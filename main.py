import cv2
import numpy as np
import os
import json
import pickle
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Dict, Tuple
import uvicorn
from pathlib import Path
import tempfile
import logging
import concurrent.futures
import threading
import time

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Disney Pin Image Matching API", version="1.0.0")

class ImageMatcher:
    def __init__(self, database_path: str = "image_data"):
        self.database_path = database_path
        self.features_cache_file = "features_cache.pkl"
        self.metadata_cache_file = "metadata_cache.json"
        
        # 🎯 CONFIGURAÇÕES DE PERFORMANCE - Ajuste estes valores conforme necessário
        self.early_stop_threshold = 0.4  # ⚠️ THRESHOLD PRINCIPAL - Score mínimo para retorno imediato
        self.min_threshold = 0.3  # Score mínimo para considerar como candidato
        self.max_workers = 14  # Número de threads para busca paralela
        self.use_parallel_search = True  # Habilita/desabilita busca paralela
        self.batch_size = 1000  # Tamanho do lote para processamento paralelo
        
        # Configuração do detector ORB
        self.orb = cv2.ORB_create(
            nfeatures=1000,  # Número máximo de features
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20
        )
        
        # Matcher para comparação de features
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Cache de features das imagens do banco
        self.database_features = {}
        self.database_metadata = {}
        
        # Carrega ou cria o banco de features
        self.load_or_create_database()
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Pré-processamento da imagem para melhorar a detecção de features"""
        # Converte para escala de cinza se necessário
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Equalização de histograma adaptativa para melhorar contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Redimensiona para tamanho padrão se muito grande
        height, width = enhanced.shape
        if max(height, width) > 1000:
            scale = 1000 / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            enhanced = cv2.resize(enhanced, (new_width, new_height))
        
        return enhanced
    
    def extract_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extrai features usando ORB"""
        processed_image = self.preprocess_image(image)
        keypoints, descriptors = self.orb.detectAndCompute(processed_image, None)
        
        if descriptors is None:
            return np.array([]), np.array([])
        
        return keypoints, descriptors
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Carrega imagem do disco"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
        return image
    
    def save_cache(self):
        """Salva cache de features e metadata"""
        with open(self.features_cache_file, 'wb') as f:
            pickle.dump(self.database_features, f)
        
        with open(self.metadata_cache_file, 'w') as f:
            json.dump(self.database_metadata, f, indent=2)
        
        logger.info(f"Cache salvo com {len(self.database_features)} imagens")
    
    def load_cache(self) -> bool:
        """Carrega cache de features e metadata"""
        try:
            if os.path.exists(self.features_cache_file) and os.path.exists(self.metadata_cache_file):
                with open(self.features_cache_file, 'rb') as f:
                    self.database_features = pickle.load(f)
                
                with open(self.metadata_cache_file, 'r') as f:
                    self.database_metadata = json.load(f)
                
                logger.info(f"Cache carregado com {len(self.database_features)} imagens")
                return True
        except Exception as e:
            logger.warning(f"Erro ao carregar cache: {e}")
        
        return False
    
    def process_database_images(self):
        """Processa todas as imagens do banco e extrai features"""
        if not os.path.exists(self.database_path):
            raise ValueError(f"Diretório do banco não encontrado: {self.database_path}")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        processed_count = 0
        
        for root, dirs, files in os.walk(self.database_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                if file_ext in image_extensions:
                    try:
                        image = self.load_image(file_path)
                        keypoints, descriptors = self.extract_features(image)
                        
                        if descriptors is not None and len(descriptors) > 0:
                            relative_path = os.path.relpath(file_path, self.database_path)
                            
                            self.database_features[relative_path] = descriptors
                            self.database_metadata[relative_path] = {
                                'filename': file,
                                'path': file_path,
                                'features_count': len(descriptors),
                                'image_shape': image.shape
                            }
                            
                            processed_count += 1
                            logger.info(f"Processada: {relative_path} ({len(descriptors)} features)")
                        
                    except Exception as e:
                        logger.error(f"Erro ao processar {file_path}: {e}")
        
        logger.info(f"Total de imagens processadas: {processed_count}")
        self.save_cache()
    
    def load_or_create_database(self):
        """Carrega cache existente ou processa imagens do banco"""
        if not self.load_cache():
            logger.info("Cache não encontrado. Processando imagens do banco...")
            self.process_database_images()
    
    def calculate_similarity(self, descriptors1: np.ndarray, descriptors2: np.ndarray) -> float:
        """Calcula similaridade entre dois conjuntos de descriptors"""
        if descriptors1 is None or descriptors2 is None or len(descriptors1) == 0 or len(descriptors2) == 0:
            return 0.0
        
        try:
            matches = self.matcher.match(descriptors1, descriptors2)
            
            if len(matches) == 0:
                return 0.0
            
            # Ordena matches por distância
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Calcula score baseado nas melhores matches
            good_matches = [m for m in matches if m.distance < 50]  # Threshold ajustável
            
            # Score baseado na proporção de boas matches
            max_features = max(len(descriptors1), len(descriptors2))
            similarity_score = len(good_matches) / max_features
            
            # Score adicional baseado na qualidade das matches
            if len(good_matches) > 0:
                avg_distance = sum(m.distance for m in good_matches) / len(good_matches)
                distance_score = max(0, (100 - avg_distance) / 100)  # Normaliza distância
                similarity_score = (similarity_score + distance_score) / 2
            
            return min(similarity_score, 1.0)
            
        except Exception as e:
            logger.error(f"Erro no cálculo de similaridade: {e}")
            return 0.0
    
    def search_similar_images(self, query_image: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Busca imagens similares no banco com otimizações de performance"""
        start_time = time.time()
        query_keypoints, query_descriptors = self.extract_features(query_image)
        
        if query_descriptors is None or len(query_descriptors) == 0:
            return []
        
        logger.info(f"🔍 Iniciando busca em {len(self.database_features)} imagens")
        logger.info(f"⚡ Configurações: threshold={self.early_stop_threshold}, parallel={self.use_parallel_search}, workers={self.max_workers}")
        
        if self.use_parallel_search:
            results = self._search_parallel(query_descriptors, top_k)
        else:
            results = self._search_sequential(query_descriptors, top_k)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Busca concluída em {elapsed_time:.2f}s - {len(results)} resultados encontrados")
        
        return results
    
    def _search_sequential(self, query_descriptors: np.ndarray, top_k: int) -> List[Dict]:
        """Busca sequencial com early stopping otimizado"""
        results = []
        images_processed = 0
        early_stopped = False
        
        for image_path, db_descriptors in self.database_features.items():
            similarity = self.calculate_similarity(query_descriptors, db_descriptors)
            images_processed += 1
            
            if similarity >= self.early_stop_threshold:
                # 🎯 EARLY STOPPING ATIVADO - Encontrou match excelente!
                result = {
                    'image_path': image_path,
                    'similarity_score': similarity,
                    'metadata': self.database_metadata.get(image_path, {}),
                    'early_stopped': True
                }
                logger.info(f"🚀 EARLY STOP! Score {similarity:.3f} >= {self.early_stop_threshold} em '{image_path}' ({images_processed}/{len(self.database_features)} processadas)")
                return [result]
            
            elif similarity >= self.min_threshold:
                result = {
                    'image_path': image_path,
                    'similarity_score': similarity,
                    'metadata': self.database_metadata.get(image_path, {}),
                    'early_stopped': False
                }
                results.append(result)
        
        # Ordena por similaridade e retorna top_k
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        logger.info(f"📊 Processamento sequencial: {images_processed} imagens analisadas")
        return results[:top_k]
    
    def _calculate_similarity_batch(self, args):
        """Função auxiliar para calcular similaridade em lote (para threading)"""
        query_descriptors, batch_items = args
        batch_results = []
        
        for image_path, db_descriptors in batch_items:
            similarity = self.calculate_similarity(query_descriptors, db_descriptors)
            
            if similarity >= self.early_stop_threshold:
                # Early stop encontrado em thread
                result = {
                    'image_path': image_path,
                    'similarity_score': similarity,
                    'metadata': self.database_metadata.get(image_path, {}),
                    'early_stopped': True
                }
                return [result]  # Retorna imediatamente apenas este resultado
            
            elif similarity >= self.min_threshold:
                result = {
                    'image_path': image_path,
                    'similarity_score': similarity,
                    'metadata': self.database_metadata.get(image_path, {}),
                    'early_stopped': False
                }
                batch_results.append(result)
        
        return batch_results
    
    def _search_parallel(self, query_descriptors: np.ndarray, top_k: int) -> List[Dict]:
        """Busca paralela com early stopping"""
        database_items = list(self.database_features.items())
        
        # Divide em lotes para processamento paralelo
        batches = [
            database_items[i:i + self.batch_size]
            for i in range(0, len(database_items), self.batch_size)
        ]
        
        logger.info(f"🔄 Processamento paralelo: {len(batches)} lotes de ~{self.batch_size} imagens")
        
        all_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Prepara argumentos para cada batch
            batch_args = [(query_descriptors, batch) for batch in batches]
            
            # Executa em paralelo
            future_to_batch = {
                executor.submit(self._calculate_similarity_batch, args): i
                for i, args in enumerate(batch_args)
            }
            
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_results = future.result()
                    
                    # Verifica se algum resultado tem early stop
                    for result in batch_results:
                        if result.get('early_stopped', False):
                            logger.info(f"🚀 EARLY STOP PARALELO! Score {result['similarity_score']:.3f} em '{result['image_path']}' (lote {batch_idx + 1})")
                            # Cancela outras tarefas pendentes
                            for pending_future in future_to_batch:
                                if not pending_future.done():
                                    pending_future.cancel()
                            return [result]
                    
                    all_results.extend(batch_results)
                    
                except Exception as e:
                    logger.error(f"Erro no lote {batch_idx}: {e}")
        
        # Ordena por similaridade e retorna top_k
        all_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        logger.info(f"📊 Processamento paralelo concluído: {len(all_results)} candidatos encontrados")
        return all_results[:top_k]

# Instância global do matcher
matcher = ImageMatcher()

@app.get("/")
async def root():
    return {
        "message": "Disney Pin Image Matching API - OTIMIZADA ⚡",
        "database_size": len(matcher.database_features),
        "performance_config": {
            "early_stop_threshold": matcher.early_stop_threshold,
            "parallel_search": matcher.use_parallel_search,
            "max_workers": matcher.max_workers
        },
        "endpoints": {
            "/search": "POST - Busca imagens similares (com early stopping)",
            "/searchtest": "POST - Retorna apenas a imagem com maior similaridade",
            "/database/info": "GET - Informações do banco",
            "/database/rebuild": "POST - Reconstrói cache do banco",
            "/performance/config": "GET - Visualiza configurações de performance",
            "/performance/config": "POST - Atualiza configurações de performance"
        }
    }

@app.get("/database/info")
async def database_info():
    """Retorna informações sobre o banco de imagens"""
    total_features = sum(
        metadata.get('features_count', 0) 
        for metadata in matcher.database_metadata.values()
    )
    
    return {
        "total_images": len(matcher.database_features),
        "total_features": total_features,
        "database_path": matcher.database_path,
        "images": list(matcher.database_metadata.keys())
    }

@app.post("/database/rebuild")
async def rebuild_database():
    """Reconstrói o cache do banco de imagens"""
    try:
        matcher.process_database_images()
        return {
            "message": "Banco reconstruído com sucesso",
            "total_images": len(matcher.database_features)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao reconstruir banco: {str(e)}")

@app.get("/performance/config")
async def get_performance_config():
    """Retorna configurações atuais de performance"""
    return {
        "early_stop_threshold": matcher.early_stop_threshold,
        "min_threshold": matcher.min_threshold,
        "max_workers": matcher.max_workers,
        "use_parallel_search": matcher.use_parallel_search,
        "batch_size": matcher.batch_size,
        "database_size": len(matcher.database_features)
    }

@app.post("/performance/config")
async def update_performance_config(
    early_stop_threshold: float = None,
    min_threshold: float = None,
    max_workers: int = None,
    use_parallel_search: bool = None,
    batch_size: int = None
):
    """Atualiza configurações de performance em tempo real"""
    updated = {}
    
    if early_stop_threshold is not None:
        if 0.1 <= early_stop_threshold <= 1.0:
            matcher.early_stop_threshold = early_stop_threshold
            updated["early_stop_threshold"] = early_stop_threshold
        else:
            raise HTTPException(status_code=400, detail="early_stop_threshold deve estar entre 0.1 e 1.0")
    
    if min_threshold is not None:
        if 0.0 <= min_threshold <= 1.0:
            matcher.min_threshold = min_threshold
            updated["min_threshold"] = min_threshold
        else:
            raise HTTPException(status_code=400, detail="min_threshold deve estar entre 0.0 e 1.0")
    
    if max_workers is not None:
        if 1 <= max_workers <= 16:
            matcher.max_workers = max_workers
            updated["max_workers"] = max_workers
        else:
            raise HTTPException(status_code=400, detail="max_workers deve estar entre 1 e 16")
    
    if use_parallel_search is not None:
        matcher.use_parallel_search = use_parallel_search
        updated["use_parallel_search"] = use_parallel_search
    
    if batch_size is not None:
        if 10 <= batch_size <= 200:
            matcher.batch_size = batch_size
            updated["batch_size"] = batch_size
        else:
            raise HTTPException(status_code=400, detail="batch_size deve estar entre 10 e 200")
    
    return {
        "message": "Configurações atualizadas com sucesso",
        "updated": updated,
        "current_config": {
            "early_stop_threshold": matcher.early_stop_threshold,
            "min_threshold": matcher.min_threshold,
            "max_workers": matcher.max_workers,
            "use_parallel_search": matcher.use_parallel_search,
            "batch_size": matcher.batch_size
        }
    }

@app.post("/search")
async def search_similar_images(
    file: UploadFile = File(...),
    top_k: int = 5
):
    """Busca imagens similares no banco"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem")
    
    try:
        # Lê a imagem enviada
        contents = await file.read()
        
        # Converte para array numpy
        nparr = np.frombuffer(contents, np.uint8)
        query_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if query_image is None:
            raise HTTPException(status_code=400, detail="Não foi possível decodificar a imagem")
        
        # Busca imagens similares
        results = matcher.search_similar_images(query_image, top_k)
        
        return {
            "query_info": {
                "filename": file.filename,
                "image_shape": query_image.shape
            },
            "results": results,
            "total_found": len(results)
        }
        
    except Exception as e:
        logger.error(f"Erro na busca: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.post("/searchtest")
async def search_test_best_match(file: UploadFile = File(...)):
    """Busca e retorna apenas a imagem com maior similaridade"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem")
    
    try:
        # Lê a imagem enviada
        contents = await file.read()
        
        # Converte para array numpy
        nparr = np.frombuffer(contents, np.uint8)
        query_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if query_image is None:
            raise HTTPException(status_code=400, detail="Não foi possível decodificar a imagem")
        
        # Busca imagens similares (apenas 1 resultado)
        results = matcher.search_similar_images(query_image, top_k=1)
        
        if not results:
            raise HTTPException(status_code=404, detail="Nenhuma imagem similar encontrada")
        
        # Pega o melhor resultado
        best_match = results[0]
        image_relative_path = best_match['image_path']
        
        # Constrói o caminho completo da imagem
        image_full_path = os.path.join(matcher.database_path, image_relative_path)
        
        # Verifica se o arquivo existe
        if not os.path.exists(image_full_path):
            raise HTTPException(status_code=404, detail="Arquivo de imagem não encontrado no disco")
        
        # Retorna a imagem diretamente
        return FileResponse(
            path=image_full_path,
            media_type='image/jpeg',
            filename=best_match['metadata'].get('filename', 'best_match.jpg'),
            headers={
                "X-Similarity-Score": str(best_match['similarity_score']),
                "X-Image-Path": image_relative_path,
                "X-Features-Count": str(best_match['metadata'].get('features_count', 0))
            }
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Erro na busca: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

if __name__ == "__main__":
    print("🎯 Iniciando Disney Pin Image Matching API...")
    print(f"📁 Banco de imagens: {matcher.database_path}")
    print(f"🖼️  Total de imagens no banco: {len(matcher.database_features)}")
    print("🚀 Servidor rodando em: http://localhost:8000")
    print("📖 Documentação em: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)