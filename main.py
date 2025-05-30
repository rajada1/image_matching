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
import psutil
import gc
from annoy import AnnoyIndex

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Disney Pin Image Matching API", version="1.0.0")

class ImageMatcher:
    def __init__(self, database_path: str = "image_data"):
        self.database_path = database_path
        self.features_cache_file = "features_cache.pkl"
        self.metadata_cache_file = "metadata_cache.json"
        self.annoy_index_file = "annoy_index.ann"
        self.annoy_mapping_file = "annoy_mapping.json"
        
        # 🎯 CONFIGURAÇÕES DE PERFORMANCE MELHORADAS
        self.early_stop_threshold = 0.98  # 🔧 Reduzido para encontrar bons matches mais rápido
        self.min_threshold = 0.2  # 🔧 Reduzido para capturar mais candidatos iniciais
        self.hybrid_threshold = 0.3  # 🔧 NOVO: Threshold específico para busca híbrida
        self.max_workers = 16  # Número de threads para busca paralela
        self.use_parallel_search = True  # Habilita/desabilita busca paralela
        self.batch_size = 100  # Tamanho do lote para processamento paralelo
        
        # 🚀 CONFIGURAÇÕES ANNOY BALANCEADAS (PRECISÃO vs MEMÓRIA)
        self.use_annoy = True  # Habilita/desabilita busca com Annoy
        self.annoy_n_trees = 50  # 🔧 BALANCEADO: Aumentado para melhor precisão (era 15)
        self.annoy_search_k = 200  # 🔧 BALANCEADO: Aumentado para melhor recall (era 50)
        self.descriptor_dim = 32  # Dimensão dos descriptors ORB (sempre 32)
        self.annoy_batch_size = 10000  # Tamanho do lote para construção incremental do índice
        
        # 🔧 CONFIGURAÇÃO ORB BALANCEADA (PRECISÃO vs MEMÓRIA)
        self.orb = cv2.ORB_create(
            nfeatures=250,  # 🔧 BALANCEADO: Aumentado para melhor precisão (era 150)
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
        
        # 🚀 Annoy - Índice e mapeamentos
        self.annoy_index = None
        self.annoy_id_to_image = {}  # Mapeia ID do Annoy para caminho da imagem
        self.annoy_id_to_descriptor_idx = {}  # Mapeia ID do Annoy para índice do descriptor
        
        # Carrega ou cria o banco de features
        self.load_or_create_database()
    
    def get_memory_usage(self) -> Dict:
        """Retorna informações sobre uso de memória"""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        return {
            'rss_gb': memory_info.rss / (1024**3),  # Resident Set Size in GB
            'vms_gb': memory_info.vms / (1024**3),  # Virtual Memory Size in GB
            'percent': memory_percent,
            'available_gb': psutil.virtual_memory().available / (1024**3)
        }
    
    def log_memory_usage(self, context: str = ""):
        """Log do uso atual de memória"""
        try:
            mem = self.get_memory_usage()
            logger.info(f"💾 Memória{' ' + context if context else ''}: "
                       f"RSS={mem['rss_gb']:.2f}GB, "
                       f"VMS={mem['vms_gb']:.2f}GB, "
                       f"Uso={mem['percent']:.1f}%, "
                       f"Disponível={mem['available_gb']:.2f}GB")
        except Exception as e:
            logger.warning(f"Erro ao obter info de memória: {e}")
    
    def cleanup_memory(self):
        """Força limpeza de memória"""
        gc.collect()
        logger.info("🧹 Limpeza de memória executada")
    
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
        
        # Salva índice Annoy se habilitado
        if self.use_annoy and self.annoy_index is not None:
            self.save_annoy_index()
    
    def save_annoy_index(self):
        """Salva índice Annoy e mapeamentos"""
        try:
            # Salva o índice
            self.annoy_index.save(self.annoy_index_file)
            
            # Salva os mapeamentos
            annoy_mapping = {
                'id_to_image': self.annoy_id_to_image,
                'id_to_descriptor_idx': self.annoy_id_to_descriptor_idx,
                'n_trees': self.annoy_n_trees,
                'descriptor_dim': self.descriptor_dim
            }
            
            with open(self.annoy_mapping_file, 'w') as f:
                json.dump(annoy_mapping, f, indent=2)
            
            logger.info(f"🚀 Índice Annoy salvo com {len(self.annoy_id_to_image)} descritores")
            
        except Exception as e:
            logger.error(f"Erro ao salvar índice Annoy: {e}")
    
    def load_annoy_index(self) -> bool:
        """Carrega índice Annoy e mapeamentos"""
        try:
            if (os.path.exists(self.annoy_index_file) and
                os.path.exists(self.annoy_mapping_file)):
                
                # Carrega mapeamentos
                with open(self.annoy_mapping_file, 'r') as f:
                    annoy_mapping = json.load(f)
                
                self.annoy_id_to_image = annoy_mapping['id_to_image']
                self.annoy_id_to_descriptor_idx = annoy_mapping['id_to_descriptor_idx']
                
                # Cria e carrega índice
                self.annoy_index = AnnoyIndex(self.descriptor_dim, 'angular')
                self.annoy_index.load(self.annoy_index_file)
                
                logger.info(f"🚀 Índice Annoy carregado com {len(self.annoy_id_to_image)} descritores")
                return True
                
        except Exception as e:
            logger.warning(f"Erro ao carregar índice Annoy: {e}")
        
        return False
    
    def build_annoy_index(self):
        """Constrói índice Annoy de forma incremental com monitoramento de memória"""
        if not self.use_annoy:
            return
            
        logger.info("🚀 Construindo índice Annoy OTIMIZADO...")
        self.log_memory_usage("(inicial)")
        
        # Estima número total de descritores
        total_descriptors = sum(len(descriptors) for descriptors in self.database_features.values())
        total_images = len(self.database_features)
        
        logger.info(f"📊 Estimativa: {total_descriptors:,} descritores de {total_images:,} imagens")
        logger.info(f"💾 Memória estimada do índice: ~{(total_descriptors * self.descriptor_dim * 4) / (1024**3):.2f} GB")
        
        # Verifica se há memória suficiente
        current_mem = self.get_memory_usage()
        estimated_gb = (total_descriptors * self.descriptor_dim * 4) / (1024**3)
        
        if estimated_gb > current_mem['available_gb'] * 0.8:  # 80% da memória disponível
            logger.warning(f"⚠️  ATENÇÃO: Memória estimada ({estimated_gb:.2f}GB) próxima do limite disponível ({current_mem['available_gb']:.2f}GB)")
            logger.info("💡 Considere reduzir ainda mais o número de features ou processar em lotes menores")
        
        # Cria novo índice
        self.annoy_index = AnnoyIndex(self.descriptor_dim, 'angular')
        self.annoy_id_to_image = {}
        self.annoy_id_to_descriptor_idx = {}
        
        annoy_id = 0
        processed_descriptors = 0
        logger.info("🚀 Adicionando descritores ao índice (processamento incremental)...")
        
        # Processa em lotes para monitoramento
        for image_idx, (image_path, descriptors) in enumerate(self.database_features.items()):
            # Adiciona todos os descriptors desta imagem
            for desc_idx, descriptor in enumerate(descriptors):
                # Converte descriptor para float32 (requerido pelo Annoy)
                desc_float = descriptor.astype(np.float32)
                
                # Adiciona ao índice
                self.annoy_index.add_item(annoy_id, desc_float)
                
                # Salva mapeamentos
                self.annoy_id_to_image[str(annoy_id)] = image_path
                self.annoy_id_to_descriptor_idx[str(annoy_id)] = desc_idx
                
                annoy_id += 1
                processed_descriptors += 1
            
            # Log do progresso e memória a cada lote de imagens
            if (image_idx + 1) % 1000 == 0 or (image_idx + 1) == total_images:
                progress_pct = ((image_idx + 1) / total_images) * 100
                logger.info(f"📈 Progresso: {image_idx + 1:,}/{total_images:,} imagens ({progress_pct:.1f}%) - {processed_descriptors:,} descritores")
                self.log_memory_usage(f"(após {image_idx + 1:,} imagens)")
                
                # Força limpeza de memória a cada 5000 imagens
                if (image_idx + 1) % 5000 == 0:
                    self.cleanup_memory()

        # Log memória antes da construção das árvores
        logger.info(f"🔨 Construindo {self.annoy_n_trees} árvores para {annoy_id:,} descritores...")
        logger.info("⏳ Este processo pode demorar alguns minutos e usar bastante memória...")
        self.log_memory_usage("(antes da construção)")
        
        build_start = time.time()
        self.annoy_index.build(self.annoy_n_trees)
        build_time = time.time() - build_start
        
        # Log final
        self.log_memory_usage("(após construção)")
        self.cleanup_memory()
        
        logger.info(f"✅ Índice Annoy construído com sucesso!")
        logger.info(f"📊 Total: {annoy_id:,} descritores de {total_images:,} imagens")
        logger.info(f"⏱️  Tempo de construção: {build_time:.2f} segundos")
        logger.info(f"🌳 Árvores construídas: {self.annoy_n_trees}")
        logger.info(f"🔧 Otimizações aplicadas: ORB features reduzidas para {self.orb.getMaxFeatures()}")
    
    def load_cache(self) -> bool:
        """Carrega cache de features e metadata"""
        try:
            if os.path.exists(self.features_cache_file) and os.path.exists(self.metadata_cache_file):
                with open(self.features_cache_file, 'rb') as f:
                    self.database_features = pickle.load(f)
                
                with open(self.metadata_cache_file, 'r') as f:
                    self.database_metadata = json.load(f)
                
                logger.info(f"Cache carregado com {len(self.database_features)} imagens")
                
                # Tenta carregar índice Annoy
                if self.use_annoy:
                    if not self.load_annoy_index():
                        logger.info("Índice Annoy não encontrado, será construído...")
                        self.build_annoy_index()
                
                return True
        except Exception as e:
            logger.warning(f"Erro ao carregar cache: {e}")
        
        return False
    
    def process_database_images(self):
        """Processa todas as imagens do banco e extrai features com otimizações de memória"""
        if not os.path.exists(self.database_path):
            raise ValueError(f"Diretório do banco não encontrado: {self.database_path}")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        processed_count = 0
        skipped_count = 0
        total_features = 0
        
        # Conta total de arquivos primeiro
        all_image_files = []
        for root, dirs, files in os.walk(self.database_path):
            for file in files:
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in image_extensions:
                    all_image_files.append(os.path.join(root, file))
        
        total_files = len(all_image_files)
        logger.info(f"🖼️  Encontrados {total_files:,} arquivos de imagem para processar")
        logger.info(f"🔧 Configuração ORB: {self.orb.getMaxFeatures()} features máximas por imagem")
        
        # Processa imagens
        for idx, file_path in enumerate(all_image_files):
            try:
                image = self.load_image(file_path)
                keypoints, descriptors = self.extract_features(image)
                
                if descriptors is not None and len(descriptors) > 0:
                    relative_path = os.path.relpath(file_path, self.database_path)
                    
                    self.database_features[relative_path] = descriptors
                    self.database_metadata[relative_path] = {
                        'filename': os.path.basename(file_path),
                        'path': file_path,
                        'features_count': len(descriptors),
                        'image_shape': image.shape
                    }
                    
                    processed_count += 1
                    total_features += len(descriptors)
                    
                    # Log progresso a cada 100 imagens
                    if processed_count % 100 == 0 or processed_count == 1:
                        progress_pct = ((idx + 1) / total_files) * 100
                        avg_features = total_features / processed_count
                        logger.info(f"📈 Progresso: {processed_count:,}/{total_files:,} ({progress_pct:.1f}%) - Média: {avg_features:.0f} features/img")
                else:
                    skipped_count += 1
                    if skipped_count <= 10:  # Log apenas as primeiras 10 falhas
                        logger.warning(f"⚠️  Imagem sem features: {os.path.basename(file_path)}")
                
                # Libera memória da imagem processada
                del image
                if descriptors is not None:
                    del descriptors, keypoints
                        
            except Exception as e:
                skipped_count += 1
                logger.error(f"❌ Erro ao processar {os.path.basename(file_path)}: {e}")
        
        # Estatísticas finais
        avg_features_per_image = total_features / processed_count if processed_count > 0 else 0
        estimated_memory_gb = (total_features * self.descriptor_dim * 4) / (1024**3)
        
        logger.info(f"✅ PROCESSAMENTO CONCLUÍDO:")
        logger.info(f"   📊 Imagens processadas: {processed_count:,}")
        logger.info(f"   ⚠️  Imagens ignoradas: {skipped_count:,}")
        logger.info(f"   🔍 Total de features: {total_features:,}")
        logger.info(f"   📈 Média de features/imagem: {avg_features_per_image:.1f}")
        logger.info(f"   💾 Memória estimada do índice: ~{estimated_memory_gb:.2f} GB")
        
        # Constrói índice Annoy se habilitado
        if self.use_annoy and processed_count > 0:
            self.build_annoy_index()
        
        self.save_cache()
    
    def load_or_create_database(self):
        """Carrega cache existente ou processa imagens do banco"""
        if not self.load_cache():
            logger.info("Cache não encontrado. Processando imagens do banco...")
            self.process_database_images()
    
    def calculate_similarity(self, descriptors1: np.ndarray, descriptors2: np.ndarray) -> float:
        """Calcula similaridade melhorada entre dois conjuntos de descriptors"""
        if descriptors1 is None or descriptors2 is None or len(descriptors1) == 0 or len(descriptors2) == 0:
            return 0.0
        
        try:
            matches = self.matcher.match(descriptors1, descriptors2)
            
            if len(matches) == 0:
                return 0.0
            
            # Ordena matches por distância (melhores primeiro)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Filtros de qualidade progressivos
            excellent_matches = [m for m in matches if m.distance < 25]  # Matches excelentes
            good_matches = [m for m in matches if m.distance < 40]       # Matches bons
            decent_matches = [m for m in matches if m.distance < 60]     # Matches decentes
            
            # Calcula métricas de qualidade
            total_features = min(len(descriptors1), len(descriptors2))
            excellent_ratio = len(excellent_matches) / total_features
            good_ratio = len(good_matches) / total_features
            decent_ratio = len(decent_matches) / total_features
            
            # Score ponderado por qualidade das matches
            quality_score = (
                excellent_ratio * 1.0 +    # Peso total para matches excelentes
                good_ratio * 0.7 +         # Peso alto para matches bons
                decent_ratio * 0.4         # Peso moderado para matches decentes
            ) / 3.0
            
            # Score baseado na distribuição de distâncias
            if len(good_matches) > 0:
                avg_distance = sum(m.distance for m in good_matches) / len(good_matches)
                distance_score = max(0, (80 - avg_distance) / 80)  # Normaliza para 0-1
            else:
                distance_score = 0.0
            
            # Score de cobertura (quantas features foram matcheadas)
            coverage_score = len(decent_matches) / total_features
            
            # Combinação final ponderada
            final_score = (
                quality_score * 0.5 +      # 50% qualidade das matches
                distance_score * 0.3 +     # 30% qualidade das distâncias
                coverage_score * 0.2       # 20% cobertura de features
            )
            
            # Bonus para imagens com muitas matches excelentes
            if excellent_ratio > 0.1:  # Mais de 10% de matches excelentes
                excellent_bonus = min(excellent_ratio * 0.2, 0.1)  # Até 10% de bonus
                final_score += excellent_bonus
            
            return min(final_score, 1.0)
            
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
        
        # 🚀 Escolhe método de busca
        if self.use_annoy and self.annoy_index is not None:
            logger.info(f"🚀 Usando busca Annoy (search_k={self.annoy_search_k})")
            results = self._search_with_annoy(query_descriptors, top_k)
        elif self.use_parallel_search:
            logger.info(f"⚡ Usando busca paralela (workers={self.max_workers})")
            results = self._search_parallel(query_descriptors, top_k)
        else:
            logger.info("🔄 Usando busca sequencial")
            results = self._search_sequential(query_descriptors, top_k)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Busca concluída em {elapsed_time:.2f}s - {len(results)} resultados encontrados")
        
        return results
    
    def _search_with_annoy(self, query_descriptors: np.ndarray, top_k: int) -> List[Dict]:
        """Busca híbrida Annoy + ORB para melhor precisão"""
        if self.annoy_index is None:
            logger.warning("Índice Annoy não disponível, usando busca sequencial")
            return self._search_sequential(query_descriptors, top_k)
        
        logger.info(f"🔍 Busca híbrida: {len(query_descriptors)} descritores de consulta")
        
        # FASE 1: Busca Annoy para encontrar candidatos rapidamente
        candidate_images = set()
        image_scores = {}
        
        # Aumenta temporariamente o search_k para melhor recall
        search_k_expanded = min(self.annoy_search_k * 3, 300)
        
        for query_desc in query_descriptors:
            query_float = query_desc.astype(np.float32)
            
            # Busca mais candidatos no Annoy
            similar_ids, distances = self.annoy_index.get_nns_by_vector(
                query_float,
                search_k_expanded,
                include_distances=True
            )
            
            # Coleta candidatos únicos
            for annoy_id, distance in zip(similar_ids, distances):
                image_path = self.annoy_id_to_image.get(str(annoy_id))
                if image_path is None:
                    continue
                
                candidate_images.add(image_path)
                
                # Score Annoy melhorado
                similarity = max(0, 1 - distance / 2)
                
                if image_path not in image_scores:
                    image_scores[image_path] = {
                        'annoy_scores': [],
                        'annoy_best': 0.0
                    }
                
                image_scores[image_path]['annoy_scores'].append(similarity)
                image_scores[image_path]['annoy_best'] = max(
                    image_scores[image_path]['annoy_best'], similarity
                )
        
        logger.info(f"🎯 Fase 1: {len(candidate_images)} candidatos via Annoy")
        
        # FASE 2: Refinamento com ORB tradicional nos top candidatos
        top_candidates = sorted(
            candidate_images,
            key=lambda img: image_scores[img]['annoy_best'],
            reverse=True
        )[:min(100, len(candidate_images))]  # Refina top 100
        
        logger.info(f"🔬 Fase 2: Refinando {len(top_candidates)} candidatos com ORB")
        
        refined_results = []
        for image_path in top_candidates:
            # Calcula similaridade ORB tradicional (mais precisa)
            db_descriptors = self.database_features.get(image_path)
            if db_descriptors is None:
                continue
            
            orb_similarity = self.calculate_similarity(query_descriptors, db_descriptors)
            
            # Score híbrido: ORB (70%) + Annoy (30%)
            annoy_avg = sum(image_scores[image_path]['annoy_scores']) / len(image_scores[image_path]['annoy_scores'])
            hybrid_score = orb_similarity * 0.7 + annoy_avg * 0.3
            
            if hybrid_score >= self.min_threshold:
                result = {
                    'image_path': image_path,
                    'similarity_score': hybrid_score,
                    'metadata': self.database_metadata.get(image_path, {}),
                    'match_details': {
                        'orb_similarity': orb_similarity,
                        'annoy_similarity': annoy_avg,
                        'hybrid_score': hybrid_score,
                        'annoy_matches': len(image_scores[image_path]['annoy_scores'])
                    },
                    'search_method': 'hybrid_annoy_orb'
                }
                refined_results.append(result)
                
                # Early stopping com score híbrido
                if hybrid_score >= self.early_stop_threshold:
                    logger.info(f"🚀 EARLY STOP HÍBRIDO! Score {hybrid_score:.3f} (ORB: {orb_similarity:.3f}) em '{image_path}'")
                    return [result]
        
        # Ordena por score híbrido
        refined_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        logger.info(f"✅ Busca híbrida: {len(refined_results)} resultados com ORB+Annoy")
        
        return refined_results[:top_k]
    
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
    annoy_status = "habilitado" if matcher.use_annoy else "desabilitado"
    annoy_loaded = matcher.annoy_index is not None if matcher.use_annoy else False
    
    return {
        "message": "Disney Pin Image Matching API - OTIMIZADA com Annoy 🚀",
        "database_size": len(matcher.database_features),
        "performance_config": {
            "early_stop_threshold": matcher.early_stop_threshold,
            "parallel_search": matcher.use_parallel_search,
            "max_workers": matcher.max_workers,
            "annoy_enabled": matcher.use_annoy,
            "annoy_loaded": annoy_loaded
        },
        "annoy_info": {
            "status": annoy_status,
            "n_trees": matcher.annoy_n_trees,
            "search_k": matcher.annoy_search_k,
            "total_descriptors": len(matcher.annoy_id_to_image) if annoy_loaded else 0
        },
        "endpoints": {
            "/search": "POST - Busca imagens similares (Annoy/early stopping)",
            "/searchtest": "POST - Retorna apenas a imagem com maior similaridade",
            "/database/info": "GET - Informações do banco",
            "/database/rebuild": "POST - Reconstrói cache do banco",
            "/performance/config": "GET - Visualiza configurações de performance",
            "/performance/config": "POST - Atualiza configurações de performance",
            "/annoy/rebuild": "POST - Reconstrói índice Annoy"
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
            "total_images": len(matcher.database_features),
            "annoy_rebuilt": matcher.use_annoy and matcher.annoy_index is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao reconstruir banco: {str(e)}")

@app.post("/annoy/rebuild")
async def rebuild_annoy_index():
    """Reconstrói apenas o índice Annoy"""
    if not matcher.use_annoy:
        raise HTTPException(status_code=400, detail="Annoy está desabilitado")
    
    if len(matcher.database_features) == 0:
        raise HTTPException(status_code=400, detail="Nenhuma feature no banco. Execute /database/rebuild primeiro")
    
    try:
        matcher.build_annoy_index()
        matcher.save_annoy_index()
        
        return {
            "message": "Índice Annoy reconstruído com sucesso",
            "total_descriptors": len(matcher.annoy_id_to_image),
            "n_trees": matcher.annoy_n_trees
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao reconstruir índice Annoy: {str(e)}")

@app.get("/performance/config")
async def get_performance_config():
    """Retorna configurações atuais de performance"""
    return {
        "early_stop_threshold": matcher.early_stop_threshold,
        "min_threshold": matcher.min_threshold,
        "max_workers": matcher.max_workers,
        "use_parallel_search": matcher.use_parallel_search,
        "batch_size": matcher.batch_size,
        "database_size": len(matcher.database_features),
        "annoy_config": {
            "use_annoy": matcher.use_annoy,
            "n_trees": matcher.annoy_n_trees,
            "search_k": matcher.annoy_search_k,
            "index_loaded": matcher.annoy_index is not None,
            "total_descriptors": len(matcher.annoy_id_to_image) if matcher.annoy_index else 0
        }
    }

@app.post("/performance/config")
async def update_performance_config(
    early_stop_threshold: float = None,
    min_threshold: float = None,
    max_workers: int = None,
    use_parallel_search: bool = None,
    batch_size: int = None,
    use_annoy: bool = None,
    annoy_search_k: int = None
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
    
    if use_annoy is not None:
        matcher.use_annoy = use_annoy
        updated["use_annoy"] = use_annoy
    
    if annoy_search_k is not None:
        if 10 <= annoy_search_k <= 1000:
            matcher.annoy_search_k = annoy_search_k
            updated["annoy_search_k"] = annoy_search_k
        else:
            raise HTTPException(status_code=400, detail="annoy_search_k deve estar entre 10 e 1000")
    
    return {
        "message": "Configurações atualizadas com sucesso",
        "updated": updated,
        "current_config": {
            "early_stop_threshold": matcher.early_stop_threshold,
            "min_threshold": matcher.min_threshold,
            "max_workers": matcher.max_workers,
            "use_parallel_search": matcher.use_parallel_search,
            "batch_size": matcher.batch_size,
            "use_annoy": matcher.use_annoy,
            "annoy_search_k": matcher.annoy_search_k
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