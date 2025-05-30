#!/usr/bin/env python3
"""
Teste rápido do sistema com Annoy
"""

import cv2
import numpy as np
import time
from main import ImageMatcher

def test_matcher():
    """Testa o matcher diretamente"""
    print("🧪 Teste Rápido do Image Matcher com Annoy")
    print("=" * 50)
    
    # Cria matcher
    matcher = ImageMatcher()
    
    print(f"📊 Banco carregado: {len(matcher.database_features)} imagens")
    print(f"🚀 Annoy habilitado: {matcher.use_annoy}")
    print(f"🚀 Índice carregado: {matcher.annoy_index is not None}")
    
    if matcher.annoy_index:
        print(f"🚀 Descritores no índice: {len(matcher.annoy_id_to_image)}")
    
    # Carrega uma imagem de teste
    test_images = [
        "image_data/3262_pinail.jpg",
        "1667_pinail.jpg",
        "dsadasdas.jpg"
    ]
    
    for test_path in test_images:
        try:
            print(f"\n🔍 Testando com: {test_path}")
            
            # Carrega imagem
            image = cv2.imread(test_path)
            if image is None:
                print(f"❌ Não foi possível carregar: {test_path}")
                continue
            
            # Teste com Annoy
            start_time = time.time()
            results_annoy = matcher.search_similar_images(image, top_k=3)
            time_annoy = time.time() - start_time
            
            print(f"🚀 Annoy: {time_annoy:.3f}s - {len(results_annoy)} resultados")
            
            # Mostra melhores resultados
            for i, result in enumerate(results_annoy[:2]):
                score = result['similarity_score']
                path = result['image_path']
                method = result.get('search_method', 'traditional')
                print(f"  {i+1}. {path} - Score: {score:.3f} ({method})")
            
            # Teste desabilitando Annoy para comparação
            matcher.use_annoy = False
            start_time = time.time()
            results_traditional = matcher.search_similar_images(image, top_k=3)
            time_traditional = time.time() - start_time
            
            print(f"🔄 Tradicional: {time_traditional:.3f}s - {len(results_traditional)} resultados")
            
            # Reabilita Annoy
            matcher.use_annoy = True
            
            # Comparação
            if time_traditional > 0:
                speedup = time_traditional / time_annoy
                print(f"⚡ Speedup: {speedup:.1f}x mais rápido com Annoy!")
            
            break  # Só testa a primeira imagem encontrada
            
        except Exception as e:
            print(f"❌ Erro: {e}")
            continue

if __name__ == "__main__":
    test_matcher()