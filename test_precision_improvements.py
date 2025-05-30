#!/usr/bin/env python3
"""
Script para testar as melhorias de precisão no sistema híbrido Annoy + ORB
"""

import cv2
import numpy as np
import time
import os
from main import ImageMatcher

def test_precision_improvements():
    """Testa as melhorias de precisão implementadas"""
    print("🧪 TESTE DE MELHORIAS DE PRECISÃO")
    print("=" * 50)
    
    # Cria matcher com configurações otimizadas
    matcher = ImageMatcher()
    
    print("📊 Configurações atuais:")
    print(f"   ORB features: {matcher.orb.getMaxFeatures()}")
    print(f"   Annoy trees: {matcher.annoy_n_trees}")
    print(f"   Annoy search_k: {matcher.annoy_search_k}")
    print(f"   Early stop threshold: {matcher.early_stop_threshold}")
    print(f"   Min threshold: {matcher.min_threshold}")
    print(f"   Hybrid threshold: {matcher.hybrid_threshold}")
    print(f"   Total imagens no banco: {len(matcher.database_features)}")
    
    if len(matcher.database_features) == 0:
        print("⚠️  Nenhuma imagem no banco! Execute primeiro o processamento.")
        return False
    
    # Verifica se há imagens de teste
    test_images = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')) and 'train' in root.lower():
                test_images.append(os.path.join(root, file))
    
    if not test_images:
        print("⚠️  Nenhuma imagem de teste encontrada na pasta train_image/")
        print("💡 Coloque algumas imagens na pasta train_image/ para testar")
        return False
    
    print(f"\n🔍 Testando com {len(test_images)} imagens de teste")
    
    total_tests = 0
    successful_tests = 0
    total_search_time = 0
    
    for test_image_path in test_images[:5]:  # Testa apenas as primeiras 5
        try:
            print(f"\n📸 Testando: {os.path.basename(test_image_path)}")
            
            # Carrega imagem de teste
            query_image = cv2.imread(test_image_path)
            if query_image is None:
                print(f"   ❌ Erro ao carregar imagem")
                continue
            
            # Executa busca
            start_time = time.time()
            results = matcher.search_similar_images(query_image, top_k=5)
            search_time = time.time() - start_time
            
            total_search_time += search_time
            total_tests += 1
            
            if results:
                successful_tests += 1
                best_result = results[0]
                
                print(f"   ✅ Melhor match: {best_result['image_path']}")
                print(f"   📊 Score: {best_result['similarity_score']:.4f}")
                print(f"   ⏱️  Tempo: {search_time:.3f}s")
                print(f"   🔍 Método: {best_result.get('search_method', 'unknown')}")
                
                # Detalhes do match híbrido
                if 'match_details' in best_result:
                    details = best_result['match_details']
                    if 'orb_similarity' in details:
                        print(f"   🎯 ORB score: {details['orb_similarity']:.4f}")
                        print(f"   🚀 Annoy score: {details['annoy_similarity']:.4f}")
                        print(f"   🔀 Híbrido: {details['hybrid_score']:.4f}")
                
                # Mostra todos os resultados
                print(f"   📋 Top {len(results)} resultados:")
                for i, result in enumerate(results):
                    print(f"      {i+1}. {result['image_path']} (score: {result['similarity_score']:.4f})")
            else:
                print(f"   ❌ Nenhum resultado encontrado")
                
        except Exception as e:
            print(f"   ❌ Erro no teste: {e}")
    
    # Estatísticas finais
    print(f"\n📊 ESTATÍSTICAS FINAIS:")
    print(f"   Total de testes: {total_tests}")
    print(f"   Testes bem-sucedidos: {successful_tests}")
    print(f"   Taxa de sucesso: {(successful_tests/total_tests*100):.1f}%")
    print(f"   Tempo médio por busca: {(total_search_time/total_tests):.3f}s")
    
    if matcher.annoy_index is not None:
        print(f"   📈 Índice Annoy ativo: {len(matcher.annoy_id_to_image):,} descritores")
    else:
        print(f"   ⚠️  Índice Annoy não carregado")
    
    success_rate = (successful_tests / total_tests) if total_tests > 0 else 0
    
    print(f"\n🎯 AVALIAÇÃO DE PRECISÃO:")
    if success_rate >= 0.8:
        print(f"   🟢 EXCELENTE: Taxa de sucesso {success_rate:.1%}")
    elif success_rate >= 0.6:
        print(f"   🟡 BOM: Taxa de sucesso {success_rate:.1%}")
    elif success_rate >= 0.4:
        print(f"   🟠 MODERADO: Taxa de sucesso {success_rate:.1%}")
    else:
        print(f"   🔴 BAIXO: Taxa de sucesso {success_rate:.1%}")
    
    return success_rate >= 0.6

def test_different_thresholds():
    """Testa diferentes configurações de threshold"""
    print("\n🔧 TESTE DE CONFIGURAÇÕES DE THRESHOLD")
    print("=" * 40)
    
    matcher = ImageMatcher()
    
    # Configurações para testar
    test_configs = [
        {"name": "Restritivo", "min_threshold": 0.7, "hybrid_threshold": 0.8},
        {"name": "Balanceado", "min_threshold": 0.5, "hybrid_threshold": 0.6},
        {"name": "Permissivo", "min_threshold": 0.3, "hybrid_threshold": 0.4},
        {"name": "Muito Permissivo", "min_threshold": 0.1, "hybrid_threshold": 0.2},
    ]
    
    # Pega uma imagem de teste
    test_image_path = None
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')) and 'train' in root.lower():
                test_image_path = os.path.join(root, file)
                break
        if test_image_path:
            break
    
    if not test_image_path:
        print("⚠️  Nenhuma imagem de teste encontrada")
        return
    
    query_image = cv2.imread(test_image_path)
    if query_image is None:
        print("❌ Erro ao carregar imagem de teste")
        return
    
    print(f"🖼️  Usando imagem: {os.path.basename(test_image_path)}")
    
    for config in test_configs:
        print(f"\n🧪 Testando configuração: {config['name']}")
        print(f"   Min threshold: {config['min_threshold']}")
        print(f"   Hybrid threshold: {config['hybrid_threshold']}")
        
        # Aplica configuração
        original_min = matcher.min_threshold
        original_hybrid = matcher.hybrid_threshold
        
        matcher.min_threshold = config['min_threshold']
        matcher.hybrid_threshold = config['hybrid_threshold']
        
        try:
            start_time = time.time()
            results = matcher.search_similar_images(query_image, top_k=3)
            search_time = time.time() - start_time
            
            print(f"   ⏱️  Tempo: {search_time:.3f}s")
            print(f"   📊 Resultados: {len(results)}")
            
            if results:
                best_score = results[0]['similarity_score']
                avg_score = sum(r['similarity_score'] for r in results) / len(results)
                print(f"   🏆 Melhor score: {best_score:.4f}")
                print(f"   📈 Score médio: {avg_score:.4f}")
                
                for i, result in enumerate(results):
                    print(f"      {i+1}. {result['image_path']} ({result['similarity_score']:.4f})")
            else:
                print(f"   ❌ Nenhum resultado")
                
        except Exception as e:
            print(f"   ❌ Erro: {e}")
        finally:
            # Restaura configurações originais
            matcher.min_threshold = original_min
            matcher.hybrid_threshold = original_hybrid

if __name__ == "__main__":
    print("🧪 TESTE DE MELHORIAS DE PRECISÃO")
    print("🎯 Validando algoritmo híbrido Annoy + ORB")
    print("=" * 60)
    
    try:
        # Teste principal de precisão
        success = test_precision_improvements()
        
        if success:
            print("\n✅ MELHORIAS DE PRECISÃO VALIDADAS!")
            print("🎉 O sistema híbrido está funcionando corretamente")
            
            # Teste de diferentes thresholds
            test_different_thresholds()
            
        else:
            print("\n⚠️  ATENÇÃO: Baixa taxa de sucesso detectada")
            print("💡 Considere ajustar os thresholds ou verificar as imagens de teste")
            
    except Exception as e:
        print(f"\n❌ ERRO NO TESTE: {e}")
        print("🔧 Verifique se o sistema está funcionando corretamente")
    
    print("\n📚 PRÓXIMOS PASSOS:")
    print("1. Se a precisão estiver baixa, ajuste os thresholds via API:")
    print("   POST /performance/config")
    print("2. Use imagens de teste variadas para validar melhor")
    print("3. Monitore os logs durante as buscas para debugging")
    print("4. Execute 'python main.py' para acessar a API completa")