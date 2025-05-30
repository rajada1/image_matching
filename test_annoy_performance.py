#!/usr/bin/env python3
"""
Script para testar a performance do Annoy vs busca tradicional
"""

import cv2
import numpy as np
import time
import requests
import json
from pathlib import Path

def test_performance():
    """Testa performance das diferentes configurações"""
    
    # URL da API
    base_url = "http://localhost:8000"
    
    # Imagem de teste (primeira imagem do diretório)
    test_image_path = "image_data/3262_pinail.jpg"
    
    if not Path(test_image_path).exists():
        print("❌ Imagem de teste não encontrada:", test_image_path)
        return
    
    print("🧪 Testando Performance do Sistema de Matching")
    print("=" * 50)
    
    # Testa configurações diferentes
    configurations = [
        {
            "name": "🔄 Busca Sequencial",
            "config": {
                "use_annoy": False,
                "use_parallel_search": False
            }
        },
        {
            "name": "⚡ Busca Paralela",
            "config": {
                "use_annoy": False,
                "use_parallel_search": True,
                "max_workers": 8
            }
        },
        {
            "name": "🚀 Annoy (Padrão)",
            "config": {
                "use_annoy": True,
                "annoy_search_k": 100
            }
        },
        {
            "name": "🚀 Annoy (Mais Preciso)",
            "config": {
                "use_annoy": True,
                "annoy_search_k": 200
            }
        },
        {
            "name": "🚀 Annoy (Mais Rápido)",
            "config": {
                "use_annoy": True,
                "annoy_search_k": 50
            }
        }
    ]
    
    results = []
    
    for config_data in configurations:
        print(f"\n📊 Testando: {config_data['name']}")
        
        # Atualiza configuração
        try:
            response = requests.post(
                f"{base_url}/performance/config",
                params=config_data['config']
            )
            
            if response.status_code != 200:
                print(f"❌ Erro ao configurar: {response.text}")
                continue
                
        except Exception as e:
            print(f"❌ Erro na configuração: {e}")
            continue
        
        # Executa teste de busca 3 vezes
        times = []
        for i in range(3):
            try:
                start_time = time.time()
                
                with open(test_image_path, 'rb') as f:
                    files = {'file': f}
                    response = requests.post(f"{base_url}/search", files=files)
                
                elapsed = time.time() - start_time
                times.append(elapsed)
                
                if response.status_code == 200:
                    result_data = response.json()
                    print(f"  Teste {i+1}: {elapsed:.3f}s - {result_data['total_found']} resultados")
                else:
                    print(f"  Teste {i+1}: ❌ Erro {response.status_code}")
                    
            except Exception as e:
                print(f"  Teste {i+1}: ❌ Erro: {e}")
                times.append(999)  # Valor alto para indicar erro
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        results.append({
            'name': config_data['name'],
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'config': config_data['config']
        })
        
        print(f"  📈 Médio: {avg_time:.3f}s | Mín: {min_time:.3f}s | Máx: {max_time:.3f}s")
    
    # Mostra relatório final
    print("\n🏆 RELATÓRIO DE PERFORMANCE")
    print("=" * 50)
    
    # Ordena por tempo médio
    results.sort(key=lambda x: x['avg_time'])
    
    for i, result in enumerate(results):
        position = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}°"
        print(f"{position} {result['name']}: {result['avg_time']:.3f}s")
    
    # Calcula melhoria
    if len(results) >= 2:
        baseline = results[-1]['avg_time']  # Pior resultado
        best = results[0]['avg_time']       # Melhor resultado
        improvement = ((baseline - best) / baseline) * 100
        print(f"\n🚀 Melhoria: {improvement:.1f}% mais rápido!")
    
    return results

def check_annoy_status():
    """Verifica se o Annoy está funcionando"""
    try:
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200:
            data = response.json()
            print("📊 Status do Sistema:")
            print(f"  Database: {data['database_size']} imagens")
            print(f"  Annoy: {'✅ Habilitado' if data['performance_config']['annoy_enabled'] else '❌ Desabilitado'}")
            print(f"  Índice carregado: {'✅ Sim' if data['performance_config']['annoy_loaded'] else '❌ Não'}")
            
            if data['performance_config']['annoy_enabled']:
                annoy_info = data['annoy_info']
                print(f"  Árvores: {annoy_info['n_trees']}")
                print(f"  Search K: {annoy_info['search_k']}")
                print(f"  Descritores: {annoy_info['total_descriptors']}")
            
            return data['performance_config']['annoy_loaded']
        else:
            print("❌ Servidor não acessível")
            return False
    except Exception as e:
        print(f"❌ Erro ao verificar status: {e}")
        return False

if __name__ == "__main__":
    print("🎯 Disney Pin Image Matching - Teste de Performance")
    print("=" * 60)
    
    if not check_annoy_status():
        print("\n⚠️  Annoy não está carregado. Execute primeiro:")
        print("  1. Inicie o servidor: python main.py")
        print("  2. Se necessário, reconstrua o índice: POST /annoy/rebuild")
        exit(1)
    
    results = test_performance()
    
    print("\n✅ Teste concluído!")
    print("\n💡 Dicas de otimização:")
    print("  - Annoy é geralmente mais rápido para bancos grandes (>100 imagens)")
    print("  - Ajuste 'annoy_search_k' para balancear velocidade vs precisão")
    print("  - Use busca paralela para bancos pequenos sem Annoy")