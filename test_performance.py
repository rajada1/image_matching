#!/usr/bin/env python3
"""
Script de teste para demonstrar as melhorias de performance
"""

import requests
import time
import json
from pathlib import Path

API_BASE = "http://localhost:8000"

def test_performance_config():
    """Testa os endpoints de configuração de performance"""
    print("🔧 Testando configurações de performance...")
    
    # Visualiza configuração atual
    response = requests.get(f"{API_BASE}/performance/config")
    if response.status_code == 200:
        config = response.json()
        print(f"📊 Configuração atual:")
        for key, value in config.items():
            print(f"   {key}: {value}")
    else:
        print("❌ Erro ao obter configurações")
        return
    
    # Testa diferentes thresholds
    test_configs = [
        {"early_stop_threshold": 0.5, "description": "Mais permissivo"},
        {"early_stop_threshold": 0.6, "description": "Balanceado (padrão)"},
        {"early_stop_threshold": 0.7, "description": "Mais rigoroso"},
    ]
    
    for config_test in test_configs:
        print(f"\n🎯 Testando threshold {config_test['early_stop_threshold']} ({config_test['description']})")
        
        # Atualiza configuração
        response = requests.post(
            f"{API_BASE}/performance/config",
            params={"early_stop_threshold": config_test["early_stop_threshold"]}
        )
        
        if response.status_code == 200:
            print(f"✅ Configuração atualizada")
        else:
            print(f"❌ Erro ao atualizar configuração: {response.text}")

def test_search_performance(image_path: str, iterations: int = 3):
    """Testa performance de busca com uma imagem"""
    if not Path(image_path).exists():
        print(f"❌ Imagem não encontrada: {image_path}")
        return
    
    print(f"\n🔍 Testando performance de busca com {image_path}")
    print(f"📊 Executando {iterations} iterações...")
    
    times = []
    
    for i in range(iterations):
        print(f"\n--- Iteração {i+1}/{iterations} ---")
        
        start_time = time.time()
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_BASE}/searchtest", files=files)
        
        end_time = time.time()
        elapsed = end_time - start_time
        times.append(elapsed)
        
        if response.status_code == 200:
            # Verifica headers para informações de performance
            similarity_score = response.headers.get('X-Similarity-Score', 'N/A')
            features_count = response.headers.get('X-Features-Count', 'N/A')
            image_path_result = response.headers.get('X-Image-Path', 'N/A')
            
            print(f"✅ Busca concluída em {elapsed:.3f}s")
            print(f"🎯 Similaridade: {similarity_score}")
            print(f"📊 Features: {features_count}")
            print(f"📁 Imagem encontrada: {image_path_result}")
        else:
            print(f"❌ Erro na busca: {response.status_code} - {response.text}")
    
    # Estatísticas
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n📈 Estatísticas de Performance:")
        print(f"   Tempo médio: {avg_time:.3f}s")
        print(f"   Tempo mínimo: {min_time:.3f}s")
        print(f"   Tempo máximo: {max_time:.3f}s")
        print(f"   Variação: {max_time - min_time:.3f}s")

def test_parallel_vs_sequential():
    """Compara performance entre busca paralela e sequencial"""
    print("\n🔄 Comparando busca paralela vs sequencial...")
    
    # Primeiro, pega uma imagem de teste do banco
    response = requests.get(f"{API_BASE}/database/info")
    if response.status_code != 200:
        print("❌ Não foi possível obter informações do banco")
        return
    
    db_info = response.json()
    if not db_info.get('images'):
        print("❌ Nenhuma imagem no banco")
        return
    
    # Usa a primeira imagem como teste
    test_image_name = db_info['images'][0]
    test_image_path = f"image_data/{test_image_name}"
    
    if not Path(test_image_path).exists():
        print(f"❌ Imagem de teste não encontrada: {test_image_path}")
        return
    
    print(f"🖼️ Usando imagem de teste: {test_image_name}")
    
    # Teste com busca paralela
    print("\n1️⃣ Testando busca PARALELA...")
    requests.post(f"{API_BASE}/performance/config", params={"use_parallel_search": True})
    test_search_performance(test_image_path, 2)
    
    # Teste com busca sequencial
    print("\n2️⃣ Testando busca SEQUENCIAL...")
    requests.post(f"{API_BASE}/performance/config", params={"use_parallel_search": False})
    test_search_performance(test_image_path, 2)
    
    # Restaura configuração paralela
    requests.post(f"{API_BASE}/performance/config", params={"use_parallel_search": True})
    print("\n✅ Configuração paralela restaurada")

def main():
    """Função principal de teste"""
    print("🚀 Iniciando testes de performance da Image Matching API")
    print("=" * 60)
    
    # Verifica se API está rodando
    try:
        response = requests.get(f"{API_BASE}/")
        if response.status_code != 200:
            print("❌ API não está respondendo corretamente")
            return
        
        api_info = response.json()
        print(f"✅ API conectada: {api_info['message']}")
        print(f"📊 Banco com {api_info['database_size']} imagens")
        
    except requests.exceptions.ConnectionError:
        print("❌ Não foi possível conectar à API")
        print("💡 Certifique-se de que a API está rodando em http://localhost:8000")
        return
    
    # Executa testes
    test_performance_config()
    test_parallel_vs_sequential()
    
    print("\n" + "=" * 60)
    print("✅ Testes de performance concluídos!")
    print("\n💡 Para monitorar em tempo real:")
    print("   - Observe os logs no console da API")
    print("   - Use GET /performance/config para ver configurações atuais")
    print("   - Use POST /performance/config para ajustar parâmetros")

if __name__ == "__main__":
    main()