#!/usr/bin/env python3
"""
Script de teste para a API de reconhecimento de imagens Disney Pin
"""

import requests
import sys
import os
from pathlib import Path

API_URL = "http://localhost:8000"

def test_api_connection():
    """Testa se a API está rodando"""
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            print("✅ API está rodando!")
            data = response.json()
            print(f"📊 Banco de dados: {data.get('database_size', 0)} imagens")
            return True
        else:
            print(f"❌ Erro ao conectar na API: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ API não está rodando. Execute 'python main.py' primeiro.")
        return False
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return False

def get_database_info():
    """Obtém informações do banco de dados"""
    try:
        response = requests.get(f"{API_URL}/database/info")
        if response.status_code == 200:
            data = response.json()
            print(f"\n📈 Informações do Banco de Dados:")
            print(f"   Total de imagens: {data['total_images']}")
            print(f"   Total de features: {data['total_features']}")
            print(f"   Caminho: {data['database_path']}")
            print(f"   Imagens disponíveis:")
            for img in data['images']:
                print(f"     - {img}")
            return True
        else:
            print(f"❌ Erro ao obter info do banco: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erro ao obter informações: {e}")
        return False

def search_image(image_path: str, top_k: int = 5):
    """Busca imagens similares"""
    if not os.path.exists(image_path):
        print(f"❌ Arquivo não encontrado: {image_path}")
        return False
    
    try:
        print(f"\n🔍 Buscando imagens similares para: {image_path}")
        
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            params = {'top_k': top_k}
            
            response = requests.post(f"{API_URL}/search", files=files, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Busca concluída!")
            print(f"📋 Informações da consulta:")
            print(f"   Arquivo: {data['query_info']['filename']}")
            print(f"   Dimensões: {data['query_info']['image_shape']}")
            print(f"   Resultados encontrados: {data['total_found']}")
            
            if data['results']:
                print(f"\n🎯 Top {len(data['results'])} resultados:")
                for i, result in enumerate(data['results'], 1):
                    score = result['similarity_score']
                    path = result['image_path']
                    features = result['metadata'].get('features_count', 'N/A')
                    
                    print(f"   {i}. {path}")
                    print(f"      Similaridade: {score:.4f} ({score*100:.2f}%)")
                    print(f"      Features: {features}")
                    
                    # Emoji baseado no score
                    if score > 0.7:
                        emoji = "🟢"
                    elif score > 0.4:
                        emoji = "🟡"
                    else:
                        emoji = "🔴"
                    print(f"      Confiança: {emoji}")
                    print()
            else:
                print("❌ Nenhuma imagem similar encontrada.")
            
            return True
        else:
            print(f"❌ Erro na busca: {response.status_code}")
            try:
                error_detail = response.json().get('detail', 'Erro desconhecido')
                print(f"   Detalhe: {error_detail}")
            except:
                pass
            return False
            
    except Exception as e:
        print(f"❌ Erro ao buscar imagem: {e}")
        return False

def rebuild_database():
    """Reconstrói o cache do banco de dados"""
    try:
        print("\n🔄 Reconstruindo banco de dados...")
        response = requests.post(f"{API_URL}/database/rebuild")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ {data['message']}")
            print(f"📊 Total de imagens: {data['total_images']}")
            return True
        else:
            print(f"❌ Erro ao reconstruir banco: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erro ao reconstruir banco: {e}")
        return False

def search_best_match(image_path: str):
    """Busca apenas a melhor imagem similar e salva localmente"""
    if not os.path.exists(image_path):
        print(f"❌ Arquivo não encontrado: {image_path}")
        return False
    
    try:
        print(f"\n🎯 Buscando melhor match para: {image_path}")
        
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            
            response = requests.post(f"{API_URL}/searchtest", files=files)
        
        if response.status_code == 200:
            # Obtém informações dos headers
            similarity_score = response.headers.get('X-Similarity-Score', 'N/A')
            image_path_header = response.headers.get('X-Image-Path', 'N/A')
            features_count = response.headers.get('X-Features-Count', 'N/A')
            
            print(f"✅ Melhor match encontrado!")
            print(f"📊 Similaridade: {similarity_score}")
            print(f"📁 Imagem: {image_path_header}")
            print(f"🔢 Features: {features_count}")
            
            # Salva a imagem retornada
            output_filename = f"best_match_{os.path.basename(image_path)}"
            with open(output_filename, 'wb') as f:
                f.write(response.content)
            
            print(f"💾 Imagem salva como: {output_filename}")
            return True
        else:
            print(f"❌ Erro na busca: {response.status_code}")
            try:
                error_detail = response.json().get('detail', 'Erro desconhecido')
                print(f"   Detalhe: {error_detail}")
            except:
                pass
            return False
            
    except Exception as e:
        print(f"❌ Erro ao buscar melhor match: {e}")
        return False

def show_help():
    """Mostra ajuda de uso"""
    print("""
🎯 Disney Pin Image Matching - Script de Teste

Uso:
    python test_api.py <comando> [argumentos]

Comandos disponíveis:
    info                    - Mostra informações do banco de dados
    search <caminho>        - Busca imagens similares (retorna lista)
    searchtest <caminho>    - Busca melhor match (retorna imagem)
    rebuild                 - Reconstrói o cache do banco
    test                    - Testa conexão com a API
    help                    - Mostra esta ajuda

Exemplos:
    python test_api.py info
    python test_api.py search train_image/telegram-cloud-photo-size-1-5026254558854229781-y.jpg
    python test_api.py searchtest train_image/telegram-cloud-photo-size-1-5026254558854229781-y.jpg
    python test_api.py search ./minha_imagem.jpg
    python test_api.py rebuild
    
Notas:
    - A API deve estar rodando em http://localhost:8000
    - Para iniciar a API: python main.py
    - Imagens do banco estão em: ./image_data/
    - O comando 'searchtest' salva a imagem encontrada localmente
    """)

def main():
    """Função principal"""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == "help" or command == "-h" or command == "--help":
        show_help()
    
    elif command == "test":
        test_api_connection()
    
    elif command == "info":
        if test_api_connection():
            get_database_info()
    
    elif command == "search":
        if len(sys.argv) < 3:
            print("❌ Caminho da imagem é obrigatório.")
            print("Uso: python test_api.py search <caminho_da_imagem>")
            return
        
        image_path = sys.argv[2]
        top_k = 5
        
        # Opção para especificar número de resultados
        if len(sys.argv) > 3:
            try:
                top_k = int(sys.argv[3])
            except ValueError:
                print("⚠️  Número de resultados inválido, usando 5.")
        
        if test_api_connection():
            search_image(image_path, top_k)
    
    elif command == "searchtest":
        if len(sys.argv) < 3:
            print("❌ Caminho da imagem é obrigatório.")
            print("Uso: python test_api.py searchtest <caminho_da_imagem>")
            return
        
        image_path = sys.argv[2]
        
        if test_api_connection():
            search_best_match(image_path)
    
    elif command == "rebuild":
        if test_api_connection():
            rebuild_database()
    
    else:
        print(f"❌ Comando desconhecido: {command}")
        show_help()

if __name__ == "__main__":
    main()