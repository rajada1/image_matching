#!/usr/bin/env python3
"""
Script de exemplo para demonstrar o uso do sistema de reconhecimento de pins Disney
"""

import os
import sys
import time
import subprocess
import threading
from pathlib import Path

def check_dependencies():
    """Verifica se as dependências estão instaladas"""
    try:
        import cv2
        import numpy
        import fastapi
        import uvicorn
        print("✅ Todas as dependências estão instaladas!")
        return True
    except ImportError as e:
        print(f"❌ Dependência faltando: {e}")
        print("Execute: pip install -r requirements.txt")
        return False

def check_images():
    """Verifica se há imagens no banco"""
    image_data_path = Path("image_data")
    train_image_path = Path("train_image")
    
    if not image_data_path.exists():
        print("❌ Pasta image_data não encontrada!")
        return False
    
    image_files = list(image_data_path.glob("*.jpg")) + list(image_data_path.glob("*.png"))
    
    if not image_files:
        print("❌ Nenhuma imagem encontrada em image_data/")
        return False
    
    print(f"✅ Encontradas {len(image_files)} imagens no banco:")
    for img in image_files:
        print(f"   - {img.name}")
    
    # Verifica imagem de teste
    test_files = list(train_image_path.glob("*.jpg")) + list(train_image_path.glob("*.png"))
    if test_files:
        print(f"✅ Encontradas {len(test_files)} imagens de teste:")
        for img in test_files:
            print(f"   - {img.name}")
    
    return True

def run_api_server():
    """Executa o servidor da API em background"""
    try:
        # Executa o servidor em um processo separado
        process = subprocess.Popen(
            [sys.executable, "main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return process
    except Exception as e:
        print(f"❌ Erro ao iniciar servidor: {e}")
        return None

def wait_for_api(max_attempts=30):
    """Aguarda a API ficar disponível"""
    import requests
    
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://localhost:8000/", timeout=2)
            if response.status_code == 200:
                print("✅ API está rodando!")
                return True
        except:
            pass
        
        print(f"⏳ Aguardando API... ({attempt + 1}/{max_attempts})")
        time.sleep(2)
    
    return False

def run_example_search():
    """Executa um exemplo de busca"""
    import requests
    
    # Procura uma imagem de teste
    train_image_path = Path("train_image")
    test_files = list(train_image_path.glob("*.jpg")) + list(train_image_path.glob("*.png"))
    
    if not test_files:
        print("❌ Nenhuma imagem de teste encontrada em train_image/")
        return False
    
    test_image = test_files[0]
    print(f"\n🔍 Testando busca com: {test_image}")
    
    try:
        with open(test_image, 'rb') as f:
            files = {'file': (test_image.name, f, 'image/jpeg')}
            params = {'top_k': 3}
            
            response = requests.post(
                "http://localhost:8000/search", 
                files=files, 
                params=params,
                timeout=30
            )
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Busca concluída!")
            print(f"📋 Arquivo: {data['query_info']['filename']}")
            print(f"📊 Resultados: {data['total_found']}")
            
            if data['results']:
                print(f"\n🎯 Melhores matches:")
                for i, result in enumerate(data['results'], 1):
                    score = result['similarity_score']
                    path = result['image_path']
                    
                    emoji = "🟢" if score > 0.7 else "🟡" if score > 0.4 else "🔴"
                    print(f"   {i}. {path} - {score:.3f} {emoji}")
            else:
                print("⚠️  Nenhum match encontrado")
            
            return True
        else:
            print(f"❌ Erro na busca: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Erro na busca: {e}")
        return False

def main():
    """Função principal"""
    print("🎯 Disney Pin Image Matching - Exemplo de Uso")
    print("=" * 50)
    
    # 1. Verificar dependências
    print("\n1. Verificando dependências...")
    if not check_dependencies():
        return
    
    # 2. Verificar imagens
    print("\n2. Verificando imagens...")
    if not check_images():
        return
    
    # 3. Iniciar servidor
    print("\n3. Iniciando servidor da API...")
    server_process = run_api_server()
    
    if not server_process:
        return
    
    try:
        # 4. Aguardar API
        print("\n4. Aguardando API ficar disponível...")
        if not wait_for_api():
            print("❌ API não ficou disponível")
            return
        
        # 5. Executar exemplo
        print("\n5. Executando exemplo de busca...")
        success = run_example_search()
        
        if success:
            print("\n✅ Exemplo executado com sucesso!")
            print("\n📖 Para mais opções, use:")
            print("   python test_api.py help")
            print("   http://localhost:8000/docs")
        
        # Manter servidor rodando
        print("\n🖥️  Servidor está rodando em: http://localhost:8000")
        print("⏹️  Pressione Ctrl+C para parar...")
        
        try:
            server_process.wait()
        except KeyboardInterrupt:
            print("\n⏹️  Parando servidor...")
        
    finally:
        # Terminar processo do servidor
        if server_process and server_process.poll() is None:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()

if __name__ == "__main__":
    main()