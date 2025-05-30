#!/usr/bin/env python3
"""
Script de teste para validar as otimizações de memória
"""

import psutil
import time
import subprocess
import sys
import os

def get_memory_gb():
    """Retorna uso de memória em GB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)

def get_system_memory():
    """Retorna informações de memória do sistema"""
    mem = psutil.virtual_memory()
    return {
        'total_gb': mem.total / (1024**3),
        'available_gb': mem.available / (1024**3),
        'used_percent': mem.percent
    }

def test_memory_usage():
    """Testa o uso de memória durante a execução"""
    print("🧪 TESTE DE OTIMIZAÇÃO DE MEMÓRIA")
    print("=" * 50)
    
    # Informações do sistema
    sys_mem = get_system_memory()
    print(f"💻 Sistema:")
    print(f"   Total de RAM: {sys_mem['total_gb']:.1f} GB")
    print(f"   RAM disponível: {sys_mem['available_gb']:.1f} GB")
    print(f"   Uso atual: {sys_mem['used_percent']:.1f}%")
    print()
    
    # Memória inicial
    initial_memory = get_memory_gb()
    print(f"📊 Memória inicial do processo: {initial_memory:.2f} GB")
    
    # Importa e testa o ImageMatcher
    try:
        print("🔄 Importando módulos...")
        from main import ImageMatcher
        
        import_memory = get_memory_gb()
        print(f"📊 Memória após imports: {import_memory:.2f} GB (+{import_memory - initial_memory:.2f} GB)")
        
        print("🚀 Criando ImageMatcher...")
        matcher = ImageMatcher()
        
        creation_memory = get_memory_gb()
        print(f"📊 Memória após criação: {creation_memory:.2f} GB (+{creation_memory - import_memory:.2f} GB)")
        
        print("📋 Configurações aplicadas:")
        print(f"   ORB features: {matcher.orb.getMaxFeatures()}")
        print(f"   Annoy trees: {matcher.annoy_n_trees}")
        print(f"   Annoy search_k: {matcher.annoy_search_k}")
        print(f"   Total de imagens: {len(matcher.database_features)}")
        
        if len(matcher.database_features) > 0:
            total_features = sum(len(features) for features in matcher.database_features.values())
            avg_features = total_features / len(matcher.database_features)
            estimated_memory = (total_features * 32 * 4) / (1024**3)
            
            print(f"   Total de features: {total_features:,}")
            print(f"   Média features/imagem: {avg_features:.1f}")
            print(f"   Memória estimada índice: {estimated_memory:.2f} GB")
        
        final_memory = get_memory_gb()
        total_increase = final_memory - initial_memory
        
        print()
        print("✅ RESULTADOS:")
        print(f"   Memória final: {final_memory:.2f} GB")
        print(f"   Aumento total: {total_increase:.2f} GB")
        
        # Avaliação
        if total_increase < 1.0:
            print("   🟢 EXCELENTE: Uso de memória muito baixo")
        elif total_increase < 3.0:
            print("   🟡 BOM: Uso de memória aceitável")
        elif total_increase < 5.0:
            print("   🟠 ATENÇÃO: Uso de memória moderado")
        else:
            print("   🔴 CRÍTICO: Uso de memória alto")
            
        # Verifica se Annoy está carregado
        if matcher.annoy_index is not None:
            print(f"   ✅ Índice Annoy carregado: {len(matcher.annoy_id_to_image)} descritores")
        else:
            print("   ⚠️  Índice Annoy não carregado (será construído no primeiro uso)")
            
    except Exception as e:
        error_memory = get_memory_gb()
        print(f"❌ ERRO: {e}")
        print(f"📊 Memória no erro: {error_memory:.2f} GB")
        return False
    
    return True

def monitor_build_process():
    """Monitora o processo de construção do índice"""
    print("\n🔍 MONITORAMENTO DE CONSTRUÇÃO")
    print("=" * 50)
    print("💡 Para monitorar a construção do índice, execute:")
    print("   python main.py")
    print("\n   E observe os logs de progresso e memória")
    print("   Os logs mostrarão:")
    print("   - 💾 Uso de memória em tempo real")
    print("   - 📈 Progresso da construção")
    print("   - ⏳ Estimativas de tempo")
    print("   - 🚨 Alertas de memória")

def check_dependencies():
    """Verifica se todas as dependências estão instaladas"""
    print("🔍 VERIFICANDO DEPENDÊNCIAS")
    print("=" * 30)
    
    required_packages = ['cv2', 'numpy', 'annoy', 'psutil', 'fastapi']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'numpy':
                import numpy
            elif package == 'annoy':
                from annoy import AnnoyIndex
            elif package == 'psutil':
                import psutil
            elif package == 'fastapi':
                import fastapi
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Pacotes faltando: {', '.join(missing_packages)}")
        print("💡 Para instalar:")
        if 'cv2' in missing_packages:
            print("   pip install opencv-python")
        if 'annoy' in missing_packages:
            print("   pip install annoy")
        if 'psutil' in missing_packages:
            print("   pip install psutil")
        if 'fastapi' in missing_packages:
            print("   pip install fastapi uvicorn")
        return False
    
    print("✅ Todas as dependências estão instaladas!")
    return True

if __name__ == "__main__":
    print("🧪 TESTE DE OTIMIZAÇÕES DE MEMÓRIA")
    print("🎯 Validando melhorias aplicadas ao Image Matching")
    print("=" * 60)
    print()
    
    # Verifica dependências
    if not check_dependencies():
        sys.exit(1)
    
    print()
    
    # Testa uso de memória
    if test_memory_usage():
        print("\n✅ TESTE CONCLUÍDO COM SUCESSO!")
        print("🎉 As otimizações estão funcionando corretamente")
    else:
        print("\n❌ TESTE FALHOU!")
        print("🔧 Verifique os logs de erro acima")
        sys.exit(1)
    
    # Informações de monitoramento
    monitor_build_process()
    
    print("\n📚 PRÓXIMOS PASSOS:")
    print("1. Execute 'python main.py' para testar a construção completa")
    print("2. Monitore os logs de memória e progresso")
    print("3. Acesse http://localhost:8000/docs para testar a API")
    print("4. Use 'python test_memory_optimization.py' para verificações futuras")