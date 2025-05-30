# 🚀 Quick Start - Disney Pin Image Matching

## Execução Rápida

### 1. Instalar dependências
```bash
pip3 install -r requirements.txt
```

### 2. Executar exemplo completo
```bash
python3 run_example.py
```

### 3. Usar a API manualmente
```bash
# Terminal 1 - Iniciar servidor
python3 main.py

# Terminal 2 - Testar busca (retorna JSON)
python3 test_api.py search train_image/telegram-cloud-photo-size-1-5026254558854229781-y.jpg

# Terminal 2 - Testar melhor match (salva imagem)
python3 test_api.py searchtest train_image/telegram-cloud-photo-size-1-5026254558854229781-y.jpg
```

## Estrutura do Banco de Dados

Suas imagens estão em:
- **Banco:** [`image_data/`](image_data) (2 imagens)
  - `telegram-cloud-photo-size-4-5889724894395087363-x.jpg`
  - `telegram-cloud-photo-size-4-5892046401527986160-y.jpg`

- **Teste:** [`train_image/`](train_image) (1 imagem)
  - `telegram-cloud-photo-size-1-5026254558854229781-y.jpg`

## Comandos Úteis

```bash
# Ver informações do banco
python3 test_api.py info

# Buscar imagens similares (retorna lista JSON)
python3 test_api.py search <caminho_da_imagem>

# Buscar melhor match (salva imagem localmente)
python3 test_api.py searchtest <caminho_da_imagem>

# Reconstruir cache (se adicionar novas imagens)
python3 test_api.py rebuild

# Documentação da API
# Acesse: http://localhost:8000/docs
```

## Como Adicionar Novas Imagens

1. Coloque arquivos `.jpg` ou `.png` na pasta [`image_data/`](image_data)
2. Execute: `python3 test_api.py rebuild`
3. Teste com: `python3 test_api.py search sua_nova_imagem.jpg`

## Parâmetros Importantes

### No arquivo [`main.py`](main.py):
- **`nfeatures=1000`**: Máximo de features por imagem
- **`distance < 50`**: Threshold para matches válidas
- **Tamanho máximo**: 1000px (redimensiona automaticamente)

### Scores de Similaridade:
- **> 0.7**: 🟢 Alta confiança
- **0.4-0.7**: 🟡 Média confiança  
- **< 0.4**: 🔴 Baixa confiança

## Troubleshooting

**API não inicia?**
```bash
# Verificar porta em uso
lsof -i :8000

# Matar processo se necessário
kill -9 <PID>
```

**Erro de dependências?**
```bash
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

**Nenhum resultado encontrado?**
- Verifique se há imagens em `image_data/`
- Execute `python3 test_api.py rebuild`
- Ajuste threshold no código se necessário