# 🎯 Disney Pin Image Matching System

Sistema de reconhecimento de imagens para pins da Disney usando OpenCV e FastAPI.

## 📋 Funcionalidades

- **Reconhecimento de imagens** usando features ORB
- **API REST** com FastAPI para busca de similaridade
- **Cache inteligente** para performance otimizada
- **Pré-processamento** avançado para diferentes condições de iluminação
- **Interface de linha de comando** para testes

## 🚀 Instalação

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Preparar banco de imagens

Coloque suas imagens de pins da Disney na pasta [`image_data/`](image_data):

```
image_data/
├── pin1.jpg
├── pin2.jpg
└── ...
```

### 3. Iniciar a API

```bash
python main.py
```

A API estará disponível em: `http://localhost:8000`

## 🔍 Como Usar

### Via API REST

#### 1. Verificar status da API
```bash
curl http://localhost:8000/
```

#### 2. Obter informações do banco
```bash
curl http://localhost:8000/database/info
```

#### 3. Buscar imagens similares
```bash
curl -X POST "http://localhost:8000/search" \
     -F "file=@sua_imagem.jpg" \
     -F "top_k=5"
```

#### 4. Buscar apenas a melhor imagem similar
```bash
curl -X POST "http://localhost:8000/searchtest" \
     -F "file=@sua_imagem.jpg" \
     --output melhor_match.jpg
```

### Via Script de Teste

O arquivo [`test_api.py`](test_api.py) fornece uma interface mais amigável:

```bash
# Testar conexão
python test_api.py test

# Ver informações do banco
python test_api.py info

# Buscar imagens similares (retorna lista JSON)
python test_api.py search train_image/telegram-cloud-photo-size-1-5026254558854229781-y.jpg

# Buscar melhor match (salva imagem localmente)
python test_api.py searchtest train_image/telegram-cloud-photo-size-1-5026254558854229781-y.jpg

# Reconstruir cache
python test_api.py rebuild

# Ajuda
python test_api.py help
```

## 📊 Exemplo de Resposta

```json
{
  "query_info": {
    "filename": "meu_pin.jpg",
    "image_shape": [480, 640, 3]
  },
  "results": [
    {
      "image_path": "pin_similar.jpg",
      "similarity_score": 0.8542,
      "metadata": {
        "filename": "pin_similar.jpg",
        "features_count": 423,
        "image_shape": [500, 600, 3]
      }
    }
  ],
  "total_found": 1
}
```

## 🛠 Configuração Avançada

### Parâmetros do ORB

No arquivo [`main.py`](main.py), você pode ajustar os parâmetros do detector ORB:

```python
self.orb = cv2.ORB_create(
    nfeatures=1000,      # Número máximo de features
    scaleFactor=1.2,     # Fator de escala da pirâmide
    nlevels=8,           # Níveis da pirâmide
    edgeThreshold=31,    # Threshold para detecção de bordas
    fastThreshold=20     # Threshold do detector FAST
)
```

### Threshold de Similaridade

Ajuste o threshold para matches no método `calculate_similarity()`:

```python
good_matches = [m for m in matches if m.distance < 50]  # Ajuste este valor
```

## 📁 Estrutura do Projeto

```
.
├── main.py                 # API principal
├── test_api.py            # Script de teste
├── requirements.txt       # Dependências
├── README.md             # Este arquivo
├── image_data/           # Banco de imagens
│   ├── telegram-cloud-photo-size-4-5889724894395087363-x.jpg
│   └── telegram-cloud-photo-size-4-5892046401527986160-y.jpg
├── train_image/          # Imagens para teste
│   └── telegram-cloud-photo-size-1-5026254558854229781-y.jpg
├── features_cache.pkl    # Cache de features (auto-gerado)
└── metadata_cache.json   # Metadata das imagens (auto-gerado)
```

## 🔧 Endpoints da API

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/` | GET | Status da API |
| [`/search`](http://localhost:8000/docs#/default/search_similar_images_search_post) | POST | Busca imagens similares (retorna JSON) |
| `/searchtest` | POST | Retorna apenas a melhor imagem similar |
| `/database/info` | GET | Informações do banco |
| `/database/rebuild` | POST | Reconstrói cache |
| [`/docs`](http://localhost:8000/docs) | GET | Documentação Swagger |

## 🚨 Troubleshooting

### Problema: "API não está rodando"
- Verifique se executou `python main.py`
- Confirme que a porta 8000 não está em uso

### Problema: "Não foi possível carregar a imagem"
- Verifique se o arquivo existe
- Confirme que é um formato suportado (jpg, png, bmp, etc.)

### Problema: "Nenhuma imagem similar encontrada"
- Verifique se há imagens no diretório `image_data/`
- Execute `python test_api.py rebuild` para reconstruir o cache
- Ajuste os parâmetros de threshold

### Problema: Performance lenta
- Reduza o número de features do ORB
- Considere redimensionar imagens muito grandes
- Use SSD para armazenamento

## 📈 Performance

### Otimizações Implementadas

- **Cache de features**: Evita reprocessar imagens do banco
- **Pré-processamento**: CLAHE para equalização adaptativa
- **Redimensionamento**: Imagens grandes são reduzidas automaticamente
- **Busca eficiente**: Ordenação por distância das matches

### Benchmarks Esperados

- **Processamento inicial**: ~1-2 segundos por imagem
- **Busca em tempo real**: ~100-500ms para banco de 90k imagens
- **Precisão**: 85-95% para pins similares em condições normais

## 🔮 Próximos Passos

1. **Integração com CNN**: Adicionar features deep learning
2. **Busca vetorial**: Implementar Faiss para bancos maiores
3. **Interface web**: Criar frontend para upload visual
4. **Clustering**: Agrupar pins similares automaticamente
5. **Fine-tuning**: Treinar modelo específico para pins Disney

## 📝 Licença

Este projeto é para fins educacionais e de demonstração.