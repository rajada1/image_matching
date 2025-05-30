# 🚀 Guia do Annoy para Image Matching

## O que é o Annoy?

**Annoy** (Approximate Nearest Neighbors Oh Yeah) é uma biblioteca que acelera drasticamente a busca de vizinhos mais próximos usando indexação inteligente em árvores.

## Por que usar Annoy neste projeto?

### ⚡ Performance
- **Antes**: Busca linear O(n) - compara com TODAS as imagens
- **Depois**: Busca em árvore O(log n) - muito mais rápido!

### 📊 Números esperados
- **169 imagens**: ~5-10x mais rápido
- **1000+ imagens**: ~50-100x mais rápido
- **10000+ imagens**: ~500x mais rápido

## Como funciona

### 1. Indexação (feita uma vez)
```python
# Constrói árvores de busca otimizadas
annoy_index.build(n_trees=50)  # Mais árvores = mais precisão
```

### 2. Busca (super rápida)
```python
# Em vez de comparar com todas as imagens
similar_ids = annoy_index.get_nns_by_vector(query, search_k=100)
```

## Configurações Importantes

### `n_trees` (Qualidade do Índice)
- **Padrão**: 50 árvores
- **Mais árvores**: Maior precisão, indexação mais lenta
- **Menos árvores**: Menor precisão, indexação mais rápida

### `search_k` (Velocidade vs Precisão)
- **Padrão**: 100
- **Maior valor**: Mais preciso, um pouco mais lento
- **Menor valor**: Mais rápido, menos preciso

## Endpoints da API

### Status do Annoy
```bash
GET /
```

### Configurações
```bash
# Ver configurações
GET /performance/config

# Atualizar configurações
POST /performance/config?use_annoy=true&annoy_search_k=150
```

### Reconstruir Índice
```bash
# Reconstrói todo o banco (features + Annoy)
POST /database/rebuild

# Reconstrói apenas o índice Annoy
POST /annoy/rebuild
```

## Teste de Performance

Execute o script de teste:
```bash
python test_annoy_performance.py
```

Ou teste rápido:
```bash
python test_quick.py
```

## Dicas de Otimização

### Para seu banco atual (~169 imagens)
- `n_trees`: 30-50 (padrão é bom)
- `search_k`: 50-100 (padrão é bom)

### Para bancos maiores (1000+ imagens)
- `n_trees`: 100-200
- `search_k`: 200-500

### Para máxima velocidade
- `search_k`: 30-50
- Aceita pequena perda de precisão

### Para máxima precisão
- `search_k`: 200-500
- `n_trees`: 100+

## Quando usar cada método

### Use Annoy quando:
- ✅ Banco com 50+ imagens
- ✅ Precisa de velocidade
- ✅ Pode aceitar ~95% de precisão

### Use busca tradicional quando:
- ✅ Banco muito pequeno (<20 imagens)
- ✅ Precisa de 100% de precisão
- ✅ Desenvolvimento/debug

## Troubleshooting

### Annoy não está carregado?
1. Verifique se o arquivo `annoy_index.ann` existe
2. Execute `POST /annoy/rebuild`
3. Verifique logs de erro

### Performance não melhorou?
1. Banco pode ser muito pequeno
2. Aumente `search_k` se precisão caiu muito
3. Verifique se `use_annoy=true`

### Erro de memória?
1. Diminua `n_trees`
2. Processe banco em lotes menores

## Arquivos gerados

- `annoy_index.ann`: Índice binário do Annoy
- `annoy_mapping.json`: Mapeamento ID → imagem
- `features_cache.pkl`: Cache de features ORB
- `metadata_cache.json`: Metadados das imagens

## Exemplo de uso via API

```python
import requests

# Ativa Annoy
requests.post("http://localhost:8000/performance/config", 
              params={"use_annoy": True, "annoy_search_k": 100})

# Busca imagem
with open("minha_imagem.jpg", "rb") as f:
    response = requests.post("http://localhost:8000/search", 
                           files={"file": f})

results = response.json()
print(f"Encontrou {len(results['results'])} imagens em segundos!")
```

## Conclusão

O Annoy transforma seu sistema de **minutos** para **segundos** em bancos grandes! 🚀

Para seu projeto Disney Pin, você deve ver melhorias significativas mesmo com ~169 imagens.