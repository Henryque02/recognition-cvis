# Design — 2ª Parte: Classificadores de Redes Neurais

**Data:** 2026-06-26  
**Arquivo de saída:** `notebooks/2aParte.ipynb`  
**Roteiro:** `docs/Atividade4.pdf`, Seção 3

---

## Contexto

A Parte 1 (já completa em `notebooks/1aParte.ipynb`) construiu o pipeline clássico:
segmentação → rejeição → extração de eixos por momentos → classificador de distância
mínima. Resultado: 100% de acurácia nos blobs isolados (classes linearmente separáveis).

A Parte 2 mantém **exatamente o mesmo dataset** e troca apenas o classificador,
percorrendo MLP → CNN → experimento de ativações → confronto comparativo.

---

## Decisões de design

| Decisão | Escolha |
|---|---|
| Notebook | Híbrido: importa `src/momentos.py`, PyTorch inline |
| MLP oculta | 16 neurônios |
| Ativação extra | SiLU (Swish) |
| Estrutura CNN | LeNet-style fiel ao Gonzalez Tabela 12.6 |

---

## Estrutura do notebook (`2aParte.ipynb`)

### Bloco 0 — Bootstrap

- Detecção Colab/local (mesmo helper de `1aParte.ipynb`)
- `!pip install` automático: `torch torchvision numpy opencv-python matplotlib scikit-learn`
- Seeds fixas: `SEED = 42` em `random`, `numpy`, `torch`, `torch.cuda`
- ROOT, DATA, RESULTS paths
- Import de `src/momentos` para extração de features

### Bloco 1 — Dados

Carrega:
- `dataset_blobs_isolados.npz` → `X` (6000×32×32 uint8), `y` (6000,)
- `splits_dataset_blobs_isolados.npz` → `idx_train`, `idx_val`, `idx_test`

Extrai features `(eixo_maior, eixo_menor)` via `mom.eixos_lote()` (mesmo código da Parte 1,
validado a ~1e-13 px contra o gerador).

Monta:
- `Xfeat` (6000×2 float32) — para MLP e distância mínima
- `Xcnn` (6000×1×32×32 float32, normalizado [0,1]) — para CNN
- `DataLoader` com batch_size=128, shuffle nas splits de treino

Inspeciona visualmente: 3 amostras por classe (imagem + eixos sobrepostos).

### Bloco 2 — MLP sobre características

**Arquitetura:**
```
Linear(2 → 16) → ReLU → Linear(16 → 3)
```

**Treinamento:**
- Loss: CrossEntropyLoss
- Otimizador: Adam(lr=1e-3)
- Máx 300 épocas
- Early stopping: paciência 30 (monitora val-loss), salva checkpoint do melhor modelo

**Tracking** por época: loss treino, loss val, acurácia treino, acurácia val.

**Avaliação:**
- Carrega best checkpoint, avalia no teste
- Matriz de confusão 3×3
- Acurácia de teste
- Tempo de inferência por amostra (µs)
- Comparação direta com distância mínima da Parte 1

### Bloco 3 — CNN ponta-a-ponta (LeNet-style)

**Arquitetura** (fiel ao Gonzalez Tabela 12.6, entrada 32×32):

```
Input: (B, 1, 32, 32)
Conv2d(1→6, 5×5) → Ativação → MaxPool2d(2×2)   # → (B, 6, 14, 14)
Conv2d(6→16, 5×5) → Ativação → MaxPool2d(2×2)  # → (B, 16, 5, 5)
Flatten                                           # → (B, 400)
Linear(400→120) → Ativação
Linear(120→84) → Ativação
Linear(84→3)
```

A ativação é injetada como parâmetro (`act_fn`) para permitir a troca no experimento.

**Treinamento:**
- Loss: CrossEntropyLoss
- Otimizador: Adam(lr=1e-3)
- Máx 100 épocas
- Early stopping: paciência 15 (val-loss), checkpoint do melhor modelo
- Seeds idênticas antes de cada variante

### Bloco 4 — Experimento comparativo de ativações

Treina 3 instâncias da mesma arquitetura CNN trocando apenas `act_fn`:

| Variante | `act_fn` | PyTorch |
|---|---|---|
| ReLU (baseline) | `nn.ReLU` | nativo |
| GELU | `nn.GELU` | nativo |
| SiLU (Swish) | `nn.SiLU` | nativo |

Para cada variante reporta:
- Curvas de loss e acurácia (treino × val) por época
- Acurácia final de teste
- Matriz de confusão 3×3
- Tempo médio por época (segundos)

Discussão: diferença prática neste problema (classes bem separáveis, dataset pequeno).

### Bloco 5 — Confronto de abordagens

Tabela comparando os 4 classificadores:

| Métrica | Dist. Mínima | MLP | CNN-ReLU | CNN-melhor |
|---|---|---|---|---|
| Acurácia teste | | | | |
| Tempo treino (s) | — | | | |
| Inferência/amostra (µs) | | | | |
| Robustez (score médio) | | | | |

"CNN-melhor" = variante com maior acurácia de teste (desempate: menor perda de val).

### Bloco 6 — Estresse / robustez

Aplica 5 degradações ao conjunto de teste (mesma seed para todas) e avalia todos
os classificadores:

1. **Sal-e-pimenta** — 5% dos pixels aleatórios viram 0 ou 255
2. **Oclusão parcial** — patch 8×8 zerado em posição aleatória no recorte 32×32
3. **Rotação extra** — ±45° (além da variação original do dataset)
4. **Jitter de escala** — ±30% de zoom (além dos ±10% do gerador)
5. **Suavização gaussiana** — σ=2 (simula desfoque do sensor)

Para cada degradação: tabela de acurácia por classificador. Gráfico radar ou
barras agrupadas comparando degradação relativa de cada método.

Discussão: quem degrada mais suavemente — momentos desenhados à mão vs features
aprendidas?

### Bloco 7 — Interpretabilidade

**(a) Feature maps da primeira camada conv (CNN melhor variante):**
- Seleciona 3 amostras de teste (uma por classe)
- Plota os 6 filtros aprendidos + os 6 feature maps de saída da Conv1
- Discussão: o que cada filtro parece detectar (bordas, orientação, tamanho)?

**(b) Regiões de decisão no espaço 2D (a, b):**
- Distância mínima e MLP: grade no espaço (a,b) → fronteiras de decisão analíticas
- CNN: a CNN recebe pixels, não (a,b) diretamente. Abordagem: pegar os blobs de teste,
  extrair seus (eixo_maior, eixo_menor) via `momentos`, e colorir cada ponto pela predição
  da CNN sobre o recorte original. Isso mostra em qual região do espaço (a,b) a CNN
  concorda/discorda do classificador geométrico, sem precisar gerar imagens sintéticas na grade.
- Confronto: o que a CNN aprende vs o que os momentos codificam

### Bloco 8 — Discussão: analogia com a linha de produção

- Custo × simplicidade: a CNN se justifica quando a distância mínima já resolve?
- Quando o aprendizado venceria: oclusão, variação maior, ruído, classes não-separáveis
- Tempo de inferência por partícula vs orçamento ~100 ms da esteira (todos os métodos)

---

## Especificações técnicas

- Framework: PyTorch (CPU ou GPU automaticamente via `device = torch.device(...)`)
- Seeds: `torch.manual_seed(SEED)` + `torch.cuda.manual_seed_all(SEED)` antes de cada instância
- Normalização CNN: divide por 255.0 (grayscale → [0,1])
- DataLoader: `batch_size=128`, `num_workers=0` (compatível com Colab)
- Checkpoint: `torch.save` / `torch.load` em arquivo temporário por instância
- Reprodutibilidade: todas as seeds documentadas, células ordenadas

---

## Arquivos gerados

- `results/b2_mlp_curvas.png` — curvas MLP
- `results/b2_mlp_confusao.png` — matriz de confusão MLP
- `results/b3_cnn_ativacoes_curvas.png` — curvas das 3 variantes
- `results/b3_cnn_confusao_{relu,gelu,silu}.png` — matrizes
- `results/b5_confronto_tabela.png` — tabela visual
- `results/b6_estresse.png` — gráfico de robustez
- `results/b7_feature_maps.png` — feature maps Conv1
- `results/b7_regioes_decisao.png` — regiões de decisão 2D
