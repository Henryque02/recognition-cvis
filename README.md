# recognition-cvis — Reconhecimento de padrões em cenas de inspeção

> **Visão Computacional (TECA2 20261) · UFG · 2026/1**
> Atividade 4 — Grupo: Henryque Oliveira · (demais integrantes)

Pipeline de **reconhecimento de partículas numa esteira de inspeção**: segmenta
cenas em tons de cinza, **rejeita** objetos cortados pela borda e aglomerados
sobrepostos, mede os **eixos principais** de cada partícula e a classifica em três
tamanhos (A grande · B média · C pequena). O projeto compara um **pipeline
clássico** (Otsu + momentos + distância mínima, implementados na mão) com
**redes neurais** (MLP sobre features e CNN ponta-a-ponta) sobre o mesmo dataset.

O contexto completo e a rubrica estão em [`CLAUDE.md`](CLAUDE.md); o enunciado
oficial em [`docs/Atividade4.pdf`](docs/Atividade4.pdf).

---

## Estado atual

| Bloco (1ª Parte) | Estado |
|---|---|
| 1. Dataset | ✅ feito (gerado + fixado) |
| 2. Segmentação + rotulação (Otsu do zero) | ✅ feito |
| 3. Módulo de rejeição (borda + sobreposição) | ✅ feito |
| 4. Eixos principais via momentos centrais (do zero) | ✅ feito |
| 5. Classificador de distância mínima (do zero) | ✅ feito |
| 6. Diagrama de blocos | ✅ feito |
| 7. Avaliação ponta-a-ponta (matriz de confusão + tempo) | ✅ feito |
| 2ª Parte — MLP / CNN / robustez (PyTorch) | ⏳ pendente |

### Resultados já obtidos (números reais, rodados)

**Bloco 2 — Segmentação + rotulação**
- **Otsu do zero** reproduz exatamente o limiar do OpenCV (`t = 112` nas 10 cenas);
  binarização concorda com `cenas_bin` em **99,99 %** dos pixels (IoU médio **0,997**).
- **Rotulação:** 391 componentes conexos nas 10 cenas — bate com os 364 objetos
  isolados do ground-truth; o déficit frente aos 434 totais vem das ~70 partículas
  sobrepostas que se fundem num único blob.

**Bloco 3 — Módulo de rejeição**
- Critérios combinados: borda (hard), área mínima, número de Euler, solidez e resíduo de elipse.
- Ponto de operação `solidez_min = 0,90`:
  **precisão 0,954 · recall 0,874 · acurácia 0,959** contra `status_referencia`
  (FP = 4, FN = 12). Os falsos aceites são sobreposições grandes que voltam a parecer
  convexas — limite intrínseco de features geométricas.

**Bloco 4 — Eixos principais via momentos centrais**
- Momentos ponderados por intensidade (sub-pixel) implementados do zero em `src/momentos.py`.
- Reproduz `eixo_*_medido_px` do gerador oficial com **erro máximo 1,41 × 10⁻¹³ px**
  (erro de ponto flutuante puro). MAE contra `cv2.fitEllipse`: ~0,37 px (maior) / ~0,34 px (menor).
- Função vetorizada `eixos_lote` processa 6000 blobs em ~311 ms (~52 µs/blob).

**Bloco 5 — Classificador de distância mínima**
- Atributo: `(eixo_maior, eixo_menor)` do Bloco 4. Protótipos: A=(26,1; 14,2), B=(20,2; 10,2), C=(15,3; 5,3) px.
- **Acurácia 100 %** em treino, validação e teste (splits oficiais, 4200/900/900).
- Matriz de confusão perfeitamente diagonal; equivalência `dⱼ(x) ↔ argmin‖x−mⱼ‖` verificada.

**Bloco 6 — Diagrama de blocos**
- Figura vetorial gerada por código, exportada em [`results/bloco6_diagrama.png`](results/bloco6_diagrama.png).

**Bloco 7 — Avaliação ponta-a-ponta nas 10 cenas**
- Pipeline completo (Blocos 2→3→4→5) sobre as cenas da esteira:
  **acurácia global 4 classes: 95,91 %** · acurácia de classe (só nos GT aceitáveis): **98,65 %**.
- Recall de rejeição: 0,874 · precisão de rejeição: 0,954 (coerente com Bloco 3).
- **Margem mínima** do discriminante no conjunto de teste: **6,10** (margem mediana 22,31) — nuvens totalmente separáveis.
- Gaps geométricos entre classes: A↔B = 2,29 px · B↔C = 3,73 px · A↔C = 10,00 px.
- **Tempo por partícula:** momentos+classificação ≈ 0,077 ms; segmentação amortizada ≈ 0,024 ms → **total ≈ 0,101 ms** — folga de **~987×** sobre orçamento de 100 ms.

Figuras exportadas em [`results/`](results/).

---

## Estrutura

```
.
├── CLAUDE.md                  # contexto fixo do projeto + rubrica
├── docs/Atividade4.pdf        # enunciado oficial
├── data/                      # dataset oficial (.npz + .json) — NÃO regenerar
├── notebooks/
│   └── 1aParte.ipynb          # pipeline clássico (Blocos 2–3 executados)
├── src/
│   ├── segmentacao.py         # Otsu (do zero) + morfologia + componentes conexas
│   ├── rejeicao.py            # critérios de descarte: borda + sobreposição
│   ├── avaliacao.py           # alinhamento ao GT via mapas_instancia + métricas
│   ├── momentos.py            # momentos brutos/centrais + eixos_principais + eixos_lote (do zero)
│   └── classificador.py       # DistanciaMinima (do zero): fit, predict, discriminantes, matriz_confusao
├── results/                   # figuras (label maps, exemplos, curva P×R)
├── pyproject.toml             # dependências (uv)
└── README.md
```

### Dataset oficial (`data/`)

Gerado por `TECA2_Gerador_Dataset_Oficial_v2.ipynb` (seed 42, scale 20 px/unid.,
supersampling 8, ±10 % nos eixos). **Não regenerar nem substituir.**

| Arquivo | Conteúdo |
|---|---|
| `dataset_blobs_isolados.npz` | 6000 recortes 32×32 rotulados (`X`, `y`, 2000/classe) |
| `splits_dataset_blobs_isolados.npz` | splits oficiais train/val/test (4200/900/900) |
| `cenas_esteira.npz` | 10 cenas 200×800: cinza, bin de referência, mapas de instância/contagem |
| `metadados_blobs_isolados.json` | eixos verdadeiros vs medidos por recorte |
| `metadados_cenas.json` | **ground-truth da rejeição** (`status_referencia` por objeto) |

---

## Como rodar

### Local (uv)

```bash
uv sync                                   # cria o ambiente a partir do pyproject
uv run jupyter lab notebooks/1aParte.ipynb
```

Executar o notebook ponta-a-ponta sem abrir a interface:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/1aParte.ipynb
```

### Google Colab

A primeira célula (bootstrap) faz `!pip install`, fixa as seeds (= 42) e
localiza `src/` + `data/`. Basta ter o repositório disponível (clonado ou via
Drive) com as pastas `src/` e `data/`.

---

## Princípios do projeto

- **Implementação na mão** do que a rubrica exige (momentos → eixos; distância
  mínima). `cv2.fitEllipse`, `regionprops`, `NearestCentroid` entram **só como
  validação cruzada**.
- **Nada de resultado inventado:** todo número sai de código que rodou.
- **Reprodutibilidade:** seeds fixas (numpy/torch/split = 42), dependências
  declaradas no `pyproject.toml`, notebooks executáveis ponta-a-ponta.
- Trabalho **por blocos**: escrever → rodar → mostrar a saída → validar contra o
  ground-truth.
