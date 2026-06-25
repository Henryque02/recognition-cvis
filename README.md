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
| 1. Dataset | ⏳ pendente |
| 2. Segmentação + rotulação (Otsu do zero) | ✅ feito |
| 3. Módulo de rejeição (borda + sobreposição) | ✅ feito |
| 4. Eixos principais via momentos centrais (do zero) | ⏳ pendente |
| 5. Classificador de distância mínima (do zero) | ⏳ pendente |
| 6. Diagrama de blocos | ⏳ pendente |
| 7. Avaliação (matriz de confusão + tempo) | ⏳ pendente |
| 2ª Parte — MLP / CNN / robustez (PyTorch) | ⏳ pendente |

### Resultados já obtidos (números reais, rodados)

- **Otsu implementado na mão** reproduz exatamente o limiar do OpenCV
  (`t = 112` nas 10 cenas); binarização concorda com `cenas_bin` em **99,99 %**
  dos pixels (IoU médio **0,997**).
- **Rotulação:** 391 componentes conexos nas 10 cenas — bate com os 364 objetos
  isolados do ground-truth; o déficit frente aos 434 totais vem das ~70
  partículas sobrepostas (que se fundem num só blob).
- **Módulo de rejeição** (ponto de operação `solidez_min = 0,90`):
  **precisão 0,954 · recall 0,874 · acurácia 0,959** contra `status_referencia`
  (FP = 4, FN = 12). Os falsos aceites são sobreposições grandes que voltam a
  parecer convexas — limite intrínseco de features geométricas.

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
│   └── avaliacao.py           # alinhamento ao GT via mapas_instancia + métricas
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
