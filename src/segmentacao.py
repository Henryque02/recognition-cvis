"""
Bloco 2 — Segmentação + rotulação (TECA2 Atividade 4, 1ª Parte).

Pipeline clássico sobre `cenas_esteira.npz`:
    cinza -> Otsu (DO ZERO) -> morfologia mínima -> componentes conexas.

O limiar de Otsu é implementado na mão (histograma + somas cumulativas da
variância entre-classes), espelhando a lógica de `segmentation-cvis/src/otsu.py`.
As funções de biblioteca (`cv2.threshold(..., THRESH_OTSU)`,
`cv2.connectedComponentsWithStats`) entram apenas como validação cruzada /
ferramenta de rotulação, conforme CLAUDE.md §0.

Convenção de polaridade: nas cenas oficiais o fundo da esteira é escuro
(nível ~0) e as partículas são claras (nível ~245). `binarizar` detecta isso
em runtime escolhendo como foreground a classe minoritária, então o objeto
sai sempre como 255 e o fundo como 0, igual a `cenas_bin`.
"""
from __future__ import annotations

import numpy as np
import cv2


# --------------------------------------------------------------------------- #
# 1. Otsu DO ZERO
# --------------------------------------------------------------------------- #
def histograma(img: np.ndarray) -> np.ndarray:
    """Histograma de 256 níveis para imagem uint8 (vetor float64 de contagens)."""
    if img.dtype != np.uint8:
        raise ValueError("histograma() espera imagem uint8 [0,255]")
    return np.bincount(img.ravel(), minlength=256).astype(np.float64)


def otsu_threshold(img: np.ndarray) -> int:
    """
    Limiar de Otsu calculado na mão maximizando a variância entre-classes.

    Para cada limiar t separamos fundo = {níveis <= t} e objeto = {níveis > t}.
    Usando as somas cumulativas:
        w(t)   = soma das probabilidades até t          (peso do fundo)
        mu(t)  = soma de i*p(i) até t                    (média acumulada)
        muT    = média global
        sigma_b(t) = (muT*w - mu)^2 / (w*(1-w))
    O limiar ótimo é argmax_t sigma_b(t). Retorna t tal que objeto = img > t.
    """
    hist = histograma(img)
    prob = hist / hist.sum()
    niveis = np.arange(256, dtype=np.float64)

    w = np.cumsum(prob)                 # peso acumulado do fundo até t
    mu = np.cumsum(prob * niveis)       # primeiro momento acumulado até t
    mu_total = mu[-1]

    denom = w * (1.0 - w)               # variância 0 quando uma classe vazia
    num = (mu_total * w - mu) ** 2
    sigma_b = np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)

    return int(np.argmax(sigma_b))


# --------------------------------------------------------------------------- #
# 2. Binarização (com detecção automática de polaridade)
# --------------------------------------------------------------------------- #
def binarizar(img: np.ndarray, t: int | None = None) -> tuple[np.ndarray, int]:
    """
    Binariza `img` (uint8) com Otsu próprio. Detecta a polaridade: assume que
    as partículas ocupam a MENOR fração de pixels da cena, então escolhe como
    foreground (255) o lado do limiar que tiver menos pixels. Retorna
    (mascara uint8 {0,255}, limiar usado).
    """
    if t is None:
        t = otsu_threshold(img)

    acima = img > t
    # foreground = classe minoritária (partículas são esparsas na esteira)
    if acima.sum() <= acima.size / 2:
        mask = acima
    else:
        mask = ~acima
    return (mask.astype(np.uint8) * 255), int(t)


# --------------------------------------------------------------------------- #
# 3. Morfologia mínima (Gonzalez, Cap. 9)
# --------------------------------------------------------------------------- #
def limpeza_morfologica(
    mask: np.ndarray, abertura: int = 3, fechamento: int = 0
) -> np.ndarray:
    """
    Limpeza morfológica mínima sobre máscara binária {0,255}.
      - abertura (erosão→dilatação) remove sal isolado / serrilhado de borda;
      - fechamento (dilatação→erosão) tapa buracos finos, se `fechamento>0`.
    `abertura`/`fechamento` são os lados (ímpares) dos elementos estruturantes
    elípticos; 0 desativa a etapa. Mantida deliberadamente leve para não fundir
    objetos vizinhos (a separação de sobrepostos é tarefa do Bloco 3).
    """
    out = mask.copy()
    if abertura and abertura > 0:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abertura, abertura))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, se)
    if fechamento and fechamento > 0:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fechamento, fechamento))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, se)
    return out


# --------------------------------------------------------------------------- #
# 4. Rotulação (componentes conexas)
# --------------------------------------------------------------------------- #
def rotular(mask: np.ndarray, connectivity: int = 8) -> dict:
    """
    Rotula componentes conexas via `cv2.connectedComponentsWithStats`.
    Retorna dict com:
        n        — nº de objetos (exclui o fundo, rótulo 0)
        labels   — mapa de rótulos HxW (0 = fundo)
        stats    — (n+1, 5): x, y, w, h, area por rótulo
        centroids— (n+1, 2): (cx, cy) por rótulo
    """
    bin01 = (mask > 0).astype(np.uint8)
    n_tot, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bin01, connectivity=connectivity
    )
    return {
        "n": int(n_tot - 1),
        "labels": labels,
        "stats": stats,
        "centroids": centroids,
    }


def filtrar_area_minima(rot: dict, area_min: int) -> dict:
    """
    Remove componentes com área < `area_min` (ruído residual), reindexando os
    rótulos de forma compacta. Opera sobre a saída de `rotular`.
    """
    labels = rot["labels"]
    stats = rot["stats"]
    novo = np.zeros_like(labels)
    keep_stats = [stats[0]]
    keep_cent = [rot["centroids"][0]]
    novo_id = 0
    for lbl in range(1, rot["n"] + 1):
        if stats[lbl, cv2.CC_STAT_AREA] >= area_min:
            novo_id += 1
            novo[labels == lbl] = novo_id
            keep_stats.append(stats[lbl])
            keep_cent.append(rot["centroids"][lbl])
    return {
        "n": novo_id,
        "labels": novo,
        "stats": np.array(keep_stats),
        "centroids": np.array(keep_cent),
    }


# --------------------------------------------------------------------------- #
# 5. Pipeline de cena
# --------------------------------------------------------------------------- #
def segmentar_cena(
    img: np.ndarray,
    abertura: int = 3,
    fechamento: int = 0,
    area_min: int = 0,
    connectivity: int = 8,
) -> dict:
    """
    Executa o pipeline completo numa cena em tons de cinza e devolve um dict
    com cada estágio intermediário (para depuração e figuras):
        limiar, mask_bin, mask_morf, n, labels, stats, centroids.
    """
    mask_bin, t = binarizar(img)
    mask_morf = limpeza_morfologica(mask_bin, abertura, fechamento)
    rot = rotular(mask_morf, connectivity)
    if area_min > 0:
        rot = filtrar_area_minima(rot, area_min)
    return {
        "limiar": t,
        "mask_bin": mask_bin,
        "mask_morf": mask_morf,
        **rot,
    }


# --------------------------------------------------------------------------- #
# 6. Utilidades de validação / visualização
# --------------------------------------------------------------------------- #
def concordancia(bin_a: np.ndarray, bin_b: np.ndarray) -> dict:
    """
    Métricas de concordância pixel-a-pixel entre duas máscaras {0,255}:
        acuracia_px — fração de pixels iguais
        iou         — interseção/união do foreground
        dice        — coeficiente de Dice do foreground
    """
    a = bin_a > 0
    b = bin_b > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return {
        "acuracia_px": float((a == b).mean()),
        "iou": float(inter / union) if union else 1.0,
        "dice": float(2 * inter / (a.sum() + b.sum())) if (a.sum() + b.sum()) else 1.0,
    }


def mapa_rotulos_colorido(labels: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Converte um mapa de rótulos num RGB uint8 com cores aleatórias estáveis
    (fundo preto). Usa seed fixa para reprodutibilidade (CLAUDE.md §0).
    """
    n = int(labels.max())
    rng = np.random.default_rng(seed)
    cores = rng.integers(60, 256, size=(n + 1, 3), dtype=np.uint8)
    cores[0] = (0, 0, 0)  # fundo
    return cores[labels]
