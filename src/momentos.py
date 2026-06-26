"""
Bloco 4 — Eixos principais via momentos centrais (TECA2 Atividade 4, 1ª Parte).

Implementação DO ZERO (CLAUDE.md §0 e §4.4): os momentos são calculados na mão
a partir das somas sobre a grade de coordenadas, sem `cv2.moments`. A partir dos
momentos centrais de 2ª ordem montamos a matriz de covariância da distribuição de
massa e extraímos seus autovalores → comprimentos dos eixos.

CONVENÇÃO (idêntica à do gerador oficial `eixos_por_momentos`, que produziu os
`eixo_*_medido_px` dos metadados — ver TECA2_Gerador_Dataset_Oficial_v2.ipynb):

    • momentos PONDERADOS POR INTENSIDADE (imagem em cinza, não binária) → sub-pixel;
    • covariância C = [[mu20, mu11], [mu11, mu02]] / mu00;
    • autovalores λ1 ≥ λ2 ≥ 0;
    • comprimento TOTAL do eixo = 4·√λ.

A escolha `4·√λ` vem da elipse de densidade uniforme: uma elipse cheia de
semieixos (a, b) tem variâncias principais a²/4 e b²/4, logo a = 2√λ e o eixo
total 2a = 4√λ. É a mesma convenção de `skimage.regionprops` (`*_axis_length`).
Por reproduzir exatamente o gerador, `eixos_principais(img, ponderado=True)` deve
casar com `eixo_*_medido_px` a menos de erro de ponto flutuante — essa é a
validação central do bloco.

`cv2.fitEllipse`, `regionprops` e `NearestCentroid` entram só como validação
cruzada, nunca como substituição (CLAUDE.md §0).
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# 1. Momentos brutos e centrais — DO ZERO (sem cv2.moments)
# --------------------------------------------------------------------------- #
def _massa(img: np.ndarray, ponderado: bool) -> np.ndarray:
    """Mapa de massa float64: intensidade (ponderado) ou indicador binário."""
    I = img.astype(np.float64)
    if not ponderado:
        I = (I > 0).astype(np.float64)
    return I


def momentos_brutos(img: np.ndarray, ponderado: bool = True) -> dict:
    """
    Momentos brutos M_pq = Σ_x Σ_y x^p y^q I(x,y) para p+q ≤ 2.
    `x` é a coluna e `y` a linha (convenção de imagem, igual ao OpenCV).
    """
    I = _massa(img, ponderado)
    H, W = I.shape
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    return {
        "m00": float(I.sum()),
        "m10": float((xs * I).sum()),
        "m01": float((ys * I).sum()),
        "m20": float((xs * xs * I).sum()),
        "m02": float((ys * ys * I).sum()),
        "m11": float((xs * ys * I).sum()),
    }


def momentos_centrais(img: np.ndarray, ponderado: bool = True) -> dict:
    """
    Momentos centrais de 2ª ordem a partir dos brutos, via fórmula de translação
    para o centroide (cx, cy) = (M10/M00, M01/M00):
        mu20 = M20 − cx·M10 ;  mu02 = M02 − cy·M01 ;  mu11 = M11 − cx·M01
    Retorna o centroide, m00 e os momentos centrais (NÃO normalizados).
    """
    m = momentos_brutos(img, ponderado)
    m00 = m["m00"]
    if m00 < 1e-9:                       # objeto vazio
        return {"cx": 0.0, "cy": 0.0, "m00": 0.0,
                "mu20": 0.0, "mu02": 0.0, "mu11": 0.0}
    cx, cy = m["m10"] / m00, m["m01"] / m00
    return {
        "cx": float(cx), "cy": float(cy), "m00": float(m00),
        "mu20": float(m["m20"] - cx * m["m10"]),
        "mu02": float(m["m02"] - cy * m["m01"]),
        "mu11": float(m["m11"] - cx * m["m01"]),
    }


# --------------------------------------------------------------------------- #
# 2. Eixos principais (autovalores da covariância de massa)
# --------------------------------------------------------------------------- #
def eixos_principais(img: np.ndarray, ponderado: bool = True) -> dict:
    """
    Eixos principais por momentos centrais normalizados. Devolve:
        eixo_maior, eixo_menor — comprimentos TOTAIS (px), convenção 4·√λ;
        lambda1, lambda2       — autovalores da covariância (λ1 ≥ λ2);
        angulo_graus           — orientação do eixo maior, ∈ [0, 180);
        excentricidade         — √(1 − λ2/λ1);
        cx, cy                 — centroide sub-pixel.
    `ponderado=True` reproduz a referência do gerador (`eixo_*_medido_px`).
    """
    mc = momentos_centrais(img, ponderado)
    m00 = mc["m00"]
    if m00 < 1e-9:
        return {"eixo_maior": 0.0, "eixo_menor": 0.0, "lambda1": 0.0,
                "lambda2": 0.0, "angulo_graus": 0.0, "excentricidade": 0.0,
                "cx": 0.0, "cy": 0.0}

    a = mc["mu20"] / m00           # var em x
    c = mc["mu02"] / m00           # var em y
    b = mc["mu11"] / m00           # covar xy
    cov = np.array([[a, b], [b, c]], dtype=np.float64)

    # autovalores de matriz simétrica 2x2 (eigvalsh dá ordem crescente)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    l1, l2 = float(max(ev[0], 0.0)), float(max(ev[1], 0.0))

    eixo_maior = 4.0 * np.sqrt(l1)
    eixo_menor = 4.0 * np.sqrt(l2)

    # orientação do eixo maior (mesma fórmula de regionprops/Hu)
    ang = 0.5 * np.arctan2(2.0 * b, a - c)        # rad, ∈ (−π/2, π/2]
    ang_graus = float(np.degrees(ang) % 180.0)

    exc = float(np.sqrt(1.0 - l2 / l1)) if l1 > 0 else 0.0

    return {
        "eixo_maior": float(eixo_maior), "eixo_menor": float(eixo_menor),
        "lambda1": l1, "lambda2": l2,
        "angulo_graus": ang_graus, "excentricidade": exc,
        "cx": mc["cx"], "cy": mc["cy"],
    }


def eixos_lote(X: np.ndarray, ponderado: bool = True) -> np.ndarray:
    """
    Aplica `eixos_principais` a um lote de recortes `X` (N,H,W) e devolve um
    array (N, 2) = [eixo_maior, eixo_menor]. Vetoriza as somas sobre o lote
    inteiro para ser rápido nos 6000 blobs (mesma matemática de `eixos_principais`).
    """
    Xf = X.astype(np.float64)
    if not ponderado:
        Xf = (Xf > 0).astype(np.float64)
    N, H, W = Xf.shape
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)

    m00 = Xf.sum(axis=(1, 2))
    m10 = (Xf * xs).sum(axis=(1, 2))
    m01 = (Xf * ys).sum(axis=(1, 2))
    m20 = (Xf * xs * xs).sum(axis=(1, 2))
    m02 = (Xf * ys * ys).sum(axis=(1, 2))
    m11 = (Xf * xs * ys).sum(axis=(1, 2))

    seguro = m00 > 1e-9
    m00s = np.where(seguro, m00, 1.0)
    cx, cy = m10 / m00s, m01 / m00s
    a = (m20 - cx * m10) / m00s
    c = (m02 - cy * m01) / m00s
    b = (m11 - cx * m01) / m00s

    # autovalores fechados da 2x2 simétrica
    tr, det = a + c, a * c - b * b
    disc = np.sqrt(np.maximum((a - c) ** 2 + 4.0 * b * b, 0.0))
    l1 = np.maximum((tr + disc) / 2.0, 0.0)
    l2 = np.maximum((tr - disc) / 2.0, 0.0)

    out = np.zeros((N, 2), dtype=np.float64)
    out[:, 0] = np.where(seguro, 4.0 * np.sqrt(l1), 0.0)
    out[:, 1] = np.where(seguro, 4.0 * np.sqrt(l2), 0.0)
    return out
