"""
Bloco 3 — Módulo de rejeição (o "coração" da 1ª Parte, TECA2 Atividade 4).

Decide, para cada componente conexo, se ele é uma partícula **aceitável** ou se
deve ser **rejeitado**, e por quê:

    rejeitar_borda          — objeto cortado pela borda / fragmento (incompleto)
    rejeitar_sobreposicao   — blob de partículas sobrepostas (não-isolado)
    aceitar                 — partícula isolada e completa

Critérios (justificados em dados — ver notebook Bloco 3, §3.x):

  • BORDA (border clearing): se o componente toca a moldura da imagem, o objeto
    está cortado e não pode ser medido confiavelmente → rejeita. É um critério
    físico/duro (no GT captura 67/70 casos de borda com apenas ~2 falsos).
  • ÁREA MÍNIMA: fragmentos minúsculos (resíduo de segmentação) → rejeita.
  • SOBREPOSIÇÃO: partículas que se fundem num blob perdem a forma elíptica
    convexa. Sinalizadores:
       - solidez (área/área_convexa) baixa  → junção côncava entre objetos;
       - euler_number ≠ 1                   → buraco entre objetos;
       - resíduo alto de fitEllipse         → forma mal aproximada por 1 elipse.
    A solidez é o limiar **varrido** para a curva precisão×recall.

IMPORTANTE: este módulo decide só a partir de features de imagem. O
`mapas_instancia` (nº de objetos por componente) é GROUND-TRUTH e entra apenas
na avaliação (`src/avaliacao.py`), nunca na decisão.
"""
from __future__ import annotations

import numpy as np
import cv2


# Limiares default (operating point escolhido na varredura — ver notebook §3.4).
LIMIARES = {
    "area_min": 30,        # px: abaixo disso é fragmento
    "solidez_min": 0.90,   # abaixo disso = sobreposição (ponto de operação)
    "residuo_max": 0.25,   # resíduo de fitEllipse acima disso = sobreposição
}


# --------------------------------------------------------------------------- #
# Features geométricas por componente
# --------------------------------------------------------------------------- #
def _maior_contorno(mask_u8: np.ndarray):
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cnts, key=cv2.contourArea) if cnts else None


def features_componente(mask_u8: np.ndarray) -> dict:
    """
    Extrai features de forma de um recorte binário {0,255} de UM componente:
        area           — nº de pixels do objeto
        solidez        — area / area do fecho convexo (preenchido, em pixels) ∈ (0,1]
        euler_number   — 1 − nº de buracos
        excentricidade — da elipse ajustada (fitEllipse), ∈ [0,1)
        residuo_elipse — |area_elipse − area| / area  (0 = elipse perfeita)
        extent         — area / area da bbox
    """
    c = _maior_contorno(mask_u8)
    area = int((mask_u8 > 0).sum())

    # solidez baseada em PIXELS (preenche o fecho convexo e conta) → sempre ≤ 1
    solidez = 0.0
    if c is not None and len(c) >= 3:
        hull = cv2.convexHull(c)
        fill = np.zeros_like(mask_u8)
        cv2.drawContours(fill, [hull], -1, 255, thickness=-1)
        hull_area = int((fill > 0).sum())
        solidez = area / hull_area if hull_area > 0 else 0.0

    # euler = 1 − buracos (contornos internos na hierarquia CCOMP)
    cnts2, hier = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    buracos = int(sum(1 for h in hier[0] if h[3] != -1)) if hier is not None else 0
    euler_number = 1 - buracos

    excentricidade = np.nan
    residuo_elipse = np.nan
    if c is not None and len(c) >= 5:
        (_cx, _cy), (eixoA, eixoB), _ang = cv2.fitEllipse(c)
        a, b = max(eixoA, eixoB) / 2.0, min(eixoA, eixoB) / 2.0
        excentricidade = float(np.sqrt(1 - (b / a) ** 2)) if a > 0 else 0.0
        area_elipse = np.pi * a * b
        residuo_elipse = float(abs(area_elipse - area) / area) if area > 0 else np.nan

    h, w = mask_u8.shape
    extent = area / (h * w) if h * w else 0.0

    return {
        "area": area,
        "solidez": float(solidez),
        "euler_number": int(euler_number),
        "excentricidade": excentricidade,
        "residuo_elipse": residuo_elipse,
        "extent": extent,
    }


def toca_borda(stats_row: np.ndarray, H: int, W: int) -> bool:
    """True se a bbox do componente encosta na moldura da imagem (border clearing)."""
    x, y, w, h = (
        stats_row[cv2.CC_STAT_LEFT],
        stats_row[cv2.CC_STAT_TOP],
        stats_row[cv2.CC_STAT_WIDTH],
        stats_row[cv2.CC_STAT_HEIGHT],
    )
    return bool(x == 0 or y == 0 or (x + w) >= W or (y + h) >= H)


# --------------------------------------------------------------------------- #
# Decisão de rejeição
# --------------------------------------------------------------------------- #
def decidir(feats: dict, na_borda: bool, lim: dict | None = None) -> tuple[str, str]:
    """
    Aplica os critérios e devolve (status, motivo).
      status ∈ {'aceitar', 'rejeitar_borda', 'rejeitar_sobreposicao'}.
    Ordem: borda → área mínima → sobreposição → aceitar.
    """
    lim = lim or LIMIARES
    if na_borda:
        return "rejeitar_borda", "toca a moldura (objeto cortado)"
    if feats["area"] < lim["area_min"]:
        return "rejeitar_borda", f"area {feats['area']} < area_min {lim['area_min']} (fragmento)"
    if feats["euler_number"] != 1:
        return "rejeitar_sobreposicao", f"euler={feats['euler_number']} (buraco entre objetos)"
    if feats["solidez"] < lim["solidez_min"]:
        return "rejeitar_sobreposicao", f"solidez {feats['solidez']:.3f} < {lim['solidez_min']}"
    res = feats["residuo_elipse"]
    if res is not None and not np.isnan(res) and res > lim["residuo_max"]:
        return "rejeitar_sobreposicao", f"residuo_elipse {res:.3f} > {lim['residuo_max']}"
    return "aceitar", "isolada e completa"


def rejeitar_cena(seg_result: dict, H: int, W: int, lim: dict | None = None) -> list[dict]:
    """
    Roda a decisão de rejeição sobre a saída de `segmentacao.segmentar_cena`.
    Retorna uma lista (1 dict por componente) com label, status, motivo e features.
    """
    labels, stats = seg_result["labels"], seg_result["stats"]
    out = []
    for lbl in range(1, seg_result["n"] + 1):
        x, y, w, h = (
            stats[lbl, cv2.CC_STAT_LEFT], stats[lbl, cv2.CC_STAT_TOP],
            stats[lbl, cv2.CC_STAT_WIDTH], stats[lbl, cv2.CC_STAT_HEIGHT],
        )
        recorte = ((labels[y:y + h, x:x + w] == lbl).astype(np.uint8)) * 255
        feats = features_componente(recorte)
        na_borda = toca_borda(stats[lbl], H, W)
        status, motivo = decidir(feats, na_borda, lim)
        out.append({"label": lbl, "status": status, "motivo": motivo, **feats,
                    "na_borda": na_borda})
    return out


# --------------------------------------------------------------------------- #
# Varredura de limiar (curva precisão × recall) — usado no Bloco 3
# --------------------------------------------------------------------------- #
def avaliar_decisoes(comp_feats: list[dict], gt_aceitar: list[bool],
                     solidez_min: float, lim: dict | None = None) -> dict:
    """
    Avalia o módulo num conjunto de componentes para um dado `solidez_min`,
    mantendo os demais limiares fixos. `gt_aceitar[k]` = True se o componente k
    deveria ser ACEITO. Positivo = REJEITAR. Retorna precisão, recall, acurácia
    e a matriz 2x2 (tp, fp, fn, tn).
    """
    base = dict(lim or LIMIARES)
    base["solidez_min"] = solidez_min
    tp = fp = fn = tn = 0
    for f, gt_ok in zip(comp_feats, gt_aceitar):
        status, _ = decidir(f, f.get("na_borda", False), base)
        pred_rejeitar = status != "aceitar"
        true_rejeitar = not gt_ok
        if pred_rejeitar and true_rejeitar:   tp += 1
        elif pred_rejeitar and not true_rejeitar: fp += 1
        elif not pred_rejeitar and true_rejeitar: fn += 1
        else: tn += 1
    n = max(tp + fp + fn + tn, 1)
    return {
        "solidez_min": solidez_min,
        "precisao": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / (tp + fn) if tp + fn else 1.0,
        "acuracia": (tp + tn) / n,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def varredura_solidez(comp_feats, gt_aceitar, valores=None, lim=None) -> list[dict]:
    """Varre `solidez_min` e devolve a lista de métricas (para a curva P×R)."""
    if valores is None:
        valores = np.round(np.arange(0.80, 0.985, 0.005), 3)
    return [avaliar_decisoes(comp_feats, gt_aceitar, float(s), lim) for s in valores]
