"""
Avaliação contra o ground-truth via `mapas_instancia` (TECA2 Atividade 4).

Alinha cada componente conexo detectado aos objetos do GT e deriva o rótulo de
referência por componente. Regra de alinhamento (CLAUDE.md §4.7):
    1 id cobre o componente   → objeto único;
    vários ids cobrem o componente → merge de sobrepostos → rejeitar.

`status_referencia` do `metadados_cenas.json` é o gabarito de aceitação.
Usado no Bloco 3 (precisão×recall da rejeição) e no Bloco 7 (matriz de confusão).
"""
from __future__ import annotations

import numpy as np


def alvo_componentes(labels: np.ndarray, inst_cena: np.ndarray,
                     meta_cena: dict) -> list[dict]:
    """
    Para cada label (1..n) de uma cena, devolve um dict de referência:
        label       — id do componente
        ids_gt      — ids de objetos do GT cobertos (de mapas_instancia)
        id_dominante— id com mais pixels no componente
        classe_id   — classe do objeto dominante
        status_ref  — status_referencia do objeto dominante
        gt_aceitar  — True sse cobre EXATAMENTE 1 id e status_ref == 'aceitar'
    A regra `gt_aceitar` trata como rejeição: bordas, fragmentos, e qualquer
    componente que funda 2+ objetos (merge) ou cujo objeto único esteja marcado
    como cortado/sobreposto no GT.
    """
    por_id = {o["id"]: o for o in meta_cena["objetos"]}
    n = int(labels.max())
    saida = []
    for lbl in range(1, n + 1):
        comp = labels == lbl
        ids = inst_cena[comp]
        ids = ids[ids >= 0]
        if ids.size == 0:
            # componente sem correspondência no GT (ruído) → rejeitar
            saida.append({"label": lbl, "ids_gt": [], "id_dominante": None,
                          "classe_id": None, "status_ref": "sem_gt",
                          "gt_aceitar": False})
            continue
        uids, cnts = np.unique(ids, return_counts=True)
        dom = int(uids[np.argmax(cnts)])
        obj = por_id.get(dom, {})
        status = obj.get("status_referencia", "?")
        gt_aceitar = (len(uids) == 1 and status == "aceitar")
        saida.append({
            "label": lbl,
            "ids_gt": [int(u) for u in uids],
            "id_dominante": dom,
            "classe_id": obj.get("classe_id"),
            "status_ref": status,
            "gt_aceitar": bool(gt_aceitar),
        })
    return saida


def matriz_confusao_binaria(pred_rejeitar: list[bool],
                            true_rejeitar: list[bool]) -> dict:
    """Matriz 2x2 aceitar/rejeitar (positivo = REJEITAR) + métricas."""
    tp = fp = fn = tn = 0
    for p, t in zip(pred_rejeitar, true_rejeitar):
        if p and t: tp += 1
        elif p and not t: fp += 1
        elif not p and t: fn += 1
        else: tn += 1
    n = max(tp + fp + fn + tn, 1)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precisao": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / (tp + fn) if tp + fn else 1.0,
        "acuracia": (tp + tn) / n,
    }
