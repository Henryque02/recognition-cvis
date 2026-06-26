"""
Bloco 5 — Classificador de distância mínima (**DO ZERO**, TECA2 Atividade 4).

Classificador linear por protótipos (Gonzalez & Woods, Tabela 12.x). Cada classe
j é representada pelo seu protótipo mⱼ = média dos vetores de atributo do treino.
O vetor de atributo é x = (eixo_maior, eixo_menor), extraído pelos momentos do
Bloco 4 (`src/momentos.py`).

Função de decisão (forma linear do discriminante de distância mínima):

    dⱼ(x) = xᵀ·mⱼ − ½‖mⱼ‖²,      decide j* = argmaxⱼ dⱼ(x)

Equivalência (prova local, sem biblioteca): minimizar a distância euclidiana ao
protótipo é o MESMO que maximizar dⱼ, pois
    ‖x − mⱼ‖² = ‖x‖² − 2·xᵀmⱼ + ‖mⱼ‖²
e ‖x‖² não depende de j, logo argminⱼ‖x − mⱼ‖² = argmaxⱼ (xᵀmⱼ − ½‖mⱼ‖²).
Por isso `predict` (via dⱼ) e `predict_por_distancia` (via ‖x − mⱼ‖) coincidem, e
ambos coincidem com `sklearn.NearestCentroid` (métrica euclidiana) — usado só como
validação cruzada (CLAUDE.md §0).

As fronteiras de decisão entre duas classes i, j são as **mediatrizes** dos
segmentos mᵢmⱼ: o lugar onde dᵢ(x) = dⱼ(x), isto é xᵀ(mᵢ−mⱼ) = ½(‖mᵢ‖²−‖mⱼ‖²).
"""
from __future__ import annotations

import numpy as np


class DistanciaMinima:
    """Classificador de distância mínima por protótipos (média de classe)."""

    def __init__(self) -> None:
        self.classes_: np.ndarray | None = None
        self.prototipos_: np.ndarray | None = None   # (C, D)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DistanciaMinima":
        """Protótipo de cada classe = média dos atributos de treino dessa classe."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.prototipos_ = np.stack([X[y == c].mean(axis=0) for c in self.classes_])
        return self

    def discriminantes(self, X: np.ndarray) -> np.ndarray:
        """Matriz (N, C) com dⱼ(x) = xᵀmⱼ − ½‖mⱼ‖² para cada amostra e classe."""
        X = np.asarray(X, dtype=np.float64)
        M = self.prototipos_
        return X @ M.T - 0.5 * np.sum(M * M, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Rótulo = argmaxⱼ dⱼ(x) (forma linear do discriminante)."""
        return self.classes_[np.argmax(self.discriminantes(X), axis=1)]

    def distancias(self, X: np.ndarray) -> np.ndarray:
        """Matriz (N, C) das distâncias euclidianas de cada amostra a cada protótipo."""
        X = np.asarray(X, dtype=np.float64)
        difs = X[:, None, :] - self.prototipos_[None, :, :]
        return np.sqrt(np.sum(difs * difs, axis=2))

    def predict_por_distancia(self, X: np.ndarray) -> np.ndarray:
        """Rótulo = argminⱼ‖x − mⱼ‖ — deve ser idêntico a `predict` (ver docstring)."""
        return self.classes_[np.argmin(self.distancias(X), axis=1)]


def acuracia(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fração de acertos."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def matriz_confusao(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Matriz de confusão (n_classes × n_classes), linhas = verdadeiro, colunas = previsto."""
    M = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        M[int(t), int(p)] += 1
    return M
