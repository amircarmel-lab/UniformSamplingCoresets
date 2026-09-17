"""Lightweight coresets (Bachem, Lucic and Krause, KDD 2018).

Importance sampling against a single center, the optimal 1-center of the
whole input:

    p_i  =  1/(2n)  +  dist(x_i, mu)^z / (2 sum_j dist(x_j, mu)^z)

The uniform half bounds the sampling probability away from zero; the
importance half gives outliers a proportionally larger chance of being
drawn, which is exactly what uniform sampling cannot do. One pass to
find mu, one to score, so construction is O(n) -- cheaper than full
sensitivity sampling's O(kn), but still linear in the input, so still
not sublinear.

The original paper states this for k-means. We use the objective's own
``dist^z`` in place of the squared distance, so the same construction
covers the l1 1-median panel.
"""
from __future__ import annotations

import time

import numpy as np

from common.coreset import Coreset
from common.evaluations import nearest_center, optimal_center


def build(
    data: np.ndarray, m: int, rng: np.random.Generator, objective: str
) -> Coreset:
    n = data.shape[0]
    t0 = time.perf_counter()

    mu = optimal_center(data, objective)
    _, point_costs = nearest_center(data, mu[None, :], objective)
    total = float(point_costs.sum())
    if total <= 0.0:
        probs = np.full(n, 1.0 / n)  # every point coincides with mu
    else:
        probs = 0.5 / n + 0.5 * point_costs / total

    indices = rng.choice(n, size=m, replace=True, p=probs)
    weights = 1.0 / (m * n * probs[indices])
    elapsed = time.perf_counter() - t0

    return Coreset.from_indices(data, indices, weights, elapsed)
