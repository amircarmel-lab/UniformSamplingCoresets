"""Sensitivity sampling (Feldman and Langberg, STOC 2011).

Compute a cheap bicriteria clustering K of the whole input, then sample
proportionally to the standard FL11 upper bound on the sensitivity,

    sens(x_i)  =  dist(x_i, q_{c(i)})^z / cost(K_{c(i)})  +  1 / |K_{c(i)}|,

where ``c(i)`` is the cluster of ``x_i``, ``q_j`` its center and
``cost(K_j)`` its total cost. Points are drawn with replacement and
weighted by the inverse sampling probability.

The bicriteria step is a full pass over the input -- this is the
construction that uniform sampling is cheap *relative to*, and the
wall-clock gap is the point of the comparison, so the clustering time is
amortized into every reported ``build_time_s``.

Split into ``prepare`` and ``build`` because the clustering does not
depend on the coreset size: one ``prepare`` per (dataset, k, trial)
serves a whole sweep over m.

k=1 uses the exact optimal center instead of a k-means++ seeding, which
makes the reference clustering exact rather than bicriteria; the
sensitivities are then the FL11 formula with a single cluster.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from common.coreset import Coreset
from common.evaluations import (
    L2_MEAN,
    kmeans_pp,
    nearest_center,
    optimal_center,
)


@dataclass(frozen=True)
class SensitivityCache:
    """Sampling distribution over the input, plus the time it took to build."""

    probs: np.ndarray
    prepare_time_s: float
    n_centers: int


def prepare(
    data: np.ndarray,
    k: int,
    rng: np.random.Generator,
    objective: str,
    seed_multiplier: int = 2,
) -> SensitivityCache:
    """Cluster the full input and turn it into a sampling distribution.

    Parameters
    ----------
    data : (n, d) array.
    k : number of centers the coreset is meant to serve.
    rng : numpy Generator, used only for the k-means++ seeding.
    objective : scores points with this objective's ``dist^z``.
    seed_multiplier : the bicriteria uses ``seed_multiplier * k`` centers.
        The FL11 analysis only needs an O(1)-approximation with O(k)
        centers, and k-means++ run for 2k iterations supplies one.
    """
    n = data.shape[0]
    t0 = time.perf_counter()

    if k == 1:
        centers = optimal_center(data, objective)[None, :]
    else:
        if objective != L2_MEAN:
            raise ValueError(
                f"k>1 seeding is l2 only, got objective={objective!r}"
            )
        centers = kmeans_pp(data, seed_multiplier * k, rng)

    labels, point_costs = nearest_center(data, centers, objective)
    n_clusters = centers.shape[0]
    sizes = np.bincount(labels, minlength=n_clusters).astype(np.float64)
    cluster_costs = np.bincount(labels, weights=point_costs, minlength=n_clusters)

    # A cluster of zero cost (all points at its center) contributes only
    # the 1/|K_j| term; guard the division rather than dropping it.
    safe = np.where(cluster_costs > 0.0, cluster_costs, 1.0)
    sens = point_costs / safe[labels] + 1.0 / sizes[labels]

    elapsed = time.perf_counter() - t0
    return SensitivityCache(
        probs=sens / sens.sum(),
        prepare_time_s=elapsed,
        n_centers=n_clusters,
    )


def build(
    data: np.ndarray,
    cache: SensitivityCache,
    m: int,
    rng: np.random.Generator,
) -> Coreset:
    """Draw ``m`` points from the cached sensitivity distribution.

    ``build_time_s`` includes the amortized ``prepare`` time, so the
    reported construction cost is what a user would actually pay to
    obtain this coreset from scratch.
    """
    n = data.shape[0]
    t0 = time.perf_counter()
    indices = rng.choice(n, size=m, replace=True, p=cache.probs)
    weights = 1.0 / (m * n * cache.probs[indices])
    elapsed = time.perf_counter() - t0

    return Coreset.from_indices(
        data, indices, weights, cache.prepare_time_s + elapsed
    )
