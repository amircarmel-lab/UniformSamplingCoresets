from __future__ import annotations

from dataclasses import dataclass

import numpy as np

L2_MEAN = "l2_mean"
L1_MEDIAN = "l1_median"
_OBJECTIVES = (L2_MEAN, L1_MEDIAN)
_MAX_ENTRIES = 4_000_000


# ======================================================================
# Distances and costs
# ======================================================================
def sq_l2_distances(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """``(n, m)`` matrix of squared Euclidean distances."""
    d = (
        np.einsum("ij,ij->i", points, points)[:, None]
        - 2.0 * (points @ centers.T)
        + np.einsum("ij,ij->i", centers, centers)[None, :]
    )
    return np.maximum(d, 0.0, out=d)


def l1_distances(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """``(n, m)`` matrix of l1 distances."""
    return np.abs(points[:, None, :] - centers[None, :, :]).sum(axis=2)


def pairwise_costs(
    points: np.ndarray, centers: np.ndarray, objective: str
) -> np.ndarray:
    """``(n, m)`` matrix of ``dist(p, c)^z`` for the given objective."""
    if objective == L2_MEAN:
        return sq_l2_distances(points, centers)
    if objective == L1_MEDIAN:
        return l1_distances(points, centers)
    raise ValueError(f"objective must be one of {_OBJECTIVES}, got {objective!r}")


def _weights(weights: np.ndarray | None, n: int) -> np.ndarray:
    if weights is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n,):
        raise ValueError(f"weights must have shape ({n},), got {w.shape}")
    return w


def _grouped_costs(
    points: np.ndarray,
    centers: np.ndarray,
    objective: str,
    weights: np.ndarray | None,
    k: int,
) -> np.ndarray:
    """Cost of each consecutive group of ``k`` rows of ``centers``.

    Chunks over points so the working matrix stays bounded; l1 needs an
    extra factor of d because its distance is not a matrix product.
    """
    points = np.ascontiguousarray(points, dtype=np.float64)
    centers = np.ascontiguousarray(centers, dtype=np.float64)
    n, d = points.shape
    if centers.shape[1] != d:
        raise ValueError(f"centers has d={centers.shape[1]}, points has d={d}")
    w = _weights(weights, n)
    n_groups = centers.shape[0] // k
    per_row = centers.shape[0] * (d if objective == L1_MEDIAN else 1)
    step = max(1, min(n, _MAX_ENTRIES // max(per_row, 1)))

    out = np.zeros(n_groups)
    for start in range(0, n, step):
        stop = min(start + step, n)
        dist = pairwise_costs(points[start:stop], centers, objective)
        out += w[start:stop] @ dist.reshape(stop - start, n_groups, k).min(axis=2)
    return out


def costs_single_centers(
    centers: np.ndarray,
    points: np.ndarray,
    objective: str,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Cost of each of ``m`` single centers. ``centers`` is ``(m, d)``."""
    return _grouped_costs(points, centers, objective, weights, k=1)


def costs_solutions(
    solutions: np.ndarray,
    points: np.ndarray,
    objective: str,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Cost of each of ``S`` solutions. ``solutions`` is ``(S, k, d)``."""
    solutions = np.asarray(solutions, dtype=np.float64)
    if solutions.ndim != 3:
        raise ValueError(f"solutions must be (S, k, d), got {solutions.shape}")
    s, k, d = solutions.shape
    return _grouped_costs(points, solutions.reshape(s * k, d), objective, weights, k)


def cost(
    centers: np.ndarray,
    points: np.ndarray,
    objective: str,
    weights: np.ndarray | None = None,
) -> float:
    """Cost of one center set. ``centers`` is ``(k, d)`` or ``(d,)``."""
    centers = np.atleast_2d(np.asarray(centers, dtype=np.float64))
    return float(_grouped_costs(points, centers, objective, weights, centers.shape[0])[0])


def nearest_center(
    points: np.ndarray, centers: np.ndarray, objective: str
) -> tuple[np.ndarray, np.ndarray]:
    """``(labels, costs)``: nearest center of each point and its ``dist^z``."""
    points = np.ascontiguousarray(points, dtype=np.float64)
    n, d = points.shape
    labels = np.empty(n, dtype=np.int64)
    best = np.empty(n)
    per_row = centers.shape[0] * (d if objective == L1_MEDIAN else 1)
    step = max(1, min(n, _MAX_ENTRIES // max(per_row, 1)))
    for start in range(0, n, step):
        stop = min(start + step, n)
        dist = pairwise_costs(points[start:stop], centers, objective)
        labels[start:stop] = dist.argmin(axis=1)
        best[start:stop] = dist.min(axis=1)
    return labels, best


# ======================================================================
# Solvers
# ======================================================================
def optimal_center(
    points: np.ndarray, objective: str, weights: np.ndarray | None = None
) -> np.ndarray:
    """Exact 1-center optimum: the centroid, or the coordinate-wise median."""
    points = np.asarray(points, dtype=np.float64)
    w = _weights(weights, points.shape[0])
    if objective == L2_MEAN:
        return (w @ points) / w.sum()
    if objective == L1_MEDIAN:
        half, out = 0.5 * w.sum(), np.empty(points.shape[1])
        for j in range(points.shape[1]):
            order = np.argsort(points[:, j], kind="stable")
            cw = np.cumsum(w[order])
            out[j] = points[order[int(np.searchsorted(cw, half))], j]
        return out
    raise ValueError(f"objective must be one of {_OBJECTIVES}, got {objective!r}")


def kmeans_pp(
    points: np.ndarray,
    k: int,
    rng: np.random.Generator,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted k-means++ seeding, ``(k, d)``. l2 only."""
    points = np.ascontiguousarray(points, dtype=np.float64)
    n = points.shape[0]
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")
    w = _weights(weights, n)

    centers = np.empty((k, points.shape[1]))
    centers[0] = points[int(rng.choice(n, p=w / w.sum()))]
    min_sq = np.sum((points - centers[0]) ** 2, axis=1)
    for j in range(1, k):
        p = min_sq * w
        total = float(p.sum())
        idx = int(rng.integers(0, n)) if total <= 0.0 else int(rng.choice(n, p=p / total))
        centers[j] = points[idx]
        np.minimum(min_sq, np.sum((points - centers[j]) ** 2, axis=1), out=min_sq)
    return centers


def lloyd(
    points: np.ndarray,
    centers: np.ndarray,
    max_iter: int = 50,
    weights: np.ndarray | None = None,
    tol: float = 1e-8,
) -> np.ndarray:
    """Weighted Lloyd from a seeding. l2 only; empty clusters keep their center.

    ``max_iter=0`` returns the seeding unchanged; small values give
    deliberately sub-optimal solutions, which is how a candidate pool
    gets members that are gamma-approximate rather than near-optimal.
    """
    points = np.ascontiguousarray(points, dtype=np.float64)
    w = _weights(weights, points.shape[0])
    centers = np.array(centers, dtype=np.float64, copy=True)
    for _ in range(max_iter):
        labels, _ = nearest_center(points, centers, L2_MEAN)
        moved = 0.0
        for i in range(centers.shape[0]):
            mask = labels == i
            wi = w[mask].sum()
            if wi <= 0.0:
                continue
            new = (w[mask] @ points[mask]) / wi
            moved = max(moved, float(np.linalg.norm(new - centers[i])))
            centers[i] = new
        if moved <= tol:
            break
    return centers


def best_of_restarts(
    points: np.ndarray,
    k: int,
    rng: np.random.Generator,
    n_restarts: int = 10,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    best_centers, best_cost = None, np.inf
    for _ in range(n_restarts):
        centers = lloyd(points, kmeans_pp(points, k, rng, weights), weights=weights)
        c = cost(centers, points, L2_MEAN, weights)
        if c < best_cost:
            best_centers, best_cost = centers, c
    return best_centers, float(best_cost)


@dataclass(frozen=True)
class InstanceStats:
    opt_k: float
    opt_km1: float
    rho2: float
    beta: float
    kurtosis: float


def instance_statistics(
    points: np.ndarray, k: int, rng: np.random.Generator, n_restarts: int = 10
) -> InstanceStats:
    """Estimate separation, balancedness and kurtosis of a k-means instance.

    ``rho2 = OPT_k / OPT_{k-1}`` with both terms replaced by
    best-of-restarts estimates. Both are upper bounds on their targets,
    so the ratio bounds the true rho^2 in neither direction.
    """
    centers, opt_k = best_of_restarts(points, k, rng, n_restarts)
    _, opt_km1 = best_of_restarts(points, max(k - 1, 1), rng, n_restarts)
    labels, sq = nearest_center(points, centers, L2_MEAN)

    sizes = np.bincount(labels, minlength=k)
    kurt = 0.0
    for i in range(k):
        s = sq[labels == i]
        if s.size and s.mean() > 0.0:
            kurt = max(kurt, float((s**2).mean() / s.mean() ** 2))
    return InstanceStats(
        opt_k=opt_k,
        opt_km1=opt_km1,
        rho2=opt_k / opt_km1,
        beta=float(k * sizes.min() / points.shape[0]),
        kurtosis=kurt,
    )


# ======================================================================
# Errors over a candidate pool (each a lower bound on its supremum)
# ======================================================================
def _check(cost_full: np.ndarray, cost_coreset: np.ndarray) -> tuple:
    cp = np.asarray(cost_full, dtype=np.float64)
    cq = np.asarray(cost_coreset, dtype=np.float64)
    if cp.shape != cq.shape or cp.ndim != 1 or cp.size == 0:
        raise ValueError(f"need equal-length non-empty 1-D arrays, got {cp.shape}, {cq.shape}")
    if np.any(cp <= 0.0):
        raise ValueError("full-data costs must be positive")
    return cp, cq


def stable_error(
    cost_full: np.ndarray, cost_coreset: np.ndarray, eps: float = 0.0
) -> float:
    cp, cq = _check(cost_full, cost_coreset)
    order = np.argsort(cq, kind="stable")
    a, b = cq[order], cp[order]
    hi = np.searchsorted(a, (1.0 + eps) * a, side="right")
    return float((np.maximum.accumulate(b)[hi - 1] / b).max() - 1.0)


def strong_error(cost_full: np.ndarray, cost_coreset: np.ndarray) -> float:
    """``max |cost(c,Q)/cost(c,P) - 1|`` over the pool."""
    cp, cq = _check(cost_full, cost_coreset)
    return float(np.abs(cq / cp - 1.0).max())


def _unit_vectors(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=(n, d))
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-300)


def pool_single_center(
    center: np.ndarray,
    direction: np.ndarray,
    ray_radii: np.ndarray,
    rng: np.random.Generator,
    scale: float = 1.0,
    n_random: int = 32,
    random_radii: np.ndarray | None = None,
    data: np.ndarray | None = None,
    n_data: int = 0,
) -> np.ndarray:
    center = np.asarray(center, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    radii = np.asarray(ray_radii, dtype=np.float64).reshape(-1, 1)
    parts = [center + radii * direction, center - radii * direction]

    if n_random > 0:
        if random_radii is None:
            random_radii = scale * np.array([0.1, 0.3, 1.0, 3.0])
        dirs = _unit_vectors(n_random, center.shape[0], rng)
        parts += [center + r * dirs for r in np.asarray(random_radii, dtype=np.float64)]

    if data is not None and n_data > 0:
        idx = rng.choice(data.shape[0], size=min(n_data, data.shape[0]), replace=False)
        parts.append(np.asarray(data[idx], dtype=np.float64))
    return np.concatenate(parts, axis=0)


def random_center_sets(
    points: np.ndarray, k: int, rng: np.random.Generator, n_sets: int = 5000
) -> np.ndarray:
    return points[rng.integers(0, len(points), (n_sets, k))]
