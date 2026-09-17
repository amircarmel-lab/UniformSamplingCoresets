"""The common return type of every coreset construction.

Weights always sum to one. Every cost in this project is a *normalized*
cost -- ``cost(C, P) = (1/n) sum_p dist(p, C)^z`` -- so a coreset whose
weights sum to one makes ``cost(C, Q)`` directly comparable to
``cost(C, P)`` with no per-method normalization. Uniform sampling is a
baseline like any other here: it returns weights ``1/m``, not counts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_WEIGHT_SUM_TOL = 1e-9


@dataclass(frozen=True)
class Coreset:
    """A weighted summary of a dataset.

    Attributes
    ----------
    points : np.ndarray, shape (m, d)
        The coreset points. Need not be a subset of the input.
    weights : np.ndarray, shape (m,)
        Non-negative weights summing to 1.
    build_time_s : float
        Wall-clock seconds spent constructing this coreset, including any
        cached preprocessing amortized into it (see the baselines).
    indices : np.ndarray or None, shape (m,)
        Positions in the input array, when the construction samples input
        points. ``None`` for constructions that synthesize points.
    """

    points: np.ndarray
    weights: np.ndarray
    build_time_s: float
    indices: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.points.ndim != 2:
            raise ValueError(f"points must be 2-D, got shape {self.points.shape}")
        if self.weights.ndim != 1:
            raise ValueError(f"weights must be 1-D, got shape {self.weights.shape}")
        if self.points.shape[0] != self.weights.shape[0]:
            raise ValueError(
                f"points has m={self.points.shape[0]} but "
                f"weights has m={self.weights.shape[0]}"
            )
        if not np.all(np.isfinite(self.points)):
            raise ValueError("points contains non-finite entries")
        if np.any(self.weights < 0.0):
            raise ValueError("weights must be non-negative")
        total = float(self.weights.sum())
        if abs(total - 1.0) > _WEIGHT_SUM_TOL:
            raise ValueError(
                f"weights must sum to 1, got {total!r}; use "
                f"Coreset.from_unnormalized to rescale explicitly"
            )
        if self.indices is not None and self.indices.shape != self.weights.shape:
            raise ValueError(
                f"indices has shape {self.indices.shape} but "
                f"weights has shape {self.weights.shape}"
            )

    @property
    def size(self) -> int:
        """Number of coreset points, counting duplicates."""
        return self.points.shape[0]

    @property
    def n_distinct(self) -> int:
        """Number of distinct input points, when indices are available.

        Sampling with replacement can draw the same point twice; the two
        copies are two rows here. Returns ``size`` when indices are not
        available.
        """
        if self.indices is None:
            return self.size
        return int(np.unique(self.indices).size)

    @classmethod
    def from_unnormalized(
        cls,
        points: np.ndarray,
        weights: np.ndarray,
        build_time_s: float,
        indices: np.ndarray | None = None,
    ) -> "Coreset":
        """Build a coreset from weights that do not yet sum to 1.

        Rescales ``weights`` by its sum. This is the entry point for
        Horvitz-Thompson style constructions, whose natural weights
        ``1 / (m p_i)`` sum to 1 only in expectation.
        """
        weights = np.asarray(weights, dtype=np.float64)
        total = float(weights.sum())
        if total <= 0.0:
            raise ValueError(f"total weight must be positive, got {total!r}")
        return cls(
            points=np.ascontiguousarray(points, dtype=np.float64),
            weights=weights / total,
            build_time_s=float(build_time_s),
            indices=None if indices is None else np.asarray(indices, dtype=np.int64),
        )

    @classmethod
    def from_indices(
        cls,
        data: np.ndarray,
        indices: np.ndarray,
        weights: np.ndarray | None,
        build_time_s: float,
    ) -> "Coreset":
        """Build a coreset from positions into ``data``.

        ``weights=None`` gives the unweighted (uniform) coreset ``1/m``.
        Otherwise the weights are rescaled to sum to 1.
        """
        indices = np.asarray(indices, dtype=np.int64)
        points = np.ascontiguousarray(data[indices], dtype=np.float64)
        if weights is None:
            w = np.full(indices.size, 1.0 / indices.size, dtype=np.float64)
            return cls(points=points, weights=w, build_time_s=float(build_time_s),
                       indices=indices)
        return cls.from_unnormalized(points, weights, build_time_s, indices)
