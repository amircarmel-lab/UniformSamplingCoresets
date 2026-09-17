from __future__ import annotations

import time

import numpy as np

from common.coreset import Coreset


def build(data: np.ndarray, m: int, rng: np.random.Generator) -> Coreset:
    """Draw ``m`` points uniformly with replacement, each of weight ``1/m``.

    Parameters
    ----------
    data : (n, d) array.
    m : coreset size.
    rng : numpy Generator.
    """
    n = data.shape[0]
    t0 = time.perf_counter()
    indices = rng.integers(0, n, size=m)
    elapsed = time.perf_counter() - t0
    return Coreset.from_indices(data, indices, weights=None, build_time_s=elapsed)
