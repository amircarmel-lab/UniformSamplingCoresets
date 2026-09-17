"""Experiment 1: relative error vs ordering error for 1-median in l1.

Relative error -- what is lost by solving on Q instead of on P:

    [cost(mu_Q, P) - cost(mu_P, P)] / cost(mu_P, P)

Exact, since the coordinate-wise weighted median is the exact l1
1-median of Q.

Ordering error -- how far the cost order of two candidates can be
inverted:

    max |cost(c1, P) - cost(c2, P)| / min(cost(c1, P), cost(c2, P))

over pairs with cost(c1, Q) <= cost(c2, Q). Unlike 1-mean, l1 has no
Koenig-Huygens identity. So this is a maximum over a
finite candidate pool, i.e. a lower bound, rather than an exact value.

    python -m coresets.exp1_k1.run
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from baselines import sensitivity, uniform
from common import evaluations as ev
from datasets.loaders import load_yt

OBJECTIVE = ev.L1_MEDIAN
SIZES = tuple(range(50, 1001, 50))
REPEATS = 20
METHODS = ("uniform", "sensitivity")
N_RAY, N_RANDOM, N_DATA = 10, 32, 64


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/exp1"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sizes, repeats = (SIZES, REPEATS) if not args.quick else ((50, 500, 1000), 2)
    rng = np.random.default_rng(0)

    data = load_yt()
    mu = ev.optimal_center(data, OBJECTIVE)
    var = float(ev.cost(mu, data, OBJECTIVE))
    print(f"n={data.shape[0]:,}  cost(mu_P, P)={var:.6g}")

    # The clustering behind the sensitivities does not depend on the
    # coreset size, so it is computed once; each build still reports it
    # in build_time_s, since that is what a user would pay.
    cache = sensitivity.prepare(data, 1, rng, OBJECTIVE)

    axis = np.eye(1, data.shape[1], 0).ravel()
    fixed = ev.pool_single_center(
        mu, axis, np.empty(0), rng, scale=var,
        n_random=N_RANDOM, data=data, n_data=N_DATA,
    )
    fixed_on_full = ev.costs_single_centers(fixed, data, OBJECTIVE)

    rows, moved = [], []
    for rep in range(repeats):
        for method in METHODS:
            for size in sizes:
                if method == "uniform":
                    q = uniform.build(data, size, rng)
                else:
                    q = sensitivity.build(data, cache, size, rng)

                mu_q = ev.optimal_center(q.points, OBJECTIVE, q.weights)
                delta = mu_q - mu
                delta_norm = float(np.linalg.norm(delta))
                direction = delta / delta_norm if delta_norm > 0 else axis
                moved.append(float(np.mean(delta != 0.0)))

                radii = var * np.geomspace(0.1, 4.0, N_RAY)
                ray = np.vstack([
                    ev.pool_single_center(mu_q, direction, radii, rng,
                                          n_random=0, n_data=0),
                    mu_q[None, :],  # last row: the coreset optimum
                ])
                ray_on_full = ev.costs_single_centers(ray, data, OBJECTIVE)

                cost_full = np.concatenate([fixed_on_full, ray_on_full])
                cost_q = ev.costs_single_centers(
                    np.vstack([fixed, ray]), q.points, OBJECTIVE, q.weights
                )
                values = {
                    "relative": (ray_on_full[-1] - var) / var,
                    "ordering": ev.stable_error(cost_full, cost_q),
                    "build_time_s": q.build_time_s,
                }
                rows += [
                    {"method": method, "size": size, "rep": rep,
                     "quantity": k, "value": v}
                    for k, v in values.items()
                ]
        print(f"  repetition {rep + 1}/{repeats} done")

    # The l1 median is a coordinate-wise median, so on columns that are
    # constant or mostly zero it does not move at all between P and Q.
    # If almost no coordinate moves, the relative error is identically
    # zero and the panel measures nothing.
    print(f"coordinates differing between mu_Q and mu_P: {np.mean(moved):.1%} on average")
    if np.mean(moved) < 0.05:
        print("WARNING: the sample median barely moves; this panel is degenerate.")

    path = args.out / "errors.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()