"""Experiment 3: strong coreset quality on Census and Covertype.

A strong coreset bounds ``|cost(C, Q) / cost(C, P) - 1|`` over *every*
k-center set, which is not computable, so we report the maximum over a
fixed pool of candidate solutions: random sets of k input points. The
pool does not depend on the coreset, so it is built once per (dataset,
k) and scored on P once; every coreset is then only scored against it.
The reported distortion is a lower bound on the true one.

    python -m coresets.exp3_kmeans.run
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from baselines import lightweight, sensitivity, uniform
from common import evaluations as ev
from datasets import loaders

OBJECTIVE = ev.L2_MEAN
SIZES = tuple(range(250, 2001, 250))
REPEATS = 20
METHODS = ("uniform", "lightweight", "sensitivity")
N_CANDIDATES = 5000
N_RESTARTS = 20          # for the reported OPT at each k
MAX_POINTS = None     # subsample; None for the full datasets

K_FACTOR = 5             # the second k is K_FACTOR times the first
K_GRID = (2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 35, 40, 50)
RHO_RESTARTS = 5         # the sweep is many optimizations; fewer restarts


def sweep_rho(data, rng, restarts, k_grid):
    """rho^2 against k over K_GRID, reusing each OPT_k twice."""
    needed = sorted(set(k_grid) | {k - 1 for k in k_grid if k > 1})
    opt = {}
    for k in needed:
        _, opt[k] = ev.best_of_restarts(data, k, rng, restarts)
        print(f"    OPT_{k} = {opt[k]:.5g}", flush=True)
    return [
        {"k": k, "rho2": opt[k] / opt[k - 1], "opt": opt[k]}
        for k in k_grid if k - 1 in opt
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/exp3"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-rho", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sizes, repeats = SIZES, REPEATS
    n_candidates, max_points = N_CANDIDATES, MAX_POINTS
    k_grid, rho_restarts = K_GRID, RHO_RESTARTS
    if args.quick:
        sizes, repeats = (250, 1000), 3
        n_candidates, max_points = 500, 20_000
        k_grid, rho_restarts = (2, 3, 5, 8), 2

    rng = np.random.default_rng(0)
    rows, stats, rho_rows = [], [], []

    for name, load in loaders.LOADERS.items():
        data = load(max_points=max_points)
        n = data.shape[0]
        k1 = loaders.DEFAULT_K[name]
        print(f"[{name}] n={n:,} d={data.shape[1]}", flush=True)

        if not args.skip_rho:
            for entry in sweep_rho(data, rng, rho_restarts, k_grid):
                rho_rows.append({"dataset": name, **entry})

        for k in (k1, K_FACTOR * k1):
            info = ev.instance_statistics(data, k, rng, n_restarts=N_RESTARTS)
            print(f"[{name}] k={k}  OPT={info.opt_k:.5g}  beta={info.beta:.3f}  "
                  f"rho^2={info.rho2:.3f}  M={info.kurtosis:.2f}", flush=True)
            stats.append({
                "dataset": name, "n": n, "d": data.shape[1], "k": k,
                "opt": info.opt_k, "beta": info.beta, "rho2": info.rho2,
                "kurtosis": info.kurtosis,
            })

            pool = ev.random_center_sets(data, k, rng, n_sets=n_candidates)
            cost_full = ev.costs_solutions(pool, data, OBJECTIVE)
            cache = sensitivity.prepare(data, k, rng, OBJECTIVE)

            for rep in range(repeats):
                for method in METHODS:
                    for size in sizes:
                        if method == "uniform":
                            q = uniform.build(data, size, rng)
                        elif method == "lightweight":
                            q = lightweight.build(data, size, rng, OBJECTIVE)
                        else:
                            q = sensitivity.build(data, cache, size, rng)

                        cost_q = ev.costs_solutions(
                            pool, q.points, OBJECTIVE, q.weights
                        )
                        values = {
                            "distortion": ev.strong_error(cost_full, cost_q),
                            "build_time_s": q.build_time_s,
                        }
                        rows += [
                            {"dataset": name, "k": k, "method": method,
                             "size": size, "rep": rep,
                             "quantity": key, "value": val}
                            for key, val in values.items()
                        ]
                print(f"  [{name}, k={k}] repetition {rep + 1}/{repeats}", flush=True)

    tables = [("distortion.csv", rows), ("datasets.csv", stats)]
    if rho_rows:
        tables.append(("rho_sweep.csv", rho_rows))
    for filename, table in tables:
        path = args.out / filename
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()