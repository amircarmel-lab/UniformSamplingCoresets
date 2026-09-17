from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PANELS = (("relative", "empirical weak coreset error (%)"), ("ordering", "empirical stable coreset error (%)"))
COLOR = {"uniform": "#1f77b4", "sensitivity": "#d62728"}
TIMING_SIZE = 500


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/exp1"))
    parser.add_argument("--spread", choices=("std", "se"), default="std")
    args = parser.parse_args()

    with (args.results / "errors.csv").open() as f:
        rows = list(csv.DictReader(f))
    series = defaultdict(lambda: defaultdict(list))
    for r in rows:
        series[(r["method"], r["quantity"])][int(r["size"])].append(float(r["value"]))

    methods = sorted({r["method"] for r in rows})
    fig, axes = plt.subplots(1, len(PANELS), figsize=(5.4 * len(PANELS), 4.0))

    print(f"{'method':<12} {'error':<10} {'slope':>7}")
    for ax, (quantity, label) in zip(axes, PANELS):
        for method in methods:
            by_size = series[(method, quantity)]
            sizes = np.array(sorted(by_size))
            vals = [np.array(by_size[s]) for s in sizes]
            mean = np.array([v.mean() for v in vals]) * 100.0
            err = np.array([v.std(ddof=1) for v in vals]) * 100.0
            if args.spread == "se":
                err = err / np.sqrt([v.size for v in vals])

            ax.plot(sizes, mean, color=COLOR.get(method), marker="o", markersize=4,
                    linewidth=1.6, label=method)
            ax.fill_between(
                sizes,
                np.maximum(mean - err, 0.0),  # the errors are non-negative
                mean + err,
                color=COLOR.get(method), alpha=0.13, linewidth=0,
            )
            good = mean > 0
            slope = np.polyfit(np.log(sizes[good]), np.log(mean[good]), 1)[0]
            print(f"{method:<12} {quantity:<10} {slope:>7.3f}")

        ax.set(xlabel="coreset size", ylabel=label, ylim=(0, None))
        ax.grid(True, which="both", alpha=0.2, linewidth=0.5)
        ax.legend(fontsize=9)

    fig.tight_layout()
    out = args.results / "exp1.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight")

    print(f"\nmean construction time at s={TIMING_SIZE}:")
    for method in methods:
        times = series[(method, "build_time_s")].get(TIMING_SIZE)
        if times:
            print(f"  {method:<12} {np.mean(times) * 1e3:8.3f} ms")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()