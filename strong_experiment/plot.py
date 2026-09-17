from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

COLOR = {"uniform": "#1f77b4", "lightweight": "#2ca02c", "sensitivity": "#d62728"}
STYLE = ("-", "--")          # smaller k solid, larger k dashed
TIMING_SIZE = 1000


def read(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/exp3"))
    args = parser.parse_args()

    rows = read(args.results / "distortion.csv")
    stats = read(args.results / "datasets.csv")

    series = defaultdict(lambda: defaultdict(list))
    for r in rows:
        series[(r["dataset"], int(r["k"]), r["method"], r["quantity"])][
            int(r["size"])
        ].append(float(r["value"]))

    datasets = list(dict.fromkeys(s["dataset"] for s in stats))
    methods = sorted({r["method"] for r in rows})
    ks = {d: sorted({int(s["k"]) for s in stats if s["dataset"] == d})
          for d in datasets}

    fig, axes = plt.subplots(1, len(datasets), figsize=(5.4 * len(datasets), 4.0),
                             squeeze=False)
    for ax, name in zip(axes[0], datasets):
        for style, k in zip(STYLE, ks[name]):
            for method in methods:
                by_size = series[(name, k, method, "distortion")]
                if not by_size:
                    continue
                sizes = np.array(sorted(by_size))
                vals = [np.array(by_size[s]) for s in sizes]
                mean = np.array([v.mean() for v in vals]) * 100.0
                err = np.array([v.std(ddof=1) for v in vals]) * 100.0
                ax.plot(sizes, mean, color=COLOR.get(method), linestyle=style,
                        marker="o", markersize=3.5, linewidth=1.5)
                ax.fill_between(sizes, np.maximum(mean - err, 0.0), mean + err,
                                color=COLOR.get(method), alpha=0.10, linewidth=0)
        ax.set(xlabel="coreset size", ylabel="empirical strong error (%)",
               ylim=(0, None), title=name)
        ax.grid(True, alpha=0.2, linewidth=0.5)

        method_key = [Line2D([], [], color=COLOR.get(m), marker="o",
                             markersize=3.5, linewidth=1.5, label=m)
                      for m in methods]
        k_key = [Line2D([], [], color="0.3", linestyle=st, linewidth=1.5,
                        label=f"$k$={k}")
                 for st, k in zip(STYLE, ks[name])]
        ax.add_artist(ax.legend(handles=method_key, fontsize=8, loc="upper right"))
        ax.legend(handles=k_key, fontsize=8, loc="lower left", handlelength=3.0)

    fig.tight_layout()
    out = args.results / "exp3.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight")

    rho_path = args.results / "rho_sweep.csv"
    if rho_path.exists():
        rho = read(rho_path)
        fig2, ax2 = plt.subplots(figsize=(5.4, 3.6))
        for name in datasets:
            pts = sorted(((int(r["k"]), float(r["rho2"])) for r in rho
                          if r["dataset"] == name))
            if pts:
                ax2.plot(*zip(*pts), marker="o", markersize=3.5, linewidth=1.5,
                         label=name)
        for name in datasets:
            for k in ks[name]:
                ax2.axvline(k, color="0.7", linewidth=0.7, linestyle=":")
        ax2.set(xlabel="$k$", ylabel=r"empirical separation $\hat\rho^2$",
                ylim=(0, 1.05))
        ax2.grid(True, alpha=0.2, linewidth=0.5)
        ax2.legend(fontsize=8)
        fig2.tight_layout()
        out2 = args.results / "exp3_rho.pdf"
        fig2.savefig(out2, bbox_inches="tight")
        fig2.savefig(out2.with_suffix(".png"), dpi=160, bbox_inches="tight")
        print(f"wrote {out2}")

    print(f"{'dataset':<11} {'n':>9} {'d':>4} {'k':>4} {'beta':>7} "
          f"{'rho^2':>7} {'M':>7}")
    for meta in stats:
        print(f"{meta['dataset']:<11} {int(meta['n']):>9,} {meta['d']:>4} "
              f"{meta['k']:>4} {float(meta['beta']):>7.3f} "
              f"{float(meta['rho2']):>7.3f} {float(meta['kurtosis']):>7.2f}")

    print(f"\nmean construction time (coreset size {TIMING_SIZE}):")
    for name in datasets:
        for k in ks[name]:
            for method in methods:
                times = series[(name, k, method, "build_time_s")].get(TIMING_SIZE)
                if times:
                    print(f"  {name:<11} k={k:<4} {method:<12} "
                          f"{np.mean(times) * 1e3:9.2f} ms")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()