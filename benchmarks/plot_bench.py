#!/usr/bin/env python
"""Plot SAMap benchmark results — log-log scaling curves.

Reads a CSV produced by ``bench_samap.py`` and renders one figure with
two rows (time, memory) and one column per benchmark phase. Each subplot
overlays the legacy and optimized curves on log-log axes so the scaling
difference is visible.

Usage
-----
::

    python benchmarks/plot_bench.py benchmarks/results/bench_<TIMESTAMP>.csv
    python benchmarks/plot_bench.py latest                        # most recent CSV in results/
    python benchmarks/plot_bench.py bench.csv -o scaling.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_CONFIG_STYLE = {
    "legacy": {"marker": "o", "linestyle": "-", "color": "#d62728"},
    "optimized": {"marker": "s", "linestyle": "-", "color": "#2ca02c"},
}


def _resolve_csv(arg: str) -> Path:
    """Resolve 'latest' to the most recent bench CSV in results/."""
    if arg == "latest":
        results_dir = Path(__file__).parent / "results"
        candidates = sorted(results_dir.glob("bench_*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"no bench_*.csv files in {results_dir}"
            )
        return candidates[-1]
    return Path(arg)


def plot(csv_path: Path, out_path: Path | None = None) -> Path:
    """Render scaling curves from a benchmark CSV.

    Returns the written figure path (derived from ``csv_path`` if ``out_path``
    is None).
    """
    df = pd.read_csv(csv_path)

    # Normalise timings to per-iteration so scales are comparable across phases
    # that use different n_iters.
    df["time_per_iter_s"] = df["wall_time_s"] / df["n_iters"]

    phases = sorted(df["phase"].unique())
    n_ph = len(phases)

    fig, axes = plt.subplots(
        2, n_ph,
        figsize=(4.5 * n_ph, 8),
        squeeze=False,
        sharex="col",
    )

    for j, phase in enumerate(phases):
        sub = df[df["phase"] == phase].sort_values("n_cells")

        # --- Row 0: wall time ---
        ax_t = axes[0, j]
        for cfg, style in _CONFIG_STYLE.items():
            s = sub[sub["config"] == cfg]
            if s.empty:
                continue
            ax_t.loglog(
                s["n_cells"], s["time_per_iter_s"],
                label=cfg, **style,
            )
        ax_t.set_title(phase, fontsize=12, fontweight="bold")
        ax_t.set_ylabel("wall time / iter (s)")
        ax_t.grid(True, which="both", ls="--", alpha=0.3)
        if j == 0:
            ax_t.legend(loc="upper left", fontsize=10)

        # --- Row 1: peak memory ---
        ax_m = axes[1, j]
        for cfg, style in _CONFIG_STYLE.items():
            s = sub[sub["config"] == cfg]
            if s.empty:
                continue
            ax_m.loglog(
                s["n_cells"], s["peak_mem_mb"],
                label=cfg, **style,
            )
        ax_m.set_xlabel("n_cells")
        ax_m.set_ylabel("peak memory (MiB)")
        ax_m.grid(True, which="both", ls="--", alpha=0.3)

    fig.suptitle(
        f"SAMap Phase-3 optimizations — {csv_path.name}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if out_path is None:
        out_path = csv_path.with_suffix(".png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "csv",
        help="path to bench CSV, or 'latest' for most recent in results/",
    )
    p.add_argument(
        "-o", "--out", type=Path, default=None,
        help="output figure path (default: same as CSV with .png suffix)",
    )
    args = p.parse_args(argv)

    csv_path = _resolve_csv(args.csv)
    out = plot(csv_path, args.out)
    print(f"Figure → {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
