#!/usr/bin/env python
"""SAMap optimization benchmark — Phase 3 wins, measured.

Compares legacy vs optimized code paths for the three SAMap iteration
phases that were rewritten in Phase 3:

* **expand** — matrix-power `_smart_expand` vs BFS `_smart_expand`
* **projection** — per-iter `_mapping_window` (rebuilds precompute) vs
  precompute-once + `_mapping_window_fast`
* **correlation** — materialized `_compute_pair_corrs(batch_size=None)`
  vs streaming `batch_size=int`

Each benchmark is phase-level (direct function call, no full SAM mock).
The toggles aren't plumbed through `_mapper` yet, so phase-level is the
cleanest way to attribute speedup to each optimization.

Usage
-----
::

    python benchmarks/bench_samap.py --max-cells 3000
    python benchmarks/bench_samap.py --max-cells 30000 --phases expand,projection
    python benchmarks/plot_bench.py benchmarks/results/bench_<TIMESTAMP>.csv

Memory measurement
------------------
Uses ``tracemalloc`` peak. This tracks numpy/scipy sparse allocations
(the bulk of what the legacy paths materialize) but misses allocations
inside numba-nopython kernels. For the configs benchmarked here, the
dominant memory cost is sparse-matrix intermediates on the legacy paths
— visible to tracemalloc.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
import tracemalloc
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as spp
from sklearn.neighbors import kneighbors_graph

# ---------------------------------------------------------------------------
# SAMap imports — the functions under test.
# ---------------------------------------------------------------------------
from samap.core.correlation import _compute_pair_corrs
from samap.core.expand import _smart_expand
from samap.core.projection import (
    _mapping_window,
    _mapping_window_fast,
    _projection_precompute,
)

# Silence the INFO logging from projection.py — distracts from timing output.
logging.getLogger("samap").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Measurement infrastructure
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Result:
    """One (phase, config, scale) measurement."""

    n_cells: int
    phase: str
    config: str
    wall_time_s: float
    peak_mem_mb: float
    n_iters: int  # how many iterations of the measured step


@contextmanager
def measure() -> Iterator[dict[str, float]]:
    """Time + peak-memory context.

    Starts a fresh tracemalloc window. On exit, the yielded dict holds
    ``wall_time_s`` and ``peak_mem_mb`` (tracemalloc peak in MiB).
    """
    # Garbage-collect first so we measure the benchmark, not leftovers.
    import gc

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    box: dict[str, float] = {}
    try:
        yield box
    finally:
        wall = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        box["wall_time_s"] = wall
        box["peak_mem_mb"] = peak / (1024 * 1024)


def run_warmup(fn: Callable[[], Any]) -> None:
    """Run ``fn`` once and discard. Forces numba JIT so it's excluded from timings."""
    fn()


# ---------------------------------------------------------------------------
# Synthetic data — per phase
# ---------------------------------------------------------------------------


def synth_knn_graph(
    n_cells: int,
    k: int,
    n_clusters: int,
    rng: np.random.Generator,
) -> tuple[spp.csr_matrix, np.ndarray]:
    """Weighted kNN graph over blob-structured points.

    Mimics a scanpy ``connectivities`` matrix: symmetric, weights in
    (0, 1], diagonal absent, cluster-local neighbourhood structure.
    Returns the graph and an integer cluster-label array.
    """
    centres = np.stack(
        [
            np.cos(2 * np.pi * np.arange(n_clusters) / n_clusters),
            np.sin(2 * np.pi * np.arange(n_clusters) / n_clusters),
        ],
        axis=1,
    ) * 10.0
    labels = rng.integers(0, n_clusters, size=n_cells)
    pts = centres[labels] + rng.normal(scale=1.0, size=(n_cells, 2))

    A = kneighbors_graph(pts, n_neighbors=k, mode="distance", include_self=False)
    A.data = np.exp(-A.data / A.data.mean())
    A = A.maximum(A.T).tocsr()
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A, labels


class _MockAdata:
    """Minimal adata stand-in for projection benchmarks.

    Provides only the fields ``_projection_precompute`` reads: ``X``,
    ``var_names``, ``var['weights']``, ``varm['PCs_SAMap']``, plus gene-name
    column slicing via ``__getitem__``.
    """

    __slots__ = ("X", "_name_to_ix", "var", "var_names", "varm")

    def __init__(
        self,
        X: spp.csr_matrix,
        var_names: np.ndarray,
        weights: np.ndarray,
        PCs: np.ndarray,
    ) -> None:
        self.X = X
        self.var_names = var_names
        self.var = pd.DataFrame({"weights": weights}, index=var_names)
        self.varm = {"PCs_SAMap": PCs}
        self._name_to_ix = {n: i for i, n in enumerate(var_names)}

    def __getitem__(self, key: tuple) -> _MockAdata:
        _, cols = key
        ix = np.array([self._name_to_ix[c] for c in cols])
        return _MockAdata(
            X=self.X[:, ix],
            var_names=self.var_names[ix],
            weights=self.var["weights"].values[ix],
            PCs=self.varm["PCs_SAMap"][ix],
        )


class _MockSAM:
    __slots__ = ("adata",)

    def __init__(self, adata: _MockAdata) -> None:
        self.adata = adata


def synth_species_pair(
    n_cells_per_species: int,
    n_genes_per_species: int,
    n_homology_edges_per_gene: int,
    density: float,
    npcs: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Build two mock species + a cross-species homology graph.

    Returns the pieces needed by projection and correlation benchmarks:
    per-species SAM mocks, concatenated gene names, and the sparse
    block-off-diagonal homology matrix.
    """
    sids = ["aa", "bb"]
    sams: dict[str, _MockSAM] = {}
    gns_list: list[np.ndarray] = []

    for sid in sids:
        var_names = np.array(
            [f"{sid}_gene{i:05d}" for i in range(n_genes_per_species)]
        )
        X = spp.random(
            n_cells_per_species,
            n_genes_per_species,
            density=density,
            format="csr",
            random_state=rng.integers(1 << 31),
            dtype=np.float32,
        )
        X.data *= 10  # roughly count-scale
        weights = rng.uniform(0.1, 1.0, n_genes_per_species)
        PCs = rng.standard_normal(
            (n_genes_per_species, npcs), dtype=np.float64
        )
        sams[sid] = _MockSAM(_MockAdata(X, var_names, weights, PCs))
        gns_list.append(var_names)

    gns = np.concatenate(gns_list)
    G = gns.size

    # Block-off-diagonal homology: species-a genes → species-b genes only.
    # ~n_homology_edges_per_gene outgoing edges per gene, random targets,
    # strictly-positive weights. Symmetrise by max.
    g_a = n_genes_per_species
    n_edges = g_a * n_homology_edges_per_gene
    src = rng.integers(0, g_a, size=n_edges)
    dst = rng.integers(g_a, G, size=n_edges)
    w = rng.uniform(0.01, 1.0, size=n_edges)
    gnnm = spp.csr_matrix((w, (src, dst)), shape=(G, G))
    gnnm = gnnm.maximum(gnnm.T)

    return {
        "sams": sams,
        "gns": gns,
        "gns_list": gns_list,
        "gnnm": gnnm,
        "sids": sids,
    }


def synth_correlation_inputs(
    n_cells_per_species: int,
    n_genes_per_species: int,
    n_pairs: int,
    knn_k: int,
    density: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Build inputs for ``_compute_pair_corrs``.

    Two-species block-diagonal expression, row-normalised averaging
    operator, and a set of cross-species gene pairs. Layout matches what
    ``_refine_corr_parallel`` feeds into the kernel.
    """
    n_a, n_b = n_cells_per_species, n_cells_per_species
    g_a, g_b = n_genes_per_species, n_genes_per_species
    N = n_a + n_b
    G = g_a + g_b

    # Averaging operator: random sparse kNN, self-loops guaranteed,
    # row-normalised. Keeping it simple (no cluster structure — doesn't
    # affect timing, only numerical output).
    rows = np.repeat(np.arange(N), knn_k)
    cols = rng.integers(0, N, size=N * knn_k)
    knn = spp.csr_matrix(
        (np.ones(N * knn_k), (rows, cols)), shape=(N, N)
    )
    knn.setdiag(1.0)
    knn.sum_duplicates()
    rs = np.asarray(knn.sum(1)).flatten()
    nnms = knn.multiply(1.0 / rs[:, None]).tocsr()

    # Block-diagonal expression. Use float32 to match production.
    Xa = spp.random(
        n_a, g_a, density=density, format="csr",
        random_state=rng.integers(1 << 31), dtype=np.float32,
    )
    Xb = spp.random(
        n_b, g_b, density=density, format="csr",
        random_state=rng.integers(1 << 31), dtype=np.float32,
    )
    Xs = spp.block_diag([Xa, Xb]).tocsc()

    # Cross-species gene pairs.
    p1 = rng.integers(0, g_a, size=n_pairs)
    p2 = rng.integers(g_a, G, size=n_pairs)
    p = np.column_stack((p1, p2)).astype(np.int64)
    ps_int = np.column_stack(
        (np.zeros(n_pairs, dtype=np.int64), np.ones(n_pairs, dtype=np.int64))
    )

    return {
        "nnms": nnms,
        "Xs": Xs,
        "p": p,
        "ps_int": ps_int,
        "sp_starts": np.array([0, n_a], dtype=np.int64),
        "sp_lens": np.array([n_a, n_b], dtype=np.int64),
        "n": N,
    }


# ---------------------------------------------------------------------------
# Phase benchmarks
# ---------------------------------------------------------------------------


def bench_expand(
    n_cells: int,
    *,
    rng: np.random.Generator,
    n_iters: int = 3,
) -> list[Result]:
    """Legacy matrix-power vs BFS neighbourhood expansion."""
    print(f"  [expand] building kNN graph: n={n_cells}", file=sys.stderr)
    nnm, labels = synth_knn_graph(
        n_cells, k=20, n_clusters=max(8, n_cells // 200), rng=rng
    )
    _, ix, counts = np.unique(labels, return_inverse=True, return_counts=True)
    K = counts[ix].astype(np.int64)

    # JIT warmup — both paths have numba kernels / scipy compile paths.
    run_warmup(lambda: _smart_expand(nnm, K.copy(), NH=3, legacy=True))
    run_warmup(lambda: _smart_expand(nnm, K.copy(), NH=3, legacy=False))

    results = []

    # --- Legacy: matrix powers --------------------------------------------
    with measure() as m:
        for _ in range(n_iters):
            _smart_expand(nnm, K.copy(), NH=3, legacy=True)
    results.append(Result(
        n_cells=n_cells, phase="expand", config="legacy",
        wall_time_s=m["wall_time_s"], peak_mem_mb=m["peak_mem_mb"],
        n_iters=n_iters,
    ))

    # --- Optimized: BFS ---------------------------------------------------
    with measure() as m:
        for _ in range(n_iters):
            _smart_expand(nnm, K.copy(), NH=3, legacy=False)
    results.append(Result(
        n_cells=n_cells, phase="expand", config="optimized",
        wall_time_s=m["wall_time_s"], peak_mem_mb=m["peak_mem_mb"],
        n_iters=n_iters,
    ))

    return results


def bench_projection(
    n_cells: int,
    *,
    rng: np.random.Generator,
    n_iters: int = 3,
) -> list[Result]:
    """Per-iter precompute rebuild vs precompute-once + fast path.

    The legacy path calls ``_mapping_window`` which internally rebuilds the
    projection precompute every call. The optimized path builds the
    precompute once (outside the timed loop — iteration-invariant) and
    calls ``_mapping_window_fast`` per iteration.

    This benchmark measures *per-iteration* cost — the one-time precompute
    cost for the optimized path is not charged to any iteration since
    SAMap runs 3+ iterations and amortizes it.
    """
    print(
        f"  [projection] building species pair: n={n_cells}/species, "
        f"genes=5000/species",
        file=sys.stderr,
    )
    synth = synth_species_pair(
        n_cells_per_species=n_cells,
        n_genes_per_species=5000,
        n_homology_edges_per_gene=5,
        density=0.08,
        npcs=50,
        rng=rng,
    )
    sams, gns, gnnm = synth["sams"], synth["gns"], synth["gnnm"]

    # JIT / cache warmup.
    run_warmup(lambda: _mapping_window(sams, gnnm, gns, K=20))

    results = []

    # --- Legacy: _mapping_window rebuilds precompute each call ------------
    with measure() as m:
        for _ in range(n_iters):
            _mapping_window(sams, gnnm, gns, K=20)
    results.append(Result(
        n_cells=n_cells, phase="projection", config="legacy",
        wall_time_s=m["wall_time_s"], peak_mem_mb=m["peak_mem_mb"],
        n_iters=n_iters,
    ))

    # --- Optimized: precompute once, fast path per iter -------------------
    # Precompute outside the measurement window — it's iteration-invariant.
    pre = _projection_precompute(sams, gns)
    with measure() as m:
        for _ in range(n_iters):
            _mapping_window_fast(gnnm, pre, K=20)
    results.append(Result(
        n_cells=n_cells, phase="projection", config="optimized",
        wall_time_s=m["wall_time_s"], peak_mem_mb=m["peak_mem_mb"],
        n_iters=n_iters,
    ))

    return results


def bench_correlation(
    n_cells: int,
    *,
    rng: np.random.Generator,
    n_iters: int = 1,
) -> list[Result]:
    """Materialised Xavg vs streaming batched correlation.

    Legacy: ``batch_size=None`` materialises the full N × G_active smoothed
    matrix. Optimized: ``batch_size`` streams pair-batches and computes only
    the columns needed per batch.
    """
    print(
        f"  [correlation] building inputs: n={n_cells}/species, "
        f"2000 genes/species, 8000 pairs",
        file=sys.stderr,
    )
    inp = synth_correlation_inputs(
        n_cells_per_species=n_cells,
        n_genes_per_species=2000,
        n_pairs=8000,
        knn_k=15,
        density=0.08,
        rng=rng,
    )

    args = (
        inp["nnms"], inp["Xs"], inp["p"], inp["ps_int"],
        inp["sp_starts"], inp["sp_lens"], inp["n"],
    )

    # Warmup (both paths share the numba _corr_kernel).
    run_warmup(lambda: _compute_pair_corrs(*args, corr_mode="pearson", batch_size=None))

    results = []

    # --- Legacy: materialised Xavg ----------------------------------------
    with measure() as m:
        for _ in range(n_iters):
            _compute_pair_corrs(*args, corr_mode="pearson", batch_size=None)
    results.append(Result(
        n_cells=n_cells, phase="correlation", config="legacy",
        wall_time_s=m["wall_time_s"], peak_mem_mb=m["peak_mem_mb"],
        n_iters=n_iters,
    ))

    # --- Optimized: streaming batches -------------------------------------
    with measure() as m:
        for _ in range(n_iters):
            _compute_pair_corrs(*args, corr_mode="pearson", batch_size=512)
    results.append(Result(
        n_cells=n_cells, phase="correlation", config="optimized",
        wall_time_s=m["wall_time_s"], peak_mem_mb=m["peak_mem_mb"],
        n_iters=n_iters,
    ))

    return results


BENCHMARKS: dict[str, Callable[..., list[Result]]] = {
    "expand": bench_expand,
    "projection": bench_projection,
    "correlation": bench_correlation,
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def write_csv(path: Path, results: list[Result]) -> None:
    """Dump results to CSV — one row per (scale, phase, config)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["n_cells", "phase", "config", "wall_time_s", "peak_mem_mb", "n_iters"]
        )
        for r in results:
            w.writerow([
                r.n_cells, r.phase, r.config,
                f"{r.wall_time_s:.6f}", f"{r.peak_mem_mb:.3f}",
                r.n_iters,
            ])


def print_summary(results: list[Result]) -> None:
    """Pretty-print speedup / memory-reduction table to stderr."""
    # Group by (phase, n_cells) → compare legacy vs optimized.
    grouped: dict[tuple[str, int], dict[str, Result]] = {}
    for r in results:
        key = (r.phase, r.n_cells)
        grouped.setdefault(key, {})[r.config] = r

    print("\n=== Summary ===", file=sys.stderr)
    print(
        f"{'phase':<12} {'n_cells':>8} {'legacy_s':>10} {'opt_s':>10} "
        f"{'speedup':>8} {'legacy_mb':>10} {'opt_mb':>10} {'mem_ratio':>9}",
        file=sys.stderr,
    )
    for (phase, n_cells), by_cfg in sorted(grouped.items()):
        if "legacy" not in by_cfg or "optimized" not in by_cfg:
            continue
        leg, opt = by_cfg["legacy"], by_cfg["optimized"]
        speedup = leg.wall_time_s / opt.wall_time_s if opt.wall_time_s > 0 else float("inf")
        mem_ratio = leg.peak_mem_mb / opt.peak_mem_mb if opt.peak_mem_mb > 0 else float("inf")
        print(
            f"{phase:<12} {n_cells:>8d} {leg.wall_time_s:>10.3f} "
            f"{opt.wall_time_s:>10.3f} {speedup:>7.1f}x "
            f"{leg.peak_mem_mb:>10.1f} {opt.peak_mem_mb:>10.1f} {mem_ratio:>8.1f}x",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--max-cells", type=int, default=10000,
        help="stop at the first scale whose n_cells exceeds this (default: 10000)",
    )
    p.add_argument(
        "--phases", default="expand,projection,correlation",
        help="comma-separated list of phases to run "
             "(expand,projection,correlation; default: all)",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="output CSV path (default: benchmarks/results/bench_<TIMESTAMP>.csv)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for synthetic data (default: 42)",
    )
    args = p.parse_args(argv)

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    for ph in phases:
        if ph not in BENCHMARKS:
            print(
                f"error: unknown phase {ph!r}; choose from {sorted(BENCHMARKS)}",
                file=sys.stderr,
            )
            return 2

    scales = [s for s in [1000, 3000, 10000, 30000] if s <= args.max_cells]
    if not scales:
        scales = [args.max_cells]

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out or (
        Path(__file__).parent / "results" / f"bench_{ts}.csv"
    )

    rng = np.random.default_rng(args.seed)
    all_results: list[Result] = []

    for n in scales:
        for ph in phases:
            print(f"[scale n_cells={n}] {ph}", file=sys.stderr)
            try:
                rs = BENCHMARKS[ph](n, rng=rng)
            except MemoryError:
                print(
                    f"  OOM at n_cells={n}, phase={ph} — stopping this phase",
                    file=sys.stderr,
                )
                continue
            all_results.extend(rs)

    write_csv(out_path, all_results)
    print(f"\nResults → {out_path}", file=sys.stderr)
    print_summary(all_results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
