# SAMap Performance Guide

SAMap 3.0 is a substantial rewrite of the core algorithm's memory model.
The algorithm is mathematically unchanged, but every step that previously
materialized an **N × G** or **N × N** intermediate has been rewritten to
stream or precompose. This document explains the new memory model, how to
select and tune the compute backend, and what to expect when scaling.

---

## Summary: what changed in 3.0

| Phase | Legacy (≤2.x) | 3.0 | Memory complexity |
|---|---|---|---|
| **Feature translation** | Materialize `Xtr = X @ G` (cells × genes, ~30% dense) per iteration | Precompose projection, one SpMM per species pair | O(N × G) → O(N × npcs) |
| **Neighbourhood expansion** | Matrix powers + LIL zeroing | BFS with per-row budget, numba kernel | O(N²) peak → O(N · k · NH) |
| **Mutual-NN stitching** | Full N × N products, chunked | Streaming per-species-pair with direct CSR output | O(N²) dense peak → O(N · k) sparse |
| **Correlation refinement** | Materialize full smoothed expression `Xavg = nnms @ Xs` | Stream pair batches; compute only columns referenced per batch | O(N × G_active) → O(N × 2·batch_size) |
| **Sparse PCA** | ARPACK | ARPACK or randomized SVD with implicit centering | Same; randomized is faster on GPU |

In all cases the **numeric output is equivalent** (correlation streaming is
bit-identical to materialized; projection precomposition agrees to ~1e-15)
with one exception: the BFS neighbourhood expansion avoids a self-loop
artefact of the matrix-power path and selects slightly different marginal
neighbours (~1% edge difference on the golden-suite data). We consider BFS
strictly more correct; pass `_smart_expand(..., legacy=True)` for exact 2.x
reproduction.

---

## The memory model

SAMap iterates a two-phase loop: **project** cells into a joint latent
space via the current homology graph, then **refine** the homology graph
from the resulting cross-species neighbourhood. Each phase had an N × G
or N × N dense chokepoint.

### Projection: precomposed feature translation

The legacy code computed, per iteration and per species pair,

```
Xtr = X_i @ G_ij          # N_i × G_j, ~30% dense — the bottleneck
Xscaled = Xtr / σ
wpca = (Xscaled * W_j) @ PCs_j
```

and assembled a block matrix of these before projecting through the PC
loadings. For realistic data this intermediate dominates both memory and
wall time.

The 3.0 path observes that the entire chain is a linear operator and can
be precomposed:

```
P_ij = G_ij · diag(W_j / σ) · PCs_j    # G_i × npcs — a few MB, regardless of N
wpca = X_i @ P_ij                       # ONE SpMM
```

The per-column standard deviation `σ` (which depends on `Xtr`) is recovered
without materializing `Xtr` via a quadratic form in the columns of `G_ij`,
using iteration-invariant precomputes of `X_iᵀX_i` and `X_i.mean(0)`. The
own-species contribution `X_i @ PCs_i` does not depend on the homology
graph at all and is computed once at the start of the run.

**When it kicks in:** Always on. The iteration-invariant state is built in
`_Samap_Iter.__init__`; each iteration runs `_mapping_window_fast` which
consumes the cached state.

### Coarsening: streaming mutual-NN

The legacy mutual-NN step built intermediate N × N products (kNN graph ×
expanded neighbourhood) chunked by rows but still O(N²)-dense at peak. The
3.0 path streams per-species-pair blocks and emits the final sparse kNN
directly via a COO builder — no dense intermediate.

**When it kicks in:** Always on. Tunable via the internal `chunksize`
parameter on `_mapper` (default 20 000 rows), but this is not currently
exposed in the public `SAMAP.run()` API.

### Correlation refinement: batched smoothed expression

The legacy path materialized `Xavg = nnms @ Xs` — an N × G_active dense
matrix — so the per-pair correlation kernel could pull columns by index.
At million-cell scale this is multiple GB.

The 3.0 default (`batch_size=512`) streams: for each batch of 512 gene
pairs, compute only the ≤1024 columns of `Xavg` actually referenced,
correlate, discard. Peak memory drops to O(N × 1024) regardless of how
many genes are active. Columns that appear in multiple batches are
recomputed — this is a cheap single-column SpMV and empirically <5%
overhead at scale.

**Trade-off:** At small scale (<10k cells), where the full `Xavg` fits
comfortably in memory, the streaming path is ~3-5× *slower* than the
materialized path (benchmark: 3.6× at 3k cells). The default is tuned for
large-scale runs where memory, not speed, is the constraint. Pass
`batch_size=None` to `_refine_corr` / `_refine_corr_parallel` to opt out.

**When it kicks in:** Default-on with `batch_size=512`. Not currently
exposed as a top-level `SAMAP.run()` parameter — tune via the internal
`_refine_corr` call if needed.

### Neighbourhood expansion: BFS

The legacy `_smart_expand` used repeated matrix powers with LIL zeroing
to collect an NH-hop neighbourhood per cell. This wastes one budget slot
per cell on a self-loop artefact (a cell's 2-hop neighbourhood always
includes itself) and has O(N²) peak memory for the power products.

The 3.0 default is a numba BFS kernel that walks neighbours directly,
tracks a per-row visited set, and respects the budget exactly.

**When it kicks in:** Default-on (`legacy=False`). Pass `legacy=True` to
`_smart_expand` for bit-exact 2.x reproduction.

---

## Backend selection (CPU / GPU)

```python
from samap import SAMAP

sm = SAMAP(sams={...}, backend="auto")   # pick CUDA if available, else CPU
sm = SAMAP(sams={...}, backend="cpu")    # force numpy/scipy
sm = SAMAP(sams={...}, backend="cuda")   # force cupy/cupyx — raises if unavailable
```

`"auto"` resolves to `"cuda"` if `cupy` is importable and a GPU is
detected, otherwise `"cpu"`. The resolved device is logged at construction.

### GPU installation

```bash
pip install "sc-samap[gpu]"
```

This pulls:

- `cupy-cuda12x` — numpy/scipy dispatch on CUDA 12.x. For CUDA 11.x,
  install `cupy-cuda11x` directly.
- `faiss-gpu` — GPU approximate kNN. **Note:** wheels are not on PyPI;
  install via conda (`pytorch` or `conda-forge` channel). The pip extra
  is advisory.
- `rapids-singlecell` — GPU Leiden/UMAP. Best installed from the
  `rapidsai` conda channel.

The kNN dispatch (`approximate_knn` in `samap.core.knn`) picks FAISS on
GPU and hnswlib on CPU automatically.

---

## Tuning parameters

Most of these live on internal functions; they are not (yet) plumbed
through the public `SAMAP.run()` API.

| Parameter | Location | Default | When to change |
|---|---|---|---|
| `batch_size` | `_refine_corr`, `_refine_corr_parallel` | `512` | Lower (256, 128) on severe memory pressure. `None` for speed on small datasets. |
| `chunksize` | `_mapper` (coarsening) | `20000` | Lower if the streaming mutual-NN step OOMs on the row-chunk. Rarely needed. |
| `legacy` | `_smart_expand` | `False` | `True` only for bit-exact 2.x reproduction. |
| `svd_solver` | `samap.sam.pca._pca_with_sparse` | `"arpack"` | `"randomized"` is faster on GPU and at high `npcs`. Slightly different numerics (randomized is an approximation). Not plumbed to public API. |
| `backend` | `SAMAP.__init__` | `"auto"` | Force `"cpu"` for reproducibility; `"cuda"` to fail loudly if GPU is missing. |

---

## Expected scaling

These are **estimates** from synthetic benchmarks and informal testing.
Actual limits depend heavily on species count, gene-set overlap, data
density, and kNN parameters.

| Setup | Approx. cell-count ceiling | Notes |
|---|---|---|
| 64 GB CPU, ≤2.x code | ~500k | `Xtr` and `Xavg` materialization are the walls |
| 64 GB CPU, 3.0 | ~2-3M | Limited by `X_iᵀX_i` Gram matrix and streaming overhead |
| 256 GB CPU + A100, 3.0 | ~5-10M | Randomized SVD helps; kNN moves to FAISS on GPU |

Dominant memory costs in 3.0, in rough order:

1. The input `X` matrices themselves (CSR, unavoidable)
2. `X_iᵀX_i` Gram matrices (G × G sparse, per species — precomputed once)
3. Per-iteration `P_ij` (G_i × npcs dense, but tiny)
4. Streaming correlation working set (N × 2·batch_size dense)

If you OOM at step 2, your gene set is too large — consider pre-filtering
to highly variable genes before running SAMap.

---

## Benchmark results

From `benchmarks/bench_samap.py`, synthetic data, 2-species, timed over
3 SAMap iterations (1 for correlation), measured on CPU (tracemalloc peak):

| Phase | n_cells | Legacy wall (s) | Optimized wall (s) | Speedup | Legacy mem (MB) | Optimized mem (MB) | Mem ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| expand | 1 000 | 1.18 | 0.34 | **3.5×** | 11.2 | 8.4 | 1.3× |
| expand | 3 000 | 5.61 | 1.16 | **4.8×** | 62.8 | 40.3 | 1.6× |
| projection | 1 000 | 6.61 | 3.32 | **2.0×** | 1157 | 571 | 2.0× |
| projection | 3 000 | 11.79 | 6.33 | **1.9×** | 1180 | 573 | 2.1× |
| correlation | 1 000 | 0.25 | 1.18 | 0.21× | 103 | 24 | **4.3×** |
| correlation | 3 000 | 0.37 | 1.33 | 0.27× | 307 | 71 | **4.3×** |

**Interpretation:**

- **expand**: Pure win. BFS is faster and smaller at every scale tested.
- **projection**: Pure win. ~2× on both axes; gains grow with N as the
  `Xtr` materialization would grow linearly in N but the precomposed
  `P_ij` does not.
- **correlation**: Memory win at the cost of speed — by design. The
  speedup axis will flip positive at the scale where materialized `Xavg`
  spills to swap or OOMs outright. On the toy benchmark sizes, streaming
  is slower.

Re-run the benchmark locally:

```bash
python benchmarks/bench_samap.py --max-cells 10000
python benchmarks/plot_bench.py benchmarks/results/bench_<TIMESTAMP>.csv
```

The `tracemalloc` peak catches numpy/scipy sparse allocations (the bulk
of legacy materialization) but misses allocations inside numba-nopython
kernels — so actual peak RSS may differ.
