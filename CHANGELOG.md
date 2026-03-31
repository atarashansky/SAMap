# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.1]

### Fixed

- `get_mapping_scores` no longer raises `IndexError` when the connectivity
  matrix contains explicit stored zeros — `_compute_csim` now calls
  `eliminate_zeros()` before `nonzero()`. (#172)
- `_calculate_blast_graph` is now compatible with pandas ≥3.0, which returns
  `StringArray` (no `.flatten()` method) from `.values` on string columns.
  Replaced `.values.flatten()` with `np.asarray(...)`. (#173)
- `GenePairFinder.find_genes` now honours the `w1t`/`w2t` SAM-weight threshold
  parameters instead of using a hardcoded `0.2`. (#166)
- `prepare_SAMap_loadings` now falls back to `adata.uns["sam"]["run_args"]`
  when `run_args` is not at the top level, supporting AnnData produced by
  `scanpy.external.tl.sam`. (#156)
- `sankey_plot` now renders all adjacent species pairs instead of being
  hardcoded for exactly 3 species. (#130)
- `_find_link_genes_avg` coerces sparse matrix indices to `int64` before
  fancy-indexing, avoiding scipy's int32 overflow (`ValueError: negative
  dimensions are not allowed`) on very large datasets. (#118)

### Changed

- README updated for v3.0.0: Python ≥3.11 requirement, simplified conda/pip
  install instructions, fixed "Anacodna" typo, updated import paths.
  (#171, #137, #136, #132)

## [3.0.0] - UNRELEASED

### Breaking

- **`sc-sam` removed as a dependency.** The SAM algorithm is now vendored
  under `samap.sam`. All internal imports route through `samap.sam` — no
  external SAM package is installed or required. If you were importing
  `samalg` directly, switch to `samap.sam`.
- `_smart_expand` default switched from matrix-power to BFS. Produces
  slightly different marginal neighbours (~1% edge difference on the
  golden-suite data) — the matpow path wasted one budget slot per cell on
  a self-loop artefact. Pass `legacy=True` for bit-exact 2.x reproduction.

### Added

- **GPU backend** via `SAMAP(backend="auto"|"cpu"|"cuda")`. Dispatches
  numpy/scipy ↔ cupy/cupyx, hnswlib ↔ FAISS for kNN, and scanpy ↔
  rapids-singlecell for Leiden/UMAP. Install with `pip install sc-samap[gpu]`
  (see `docs/performance.md` for conda details). `"auto"` picks CUDA if
  available, else CPU.
- **N² → N-linear memory rewrites** (see `docs/performance.md` for the full
  model):
  - *Precomposed feature translation* — projection precomposes
    `G · diag(W/σ) · PCs` so the cells × genes `Xtr` intermediate is never
    materialised. Iteration-invariant state (`XᵀX`, means, own-species
    projection) is computed once. ~2× wall and ~2× memory on the benchmark
    suite; gains grow with N.
  - *Streaming mutual-NN* — coarsening streams per-species-pair blocks
    directly into a CSR builder instead of materialising dense N × N products.
  - *Batched correlation refinement* — streams gene-pair batches
    (default `batch_size=512`); computes only the columns of the smoothed
    expression matrix referenced per batch. Peak memory drops from
    O(N × G_active) to O(N × 1024). ~4× less memory; ~3-5× slower on small
    data where the full matrix fits — pass `batch_size=None` to opt out.
  - *BFS neighbourhood expansion* — numba BFS kernel replaces matrix-power
    `_smart_expand`. ~5× faster at 3k cells, memory-bounded.
- **Randomized SVD with implicit centering** for sparse PCA — available
  via `svd_solver="randomized"` on `samap.sam.pca._pca_with_sparse`. Faster
  on GPU and at high PC counts; slightly different numerics. Default remains
  ARPACK.
- **Phase-level benchmark suite** — `benchmarks/bench_samap.py` compares
  legacy vs optimized paths for each rewritten phase.
- `docs/performance.md` — memory model, backend selection, tuning, scaling
  estimates.

### Fixed

- Dead random-walk computation in `_mapper` (result written then immediately
  discarded; preserved only the binarization side effect).
- `thr` → `align_thr` kwarg misroute in `analysis.enrichment` (was falling
  through to an unrelated p-value threshold).
- Deprecated `.A` matrix attribute → `np.asarray()` in several hot paths.
- Stale root `setup.py` removed (pyproject.toml is authoritative).
- Broken `SAMGUI` import and dead `gui()` method removed.
- Duplicated `_q` helper consolidated into `samap.utils.q`.
- Dead `mdata['xsim']` store removed.
- `__version__` is now dynamic via `importlib.metadata`.

### Changed

- `src/samap/core/mapping.py` split into focused modules: `homology.py`,
  `correlation.py`, `projection.py`, `coarsening.py`, `expand.py`. The
  `SAMAP` class remains in `mapping.py`; all existing imports work unchanged.
- `_refine_corr` / `_refine_corr_parallel` default `batch_size` changed
  from `None` (materialized) to `512` (streaming).
- `_smart_expand` default `legacy` changed from `True` (matpow) to
  `False` (BFS).
- Golden regression fixture regenerated to reflect the BFS and streaming
  defaults.
