# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`samap.analysis` interpretation helpers** (driven by a 105-pair
  pan-metazoan benchmark sweep):
  - `get_mapping_scores(..., which_iter=0|'final')` — score against the
    iter-0 (raw-BLAST-homology) manifold as well as the converged one.
    `SAMAP.run` now stores `obsp['connectivities_iter0']` and
    `sm.nnm_per_iter[]` so the "find" vs "measure" axes are separable.
  - `permutation_null_scores` — label-permutation null for mapping
    scores (no manifold rerun; cheap empirical p-values per pair).
  - `homology_graph_delta` / `find_paralog_substitutions` — per-edge
    sequence-similarity vs expression-correlation residuals; ranked
    paralog-substitution candidates without an external orthology DB.
  - `gene_modules` / `module_factored_scores` — Leiden-partition the
    homology graph and decompose each cell-type alignment by how many
    independent gene modules support it (`n_modules`, `top_module_frac`,
    `module_entropy`).
  - `cluster_to_k` / `mapping_degeneracy` — leiden-to-target-k for
    granularity-matched comparison; reciprocal-best fraction, entropy,
    and effective-rank summaries of the score matrix.
  - `persist_pair` / `score_from_connectivities` / `build_union_graph` /
    `cluster_families` / `family_phylogenetic_signal`
    (`samap.analysis.ontology`) — disk-based re-scoring of persisted
    `obsp['connectivities']` against arbitrary per-species label sets
    without instantiating SAM/SAMAP objects (~0.2s/pair); union-graph
    assembly and Leiden community detection for many-species cell-type-
    family discovery; per-family lineage-vs-program classification via
    within-family score–divergence correlation with same-study-clade
    exclusion control.
  - `SAMAP.run(..., joint_weights=True)` — *experimental*: recompute SAM
    gene weights on the joint manifold after iteration 1 so iterations
    2..N project through cross-species-informative genes (down-weights
    pan-conserved RNA-processing genes that dominate enriched-pair
    lists). Off by default; no behaviour change.

### Fixed

- Tests: hoisted `NUMBA_NUM_THREADS` pinning from
  `tests/regression/test_golden_output.py` to the root `tests/conftest.py`
  so it applies before any test module imports numba (the previous
  location was collected after `tests/integration/`, causing a
  `RuntimeError: Cannot set NUMBA_NUM_THREADS to a different value once
  the threads have been launched` whenever the full suite was run on a
  >1-CPU host without the env var pre-set).

### Added

- **`samap.io` onboarding helpers** (see `docs/io.md`):
  - `detect_id_flavor` — regex classifier for `var_names` namespace
    (Ensembl/RefSeq/NCBI GeneID/UniProt/model-org DBs/symbol/unknown).
  - `fetch_proteome` — derive a protein FASTA whose headers *are*
    `var_names` from Ensembl REST or NCBI Datasets v2. No new runtime
    deps (stdlib `urllib`).
  - `match_fasta` — score a header-transform cascade against `var_names`,
    pick the best, emit a renamed FASTA + `names[]` array. Optional GTF
    transcript→gene mapping.
  - `gnnm_from_pairs` / `homology_from_eggnog` — build the
    `(gnnm, gns, gns_dict)` tuple for `SAMAP(gnnm=...)` from any
    ortholog table or eggNOG-mapper OG co-membership.
  - `run_blast` — Python port of `map_genes.sh`: all N-choose-2
    reciprocal alignments, DIAMOND-first with BLAST+ fallback.
  - `save_gnnm` / `load_gnnm` — `.npz` cache for the homology graph.
- **`samap` CLI** (`detect-ids`, `fetch-proteome`, `match-fasta`,
  `blast`) via `[project.scripts]`.
- `SAMAP.__init__` now logs the per-species `var_names ↔ homology graph`
  overlap fraction and **warns with example IDs from each side** when
  overlap < 30 % (`HOMOLOGY_OVERLAP_WARN_THRESHOLD`).
- `network` pytest marker for tests that hit external services.

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
