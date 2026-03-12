"""Gene-gene correlation refinement for the homology graph.

Contains the numba-accelerated kernels for computing Pearson / Xi correlations
between homologous gene pairs across the stitched manifold, and the driver
routines that chunk the graph for parallel refinement.

Implementation notes
--------------------
The memory bottleneck here is ``Xavg = nnms @ Xs`` — an N × G_active matrix at
10-60% density (multiple GB at million-cell scale). It exists solely to feed
per-gene-pair Pearson correlations. We offer two paths:

* **Materialized** (``batch_size=None``): builds the full ``Xavg`` up front.
  Faster for moderate-scale runs (~3-5× at <10k cells); the right choice when
  the estimated ``Xavg`` fits comfortably in memory.
* **Streaming** (``batch_size=int``): processes pairs in batches. For each
  batch, computes ``Xavg`` only for the genes appearing in that batch's pairs
  (at most ``2 * batch_size`` columns), correlates, discards. Peak memory
  drops from O(N × G_active) to O(N × 2·batch_size). Some columns are
  recomputed across batches if a gene appears in multiple pair-batches;
  this is a cheap single-column SpMV and empirically <5% overhead.
* **Auto** (``batch_size="auto"``, the default): estimates the materialised
  ``Xavg`` size from cell/gene counts, expression density, and kNN degree.
  If the estimate is under ``correlation_mem_threshold`` (default 2 GB),
  materialise; otherwise stream at ``batch_size=512``. See
  :func:`_resolve_batch_size`.

Separately, the numba kernel no longer uses a Python dict for species lookup.
Species cell ranges are passed as integer ``sp_starts`` / ``sp_lens`` arrays,
indexed by integer species ID. This is a prerequisite for a future CUDA port.
"""

from __future__ import annotations

import gc
import os
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy as sp
from numba import njit, prange
from numba.core.errors import NumbaPerformanceWarning, NumbaWarning

from samap._logging import logger
from samap.utils import q as _q
from samap.utils import to_vn

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray

    from samap.sam import SAM

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
warnings.filterwarnings("ignore", category=NumbaWarning)


@njit(parallel=True)
def _replace(X: NDArray[Any], xi: NDArray[Any], yi: NDArray[Any]) -> NDArray[np.float64]:
    """Per-pair Pearson over rows of a dense matrix (CPU fast path, numba).

    For each pair ``(xi[i], yi[i])``, gather the two rows of ``X`` and compute
    Pearson correlation. ``prange`` parallelises over pairs.

    On CPU this outperforms the vectorised form below because numba can fuse
    the mean/std/dot into a single tight loop per pair with no intermediate
    ``(n_pairs × d)`` allocation. On GPU, use :func:`_replace_vectorized`.
    """
    data = np.zeros(xi.size)
    for i in prange(xi.size):
        x = X[xi[i]]
        y = X[yi[i]]
        data[i] = ((x - x.mean()) * (y - y.mean()) / x.std() / y.std()).sum() / x.size
    return data


def _replace_vectorized(
    X: Any,
    xi: Any,
    yi: Any,
    bk: Any,
    batch_size: int | None = None,
) -> Any:
    """Per-pair Pearson over rows of a dense matrix — vectorised, backend-agnostic.

    Algebraically identical to :func:`_replace`. Works on both numpy and cupy
    via ``bk.xp`` dispatch. Each batch gathers ``2 × batch`` rows of ``X``
    and computes Pearson in one shot — one reduction kernel on GPU instead of
    ``n_pairs`` launches.

    Parameters
    ----------
    X
        Dense (N × d) array on the active backend.
    xi, yi
        Integer index arrays of shape (n_pairs,) on the active backend.
    bk
        :class:`Backend` instance for xp dispatch.
    batch_size
        If ``None``, process all pairs at once (requires ``2 * n_pairs * d``
        floats of scratch). If an integer, chunk pairs to cap the working
        set at ``2 * batch_size * d`` floats. Use when ``n_pairs × d × 8``
        bytes approaches device memory.

    Returns
    -------
    Array of shape (n_pairs,) on the active backend, dtype float64.
    """
    xp = bk.xp
    n_pairs = xi.shape[0]

    def _one_batch(ii: Any, jj: Any) -> Any:
        Xa = X[ii].astype(xp.float64, copy=True)
        Xb = X[jj].astype(xp.float64, copy=True)
        Xa -= Xa.mean(axis=1, keepdims=True)
        Xb -= Xb.mean(axis=1, keepdims=True)
        num = (Xa * Xb).sum(axis=1)
        den = xp.sqrt((Xa * Xa).sum(axis=1) * (Xb * Xb).sum(axis=1))
        # den==0 (zero-variance row) → 0/0 → nan. Match _replace's behaviour
        # (callers already do vals[isnan]=0). Suppress the expected warning.
        with np.errstate(invalid="ignore"):
            return num / den

    if batch_size is None or batch_size >= n_pairs:
        return _one_batch(xi, yi)

    out = xp.empty(n_pairs, dtype=xp.float64)
    for start in range(0, n_pairs, batch_size):
        end = min(start + batch_size, n_pairs)
        out[start:end] = _one_batch(xi[start:end], yi[start:end])
    return out


def replace_corr(
    X: Any,
    xi: Any,
    yi: Any,
    bk: Any = None,
    batch_size: int | None = None,
) -> Any:
    """Dispatch per-pair Pearson to numba (CPU) or vectorised (GPU).

    Drop-in entry point for callers that have a :class:`Backend`. When
    ``bk is None`` or ``not bk.gpu``, uses the numba :func:`_replace` (fastest
    on CPU: fused loop, no intermediate allocation). When ``bk.gpu`` is True,
    uses :func:`_replace_vectorized` (single large reduction kernel on
    device).

    Numerically equivalent to :func:`_replace` to machine precision.
    """
    if bk is None or not bk.gpu:
        # CPU fast path — numba prange beats numpy vectorisation here
        return _replace(np.asarray(X), np.asarray(xi), np.asarray(yi))
    return _replace_vectorized(X, xi, yi, bk, batch_size=batch_size)


@njit
def nb_unique1d(ar: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Find unique elements of an array (numba-optimized)."""
    ar = ar.flatten()
    perm = ar.argsort(kind="mergesort")
    aux = ar[perm]
    mask = np.empty(aux.shape, dtype=np.bool_)
    mask[:1] = True
    mask[1:] = aux[1:] != aux[:-1]

    imask = np.cumsum(mask) - 1
    inv_idx = np.empty(mask.shape, dtype=np.intp)
    inv_idx[perm] = imask
    idx = np.append(np.nonzero(mask)[0], mask.size)

    return aux[mask], perm[mask], inv_idx, np.diff(idx)


@njit
def _xicorr(X: NDArray[Any], Y: NDArray[Any]) -> float:
    """Xi correlation coefficient."""
    n = X.size
    xi = np.argsort(X, kind="quicksort")
    Y = Y[xi]
    _, _, b, c = nb_unique1d(Y)
    r = np.cumsum(c)[b]
    _, _, b, c = nb_unique1d(-Y)
    left_counts = np.cumsum(c)[b]
    denominator = 2 * (left_counts * (n - left_counts)).sum()
    if denominator > 0:
        return 1 - n * np.abs(np.diff(r)).sum() / denominator
    else:
        return 0.0


# ---------------------------------------------------------------------------
# CUDA-porting notes for _corr_kernel
# ---------------------------------------------------------------------------
# The kernel below is structurally dict-free and uses only integer species
# indexing, but is NOT yet directly compilable under @cuda.jit. Remaining
# blockers and the restructuring they imply:
#
# 1. Dynamic allocation. `np.zeros(n)`, `np.empty(total)` allocate per-call
#    with runtime sizes. CUDA kernels cannot heap-allocate.
#    → Fix: do not densify. Instead of scattering CSC column → dense N-vector
#      → slicing by species, iterate directly over the CSC nonzeros,
#      test each index against the two species ranges [s1,s1+l1)∪[s2,s2+l2),
#      and accumulate Pearson sums (sum_x, sum_y, sum_xx, sum_xy, sum_yy,
#      count) in scalar registers. This is a two-pass loop per pair (mean
#      first, then centred products) but needs only O(1) per-thread state.
#
# 2. Fancy-index scatter `x[pl1i] = pl1d`. Not supported on cuda device
#    arrays. Becomes irrelevant once (1) removes the dense scatter.
#
# 3. `prange` → `cuda.grid(1)` + `if j >= n_pairs: return`. Trivial.
#
# 4. `.mean()`, `.std()` on arrays. Not available on cuda.local arrays.
#    Irrelevant after (1) — sums are accumulated manually.
#
# 5. `_xicorr` device call. Xi correlation uses `np.argsort` and rank
#    computations that have no @cuda.jit equivalent. A GPU Xi path would
#    need cub/thrust sort (via cupy) + a separate device kernel for the
#    rank statistics. For a first CUDA port, gate the kernel to
#    `pearson=True` only and fall back to CPU for Xi.
#
# With (1), the kernel becomes a CSR/CSC-walking Pearson that reads CSC
# columns twice (two passes) and writes one float per pair. Workspace:
# ~6 float64 + ~6 int64 registers per thread. No shared memory needed.
# ---------------------------------------------------------------------------


@njit(parallel=True)
def _corr_kernel(
    p1: NDArray[np.int64],
    p2: NDArray[np.int64],
    ps1: NDArray[np.int64],
    ps2: NDArray[np.int64],
    sp_starts: NDArray[np.int64],
    sp_lens: NDArray[np.int64],
    indptr: NDArray[Any],
    indices: NDArray[Any],
    data: NDArray[Any],
    n: int,
    pearson: bool,
) -> NDArray[np.float64]:
    """Compute per-pair gene correlations (dict-free, GPU-portable).

    This replaces the original dict-based ``_refine_corr_kernel``. Species
    cell ranges are passed as integer ``sp_starts`` / ``sp_lens`` arrays
    (contiguous by construction in the stitched manifold), indexed by integer
    species ID — no Python dict inside the hot loop. Numerically bit-identical
    to the original (same Pearson formula, same Xi path).

    Parameters
    ----------
    p1, p2
        Column indices into the CSC ``Xavg``, one pair per entry.
    ps1, ps2
        Integer species IDs for each gene (index into ``sp_starts``).
    sp_starts, sp_lens
        Species ``s`` spans cells ``[sp_starts[s], sp_starts[s]+sp_lens[s])``.
    indptr, indices, data
        CSC components of ``Xavg``.
    n
        Number of cells (rows in ``Xavg``).
    pearson
        True → Pearson; False → Xi correlation.
    """
    res = np.zeros(p1.size)

    for j in prange(len(p1)):
        j1, j2 = p1[j], p2[j]
        pl1d = data[indptr[j1] : indptr[j1 + 1]]
        pl1i = indices[indptr[j1] : indptr[j1 + 1]]

        sc1d = data[indptr[j2] : indptr[j2 + 1]]
        sc1i = indices[indptr[j2] : indptr[j2 + 1]]

        x = np.zeros(n)
        x[pl1i] = pl1d
        y = np.zeros(n)
        y[sc1i] = sc1d

        s1 = sp_starts[ps1[j]]
        l1 = sp_lens[ps1[j]]
        s2 = sp_starts[ps2[j]]
        l2 = sp_lens[ps2[j]]

        total = l1 + l2
        xx = np.empty(total)
        yy = np.empty(total)
        xx[:l1] = x[s1 : s1 + l1]
        xx[l1:] = x[s2 : s2 + l2]
        yy[:l1] = y[s1 : s1 + l1]
        yy[l1:] = y[s2 : s2 + l2]

        if pearson:
            c = ((xx - xx.mean()) * (yy - yy.mean()) / xx.std() / yy.std()).sum() / xx.size
        else:
            c = _xicorr(xx, yy)
        res[j] = c
    return res


def _compute_pair_corrs(
    nnms: Any,
    Xs: Any,
    p: NDArray[np.int64],
    ps_int: NDArray[np.int64],
    sp_starts: NDArray[np.int64],
    sp_lens: NDArray[np.int64],
    n: int,
    corr_mode: str,
    batch_size: int | None,
) -> NDArray[np.float64]:
    """Compute correlations for all gene pairs, materialised or streaming.

    Parameters
    ----------
    nnms
        (N × N) row-normalised neighbour-averaging operator (CSR).
    Xs
        (N × G_active) block-diagonal expression matrix. CSC preferred for
        column slicing in the streaming path.
    p
        (n_pairs × 2) integer column-indices into ``Xs`` for each gene pair.
    ps_int
        (n_pairs × 2) integer species IDs for each gene pair.
    sp_starts, sp_lens, n
        Species layout (see :func:`_corr_kernel`).
    corr_mode
        ``"pearson"`` or ``"xi"``.
    batch_size
        ``None`` → materialise full ``Xavg = nnms @ Xs`` (legacy path,
        golden-compatible). ``int`` → stream in pair-batches; per batch,
        compute only the ≤ ``2 * batch_size`` columns of ``Xavg`` actually
        needed, correlate, discard. Peak memory O(N × 2·batch_size) instead
        of O(N × G_active).
    """
    pearson = corr_mode == "pearson"
    p1 = np.ascontiguousarray(p[:, 0], dtype=np.int64)
    p2 = np.ascontiguousarray(p[:, 1], dtype=np.int64)
    ps1 = np.ascontiguousarray(ps_int[:, 0], dtype=np.int64)
    ps2 = np.ascontiguousarray(ps_int[:, 1], dtype=np.int64)

    if batch_size is None:
        # --- Materialised path (golden-compatible) --------------------------
        Xavg = nnms.dot(Xs).tocsc()
        return _corr_kernel(
            p1, p2, ps1, ps2,
            sp_starts, sp_lens,
            Xavg.indptr, Xavg.indices, Xavg.data,
            n, pearson,
        )

    # --- Streaming path -----------------------------------------------------
    if not sp.sparse.isspmatrix_csc(Xs):
        Xs = Xs.tocsc()

    n_pairs = p1.size
    res = np.zeros(n_pairs)

    for start in range(0, n_pairs, batch_size):
        end = min(start + batch_size, n_pairs)

        p1b = p1[start:end]
        p2b = p2[start:end]

        # Genes needed for this batch (sorted, unique) — at most 2·batch_size.
        needed = np.unique(np.concatenate((p1b, p2b)))
        # Map original column index → local column index in Xavg_batch.
        # `needed` is sorted so searchsorted gives the exact local position.
        p1_local = np.searchsorted(needed, p1b).astype(np.int64)
        p2_local = np.searchsorted(needed, p2b).astype(np.int64)

        # One SpMM for just the needed columns, then CSC for kernel access.
        Xavg_batch = nnms.dot(Xs[:, needed]).tocsc()

        res[start:end] = _corr_kernel(
            p1_local, p2_local,
            np.ascontiguousarray(ps1[start:end]),
            np.ascontiguousarray(ps2[start:end]),
            sp_starts, sp_lens,
            Xavg_batch.indptr, Xavg_batch.indices, Xavg_batch.data,
            n, pearson,
        )

        del Xavg_batch
        gc.collect()

    return res


def _resolve_batch_size(
    batch_size: int | str | None,
    nnms: Any,
    Xs: Any,
    mem_threshold_gb: float = 2.0,
) -> int | None:
    """Auto-select batched vs materialised correlation based on estimated memory.

    The materialised path (``batch_size=None``) is 3-5× faster on small data
    (fewer SpMM dispatches, better cache reuse) but requires holding the full
    ``Xavg = nnms @ Xs`` in memory. The streaming path (``batch_size=int``)
    caps memory but pays per-batch overhead.

    Heuristic: estimate the materialised ``Xavg`` size. The output density
    after kNN smoothing is roughly ``1 - (1 - p)^k`` where ``p`` is the input
    expression density and ``k`` is the average neighbour degree — each output
    entry is zero only if all ``k`` contributing inputs are zero. If the
    estimate is comfortably under ``mem_threshold_gb``, materialise; otherwise
    stream at 512.

    Parameters
    ----------
    batch_size
        User-supplied batch_size. If not the string ``"auto"``, returned
        unchanged (respects explicit user choice including ``None``).
    nnms
        Row-normalised averaging operator (N × N sparse).
    Xs
        Block-diagonal expression (N × G_active sparse).
    mem_threshold_gb
        If estimated ``Xavg`` size is below this, materialise. Default 2 GB
        leaves ample headroom on a 16 GB laptop; large-memory nodes can
        raise it via ``correlation_mem_threshold``.

    Returns
    -------
    ``None`` to materialise, or an integer batch size to stream.
    """
    if batch_size != "auto":
        return batch_size

    n_cells, n_genes = Xs.shape
    if n_cells == 0 or n_genes == 0:
        return None  # trivial — materialise

    k_avg = nnms.nnz / max(nnms.shape[0], 1)
    expr_density = Xs.nnz / (n_cells * n_genes)
    # Output entry is zero iff all k contributing inputs are zero: (1-p)^k.
    # This overestimates density slightly (ignores structural zeros from
    # block-diag Xs outside a species' gene range), which is the safe direction.
    out_density = min(1.0, 1.0 - (1.0 - expr_density) ** max(k_avg, 1.0))

    # CSC storage: data (float64) + indices (int32) + indptr. Dominated by
    # data + indices ≈ 12 bytes/nonzero. Use 12 not 8 to be conservative.
    est_bytes = n_cells * n_genes * out_density * 12.0
    est_gb = est_bytes / 1e9

    if est_gb < mem_threshold_gb:
        logger.info(
            "Correlation: estimated Xavg %.3f GB (density~%.1f%%, %d cells x %d genes) "
            "< %.1f GB threshold — using materialised path.",
            est_gb, out_density * 100, n_cells, n_genes, mem_threshold_gb,
        )
        return None

    logger.info(
        "Correlation: estimated Xavg %.3f GB (density~%.1f%%, %d cells x %d genes) "
        ">= %.1f GB threshold — using streaming path (batch_size=512).",
        est_gb, out_density * 100, n_cells, n_genes, mem_threshold_gb,
    )
    return 512


def _refine_corr(
    sams: dict[str, SAM],
    st: SAM,
    gnnm: sp.sparse.csr_matrix,
    gns_dict: dict[str, NDArray[Any]],
    corr_mode: str = "pearson",
    THR: float = 0,
    use_seq: bool = False,
    T1: float = 0.25,
    NCLUSTERS: int = 1,
    ncpus: int | None = None,
    wscale: bool = False,
    batch_size: int | str | None = "auto",
    correlation_mem_threshold: float = 2.0,
) -> sp.sparse.csr_matrix:
    """Refine correlation matrix for homology graph.

    Parameters
    ----------
    batch_size : int | str | None
        ``"auto"`` (default) → :func:`_resolve_batch_size` decides: materialise
        when the estimated ``Xavg`` fits under ``correlation_mem_threshold``
        GB (3-5× faster on small data), stream at 512 otherwise. ``None``
        forces the materialised path unconditionally. An integer forces
        streaming at that batch size. See the module docstring for the full
        memory model.
    correlation_mem_threshold : float
        Memory threshold (GB) for ``batch_size="auto"``. Default 2.0 — leaves
        ample headroom on a 16 GB laptop. Raise on large-memory nodes to keep
        the faster materialised path for larger datasets; lower for
        memory-constrained environments.
    (other parameters unchanged)
    """
    if ncpus is None:
        ncpus = os.cpu_count() or 1

    gns = np.concatenate(list(gns_dict.values()))

    x, y = gnnm.nonzero()
    sam = next(iter(sams.values()))
    cl = sam.leiden_clustering(gnnm, res=0.5)
    ix = np.argsort(cl)
    NGPC = gns.size // NCLUSTERS + 1

    ixs = []
    for i in range(NCLUSTERS):
        ixs.append(np.sort(ix[i * NGPC : (i + 1) * NGPC]))

    assert np.concatenate(ixs).size == gns.size

    GNNMSUBS = []
    GNSUBS = []
    for i in range(len(ixs)):
        ixs[i] = np.unique(np.append(ixs[i], gnnm[ixs[i], :].nonzero()[1]))
        gnnm_sub = gnnm[ixs[i], :][:, ixs[i]]
        gnsub = gns[ixs[i]]
        gns_dict_sub = {}
        for sid in gns_dict:
            gn = gns_dict[sid]
            gns_dict_sub[sid] = gn[np.isin(gn, gnsub)]

        gnnm2_sub = _refine_corr_parallel(
            sams,
            st,
            gnnm_sub,
            gns_dict_sub,
            corr_mode=corr_mode,
            THR=THR,
            use_seq=use_seq,
            T1=T1,
            ncpus=ncpus,
            wscale=wscale,
            batch_size=batch_size,
            correlation_mem_threshold=correlation_mem_threshold,
        )
        GNNMSUBS.append(gnnm2_sub)
        GNSUBS.append(gnsub)
        gc.collect()

    indices_list = []
    pairs_list = []
    for i in range(len(GNNMSUBS)):
        indices_list.append(np.unique(np.sort(np.vstack(GNNMSUBS[i].nonzero()).T, axis=1), axis=0))
        pairs_list.append(GNSUBS[i][indices_list[-1]])

    GNS = pd.DataFrame(data=np.arange(gns.size)[None, :], columns=gns)
    gnnm3 = sp.sparse.lil_matrix(gnnm.shape)
    for i in range(len(indices_list)):
        x, y = GNS[pairs_list[i][:, 0]].values.flatten(), GNS[pairs_list[i][:, 1]].values.flatten()
        gnnm3[x, y] = np.asarray(
            GNNMSUBS[i][indices_list[i][:, 0], indices_list[i][:, 1]]
        ).flatten()

    gnnm3 = gnnm3.tocsr()
    x, y = gnnm3.nonzero()
    gnnm3 = gnnm3.tolil()
    gnnm3[y, x] = np.asarray(gnnm3[x, y].tocsr().todense()).flatten()
    return gnnm3.tocsr()


def _refine_corr_parallel(
    sams: dict[str, SAM],
    st: SAM,
    gnnm: sp.sparse.csr_matrix,
    gns_dict: dict[str, NDArray[Any]],
    corr_mode: str = "pearson",
    THR: float = 0,
    use_seq: bool = False,
    T1: float = 0.0,
    ncpus: int | None = None,
    wscale: bool = False,
    batch_size: int | str | None = "auto",
    correlation_mem_threshold: float = 2.0,
) -> sp.sparse.csr_matrix:
    """Parallel correlation refinement.

    Parameters
    ----------
    batch_size
        ``"auto"`` (default) → pick materialised vs streaming based on the
        estimated ``Xavg`` memory footprint (see :func:`_resolve_batch_size`).
        ``None`` → force materialised (legacy, fast on small data). Integer →
        force streaming at that batch size.
    correlation_mem_threshold
        GB threshold for auto-selection. Default 2.0.
    (other parameters unchanged)
    """
    if ncpus is None:
        ncpus = os.cpu_count() or 1

    gn = np.concatenate(list(gns_dict.values()))

    Ws = []
    ix = []
    for sid in sams:
        Ws.append(sams[sid].adata.var["weights"][gns_dict[sid]].values)
        ix += [sid] * gns_dict[sid].size
    ix = np.array(ix)
    w = np.concatenate(Ws)

    w[w > T1] = 1
    w[w < 1] = 0

    gnO = gn[w > 0]
    ix = ix[w > 0]
    gns_dictO = {}
    for sid in gns_dict:
        gns_dictO[sid] = gnO[ix == sid]

    gnnmO = gnnm[w > 0, :][:, w > 0]
    x, y = gnnmO.nonzero()

    pairs = np.unique(np.sort(np.vstack((x, y)).T, axis=1), axis=0)

    xs = _q([i.split("_")[0] for i in gnO[pairs[:, 0]]])
    ys = _q([i.split("_")[0] for i in gnO[pairs[:, 1]]])
    pairs_species = np.vstack((xs, ys)).T

    nnm = st.adata.obsp["connectivities"]
    xs_list = []
    nnms = []
    for i, sid in enumerate(sams.keys()):
        batch_mask = (st.adata.obs["batch"] == f"batch{i + 1}").values
        nnms.append(nnm[:, batch_mask])
        s1 = np.asarray(nnms[-1].sum(1))
        s1[s1 < 1e-3] = 1
        s1 = s1.flatten()[:, None]
        nnms[-1] = nnms[-1].multiply(1 / s1)

        xs_list.append(sams[sid].adata[:, gns_dictO[sid]].X.astype("float32"))

    Xs = sp.sparse.block_diag(xs_list).tocsc()
    nnms = sp.sparse.hstack(nnms).tocsr()

    # Resolve "auto" to a concrete batch_size now that nnms/Xs shapes are known.
    batch_size = _resolve_batch_size(batch_size, nnms, Xs, correlation_mem_threshold)

    p = pairs
    ps = pairs_species

    gnnm2 = gnnm.multiply(w[:, None]).multiply(w[None, :]).tocsr()
    x, y = gnnm2.nonzero()
    pairs = np.unique(np.sort(np.vstack((x, y)).T, axis=1), axis=0)

    # --- Species layout as integer start/len arrays (dict-free kernel) ------
    species = _q(st.adata.obs["species"])
    sidss = np.unique(species)
    sp_starts = np.empty(sidss.size, dtype=np.int64)
    sp_lens = np.empty(sidss.size, dtype=np.int64)
    for i, sid in enumerate(sidss):
        where = np.where(species == sid)[0]
        sp_starts[i] = where[0]
        sp_lens[i] = where.size
        # Contiguity sanity check — true by construction in _concatenate_sam
        # (cells are grouped by species); guard against future changes.
        if where[-1] - where[0] + 1 != where.size:
            raise RuntimeError(
                f"Species {sid!r} cells are not contiguous in the stitched "
                f"manifold. This violates the dict-free kernel's assumption "
                f"(cells must be species-grouped). Please report."
            )

    # Map string species IDs in `ps` → integer positions in `sidss`.
    sid_to_int = pd.Series(np.arange(sidss.size, dtype=np.int64), index=sidss)
    ps_int = np.column_stack(
        (sid_to_int.loc[ps[:, 0]].values, sid_to_int.loc[ps[:, 1]].values)
    ).astype(np.int64)

    vals = _compute_pair_corrs(
        nnms, Xs, p.astype(np.int64), ps_int,
        sp_starts, sp_lens, nnms.shape[0], corr_mode, batch_size,
    )
    vals[np.isnan(vals)] = 0

    CORR = dict(zip(to_vn(np.vstack((gnO[p[:, 0]], gnO[p[:, 1]])).T), vals))

    for k in CORR:
        CORR[k] = 0 if CORR[k] < THR else CORR[k]
        if wscale:
            id1, id2 = [x.split("_")[0] for x in k.split(";")]
            weight1 = sams[id1].adata.var["weights"][k.split(";")[0]]
            weight2 = sams[id2].adata.var["weights"][k.split(";")[1]]
            CORR[k] = np.sqrt(CORR[k] * np.sqrt(weight1 * weight2))

    CORR_arr = np.array([CORR[x] for x in to_vn(gn[pairs])])

    gnnm3 = sp.sparse.lil_matrix(gnnm.shape)

    if use_seq:
        gnnm3[pairs[:, 0], pairs[:, 1]] = (
            CORR_arr * np.asarray(gnnm2[pairs[:, 0], pairs[:, 1]]).flatten()
        )
        gnnm3[pairs[:, 1], pairs[:, 0]] = (
            CORR_arr * np.asarray(gnnm2[pairs[:, 1], pairs[:, 0]]).flatten()
        )
    else:
        gnnm3[pairs[:, 0], pairs[:, 1]] = CORR_arr
        gnnm3[pairs[:, 1], pairs[:, 0]] = CORR_arr

    gnnm3 = gnnm3.tocsr()
    gnnm3.eliminate_zeros()
    return gnnm3
