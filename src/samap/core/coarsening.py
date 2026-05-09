"""Cross-species kNN graph stitching and manifold assembly.

The `_mapper` function here is the core graph-coarsening step: it takes
per-species neighbourhoods and the cross-species projection kNN, stitches them
together via in-degree coarsening, and produces the combined SAM manifold.

Implementation notes
--------------------
The mutual-NN construction exploits block structure to avoid materialising the
full N×N intermediate ``D = B @ nnm_internal.T``:

* ``B`` (cross-species kNN, from projection) is **block-off-diagonal** —
  within-species blocks are zero by construction.
* ``nnm_internal`` (expanded within-species kNN) is **block-diagonal**.
* Therefore ``D`` is also block-off-diagonal: ``D[a,b] = B[a,b] @ nnm_b.T``.
* The mutualisation ``M = sqrt(D ⊙ D.T)`` factors per species pair:
  ``M[a,b] = sqrt(D[a,b] ⊙ D[b,a].T)``, and the two factors can be computed
  chunk-by-chunk for the source species ``a`` without ever holding the full D.

This brings peak memory from O(N²) down to O(N_a × N_b) per pair (and further
down to O(chunk × N_b) when chunking within a large species).
"""

from __future__ import annotations

import gc
import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scanpy as sc
import scipy as sp

from samap._constants import (
    UMAP_MAXITER_LARGE,
    UMAP_MAXITER_SMALL,
    UMAP_MIN_DIST,
    UMAP_SIZE_THRESHOLD,
)
from samap._logging import logger
from samap.sam import SAM
from samap.utils import q as _q
from samap.utils import sparse_knn

from ._backend import Backend, COOBuilder
from .correlation import _replace, _replace_pair
from .expand import _smart_expand
from .homology import _tanh_scale
from .projection import _mapping_window, _mapping_window_fast

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray


def _generate_coclustering_matrix(cl: NDArray[Any]) -> sp.sparse.csr_matrix:
    """Generate a co-clustering indicator matrix."""
    from samap.sam.utils import convert_annotations

    cl_arr = convert_annotations(np.array(list(cl)))
    clu, _cluc = np.unique(cl_arr, return_counts=True)
    v = np.zeros((cl_arr.size, clu.size))
    v[np.arange(v.shape[0]), cl_arr] = 1
    return sp.sparse.csr_matrix(v)


def _scale_by_corr(
    M_chunk: Any,
    global_rows: NDArray[np.int64],
    wPCA: Any,
) -> Any:
    """Rescale mutual-NN edge weights by cell-cell correlation in wPCA space.

    Operates on a chunk of rows (all columns present for those rows), so
    per-row maxima are exact. Returns a CSR with the same sparsity pattern as
    the input and rescaled data — matches the original full-matrix path
    exactly.

    ``wPCA`` may be either a dense ``(N_total, S·npcs)`` array (legacy)
    or any object supporting fancy-indexing by row (``_TiledWPCA``). Rows for
    both endpoints are gathered up-front and the Pearson is computed on the
    gathered float32 blocks, so the inner numba kernel sees plain ndarrays
    either way.
    """
    M_chunk = M_chunk.tocsr()
    x, y = M_chunk.nonzero()
    # map chunk-local row indices → global cell indices for the correlation lookup
    if not isinstance(wPCA, np.ndarray):
        # _TiledWPCA or other lazy view — materialise once. This path is only
        # reached for pairwise=False or 2-species runs (S<=2), where the full
        # wPCA is N_total × S·npcs ≈ 2× the per-species blocks already held.
        # For 3+ species pairwise=True, _compute_mutual_graph routes through
        # _scale_by_corr_pair instead and never calls this.
        wPCA = np.asarray(wPCA)
    vals = _replace(wPCA, global_rows[x], y)
    # floor at 1e-3 (no eliminate_zeros — preserve M_chunk's sparsity pattern)
    vals[vals < 1e-3] = 1e-3

    F = M_chunk.copy()
    F.data[:] = vals

    Fmax = np.asarray(F.max(1).todense()).flatten()
    Fmax[Fmax == 0] = 1
    F = F.multiply(1 / Fmax[:, None]).tocsr()
    F.data[:] = _tanh_scale(F.data, center=0.7, scale=10)

    Mmax = np.asarray(M_chunk.max(1).todense()).flatten()
    Mmax[Mmax == 0] = 1

    scaled = F.multiply(M_chunk).tocsr()
    scaled.data[:] = np.sqrt(scaled.data)

    scaled_max = np.asarray(scaled.max(1).todense()).flatten()
    scaled_max[scaled_max == 0] = 1

    return scaled.multiply((Mmax / scaled_max)[:, None]).tocsr()


def _scale_by_corr_pair(
    M_block: Any,
    A: NDArray[Any],
    B: NDArray[Any],
    local_rows: NDArray[np.int64],
) -> Any:
    """Per-pair variant of :func:`_scale_by_corr` (pairwise tiling).

    ``M_block`` is the (chunk_len × N_b) mutual-NN block for a single
    species pair (a, b). ``A`` is species a's cells in the per-pair
    [PCs_a|PCs_b] space (N_a × 2·npcs); ``B`` is species b's cells in the
    same space (N_b × 2·npcs). ``local_rows`` maps chunk-local row index →
    species-a-local row index into ``A``.

    Row-max normalisations are taken over this pair's columns only — for
    ``pairwise=True`` (per-partner top-k) this is the semantically correct
    scope; the legacy code took row-max over *all* partners' columns, which
    let a strong a↔b correlation suppress a↔c weights. The change is
    confined to runs with 3+ species under ``pairwise=True``.
    """
    M_block = M_block.tocsr()
    x, y = M_block.nonzero()
    vals = _replace_pair(A, B, local_rows[x], y.astype(np.int64))
    vals[vals < 1e-3] = 1e-3

    F = M_block.copy()
    F.data[:] = vals
    Fmax = np.asarray(F.max(1).todense()).flatten()
    Fmax[Fmax == 0] = 1
    F = F.multiply(1 / Fmax[:, None]).tocsr()
    F.data[:] = _tanh_scale(F.data, center=0.7, scale=10)

    Mmax = np.asarray(M_block.max(1).todense()).flatten()
    Mmax[Mmax == 0] = 1
    scaled = F.multiply(M_block).tocsr()
    scaled.data[:] = np.sqrt(scaled.data)
    scaled_max = np.asarray(scaled.max(1).todense()).flatten()
    scaled_max[scaled_max == 0] = 1
    return scaled.multiply((Mmax / scaled_max)[:, None]).tocsr()


def _compute_mutual_graph(
    nnms_in: dict[str, Any],
    neigh_from_keys: dict[str, bool],
    B: Any,
    offsets: dict[str, int],
    n_cells: dict[str, int],
    sids: list[str],
    k1: int,
    N: int,
    *,
    pairwise: bool,
    chunksize: int,
    threshold: float,
    scale_edges_by_corr: bool,
    wPCA: NDArray[Any] | None,
    bk: Backend | None = None,
) -> Any:
    """Streaming per-species-pair mutual-NN construction.

    For each source species ``a``, iterates over row chunks and over partner
    species ``b ≠ a``, computing::

        left  = D[a,b][chunk] = B[a,b][chunk] @ nnm_b.T
        right = D[b,a].T[chunk] = nnm_a[chunk] @ B[b,a].T
        M[a,b][chunk] = sqrt(left ⊙ right)   # mutual geometric mean

    then assembles the chunk's full row (all partners), optionally rescales by
    wPCA correlation, top-k sparsifies, and accumulates into a COO builder.

    Parameters
    ----------
    nnms_in
        Per-species within-species neighbour matrices. For a species with
        ``neigh_from_keys[sid]`` false, this is an (N_i × N_i) expanded kNN.
        For ``neigh_from_keys[sid]`` true, this is an (N_i × n_clusters)
        one-hot cluster-membership matrix; the effective neighbour block is
        ``M @ M.T`` (cells sharing a cluster), kept factored to avoid
        materialising a potentially dense N_i² block.
    neigh_from_keys
        Per-species flag for the coclustering path (see above).
    B
        Cross-species kNN, (N × N), block-off-diagonal in global indices.
    offsets, n_cells, sids, N
        Species layout in global index space.
    k1
        Neighbours to keep per row (per species-pair if ``pairwise`` and
        more than two species; otherwise global per row).
    pairwise
        If True and ``len(sids) > 2``, top-k is applied per species-pair
        block rather than globally per row.
    chunksize
        Row-chunk size for the source species loop.
    threshold
        Elementwise floor applied to both ``left`` and ``right`` before
        mutualisation (entries below it are zeroed). Set to 0 to disable.
    scale_edges_by_corr, wPCA
        If True, rescale mutualised weights by tanh-scaled cell-cell
        correlation in ``wPCA`` space.

    Returns
    -------
    scipy.sparse.csr_matrix
        The mutualised, sparsified cross-species graph (N × N).
    """
    if bk is None:
        bk = Backend("cpu")
    builder = COOBuilder(bk, shape=(N, N))
    pairwise_topk = pairwise and len(sids) > 2

    # When pairwise_topk and wPCA exposes per-pair tiles, the
    # *correctness-preserving* path materialises one f32 N_total×S·npcs array
    # (releasing tiles in-place so peak ≈ 1×, half the legacy f64 buffer) and
    # runs the original full-width _scale_by_corr. The opt-in
    # SAMAP_TILED_SCALE_BY_CORR=1 path correlates each (a,b) block in its own
    # [PCs_a|PCs_b] subspace with per-partner row-max — semantically the right
    # scope for pairwise=True but a small (≈0.01 mean_top1, 0.97 hgr cosine)
    # deviation from legacy on a 3-species golden. Kept opt-in until validated
    # at S≫3.
    sid_to_idx = {sid: i for i, sid in enumerate(sids)}
    tiled_pair_corr = (
        pairwise_topk
        and scale_edges_by_corr
        and hasattr(wPCA, "pair_corr_basis")
        and os.environ.get("SAMAP_TILED_SCALE_BY_CORR") == "1"
    )
    if (
        not tiled_pair_corr
        and scale_edges_by_corr
        and hasattr(wPCA, "materialise_full")
    ):
        wPCA = wPCA.materialise_full(free=True)

    # Precompute per-species slices into B for cheap block extraction.
    gslice: dict[str, slice] = {
        sid: slice(offsets[sid], offsets[sid] + n_cells[sid]) for sid in sids
    }

    for a in sids:
        partners = [b for b in sids if b != a]
        if not partners:
            continue

        na = n_cells[a]
        off_a = offsets[a]
        nnm_a = nnms_in[a]
        nfk_a = neigh_from_keys[a]

        # Cache per-partner blocks of B once (row slicing is cheap on CSR).
        # B_ab[b]: (N_a × N_b), B_baT[b]: (N_a × N_b) = B[b,a].T
        B_ab: dict[str, Any] = {}
        B_baT: dict[str, Any] = {}
        for b in partners:
            B_ab[b] = B[gslice[a], gslice[b]].tocsr()
            B_baT[b] = B[gslice[b], gslice[a]].T.tocsr()

        # For nfk_a, precompute Ma.T @ B_ba.T per partner
        # (n_clusters_a × N_b, small). Reused across all chunks of species a.
        pre_right: dict[str, Any] = {}
        if nfk_a:
            for b in partners:
                pre_right[b] = nnm_a.T.dot(B_baT[b])

        # Per-pair correlation bases — assembled once per source species.
        pair_A: dict[str, NDArray[Any]] = {}
        pair_B: dict[str, NDArray[Any]] = {}
        if tiled_pair_corr:
            ia = sid_to_idx[a]
            for b in partners:
                ib = sid_to_idx[b]
                pair_A[b], pair_B[b] = wPCA.pair_corr_basis(ia, ib)

        for start in range(0, na, chunksize):
            end = min(start + chunksize, na)
            local = slice(start, end)
            chunk_len = end - start
            global_rows = np.arange(off_a + start, off_a + end, dtype=np.int64)
            local_rows = np.arange(start, end, dtype=np.int64)

            row_l: list[NDArray[np.intp]] = []
            col_l: list[NDArray[np.int64]] = []
            val_l: list[NDArray[np.float64]] = []

            for b in partners:
                nnm_b = nnms_in[b]
                nfk_b = neigh_from_keys[b]
                B_ab_chunk = B_ab[b][local]  # (chunk × N_b)

                # left = D_ab[chunk] = B_ab[chunk] @ nnm_block_b.T
                if nfk_b:
                    # nnm_block_b = M_b @ M_b.T  →  left = (B_ab_chunk @ M_b) @ M_b.T
                    left = B_ab_chunk.dot(nnm_b).dot(nnm_b.T)
                else:
                    left = B_ab_chunk.dot(nnm_b.T)

                # right = D_ba.T[chunk] = nnm_block_a[chunk] @ B_ba.T
                if nfk_a:
                    # nnm_block_a = M_a @ M_a.T  →  right = M_a[chunk] @ (M_a.T @ B_ba.T)
                    right = nnm_a[local].dot(pre_right[b])
                else:
                    right = nnm_a[local].dot(B_baT[b])

                if threshold > 0:
                    left = left.tocsr()
                    left.data[left.data < threshold] = 0
                    left.eliminate_zeros()
                    right = right.tocsr()
                    right.data[right.data < threshold] = 0
                    right.eliminate_zeros()

                Mb = left.multiply(right).tocsr()
                if Mb.nnz == 0:
                    continue
                Mb.data[:] = np.sqrt(Mb.data)

                if tiled_pair_corr:
                    # Tiled fast path: scale + top-k this (a,b) block in its
                    # own [PCs_a|PCs_b] subspace, then emit directly. Never
                    # touches columns outside species b → no full-width wPCA.
                    Mb = _scale_by_corr_pair(Mb, pair_A[b], pair_B[b], local_rows)
                    Mk = sparse_knn(Mb, k1).tocoo()
                    row_l.append(Mk.row)
                    col_l.append(Mk.col.astype(np.int64) + offsets[b])
                    val_l.append(Mk.data)
                    continue

                coo = Mb.tocoo()
                row_l.append(coo.row)
                col_l.append(coo.col.astype(np.int64) + offsets[b])
                val_l.append(coo.data)

            if not row_l:
                continue

            if tiled_pair_corr:
                # Already scaled + top-k'd per partner above — just emit.
                rows = np.concatenate(row_l)
                cols = np.concatenate(col_l)
                vals = np.concatenate(val_l)
                builder.add_batch(global_rows[rows], cols, vals)
                continue

            M_chunk = sp.sparse.csr_matrix(
                (np.concatenate(val_l), (np.concatenate(row_l), np.concatenate(col_l))),
                shape=(chunk_len, N),
            )

            if scale_edges_by_corr:
                M_chunk = _scale_by_corr(M_chunk, global_rows, wPCA)

            if pairwise_topk:
                out_rows: list[NDArray[np.intp]] = []
                out_cols: list[NDArray[np.int64]] = []
                out_vals: list[NDArray[np.float64]] = []
                for b in partners:
                    Msub = M_chunk[:, gslice[b]]
                    if Msub.nnz == 0:
                        continue
                    Mk = sparse_knn(Msub, k1).tocoo()
                    out_rows.append(Mk.row)
                    out_cols.append(Mk.col.astype(np.int64) + offsets[b])
                    out_vals.append(Mk.data)
                if not out_rows:
                    continue
                rows = np.concatenate(out_rows)
                cols = np.concatenate(out_cols)
                vals = np.concatenate(out_vals)
            else:
                Mk = sparse_knn(M_chunk, k1).tocoo()
                rows, cols, vals = Mk.row, Mk.col.astype(np.int64), Mk.data

            builder.add_batch(global_rows[rows], cols, vals)

    return builder.finalize("csr")


def _mapper(
    sams: dict[str, SAM],
    gnnm: sp.sparse.csr_matrix | None = None,
    gn: NDArray[Any] | None = None,
    NHS: dict[str, int] | None = None,
    umap: bool = False,
    mdata: dict[str, Any] | None = None,
    k: int | None = None,
    K: int = 20,
    chunksize: int = 20000,
    coarsen: bool = True,
    keys: dict[str, str] | None = None,
    scale_edges_by_corr: bool = False,
    neigh_from_keys: dict[str, bool] | None = None,
    pairwise: bool = True,
    proj_cache: dict[str, Any] | None = None,
    bk: Backend | None = None,
    **kwargs: Any,
) -> SAM:
    """Map cells between species."""
    if NHS is None:
        NHS = dict.fromkeys(sams.keys(), 3)

    if neigh_from_keys is None:
        neigh_from_keys = dict.fromkeys(sams.keys(), False)

    if mdata is None:
        if proj_cache is not None:
            # Fast path: precomputed iteration-invariant state; the expensive
            # ss/XtX/wpca_own are read from cache, not rebuilt.
            mdata = _mapping_window_fast(gnnm, proj_cache, K=K, pairwise=pairwise)
        else:
            # Legacy path: rebuild precompute on the fly (wasteful but correct).
            mdata = _mapping_window(sams, gnnm, gn, K=K, pairwise=pairwise)

    k1 = K

    if keys is None:
        keys = dict.fromkeys(sams.keys(), "leiden_clusters")

    nnms_in: dict[str, Any] = {}
    nnms_in0: dict[str, Any] = {}
    any_nfk = False
    for sid in sams:
        logger.info("Expanding neighbourhoods of species %s...", sid)
        cl = sams[sid].get_labels(keys[sid])
        _, ix, cluc = np.unique(cl, return_counts=True, return_inverse=True)
        K_arr = cluc[ix]
        nnms_in0[sid] = sams[sid].adata.obsp["connectivities"].copy()
        if not neigh_from_keys[sid]:
            nnm_in = _smart_expand(nnms_in0[sid], K_arr, NH=NHS[sid], bk=bk)
            nnm_in.data[:] = 1
            nnms_in[sid] = nnm_in
        else:
            nnms_in[sid] = _generate_coclustering_matrix(cl)
            any_nfk = True

    # --- Species layout in global index space -------------------------------
    sids = list(sams.keys())
    n_cells: dict[str, int] = {sid: nnms_in0[sid].shape[0] for sid in sids}
    offsets: dict[str, int] = {}
    _off = 0
    for sid in sids:
        offsets[sid] = _off
        _off += n_cells[sid]
    N = _off

    nnm_internal0 = sp.sparse.block_diag(list(nnms_in0.values())).tocsr()

    logger.info("Indegree coarsening")

    # Original non-coclustering path applied a 0.1 floor to D before
    # mutualisation; the coclustering path did not. Preserve that asymmetry.
    threshold = 0.0 if any_nfk else 0.1

    if scale_edges_by_corr:
        logger.info("Rescaling edge weights by expression correlations.")

    Dk = _compute_mutual_graph(
        nnms_in,
        neigh_from_keys,
        mdata["knn"],
        offsets,
        n_cells,
        sids,
        k1,
        N,
        pairwise=pairwise,
        chunksize=chunksize,
        threshold=threshold,
        scale_edges_by_corr=scale_edges_by_corr,
        wPCA=mdata["wPCA"] if scale_edges_by_corr else None,
        bk=bk,
    )

    del nnms_in
    gc.collect()

    if not pairwise or len(sids) == 2:
        denom = k1
    else:
        denom = k1 * (len(sids) - 1)

    species_list = []
    for sid in sids:
        species_list += [sid] * n_cells[sid]
    species_list = np.array(species_list)

    sr = np.asarray(Dk.sum(1))

    x = 1 - sr.flatten() / denom

    omp = nnm_internal0.tocsr()
    omp.data[:] = 1
    NNM = omp.multiply(x[:, None])
    NNM = (NNM + Dk).tolil()
    NNM.setdiag(0)

    logger.info("Concatenating SAM objects...")
    sam3 = _concatenate_sam(sams, NNM)

    sam3.adata.obs["species"] = pd.Categorical(species_list)

    sam3.adata.uns["gnnm_corr"] = mdata.get("gnnm_corr", None)

    if umap:
        logger.info("Computing UMAP projection...")
        maxiter = (
            UMAP_MAXITER_SMALL if sam3.adata.shape[0] <= UMAP_SIZE_THRESHOLD else UMAP_MAXITER_LARGE
        )
        sc.tl.umap(sam3.adata, min_dist=UMAP_MIN_DIST, maxiter=maxiter)
    return sam3


def _concatenate_sam(sams: dict[str, SAM], nnm: sp.sparse.lil_matrix) -> SAM:
    """Concatenate SAM objects."""
    acns = []
    exps = []
    agns = []
    sps = []
    for i, sid in enumerate(sams.keys()):
        acns.append(_q(sams[sid].adata.obs_names))
        sps.append([sid] * acns[-1].size)
        exps.append(sams[sid].adata.X)
        agns.append(_q(sams[sid].adata.var_names))

    acn = np.concatenate(acns)
    agn = np.concatenate(agns)
    sps_arr = np.concatenate(sps)

    xx = sp.sparse.block_diag(exps, format="csr")

    sam = SAM(counts=(xx, agn, acn))

    sam.adata.uns["neighbors"] = {}
    nnm = nnm.tocsr()
    nnm.eliminate_zeros()
    sam.adata.obsp["connectivities"] = nnm
    sam.adata.uns["neighbors"]["params"] = {
        "n_neighbors": 15,
        "method": "umap",
        "use_rep": "X",
        "metric": "euclidean",
    }
    for i in sams:
        for k in sams[i].adata.obs:
            if sams[i].adata.obs[k].dtype.name == "category":
                z = np.array(["unassigned"] * sam.adata.shape[0], dtype="object")
                z[sps_arr == i] = _q(sams[i].adata.obs[k])
                sam.adata.obs[i + "_" + k] = pd.Categorical(z)

    a = []
    for i, sid in enumerate(sams.keys()):
        a.extend(["batch" + str(i + 1)] * sams[sid].adata.shape[0])
    sam.adata.obs["batch"] = pd.Categorical(np.array(a))
    sam.adata.obs.columns = sam.adata.obs.columns.astype("str")
    sam.adata.var.columns = sam.adata.var.columns.astype("str")

    for i in sam.adata.obs:
        sam.adata.obs[i] = sam.adata.obs[i].astype("str")

    return sam
