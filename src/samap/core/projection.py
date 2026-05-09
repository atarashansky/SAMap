"""Cross-species feature projection and kNN construction.

Projects cells from each species into a joint latent space via the homology
graph, then builds the cross-species kNN graph with HNSW.

Implementation notes — precomposed feature translation
------------------------------------------------------
The legacy algorithm materialised a cells × genes translated-feature matrix
``Xtr = X_i @ G_ij`` per species pair, scaled it column-wise, weighted it by
gene weights, then projected it through the target species' PC loadings.
For realistic datasets that intermediate is ~30% dense and dominates both
memory and wall time.

We now precompose the projection. Writing the column-wise scaling as a
diagonal matrix ``D = diag(W_j / σ)``, the cross contribution is

    wpca_cross = (X_i @ G_ij) · D · PCs_j
               = X_i @ (G_ij · D · PCs_j)
               = X_i @ P_ij

where ``P_ij`` has shape (G_i × npcs) — typically a few MB — and the final
result follows from one SpMM. The per-column standard deviation ``σ`` is
recovered from iteration-invariant precomputes (``X_i^T X_i`` and
``X_i.mean(0)``) via a quadratic form in the columns of ``G_ij``, so the
dense intermediate is never materialised.

The own-species contribution ``X_i @ PCs_i`` and its mean correction do not
depend on the homology graph at all and are cached in :func:`_projection_precompute`.
"""

from __future__ import annotations

import gc
import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy as sp
from sklearn.preprocessing import StandardScaler

from samap._logging import logger
from samap.core._backend import Backend
from samap.core.knn import _hnswlib_build, _hnswlib_query, approximate_knn
from samap.utils import q as _q

from .homology import _tanh_scale

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray

    from samap.sam import SAM


def prepare_SAMap_loadings(sam: SAM, npcs: int = 300) -> None:
    """Prepare SAM object with PC loadings for manifold.

    Parameters
    ----------
    sam : SAM
        SAM object to prepare.
    npcs : int, optional
        Number of PCs to calculate. Default 300.
    """
    # scanpy.external.tl.sam nests run_args under uns["sam"] (#156)
    ra = sam.adata.uns.get("run_args") or sam.adata.uns.get("sam", {}).get("run_args")
    if ra is None:
        raise KeyError(
            "run_args not found in adata.uns — was this AnnData processed with SAM.run()?"
        )
    preprocessing = ra.get("preprocessing", "StandardScaler")
    weight_PCs = ra.get("weight_PCs", False)
    A, _ = sam.calculate_nnm(
        n_genes=sam.adata.shape[1],
        preprocessing=preprocessing,
        npcs=npcs,
        weight_PCs=weight_PCs,
        sparse_pca=True,
        update_manifold=False,
        weight_mode="dispersion",
    )
    sam.adata.varm["PCs_SAMap"] = A


def _united_proj(
    wpca1: NDArray[Any],
    wpca2: NDArray[Any],
    k: int = 20,
    metric: str = "cosine",
    bk: Backend | None = None,
) -> sp.sparse.csr_matrix:
    """Build cross-species kNN sparse graph with similarity weights.

    Finds the ``k`` nearest neighbours of each row of ``wpca1`` in
    ``wpca2``, transforms distances into similarity weights, and returns a
    sparse (n_q, n_d) CSR.

    The kNN search is delegated to :func:`samap.core.knn.approximate_knn`,
    which dispatches between CPU HNSW (hnswlib) and GPU brute-force (FAISS)
    based on ``bk``. When ``bk`` is ``None`` a CPU backend is used.
    """
    metric = "l2" if metric == "euclidean" else metric
    metric = "cosine" if metric == "correlation" else metric

    idx1, dist1 = approximate_knn(wpca1, wpca2, k=k, metric=metric, bk=bk)

    if metric == "cosine":
        dist1 = 1 - dist1
        dist1[dist1 < 1e-3] = 1e-3
        dist1 = dist1 / dist1.max(1)[:, None]
        dist1 = _tanh_scale(dist1, scale=10, center=0.7)
    else:
        sigma1 = dist1[:, 4]
        sigma1[sigma1 < 1e-3] = 1e-3
        dist1 = np.exp(-dist1 / sigma1[:, None])

    Sim1 = dist1
    n_q = wpca1.shape[0]
    n_d = wpca2.shape[0]
    rows = np.repeat(np.arange(n_q, dtype=np.int32), k)
    cols = idx1.ravel().astype(np.int32)
    vals = Sim1.ravel()
    return sp.sparse.coo_matrix((vals, (rows, cols)), shape=(n_q, n_d)).tocsr()


# --------------------------------------------------------------------------- #
# Sigma from precomputes                                                      #
# --------------------------------------------------------------------------- #


def _compute_sigma(
    XtX: Any,
    mu: NDArray[Any],
    G: Any,
    n: int,
    bk: Backend | None = None,
) -> NDArray[Any]:
    """Column-wise standard deviation of ``X @ G`` without materialising it.

    Equivalent to ``StandardScaler(with_mean=False).fit(X @ G).scale_``.

    Parameters
    ----------
    XtX : sparse (G_i × G_i)
        Precomputed Gram matrix ``X.T @ X``.
    mu : 1-d array, length G_i
        Precomputed column means ``X.mean(axis=0)``.
    G : sparse (G_i × G_j)
        The homology sub-block whose columns we're scaling.
    n : int
        Number of rows in ``X`` (cells in the source species).
    bk : Backend or None
        Array backend. If None, uses numpy/scipy directly.

    Returns
    -------
    sigma : 1-d array, length G_j
        Per-column biased standard deviation, with zero-variance columns
        mapped to 1.0 (matching sklearn's ``_handle_zeros_in_scale``).

    Notes
    -----
    Uses the identity ``diag(Gᵀ·XtX·G)_k = Σ_r (X·g_k)[r]²`` so that
    ``σ_k² = diag(Gᵀ·XtX·G)_k / n − (μ·g_k)²``. The diagonal is extracted
    as ``((XtX @ G) ⊙ G).sum(0)`` — one SpGEMM + one elementwise product,
    never forming the full G_j × G_j outer product.
    """
    xp = bk.xp if bk is not None else np
    # diag(Gᵀ · XtX · G) — elementwise-multiply trick avoids G_j × G_j dense
    sq = xp.asarray((XtX @ G).multiply(G).sum(0)).flatten()
    mu_terms = xp.asarray(mu @ G).flatten()
    var = sq / n - mu_terms * mu_terms
    # numerical guard — floating-point cancellation can produce tiny negatives
    var = xp.maximum(var, 0.0)
    sigma = xp.sqrt(var)
    # sklearn maps zero-variance columns to scale_=1.0
    sigma = xp.where(sigma == 0.0, 1.0, sigma)
    return sigma


# --------------------------------------------------------------------------- #
# Iteration-invariant precompute                                              #
# --------------------------------------------------------------------------- #


def _projection_precompute(
    sams: dict[str, SAM],
    gns: NDArray[Any],
    bk: Backend | None = None,
    precomputed: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the iteration-invariant state for :func:`_mapping_window_fast`.

    Everything that depends only on the input SAM objects — not on the
    homology graph — is computed once here and cached. Specifically: the
    standardised, gene-weighted expression matrices; their Gram matrices and
    column means (for the sigma quadratic form); and the own-species PC
    projection (which never changes across SAMap iterations).

    Parameters
    ----------
    sams : dict[str, SAM]
        Input per-species SAM objects. Must have ``adata.varm['PCs_SAMap']``
        and ``adata.var['weights']`` populated.
    gns : array of str
        Concatenated homology-graph gene names, species-prefixed and ordered
        so that species blocks are contiguous.
    bk : Backend or None
        Array backend for device placement. Default: CPU.
    precomputed : dict[str, dict] or None
        Optional per-species cache loaded by :func:`samap.io.load_precompute`.
        When an entry for ``sid`` is present its full-gene ``XtX`` is sliced
        to ``gs[sid]`` instead of recomputing the SpGEMM. The cached
        ``PCs_SAMap`` is expected to have already been placed into
        ``sams[sid].adata.varm`` by the caller.

    Returns
    -------
    dict
        Keys: ``sids``, ``gs``, ``W``, ``species_indexer``, ``genes_indexer``,
        ``ss``, ``PCs``, ``n_cells``, ``XtX``, ``mu_ss``, ``wpca_own``,
        ``M_own``, ``bk``. All array-valued entries live on ``bk``'s device.
    """
    if bk is None:
        bk = Backend("cpu")
    if precomputed is None:
        precomputed = {}

    std = StandardScaler(with_mean=False)

    sids = list(sams.keys())
    gs: dict[str, NDArray[Any]] = {}
    W: dict[str, Any] = {}
    ss: dict[str, Any] = {}
    PCs: dict[str, Any] = {}
    n_cells: dict[str, int] = {}
    species_indexer: list[NDArray[Any]] = []
    genes_indexer: list[NDArray[Any]] = []
    gs_ix: dict[str, NDArray[Any]] = {}

    for sid in sids:
        var_names = _q(sams[sid].adata.var_names)
        gs[sid] = gns[np.isin(gns, var_names)]
        # positional indices of gs[sid] in the species' var_names — used to
        # slice the full-gene cached XtX
        gs_ix[sid] = pd.Index(var_names).get_indexer(gs[sid])
        sub = sams[sid].adata[:, gs[sid]]
        W[sid] = bk.to_device(bk.xp.asarray(sub.var["weights"].values))
        # StandardScaler runs on host; move result to device after
        ss_host = std.fit_transform(sub.X).multiply(sub.var["weights"].values[None, :]).tocsr()
        ss[sid] = bk.to_device(ss_host)
        PCs[sid] = bk.to_device(bk.xp.asarray(sub.varm["PCs_SAMap"]))
        n_cells[sid] = ss_host.shape[0]
        species_indexer.append(np.arange(n_cells[sid]))
        genes_indexer.append(np.arange(gs[sid].size))

    for i in range(1, len(species_indexer)):
        species_indexer[i] = species_indexer[i] + species_indexer[i - 1].max() + 1
        genes_indexer[i] = genes_indexer[i] + genes_indexer[i - 1].max() + 1

    # Gram matrices + column means — feed the sigma quadratic form
    XtX: dict[str, Any] = {}
    mu_ss: dict[str, Any] = {}
    for sid in sids:
        cached = precomputed.get(sid)
        if cached is not None:
            # Slice the cached full-gene Gram to this run's homology-connected
            # genes. XtX over a column subset equals the row+column slice of
            # the full Gram (XᵀX is bilinear in column selection).
            ix = gs_ix[sid]
            XtX_full = cached["XtX"]
            XtX[sid] = bk.to_device(XtX_full[ix, :][:, ix].tocsr())
            logger.info(
                "_projection_precompute[%s]: using cached XtX (sliced %d→%d genes).",
                sid,
                XtX_full.shape[0],
                ix.size,
            )
        else:
            XtX[sid] = (ss[sid].T @ ss[sid]).tocsr()
        mu_ss[sid] = bk.xp.asarray(ss[sid].mean(0)).flatten()

    # Own-species PC projection — fully iteration-invariant
    wpca_own: dict[str, Any] = {}
    M_own: dict[str, Any] = {}
    for sid in sids:
        wpca_own[sid] = ss[sid] @ PCs[sid]  # N_sid × npcs_sid
        M_own[sid] = mu_ss[sid] @ PCs[sid]  # npcs_sid

    return {
        "sids": sids,
        "gs": gs,
        "W": W,
        "species_indexer": species_indexer,
        "genes_indexer": genes_indexer,
        "ss": ss,
        "PCs": PCs,
        "n_cells": n_cells,
        "XtX": XtX,
        "mu_ss": mu_ss,
        "wpca_own": wpca_own,
        "M_own": M_own,
        "bk": bk,
    }


# --------------------------------------------------------------------------- #
# Tiled wPCA                                                                  #
# --------------------------------------------------------------------------- #


class _TiledWPCA:
    """Lazy per-pair view of the joint embedding.

    The full joint embedding is conceptually ``wpca[N_total, S·npcs]`` —
    the row-block for species ``i`` is ``hstack(row_blocks[i][s] for s)``
    minus the per-species mean-correction ``M_blocks[i][s]``. At S=21,
    npcs=300 that buffer is ~35 GB f64; at S=100 it is 840 GB. This path
    never allocates it. Instead this object stores the mean-corrected
    ``row_blocks`` (each N_i × npcs_s, float32) and exposes:

    * :meth:`pair_view` — for ``pairwise=True`` kNN: assembles
      ``[row_blocks[i][i] | row_blocks[i][j]]`` (N_i × 2·npcs) and the
      matching reference ``[row_blocks[j][i] | row_blocks[j][j]]``
      (N_j × 2·npcs) on demand. Peak ~2·N_max·2·npcs f32 ≈ 0.24 GB at
      S=21.
    * :meth:`row_embedding` — returns species ``i``'s full-width
      mean-corrected row block (N_i × S·npcs). Used by the
      ``pairwise=False`` index-reuse path and as a compatibility fallback
      for callers that need a global embedding.
    * :meth:`materialise_full` — assembles the full N_total × S·npcs
      matrix. Float32. Only called on the ``pairwise=False`` path, where
      a single shared latent space is the algorithm's intent; still 2×
      smaller than the legacy f64 buffer.

    All blocks are stored mean-corrected and as float32 — hnswlib casts to
    f32 anyway, so the f64 storage was pure waste.

    Implements ``__getitem__`` over global cell indices so legacy callers
    (``_replace`` in ``_scale_by_corr``) keep working when handed this
    object in place of a dense ``wPCA`` array. Indexing materialises only
    the requested rows.
    """

    def __init__(
        self,
        row_blocks: list[list[Any]],
        species_indexer: list[NDArray[Any]],
        npcs_blocks: list[int],
        sids: list[str],
    ) -> None:
        self._rb = row_blocks  # row_blocks[i][s] : (N_i × npcs_s), mean-corrected, f32
        self._spix = species_indexer
        self._npcs = npcs_blocks
        self._col_off = np.cumsum([0, *npcs_blocks])
        self._sids = sids
        self._n_species = len(sids)
        self._N = int(species_indexer[-1][-1] + 1)
        self.shape = (self._N, int(self._col_off[-1]))
        # global cell index → (species_idx, local_idx)
        self._sp_of = np.empty(self._N, dtype=np.int32)
        self._loc_of = np.empty(self._N, dtype=np.int32)
        for i, ix in enumerate(species_indexer):
            self._sp_of[ix] = i
            self._loc_of[ix] = np.arange(ix.size, dtype=np.int32)

    # ---- pairwise kNN tiles -------------------------------------------- #
    def pair_view(self, i: int, j: int) -> tuple[NDArray[Any], NDArray[Any]]:
        """``(query, reference)`` for the directed (i→j) kNN at 2·npcs width.

        ``query``     = species i's cells in [PCs_i | PCs_j]  (N_i × (npcs_i+npcs_j))
        ``reference`` = species j's cells in [PCs_i | PCs_j]  (N_j × (npcs_i+npcs_j))

        Both float32, C-contiguous. Allocated fresh per call (cheap — MB-scale).
        """
        q = np.ascontiguousarray(np.hstack((self._rb[i][i], self._rb[i][j])))
        r = np.ascontiguousarray(np.hstack((self._rb[j][i], self._rb[j][j])))
        return q, r

    # ---- pairwise=False index reuse ------------------------------------ #
    def row_embedding(self, i: int) -> NDArray[Any]:
        """Species ``i``'s full-width row block (N_i × S·npcs, float32)."""
        return np.ascontiguousarray(np.hstack(self._rb[i]))

    def materialise_full(self, *, free: bool = False) -> NDArray[Any]:
        """Assemble the full N_total × S·npcs float32 matrix.

        Provided for ``pairwise=False`` and for downstream consumers that
        still need a global embedding (e.g. UMAP on wPCA, if ever added).
        Memory cost is half the legacy path (f32 vs f64).

        With ``free=True``, releases each row-block tile immediately after
        it's been copied into ``out``, so peak memory stays at
        ~N_total × S·npcs × 4 bytes (one f32 copy of the data) instead of
        2×. After a freeing materialise, :meth:`pair_view` /
        :meth:`row_embedding` are no longer usable — call this last.
        """
        out = np.empty(self.shape, dtype=np.float32)
        for i in range(self._n_species):
            r0, r1 = self._spix[i][0], self._spix[i][-1] + 1
            for s in range(self._n_species):
                c0, c1 = self._col_off[s], self._col_off[s + 1]
                out[r0:r1, c0:c1] = self._rb[i][s]
                if free:
                    self._rb[i][s] = None  # type: ignore[assignment]
        if free:
            self._rb = None  # type: ignore[assignment]
        return out

    # ---- _scale_by_corr per-pair correlation ---------------------------- #
    def pair_corr_basis(self, i: int, j: int) -> tuple[NDArray[Any], NDArray[Any]]:
        """Row sets for correlating cells of species i against species j.

        Returns ``(A, B)`` where ``A`` is N_i × d and ``B`` is N_j × d, with
        ``d = npcs_i + npcs_j``. Same tiles as :meth:`pair_view` — separate
        method only to make the call site self-documenting.
        """
        return self.pair_view(i, j)

    # ---- numpy interop -------------------------------------------------- #
    def __array__(self, dtype: Any = None, copy: bool | None = None) -> NDArray[Any]:
        a = self.materialise_full()
        if dtype is not None:
            a = a.astype(dtype, copy=False)
        return a

    # ---- legacy compatibility: fancy-index by global row ---------------- #
    def __getitem__(self, rows: Any) -> NDArray[Any]:
        rows = np.asarray(rows, dtype=np.int64)
        out = np.empty((rows.size, self.shape[1]), dtype=np.float32)
        # group by source species so each row_block is touched once
        sp = self._sp_of[rows]
        loc = self._loc_of[rows]
        for i in range(self._n_species):
            mask = sp == i
            if not mask.any():
                continue
            li = loc[mask]
            for s in range(self._n_species):
                c0, c1 = self._col_off[s], self._col_off[s + 1]
                out[mask, c0:c1] = self._rb[i][s][li]
        return out


# --------------------------------------------------------------------------- #
# Fast per-iteration path                                                     #
# --------------------------------------------------------------------------- #


def _mapping_window_fast(
    gnnm: Any,
    precompute: dict[str, Any],
    K: int = 20,
    pairwise: bool = True,
) -> dict[str, Any]:
    """Cross-species projection using precomposed feature translation.

    Per-iteration worker: consumes the current homology graph ``gnnm`` and
    the cached invariants from :func:`_projection_precompute`, and produces
    the cross-species kNN graph. Never materialises the cells × genes
    translated feature matrix — see the module docstring for the algebra.

    Parameters
    ----------
    gnnm : sparse (G_total × G_total)
        Current gene-homology graph. Row/column order must match
        ``precompute['gs']`` block structure.
    precompute : dict
        Output of :func:`_projection_precompute`.
    K : int
        Number of nearest neighbours per species pair.
    pairwise : bool
        If True (default), the homology sub-block is re-normalised per
        species pair. If False, the global column normalisation is used as-is.
        These differ for 3+ species.

    Returns
    -------
    dict
        Same shape as legacy ``_mapping_window``: keys ``knn`` (CSR, host),
        ``wPCA`` (dense, host), ``gnnm_corr`` (CSR, host; globally
        column-normalised — downstream consumers rely on this).
    """
    bk: Backend = precompute["bk"]
    sids: list[str] = precompute["sids"]
    n_species = len(sids)
    species_indexer = precompute["species_indexer"]
    genes_indexer = precompute["genes_indexer"]
    ss = precompute["ss"]
    PCs = precompute["PCs"]
    W = precompute["W"]
    n_cells = precompute["n_cells"]
    XtX = precompute["XtX"]
    mu_ss = precompute["mu_ss"]
    wpca_own = precompute["wpca_own"]
    M_own = precompute["M_own"]

    logger.info("Prepping datasets for translation.")

    # ---- Global gnnm_corr preparation (tanh-scale + column-normalise) ---- #
    #  This normalised graph is also a pipeline output (output_dict['gnnm_corr']).
    gnnm_corr = bk.to_device(gnnm.copy())
    gnnm_corr.data[:] = _tanh_scale(bk.to_host(gnnm_corr.data))  # tanh is cheap; stay on host np
    gnnm_corr = bk.to_device(gnnm_corr)
    su = bk.xp.asarray(gnnm_corr.sum(0))
    su = bk.xp.where(su == 0, 1.0, su)
    gnnm_corr = gnnm_corr.multiply(1.0 / su).tocsr()

    ttt = time.time()
    if pairwise:
        logger.info("Translating feature spaces pairwise.")
    else:
        logger.info("Translating feature spaces all-to-all.")

    # Per-species row-block of wpca, each decomposed into per-species column-blocks
    #   row_blocks[i][s]  = N_i × npcs_s contribution
    #   M_blocks[i][s]    = npcs_s mean-correction vector
    row_blocks: list[list[Any]] = [[None] * n_species for _ in range(n_species)]
    M_blocks: list[list[Any]] = [[None] * n_species for _ in range(n_species)]

    for i, sid_i in enumerate(sids):
        gi = genes_indexer[i]
        n_i = n_cells[sid_i]

        for s, sid_s in enumerate(sids):
            if s == i:
                # Own-species contribution — iteration-invariant, just reference
                row_blocks[i][s] = wpca_own[sid_i]
                M_blocks[i][s] = M_own[sid_i]
                continue

            gs = genes_indexer[s]
            # Extract cross-species homology sub-block
            G_is = gnnm_corr[gi[0] : gi[-1] + 1, gs[0] : gs[-1] + 1]

            if pairwise:
                # Re-normalise locally — matches legacy pairwise branch
                col_sum = bk.xp.asarray(G_is.sum(0))
                col_sum = bk.xp.where(col_sum == 0, 1.0, col_sum)
                G_is = G_is.multiply(1.0 / col_sum).tocsr()

            # ---- Sigma via quadratic form (no Xtr materialised) -------- #
            sigma = _compute_sigma(XtX[sid_i], mu_ss[sid_i], G_is, n_i, bk)
            # mu_terms needed below for the mean-correction; recompute (cheap)
            mu_terms = bk.xp.asarray(mu_ss[sid_i] @ G_is).flatten()

            # ---- Precompose: P_is = G · diag(W/σ) · PCs_s --------------- #
            #  Shape: (G_i × npcs_s), dense — typically a few MB regardless of N
            scale = W[sid_s] / sigma
            P_is = G_is.multiply(scale).tocsr() @ PCs[sid_s]

            # ---- ONE SpMM replaces the N_i × G_s dense intermediate ---- #
            row_blocks[i][s] = ss[sid_i] @ P_is  # N_i × npcs_s

            # Mean-correction vector for this block — same identity as sigma,
            #   mu(Xtr_weighted) = (mu_ss · G) / σ · W
            mu_cross = mu_terms / sigma * W[sid_s]
            M_blocks[i][s] = mu_cross @ PCs[sid_s]

    gc.collect()

    logger.info("Projecting data into joint latent space. %.2fs", time.time() - ttt)
    ttt = time.time()

    # ---- Never assemble full N_total × S·npcs wpca. Instead store --------
    #  mean-corrected row_blocks (each N_i × npcs_s) as float32 and wrap them
    #  in a _TiledWPCA that materialises per-pair tiles on demand.
    npcs_blocks = [PCs[sid].shape[1] for sid in sids]
    for i in range(n_species):
        for s in range(n_species):
            block = row_blocks[i][s]
            if hasattr(block, "toarray"):
                block = block.toarray()
            # apply mean correction in-place at block granularity (was done as
            # one wide subtraction on the full wpca before)
            block = bk.to_host(block).astype(np.float32, copy=False)
            block = block - bk.to_host(M_blocks[i][s]).astype(np.float32)
            row_blocks[i][s] = np.ascontiguousarray(block)
    # M_blocks no longer needed
    del M_blocks
    gc.collect()

    wpca_tiled = _TiledWPCA(row_blocks, species_indexer, npcs_blocks, sids)
    N_total = wpca_tiled.shape[0]

    logger.info("Correcting data with means. %.2fs", time.time() - ttt)

    # ---- Cross-species kNN ------------------------------------------------ #
    gnnm_corr_host = bk.to_host(gnnm_corr)
    k = K
    Xs: list[Any] = []
    Ys: list[Any] = []
    Vs: list[Any] = []

    def _emit(b_csr: Any, ixq: NDArray[Any], ixr: NDArray[Any]) -> None:
        su = np.asarray(b_csr.sum(1))
        su[su == 0] = 1
        bn = b_csr.multiply(1 / su).tocsr()
        x, y = bn.nonzero()
        Xs.extend(ixq[x])
        Ys.extend(ixr[y])
        Vs.extend(bn.data)

    if pairwise:
        # pairwise=True: each (i→j) search runs in 2·npcs dims
        # using only the [PCs_i | PCs_j] columns. The other S−2 column blocks
        # are species-i-projected-through-other-species — zero-mean noise
        # w.r.t. the i↔j cosine. Dropping them is correctness-neutral up to
        # the small norm perturbation those noise columns add to the cosine
        # denominator. Per-pair index builds remain unavoidable (the
        # reference for j depends on i via row_blocks[j][i]) but at 2·npcs
        # not S·npcs — the 10.5× kNN win at S=21.
        for i in range(n_species):
            ixq = species_indexer[i]
            for j in range(n_species):
                if i == j:
                    continue
                ixr = species_indexer[j]
                query, reference = wpca_tiled.pair_view(i, j)
                b = _united_proj(query, reference, k=k, bk=bk)
                _emit(b, ixq, ixr)
    elif bk.gpu:
        # GPU brute-force is exact and doesn't amortise across queries the
        # way HNSW does — keep the existing per-pair _united_proj dispatch,
        # but on the f32 full-width tiles (no f64 buffer).
        for i in range(n_species):
            ixq = species_indexer[i]
            query = wpca_tiled.row_embedding(i)
            for j in range(n_species):
                if i == j:
                    continue
                ixr = species_indexer[j]
                reference = wpca_tiled.row_embedding(j)
                b = _united_proj(query, reference, k=k, bk=bk)
                _emit(b, ixq, ixr)
    else:
        # pairwise=False (CPU): the reference embedding for species j
        # in the joint space is shared across all queriers i. Build S HNSW
        # indices once at full S·npcs width, query each from all S−1 others.
        # 21× index-build saving at S=21 vs the legacy S(S−1) builds.
        logger.info("pairwise=False: building %d shared HNSW indices.", n_species)
        # Precompute each species' full-width embedding once.
        embeds = [wpca_tiled.row_embedding(i) for i in range(n_species)]
        for j in range(n_species):
            ixr = species_indexer[j]
            index = _hnswlib_build(embeds[j], metric="cosine")
            for i in range(n_species):
                if i == j:
                    continue
                ixq = species_indexer[i]
                idx, dist = _hnswlib_query(index, embeds[i], k=k)
                # replicate _united_proj's similarity transform
                d = 1 - dist
                d[d < 1e-3] = 1e-3
                d = d / d.max(1)[:, None]
                d = _tanh_scale(d, scale=10, center=0.7)
                rows = np.repeat(np.arange(idx.shape[0], dtype=np.int32), k)
                b = sp.sparse.coo_matrix(
                    (d.ravel(), (rows, idx.ravel().astype(np.int32))),
                    shape=(idx.shape[0], embeds[j].shape[0]),
                ).tocsr()
                _emit(b, ixq, ixr)
            del index
        del embeds
        gc.collect()

    knn = sp.sparse.coo_matrix((Vs, (Xs, Ys)), shape=(N_total, N_total))

    return {
        "knn": knn.tocsr(),
        "wPCA": wpca_tiled,
        "gnnm_corr": gnnm_corr_host,
    }


# --------------------------------------------------------------------------- #
# Backward-compat wrapper                                                     #
# --------------------------------------------------------------------------- #


def _mapping_window(
    sams: dict[str, SAM],
    gnnm: sp.sparse.csr_matrix | None = None,
    gns: NDArray[Any] | None = None,
    K: int = 20,
    pairwise: bool = True,
) -> dict[str, Any]:
    """Cross-species projection — backward-compatible entry point.

    Builds the precompute dict on-the-fly and delegates to
    :func:`_mapping_window_fast`. For iterative use, prefer calling
    :func:`_projection_precompute` once and :func:`_mapping_window_fast`
    per iteration — the precompute is iteration-invariant and expensive.

    When ``gnnm is None`` (own-species-only projection, used by the no-graph
    bootstrap path), falls back to the legacy implementation.
    """
    if gnnm is not None and gns is not None:
        pre = _projection_precompute(sams, gns, bk=Backend("cpu"))
        return _mapping_window_fast(gnnm, pre, K=K, pairwise=pairwise)

    # ---- No-graph path (legacy, unchanged) ------------------------------- #
    #  Only the own-species projection — no homology graph, no cross blocks.
    std = StandardScaler(with_mean=False)
    adatas: dict[str, Any] = {}
    Ws: dict[str, Any] = {}
    ss: dict[str, Any] = {}
    species_indexer: list[NDArray[Any]] = []
    mus: list[Any] = []
    for sid in sams:
        adatas[sid] = sams[sid].adata
        Ws[sid] = adatas[sid].var["weights"].values
        ss[sid] = std.fit_transform(adatas[sid].X).multiply(Ws[sid][None, :]).tocsr()
        mus.append(np.asarray(ss[sid].mean(0)).flatten())
        species_indexer.append(np.arange(ss[sid].shape[0]))
    for i in range(1, len(species_indexer)):
        species_indexer[i] = species_indexer[i] + species_indexer[i - 1].max() + 1
    X = sp.sparse.vstack(list(ss.values()))
    C = np.hstack([adatas[sid].varm["PCs_SAMap"] for sid in sams])
    wpca = X.dot(C)
    M = np.vstack(mus).dot(C)
    for i, sid in enumerate(sams.keys()):
        ixq = species_indexer[i]
        wpca[ixq] -= M[i]

    k = K
    ixg = np.arange(wpca.shape[0])
    Xs = []
    Ys = []
    Vs = []
    for i, sid in enumerate(sams.keys()):
        ixq = species_indexer[i]
        query = wpca[ixq]
        for j, _sid2 in enumerate(sams.keys()):
            if i != j:
                ixr = species_indexer[j]
                reference = wpca[ixr]

                b = _united_proj(query, reference, k=k)

                su = np.asarray(b.sum(1))
                su[su == 0] = 1
                b = b.multiply(1 / su).tocsr()

                A = pd.Series(index=np.arange(b.shape[0]), data=ixq)
                B = pd.Series(index=np.arange(b.shape[1]), data=ixr)

                x, y = b.nonzero()
                x, y = A[x].values, B[y].values
                Xs.extend(x)
                Ys.extend(y)
                Vs.extend(b.data)

    knn = sp.sparse.coo_matrix((Vs, (Xs, Ys)), shape=(ixg.size, ixg.size))

    return {"knn": knn.tocsr(), "wPCA": wpca}
