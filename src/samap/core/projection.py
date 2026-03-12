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

import hnswlib
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.preprocessing import StandardScaler

from samap._logging import logger
from samap.core._backend import Backend
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
    ra = sam.adata.uns["run_args"]
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
    ef: int = 200,
    M: int = 48,
) -> sp.sparse.csr_matrix:
    """Project between feature spaces using HNSW."""
    metric = "l2" if metric == "euclidean" else metric
    metric = "cosine" if metric == "correlation" else metric
    labels2 = np.arange(wpca2.shape[0])
    p2 = hnswlib.Index(space=metric, dim=wpca2.shape[1])
    p2.init_index(max_elements=wpca2.shape[0], ef_construction=ef, M=M)
    p2.add_items(wpca2, labels2)
    p2.set_ef(ef)
    idx1, dist1 = p2.knn_query(wpca1, k=k)

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
    knn1v2 = sp.sparse.lil_matrix((wpca1.shape[0], wpca2.shape[0]))
    x1 = np.tile(np.arange(idx1.shape[0])[:, None], (1, idx1.shape[1])).flatten()
    knn1v2[x1.astype("int32"), idx1.flatten().astype("int32")] = Sim1.flatten()
    return knn1v2.tocsr()


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

    Returns
    -------
    dict
        Keys: ``sids``, ``gs``, ``W``, ``species_indexer``, ``genes_indexer``,
        ``ss``, ``PCs``, ``n_cells``, ``XtX``, ``mu_ss``, ``wpca_own``,
        ``M_own``, ``bk``. All array-valued entries live on ``bk``'s device.
    """
    if bk is None:
        bk = Backend("cpu")

    std = StandardScaler(with_mean=False)

    sids = list(sams.keys())
    gs: dict[str, NDArray[Any]] = {}
    W: dict[str, Any] = {}
    ss: dict[str, Any] = {}
    PCs: dict[str, Any] = {}
    n_cells: dict[str, int] = {}
    species_indexer: list[NDArray[Any]] = []
    genes_indexer: list[NDArray[Any]] = []

    for sid in sids:
        gs[sid] = gns[np.isin(gns, _q(sams[sid].adata.var_names))]
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

    # ---- Assemble wpca: row-block i = hstack of column-blocks, minus M --- #
    N_total = sum(n_cells.values())
    npcs_blocks = [PCs[sid].shape[1] for sid in sids]
    npcs_total = sum(npcs_blocks)
    wpca = bk.xp.zeros((N_total, npcs_total), dtype=bk.xp.float64)

    col_offsets = np.cumsum([0, *npcs_blocks])
    for i, sid_i in enumerate(sids):
        r0, r1 = species_indexer[i][0], species_indexer[i][-1] + 1
        M_i = bk.xp.concatenate(M_blocks[i])  # full-width correction vector for species i
        for s in range(n_species):
            c0, c1 = col_offsets[s], col_offsets[s + 1]
            block = row_blocks[i][s]
            # row_blocks may come out sparse (rare, e.g. all-zero G_is); coerce
            if hasattr(block, "toarray"):
                block = block.toarray()
            wpca[r0:r1, c0:c1] = block
        wpca[r0:r1] -= M_i

    logger.info("Correcting data with means. %.2fs", time.time() - ttt)

    # ---- Cross-species kNN via HNSW (host, numpy — hnswlib is CPU-only) -- #
    wpca_host = bk.to_host(wpca)
    gnnm_corr_host = bk.to_host(gnnm_corr)

    k = K
    ixg = np.arange(wpca_host.shape[0])
    Xs: list[Any] = []
    Ys: list[Any] = []
    Vs: list[Any] = []
    for i in range(n_species):
        ixq = species_indexer[i]
        query = wpca_host[ixq]
        for j in range(n_species):
            if i == j:
                continue
            ixr = species_indexer[j]
            reference = wpca_host[ixr]

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

    return {
        "knn": knn.tocsr(),
        "wPCA": wpca_host,
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

