"""Gene-gene correlation refinement for the homology graph.

Contains the numba-accelerated kernels for computing Pearson / Xi correlations
between homologous gene pairs across the stitched manifold, and the driver
routines that chunk the graph for parallel refinement.
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
    """Compute correlations for pairs in parallel."""
    data = np.zeros(xi.size)
    for i in prange(xi.size):
        x = X[xi[i]]
        y = X[yi[i]]
        data[i] = ((x - x.mean()) * (y - y.mean()) / x.std() / y.std()).sum() / x.size
    return data


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


@njit(parallel=True)
def _refine_corr_kernel(
    p: NDArray[Any],
    ps: NDArray[Any],
    sids: NDArray[Any],
    sixs: list[NDArray[Any]],
    indptr: NDArray[Any],
    indices: NDArray[Any],
    data: NDArray[Any],
    n: int,
    corr_mode: str,
) -> NDArray[np.float64]:
    """Kernel for computing gene correlations in parallel."""
    p1 = p[:, 0]
    p2 = p[:, 1]

    ps1 = ps[:, 0]
    ps2 = ps[:, 1]

    d = {}
    for i in range(len(sids)):
        d[sids[i]] = sixs[i]

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

        a1, a2 = ps1[j], ps2[j]
        ix1 = d[a1]
        ix2 = d[a2]

        xa, xb, ya, yb = x[ix1], x[ix2], y[ix1], y[ix2]
        xx = np.append(xa, xb)
        yy = np.append(ya, yb)

        if corr_mode == "pearson":
            c = ((xx - xx.mean()) * (yy - yy.mean()) / xx.std() / yy.std()).sum() / xx.size
        else:
            c = _xicorr(xx, yy)
        res[j] = c
    return res


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
) -> sp.sparse.csr_matrix:
    """Refine correlation matrix for homology graph."""
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
) -> sp.sparse.csr_matrix:
    """Parallel correlation refinement."""
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
    Xavg = nnms.dot(Xs).tocsc()

    p = pairs
    ps = pairs_species

    gnnm2 = gnnm.multiply(w[:, None]).multiply(w[None, :]).tocsr()
    x, y = gnnm2.nonzero()
    pairs = np.unique(np.sort(np.vstack((x, y)).T, axis=1), axis=0)

    species = _q(st.adata.obs["species"])
    sixs = []
    sidss = np.unique(species)
    for sid in sidss:
        sixs.append(np.where(species == sid)[0])

    vals = _refine_corr_kernel(
        p, ps, sidss, sixs, Xavg.indptr, Xavg.indices, Xavg.data, Xavg.shape[0], corr_mode
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
