"""Cross-species kNN graph stitching and manifold assembly.

The `_mapper` function here is the core graph-coarsening step: it takes
per-species neighbourhoods and the cross-species projection kNN, stitches them
together via in-degree coarsening, and produces the combined SAM manifold.
"""

from __future__ import annotations

import gc
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

from .correlation import _replace
from .expand import _smart_expand
from .homology import _tanh_scale
from .projection import _mapping_window

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
    **kwargs: Any,
) -> SAM:
    """Map cells between species."""
    if NHS is None:
        NHS = dict.fromkeys(sams.keys(), 3)

    if neigh_from_keys is None:
        neigh_from_keys = dict.fromkeys(sams.keys(), False)

    if mdata is None:
        mdata = _mapping_window(sams, gnnm, gn, K=K, pairwise=pairwise)

    k1 = K

    if keys is None:
        keys = dict.fromkeys(sams.keys(), "leiden_clusters")

    nnms_in: dict[str, Any] = {}
    nnms_in0: dict[str, Any] = {}
    flag = False
    species_indexer = []
    for sid in sams:
        logger.info("Expanding neighbourhoods of species %s...", sid)
        cl = sams[sid].get_labels(keys[sid])
        _, ix, cluc = np.unique(cl, return_counts=True, return_inverse=True)
        K_arr = cluc[ix]
        nnms_in0[sid] = sams[sid].adata.obsp["connectivities"].copy()
        species_indexer.append(np.arange(sams[sid].adata.shape[0]))
        if not neigh_from_keys[sid]:
            nnm_in = _smart_expand(nnms_in0[sid], K_arr, NH=NHS[sid])
            nnm_in.data[:] = 1
            nnms_in[sid] = nnm_in
        else:
            nnms_in[sid] = _generate_coclustering_matrix(cl)
            flag = True

    for i in range(1, len(species_indexer)):
        species_indexer[i] += species_indexer[i - 1].max() + 1

    if not flag:
        nnm_internal = sp.sparse.block_diag(list(nnms_in.values())).tocsr()
    nnm_internal0 = sp.sparse.block_diag(list(nnms_in0.values())).tocsr()

    ovt = mdata["knn"]
    ovt0 = ovt.copy()
    ovt0.data[:] = 1

    B = ovt

    logger.info("Indegree coarsening")

    numiter = nnm_internal0.shape[0] // chunksize + 1

    D = sp.sparse.csr_matrix((0, nnm_internal0.shape[0]))
    if flag:
        Cs = []
        for it, sid in enumerate(sams.keys()):
            nfk = neigh_from_keys[sid]
            if nfk:
                Cs.append(nnms_in[sid].dot(nnms_in[sid].T.dot(B.T[species_indexer[it]])))
            else:
                Cs.append(nnms_in[sid].dot(B.T[species_indexer[it]]))
        D = sp.sparse.vstack(Cs).T
        del Cs
        gc.collect()
    else:
        for bl in range(numiter):
            logger.debug("%d/%d, shape %s", bl, numiter, D.shape)
            C = B[bl * chunksize : (bl + 1) * chunksize].dot(nnm_internal.T)
            C.data[C.data < 0.1] = 0
            C.eliminate_zeros()

            D = sp.sparse.vstack((D, C))
            del C
            gc.collect()

    D = D.multiply(D.T).tocsr()
    D.data[:] = D.data**0.5

    if scale_edges_by_corr:
        logger.info("Rescaling edge weights by expression correlations.")
        x, y = D.nonzero()
        vals = _replace(mdata["wPCA"], x, y)
        vals[vals < 1e-3] = 1e-3

        F = D.copy()
        F.data[:] = vals

        ma = np.asarray(F.max(1).todense())
        ma[ma == 0] = 1
        F = F.multiply(1 / ma).tocsr()
        F.data[:] = _tanh_scale(F.data, center=0.7, scale=10)

        ma = np.asarray(D.max(1).todense())
        ma[ma == 0] = 1

        D = F.multiply(D).tocsr()
        D.data[:] = np.sqrt(D.data)

        ma2 = np.asarray(D.max(1).todense())
        ma2[ma2 == 0] = 1

        D = D.multiply(ma / ma2).tocsr()

    species_list = []
    for sid in sams:
        species_list += [sid] * sams[sid].adata.shape[0]
    species_list = np.array(species_list)

    if not pairwise or len(sams.keys()) == 2:
        Dk = sparse_knn(D, k1).tocsr()
        denom = k1
    else:
        Dk = []
        for sid1 in sams:
            row = []
            for sid2 in sams:
                if sid1 != sid2:
                    Dsubk = sparse_knn(D[species_list == sid1][:, species_list == sid2], k1).tocsr()
                else:
                    Dsubk = sp.sparse.csr_matrix((sams[sid1].adata.shape[0],) * 2)
                row.append(Dsubk)
            Dk.append(sp.sparse.hstack(row))
        Dk = sp.sparse.vstack(Dk).tocsr()
        denom = k1 * (len(sams.keys()) - 1)

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
