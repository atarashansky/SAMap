"""Cross-species feature projection and kNN construction.

Projects cells from each species into a joint latent space via the homology
graph, then builds the cross-species kNN graph with HNSW.
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


def _mapping_window(
    sams: dict[str, SAM],
    gnnm: sp.sparse.csr_matrix | None = None,
    gns: NDArray[Any] | None = None,
    K: int = 20,
    pairwise: bool = True,
) -> dict[str, Any]:
    """Create mapping window for cross-species projection."""
    k = K
    output_dict: dict[str, Any] = {}
    if gnnm is not None and gns is not None:
        logger.info("Prepping datasets for translation.")
        gnnm_corr = gnnm.copy()
        gnnm_corr.data[:] = _tanh_scale(gnnm_corr.data)

        std = StandardScaler(with_mean=False)

        gs = {}
        adatas = {}
        Ws = {}
        ss = {}
        species_indexer = []
        genes_indexer = []
        for sid in sams:
            gs[sid] = gns[np.isin(gns, _q(sams[sid].adata.var_names))]
            adatas[sid] = sams[sid].adata[:, gs[sid]]
            Ws[sid] = adatas[sid].var["weights"].values
            ss[sid] = std.fit_transform(adatas[sid].X).multiply(Ws[sid][None, :]).tocsr()
            species_indexer.append(np.arange(ss[sid].shape[0]))
            genes_indexer.append(np.arange(gs[sid].size))

        for i in range(1, len(species_indexer)):
            species_indexer[i] = species_indexer[i] + species_indexer[i - 1].max() + 1
            genes_indexer[i] = genes_indexer[i] + genes_indexer[i - 1].max() + 1

        su = np.asarray(gnnm_corr.sum(0))
        su[su == 0] = 1
        gnnm_corr = gnnm_corr.multiply(1 / su).tocsr()

        X = sp.sparse.block_diag(list(ss.values())).tocsr()
        W = np.concatenate(list(Ws.values())).flatten()

        ttt = time.time()
        if pairwise:
            logger.info("Translating feature spaces pairwise.")
            Xtr = []
            for i, _sid1 in enumerate(sams.keys()):
                xtr = []
                for j, _sid2 in enumerate(sams.keys()):
                    if i != j:
                        gnnm_corr_sub = gnnm_corr[genes_indexer[i]][:, genes_indexer[j]]
                        su = np.asarray(gnnm_corr_sub.sum(0))
                        su[su == 0] = 1
                        gnnm_corr_sub = gnnm_corr_sub.multiply(1 / su).tocsr()
                        xtr.append(X[species_indexer[i]][:, genes_indexer[i]].dot(gnnm_corr_sub))
                        xtr[-1] = std.fit_transform(xtr[-1]).multiply(W[genes_indexer[j]][None, :])
                    else:
                        xtr.append(
                            sp.sparse.csr_matrix((species_indexer[i].size, genes_indexer[i].size))
                        )
                Xtr.append(sp.sparse.hstack(xtr))
            Xtr = sp.sparse.vstack(Xtr)
        else:
            logger.info("Translating feature spaces all-to-all.")

            Xtr = []
            for i, sid in enumerate(sams.keys()):
                Xtr.append(X[species_indexer[i]].dot(gnnm_corr))
                Xtr[-1] = std.fit_transform(Xtr[-1]).multiply(W[None, :])
            Xtr = sp.sparse.vstack(Xtr)
        Xc = (X + Xtr).tocsr()

        mus = []
        for i, sid in enumerate(sams.keys()):
            mus.append(np.asarray(Xc[species_indexer[i]].mean(0)).flatten())

        gc.collect()

        logger.info("Projecting data into joint latent space. %.2fs", time.time() - ttt)
        C = sp.linalg.block_diag(*[adatas[sid].varm["PCs_SAMap"] for sid in sams])
        M = np.vstack(mus).dot(C)
        ttt = time.time()
        it = 0
        PCAs = []
        for sid in sams:
            PCAs.append(Xc[:, it : it + gs[sid].size].dot(adatas[sid].varm["PCs_SAMap"]))
            it += gs[sid].size
        wpca = np.hstack(PCAs)

        logger.info("Correcting data with means. %.2fs", time.time() - ttt)
        for i, sid in enumerate(sams.keys()):
            ixq = species_indexer[i]
            wpca[ixq] -= M[i]
        output_dict["gnnm_corr"] = gnnm_corr
    else:
        std = StandardScaler(with_mean=False)

        gs = {}
        adatas = {}
        Ws = {}
        ss = {}
        species_indexer = []
        mus = []
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

    output_dict["knn"] = knn.tocsr()
    output_dict["wPCA"] = wpca
    return output_dict
