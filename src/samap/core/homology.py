"""Gene homology graph construction from BLAST results.

Functions for building, coarsening, and filtering the cross-species gene
homology graph that seeds the SAMap iteration.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy as sp

from samap.utils import coo_to_csr_overwrite, df_to_dict
from samap.utils import q as _q

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray

    from samap.sam import SAM


def _tanh_scale(x: NDArray[Any], scale: float = 10, center: float = 0.5) -> NDArray[Any]:
    """Apply tanh scaling to values."""
    return center + (1 - center) * np.tanh(scale * (x - center))


def _calculate_blast_graph(
    ids: list[str],
    f_maps: str = "maps/",
    eval_thr: float = 1e-6,
    reciprocate: bool = False,
) -> tuple[sp.sparse.csr_matrix, NDArray[Any], dict[str, NDArray[Any]]]:
    """Calculate gene homology graph from BLAST results."""
    gns: list[str] = []
    Xs: list[Any] = []
    Ys: list[Any] = []
    Vs: list[Any] = []

    for i in range(len(ids)):
        id1 = ids[i]
        for j in range(i, len(ids)):
            id2 = ids[j]
            if i != j:
                if os.path.exists(f_maps + f"{id1}{id2}"):
                    fA = f_maps + f"{id1}{id2}/{id1}_to_{id2}.txt"
                    fB = f_maps + f"{id1}{id2}/{id2}_to_{id1}.txt"
                elif os.path.exists(f_maps + f"{id2}{id1}"):
                    fA = f_maps + f"{id2}{id1}/{id1}_to_{id2}.txt"
                    fB = f_maps + f"{id2}{id1}/{id2}_to_{id1}.txt"
                else:
                    raise FileNotFoundError(
                        f"BLAST mapping tables with the input IDs ({id1} and {id2}) "
                        f"not found in the specified path."
                    )

                A = pd.read_csv(fA, sep="\t", header=None, index_col=0)
                B = pd.read_csv(fB, sep="\t", header=None, index_col=0)

                A.columns = A.columns.astype("<U100")
                B.columns = B.columns.astype("<U100")

                A = A[A.index.astype("str") != "nan"]
                A = A[A.iloc[:, 0].astype("str") != "nan"]
                B = B[B.index.astype("str") != "nan"]
                B = B[B.iloc[:, 0].astype("str") != "nan"]

                A.index = _prepend_blast_prefix(A.index, id1)
                B[B.columns[0]] = _prepend_blast_prefix(np.asarray(B.iloc[:, 0]), id1)

                B.index = _prepend_blast_prefix(B.index, id2)
                A[A.columns[0]] = _prepend_blast_prefix(np.asarray(A.iloc[:, 0]), id2)

                i1 = np.where(A.columns == "10")[0][0]
                i3 = np.where(A.columns == "11")[0][0]

                inA = _q(A.index)
                inB = _q(B.index)

                inA2 = _q(A.iloc[:, 0])
                inB2 = _q(B.iloc[:, 0])
                gn1 = np.unique(np.append(inB2, inA))
                gn2 = np.unique(np.append(inA2, inB))
                gn = np.append(gn1, gn2)
                gnind = pd.DataFrame(data=np.arange(gn.size)[None, :], columns=gn)

                A.index = pd.Index(np.asarray(gnind[A.index]).flatten())
                B.index = pd.Index(np.asarray(gnind[B.index]).flatten())
                A[A.columns[0]] = np.asarray(gnind[np.asarray(A.iloc[:, 0])]).flatten()
                B[B.columns[0]] = np.asarray(gnind[np.asarray(B.iloc[:, 0])]).flatten()

                Arows = np.vstack((A.index, A.iloc[:, 0], A.iloc[:, i3])).T
                Arows = Arows[np.asarray(A.iloc[:, i1]) <= eval_thr, :]
                gnnm1 = coo_to_csr_overwrite(
                    Arows[:, 0], Arows[:, 1], Arows[:, 2], (gn.size, gn.size)
                )

                Brows = np.vstack((B.index, B.iloc[:, 0], B.iloc[:, i3])).T
                Brows = Brows[np.asarray(B.iloc[:, i1]) <= eval_thr, :]
                gnnm2 = coo_to_csr_overwrite(
                    Brows[:, 0], Brows[:, 1], Brows[:, 2], (gn.size, gn.size)
                )

                gnnm = (gnnm1 + gnnm2).tocsr()
                gnnms = (gnnm + gnnm.T) / 2
                if reciprocate:
                    gnnm.data[:] = 1
                    gnnms = gnnms.multiply(gnnm).multiply(gnnm.T).tocsr()
                gnnm = gnnms

                f1 = np.where(np.isin(gn, gn1))[0]
                f2 = np.where(np.isin(gn, gn2))[0]
                f = np.append(f1, f2)
                gn = gn[f]
                gnnm = gnnm[f, :][:, f]

                V = gnnm.data
                X, Y = gnnm.nonzero()

                Xs.extend(gn[X])
                Ys.extend(gn[Y])
                Vs.extend(V)
                gns.extend(gn)

    gns_arr = np.unique(gns)
    gns_sp = np.array([x.split("_")[0] for x in gns_arr])
    gns2 = []
    gns_dict: dict[str, NDArray[Any]] = {}
    for sid in ids:
        gns2.append(gns_arr[gns_sp == sid])
        gns_dict[sid] = gns2[-1]
    gns_arr = np.concatenate(gns2)
    indexer = pd.Series(index=gns_arr, data=np.arange(gns_arr.size))

    X = indexer[Xs].values
    Y = indexer[Ys].values
    gnnm = sp.sparse.coo_matrix((Vs, (X, Y)), shape=(gns_arr.size, gns_arr.size)).tocsr()

    return gnnm, gns_arr, gns_dict


def _prepend_blast_prefix(data: Any, pre: str) -> NDArray[np.str_]:
    """Add species prefix to gene names."""
    x = [str(item).split("_")[0] for item in data]
    vn = []
    for i, g in enumerate(data):
        if x[i] != pre:
            vn.append(pre + "_" + g)
        else:
            vn.append(g)
    return np.array(vn).astype("str").astype("object")


def _coarsen_blast_graph(
    gnnm: sp.sparse.csr_matrix,
    gns: NDArray[Any],
    names: dict[str, Any],
) -> tuple[sp.sparse.csr_matrix, dict[str, NDArray[Any]], NDArray[Any]]:
    """Coarsen BLAST graph by collapsing transcripts to genes."""
    gnnm = gnnm.tocsr()
    gnnm.eliminate_zeros()

    sps = np.array([x.split("_")[0] for x in gns])
    sids = np.unique(sps)
    ss = []
    for sid in sids:
        n = names.get(sid)
        if n is not None:
            n = np.array(n)
            n = (sid + "_" + n.astype("object")).astype("str")
            s1 = pd.Series(index=n[:, 0], data=n[:, 1])
            g = gns[sps == sid]
            g = g[np.isin(g, n[:, 0], invert=True)]
            s2 = pd.Series(index=g, data=g)
            s = pd.concat([s1, s2])
        else:
            s = pd.Series(index=gns[sps == sid], data=gns[sps == sid])
        ss.append(s)
    ss_combined = pd.concat(ss)
    ss_combined = ss_combined[np.unique(_q(ss_combined.index), return_index=True)[1]]
    x, y = gnnm.nonzero()
    s = pd.Series(data=gns, index=np.arange(gns.size))
    xn, yn = s[x].values, s[y].values
    xg, yg = ss_combined[xn].values, ss_combined[yn].values

    da = gnnm.data

    zgu, ix, _ivx, cu = np.unique(
        np.array([xg, yg]).astype("str"),
        axis=1,
        return_counts=True,
        return_index=True,
        return_inverse=True,
    )

    xgu, ygu = zgu[:, cu > 1]
    xgyg = _q(xg.astype("object") + ";" + yg.astype("object"))
    xguygu = _q(xgu.astype("object") + ";" + ygu.astype("object"))

    filt = np.isin(xgyg, xguygu)

    DF = pd.DataFrame(data=xgyg[filt][:, None], columns=["key"])
    DF["val"] = da[filt]

    dic = df_to_dict(DF, key_key="key")

    xgu = _q([x.split(";")[0] for x in dic])
    ygu = _q([x.split(";")[1] for x in dic])
    replz = _q([max(dic[x]) for x in dic])

    xgu1, ygu1 = zgu[:, cu == 1]
    xg = np.append(xgu1, xgu)
    yg = np.append(ygu1, ygu)
    da = np.append(da[ix][cu == 1], replz)
    gn = np.unique(np.append(xg, yg))

    s = pd.Series(data=np.arange(gn.size), index=gn)
    xn, yn = s[xg].values, s[yg].values
    gnnm = sp.sparse.coo_matrix((da, (xn, yn)), shape=(gn.size,) * 2).tocsr()

    f = np.asarray(gnnm.sum(1)).flatten() != 0
    gn = gn[f]
    sps = np.array([x.split("_")[0] for x in gn])

    gns_dict: dict[str, NDArray[Any]] = {}
    for sid in sids:
        gns_dict[sid] = gn[sps == sid]

    return gnnm, gns_dict, gn


def _filter_gnnm(gnnm: sp.sparse.csr_matrix, thr: float = 0.25) -> sp.sparse.csr_matrix:
    """Filter edges in homology graph below threshold."""
    x, y = gnnm.nonzero()
    mas = np.asarray(gnnm.max(1).todense()).flatten()
    gnnm4 = gnnm.copy()
    # Use np.asarray to handle both sparse matrix and numpy.matrix returns
    edge_values = np.asarray(gnnm4[x, y]).flatten()
    gnnm4.data[edge_values < mas[x] * thr] = 0
    gnnm4.eliminate_zeros()
    x, y = gnnm4.nonzero()
    z = gnnm4.data
    # Symmetrise: ensure (y, x) has the (x, y) value. Original entries first,
    # transpose second — last-write-wins matches the old LIL [y,x]=z behaviour.
    return coo_to_csr_overwrite(
        np.concatenate([x, y]),
        np.concatenate([y, x]),
        np.concatenate([z, z]),
        gnnm4.shape,
    )


def _get_pairs(
    sams: dict[str, SAM],
    gnnm: sp.sparse.csr_matrix,
    gns_dict: dict[str, NDArray[Any]],
    NOPs1: int = 0,
    NOPs2: int = 0,
) -> sp.sparse.csr_matrix:
    """Get gene pairs weighted by SAM weights."""
    su = np.asarray(gnnm.max(1).todense())
    su[su == 0] = 1
    gnnm = gnnm.multiply(1 / su).tocsr()
    Ws = {}
    for sid in sams:
        Ws[sid] = sams[sid].adata.var["weights"][gns_dict[sid]].values

    W = np.concatenate(list(Ws.values()))
    W[W < 0.0] = 0
    W[W > 0.0] = 1

    B = gnnm.multiply(W[None, :]).multiply(W[:, None]).tocsr()
    B.eliminate_zeros()

    return B
