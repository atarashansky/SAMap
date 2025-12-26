"""Mapping score functions for SAMap."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp
from samalg import SAM

from samap.utils import df_to_dict, substr, to_vn, to_vo

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray

    from samap.core.mapping import SAMAP


def _q(x: Any) -> NDArray[Any]:
    """Convert input to numpy array."""
    return np.array(list(x))


def _compute_csim(
    samap: SAM,
    key: str,
    X: sp.sparse.csr_matrix | None = None,
    prepend: bool = True,
    n_top: int = 0,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Compute cell similarity matrix."""
    splabels = _q(samap.adata.obs["species"])
    skeys = splabels[np.sort(np.unique(splabels, return_index=True)[1])]

    cl = []
    clu = []
    for sid in skeys:
        if prepend:
            cl.append(
                sid
                + "_"
                + _q(samap.adata.obs[key])[samap.adata.obs["species"] == sid]
                .astype("str")
                .astype("object")
            )
        else:
            cl.append(_q(samap.adata.obs[key])[samap.adata.obs["species"] == sid])
        clu.append(np.unique(cl[-1]))

    clu = np.concatenate(clu)
    cl = np.concatenate(cl)

    CSIM = np.zeros((clu.size, clu.size))
    if X is None:
        X = samap.adata.obsp["connectivities"].copy()

    xi, yi = X.nonzero()
    spxi = splabels[xi]
    spyi = splabels[yi]

    filt = spxi != spyi
    di = X.data[filt]
    xi = xi[filt]
    yi = yi[filt]

    px, py = xi, cl[yi]
    p = px.astype("str").astype("object") + ";" + py.astype("object")

    A = pd.DataFrame(data=np.vstack((p, di)).T, columns=["x", "y"])
    valdict = df_to_dict(A, key_key="x", val_key="y")
    cell_scores = [valdict[k].sum() for k in valdict.keys()]
    ixer = pd.Series(data=np.arange(clu.size), index=clu)
    if len(valdict.keys()) > 0:
        xc, yc = substr(list(valdict.keys()), ";")
        xc = xc.astype("int")
        yc = ixer[yc].values
        cell_cluster_scores = sp.sparse.coo_matrix(
            (cell_scores, (xc, yc)), shape=(X.shape[0], clu.size)
        ).toarray()

        for i, c in enumerate(clu):
            if n_top > 0:
                CSIM[i, :] = np.sort(cell_cluster_scores[cl == c], axis=0)[-n_top:].mean(0)
            else:
                CSIM[i, :] = cell_cluster_scores[cl == c].mean(0)

        CSIM = np.stack((CSIM, CSIM.T), axis=2).max(2)
        CSIMth = CSIM / samap.adata.uns["mapping_K"]
        return CSIMth, clu
    else:
        return np.zeros((clu.size, clu.size)), clu


def get_mapping_scores(
    sm: SAMAP,
    keys: dict[str, str],
    n_top: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate mapping scores between cell types.

    Parameters
    ----------
    sm : SAMAP
        SAMAP object.
    keys : dict
        Annotation vector keys for species.
    n_top : int, optional
        Number of top cells for averaging. Default 0.

    Returns
    -------
    tuple
        (D, A) - table of highest scores and pairwise mapping table.
    """
    if len(list(keys.keys())) < len(list(sm.sams.keys())):
        samap = SAM(
            counts=sm.samap.adata[np.in1d(sm.samap.adata.obs["species"], list(keys.keys()))]
        )
    else:
        samap = sm.samap

    ix = np.unique(samap.adata.obs["species"], return_index=True)[1]
    skeys = _q(samap.adata.obs["species"])[np.sort(ix)]

    clusters = []
    for sid in skeys:
        clusters.append(_q([sid + "_" + str(x) for x in sm.sams[sid].adata.obs[keys[sid]]]))

    cl = np.concatenate(clusters)
    label = "{}_mapping_scores".format(";".join([keys[sid] for sid in skeys]))
    samap.adata.obs[label] = pd.Categorical(cl)

    CSIMth, clu = _compute_csim(samap, label, n_top=n_top, prepend=False)

    A = pd.DataFrame(data=CSIMth, index=clu, columns=clu)
    i = np.argsort(-A.values.max(0).flatten())
    H = []
    C = []
    for idx in range(A.shape[1]):
        x = A.iloc[:, i[idx]].sort_values(ascending=False)
        H.append(np.vstack((x.index, x.values)).T)
        C.append(A.columns[i[idx]])
        C.append(A.columns[i[idx]])
    H = np.hstack(H)
    D = pd.DataFrame(data=H, columns=[C, ["Cluster", "Alignment score"] * (H.shape[1] // 2)])
    return D, A


def ParalogSubstitutions(
    sm: SAMAP,
    ortholog_pairs: NDArray[Any],
    paralog_pairs: NDArray[Any] | None = None,
    psub_thr: float = 0.3,
) -> pd.DataFrame:
    """Identify paralog substitutions.

    Parameters
    ----------
    sm : SAMAP
        SAMAP object.
    ortholog_pairs : ndarray
        Nx2 array of ortholog pairs.
    paralog_pairs : ndarray, optional
        Nx2 array of paralog pairs.
    psub_thr : float, optional
        Correlation difference threshold. Default 0.3.

    Returns
    -------
    pd.DataFrame
        Table of paralog substitutions.
    """
    if paralog_pairs is not None:
        ids1 = np.array([x.split("_")[0] for x in paralog_pairs[:, 0]])
        ids2 = np.array([x.split("_")[0] for x in paralog_pairs[:, 1]])
        ix = np.where(ids1 == ids2)[0]
        ixnot = np.where(ids1 != ids2)[0]

        if ix.size > 0:
            pps = paralog_pairs[ix]

            ZZ1: dict[str, list[str]] = {}
            ZZ2: dict[str, list[str]] = {}
            for i in range(pps.shape[0]):
                L = ZZ1.get(pps[i, 0], [])
                L.append(pps[i, 1])
                ZZ1[pps[i, 0]] = L

                L = ZZ2.get(pps[i, 1], [])
                L.append(pps[i, 0])
                ZZ2[pps[i, 1]] = L

            keys = list(ZZ1.keys())
            for k in keys:
                L = ZZ2.get(k, [])
                L.extend(ZZ1[k])
                ZZ2[k] = list(np.unique(L))

            ZZ = ZZ2

            L1 = []
            L2 = []
            for i in range(ortholog_pairs.shape[0]):
                x = ZZ.get(ortholog_pairs[i, 0], [])
                L1.extend([ortholog_pairs[i, 1]] * len(x))
                L2.extend(x)

                x = ZZ.get(ortholog_pairs[i, 1], [])
                L1.extend([ortholog_pairs[i, 0]] * len(x))
                L2.extend(x)

            L = np.vstack((L2, L1)).T
            pps = np.unique(np.sort(L, axis=1), axis=0)

            paralog_pairs = np.unique(np.sort(np.vstack((pps, paralog_pairs[ixnot])), axis=1), axis=0)

    smp = sm.samap

    gnnm = smp.adata.varp["homology_graph_reweighted"]
    gn = _q(smp.adata.var_names)

    ortholog_pairs = np.sort(ortholog_pairs, axis=1)

    ortholog_pairs = ortholog_pairs[
        np.logical_and(np.in1d(ortholog_pairs[:, 0], gn), np.in1d(ortholog_pairs[:, 1], gn))
    ]
    if paralog_pairs is None:
        paralog_pairs = gn[np.vstack(smp.adata.varp["homology_graph"].nonzero()).T]
    else:
        paralog_pairs = paralog_pairs[
            np.logical_and(np.in1d(paralog_pairs[:, 0], gn), np.in1d(paralog_pairs[:, 1], gn))
        ]

    paralog_pairs = np.sort(paralog_pairs, axis=1)

    paralog_pairs = paralog_pairs[
        np.in1d(
            to_vn(paralog_pairs),
            np.append(to_vn(ortholog_pairs), to_vn(ortholog_pairs[:, ::-1])),
            invert=True,
        )
    ]

    A = pd.DataFrame(data=np.arange(gn.size)[None, :], columns=gn)
    xp, yp = (
        A[paralog_pairs[:, 0]].values.flatten(),
        A[paralog_pairs[:, 1]].values.flatten(),
    )
    xp, yp = np.unique(np.vstack((np.vstack((xp, yp)).T, np.vstack((yp, xp)).T)), axis=0).T

    xo, yo = (
        A[ortholog_pairs[:, 0]].values.flatten(),
        A[ortholog_pairs[:, 1]].values.flatten(),
    )
    xo, yo = np.unique(np.vstack((np.vstack((xo, yo)).T, np.vstack((yo, xo)).T)), axis=0).T
    A = pd.DataFrame(data=np.vstack((xp, yp)).T, columns=["x", "y"])
    pairdict = df_to_dict(A, key_key="x", val_key="y")
    Xp = []
    Yp = []
    Xo = []
    Yo = []
    for i in range(xo.size):
        y = pairdict.get(xo[i], np.array([]))
        Yp.extend(y)
        Xp.extend([xo[i]] * y.size)
        Xo.extend([xo[i]] * y.size)
        Yo.extend([yo[i]] * y.size)

    orths = to_vn(gn[np.vstack((np.array(Xo), np.array(Yo))).T])
    paras = to_vn(gn[np.vstack((np.array(Xp), np.array(Yp))).T])
    orth_corrs = np.asarray(gnnm[Xo, Yo]).flatten()
    par_corrs = np.asarray(gnnm[Xp, Yp]).flatten()
    diff_corrs = par_corrs - orth_corrs

    RES = pd.DataFrame(
        data=np.vstack((orths, paras)).T, columns=["ortholog pairs", "paralog pairs"]
    )
    RES["ortholog corrs"] = orth_corrs
    RES["paralog corrs"] = par_corrs
    RES["corr diff"] = diff_corrs
    RES = RES.sort_values("corr diff", ascending=False)
    RES = RES[RES["corr diff"] > psub_thr]
    orths = RES["ortholog pairs"].values.flatten()
    paras = RES["paralog pairs"].values.flatten()
    if orths.size > 0:
        orthssp = np.vstack([np.array([x.split("_")[0] for x in xx]) for xx in to_vo(orths)])
        parassp = np.vstack([np.array([x.split("_")[0] for x in xx]) for xx in to_vo(paras)])
        filt = []
        for i in range(orthssp.shape[0]):
            filt.append(np.in1d(orthssp[i], parassp[i]).mean() == 1.0)
        filt = np.array(filt)
        return RES[filt]
    else:
        return RES


def convert_eggnog_to_homologs(
    sm: SAMAP,
    eggs: dict[str, pd.DataFrame],
    og_key: str = "eggNOG_OGs",
    taxon: int = 2759,
) -> NDArray[Any]:
    """Convert eggNOG results to homolog pairs.

    Parameters
    ----------
    sm : SAMAP
        SAMAP object.
    eggs : dict
        Dict of eggNOG DataFrames keyed by species.
    og_key : str, optional
        Column name for orthology groups. Default 'eggNOG_OGs'.
    taxon : int, optional
        Taxonomic level ID. Default 2759 (Eukaryotes).

    Returns
    -------
    ndarray
        Nx2 array of homolog pairs.
    """
    smp = sm.samap

    taxon_str = str(taxon)
    eggs_copy = dict(zip(list(eggs.keys()), list(eggs.values())))
    for k in eggs_copy.keys():
        eggs_copy[k] = eggs_copy[k].copy()

    Es = []
    for k in eggs_copy.keys():
        A = eggs_copy[k]
        A.index = k + "_" + A.index
        Es.append(A)

    A = pd.concat(Es, axis=0)
    gn = _q(smp.adata.var_names)
    A = A[np.in1d(_q(A.index), gn)]

    orthology_groups = A[og_key]
    og = _q(orthology_groups)
    x = np.unique(",".join(og).split(","))
    D = pd.DataFrame(data=np.arange(x.size)[None, :], columns=x)

    for i in range(og.size):
        n = orthology_groups[i].split(",")
        taxa = substr(substr(n, "@", 1), "|", 0)
        if (taxa == "2759").sum() > 1 and taxon_str == "2759":
            og[i] = ""
        else:
            og[i] = "".join(np.array(n)[taxa == taxon_str])

    A[og_key] = og

    og = _q(A[og_key].reindex(gn))
    og[og == "nan"] = ""

    X = []
    Y = []
    for i in range(og.size):
        x = og[i]
        if x != "":
            X.extend(D[x].values.flatten())
            Y.extend([i])

    X = np.array(X)
    Y = np.array(Y)
    B = sp.sparse.lil_matrix((og.size, D.size))
    B[Y, X] = 1
    B = B.tocsr()
    B = B.dot(B.T)
    B.data[:] = 1
    pairs = gn[np.vstack((B.nonzero())).T]
    pairssp = np.vstack([_q([x.split("_")[0] for x in xx]) for xx in pairs])
    return np.unique(np.sort(pairs[pairssp[:, 0] != pairssp[:, 1]], axis=1), axis=0)


def CellTypeTriangles(
    sm: SAMAP,
    keys: dict[str, str],
    align_thr: float = 0.1,
) -> pd.DataFrame:
    """Output table of cell type triangles.

    Parameters
    ----------
    sm : SAMAP
        SAMAP object with at least three species.
    keys : dict
        Annotation keys per species.
    align_thr : float, optional
        Minimum alignment score threshold. Default 0.1.

    Returns
    -------
    pd.DataFrame
        Table of cell type triangles.
    """
    D, A = get_mapping_scores(sm, keys=keys)
    x, y = A.values.nonzero()
    all_pairsf = np.array([A.index[x], A.columns[y]]).T.astype("str")
    alignmentf = A.values[x, y].flatten()

    alignment = alignmentf.copy()
    all_pairs = all_pairsf.copy()
    all_pairs = all_pairs[alignment > align_thr]
    alignment = alignment[alignment > align_thr]
    all_pairs = to_vn(np.sort(all_pairs, axis=1))

    x, y = substr(all_pairs, ";")
    ctu = np.unique(np.concatenate((x, y)))
    Z = pd.DataFrame(data=np.arange(ctu.size)[None, :], columns=ctu)
    nnm = sp.sparse.lil_matrix((ctu.size,) * 2)
    nnm[Z[x].values.flatten(), Z[y].values.flatten()] = alignment
    nnm[Z[y].values.flatten(), Z[x].values.flatten()] = alignment
    nnm = nnm.tocsr()

    G = nx.Graph()
    gps = ctu[np.vstack(nnm.nonzero()).T]
    G.add_edges_from(gps)
    alignment = pd.Series(index=to_vn(gps), data=nnm.data)
    all_triangles = [c for c in nx.enumerate_all_cliques(G) if len(c) == 3]
    Z = np.sort(np.vstack(all_triangles), axis=1)
    DF = pd.DataFrame(data=Z, columns=[x.split("_")[0] for x in Z[0]])
    for i, sid1 in enumerate(sm.ids):
        for sid2 in sm.ids[i:]:
            if sid1 != sid2:
                DF[sid1 + ";" + sid2] = [
                    alignment[x]
                    for x in DF[sid1].values.astype("str").astype("object")
                    + ";"
                    + DF[sid2].values.astype("str").astype("object")
                ]
    DF = DF[sm.ids]
    return DF


def GeneTriangles(
    sm: SAMAP,
    orth: NDArray[Any],
    keys: dict[str, str] | None = None,
    compute_markers: bool = True,
    corr_thr: float = 0.3,
    psub_thr: float = 0.3,
    pval_thr: float = 1e-10,
) -> pd.DataFrame:
    """Output table of gene triangles.

    Parameters
    ----------
    sm : SAMAP
        SAMAP object with at least three species.
    orth : ndarray
        Nx2 ortholog pairs.
    keys : dict, optional
        Annotation keys per species.
    compute_markers : bool, optional
        Whether to compute differential expression. Default True.
    corr_thr : float, optional
        Minimum correlation threshold. Default 0.3.
    psub_thr : float, optional
        Paralog substitution threshold. Default 0.3.
    pval_thr : float, optional
        P-value threshold for differential expression. Default 1e-10.

    Returns
    -------
    pd.DataFrame
        Table of gene triangles.
    """
    import itertools

    from samap.analysis.gene_pairs import find_cluster_markers

    FINALS = []

    orth = np.sort(orth, axis=1)
    orthsp = np.vstack([_q([x.split("_")[0] for x in xx]) for xx in orth])

    RES = ParalogSubstitutions(sm, orth, psub_thr=psub_thr)
    if RES.shape[0] > 0:
        op = to_vo(_q(RES["ortholog pairs"]))
        pp = to_vo(_q(RES["paralog pairs"]))
        ops = np.vstack([_q([x.split("_")[0] for x in xx]) for xx in op])
        pps = np.vstack([_q([x.split("_")[0] for x in xx]) for xx in pp])
        doPsubsAll = True
    else:
        doPsubsAll = False
    gnnm = sm.samap.adata.varp["homology_graph_reweighted"]
    gn = _q(sm.samap.adata.var_names)
    gnsp = _q([x.split("_")[0] for x in gn])

    combs = list(itertools.combinations(sm.ids, 3))
    for comb in combs:
        A_sid, B_sid, C_sid = comb
        sam1 = sm.sams[A_sid]
        sam2 = sm.sams[B_sid]
        sam3 = sm.sams[C_sid]

        f1 = (
            (orthsp[:, 0] == A_sid) * (orthsp[:, 1] == B_sid)
            + (orthsp[:, 0] == B_sid) * (orthsp[:, 1] == A_sid)
        ) > 0
        f2 = (
            (orthsp[:, 0] == A_sid) * (orthsp[:, 1] == C_sid)
            + (orthsp[:, 0] == C_sid) * (orthsp[:, 1] == A_sid)
        ) > 0
        f3 = (
            (orthsp[:, 0] == B_sid) * (orthsp[:, 1] == C_sid)
            + (orthsp[:, 0] == C_sid) * (orthsp[:, 1] == B_sid)
        ) > 0
        orth1 = orth[f1]
        orth2 = orth[f2]
        orth3 = orth[f3]

        gnnm1 = sp.sparse.vstack(
            (
                sp.sparse.hstack(
                    (
                        sp.sparse.csr_matrix(((gnsp == A_sid).sum(),) * 2),
                        gnnm[gnsp == A_sid, :][:, gnsp == B_sid],
                    )
                ),
                sp.sparse.hstack(
                    (
                        gnnm[gnsp == B_sid, :][:, gnsp == A_sid],
                        sp.sparse.csr_matrix(((gnsp == B_sid).sum(),) * 2),
                    )
                ),
            )
        ).tocsr()
        gnnm2 = sp.sparse.vstack(
            (
                sp.sparse.hstack(
                    (
                        sp.sparse.csr_matrix(((gnsp == A_sid).sum(),) * 2),
                        gnnm[gnsp == A_sid, :][:, gnsp == C_sid],
                    )
                ),
                sp.sparse.hstack(
                    (
                        gnnm[gnsp == C_sid, :][:, gnsp == A_sid],
                        sp.sparse.csr_matrix(((gnsp == C_sid).sum(),) * 2),
                    )
                ),
            )
        ).tocsr()
        gnnm3 = sp.sparse.vstack(
            (
                sp.sparse.hstack(
                    (
                        sp.sparse.csr_matrix(((gnsp == B_sid).sum(),) * 2),
                        gnnm[gnsp == B_sid, :][:, gnsp == C_sid],
                    )
                ),
                sp.sparse.hstack(
                    (
                        gnnm[gnsp == C_sid, :][:, gnsp == B_sid],
                        sp.sparse.csr_matrix(((gnsp == C_sid).sum(),) * 2),
                    )
                ),
            )
        ).tocsr()
        gn1 = np.append(gn[gnsp == A_sid], gn[gnsp == B_sid])
        gn2 = np.append(gn[gnsp == A_sid], gn[gnsp == C_sid])
        gn3 = np.append(gn[gnsp == B_sid], gn[gnsp == C_sid])

        if doPsubsAll:
            f1 = np.logical_and(
                ((ops[:, 0] == A_sid) * (ops[:, 1] == B_sid) + (ops[:, 0] == B_sid) * (ops[:, 1] == A_sid)) > 0,
                ((pps[:, 0] == A_sid) * (pps[:, 1] == B_sid) + (pps[:, 0] == B_sid) * (pps[:, 1] == A_sid)) > 0,
            )
            f2 = np.logical_and(
                ((ops[:, 0] == A_sid) * (ops[:, 1] == C_sid) + (ops[:, 0] == C_sid) * (ops[:, 1] == A_sid)) > 0,
                ((pps[:, 0] == A_sid) * (pps[:, 1] == C_sid) + (pps[:, 0] == C_sid) * (pps[:, 1] == A_sid)) > 0,
            )
            f3 = np.logical_and(
                ((ops[:, 0] == B_sid) * (ops[:, 1] == C_sid) + (ops[:, 0] == C_sid) * (ops[:, 1] == B_sid)) > 0,
                ((pps[:, 0] == B_sid) * (pps[:, 1] == C_sid) + (pps[:, 0] == C_sid) * (pps[:, 1] == B_sid)) > 0,
            )
            _ = f1.sum() > 0 and f2.sum() > 0 and f3.sum() > 0  # Not used in simplified version
        else:
            pass

        pairs1 = gn1[np.vstack(gnnm1.nonzero()).T]
        pairs2 = gn2[np.vstack(gnnm2.nonzero()).T]
        pairs3 = gn3[np.vstack(gnnm3.nonzero()).T]
        data = np.concatenate((gnnm1.data, gnnm2.data, gnnm3.data))

        CORR1 = pd.DataFrame(data=gnnm1.data[None, :], columns=to_vn(pairs1))
        CORR2 = pd.DataFrame(data=gnnm2.data[None, :], columns=to_vn(pairs2))
        CORR3 = pd.DataFrame(data=gnnm3.data[None, :], columns=to_vn(pairs3))

        pairs = np.vstack((pairs1, pairs2, pairs3))
        all_genes = np.unique(pairs.flatten())
        Z = pd.DataFrame(data=np.arange(all_genes.size)[None, :], columns=all_genes)
        x, y = Z[pairs[:, 0]].values.flatten(), Z[pairs[:, 1]].values.flatten()
        GNNM = sp.sparse.lil_matrix((all_genes.size,) * 2)
        GNNM[x, y] = data
        GNNM = GNNM.tocsr()
        GNNM.data[GNNM.data < corr_thr] = 0
        GNNM.eliminate_zeros()

        G = nx.from_scipy_sparse_array(GNNM, create_using=nx.Graph)
        all_triangles = [c for c in nx.enumerate_all_cliques(G) if len(c) == 3]
        if len(all_triangles) == 0:
            continue
        Z = all_genes[np.sort(np.vstack(all_triangles), axis=1)]

        DF = pd.DataFrame(data=Z, columns=[x.split("_")[0] for x in Z[0]])
        DF = DF[[A_sid, B_sid, C_sid]]

        orth1DF = pd.DataFrame(data=orth1, columns=[x.split("_")[0] for x in orth1[0]])[[A_sid, B_sid]]
        orth2DF = pd.DataFrame(data=orth2, columns=[x.split("_")[0] for x in orth2[0]])[[A_sid, C_sid]]
        orth3DF = pd.DataFrame(data=orth3, columns=[x.split("_")[0] for x in orth3[0]])[[B_sid, C_sid]]

        AB = to_vn(DF[[A_sid, B_sid]].values)
        AC = to_vn(DF[[A_sid, C_sid]].values)
        BC = to_vn(DF[[B_sid, C_sid]].values)

        AVs = []
        CATs = []
        CORRs = []
        for _, pairs_x, orth_df, corr_r in zip(
            [0, 1, 2], [AB, AC, BC], [orth1DF, orth2DF, orth3DF], [CORR1, CORR2, CORR3]
        ):
            cat = _q(["homolog"] * pairs_x.size).astype("object")
            cat[np.in1d(pairs_x, to_vn(orth_df.values))] = "ortholog"
            AV = np.zeros(pairs_x.size, dtype="object")
            corr = corr_r[pairs_x].values.flatten()
            AVs.append(AV)
            CATs.append(cat)
            CORRs.append(corr)

        tri_pairs = np.vstack((AB, AC, BC)).T
        cat_pairs = np.vstack(CATs).T
        corr_pairs = np.vstack(CORRs).T
        homology_triangles = DF.values
        substituted_genes = np.vstack(AVs).T
        substituted_genes[substituted_genes == 0] = "N.S."
        data = np.hstack(
            (
                homology_triangles.astype("object"),
                substituted_genes.astype("object"),
                tri_pairs.astype("object"),
                corr_pairs.astype("object"),
                cat_pairs.astype("object"),
            )
        )

        FINAL = pd.DataFrame(
            data=data,
            columns=[
                f"{A_sid} gene",
                f"{B_sid} gene",
                f"{C_sid} gene",
                f"{A_sid}/{B_sid} subbed",
                f"{A_sid}/{C_sid} subbed",
                f"{B_sid}/{C_sid} subbed",
                f"{A_sid}/{B_sid}",
                f"{A_sid}/{C_sid}",
                f"{B_sid}/{C_sid}",
                f"{A_sid}/{B_sid} corr",
                f"{A_sid}/{C_sid} corr",
                f"{B_sid}/{C_sid} corr",
                f"{A_sid}/{B_sid} type",
                f"{A_sid}/{C_sid} type",
                f"{B_sid}/{C_sid} type",
            ],
        )
        FINAL["#orthologs"] = (cat_pairs == "ortholog").sum(1)
        FINAL["#substitutions"] = (cat_pairs == "substitution").sum(1)
        FINAL = FINAL[(FINAL["#orthologs"] + FINAL["#substitutions"]) == 3]
        x = FINAL[[f"{A_sid}/{B_sid} corr", f"{A_sid}/{C_sid} corr", f"{B_sid}/{C_sid} corr"]].min(1)
        FINAL["min_corr"] = x
        FINAL = FINAL[x > corr_thr]
        if keys is not None:
            keys_local = [keys[A_sid], keys[B_sid], keys[C_sid]]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for i, sam, n in zip([0, 1, 2], [sam1, sam2, sam3], [A_sid, B_sid, C_sid]):
                    if compute_markers:
                        find_cluster_markers(sam, keys_local[i])
                    a = sam.adata.varm[keys_local[i] + "_scores"].T[_q(FINAL[n + " gene"])].T
                    p = sam.adata.varm[keys_local[i] + "_pvals"].T[_q(FINAL[n + " gene"])].T.values
                    p[p > pval_thr] = 1
                    p[p < 1] = 0
                    p = 1 - p
                    f = a.columns[a.values.argmax(1)]
                    res = []
                    for j in range(p.shape[0]):
                        res.append(";".join(np.unique(np.append(f[j], a.columns[p[j, :] == 1]))))
                    FINAL[n + " cell type"] = res
        FINAL = FINAL.sort_values("min_corr", ascending=False)
        FINALS.append(FINAL)
    return pd.concat(FINALS, axis=0)
