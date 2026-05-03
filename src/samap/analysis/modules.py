"""Module-factored mapping scores.

A high alignment score between two cell types can be driven by a *single*
co-expressed gene module (e.g., the actomyosin program pulling colloblasts
and cardiomyocytes together) or by *many independent* modules — and only the
latter is good evidence for cell-type homology in the Arendt CoRC sense.

This module clusters the cross-species homology graph into gene modules
(Leiden on the union of the BLAST graph and the reweighted-correlation
graph), then decomposes each cell-type alignment by which gene modules
contribute to it (via the per-mapping enriched gene-pair list from
``GenePairFinder``). The output adds three columns to the score table:
``n_modules`` (how many modules contribute ≥ ``min_module_frac``),
``top_module_frac`` (fraction of enriched-pair weight in the dominant
module), and ``module_entropy`` (Shannon entropy across module
contributions).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.sparse as sp

from samap.analysis.gene_pairs import GenePairFinder
from samap.analysis.scores import get_mapping_scores
from samap.utils import q as _q

if TYPE_CHECKING:
    from samap.core.mapping import SAMAP


def _snn_from_adjacency(H: sp.csr_matrix, k: int = 15, prune: float = 1 / 15) -> sp.csr_matrix:
    """Jaccard shared-nearest-neighbour graph from a (sparse) adjacency.

    For each node, take its top-``k`` neighbours by weight; two nodes get an
    SNN edge weighted by the Jaccard overlap of their top-k neighbour sets.
    Edges with Jaccard < ``prune`` are dropped. Returned graph is symmetric.
    """
    H = H.tocsr()
    n = H.shape[0]
    # top-k per row
    rows, cols = [], []
    for i in range(n):
        lo, hi = H.indptr[i], H.indptr[i + 1]
        if hi == lo:
            continue
        idx = H.indices[lo:hi]
        dat = H.data[lo:hi]
        if idx.size > k:
            top = np.argpartition(-dat, k - 1)[:k]
            idx = idx[top]
        rows.extend([i] * idx.size)
        cols.extend(idx.tolist())
    K = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))
    # shared-neighbour count
    shared = (K @ K.T).tocoo()
    deg = np.asarray(K.sum(axis=1)).ravel()
    r, c, s = shared.row, shared.col, shared.data
    keep = r != c
    r, c, s = r[keep], c[keep], s[keep]
    jac = s / (deg[r] + deg[c] - s)
    keep = jac >= prune
    S = sp.csr_matrix((jac[keep], (r[keep], c[keep])), shape=(n, n))
    return ((S + S.T) / 2).tocsr()


def _coexpression_edges(
    sm: SAMAP,
    nodes: np.ndarray,
    var: np.ndarray,
    sp_of: np.ndarray,
    k: int,
    min_corr: float,
) -> sp.csr_matrix:
    """Within-species gene-gene Pearson-correlation kNN edges.

    For each species, restrict to that species' genes among ``nodes``,
    kNN-average their expression over the species' own cell-cell graph
    (so correlation reflects co-regulation, not technical co-detection),
    compute the gene×gene Pearson matrix, and keep each gene's top-``k``
    partners with correlation ≥ ``min_corr``. Returned matrix is over the
    ``nodes`` index space (n_nodes × n_nodes), symmetric.
    """
    n = nodes.size
    rows, cols, vals = [], [], []
    full_of = var[nodes]
    bare_of = np.array([g.split("_", 1)[1] for g in full_of])
    for sid, sam in sm.sams.items():
        local = np.where(sp_of[nodes] == sid)[0]
        if local.size < 3:
            continue
        ad = sam.adata
        vn = pd.Index(ad.var_names)
        # Per-species SAM var_names may be either species-prefixed or bare —
        # try prefixed first (the SAMAP-processed case), fall back to bare.
        present = vn.get_indexer(full_of[local])
        if (present < 0).all():
            present = vn.get_indexer(bare_of[local])
        ok = present >= 0
        if ok.sum() < 3:
            continue
        loc_ok = local[ok]
        cols_ok = present[ok]
        X = ad.X[:, cols_ok]
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        # kNN-average over the cell graph (row-normalised connectivities)
        nnm = ad.obsp.get("connectivities")
        if nnm is None:
            nnm = ad.obsp.get("nnm")
        if nnm is not None:
            nnm = sp.csr_matrix(nnm, dtype=np.float32)
            rs = np.asarray(nnm.sum(axis=1)).ravel()
            rs[rs == 0] = 1.0
            D = sp.diags(1.0 / rs)
            X = D @ nnm @ X
        # Pearson on columns
        Xc = X - X.mean(axis=0, keepdims=True)
        sd = Xc.std(axis=0, ddof=0)
        sd[sd == 0] = 1.0
        Z = Xc / sd
        C = (Z.T @ Z) / Z.shape[0]
        np.fill_diagonal(C, -np.inf)
        m = C.shape[0]
        kk = min(k, m - 1)
        for i in range(m):
            top = np.argpartition(-C[i], kk - 1)[:kk]
            for j in top:
                cij = C[i, j]
                if cij >= min_corr:
                    rows.append(loc_ok[i])
                    cols.append(loc_ok[j])
                    vals.append(float(cij))
    if not rows:
        return sp.csr_matrix((n, n), dtype=np.float32)
    E = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return ((E + E.T) / 2).tocsr()


def gene_modules(
    sm: SAMAP,
    *,
    resolution: float = 1.0,
    use_reweighted: bool = True,
    snn: bool | int = False,
    with_coexpression: bool | int = False,
    coexpr_min_corr: float = 0.2,
    seed: int = 0,
) -> pd.Series:
    """Partition the cross-species homology graph into gene modules.

    Runs Leiden community detection on the (symmetric, undirected) homology
    graph. With ``use_reweighted=True`` (default) the partition is over the
    expression-correlation–weighted graph, so modules correspond to sets of
    homologous genes that are also *co-expressed* on the joint manifold.

    The raw homology graph is bipartite-per-species-pair (edges only ever go
    *between* species), so on its own it Leiden-fragments into orthogroup-
    sized components and the module decomposition degenerates to a gene-pair
    count. Two densification options bridge within-species genes so modules
    become *program*-sized — analogous to how the cell-level SAMap manifold
    unions cross-species kNN with within-species kNN:

    - ``snn``: shared-nearest-neighbour on the homology graph itself. Two
      genes (any species) get an edge weighted by the Jaccard overlap of
      their top-k homology partners. Cheap; sequence-driven.
    - ``with_coexpression``: per-species gene-gene Pearson-correlation kNN
      on kNN-averaged expression. Expression-driven; closest to a CoRC
      "shared regulation" definition of a module.

    Parameters
    ----------
    sm : SAMAP
        Run SAMAP object.
    resolution : float, optional
        Leiden resolution. Higher → more, smaller modules. Default 1.0.
    use_reweighted : bool, optional
        Partition the reweighted (corr) graph rather than the raw BLAST
        graph. Default True.
    snn : bool or int, optional
        If truthy, augment with a Jaccard-SNN graph over the homology
        adjacency. If an int, that's the per-node ``k`` for the SNN
        neighbour sets (default 15 when ``True``). Default False.
    with_coexpression : bool or int, optional
        If truthy, augment with within-species gene-gene correlation kNN
        edges. If an int, that's the per-gene ``k`` (default 10 when
        ``True``). Default False.
    coexpr_min_corr : float, optional
        Minimum Pearson correlation for a coexpression edge. Default 0.2.
    seed : int, optional
        Leiden RNG seed. Default 0.

    Returns
    -------
    pandas.Series
        Index = species-prefixed gene name, value = module id (int). Genes
        with no homology-graph edge are assigned ``-1``.
    """
    import igraph as ig
    import leidenalg

    ad = sm.samap.adata
    key = "homology_graph_reweighted" if use_reweighted else "homology_graph"
    G = ad.varp[key]
    if G is None:
        raise ValueError(f"adata.varp['{key}'] is missing — run sm.run() first.")
    G = sp.csr_matrix(G)
    var = _q(ad.var_names)
    sp_of = np.array([g.split("_", 1)[0] for g in var])
    # restrict to nodes that participate
    deg = np.asarray((G != 0).sum(axis=1)).ravel()
    nodes = np.where(deg > 0)[0]
    H = G[nodes][:, nodes].tocsr()

    # densification
    aug = H.copy()
    if snn:
        k_snn = 15 if snn is True else int(snn)
        aug = aug + _snn_from_adjacency(H, k=k_snn)
    if with_coexpression:
        k_co = 10 if with_coexpression is True else int(with_coexpression)
        aug = aug + _coexpression_edges(sm, nodes, var, sp_of, k=k_co, min_corr=coexpr_min_corr)
    aug = aug.tocoo()
    upper = aug.row < aug.col
    edges = list(zip(aug.row[upper].tolist(), aug.col[upper].tolist()))
    weights = aug.data[upper].astype(float).tolist()

    g = ig.Graph(n=len(nodes), edges=edges, directed=False)
    g.es["weight"] = weights
    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        weights="weight",
        seed=seed,
    )
    membership = np.asarray(part.membership)

    out = pd.Series(-1, index=var, dtype=int, name="module")
    out.iloc[nodes] = membership
    # stash size table for diagnostics
    sizes = pd.Series(membership).value_counts()
    out.attrs["n_modules"] = int(sizes.size)
    out.attrs["median_size"] = int(sizes.median())
    out.attrs["n_components"] = int(g.connected_components().__len__())
    return out


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def module_factored_scores(
    sm: SAMAP,
    keys: dict[str, str],
    *,
    modules: pd.Series | None = None,
    align_thr: float = 0.1,
    min_module_frac: float = 0.05,
    resolution: float = 1.0,
) -> pd.DataFrame:
    """Annotate each cell-type mapping with its gene-module support profile.

    For every cross-species cell-type pair with alignment score above
    ``align_thr``, runs ``GenePairFinder`` to get the enriched homologous
    gene pairs that drive it, assigns each pair to the module of its
    species-a gene (modules are cross-species so either side gives the same
    answer for genes connected in the homology graph), and reports how many
    independent modules support the mapping.

    Parameters
    ----------
    sm : SAMAP
        Run SAMAP object.
    keys : dict[str, str]
        Annotation key per species.
    modules : pandas.Series, optional
        Precomputed output of ``gene_modules``. If None, computed at
        ``resolution``.
    align_thr : float, optional
        Only decompose mappings with score above this. Default 0.1.
    min_module_frac : float, optional
        A module "supports" a mapping if it contributes at least this
        fraction of enriched-pair weight. Default 0.05.
    resolution : float, optional
        Passed to ``gene_modules`` when ``modules`` is None.

    Returns
    -------
    pandas.DataFrame
        Columns: ``a``, ``b``, ``type_a``, ``type_b``, ``score``,
        ``n_gene_pairs``, ``n_modules``, ``top_module_frac``,
        ``module_entropy``, ``top_module``. Sorted by score descending.
    """
    if len(keys) != 2:
        raise ValueError("module_factored_scores is defined for two-species comparisons.")
    a, b = list(keys.keys())

    if modules is None:
        modules = gene_modules(sm, resolution=resolution)

    _, MT = get_mapping_scores(sm, keys)

    gpf = GenePairFinder(sm, keys=keys)
    gp = gpf.find_all(align_thr=align_thr)

    rows = []
    for col in gp.columns:
        if "_pval" in col or ";" not in col:
            continue
        ta_full, tb_full = col.split(";", 1)
        # GenePairFinder.find_all lexically sorts each (ct1, ct2) pair by full
        # species-prefixed label, so the (ta_full, tb_full) species order need
        # not match (a, b). Detect from prefix and swap if needed.
        sp1 = ta_full.split("_", 1)[0]
        if sp1 == a:
            ta = ta_full.split("_", 1)[1]
            tb = tb_full.split("_", 1)[1]
        else:
            ta = tb_full.split("_", 1)[1]
            tb = ta_full.split("_", 1)[1]
        score = float(MT.loc[f"{a}_{ta}", f"{b}_{tb}"])
        pairs = gp[col].dropna()
        mods = []
        for cell in pairs:
            if ";" not in str(cell):
                continue
            ga, gb = str(cell).split(";", 1)
            m = int(modules.get(ga, modules.get(gb, -1)))
            if m >= 0:
                mods.append(m)
        if not mods:
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "type_a": ta,
                    "type_b": tb,
                    "score": score,
                    "n_gene_pairs": len(pairs),
                    "n_modules": 0,
                    "top_module_frac": np.nan,
                    "module_entropy": 0.0,
                    "top_module": -1,
                }
            )
            continue
        counts = pd.Series(mods).value_counts()
        frac = counts / counts.sum()
        rows.append(
            {
                "a": a,
                "b": b,
                "type_a": ta,
                "type_b": tb,
                "score": score,
                "n_gene_pairs": len(pairs),
                "n_modules": int((frac >= min_module_frac).sum()),
                "top_module_frac": float(frac.iloc[0]),
                "module_entropy": _entropy(frac.values),
                "top_module": int(frac.index[0]),
            }
        )
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
