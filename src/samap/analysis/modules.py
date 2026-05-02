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


def gene_modules(
    sm: SAMAP,
    *,
    resolution: float = 1.0,
    use_reweighted: bool = True,
    seed: int = 0,
) -> pd.Series:
    """Partition the cross-species homology graph into gene modules.

    Runs Leiden community detection on the (symmetric, undirected) homology
    graph. With ``use_reweighted=True`` (default) the partition is over the
    expression-correlation–weighted graph, so modules correspond to sets of
    homologous genes that are also *co-expressed* on the joint manifold.

    Parameters
    ----------
    sm : SAMAP
        Run SAMAP object.
    resolution : float, optional
        Leiden resolution. Higher → more, smaller modules. Default 1.0.
    use_reweighted : bool, optional
        Partition the reweighted (corr) graph rather than the raw BLAST
        graph. Default True.
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
    # restrict to nodes that participate
    deg = np.asarray((G != 0).sum(axis=1)).ravel()
    nodes = np.where(deg > 0)[0]
    sub = G[nodes][:, nodes].tocoo()
    upper = sub.row < sub.col
    edges = list(zip(sub.row[upper].tolist(), sub.col[upper].tolist()))
    weights = sub.data[upper].astype(float).tolist()

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

    var = _q(ad.var_names)
    out = pd.Series(-1, index=var, dtype=int, name="module")
    out.iloc[nodes] = membership
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
    ar = [r for r in MT.index if r.startswith(f"{a}_")]
    bc = [c for c in MT.columns if c.startswith(f"{b}_")]
    A = MT.loc[ar, bc]

    gpf = GenePairFinder(sm, keys=keys)
    gp = gpf.find_all(align_thr=align_thr)

    rows = []
    for col in gp.columns:
        if "_pval" in col or ";" not in col:
            continue
        ta_full, tb_full = col.split(";", 1)
        ta = ta_full[len(a) + 1 :]
        tb = tb_full[len(b) + 1 :]
        score = float(A.loc[f"{a}_{ta}", f"{b}_{tb}"])
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
