"""Gene enrichment analysis functions for SAMap."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from samap._logging import logger
from samap.utils import substr

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray

    from samap.core.mapping import SAMAP


def _log_factorial(n: int) -> float:
    """Compute log factorial."""
    return np.log(np.arange(1, n + 1)).sum()


def _log_binomial(n: int, k: int) -> float:
    """Compute log binomial coefficient."""
    return _log_factorial(n) - (_log_factorial(k) + _log_factorial(n - k))


def GOEA(
    target_genes: NDArray[Any],
    gene_sets: dict[str, NDArray[Any]] | pd.DataFrame,
    df_key: str = "GO",
    goterms: NDArray[Any] | None = None,
    fdr_thresh: float = 0.25,
    p_thresh: float = 1e-3,
) -> pd.DataFrame:
    """Perform GO term Enrichment Analysis using the hypergeometric distribution.

    Parameters
    ----------
    target_genes : array-like
        List of target genes from which to find enriched GO terms.
    gene_sets : dict or pd.DataFrame
        Dictionary where keys are GO terms and values are gene lists.
        OR DataFrame with genes as index and GO terms as values.
    df_key : str, optional
        Column name for GO terms if gene_sets is a DataFrame. Default 'GO'.
    goterms : array-like, optional
        If provided, only test these GO terms.
    fdr_thresh : float, optional
        FDR q-value threshold. Default 0.25.
    p_thresh : float, optional
        P-value threshold. Default 1e-3.

    Returns
    -------
    pd.DataFrame
        Enriched GO terms with FDR q-values, p-values, and associated genes.
    """
    if isinstance(gene_sets, pd.DataFrame):
        logger.info("Converting DataFrame into dictionary")
        genes = np.array(list(gene_sets.index))
        agt = np.array(list(gene_sets[df_key].values))
        idx = np.argsort(agt)
        genes = genes[idx]
        agt = agt[idx]
        bounds = np.where(agt[:-1] != agt[1:])[0] + 1
        bounds = np.append(np.append(0, bounds), agt.size)
        bounds_left = bounds[:-1]
        bounds_right = bounds[1:]
        genes_lists = [genes[bounds_left[i] : bounds_right[i]] for i in range(bounds_left.size)]
        gene_sets = dict(zip(np.unique(agt), genes_lists))

    all_genes = np.unique(np.concatenate(list(gene_sets.values())))
    all_genes = np.array(all_genes)

    if goterms is None:
        goterms = np.unique(list(gene_sets.keys()))
    else:
        goterms = goterms[np.in1d(goterms, np.unique(list(gene_sets.keys())))]

    _, ix = np.unique(target_genes, return_index=True)
    target_genes = target_genes[np.sort(ix)]
    target_genes = target_genes[np.in1d(target_genes, all_genes)]

    N = all_genes.size

    probs = []
    probs_genes = []

    for goterm in goterms:
        gene_set = np.array(gene_sets[goterm])
        B = gene_set.size
        gene_set_in_target = gene_set[np.in1d(gene_set, target_genes)]
        b = gene_set_in_target.size
        if b != 0:
            n = target_genes.size
            num_iter = min(n, B)
            rng = np.arange(b, num_iter + 1)
            probs.append(
                sum(
                    [
                        np.exp(_log_binomial(n, i) + _log_binomial(N - n, B - i) - _log_binomial(N, B))
                        for i in rng
                    ]
                )
            )
        else:
            probs.append(1.0)
        probs_genes.append(gene_set_in_target)

    probs = np.array(probs)
    probs_genes = np.array([";".join(x) for x in probs_genes])

    fdr_q_probs = probs.size * probs / rankdata(probs, method="ordinal")

    filt = np.logical_and(fdr_q_probs < fdr_thresh, probs < p_thresh)
    enriched_goterms = goterms[filt]
    p_values = probs[filt]
    fdr_q_probs = fdr_q_probs[filt]
    probs_genes = probs_genes[filt]

    gns = probs_genes
    result = pd.DataFrame(data=fdr_q_probs, index=enriched_goterms, columns=["fdr_q_value"])
    result["p_value"] = p_values
    result["genes"] = gns

    return result.sort_values("p_value")


class FunctionalEnrichment:
    """Functional enrichment analysis on gene pairs in mapped cell types.

    Parameters
    ----------
    sm : SAMAP
        SAMAP object.
    dfs : dict
        Dict of pandas DataFrames with functional annotations, keyed by species.
    col_key : str
        Column name with functional annotations.
    keys : dict
        Dict of obs column keys keyed by species.
    delimiter : str, optional
        Delimiter for multiple annotations. Default ''.
    align_thr : float, optional
        Alignment score threshold. Default 0.1.
    limit_reference : bool, optional
        If True, limit background genes to enriched ones. Default False.
    n_top : int, optional
        Number of top cells for alignment averaging. Default 0.
    """

    def __init__(
        self,
        sm: SAMAP,
        dfs: dict[str, pd.DataFrame],
        col_key: str,
        keys: dict[str, str],
        delimiter: str = "",
        align_thr: float = 0.1,
        limit_reference: bool = False,
        n_top: int = 0,
    ) -> None:
        from samap.analysis.gene_pairs import GenePairFinder

        sams = sm.sams

        for sid in sm.ids:
            sm.sams[sid] = sams[sid]
            gc.collect()

        for k in dfs.keys():
            dfs[k].index = k + "_" + dfs[k].index

        A = pd.concat(list(dfs.values()), axis=0)
        RES = pd.DataFrame(A[col_key])
        RES.columns = ["GO"]
        RES = RES[(np.array(list(RES.values.flatten())).astype("str") != "nan")]

        data = []
        index = []
        for i in range(RES.shape[0]):
            if delimiter == "":
                items = list(RES.values[i][0])
                items = np.array([str(x) if str(x).isalpha() else "" for x in items])
                items = items[items != ""]
                items = list(items)
            else:
                items = RES.values[i][0].split(delimiter)

            data.extend(items)
            index.extend([RES.index[i]] * len(items))

        RES = pd.DataFrame(index=index, data=data, columns=["GO"])
        genes = np.array(list(RES.index))
        agt = np.array(list(RES["GO"].values))
        idx = np.argsort(agt)
        genes = genes[idx]
        agt = agt[idx]
        bounds = np.where(agt[:-1] != agt[1:])[0] + 1
        bounds = np.append(np.append(0, bounds), agt.size)
        bounds_left = bounds[:-1]
        bounds_right = bounds[1:]
        genes_lists = [genes[bounds_left[i] : bounds_right[i]] for i in range(bounds_left.size)]
        gene_sets = dict(zip(np.unique(agt), genes_lists))
        for cc in gene_sets.keys():
            gene_sets[cc] = np.unique(gene_sets[cc])

        logger.info("Finding enriched gene pairs...")
        gpf = GenePairFinder(sm, keys=keys)
        gene_pairs = gpf.find_all(thr=align_thr, n_top=n_top)

        self.DICT: dict[str, NDArray[Any]] = {}
        for c in gene_pairs.columns:
            if "_pval1" not in c and "_pval2" not in c:
                x = np.array(list(gene_pairs[c].values.flatten())).astype("str")
                ff = x != "nan"
                if ff.sum() > 0:
                    self.DICT[c] = x[ff]

        if limit_reference:
            all_genes = np.unique(np.concatenate(substr(np.concatenate(list(self.DICT.values())), ";")))
        else:
            all_genes = np.unique(np.array(list(A.index)))

        for d in gene_sets.keys():
            gene_sets[d] = gene_sets[d][np.in1d(gene_sets[d], all_genes)]

        self.gene_pairs = gene_pairs
        self.CAT_NAMES = np.unique(np.array(list(RES["GO"])))
        self.GENE_SETS = gene_sets
        self.RES = RES

    def calculate_enrichment(self, verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Calculate functional enrichment.

        Parameters
        ----------
        verbose : bool, optional
            If True, log progress. Default False.

        Returns
        -------
        tuple
            (enrichment_scores, num_enriched_genes, enriched_genes) DataFrames
        """
        DICT = self.DICT
        RES = self.RES
        CAT_NAMES = self.CAT_NAMES
        GENE_SETS = self.GENE_SETS
        pairs = np.array(list(DICT.keys()))
        all_nodes = np.unique(np.concatenate(substr(pairs, ";")))

        CCG: dict[str, NDArray[Any]] = {}
        for ik in range(len(all_nodes)):
            genes = []
            nodes = all_nodes[ik]
            for j in range(len(pairs)):
                n1, n2 = pairs[j].split(";")
                if n1 == nodes or n2 == nodes:
                    g1, g2 = substr(DICT[pairs[j]], ";")
                    genes.append(np.append(g1, g2))
            if len(genes) > 0:
                genes = np.concatenate(genes)
                genes = np.unique(genes)
            else:
                genes = np.array([])
            CCG[all_nodes[ik]] = genes

        HM = np.zeros((len(CAT_NAMES), len(all_nodes)))
        HMe = np.zeros((len(CAT_NAMES), len(all_nodes)))
        HMg = np.zeros((len(CAT_NAMES), len(all_nodes)), dtype="object")
        for ii, cln in enumerate(all_nodes):
            if verbose:
                logger.info("Calculating functional enrichment for cell type %s", cln)

            g = CCG[cln]

            if g.size > 0:
                gi = g[np.in1d(g, np.array(list(RES.index)))]
                ix = np.where(np.in1d(np.array(list(RES.index)), gi))[0]
                res = RES.iloc[ix]
                goterms = np.unique(np.array(list(res["GO"])))
                goterms = goterms[goterms != "S"].flatten()
                if goterms.size > 0:
                    result = GOEA(gi, GENE_SETS, goterms=goterms, fdr_thresh=100, p_thresh=100)

                    lens = np.array([len(np.unique(x.split(";"))) for x in result["genes"].values])
                    F = -np.log10(result["p_value"])
                    gt, vals = F.index, F.values
                    Z = pd.DataFrame(data=np.arange(CAT_NAMES.size)[None, :], columns=CAT_NAMES)
                    if gt.size > 0:
                        HM[Z[gt].values.flatten(), ii] = vals
                        HMe[Z[gt].values.flatten(), ii] = lens
                        HMg[Z[gt].values.flatten(), ii] = [
                            ";".join(np.unique(x.split(";"))) for x in result["genes"].values
                        ]

        SC = pd.DataFrame(data=HM, index=CAT_NAMES, columns=all_nodes).T
        SCe = pd.DataFrame(data=HMe, index=CAT_NAMES, columns=all_nodes).T
        SCg = pd.DataFrame(data=HMg, index=CAT_NAMES, columns=all_nodes).T
        SCg.values[SCg.values == 0] = ""

        self.ENRICHMENT_SCORES = SC
        self.NUM_ENRICHED_GENES = SCe
        self.ENRICHED_GENES = SCg

        return self.ENRICHMENT_SCORES, self.NUM_ENRICHED_GENES, self.ENRICHED_GENES

    def plot_enrichment(
        self,
        cell_types: list[str] | None = None,
        pval_thr: float = 2.0,
        msize: float = 50,
    ) -> tuple[Any, Any]:
        """Create enrichment dot plot.

        Parameters
        ----------
        cell_types : list, optional
            Cell types to include. Default all.
        pval_thr : float, optional
            -log10 p-value threshold. Default 2.0.
        msize : float, optional
            Marker size. Default 50.

        Returns
        -------
        tuple
            (fig, ax) matplotlib objects
        """
        import matplotlib
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram, linkage

        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42

        if cell_types is None:
            cell_types = []

        SC = self.ENRICHMENT_SCORES
        SCe = self.NUM_ENRICHED_GENES
        SCg = self.ENRICHED_GENES

        if len(cell_types) > 0:
            SC = SC.T[cell_types].T
            SCe = SCe.T[cell_types].T
            SCg = SCg.T[cell_types].T

        SC.values[SC.values < pval_thr] = 0
        SCe.values[SC.values < pval_thr] = 0
        SCg.values[SC.values < pval_thr] = ""
        SCg = SCg.astype("str")
        SCg.values[SCg.values == "nan"] = ""

        ixrow = np.array(
            dendrogram(linkage(SC.values.T, method="ward", metric="euclidean"), no_plot=True)["ivl"]
        ).astype("int")
        ixcol = np.array(
            dendrogram(linkage(SC.values, method="ward", metric="euclidean"), no_plot=True)["ivl"]
        ).astype("int")

        SC = SC.iloc[ixcol].iloc[:, ixrow]
        SCe = SCe.iloc[ixcol].iloc[:, ixrow]
        SCg = SCg.iloc[ixcol].iloc[:, ixrow]

        x, y = np.tile(np.arange(SC.shape[0]), SC.shape[1]), np.repeat(
            np.arange(SC.shape[1]), SC.shape[0]
        )
        co = SC.values[x, y].flatten()
        ms = SCe.values[x, y].flatten()
        ms = ms / ms.max()
        x = x.max() - x
        ms = ms * msize
        ms[np.logical_and(ms < 0.15, ms > 0)] = 0.15

        fig, ax = plt.subplots()
        fig.set_size_inches((7 * SC.shape[0] / SC.shape[1], 7))

        scat = ax.scatter(x, y, c=co, s=ms, cmap="seismic", edgecolor="k", linewidth=0.5, vmin=3)
        fig.colorbar(scat, pad=0.02)
        ax.set_yticks(np.arange(SC.shape[1]))
        ax.set_yticklabels(SC.columns, ha="right", rotation=0)
        ax.set_xticks(np.arange(SC.shape[0]))
        ax.set_xticklabels(SC.index[::-1], ha="right", rotation=45)
        ax.invert_yaxis()
        ax.invert_xaxis()
        return fig, ax
