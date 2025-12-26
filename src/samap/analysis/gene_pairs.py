"""Gene pair finding functions for SAMap."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scanpy as sc
import sklearn.utils.sparsefuncs as sf

from samap._logging import logger
from samap.utils import substr, to_vn

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray
    from samalg import SAM

    from samap.core.mapping import SAMAP


def _q(x: Any) -> NDArray[Any]:
    """Convert input to numpy array."""
    return np.array(list(x))


class GenePairFinder:
    """Find enriched gene pairs in cell type mappings.

    Parameters
    ----------
    sm : SAMAP
        SAMAP object.
    keys : dict, optional
        Annotation keys per species. Default uses 'leiden_clusters'.
    """

    def __init__(self, sm: SAMAP, keys: dict[str, str] | None = None) -> None:
        if keys is None:
            keys = {sid: "leiden_clusters" for sid in sm.sams.keys()}
        self.sm = sm
        self.sams = sm.sams
        self.s3 = sm.samap
        self.gns = _q(sm.samap.adata.var_names)
        self.gnnm = sm.samap.adata.varp["homology_graph_reweighted"]
        self.gns_dict = sm.gns_dict

        self.ids = sm.ids

        mus = {}
        stds = {}
        for sid in self.sams.keys():
            self.sams[sid].adata.obs[keys[sid]] = self.sams[sid].adata.obs[keys[sid]].astype("str")
            mu, var = sf.mean_variance_axis(self.sams[sid].adata[:, self.gns_dict[sid]].X, axis=0)
            var[var == 0] = 1
            var = var**0.5
            mus[sid] = pd.Series(data=mu, index=self.gns_dict[sid])
            stds[sid] = pd.Series(data=var, index=self.gns_dict[sid])

        self.mus = mus
        self.stds = stds
        self.keys = keys
        self.find_markers()

    def find_markers(self) -> None:
        """Find cluster-specific markers for all species."""
        for sid in self.sams.keys():
            logger.info("Finding cluster-specific markers in %s:%s.", sid, self.keys[sid])
            import gc

            if self.keys[sid] + "_scores" not in self.sams[sid].adata.varm.keys():
                find_cluster_markers(self.sams[sid], self.keys[sid])
                gc.collect()

    def find_all(
        self,
        n: str | None = None,
        align_thr: float = 0.1,
        n_top: int = 0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Find enriched gene pairs in all mapped cell type pairs.

        Parameters
        ----------
        n : str, optional
            If provided, find pairs for cell types connected to `n`.
        align_thr : float, optional
            Alignment score threshold. Default 0.1.
        n_top : int, optional
            Number of top cells for averaging. Default 0.

        Returns
        -------
        pd.DataFrame
            Table of enriched gene pairs for each cell type pair.
        """
        from samap.analysis.scores import get_mapping_scores

        _, M = get_mapping_scores(self.sm, self.keys, n_top=n_top)
        ax = _q(M.index)
        data = M.values.copy()
        data[data < align_thr] = 0
        x, y = data.nonzero()
        ct1, ct2 = ax[x], ax[y]
        if n is not None:
            f1 = ct1 == n
            f2 = ct2 == n
            f = np.logical_or(f1, f2)
        else:
            f = np.array([True] * ct2.size)

        ct1 = ct1[f]
        ct2 = ct2[f]
        ct1, ct2 = np.unique(np.sort(np.vstack((ct1, ct2)).T, axis=1), axis=0).T
        res = {}
        for i in range(ct1.size):
            a = "_".join(ct1[i].split("_")[1:])
            b = "_".join(ct2[i].split("_")[1:])
            logger.info(
                "Calculating gene pairs for the mapping: %s;%s to %s;%s",
                ct1[i].split("_")[0],
                a,
                ct2[i].split("_")[0],
                b,
            )
            res[f"{ct1[i]};{ct2[i]}"] = self.find_genes(ct1[i], ct2[i], **kwargs)

        cols = []
        col_names = []
        for k in res:
            col_names.append(k)
            col_names.append(k + "_pval1")
            col_names.append(k + "_pval2")
            cols.append(res[k][0])
            cols.append(res[k][-2])
            cols.append(res[k][-1])
        return pd.DataFrame(cols, index=col_names).fillna(np.nan).T

    def find_genes(
        self,
        n1: str,
        n2: str,
        w1t: float = 0.2,
        w2t: float = 0.2,
        n_genes: int = 1000,
        thr: float = 1e-2,
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Find enriched gene pairs in a particular cell type pair.

        Parameters
        ----------
        n1 : str
            Cell type ID from species 1.
        n2 : str
            Cell type ID from species 2.
        w1t, w2t : float, optional
            SAM weight thresholds. Default 0.2.
        n_genes : int, optional
            Top ranked gene pairs to consider. Default 1000.
        thr : float, optional
            Differential expression p-value threshold. Default 0.01.

        Returns
        -------
        tuple
            (gene_pairs, genes1, genes2, pvals1, pvals2)
        """
        n1 = str(n1)
        n2 = str(n2)
        id1, id2 = n1.split("_")[0], n2.split("_")[0]
        sam1, sam2 = self.sams[id1], self.sams[id2]

        n1, n2 = "_".join(n1.split("_")[1:]), "_".join(n2.split("_")[1:])
        assert n1 in _q(self.sams[id1].adata.obs[self.keys[id1]])
        assert n2 in _q(self.sams[id2].adata.obs[self.keys[id2]])

        m, gpairs = self._find_link_genes_avg(n1, n2, id1, id2, w1t=w1t, w2t=w2t, expr_thr=0.05)

        self.gene_pair_scores = pd.Series(index=gpairs, data=m)

        G = _q(gpairs[np.argsort(-m)[:n_genes]])
        G1 = substr(G, ";", 0)
        G2 = substr(G, ";", 1)
        pvals1 = _q(sam1.adata.varm[self.keys[id1] + "_pvals"][n1][G1])
        pvals2 = _q(sam2.adata.varm[self.keys[id2] + "_pvals"][n2][G2])
        filt = np.logical_and(pvals1 < thr, pvals2 < thr)
        G = _q(G[filt])
        G1 = substr(G, ";", 0)
        G2 = substr(G, ";", 1)
        _, ix1 = np.unique(G1, return_index=True)
        _, ix2 = np.unique(G2, return_index=True)
        G1 = G1[np.sort(ix1)]
        G2 = G2[np.sort(ix2)]
        return G, G1, G2, pvals1[filt], pvals2[filt]

    def _find_link_genes_avg(
        self,
        c1: str,
        c2: str,
        id1: str,
        id2: str,
        w1t: float = 0.35,
        w2t: float = 0.35,
        expr_thr: float = 0.05,
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """Find linked genes by average expression."""
        mus = self.mus
        stds = self.stds
        sams = self.sams

        keys = self.keys
        sam3 = self.s3
        gnnm = self.gnnm
        gns = self.gns

        xs = []
        for sid in [id1, id2]:
            xs.append(sams[sid].get_labels(keys[sid]).astype("str").astype("object"))
        x1, x2 = xs
        g1, g2 = gns[np.vstack(gnnm.nonzero())]
        gs1, gs2 = _q([x.split("_")[0] for x in g1]), _q([x.split("_")[0] for x in g2])
        filt = np.logical_and(gs1 == id1, gs2 == id2)
        g1 = g1[filt]
        g2 = g2[filt]
        sam1, sam2 = sams[id1], sams[id2]
        mu1, std1, mu2, std2 = (
            mus[id1][g1].values,
            stds[id1][g1].values,
            mus[id2][g2].values,
            stds[id2][g2].values,
        )

        X1 = _sparse_sub_standardize(sam1.adata[:, g1].X[x1 == c1, :], mu1, std1)
        X2 = _sparse_sub_standardize(sam2.adata[:, g2].X[x2 == c2, :], mu2, std2)
        species_mask1 = (sam3.adata.obs["species"] == id1).values
        species_mask2 = (sam3.adata.obs["species"] == id2).values
        a, b = sam3.adata.obsp["connectivities"][species_mask1, :][
            :, species_mask2
        ][x1 == c1, :][:, x2 == c2].nonzero()
        c, d = sam3.adata.obsp["connectivities"][species_mask2, :][
            :, species_mask1
        ][x2 == c2, :][:, x1 == c1].nonzero()

        pairs = np.unique(np.vstack((np.vstack((a, b)).T, np.vstack((d, c)).T)), axis=0)

        av1 = np.asarray(X1[np.unique(pairs[:, 0]), :].mean(0)).flatten()
        av2 = np.asarray(X2[np.unique(pairs[:, 1]), :].mean(0)).flatten()
        sav1 = (av1 - av1.mean()) / av1.std()
        sav2 = (av2 - av2.mean()) / av2.std()
        sav1[sav1 < 0] = 0
        sav2[sav2 < 0] = 0
        val = sav1 * sav2 / sav1.size
        X1.data[:] = 1
        X2.data[:] = 1
        min_expr = (np.asarray(X1.mean(0)).flatten() > expr_thr) * (np.asarray(X2.mean(0)).flatten() > expr_thr)

        w1 = sam1.adata.var["weights"][g1].values.copy()
        w2 = sam2.adata.var["weights"][g2].values.copy()
        w1[w1 < 0.2] = 0
        w2[w2 < 0.2] = 0
        w1[w1 > 0] = 1
        w2[w2 > 0] = 1
        return val * w1 * w2 * min_expr, to_vn(np.array([g1, g2]).T)


def find_cluster_markers(sam: SAM, key: str, inplace: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Find differentially expressed genes for cell type labels.

    Parameters
    ----------
    sam : SAM
        SAM object.
    key : str
        Column in `sam.adata.obs` for differential expression.
    inplace : bool, optional
        If True, deposit results in sam.adata.varm. Default True.

    Returns
    -------
    tuple or None
        If not inplace, returns (NAMES, PVALS, SCORES) DataFrames.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        a, c = np.unique(_q(sam.adata.obs[key]), return_counts=True)
        t = a[c == 1]

        adata = sam.adata[np.in1d(_q(sam.adata.obs[key]), a[c == 1], invert=True)].copy()
        sc.tl.rank_genes_groups(
            adata,
            key,
            method="wilcoxon",
            n_genes=sam.adata.shape[1],
            use_raw=False,
            layer=None,
        )

        sam.adata.uns["rank_genes_groups"] = adata.uns["rank_genes_groups"]

        NAMES = pd.DataFrame(sam.adata.uns["rank_genes_groups"]["names"])
        PVALS = pd.DataFrame(sam.adata.uns["rank_genes_groups"]["pvals"])
        SCORES = pd.DataFrame(sam.adata.uns["rank_genes_groups"]["scores"])
        if not inplace:
            return NAMES, PVALS, SCORES
        dfs1 = []
        dfs2 = []
        for i in range(SCORES.shape[1]):
            names = NAMES.iloc[:, i]
            scores = SCORES.iloc[:, i]
            pvals = PVALS.iloc[:, i]
            pvals[scores < 0] = 1.0
            scores[scores < 0] = 0
            pvals = _q(pvals)
            scores = _q(scores)

            dfs1.append(
                pd.DataFrame(data=scores[None, :], index=[SCORES.columns[i]], columns=names)[
                    sam.adata.var_names
                ].T
            )
            dfs2.append(
                pd.DataFrame(data=pvals[None, :], index=[SCORES.columns[i]], columns=names)[
                    sam.adata.var_names
                ].T
            )
        df1 = pd.concat(dfs1, axis=1)
        df2 = pd.concat(dfs2, axis=1)

        # Safer attribute assignment
        if not hasattr(sam.adata.varm, "dim_names") or sam.adata.varm.dim_names is None:
            sam.adata.varm.dim_names = sam.adata.var_names
        sam.adata.varm[key + "_scores"] = df1
        sam.adata.varm[key + "_pvals"] = df2

        for i in range(t.size):
            sam.adata.varm[key + "_scores"][t[i]] = 0
            sam.adata.varm[key + "_pvals"][t[i]] = 1

    return None


def _sparse_sub_standardize(
    X: Any,
    mu: NDArray[Any],
    var: NDArray[Any],
    rows: bool = False,
) -> Any:
    """Standardize sparse matrix by subtraction."""
    x, y = X.nonzero()
    if not rows:
        Xs = X.copy()
        Xs.data[:] = (X.data - mu[y]) / var[y]
    else:
        mu, var = sf.mean_variance_axis(X, axis=1)
        var = var**0.5
        var[var == 0] = 1
        Xs = X.copy()
        Xs.data[:] = (X.data - mu[x]) / var[x]
    Xs.data[Xs.data < 0] = 0
    Xs.eliminate_zeros()
    return Xs
