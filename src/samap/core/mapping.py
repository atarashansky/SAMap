"""Core SAMap mapping algorithm."""

from __future__ import annotations

import gc
import os
import time
import warnings
from typing import TYPE_CHECKING

import hnswlib
import numpy as np
import pandas as pd
import scanpy as sc
import scipy as sp
from numba import njit, prange
from numba.core.errors import NumbaPerformanceWarning, NumbaWarning
from samalg import SAM
from sklearn.preprocessing import StandardScaler

from samap._constants import (
    DEFAULT_CROSS_K,
    DEFAULT_EVAL_THRESHOLD,
    DEFAULT_FILTER_THRESHOLD,
    DEFAULT_K_NEIGHBORS,
    DEFAULT_LEIDEN_RESOLUTION,
    DEFAULT_MIN_EXPRESSION,
    DEFAULT_N_GENES,
    DEFAULT_N_PCS,
    DEFAULT_NEIGHBORHOOD_SIZE,
    DEFAULT_NUM_ITERATIONS,
    DEFAULT_THRESH_HIGH,
    DEFAULT_THRESH_LOW,
    UMAP_MAXITER_LARGE,
    UMAP_MAXITER_SMALL,
    UMAP_MIN_DIST,
    UMAP_SIZE_THRESHOLD,
)
from samap._logging import logger
from samap.utils import df_to_dict, prepend_var_prefix, sparse_knn, to_vn

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
warnings.filterwarnings("ignore", category=NumbaWarning)


def _q(x: Any) -> NDArray[Any]:
    """Convert input to numpy array."""
    return np.array(list(x))


class SAMAP:
    """Self-Assembling Manifold Mapping for cross-species single-cell analysis.

    Parameters
    ----------
    sams : dict
        Dictionary mapping species IDs to either:
        - Path to an unprocessed '.h5ad' AnnData object
        - A processed and already-run SAM object

    f_maps : str, optional
        Path to the `maps` directory output by `map_genes.sh`.
        By default 'maps/'.

    names : dict, optional
        If BLAST was run on a transcriptome with FASTA headers that don't match
        gene symbols, pass a dict mapping species ID to a list of tuples:
        (FASTA header name, Dataset gene symbol).

    keys : dict, optional
        Dictionary of obs keys indexed by species for determining maximum
        neighborhood size. Defaults to 'leiden_clusters' for all species.

    resolutions : dict, optional
        Dictionary of leiden clustering resolutions indexed by species.
        Ignored if `keys` is set.

    gnnm : tuple, optional
        Pre-computed homology graph as (sparse matrix, gene names, gene dict).

    save_processed : bool, optional
        If True, saves processed SAM objects to '.h5ad' files.

    eval_thr : float, optional
        E-value threshold for BLAST results filtering. Default 1e-6.

    Attributes
    ----------
    sams : dict
        Dictionary of SAM objects indexed by species ID.
    gnnm : scipy.sparse matrix
        Gene homology graph.
    gns : ndarray
        Gene names in the homology graph.
    gns_dict : dict
        Gene names per species.
    ids : list
        Species IDs.
    samap : SAM
        Combined SAM object after running.
    """

    def __init__(
        self,
        sams: dict[str, str | SAM],
        f_maps: str = "maps/",
        names: dict[str, Any] | None = None,
        keys: dict[str, str] | None = None,
        resolutions: dict[str, float] | None = None,
        gnnm: tuple[Any, NDArray[Any], dict[str, NDArray[Any]]] | None = None,
        save_processed: bool = True,
        eval_thr: float = DEFAULT_EVAL_THRESHOLD,
    ) -> None:
        for key, data in sams.items():
            if not (isinstance(data, str) or isinstance(data, SAM)):
                raise TypeError(f"Input data {key} must be either a path or a SAM object.")

        ids = list(sams.keys())

        if keys is None:
            keys = {sid: "leiden_clusters" for sid in ids}

        if resolutions is None:
            resolutions = {sid: DEFAULT_LEIDEN_RESOLUTION for sid in ids}

        for sid in ids:
            data = sams[sid]
            key = keys[sid]
            res = resolutions[sid]

            if isinstance(data, str):
                logger.info("Processing data %s from: %s", sid, data)
                sam = SAM()
                sam.load_data(data)
                sam.preprocess_data(
                    sum_norm="cell_median",
                    norm="log",
                    thresh_low=DEFAULT_THRESH_LOW,
                    thresh_high=DEFAULT_THRESH_HIGH,
                    min_expression=DEFAULT_MIN_EXPRESSION,
                )
                sam.run(
                    preprocessing="StandardScaler",
                    npcs=DEFAULT_N_PCS,
                    weight_PCs=False,
                    k=DEFAULT_K_NEIGHBORS,
                    n_genes=DEFAULT_N_GENES,
                    weight_mode="rms",
                )
            else:
                sam = data

            if key == "leiden_clusters":
                sam.leiden_clustering(res=res)

            if "PCs_SAMap" not in sam.adata.varm.keys():
                prepare_SAMap_loadings(sam)

            if save_processed and isinstance(data, str):
                sam.save_anndata(data.split(".h5ad")[0] + "_pr.h5ad")

            sams[sid] = sam

        if gnnm is None:
            gnnm_matrix, gns, gns_dict = _calculate_blast_graph(
                ids, f_maps=f_maps, reciprocate=True, eval_thr=eval_thr
            )
            if names is not None:
                gnnm_matrix, gns_dict, gns = _coarsen_blast_graph(gnnm_matrix, gns, names)

            gnnm_matrix = _filter_gnnm(gnnm_matrix, thr=DEFAULT_FILTER_THRESHOLD)
        else:
            gnnm_matrix, gns, gns_dict = gnnm

        gns_list = []
        ges_list = []
        for sid in ids:
            prepend_var_prefix(sams[sid], sid)
            ge = _q(sams[sid].adata.var_names)
            gn = gns_dict[sid]
            gns_list.append(gn[np.in1d(gn, ge)])
            ges_list.append(ge)

        f = np.in1d(gns, np.concatenate(gns_list))
        gns = gns[f]
        gnnm_matrix = gnnm_matrix[f][:, f]
        A = pd.DataFrame(data=np.arange(gns.size)[None, :], columns=gns)
        ges = np.concatenate(ges_list)
        ges = ges[np.in1d(ges, gns)]
        ix = A[ges].values.flatten()
        gnnm_matrix = gnnm_matrix[ix][:, ix]
        gns = ges

        gns_dict = {}
        for i, sid in enumerate(ids):
            gns_dict[sid] = ges[np.in1d(ges, gns_list[i])]
            logger.info(
                "%d '%s' gene symbols match between the datasets and the BLAST graph.",
                gns_dict[sid].size,
                sid,
            )

        for sid in sams:
            if not sp.sparse.issparse(sams[sid].adata.X):
                sams[sid].adata.X = sp.sparse.csr_matrix(sams[sid].adata.X)

        smap = _Samap_Iter(sams, gnnm_matrix, gns_dict, keys=keys)
        self.sams = sams
        self.gnnm = gnnm_matrix
        self.gns_dict = gns_dict
        self.gns = gns
        self.ids = ids
        self.smap = smap

    def run(
        self,
        n_iterations: int = DEFAULT_NUM_ITERATIONS,
        neighborhood_sizes: dict[str, int] | None = None,
        cross_species_k: int = DEFAULT_CROSS_K,
        n_gene_chunks: int = 1,
        umap: bool = True,
        ncpus: int | None = None,
        hom_edge_thr: float = 0,
        hom_edge_mode: str = "pearson",
        scale_edges_by_corr: bool = True,
        neigh_from_keys: dict[str, bool] | None = None,
        pairwise: bool = True,
        # Deprecated parameter aliases
        NUMITERS: int | None = None,
        NHS: dict[str, int] | None = None,
        crossK: int | None = None,
        N_GENE_CHUNKS: int | None = None,
    ) -> SAM:
        """Run the SAMap algorithm.

        Parameters
        ----------
        n_iterations : int, optional
            Number of SAMap iterations. Default 3.
        neighborhood_sizes : dict, optional
            Maximum neighborhood sizes per species. Default 3 for all.
        cross_species_k : int, optional
            Number of cross-species edges per cell. Default 20.
        n_gene_chunks : int, optional
            Number of chunks for gene correlation computation. Default 1.
        umap : bool, optional
            Whether to compute UMAP projection. Default True.
        ncpus : int, optional
            Number of CPUs for parallel computation. Default all available.
        hom_edge_thr : float, optional
            Minimum edge weight threshold in homology graph. Default 0.
        hom_edge_mode : str, optional
            Correlation mode: 'pearson'. Default 'pearson'.
        scale_edges_by_corr : bool, optional
            Whether to scale edges by expression correlation. Default True.
        neigh_from_keys : dict, optional
            Whether to use clustering for neighborhoods per species.
        pairwise : bool, optional
            If True, compute neighborhoods pairwise. Default True.

        Returns
        -------
        SAM
            Species-merged SAM object.
        """
        # Handle deprecated parameter names
        if NUMITERS is not None:
            warnings.warn(
                "NUMITERS is deprecated, use n_iterations instead",
                DeprecationWarning,
                stacklevel=2,
            )
            n_iterations = NUMITERS
        if NHS is not None:
            warnings.warn(
                "NHS is deprecated, use neighborhood_sizes instead",
                DeprecationWarning,
                stacklevel=2,
            )
            neighborhood_sizes = NHS
        if crossK is not None:
            warnings.warn(
                "crossK is deprecated, use cross_species_k instead",
                DeprecationWarning,
                stacklevel=2,
            )
            cross_species_k = crossK
        if N_GENE_CHUNKS is not None:
            warnings.warn(
                "N_GENE_CHUNKS is deprecated, use n_gene_chunks instead",
                DeprecationWarning,
                stacklevel=2,
            )
            n_gene_chunks = N_GENE_CHUNKS

        if ncpus is None:
            ncpus = os.cpu_count() or 1

        self.pairwise = pairwise

        ids = self.ids
        sams = self.sams
        gnnm = self.gnnm
        gns_dict = self.gns_dict
        gns = self.gns
        smap = self.smap

        if neighborhood_sizes is None:
            neighborhood_sizes = {sid: DEFAULT_NEIGHBORHOOD_SIZE for sid in ids}
        if neigh_from_keys is None:
            neigh_from_keys = {sid: False for sid in ids}

        start_time = time.time()

        smap.run(
            NUMITERS=n_iterations,
            NHS=neighborhood_sizes,
            K=cross_species_k,
            NCLUSTERS=n_gene_chunks,
            ncpus=ncpus,
            THR=hom_edge_thr,
            corr_mode=hom_edge_mode,
            scale_edges_by_corr=scale_edges_by_corr,
            neigh_from_keys=neigh_from_keys,
            pairwise=pairwise,
        )
        samap = smap.final_sam
        self.samap = samap
        self.ITER_DATA = smap.ITER_DATA

        if umap:
            logger.info("Running UMAP on the stitched manifolds.")
            maxiter = (
                UMAP_MAXITER_SMALL
                if self.samap.adata.shape[0] <= UMAP_SIZE_THRESHOLD
                else UMAP_MAXITER_LARGE
            )
            sc.tl.umap(self.samap.adata, min_dist=UMAP_MIN_DIST, init_pos="random", maxiter=maxiter)

        ix = pd.Series(data=np.arange(samap.adata.shape[1]), index=samap.adata.var_names)[gns].values
        rixer = pd.Series(index=np.arange(gns.size), data=ix)

        if smap.GNNMS_corr:
            hom_graph = smap.GNNMS_corr[-1]
            x, y = hom_graph.nonzero()
            d = hom_graph.data
            hom_graph = sp.sparse.coo_matrix(
                (d, (rixer[x].values, rixer[y].values)), shape=(samap.adata.shape[1],) * 2
            ).tocsr()
            samap.adata.varp["homology_graph_reweighted"] = hom_graph
            self.gnnm_refined = hom_graph

        x, y = gnnm.nonzero()
        d = gnnm.data
        gnnm = sp.sparse.coo_matrix(
            (d, (rixer[x].values, rixer[y].values)), shape=(samap.adata.shape[1],) * 2
        ).tocsr()
        samap.adata.varp["homology_graph"] = gnnm
        samap.adata.uns["homology_gene_names_dict"] = gns_dict

        self.gnnm = gnnm
        self.gns = _q(samap.adata.var_names)

        gns_dict = {}
        for sid in ids:
            gns_dict[sid] = self.gns[np.in1d(self.gns, _q(self.sams[sid].adata.var_names))]
        self.gns_dict = gns_dict

        if umap:
            for sid in ids:
                sams[sid].adata.obsm["X_umap_samap"] = self.samap.adata[
                    sams[sid].adata.obs_names
                ].obsm["X_umap"]

        self.run_time = time.time() - start_time
        logger.info("Elapsed time: %.2f minutes.", self.run_time / 60)
        return samap

    def run_umap(self) -> None:
        """Run UMAP on the stitched manifolds."""
        logger.info("Running UMAP on the stitched manifolds.")
        ids = self.ids
        sams = self.sams
        maxiter = (
            UMAP_MAXITER_SMALL
            if self.samap.adata.shape[0] <= UMAP_SIZE_THRESHOLD
            else UMAP_MAXITER_LARGE
        )
        sc.tl.umap(
            self.samap.adata, min_dist=UMAP_MIN_DIST, init_pos="random", maxiter=maxiter
        )
        for sid in ids:
            sams[sid].adata.obsm["X_umap_samap"] = self.samap.adata[
                sams[sid].adata.obs_names
            ].obsm["X_umap"]

    def query_gene_pairs(self, gene: str) -> dict[str, pd.Series]:
        """Get BLAST and correlation scores for all genes connected to query gene.

        Parameters
        ----------
        gene : str
            Query gene (preferably with species prefix, e.g., "hu_SOX2").

        Returns
        -------
        dict
            Dictionary with "blast" and "correlation" Series.
        """
        ids = self.ids
        qgene = None
        if gene in self.gns:
            qgene = gene
        else:
            for sid in ids:
                if sid + "_" + gene in self.gns:
                    qgene = sid + "_" + gene
                    break
        if qgene is None:
            raise ValueError(f"Query gene {gene} not found in dataset.")

        a = self.gnnm[self.gns == qgene]
        b = self.gnnm_refined[self.gns == qgene]

        i1 = self.gns[a.nonzero()[1]]
        i2 = self.gns[b.nonzero()[1]]
        d1 = a.data
        d2 = b.data
        return {"blast": pd.Series(index=i1, data=d1), "correlation": pd.Series(index=i2, data=d2)}

    def query_gene_pair(self, gene1: str, gene2: str) -> dict[str, float]:
        """Get BLAST and correlation score for a pair of genes.

        Parameters
        ----------
        gene1, gene2 : str
            Query genes (preferably with species prefixes).

        Returns
        -------
        dict
            Dictionary with "blast" and "correlation" scores.
        """
        ids = self.ids

        def find_gene(gene: str) -> str:
            if gene in self.gns:
                return gene
            for sid in ids:
                if sid + "_" + gene in self.gns:
                    return sid + "_" + gene
            raise ValueError(f"Query gene {gene} not found in dataset.")

        qgene1 = find_gene(gene1)
        qgene2 = find_gene(gene2)

        a = self.gnnm[self.gns == qgene1].toarray().flatten()[self.gns == qgene2][0]
        b = self.gnnm_refined[self.gns == qgene1].toarray().flatten()[self.gns == qgene2][0]
        return {"blast": a, "correlation": b}

    def scatter(
        self,
        axes: Any = None,
        colors: dict[str, str] | None = None,
        sizes: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Plot species on combined UMAP.

        Parameters
        ----------
        axes : matplotlib.axes.Axes, optional
            Axes to plot on.
        colors : dict, optional
            Colors per species.
        sizes : dict, optional
            Marker sizes per species.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if sizes is None:
            sizes = {sid: 3 for sid in self.ids}

        if colors is None:
            colors = {}
            for sid in self.ids:
                s = "".join(hex(np.random.randint(16))[-1].upper() for _ in range(6))
                colors[sid] = "#" + s

        for sid in self.ids:
            axes = self.sams[sid].scatter(
                projection="X_umap_samap",
                colorspec=colors[sid],
                axes=axes,
                s=sizes[sid],
                colorbar=False,
                **kwargs,
            )

        return axes

    def plot_expression_overlap(
        self,
        gs: dict[str, str],
        axes: Any = None,
        color0: str = "gray",
        colors: dict[str, str] | None = None,
        colorc: str = "#00ceb5",
        s0: int = 1,
        ss: dict[str, int] | None = None,
        sc: int = 10,
        thr: float = 0.1,
        **kwargs: Any,
    ) -> Any:
        """Display expression overlap of genes on the combined manifold.

        Parameters
        ----------
        gs : dict
            Dictionary of genes to display, keyed by species IDs.
            For example: {'hu': 'TOP2A', 'ms': 'Top2a'}
        axes : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        color0 : str, optional
            Color for cells not expressing genes. Default 'gray'.
        colors : dict, optional
            Colors per species. If None, randomly generated.
        colorc : str, optional
            Color for overlapping expression. Default '#00ceb5'.
        s0 : int, optional
            Marker size for non-expressing cells. Default 1.
        ss : dict, optional
            Marker sizes per species. Default 3 for all.
        sc : int, optional
            Marker size for overlap. Default 10.
        thr : float, optional
            Threshold for imputed expression. Default 0.1.
        **kwargs
            Additional arguments for scatter.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if len(list(gs.keys())) < len(list(self.sams.keys())):
            samap = SAM(
                counts=self.samap.adata[np.in1d(self.samap.adata.obs["species"], list(gs.keys()))]
            )
        else:
            samap = self.samap

        if ss is None:
            ss = {sid: 3 for sid in self.ids}

        if colors is None:
            colors = {}
            for sid in self.ids:
                s = "".join(hex(np.random.randint(16))[-1].upper() for _ in range(6))
                colors[sid] = "#" + s

        def hex_to_rgb(value: str) -> list[float]:
            value = value.lstrip("#")
            lv = len(value)
            rgb = [int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3)]
            return [x / 255 for x in rgb]

        nnm = samap.adata.obsp["connectivities"]
        su = np.asarray(nnm.sum(1)).flatten()[:, None]
        su[su == 0] = 1

        nnm = nnm.multiply(1 / su).tocsr()
        AS: dict[str, NDArray[Any]] = {}
        for sid in gs.keys():
            g = gs[sid]
            try:
                AS[sid] = self.sams[sid].adata[:, g].X.toarray().flatten()
            except KeyError:
                try:
                    AS[sid] = self.sams[sid].adata[:, sid + "_" + g].X.toarray().flatten()
                except KeyError:
                    raise KeyError(f"Gene not found in species {sid}") from None

        davgs: dict[str, NDArray[Any]] = {}
        for sid in gs.keys():
            d = np.zeros(samap.adata.shape[0])
            d[samap.adata.obs["species"] == sid] = AS[sid]
            davg = np.asarray(nnm.dot(d)).flatten()
            davg[davg < thr] = 0
            davgs[sid] = davg
        davg = np.vstack(list(davgs.values())).min(0)
        for sid in gs.keys():
            if davgs[sid].max() > 0:
                davgs[sid] = davgs[sid] / davgs[sid].max()
        if davg.max() > 0:
            davg = davg / davg.max()

        cs: dict[str, NDArray[Any]] = {}
        for sid in gs.keys():
            c = hex_to_rgb(colors[sid]) + [0.0]
            cs[sid] = np.vstack([c] * davg.size)
            cs[sid][:, -1] = davgs[sid]
        cc = hex_to_rgb(colorc) + [0.0]
        cc = np.vstack([cc] * davg.size)
        cc[:, -1] = davg

        ax = samap.scatter(projection="X_umap", colorspec=color0, axes=axes, s=s0)

        for sid in gs.keys():
            samap.scatter(
                projection="X_umap", c=cs[sid], axes=ax, s=ss[sid], colorbar=False, **kwargs
            )

        samap.scatter(projection="X_umap", c=cc, axes=ax, s=sc, colorbar=False, **kwargs)

        return ax

    def gui(self) -> Any:
        """Launch a SAMGUI instance containing the SAM objects."""
        if "SamapGui" not in self.__dict__:
            try:
                from samalg.gui import SAMGUI
            except ImportError:
                raise ImportError(
                    "Please install SAMGUI dependencies. See the README in the SAM github repository."
                ) from None

            sg = SAMGUI(
                sam=list(self.sams.values()),
                title=list(self.ids),
                default_proj="X_umap_samap",
            )
            self.SamapGui = sg
            return sg.SamPlot
        else:
            return self.SamapGui.SamPlot

    def refine_homology_graph(
        self,
        thr: float = 0,
        n_clusters: int = 1,
        ncpus: int | None = None,
        corr_mode: str = "pearson",
        wscale: bool = False,
    ) -> sp.sparse.csr_matrix:
        """Refine the homology graph using expression correlations.

        Parameters
        ----------
        thr : float, optional
            Threshold for edge weights. Default 0.
        n_clusters : int, optional
            Number of gene clusters for chunked computation. Default 1.
        ncpus : int, optional
            Number of CPUs. Default all available.
        corr_mode : str, optional
            Correlation mode: 'pearson'. Default 'pearson'.
        wscale : bool, optional
            Whether to scale by weights. Default False.

        Returns
        -------
        scipy.sparse.csr_matrix
            Refined homology graph.
        """
        if ncpus is None:
            ncpus = os.cpu_count() or 1

        gnnm = self.smap.refine_homology_graph(
            NCLUSTERS=n_clusters, ncpus=ncpus, THR=thr, corr_mode=corr_mode, wscale=wscale
        )
        samap = self.smap.samap
        gns_dict = self.smap.gns_dict
        gns = []
        for sid in _q(samap.adata.obs["species"])[
            np.sort(np.unique(samap.adata.obs["species"], return_index=True)[1])
        ]:
            gns.extend(gns_dict[sid])
        gns = _q(gns)
        ix = pd.Series(data=np.arange(samap.adata.shape[1]), index=samap.adata.var_names)[gns].values
        rixer = pd.Series(index=np.arange(gns.size), data=ix)
        x, y = gnnm.nonzero()
        d = gnnm.data
        gnnm = sp.sparse.coo_matrix(
            (d, (rixer[x].values, rixer[y].values)), shape=(samap.adata.shape[1],) * 2
        ).tocsr()
        return gnnm


class _Samap_Iter:
    """Internal iterator class for SAMap algorithm."""

    def __init__(
        self,
        sams: dict[str, SAM],
        gnnm: sp.sparse.csr_matrix,
        gns_dict: dict[str, NDArray[Any]],
        keys: dict[str, str] | None = None,
    ) -> None:
        self.sams = sams
        self.gnnm = gnnm
        self.gnnmu = gnnm
        self.gns_dict = gns_dict

        if keys is None:
            keys = {sid: "leiden_clusters" for sid in sams.keys()}

        self.keys = keys

        self.GNNMS_corr: list[Any] = []
        self.GNNMS_pruned: list[Any] = []
        self.GNNMS_nnm: list[Any] = []

        self.ITER_DATA = [
            self.GNNMS_nnm,
            self.GNNMS_corr,
            self.GNNMS_pruned,
        ]
        self.iter = 0

    def refine_homology_graph(
        self,
        NCLUSTERS: int = 1,
        ncpus: int | None = None,
        THR: float = 0,
        corr_mode: str = "pearson",
        wscale: bool = False,
    ) -> sp.sparse.csr_matrix:
        """Refine homology graph using correlations."""
        if ncpus is None:
            ncpus = os.cpu_count() or 1

        gnnmu = _refine_corr(
            self.sams,
            self.samap,
            self.gnnm,
            self.gns_dict,
            THR=THR,
            use_seq=False,
            T1=0,
            NCLUSTERS=NCLUSTERS,
            ncpus=ncpus,
            corr_mode=corr_mode,
            wscale=wscale,
        )
        return gnnmu

    def run(
        self,
        NUMITERS: int = 3,
        NHS: dict[str, int] | None = None,
        K: int = 20,
        corr_mode: str = "pearson",
        NCLUSTERS: int = 1,
        scale_edges_by_corr: bool = True,
        THR: float = 0,
        neigh_from_keys: dict[str, bool] | None = None,
        pairwise: bool = True,
        ncpus: int | None = None,
    ) -> None:
        """Run the SAMap iterations."""
        if ncpus is None:
            ncpus = os.cpu_count() or 1

        sams = self.sams
        gns_dict = self.gns_dict
        gnnmu = self.gnnmu
        keys = self.keys

        if NHS is None:
            NHS = {sid: 2 for sid in sams.keys()}
        if neigh_from_keys is None:
            neigh_from_keys = {sid: False for sid in sams}
        gns = np.concatenate(list(gns_dict.values()))

        if self.iter > 0:
            sam4 = self.samap

        for i in range(NUMITERS):
            if self.iter > 0 and i == 0:
                logger.info("Calculating gene-gene correlations in the homology graph...")
                gnnmu = self.refine_homology_graph(
                    ncpus=ncpus, NCLUSTERS=NCLUSTERS, THR=THR, corr_mode=corr_mode
                )

                self.GNNMS_corr.append(gnnmu)
                self.gnnmu = gnnmu

            gnnm2 = _get_pairs(sams, gnnmu, gns_dict, NOPs1=0, NOPs2=0)
            self.GNNMS_pruned.append(gnnm2)

            sam4 = _mapper(
                sams,
                gnnm2,
                gns,
                umap=False,
                K=K,
                NHS=NHS,
                coarsen=True,
                keys=keys,
                scale_edges_by_corr=scale_edges_by_corr,
                neigh_from_keys=neigh_from_keys,
                pairwise=pairwise,
            )
            sam4.adata.uns["mapping_K"] = K
            self.samap = sam4
            self.GNNMS_nnm.append(sam4.adata.obsp["connectivities"])

            logger.info("Iteration %d complete.", i + 1)
            logger.info("Alignment scores:\n%s", _avg_as(sam4))

            self.iter += 1
            if i < NUMITERS - 1:
                logger.info("Calculating gene-gene correlations in the homology graph...")
                self.samap = sam4
                gnnmu = self.refine_homology_graph(
                    ncpus=ncpus, NCLUSTERS=NCLUSTERS, THR=THR, corr_mode=corr_mode
                )

                self.GNNMS_corr.append(gnnmu)
                self.gnnmu = gnnmu

            gc.collect()

        self.final_sam = sam4


def _avg_as(s: SAM) -> pd.DataFrame:
    """Calculate average alignment scores between species."""
    x = _q(s.adata.obs["species"])
    xu = np.unique(x)
    a = np.zeros((xu.size, xu.size))
    for i in range(xu.size):
        for j in range(xu.size):
            if i != j:
                a[i, j] = (
                    np.asarray(
                        s.adata.obsp["connectivities"][x == xu[i], :][:, x == xu[j]]
                        .sum(1)
                    )
                    .flatten()
                    .mean()
                    / s.adata.uns["mapping_K"]
                )
    return pd.DataFrame(data=a, index=xu, columns=xu)


@njit(parallel=True)
def _replace(X: NDArray[Any], xi: NDArray[Any], yi: NDArray[Any]) -> NDArray[np.float64]:
    """Compute correlations for pairs in parallel."""
    data = np.zeros(xi.size)
    for i in prange(xi.size):
        x = X[xi[i]]
        y = X[yi[i]]
        data[i] = ((x - x.mean()) * (y - y.mean()) / x.std() / y.std()).sum() / x.size
    return data


def _generate_coclustering_matrix(cl: NDArray[Any]) -> sp.sparse.csr_matrix:
    """Generate a co-clustering indicator matrix."""
    import samalg.utilities as ut

    cl_arr = ut.convert_annotations(np.array(list(cl)))
    clu, cluc = np.unique(cl_arr, return_counts=True)
    v = np.zeros((cl_arr.size, clu.size))
    v[np.arange(v.shape[0]), cl_arr] = 1
    return sp.sparse.csr_matrix(v)


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


# Include remaining internal functions from original mapping.py
# These are simplified versions with proper type hints and error handling

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
                B[B.columns[0]] = _prepend_blast_prefix(B.iloc[:, 0].values.flatten(), id1)

                B.index = _prepend_blast_prefix(B.index, id2)
                A[A.columns[0]] = _prepend_blast_prefix(A.iloc[:, 0].values.flatten(), id2)

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

                A.index = pd.Index(gnind[A.index].values.flatten())
                B.index = pd.Index(gnind[B.index].values.flatten())
                A[A.columns[0]] = gnind[A.iloc[:, 0].values.flatten()].values.flatten()
                B[B.columns[0]] = gnind[B.iloc[:, 0].values.flatten()].values.flatten()

                Arows = np.vstack((A.index, A.iloc[:, 0], A.iloc[:, i3])).T
                Arows = Arows[A.iloc[:, i1].values.flatten() <= eval_thr, :]
                gnnm1 = sp.sparse.lil_matrix((gn.size,) * 2)
                gnnm1[Arows[:, 0].astype("int32"), Arows[:, 1].astype("int32")] = Arows[:, 2]

                Brows = np.vstack((B.index, B.iloc[:, 0], B.iloc[:, i3])).T
                Brows = Brows[B.iloc[:, i1].values.flatten() <= eval_thr, :]
                gnnm2 = sp.sparse.lil_matrix((gn.size,) * 2)
                gnnm2[Brows[:, 0].astype("int32"), Brows[:, 1].astype("int32")] = Brows[:, 2]

                gnnm = (gnnm1 + gnnm2).tocsr()
                gnnms = (gnnm + gnnm.T) / 2
                if reciprocate:
                    gnnm.data[:] = 1
                    gnnms = gnnms.multiply(gnnm).multiply(gnnm.T).tocsr()
                gnnm = gnnms

                f1 = np.where(np.in1d(gn, gn1))[0]
                f2 = np.where(np.in1d(gn, gn2))[0]
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
        n = names.get(sid, None)
        if n is not None:
            n = np.array(n)
            n = (sid + "_" + n.astype("object")).astype("str")
            s1 = pd.Series(index=n[:, 0], data=n[:, 1])
            g = gns[sps == sid]
            g = g[np.in1d(g, n[:, 0], invert=True)]
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

    zgu, ix, ivx, cu = np.unique(
        np.array([xg, yg]).astype("str"), axis=1, return_counts=True, return_index=True, return_inverse=True
    )

    xgu, ygu = zgu[:, cu > 1]
    xgyg = _q(xg.astype("object") + ";" + yg.astype("object"))
    xguygu = _q(xgu.astype("object") + ";" + ygu.astype("object"))

    filt = np.in1d(xgyg, xguygu)

    DF = pd.DataFrame(data=xgyg[filt][:, None], columns=["key"])
    DF["val"] = da[filt]

    dic = df_to_dict(DF, key_key="key")

    xgu = _q([x.split(";")[0] for x in dic.keys()])
    ygu = _q([x.split(";")[1] for x in dic.keys()])
    replz = _q([max(dic[x]) for x in dic.keys()])

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
    gnnm4 = gnnm4.tolil()
    gnnm4[y, x] = z
    return gnnm4.tocsr()


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
    for sid in sams.keys():
        Ws[sid] = sams[sid].adata.var["weights"][gns_dict[sid]].values

    W = np.concatenate(list(Ws.values()))
    W[W < 0.0] = 0
    W[W > 0.0] = 1

    B = gnnm.multiply(W[None, :]).multiply(W[:, None]).tocsr()
    B.eliminate_zeros()

    return B


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


def _tanh_scale(x: NDArray[Any], scale: float = 10, center: float = 0.5) -> NDArray[Any]:
    """Apply tanh scaling to values."""
    return center + (1 - center) * np.tanh(scale * (x - center))


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
    sam = list(sams.values())[0]
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
        for sid in gns_dict.keys():
            gn = gns_dict[sid]
            gns_dict_sub[sid] = gn[np.in1d(gn, gnsub)]

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
        indices_list.append(np.unique(np.sort(np.vstack((GNNMSUBS[i].nonzero())).T, axis=1), axis=0))
        pairs_list.append(GNSUBS[i][indices_list[-1]])

    GNS = pd.DataFrame(data=np.arange(gns.size)[None, :], columns=gns)
    gnnm3 = sp.sparse.lil_matrix(gnnm.shape)
    for i in range(len(indices_list)):
        x, y = GNS[pairs_list[i][:, 0]].values.flatten(), GNS[pairs_list[i][:, 1]].values.flatten()
        gnnm3[x, y] = np.asarray(GNNMSUBS[i][indices_list[i][:, 0], indices_list[i][:, 1]]).flatten()

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
    for sid in sams.keys():
        Ws.append(sams[sid].adata.var["weights"][gns_dict[sid]].values)
        ix += [sid] * gns_dict[sid].size
    ix = np.array(ix)
    w = np.concatenate(Ws)

    w[w > T1] = 1
    w[w < 1] = 0

    gnO = gn[w > 0]
    ix = ix[w > 0]
    gns_dictO = {}
    for sid in gns_dict.keys():
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

    for k in CORR.keys():
        CORR[k] = 0 if CORR[k] < THR else CORR[k]
        if wscale:
            id1, id2 = [x.split("_")[0] for x in k.split(";")]
            weight1 = sams[id1].adata.var["weights"][k.split(";")[0]]
            weight2 = sams[id2].adata.var["weights"][k.split(";")[1]]
            CORR[k] = np.sqrt(CORR[k] * np.sqrt(weight1 * weight2))

    CORR_arr = np.array([CORR[x] for x in to_vn(gn[pairs])])

    gnnm3 = sp.sparse.lil_matrix(gnnm.shape)

    if use_seq:
        gnnm3[pairs[:, 0], pairs[:, 1]] = CORR_arr * np.asarray(gnnm2[pairs[:, 0], pairs[:, 1]]).flatten()
        gnnm3[pairs[:, 1], pairs[:, 0]] = CORR_arr * np.asarray(gnnm2[pairs[:, 1], pairs[:, 0]]).flatten()
    else:
        gnnm3[pairs[:, 0], pairs[:, 1]] = CORR_arr
        gnnm3[pairs[:, 1], pairs[:, 0]] = CORR_arr

    gnnm3 = gnnm3.tocsr()
    gnnm3.eliminate_zeros()
    return gnnm3


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
        NHS = {sid: 3 for sid in sams.keys()}

    if neigh_from_keys is None:
        neigh_from_keys = {sid: False for sid in sams.keys()}

    if mdata is None:
        mdata = _mapping_window(sams, gnnm, gn, K=K, pairwise=pairwise)

    k1 = K

    if keys is None:
        keys = {sid: "leiden_clusters" for sid in sams.keys()}

    nnms_in: dict[str, Any] = {}
    nnms_in0: dict[str, Any] = {}
    flag = False
    species_indexer = []
    for sid in sams.keys():
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
    mdata["xsim"] = D

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
    for sid in sams.keys():
        species_list += [sid] * sams[sid].adata.shape[0]
    species_list = np.array(species_list)

    if not pairwise or len(sams.keys()) == 2:
        Dk = sparse_knn(D, k1).tocsr()
        denom = k1
    else:
        Dk = []
        for sid1 in sams.keys():
            row = []
            for sid2 in sams.keys():
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

    sr[sr == 0] = 1
    st = np.asarray(Dk.sum(0)).flatten()[None, :]
    st[st == 0] = 1
    proj = Dk.multiply(1 / sr).dot(Dk.multiply(1 / st)).tocsr()
    z = proj.copy()
    z.data[:] = 1
    idx = np.where(np.asarray(z.sum(1)).flatten() >= k1)[0]

    omp = nnm_internal0
    omp.data[:] = 1
    s = np.asarray(proj.max(1).todense())
    s[s == 0] = 1
    proj = proj.multiply(1 / s).tocsr()
    X, Y = omp.nonzero()
    X2 = X[np.in1d(X, idx)]
    Y2 = Y[np.in1d(X, idx)]

    omp = omp.tolil()
    omp[X2, Y2] = np.vstack((np.asarray(proj[X2, Y2]).flatten(), np.ones(X2.size) * 0.3)).max(0)

    omp = nnm_internal0.tocsr()
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
    for i in sams.keys():
        for k in sams[i].adata.obs.keys():
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
        for sid in sams.keys():
            gs[sid] = gns[np.in1d(gns, _q(sams[sid].adata.var_names))]
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
            for i, sid1 in enumerate(sams.keys()):
                xtr = []
                for j, sid2 in enumerate(sams.keys()):
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
        C = sp.linalg.block_diag(*[adatas[sid].varm["PCs_SAMap"] for sid in sams.keys()])
        M = np.vstack(mus).dot(C)
        ttt = time.time()
        it = 0
        PCAs = []
        for sid in sams.keys():
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
        for sid in sams.keys():
            adatas[sid] = sams[sid].adata
            Ws[sid] = adatas[sid].var["weights"].values
            ss[sid] = std.fit_transform(adatas[sid].X).multiply(Ws[sid][None, :]).tocsr()
            mus.append(np.asarray(ss[sid].mean(0)).flatten())
            species_indexer.append(np.arange(ss[sid].shape[0]))
        for i in range(1, len(species_indexer)):
            species_indexer[i] = species_indexer[i] + species_indexer[i - 1].max() + 1
        X = sp.sparse.vstack(list(ss.values()))
        C = np.hstack([adatas[sid].varm["PCs_SAMap"] for sid in sams.keys()])
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

        for j, sid2 in enumerate(sams.keys()):
            if i != j:
                ixr = species_indexer[j]
                reference = wpca[ixr]

                b = _united_proj(query, reference, k=k)

                su = b.sum(1).A
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


def _sparse_knn_ks(D: sp.sparse.coo_matrix, ks: NDArray[Any]) -> sp.sparse.coo_matrix:
    """Keep variable top-k values per row in sparse matrix."""
    D1 = D.tocoo()
    idr = np.argsort(D1.row)
    D1.row[:] = D1.row[idr]
    D1.col[:] = D1.col[idr]
    D1.data[:] = D1.data[idr]

    row, ind = np.unique(D1.row, return_index=True)
    ind = np.append(ind, D1.data.size)
    for i in range(ind.size - 1):
        idx = np.argsort(D1.data[ind[i] : ind[i + 1]])
        k = ks[row[i]]
        if idx.size > k:
            if k != 0:
                idx = idx[:-k]
            else:
                idx = idx
            D1.data[np.arange(ind[i], ind[i + 1])[idx]] = 0
    D1.eliminate_zeros()
    return D1


def _smart_expand(
    nnm: sp.sparse.csr_matrix, K: NDArray[Any], NH: int = 3
) -> sp.sparse.csr_matrix:
    """Expand neighborhoods progressively."""
    stage0 = nnm.copy()
    S = [stage0]
    running = stage0
    for i in range(1, NH + 1):
        stage = running.dot(stage0)
        running = stage
        stage = stage.tolil()
        for j in range(i):
            stage[S[j].nonzero()] = 0
        stage = stage.tocsr()
        S.append(stage)

    for i in range(len(S)):
        s = _sparse_knn_ks(S[i], K).tocsr()
        a, c = np.unique(s.nonzero()[0], return_counts=True)
        numnz = np.zeros(s.shape[0], dtype="int32")
        numnz[a] = c
        K = K - numnz
        K[K < 0] = 0
        S[i] = s
    res = S[0]
    for i in range(1, len(S)):
        res = res + S[i]
    return res
