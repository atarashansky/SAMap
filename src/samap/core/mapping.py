"""Core SAMap mapping algorithm."""

from __future__ import annotations

import gc
import os
import time
import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import scanpy as sc
import scipy as sp

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
from samap.sam import SAM
from samap.utils import prepend_var_prefix
from samap.utils import q as _q

from ._backend import Backend
from .coarsening import _mapper
from .correlation import _refine_corr
from .homology import (
    _calculate_blast_graph,
    _coarsen_blast_graph,
    _filter_gnnm,
    _get_pairs,
)
from .projection import _projection_precompute, prepare_SAMap_loadings

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray


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

    backend : {"auto", "cpu", "cuda"}, optional
        Compute backend. "auto" picks CUDA if a GPU is available, else CPU.

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
        backend: Literal["auto", "cpu", "cuda"] = "auto",
    ) -> None:
        self._bk = Backend(backend)
        logger.info("Using backend: %s", self._bk.device)

        for key, data in sams.items():
            if not (isinstance(data, str | SAM)):
                raise TypeError(f"Input data {key} must be either a path or a SAM object.")

        ids = list(sams.keys())

        if keys is None:
            keys = dict.fromkeys(ids, "leiden_clusters")

        if resolutions is None:
            resolutions = dict.fromkeys(ids, DEFAULT_LEIDEN_RESOLUTION)

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

            if "PCs_SAMap" not in sam.adata.varm:
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
            gns_list.append(gn[np.isin(gn, ge)])
            ges_list.append(ge)

        f = np.isin(gns, np.concatenate(gns_list))
        gns = gns[f]
        gnnm_matrix = gnnm_matrix[f][:, f]
        A = pd.DataFrame(data=np.arange(gns.size)[None, :], columns=gns)
        ges = np.concatenate(ges_list)
        ges = ges[np.isin(ges, gns)]
        ix = A[ges].values.flatten()
        gnnm_matrix = gnnm_matrix[ix][:, ix]
        gns = ges

        gns_dict = {}
        for i, sid in enumerate(ids):
            gns_dict[sid] = ges[np.isin(ges, gns_list[i])]
            logger.info(
                "%d '%s' gene symbols match between the datasets and the BLAST graph.",
                gns_dict[sid].size,
                sid,
            )

        for sid in sams:
            if not sp.sparse.issparse(sams[sid].adata.X):
                sams[sid].adata.X = sp.sparse.csr_matrix(sams[sid].adata.X)

        smap = _Samap_Iter(sams, gnnm_matrix, gns_dict, keys=keys, bk=self._bk)
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
            neighborhood_sizes = dict.fromkeys(ids, DEFAULT_NEIGHBORHOOD_SIZE)
        if neigh_from_keys is None:
            neigh_from_keys = dict.fromkeys(ids, False)

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

        ix = pd.Series(data=np.arange(samap.adata.shape[1]), index=samap.adata.var_names)[
            gns
        ].values
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
            gns_dict[sid] = self.gns[np.isin(self.gns, _q(self.sams[sid].adata.var_names))]
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
        sc.tl.umap(self.samap.adata, min_dist=UMAP_MIN_DIST, init_pos="random", maxiter=maxiter)
        for sid in ids:
            sams[sid].adata.obsm["X_umap_samap"] = self.samap.adata[sams[sid].adata.obs_names].obsm[
                "X_umap"
            ]

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
            sizes = dict.fromkeys(self.ids, 3)

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
                counts=self.samap.adata[np.isin(self.samap.adata.obs["species"], list(gs.keys()))]
            )
        else:
            samap = self.samap

        if ss is None:
            ss = dict.fromkeys(self.ids, 3)

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
        for sid in gs:
            g = gs[sid]
            try:
                AS[sid] = self.sams[sid].adata[:, g].X.toarray().flatten()
            except KeyError:
                try:
                    AS[sid] = self.sams[sid].adata[:, sid + "_" + g].X.toarray().flatten()
                except KeyError:
                    raise KeyError(f"Gene not found in species {sid}") from None

        davgs: dict[str, NDArray[Any]] = {}
        for sid in gs:
            d = np.zeros(samap.adata.shape[0])
            d[samap.adata.obs["species"] == sid] = AS[sid]
            davg = np.asarray(nnm.dot(d)).flatten()
            davg[davg < thr] = 0
            davgs[sid] = davg
        davg = np.vstack(list(davgs.values())).min(0)
        for sid in gs:
            if davgs[sid].max() > 0:
                davgs[sid] = davgs[sid] / davgs[sid].max()
        if davg.max() > 0:
            davg = davg / davg.max()

        cs: dict[str, NDArray[Any]] = {}
        for sid in gs:
            c = [*hex_to_rgb(colors[sid]), 0.0]
            cs[sid] = np.vstack([c] * davg.size)
            cs[sid][:, -1] = davgs[sid]
        cc = [*hex_to_rgb(colorc), 0.0]
        cc = np.vstack([cc] * davg.size)
        cc[:, -1] = davg

        ax = samap.scatter(projection="X_umap", colorspec=color0, axes=axes, s=s0)

        for sid in gs:
            samap.scatter(
                projection="X_umap", c=cs[sid], axes=ax, s=ss[sid], colorbar=False, **kwargs
            )

        samap.scatter(projection="X_umap", c=cc, axes=ax, s=sc, colorbar=False, **kwargs)

        return ax

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
        ix = pd.Series(data=np.arange(samap.adata.shape[1]), index=samap.adata.var_names)[
            gns
        ].values
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
        bk: Backend | None = None,
    ) -> None:
        self._bk = bk if bk is not None else Backend("cpu")
        self.sams = sams
        self.gnnm = gnnm
        self.gnnmu = gnnm
        self.gns_dict = gns_dict

        if keys is None:
            keys = dict.fromkeys(sams.keys(), "leiden_clusters")

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

        # Iteration-invariant projection state: standardised expression matrices,
        # their Gram matrices/means (for the sigma quadratic form), and the
        # own-species PC projection. Built once here, consumed every iteration
        # inside _mapper → _mapping_window_fast.
        self._gns = np.concatenate(list(gns_dict.values()))
        self._proj_cache = _projection_precompute(sams, self._gns, self._bk)

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
            NHS = dict.fromkeys(sams.keys(), 2)
        if neigh_from_keys is None:
            neigh_from_keys = dict.fromkeys(sams, False)
        gns = self._gns

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
                proj_cache=self._proj_cache,
                bk=self._bk,
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
                    np.asarray(s.adata.obsp["connectivities"][x == xu[i], :][:, x == xu[j]].sum(1))
                    .flatten()
                    .mean()
                    / s.adata.uns["mapping_K"]
                )
    return pd.DataFrame(data=a, index=xu, columns=xu)


