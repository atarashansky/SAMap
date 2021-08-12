from scipy.stats import binned_statistic
import hnswlib
import typing
from numba import njit, prange
import os
from os import path
import gc
from samalg import SAM
import time
from sklearn.preprocessing import StandardScaler

from . import q, ut, pd, sp, np, warnings, sc
from .analysis import _compute_csim
from .utils import prepend_var_prefix, to_vn, substr, sparse_knn

from numba.core.errors import NumbaPerformanceWarning, NumbaWarning

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
warnings.filterwarnings("ignore", category=NumbaWarning)


class SAMAP(object):
    def __init__(
        self,
        data1: typing.Union[str, SAM],
        data2: typing.Union[str, SAM],
        id1: typing.Optional[str] = "Aa",
        id2: typing.Optional[str] = "Bb",
        f_maps: typing.Optional[str] = "maps/",
        names1: typing.Optional[typing.Union[list, np.ndarray]] = None,
        names2: typing.Optional[typing.Union[list, np.ndarray]] = None,
        gnnm: typing.Optional[tuple] = None,
        taxa: typing.Optional[int] = 33208,
        E: typing.Optional[pd.DataFrame] = None,
        reciprocal_blast: typing.Optional[bool] = True,
        key1: typing.Optional[str] = "leiden_clusters",
        key2: typing.Optional[str] = "leiden_clusters",
        leiden_res1: typing.Optional[int] = 3,
        leiden_res2: typing.Optional[int] = 3,
        save_processed: typing.Optional[bool] = False
    ):

        """Initializes and preprocess data structures for SAMap algorithm.

        Parameters
        ----------
        data1 : string OR SAM
            The path to an unprocessed '.h5ad' `AnnData` object for organism 1.
            OR
            A processed and already-run SAM object.

        data2 : string OR SAM
            The path to an unprocessed '.h5ad' `AnnData` object for organism 2.
            OR
            A processed and already-run SAM object.

        id1 : string
            Organism 1 identifier (corresponds to the transcriptome ID provided
            when using `map_genes.sh`)

        id2 : string
            Organism 2 identifier (corresponds to the transcriptome ID provided
            when using `map_genes.sh`)

        f_maps : string, optional, default 'maps/'
            Path to the `maps` directory output by `map_genes.sh`.
            By default assumes it is in the local directory.

        names1 & names2 : list of 2D tuples or Nx2 numpy.ndarray, optional, default None
            If BLAST was run on a transcriptome with Fasta headers that do not match
            the gene symbols used in the dataset, you can pass a list of tuples mapping
            the Fasta header name to the Dataset gene symbol:
            (Fasta header name , Dataset gene symbol). Transcripts with the same gene
            symbol will be collapsed into a single node in the gene homology graph.
            By default, the Fasta header IDs are assumed to be equivalent to the
            gene symbols used in the dataset.

            names1 corresponds to the mapping for organism 1 and names2 corresponds to
            the mapping for organism 2.

        gnnm : tuple(scipy.sparse.csr_matrix,numpy array, numpy array)
            If the homology graph was already computed, you can pass it here in the form of a tuple:
            (sparse adjacency matrix, species 1 genes, species 2 genes).
            This is the tuple returned by `_calculate_blast_graph(...)` or `_coarsen_eggnog_graph(...)`.

        taxa : int, optional, default 33208
            Specifies the taxonomic level at which genes with overlapping orthology groups will be linked.
            Only used if EGGNOG is True.

        E : tuple(pd.DataFrame,pd.DataFrame), optional, default None
            A tuple of EGGNOG orthology group mapping tables corresponding to species 1 and 2,
            respectively. If not `None` (default) SAMap will construct its homology graph using
            the EGGNOG mapping tables instead of the BLAST results.
        
        reciprocal_blast : bool, optional, default True
            If True, only keep reciprocal edges in the computed BLAST graph.

        key1 & key2 : str, optional, default 'leiden_clusters'
            The `sam.adata.obs` key to use for determining maximum neighborhood size of each cell. 
            If set to 'leiden_clusters', leiden clustering will be performed using the resolution
            parameter specified by `leiden_res1/2`
            
        leiden_res1 & leiden_res2 : float, optional, default 3
            The resolution parameter for the leiden clustering to be done on the manifold of organisms
            1/2. Each cell's neighborhood size will be capped to be the size of its leiden cluster.
            
        save_processed : bool, optional, default False
            If True saves the processed SAM objects corresponding to each species to an `.h5ad` file.
            This argument is unused if preloaded SAM objects are passed in to SAMAP.
        """

        if not (isinstance(data1, str) or isinstance(data1, SAM)):
            raise TypeError("Input data 1 must be either a path or a SAM object.")

        if not (isinstance(data2, str) or isinstance(data2, SAM)):
            raise TypeError("Input data 2 must be either a path or a SAM object.")

        if isinstance(data1, str):
            print("Processing data 1 from:\n{}".format(data1))
            sam1 = SAM()
            sam1.load_data(data1)
            sam1.preprocess_data(
                sum_norm="cell_median",
                norm="log",
                thresh_low=0.0,
                thresh_high=0.96,
                min_expression=1,
            )
            sam1.run(
                preprocessing="StandardScaler",
                npcs=150,
                weight_PCs=False,
                k=20,
                n_genes=3000,
                weight_mode='combined'
            )
            f1n = ".".join(data1.split(".")[:-1]) + "_pr.h5ad"
            
            print("Preparing data 1 for SAMap.")
            if key1 == "leiden_clusters":
                sam1.leiden_clustering(res=leiden_res1)

            if "PCs_SAMap" not in sam1.adata.varm.keys():
                prepare_SAMap_loadings(sam1)
    
            if save_processed:
                print("Saving processed data to:\n{}".format(f1n))
                sam1.save_anndata(f1n)
        else:
            sam1 = data1
            print("Preparing data 1 for SAMap.")
            if key1 == "leiden_clusters":
                sam1.leiden_clustering(res=leiden_res1)

            if "PCs_SAMap" not in sam1.adata.varm.keys():
                prepare_SAMap_loadings(sam1)            

        if isinstance(data2, str):
            print("Processing data 2 from:\n{}".format(data2))
            sam2 = SAM()
            sam2.load_data(data2)
            sam2.preprocess_data(
                sum_norm="cell_median",
                norm="log",
                thresh_low=0.0,
                thresh_high=0.96,
                min_expression=1,
            )
            sam2.run(
                preprocessing="StandardScaler",
                npcs=150,
                weight_PCs=False,
                k=20,
                n_genes=3000,
                weight_mode='combined'
            )
            f2n = ".".join(data2.split(".")[:-1]) + "_pr.h5ad"
            
            print("Preparing data 2 for SAMap.")
            if key2 == "leiden_clusters":
                sam2.leiden_clustering(res=leiden_res2)

            if "PCs_SAMap" not in sam2.adata.varm.keys():
                prepare_SAMap_loadings(sam2)            
            
            if save_processed:            
                print("Saving processed data to:\n{}".format(f2n))
                sam2.save_anndata(f2n)
        else:         
            sam2 = data2
            print("Preparing data 2 for SAMap.")
            if key2 == "leiden_clusters":
                sam2.leiden_clustering(res=leiden_res2)

            if "PCs_SAMap" not in sam2.adata.varm.keys():
                prepare_SAMap_loadings(sam2)               

        if gnnm is None:
            if E is None:
                gnnm, gn1, gn2 = _calculate_blast_graph(
                    id1, id2, f_maps=f_maps, reciprocate=reciprocal_blast
                )
            else:
                gnnm, gn1, gn2 = _calculate_eggnog_graph(E[0], E[1], id1, id2, taxa=taxa)

            gn = np.append(gn1,gn2)
            
            if names1 is not None or names2 is not None:
                gnnm, gn1, gn2, gn = _coarsen_blast_graph(
                    gnnm, gn1, gn2, gn, id1, id2, names1, names2
                )

            gnnm = _filter_gnnm(gnnm, thr=0.25)
        else:
            gnnm, gn1, gn2 = gnnm
            gn = np.append(gn1,gn2)
            id1 = gn1[0].split("_")[0]
            id2 = gn2[0].split("_")[0]

        prepend_var_prefix(sam1, id1)
        prepend_var_prefix(sam2, id2)

        ge1 = q(sam1.adata.var_names)
        ge2 = q(sam2.adata.var_names)

        gn1 = gn1[np.in1d(gn1, ge1)]
        gn2 = gn2[np.in1d(gn2, ge2)]
        f = np.in1d(gn, np.append(gn1, gn2))
        gn = gn[f]
        gnnm = gnnm[f][:, f]
        A = pd.DataFrame(data=np.arange(gn.size)[None, :], columns=gn)
        ge = np.append(ge1, ge2)
        ge = ge[np.in1d(ge, gn)]
        ix = A[ge].values.flatten()
        gnnm = gnnm[ix][:, ix]
        gn = ge
        gn1 = ge[np.in1d(ge, gn1)]
        gn2 = ge[np.in1d(ge, gn2)]

        print(
            "{} `{}` genes and {} `{}` gene symbols match between the datasets and the BLAST graph.".format(
                gn1.size, id1, gn2.size, id2
            )
        )

        smap = _Samap_Iter(sam1, sam2, gnnm, gn1, gn2, key1=key1, key2=key2)
        self.sam1 = sam1
        self.sam2 = sam2
        self.gnnm = gnnm
        self.gn1 = gn1
        self.gn2 = gn2
        self.gn = gn
        self.id1 = id1
        self.id2 = id2
        self.smap = smap

    def run(
        self,
        NUMITERS: typing.Optional[int] = 3,
        NH1: typing.Optional[int] = 3,
        NH2: typing.Optional[int] = 3,
        K: typing.Optional[int] = 20,
        N_GENE_CHUNKS: typing.Optional[int] = 1,
        umap: typing.Optional[bool] = True,
        ncpus=os.cpu_count(),
        hom_edge_thr=0,
        hom_edge_mode = "pearson",
        scale_edges_by_corr = False,
        neigh_from_key1 = False,
        neigh_from_key2 = False
        
    ):
        """Runs the SAMap algorithm.

        Parameters
        ----------
        NUMITERS : int, optional, default 3
            Runs SAMap for `NUMITERS` iterations.        
            
        NH1 : int, optional, default 3
            Cells up to `NH1` hops away from a particular cell in organism 1
            will be included in its neighborhood.

        NH2 : int, optional, default 3
            Cells up to `NH2` hops away from a particular cell in organism 2
            will be included in its neighborhood.

        K : int, optional, default 20
            The number of cross-species edges to identify per cell.

        N_GENE_CHUNKS : int, optional, default 1
            When updating the edge weights in the BLAST homology graph, the operation
            will be split up into `N_GENE_CHUNKS` chunks. For large datasets
            (>50,000 cells), use more chunks (e.g. 4) to avoid running out of
            memory.
            
        umap : bool, optional, default True
            If True, performs UMAP on the combined manifold to generate a 2D visualization.
            If False, skips this step. 
            
        ncpus : int, optional, default `os.cpu_count()`
            The number of CPUs to use when computing gene-gene correlations.
            Defaults to using all available CPUs.
            
        hom_edge_thr : float, optional, default 0
            Edges with weight below `hom_edge_thr` in the homology graph will be set to zero.
            
        hom_edge_mode: str, optional, default "pearson"
            If "pearson", edge weights in the homology graph will be calculated using Pearson
            correlation. If "mutual_info", edge weights will be calculated using normalized
            mutual information. The latter requires package `fast-histogram` to be installed.
            
        scale_edges_by_corr: bool, optional, default False
            If True, scale cell-cell cross-species edges by their expression similarities
            (correlations).
            
        neigh_from_key1 : bool, optional, default False
            If True, species 1 neighborhoods are calculated directly from the chosen clustering (`self.key1`).
            Cells within the same cluster belong to the same neighborhood.
            
        neigh_from_key2 : bool, optional, default False
            If True, species 2 neighborhoods are calculated directly from the chosen clustering (`self.key2`).
            Cells within the same cluster belong to the same neighborhood.
            
        Returns
        -------
        samap - Species-merged SAM object
        """
        start_time = time.time()

        sam1 = self.sam1
        sam2 = self.sam2
        gnnm = self.gnnm
        gn1 = self.gn1
        gn2 = self.gn2
        gn = self.gn
        smap = self.smap

        smap.run(
            NUMITERS=NUMITERS,
            NOPs1=0,
            NOPs2=0,
            NH1=NH1,
            NH2=NH2,
            K=K,
            NCLUSTERS=N_GENE_CHUNKS,
            ncpus=ncpus,
            THR=hom_edge_thr,
            corr_mode=hom_edge_mode,
            scale_edges_by_corr = scale_edges_by_corr,
            neigh_from_key1=neigh_from_key1,
            neigh_from_key2=neigh_from_key2
        )
        samap = smap.final_sam
        self.samap = samap
        self.ITER_DATA = smap.ITER_DATA

        print("Alignment score ---", _avg_as(samap).mean())
        if umap:
            print("Running UMAP on the stitched manifolds.")
            sc.tl.umap(self.samap.adata,min_dist=0.1,init_pos='random')
        
        try:
            hom_graph = smap.GNNMS_corr[-1]
            samap.adata.uns["homology_graph_reweighted"] = hom_graph
        except:
            pass
        
        samap.adata.uns["homology_graph"] = gnnm
        samap.adata.uns["homology_gene_names"] = gn
        samap.adata.uns["homology_gene_names1"] = gn1
        samap.adata.uns["homology_gene_names2"] = gn2

        samap.adata.obs["species"] = pd.Categorical(
            [self.id1] * sam1.adata.shape[0] + [self.id2] * sam2.adata.shape[0]
        )
        
        if umap:
            self.sam1.adata.obsm['X_umap_samap'] = self.samap.adata[self.sam1.adata.obs_names].obsm['X_umap']
            self.sam2.adata.obsm['X_umap_samap'] = self.samap.adata[self.sam2.adata.obs_names].obsm['X_umap']        
        
        self.run_time = time.time() - start_time
        print("Elapsed time: {} minutes.".format(self.run_time / 60))
        return samap

    def run_umap(self):
        print("Running UMAP on the stitched manifolds.")
        sc.tl.umap(self.samap.adata,min_dist=0.1,init_pos='random')
        self.sam1.adata.obsm['X_umap_samap'] = self.samap.adata[self.sam1.adata.obs_names].obsm['X_umap']
        self.sam2.adata.obsm['X_umap_samap'] = self.samap.adata[self.sam2.adata.obs_names].obsm['X_umap']        

    def plot_expression_overlap(self,g1,g2,axes=None,
                                COLOR0='gray', COLOR1='#000098', COLOR2='#ffb900', COLOR3='#00ceb5',
                                s0 = 1, s1 = 3, s2 = 3, s3 = 10,
                                thr = 0.1,**kwargs):
        """Displays the expression overlap of two genes on the combined manifold.

        Parameters
        ----------
        g1 : str
            Gene ID from species 1.
        
        g2 : str
            Gene ID from species 2.
            
        axes : matplotlib.pyplot.Axes, optional, default None
            Displays the scatter plot on the provided axes if specified.
            Otherwise creates a new figure.
            
        COLOR0 : str, optional, default 'gray'
            The color for cells that do not express `g1` or `g2`.
        
        COLOR1 : str, optional, default '#000098'
            The color for cells expressing `g1`.
        
        COLOR2 : str, optional, default '#ffb900'
            The color for cells expressing `g2`.
        
        COLOR3 : str, optional, default '#00ceb5'
            The color for cells that overlap in
            expression of `g1` and `g2`.
        
        s0 : int, optional, default 1
            Marker size corresponding to `COLOR0`.
            
        s1 : int, optional, default 3
            Marker size corresponding to `COLOR1`.
            
        s2 : int, optional, default 3
            Marker size corresponding to `COLOR2`.
            
        s3 : int, optional, default 10
            Marker size corresponding to `COLOR3`.
        
        thr : float, optional, default 0.1
            Threshold below which imputed expressions across species are zero'd out. 
        
        Keyword Arguments (**kwargs)
        ----------------------------
        Most arguments accepted by matplotlib.pyplot.scatter are available.
        

        Returns
        -------
        ax - matplotlib.pyplot.Axes
        """
        
        from matplotlib import cm
        from matplotlib.colors import to_rgba
        def hex_to_rgb(value):
            value = value.lstrip('#')
            lv = len(value)
            lv = list(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
            lv = [x/255 for x in lv]
            return lv
        
        sm = self
        
        nnm = sm.samap.adata.obsp['connectivities']
        su = nnm.sum(1).A.flatten()[:,None]
        su[su==0]=1

        nnm = nnm.multiply(1/su).tocsr()
        B1 = True
        if g1 not in sm.sam1.adata.var_names:
            if sm.id1+"_"+g1 not in sm.sam1.adata.var_names:
                if g1 not in sm.sam2.adata.var_names:
                    if sm.id2+"_"+g1 not in sm.sam2.adata.var_names:
                        raise ValueError(f"{g1} not found in either dataset.")
                    else:
                        a1 = sm.sam2.adata[:,sm.id2+"_"+g1].X.A.flatten()[:,None]
                        B1=False
                else:
                    a1 = sm.sam2.adata[:,g1].X.A.flatten()[:,None]
                    B1=False
            else:
                a1 = sm.sam1.adata[:,sm.id1+"_"+g1].X.A.flatten()[:,None]
        else:
            a1 = sm.sam1.adata[:,g1].X.A.flatten()[:,None]
            
        B2=True
        if g2 not in sm.sam2.adata.var_names:
            if sm.id2+"_"+g2 not in sm.sam2.adata.var_names:
                if g2 not in sm.sam1.adata.var_names:
                    if sm.id1+"_"+g2 not in sm.sam1.adata.var_names:
                        raise ValueError(f"{g2} not found in either dataset.")
                    else:
                        a2 = sm.sam1.adata[:,sm.id1+"_"+g2].X.A.flatten()[:,None]
                        B2=False
                else:
                    a2 = sm.sam1.adata[:,g2].X.A.flatten()[:,None]
                    B2=False
            else:
                a2 = sm.sam2.adata[:,sm.id2+"_"+g2].X.A.flatten()[:,None]
        else:
            a2 = sm.sam2.adata[:,g2].X.A.flatten()[:,None]            

        d1 = np.append(a1,np.zeros(sm.sam2.adata.shape[0])) if B1 else np.append(np.zeros(sm.sam1.adata.shape[0]),a1)
        d2 = np.append(np.zeros(sm.sam1.adata.shape[0]),a2) if B2 else np.append(a2,np.zeros(sm.sam2.adata.shape[0]))

        davg1 = nnm.dot(d1).flatten()
        davg2 = nnm.dot(d2).flatten()
        davg1[davg1<thr]=0
        davg2[davg2<thr]=0
        
        davg3 = np.vstack((davg1,davg2)).min(0)
        ma = max([davg1.max(),davg2.max(),davg3.max()])
        if davg1.max()>0:
            davg1 = davg1/davg1.max()
        if davg2.max()>0:
            davg2 = davg2/davg2.max()
        if davg3.max()>0:
            davg3 = davg3/davg3.max()            

        c1 = hex_to_rgb(COLOR1)+[0.0]
        c2 = hex_to_rgb(COLOR2)+[0.0]
        c3 = hex_to_rgb(COLOR3)+[0.0]

        c1 = np.vstack([c1]*davg1.size)
        c2 = np.vstack([c2]*davg1.size)
        c3 = np.vstack([c3]*davg1.size)
        c1[:,-1] = davg1# if not reverse else davg2
        c2[:,-1] = davg2# if not reverse else davg1
        c3[:,-1] = davg3

        ax = sm.samap.scatter(projection = 'X_umap', colorspec = COLOR0, axes=axes, s = s0)
        sm.samap.scatter(projection = 'X_umap', c = c1, axes = ax, s = s1,colorbar=False,**kwargs)
        sm.samap.scatter(projection = 'X_umap', c = c2, axes = ax, s = s2,colorbar=False,**kwargs)
        sm.samap.scatter(projection = 'X_umap', c = c3, axes = ax, s = s3,colorbar=False,**kwargs)
        return ax    
    
    def scatter(self,axes=None, c1='#000098', c2='#ffb900', s1=10,s2=10,alpha1=1.0,alpha2=1.0, **kwargs):
        """Displays the UMAP projection of the combined manifold.

        Parameters
        ----------
        axes : matplotlib.pyplot.Axes, optional, default None
            Displays the scatter plot on the provided axes if specified.
            Otherwise creates a new figure.
            
        c1 : str, optional, default '#000098'
            The color for cells from species 1.
        
        c2 : str, optional, default '#ffb900'
            The color for cells from species 2.
            
        s1 : int, optional, default 10
            Marker size corresponding to `c1`.
            
        s2 : int, optional, default 10
            Marker size corresponding to `c2`.
        
        alpha1 : float, optional, default 1.0
            The transparency of dots from species 1.
            
        alpha2 : float, optional, default 1.0
            The transparency of dots from species 2.
                    
        Keyword Arguments (**kwargs)
        ----------------------------
        Most arguments accepted by matplotlib.pyplot.scatter are available.
        

        Returns
        -------
        ax - matplotlib.pyplot.Axes
        """
        
        ax = self.sam1.scatter(projection = 'X_umap_samap',axes=axes,colorspec=c1, s=s1, alpha=alpha1,**kwargs)
        ax = self.sam2.scatter(projection = 'X_umap_samap',axes=ax,colorspec=c2, s=s2, alpha=alpha2,**kwargs)
        return ax
        
    def gui(self):
        """Launches a SAMGUI instance containing the two SAM objects."""
        if 'SamapGui' not in self.__dict__:
            try:
                from samalg.gui import SAMGUI
            except ImportError:
                raise ImportError('Please install SAMGUI dependencies. See the README in the SAM github repository.')

            sg = SAMGUI(sam = [self.sam1,self.sam2], title = [self.id1,self.id2],default_proj='X_umap_samap')
            self.SamapGui = sg
            return sg.SamPlot
        else:
            return self.SamapGui.SamPlot
        
    def refine_homology_graph(self, THR=0, NCLUSTERS=1, ncpus = os.cpu_count(), corr_mode='pearson', ct_labels = None):
        return self.smap.refine_homology_graph(NCLUSTERS=NCLUSTERS, ncpus=ncpus, THR=THR, corr_mode=corr_mode, ct_labels=ct_labels)
        
class _Samap_Iter(object):
    def __init__(
        self, sam1, sam2, gnnm, gn1, gn2, key1="leiden_clusters", key2="leiden_clusters"
    ):
        self.sam1 = sam1
        self.sam2 = sam2
        self.gnnm = gnnm
        self.gnnmu = gnnm
        self.gn1 = gn1
        self.gn2 = gn2
        self.key1 = key1
        self.key2 = key2
        self.SCORE_VEC = []
        self.GNNMS_corr = []
        self.GNNMS_pruned = []
        self.GNNMS_nnm = []
        self.ITER_DATA = [
            self.GNNMS_nnm,
            self.GNNMS_corr,
            self.GNNMS_pruned,
            self.SCORE_VEC,
        ]
        self.iter = 0

    def refine_homology_graph(self, NCLUSTERS=1, ncpus=os.cpu_count(), THR=0, corr_mode='pearson', ct_labels=None):
        if corr_mode=='mutual_info':
            try:
                from fast_histogram import histogram2d
            except:
                raise ImportError("Package `fast_histogram` must be installed for `corr_mode='mutual_info'`.");
                
        sam1 = self.sam1
        sam2 = self.sam2
        sam4 = self.samap
        
        gnnm = self.gnnm
        gn1 = self.gn1
        gn2 = self.gn2
        gnnmu = self.gnnmu
        
        gnnmu = _refine_corr(
            sam1,
            sam2,
            sam4,
            gnnm,
            gn1,
            gn2,
            THR=THR,
            use_seq=False,
            T1=0,
            T2=0,
            NCLUSTERS=NCLUSTERS,
            ncpus=ncpus,
            corr_mode=corr_mode,
            ct_labels=ct_labels
        )
        return gnnmu

    def run(self, NUMITERS=3, NOPs1=0, NOPs2=0, NH1=2, NH2=2, K=20, corr_mode='pearson', NCLUSTERS=1,
                  scale_edges_by_corr=False, THR=0, neigh_from_key1=False, neigh_from_key2=False, ncpus=os.cpu_count()):
        sam1 = self.sam1
        sam2 = self.sam2
        gnnm = self.gnnm
        gn1 = self.gn1
        gn2 = self.gn2
        gnnmu = self.gnnmu
        
        gn = np.append(gn1, gn2)

        if self.iter > 0:
            sam4 = self.samap

        for i in range(NUMITERS):
            if self.iter > 0 and i == 0:
                print("Calculating gene-gene correlations in the homology graph...")
                gnnmu = self.refine_homology_graph(ncpus = ncpus, NCLUSTERS = NCLUSTERS, THR=THR, corr_mode=corr_mode)

                self.GNNMS_corr.append(gnnmu)
                self.gnnmu = gnnmu

            gnnm2 = _get_pairs(sam1, sam2, gnnmu, gn1, gn2, NOPs1=NOPs1, NOPs2=NOPs2)
            self.GNNMS_pruned.append(gnnm2)

            sam4 = _mapper(
                [sam1, sam2],
                gnnm2,
                gn,
                umap=False,
                K=K,
                NH1=NH1,
                NH2=NH2,
                coarsen=True,
                key1=self.key1,
                key2=self.key2,
                scale_edges_by_corr=scale_edges_by_corr,
                neigh_from_key1=neigh_from_key1,
                neigh_from_key2=neigh_from_key2
            )

            self.samap = sam4
            self.GNNMS_nnm.append(sam4.adata.obsp["connectivities"])

            _, _, _, CSIMth = _compute_csim(sam4, "leiden_clusters")

            self.SCORE_VEC.append(CSIMth.flatten())
            if len(self.SCORE_VEC) > 1:
                diff = self.SCORE_VEC[-1] - self.SCORE_VEC[-2]
            elif len(self.SCORE_VEC) > 0:
                diff = self.SCORE_VEC[-1] - np.zeros(self.SCORE_VEC[-1].size)

            diffmax = diff.max()
            diffmin = diff.min()
            print(
                "ITERATION: " + str(i),
                "\nAverage alignment score (A.S.): ",
                _avg_as(sam4).mean(),
                "\nMax A.S. improvement:",
                diffmax,
                "\nMin A.S. improvement:",
                diffmin,
            )
            self.iter += 1
            if i < NUMITERS - 1:
                print("Calculating gene-gene correlations in the homology graph...")
                self.samap = sam4
                gnnmu = self.refine_homology_graph(ncpus = ncpus,  NCLUSTERS = NCLUSTERS,  THR=THR, corr_mode=corr_mode)

                self.GNNMS_corr.append(gnnmu)
                self.gnnmu = gnnmu

            gc.collect()

        self.final_sam = sam4
        self.final_sam.adata.uns["edge_weights"] = self.final_sam.adata.uns["mdata"][
            "edge_weights"
        ]
        
@njit(parallel=True)        
def _replace(X,Y,xi,yi):
    data = np.zeros(xi.size)
    for i in prange(xi.size):
        x=X[xi[i]]
        y=Y[yi[i]]
        data[i] = ((x-x.mean())*(y-y.mean()) / x.std() / y.std()).sum() / x.size
    return data
    
    
def _generate_coclustering_matrix(cl):
    cl = ut.convert_annotations(np.array(list(cl)))
    clu,cluc=np.unique(cl,return_counts=True)    
    v = np.zeros((cl.size,clu.size))
    v[np.arange(v.shape[0]),cl]=1
    v = sp.sparse.csr_matrix(v)
    return v

    
def _mapper(
    sams,
    gnnm,
    gn,
    NH1=3,
    NH2=3,
    umap=False,
    mdata=None,
    k=None,
    K=20,
    chunksize=20000,
    coarsen=True,
    key1="leiden_clusters",
    key2="leiden_clusters",
    scale_edges_by_corr=False,
    neigh_from_key1=False,
    neigh_from_key2=False,
    **kwargs
):
    n = len(sams)
    DS = {}
    for I in range(n):
        sam = sams[I]
        for J in range(I + 1, n):
            print("Stitching SAM " + str(I) + " and SAM " + str(J))
            sam2 = sams[J]

            if mdata is None:
                mdata = _mapping_window(sam, sam2, gnnm, gn, K=K)

            if k is None:
                k1 = sam.run_args.get("k", 20)
            else:
                k1 = k

            print("Using " + key1 + " and " + key2 + " cluster labels.")

            CL1 = sam.get_labels(key1)
            CL2 = sam2.get_labels(key2)

            clu1, ix1, cluc1 = np.unique(CL1, return_counts=True, return_inverse=True)
            clu2, ix2, cluc2 = np.unique(CL2, return_counts=True, return_inverse=True)

            K1 = cluc1[ix1]
            K2 = cluc2[ix2]

            h2m = mdata["knn_1v2"]
            m2h = mdata["knn_2v1"]

            if coarsen:
                h2m0 = h2m.copy()
                m2h0 = m2h.copy()
                h2m0.data[:] = 1
                m2h0.data[:] = 1

                print("Out-neighbor smart expansion 1")
                nnm = sam.adata.obsp["connectivities"].copy()
                nnm1_out = nnm
                if not neigh_from_key1:
                    nnm1_in = _smart_expand(nnm, K1, NH=NH1)
                    nnm1_in.data[:] = 1
                else:
                    nnm1_in = _generate_coclustering_matrix(CL1)

                print("Out-neighbor smart expansion 2")
                nnm = sam2.adata.obsp["connectivities"].copy()
                nnm2_out = nnm
                if not neigh_from_key2:
                    nnm2_in = _smart_expand(nnm, K2, NH=NH2)
                    nnm2_in.data[:] = 1
                else:
                    nnm2_in = _generate_coclustering_matrix(CL2)                    


                mdata["nnm1_out"] = nnm1_out
                mdata["nnm1_in"] = nnm1_in
                mdata["nnm2_out"] = nnm2_out
                mdata["nnm2_in"] = nnm2_in

                B = h2m
                B2 = m2h
                s = B.sum(1).A
                s2 = B2.sum(1).A
                s[s == 0] = 1
                s2[s2 == 0] = 1

                B = B.multiply(1 / s).tocsr()
                B2 = B2.multiply(1 / s2).tocsr()

                print("Indegree coarsening")

                numiter = max(nnm2_in.shape[0], nnm1_in.shape[0]) // chunksize + 1

                if nnm2_in.shape[0] < nnm1_in.shape[0]:
                    R = True
                else:
                    R = False

                D = sp.sparse.csr_matrix((0, min(nnm2_in.shape[0], nnm1_in.shape[0])))
                for bl in range(numiter):
                    print(str(bl) + "/" + str(numiter), D.shape, R)
                    if not R:
                        if neigh_from_key2:
                            C = nnm2_in[bl * chunksize : (bl + 1) * chunksize].dot(nnm2_in.T.dot(B.T))
                        else:
                            C = nnm2_in[bl * chunksize : (bl + 1) * chunksize].dot(B.T)
                        C.data[C.data < 0.1] = 0
                        C.eliminate_zeros()
                        if neigh_from_key1:
                            C2 = nnm1_in.dot(nnm1_in.T.dot(B2[bl * chunksize : (bl + 1) * chunksize].T)).T
                        else:
                            C2 = nnm1_in.dot(B2[bl * chunksize : (bl + 1) * chunksize].T).T
                        C2.data[C2.data < 0.1] = 0
                        C2.eliminate_zeros()
                    else:
                        if neigh_from_key2:
                            C = nnm2_in.dot(nnm2_in.T.dot(B[bl * chunksize : (bl + 1) * chunksize].T)).T
                        else:
                            C = nnm2_in.dot(B[bl * chunksize : (bl + 1) * chunksize].T).T
                        C.data[C.data < 0.1] = 0
                        C.eliminate_zeros()
                        if neigh_from_key1:
                            C2 = nnm1_in[bl * chunksize : (bl + 1) * chunksize].dot(nnm1_in.T.dot(B2.T))
                        else:
                            C2 = nnm1_in[bl * chunksize : (bl + 1) * chunksize].dot(B2.T)
                        C2.data[C2.data < 0.1] = 0
                        C2.eliminate_zeros()

                    X = C.multiply(C2)
                    X.data[:] = X.data ** 0.5
                    del C
                    del C2
                    gc.collect()
                    D = sp.sparse.vstack((D, X))
                    del X
                    gc.collect()

                if not R:
                    D = D.T
                D = D.tocsr()

                mdata["xsim"] = D
                if scale_edges_by_corr:
                    print('Scaling edge weights by expression correlations.')                
                    x,y = D.nonzero()
                    vals = _replace(mdata["wPCA1"],mdata["wPCA2"],x,y)
                    vals[vals<0]=0
                    D.data[:] = np.sqrt(vals*D.data)
                
                D1 = sparse_knn(D, k1).tocsr()
                D2 = sparse_knn(D.T, k1).tocsr()                

            else:
                D1 = h2m
                D2 = m2h
                if k1 < K:
                    print("Redoing sparse kNN selection...")
                    D1 = sparse_knn(D1, k1).tocsr()
                    D2 = sparse_knn(D2, k1).tocsr()

            try:
                DS[I][J] = D1
            except:
                DS[I] = {}
                DS[I][J] = D1

            try:
                DS[J][I] = D2
            except:
                DS[J] = {}
                DS[J][I] = D2

    ROWS = []
    for I in range(n):
        ROW = []
        ROWt = []
        for J in range(n):
            if I != J:
                ROW.append(DS[I][J])
                ROWt.append(DS[J][I])

        nnm = sams[I].adata.obsp["connectivities"]

        row = sp.sparse.hstack(ROW)
        rowt = sp.sparse.vstack(ROWt)
        x = 1 - row.sum(1).A.flatten() / k1 / (n - 1)

        # onemode projection
        s = row.sum(1).A
        s[s == 0] = 1
        s2 = rowt.sum(1).A
        s2[s2 == 0] = 1
        proj = row.multiply(1 / s).dot(rowt.multiply(1 / s2)).tocsr()

        # find rows with abnormally small # edges in projection
        z = proj.copy()
        z.data[:] = 1
        idx = np.where(z.sum(1).A.flatten() >= k1)[0]

        # copy nearest neighbor graph
        omp = nnm.copy().astype("float")
        omp.data[:] = 1

        # renormalize edge weights to max 1
        s = proj.max(1).A
        s[s == 0] = 1
        proj = proj.multiply(1 / s).tocsr()

        # find edges in original graph and only choose ones from nonzero rows
        X, Y = omp.nonzero()
        X2 = X[np.in1d(X, idx)]
        Y2 = Y[np.in1d(X, idx)]

        omp = omp.tolil()
        omp[X2, Y2] = np.vstack((proj[X2, Y2].A.flatten(), np.ones(X2.size) * 0.3)).max(
            0
        )
        omp = omp.tocsr()
        omp = omp.multiply(x[:, None]).tocsr()
        ROW.insert(I, omp)
        ROWS.append(sp.sparse.hstack(ROW))
    NNM = sp.sparse.vstack((ROWS)).tolil()

    NNM.setdiag(0)
    # """

    print("Concatenating SAM objects...")
    sam3 = _concatenate_sam(sams, NNM, mdata["ortholog_pairs"])

    sam3.adata.uns["mdata"] = mdata

    if umap:
        print("Computing UMAP projection...")
        sc.tl.umap(sam3.adata, min_dist=0.1)
    return sam3

def _refine_corr(
    sam1,
    sam2,
    st,
    gnnm,
    gn1,
    gn2,
    corr_mode="mutual_info",
    THR=0,
    use_seq=False,
    T1=0.25,
    T2=0,
    NCLUSTERS=1,
    ncpus=os.cpu_count(),
    ct_labels=None
):
    # import networkx as nx
    gn = np.append(gn1, gn2)

    x, y = gnnm.nonzero()
    cl = sam1.leiden_clustering(gnnm, res=0.5)
    ix = np.argsort(cl)
    NGPC = gn.size // NCLUSTERS + 1
    ixs = []
    for i in range(NCLUSTERS):
        ixs.append(np.sort(ix[i * NGPC : (i + 1) * NGPC]))

    assert np.concatenate(ixs).size == gn.size

    GNNMSUBS = []
    CORRSUBS = []
    GNSUBS = []
    for i in range(len(ixs)):
        ixs[i] = np.unique(np.append(ixs[i], gnnm[ixs[i], :].nonzero()[1]))
        gnnm_sub = gnnm[ixs[i], :][:, ixs[i]]
        gnsub = gn[ixs[i]]
        gn1_sub = gn1[np.in1d(gn1, gnsub)]
        gn2_sub = gn2[np.in1d(gn2, gnsub)]
        gnnm2_sub, CORR_sub = _refine_corr_parallel(
            sam1,
            sam2,
            st,
            gnnm_sub,
            gn1_sub,
            gn2_sub,
            corr_mode=corr_mode,
            THR=THR,
            use_seq=use_seq,
            T1=T1,
            T2=T2,
            ncpus=ncpus,
            ct_labels=ct_labels
        )
        GNNMSUBS.append(gnnm2_sub)
        CORRSUBS.append(CORR_sub)
        GNSUBS.append(gnsub)
        gc.collect()
    I = []
    P = []
    for i in range(len(GNNMSUBS)):
        I.append(
            np.unique(np.sort(np.vstack((GNNMSUBS[i].nonzero())).T, axis=1), axis=0)
        )
        P.append(GNSUBS[i][I[-1]])

    GN = pd.DataFrame(data=np.arange(gn.size)[None, :], columns=gn)
    gnnm3 = sp.sparse.lil_matrix(gnnm.shape)
    for i in range(len(I)):
        x, y = GN[P[i][:, 0]].values.flatten(), GN[P[i][:, 1]].values.flatten()
        gnnm3[x, y] = GNNMSUBS[i][I[i][:, 0], I[i][:, 1]].A.flatten()

    gnnm3 = gnnm3.tocsr()
    x, y = gnnm3.nonzero()
    # gnnm3[y,x]=gnnm3.data
    gnnm3 = gnnm3.tolil()
    gnnm3[y, x] = gnnm3[x, y].A.flatten()
    return gnnm3.tocsr()


def _calculate_eggnog_graph(A, B, id1, id2, taxa=33208):
    import networkx as nx

    A = A.copy()
    B = B.copy()
    A.index = id1 + "_" + A.index
    B.index = id2 + "_" + B.index

    x, ix = ut.search_string(A.values, "@{}".format(taxa))
    A = A.iloc[ix]
    ogA = substr(substr(x, "@{}".format(taxa), 0), ",", -1)

    x, ix = ut.search_string(B.values, "@{}".format(taxa))
    B = B.iloc[ix]
    ogB = substr(substr(x, "@{}".format(taxa), 0), ",", -1)

    ogs = np.unique(np.append(ogA, ogB))
    T = pd.DataFrame(data=np.arange(ogs.size)[None, :], columns=ogs)
    Am = sp.sparse.coo_matrix(
        (np.ones(A.size), (np.arange(A.size), T[ogA].values.flatten())),
        shape=(A.size, ogs.size),
    )
    Bm = sp.sparse.coo_matrix(
        (np.ones(B.size), (np.arange(B.size), T[ogB].values.flatten())),
        shape=(B.size, ogs.size),
    )

    iA, iB = np.vstack(Am.dot(Bm.T).nonzero())
    gA = q(A.index)[iA]
    gB = q(B.index)[iB]

    G = nx.Graph()
    G.add_edges_from(np.vstack((gA, gB)).T)
    gn = q(G.nodes)
    gnnm = nx.convert_matrix.to_scipy_sparse_matrix(G)

    gn1 = gn[np.array([x.split("_")[0] for x in gn]) == id1]
    gn2 = gn[np.array([x.split("_")[0] for x in gn]) == id2]
    
    f1 = np.where(np.in1d(gn,gn1))[0]
    f2 = np.where(np.in1d(gn,gn2))[0]
    f = np.append(f1,f2)
    gnnm = gnnm[f,:][:,f]
    gn1 = gn[f1]
    gn2 = gn[f2]
    return gnnm, gn1, gn2


def _calculate_blast_graph(id1, id2, f_maps="maps/", eval_thr=1e-6, reciprocate=False):

    if os.path.exists(f_maps + "{}{}".format(id1, id2)):
        fA = f_maps + "{}{}/{}_to_{}.txt".format(id1, id2, id1, id2)
        fB = f_maps + "{}{}/{}_to_{}.txt".format(id1, id2, id2, id1)
    elif os.path.exists(f_maps + "{}{}".format(id2, id1)):
        fA = f_maps + "{}{}/{}_to_{}.txt".format(id2, id1, id1, id2)
        fB = f_maps + "{}{}/{}_to_{}.txt".format(id2, id1, id2, id1)
    else:
        raise FileExistsError(
            "BLAST mapping tables with the input IDs ({} and {}) not found in the specified path.".format(
                id1, id2
            )
        )

    A = pd.read_csv(fA, sep="\t", header=None, index_col=0)
    B = pd.read_csv(fB, sep="\t", header=None, index_col=0)

    A.columns = A.columns.astype("<U100")
    B.columns = B.columns.astype("<U100")

    A = A[A.index.astype("str") != "nan"]
    A = A[A.iloc[:, 0].astype("str") != "nan"]
    B = B[B.index.astype("str") != "nan"]
    B = B[B.iloc[:, 0].astype("str") != "nan"]

    A.index = id1 + "_" + A.index.astype("str").astype("object")
    B.iloc[:, 0] = (
        id1 + "_" + B.iloc[:, 0].values.flatten().astype("str").astype("object")
    )

    B.index = id2 + "_" + B.index.astype("str").astype("object")
    A.iloc[:, 0] = (
        id2 + "_" + A.iloc[:, 0].values.flatten().astype("str").astype("object")
    )

    i1 = np.where(A.columns == "10")[0][0]
    i3 = np.where(A.columns == "11")[0][0]

    inA = q(A.index)
    inB = q(B.index)

    inA2 = q(A.iloc[:, 0])
    inB2 = q(B.iloc[:, 0])
    gn1 = np.unique(np.append(inB2, inA))
    gn2 = np.unique(np.append(inA2, inB))
    gn = np.append(gn1, gn2)
    gnind = pd.DataFrame(data=np.arange(gn.size)[None, :], columns=gn)

    A.index = pd.Index(gnind[A.index].values.flatten())
    B.index = pd.Index(gnind[B.index].values.flatten())
    A.iloc[:, 0] = gnind[A.iloc[:, 0].values.flatten()].values.flatten()
    B.iloc[:, 0] = gnind[B.iloc[:, 0].values.flatten()].values.flatten()

    Arows = np.vstack((A.index, A.iloc[:, 0], A.iloc[:, i3])).T
    Arows = Arows[A.iloc[:, i1].values.flatten() <= eval_thr, :]
    gnnm1 = sp.sparse.lil_matrix((gn.size,) * 2)
    gnnm1[Arows[:, 0].astype("int32"), Arows[:, 1].astype("int32")] = Arows[
        :, 2
    ]  # -np.log10(Arows[:,2]+1e-200)

    Brows = np.vstack((B.index, B.iloc[:, 0], B.iloc[:, i3])).T
    Brows = Brows[B.iloc[:, i1].values.flatten() <= eval_thr, :]
    gnnm2 = sp.sparse.lil_matrix((gn.size,) * 2)
    gnnm2[Brows[:, 0].astype("int32"), Brows[:, 1].astype("int32")] = Brows[
        :, 2
    ]  # -np.log10(Brows[:,2]+1e-200)

    gnnm = (gnnm1 + gnnm2).tocsr()
    gnnms = (gnnm + gnnm.T) / 2
    if reciprocate:
        gnnm.data[:] = 1
        gnnms = gnnms.multiply(gnnm).multiply(gnnm.T).tocsr()
    gnnm = gnnms

    f1 = np.where(np.in1d(gn,gn1))[0]
    f2 = np.where(np.in1d(gn,gn2))[0]
    f = np.append(f1,f2)
    gnnm = gnnm[f,:][:,f]
    gn1 = gn[f1]
    gn2 = gn[f2]    
    return gnnm, gn1, gn2


def _coarsen_blast_graph(gnnm, gn1, gn2, gn, id1, id2, namesA, namesB):
    if namesA is not None:
        groupsA = {}
        for i in range(len(namesA)):
            name = namesA[i][1]
            L = groupsA.get(name, [])
            L.append(namesA[i][0])
            groupsA[name] = L
    else:
        groupsA = None

    if namesB is not None:
        groupsB = {}
        for i in range(len(namesB)):
            name = namesB[i][1]
            L = groupsB.get(name, [])
            L.append(namesB[i][0])
            groupsB[name] = L
    else:
        groupsB = None

    for n, groups, ids, g in zip(
        ["A", "B"], [groupsA, groupsB], [id1, id2], [gn1, gn2]
    ):
        if groups is not None:
            if isinstance(groups, dict):
                xdim = len(list(groups.keys()))
                for I in range(2):
                    print(
                        "Coarsening gene connectivity graph using labels `{}`, round {}.".format(
                            ids, I
                        )
                    )
                    X = []
                    Y = []
                    D = []
                    for i, k in enumerate(groups.keys()):
                        if len(groups[k]) > 1:
                            f = np.in1d(gn, [ids + "_" + x for x in groups[k]])
                            if f.sum() > 0:
                                z = gnnm[f].max(0).A.flatten()
                            else:
                                z = np.array([])
                        else:
                            f = gn == ids + "_" + groups[k][0]
                            if np.any(f):
                                z = gnnm[f].A.flatten()
                            else:
                                z = np.array([])
                        y = z.nonzero()[0]
                        d = z[y]
                        x = np.ones(y.size) * i
                        X.extend(x)
                        Y.extend(y)
                        D.extend(d)
                    if n == "A":
                        xa, ya = gnnm[gn1.size :].nonzero()
                        da = gnnm[gn1.size :].data
                        xa += xdim
                    else:
                        xa, ya = gnnm[: gn1.size].nonzero()
                        da = gnnm[: gn1.size].data
                        X = list(np.array(X) + gn1.size)
                    X.extend(xa)
                    Y.extend(ya)
                    D.extend(da)
                    gnnm = sp.sparse.coo_matrix(
                        (D, (X, Y)),
                        shape=(xdim + gnnm.shape[0] - g.size, gnnm.shape[1]),
                    ).T.tocsr()
                g = np.array([ids + "_" + x for x in list(groups.keys())])
                if n == "A":
                    gn1 = g
                else:
                    gn2 = g
                gn = np.append(gn1, gn2)
            else:
                raise TypeError(
                    "Gene groupings ({}) must be in dictionary form.".format(n)
                )
    f = gnnm.sum(1).A.flatten() != 0
    gn = gn[f]
    gn1 = gn1[np.in1d(gn1, gn)]
    gn2 = gn2[np.in1d(gn2, gn)]
    gnnm = gnnm[f, :][:, f]
    return gnnm, gn1, gn2, gn


def prepare_SAMap_loadings(sam, npcs=300):
    """ Prepares SAM object to contain the proper PC loadings associated with its manifold.
    Deposits the loadings in `sam.adata.varm['PCs_SAMap']`.
    
    Parameters
    ----------    
    sam - SAM object
    
    npcs - int, optional, default 300
        The number of PCs to calculate loadings for.
    
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
        weight_mode='dispersion'
    )
    sam.adata.varm["PCs_SAMap"] = A


def _concatenate_sam(sams, nnm, op):

    acns = []
    obsks = []
    exps = []
    agns = []
    for i in range(len(sams)):
        acns.append(q(sams[i].adata.obs_names))
        obsks.append(np.array(sams[i].adata.obs_keys()))
        exps.append(sams[i].adata.X)
        agns.append(q(sams[i].adata.var_names))
    
    obsk = np.unique(np.concatenate(obsks))

    acn = np.concatenate(acns)
    agn = np.concatenate(agns)
    xx = sp.sparse.block_diag(exps,format='csr')
    
    gST = op[:, 0].astype("object") + ";" + op[:, 1].astype("object")

    
    sam = SAM(counts=(xx, agn, acn))
    sam.adata.uns['gene_pairs'] = gST
    
    sam.adata.uns["neighbors"] = {}
    nnm.setdiag(0)
    nnm = nnm.tocsr()
    nnm.eliminate_zeros()
    sam.adata.obsp["connectivities"] = nnm
    sam.adata.uns["neighbors"]["params"] = {
        "n_neighbors": 15,
        "method": "umap",
        "use_rep": "X",
        "metric": "euclidean",
    }

    for k in obsk:
        ann = []
        for i in range(len(sams)):
            if k in sams[i].adata.obs.keys():
                ann.append(
                    sams[i].adata.var_names[0].split("_")[0]
                    + "_"
                    + sams[i].get_labels(k).astype('str').astype("object")
                )
            else:
                ann.append(q([""] * sams[i].adata.shape[0]))

        sam.adata.obs[k] = pd.Categorical(np.concatenate(ann))

    a = []
    for i in range(len(sams)):
        a.extend(["batch" + str(i + 1)] * sams[i].adata.shape[0])
    sam.adata.obs["batch"] = pd.Categorical(np.array(a))
    sam.adata.obs.columns = sam.adata.obs.columns.astype("str")
    sam.adata.var.columns = sam.adata.var.columns.astype("str")
    return sam

def _map_features_un(A, B, sam1, sam2, thr=1e-6):
    i1 = np.where(A.columns == "10")[0][0]
    i3 = np.where(A.columns == "11")[0][0]

    inA = q(A.index)
    inB = q(B.index)

    gn1 = q(sam1.adata.var_names)
    gn2 = q(sam2.adata.var_names)

    gn1 = gn1[np.in1d(gn1, inA)]
    gn2 = gn2[np.in1d(gn2, inB)]

    A = A.iloc[np.in1d(inA, gn1), :]
    B = B.iloc[np.in1d(inB, gn2), :]

    inA2 = q(A.iloc[:, 0])
    inB2 = q(B.iloc[:, 0])

    A = A.iloc[np.in1d(inA2, gn2), :]
    B = B.iloc[np.in1d(inB2, gn1), :]

    gn = np.append(gn1, gn2)
    gnind = pd.DataFrame(data=np.arange(gn.size)[None, :], columns=gn)

    A.index = pd.Index(gnind[A.index].values.flatten())
    B.index = pd.Index(gnind[B.index].values.flatten())
    A.iloc[:, 0] = gnind[A.iloc[:, 0].values.flatten()].values.flatten()
    B.iloc[:, 0] = gnind[B.iloc[:, 0].values.flatten()].values.flatten()

    Arows = np.vstack((A.index, A.iloc[:, 0], A.iloc[:, i3])).T
    Arows = Arows[A.iloc[:, i1].values.flatten() <= thr, :]
    gnnm1 = sp.sparse.lil_matrix((gn.size,) * 2)
    gnnm1[Arows[:, 0].astype("int32"), Arows[:, 1].astype("int32")] = Arows[
        :, 2
    ]  # -np.log10(Arows[:,2]+1e-200)

    Brows = np.vstack((B.index, B.iloc[:, 0], B.iloc[:, i3])).T
    Brows = Brows[B.iloc[:, i1].values.flatten() <= thr, :]
    gnnm2 = sp.sparse.lil_matrix((gn.size,) * 2)
    gnnm2[Brows[:, 0].astype("int32"), Brows[:, 1].astype("int32")] = Brows[
        :, 2
    ]  # -np.log10(Brows[:,2]+1e-200)

    gnnm = (gnnm1 + gnnm2).tocsr()
    gnnms = (gnnm + gnnm.T) / 2
    gnnm.data[:] = 1
    gnnms = gnnms.multiply(gnnm).multiply(gnnm.T).tocsr()
    return gnnms, gn1, gn2


def _filter_gnnm(gnnm, thr=0.25):
    x, y = gnnm.nonzero()
    mas = gnnm.max(1).A.flatten()
    gnnm4 = gnnm.copy()
    gnnm4.data[gnnm4[x, y].A.flatten() < mas[x] * thr] = 0
    gnnm4.eliminate_zeros()
    x, y = gnnm4.nonzero()
    z = gnnm4.data
    gnnm4 = gnnm4.tolil()
    gnnm4[y, x] = z
    gnnm4 = gnnm4.tocsr()
    return gnnm4


def _get_pairs(sam1, sam2, gnnm, gn1, gn2, NOPs1=2, NOPs2=5):
    # gnnm = filter_gnnm(gnnm)
    su = gnnm.max(1).A
    su[su == 0] = 1
    #gnnm = gnnm.multiply(1 / su).tocsr()
    W1 = sam1.adata.var["weights"][gn1].values
    W2 = sam2.adata.var["weights"][gn2].values
    W = np.append(W1, W2)
    W[W < 0.0] = 0
    W[W > 0.0] = 1

    if NOPs1 == 0 and NOPs2 == 0:
        B = gnnm.multiply(W[None, :]).multiply(W[:, None]).tocsr()
        B.eliminate_zeros()
    else:
        B = sparse_knn(
            gnnm.multiply(W[None, :]).multiply(W[:, None]).tocsr(), NOPs1
        ).tocsr()
        B = sparse_knn(B.T, NOPs2).T.tocsr()
        B.eliminate_zeros()

        x, y = B.nonzero()
        data = np.vstack((B[x, y].A.flatten(), B[y, x].A.flatten())).max(0)
        B = sp.sparse.lil_matrix(B.shape)
        B[x, y] = data
        B[y, x] = data
        B = B.tocsr()
    return B


def _avg_as(s):
    return (np.append(
        s.adata.obsp["connectivities"][np.array(s.adata.obs["batch"]) == "batch1", :][
            :, np.array(s.adata.obs["batch"]) == "batch2"
        ]
        .sum(1)
        .A.flatten(),
        s.adata.obsp["connectivities"][np.array(s.adata.obs["batch"]) == "batch2", :][
            :, np.array(s.adata.obs["batch"]) == "batch1"
        ]
        .sum(1)
        .A.flatten()
    )  / s.adata.uns['mdata']['knn_1v2'][0].data.size )


def _parallel_init(ipl1x, isc1x, ipairs, ign1O, ign2O, iT2, iCORR, icorr_mode,icl,ics):
    global pl1
    global sc1
    global p
    global gn1O
    global gn2O
    global T2
    global CORR
    global cs
    global cl
    global corr_mode
    cs = ics
    cl = icl
    pl1 = ipl1x
    sc1 = isc1x
    p = ipairs
    gn1O = ign1O
    gn2O = ign2O
    T2 = iT2
    CORR = iCORR
    corr_mode = icorr_mode

@njit(parallel=True)
def _refine_corr_kernel(filt, p,indptr1,indices1,data1,indptr2,indices2,data2,n1,n2):
    p1 = p[:,0]
    p2 = p[:,1]
    res = np.zeros(p1.size)
    
    for j in prange(len(p1)):
        j1, j2 = p1[j], p2[j]

        pl1d = data1[indptr1[j1] : indptr1[j1 + 1]]
        pl1i = indices1[indptr1[j1] : indptr1[j1 + 1]]

        sc1d = data2[indptr2[j2] : indptr2[j2 + 1]]
        sc1i = indices2[indptr2[j2] : indptr2[j2 + 1]]

        x = np.zeros(n1)
        x[pl1i] = pl1d
        y = np.zeros(n2)
        y[sc1i] = sc1d
        iz = np.logical_or(x>0,y>0)
        izf = np.logical_and(x>0,y>0)

        if izf.sum()>0:
            x=x[iz]
            y=y[iz]
            res[j] = ((x-x.mean())*(y-y.mean()) / x.std() / y.std())[filt[iz]].sum() / x.size
        else:
            res[j] = 0
            
    return res
            
def _xicorr(X,Y):
    n = X.size
    xi = np.argsort(X,kind='quicksort')
    Y = Y[xi]
    _,b,c = np.unique(Y,return_counts=True,return_inverse=True)
    r = np.cumsum(c)[b]
    _,b,c = np.unique(-Y,return_counts=True,return_inverse=True)
    l = np.cumsum(c)[b]
    denominator = (2*(l*(n-l)).sum())
    if denominator > 0:
        return 1 - n*np.abs(np.diff(r)).sum() / denominator
    else:
        return 0        

def _refine_corr_parallel(
    sam1,
    sam2,
    st,
    gnnm,
    gn1,
    gn2,
    corr_mode="pearson",
    THR=0,
    use_seq=False,
    T1=0.0,
    T2=0.0,
    ncpus=os.cpu_count(),
    ct_labels=None    
):

    import scipy as sp

    gn = np.append(gn1, gn2)

    w1 = sam1.adata.var["weights"][gn1].values
    w2 = sam2.adata.var["weights"][gn2].values
    w = np.append(w1, w2)

    w[w > T1] = 1
    w[w < 1] = 0
    ix = np.array(["a"] * gn1.size + ["b"] * gn2.size)
    gnO = gn[w > 0]
    ix = ix[w > 0]
    gn1O = gnO[ix == "a"]
    gn2O = gnO[ix == "b"]  #
    gnnmO = gnnm[w > 0, :][:, w > 0]
    x, y = gnnmO.nonzero()
    pairs = np.unique(np.sort(np.vstack((x, y)).T, axis=1), axis=0)
    pairs[pairs >= gn1O.size] = pairs[pairs >= gn1O.size] - gn1O.size

    idx1 = np.where(st.adata.obs["batch"] == "batch1")[0]
    idx2 = np.where(st.adata.obs["batch"] == "batch2")[0]
    nnm = st.adata.obsp["connectivities"]
    x1 = sam1.adata[:, gn1O].X.tocsc().astype("float16")  # [:,pairs[:,0]]
    x2 = sam2.adata[:, gn2O].X.tocsc().astype("float16")  # [:,pairs[:,1]]

    nnm1 = nnm[:, idx1].astype("float16")
    nnm2 = nnm[:, idx2].astype("float16")

    s1 = nnm1.sum(1).A
    s1[s1 < 1e-3] = 1
    s1 = s1.flatten()[:, None]
    s2 = nnm2.sum(1).A
    s2[s2 < 1e-3] = 1
    s2 = s2.flatten()[:, None]

    pl1x = nnm1.dot(x1).multiply(1 / s1).tocsc()

    sc1x = nnm2.dot(x2).multiply(1 / s2).tocsc()

    p = pairs
    pl1 = pl1x
    sc1 = sc1x
    
    if corr_mode == 'mutual_info' or corr_mode == "xicorr":
        from multiprocessing import Pool, Manager
        CORR = Manager().dict()
        pc_chunksize = pl1.shape[1] // ncpus + 1
        if ct_labels is None:
            cl=cs=None
        else:
            cl,cs = ct_labels
            if not (isinstance(cs,list) or isinstance(cs,tuple)):
                cs = [cs]           
            if np.in1d(cl,cs).sum()==0:
                raise ValueError('Cell types not found in provided labels.')
                
        pool = Pool(
            ncpus, _parallel_init, [pl1, sc1, p, gn1O, gn2O, T2, CORR, corr_mode, cl, cs]
        )
        try:
            pool.map(_parallel_wrapper, range(p.shape[0]), chunksize=pc_chunksize)
        finally:
            pool.close()
            pool.join()
        CORR = CORR._getvalue()
    else:
        if ct_labels is not None:
            cl,cs = ct_labels
            if not (isinstance(cs,list) or isinstance(cs,tuple)):
                cs = [cs]                      
            filt = np.in1d(cl,cs)
            if filt.sum()==0:
                raise ValueError('Cell types not found in provided labels.')
                
        else:
            filt = np.array([True]*pl1.shape[0])       
        vals = _refine_corr_kernel(filt, p,pl1.indptr,pl1.indices,pl1.data,sc1.indptr,sc1.indices,sc1.data,pl1.shape[0],sc1.shape[0])
        CORR = dict(zip(to_vn(np.vstack((gn1O[p[:,0]],gn2O[p[:,1]])).T),vals))
    
    
    for k in CORR.keys():
        weight1 = sam1.adata.var["weights"][k.split(';')[0]]
        weight2 = sam2.adata.var["weights"][k.split(';')[1]]
        CORR[k] = 0 if CORR[k] < THR else CORR[k]
        CORR[k] = np.sqrt(CORR[k] * np.sqrt(weight1 * weight2))

    gnnm2 = gnnm.multiply(w[:, None]).multiply(w[None, :]).tocsr()
    x, y = gnnm2.nonzero()
    pairs = np.unique(np.sort(np.vstack((x, y)).T, axis=1), axis=0)
    CORR = np.array([CORR[x] for x in to_vn(gn[pairs])])

    
    gnnm3 = sp.sparse.lil_matrix(gnnm.shape)

    if use_seq:
        gnnm3[pairs[:, 0], pairs[:, 1]] = (
            CORR * gnnm2[pairs[:, 0], pairs[:, 1]].A.flatten()
        )
        gnnm3[pairs[:, 1], pairs[:, 0]] = (
            CORR * gnnm2[pairs[:, 1], pairs[:, 0]].A.flatten()
        )
    else:
        gnnm3[pairs[:, 0], pairs[:, 1]] = CORR  # *gnnm2[x,y].A.flatten()
        gnnm3[pairs[:, 1], pairs[:, 0]] = CORR  # *gnnm2[x,y].A.flatten()

    gnnm3 = gnnm3.tocsr()
    gnnm3.eliminate_zeros()

    return gnnm3, CORR

try:
    from fast_histogram import histogram2d
except:
    pass;

def hist2d(X,Y,bins=100,domain=None):
    if domain is None:
        xmin = X.min()
        xmax = X.max()
        ymin=Y.min()
        ymax=Y.max()    
        domain = [(xmin,xmax),(ymin,ymax)]
    return histogram2d(X,Y,bins,domain)

def calc_MI(X,Y,bins=100,cl=None,cs=None):    
    xmin = X.min()
    xmax = X.max()
    ymin=Y.min()
    ymax=Y.max()    
    domain = [(xmin,xmax),(ymin,ymax)]
    c_XY = hist2d(X,Y,bins=bins,domain=domain)
    c_X = c_XY.sum(1)
    c_Y = c_XY.sum(0)
    
    c = c_XY
    c_normalized = c / c.sum()
    c1 = c_normalized.sum(1)
    c2 = c_normalized.sum(0)
    c1[c1==0]=1
    c2[c2==0]=1
    c_normalized[c_normalized==0]=1
    H_X = -(c1*np.log2(c1)).sum()
    H_Y = -(c2*np.log2(c2)).sum()
    H_XY = -(c_normalized*np.log2(c_normalized)).sum()
    
    H_Y = 0 if H_Y < 0 else H_Y
    H_X = 0 if H_X < 0 else H_X
    H_XY = 0 if H_XY < 0 else H_XY

    MI = H_X + H_Y - H_XY
    if MI <= 0 or H_X <= 0 or H_Y <= 0:
        return 0
    else:
        return MI / np.sqrt(H_X*H_Y)

def _parallel_wrapper(j):
    j1, j2 = p[j, 0], p[j, 1]

    pl1d = pl1.data[pl1.indptr[j1] : pl1.indptr[j1 + 1]]
    pl1i = pl1.indices[pl1.indptr[j1] : pl1.indptr[j1 + 1]]

    sc1d = sc1.data[sc1.indptr[j2] : sc1.indptr[j2 + 1]]
    sc1i = sc1.indices[sc1.indptr[j2] : sc1.indptr[j2 + 1]]

    x = np.zeros(pl1.shape[0])
    x[pl1i] = pl1d
    y = np.zeros(sc1.shape[0])
    y[sc1i] = sc1d

    ha = gn1O[j1] + ";" + gn2O[j2]

    try:
        if corr_mode == 'xicorr':
            CORR[ha] = _xicorr(x,y)
        elif corr_mode == 'mutual_info':
            CORR[ha] = calc_MI(x,y,cl=cl,cs=cs)
        else:
            raise ValueError(f'`{corr_mode}` not recognized.')
    except:
        CORR[ha] = 0


def _united_proj(wpca1, wpca2, k=20, metric="correlation", sigma=500, ef=200, M=48):

    print("Running hsnwlib")

    labels1 = np.arange(wpca1.shape[0])
    labels2 = np.arange(wpca2.shape[0])

    p1 = hnswlib.Index(space="cosine", dim=wpca1.shape[1])
    p2 = hnswlib.Index(space="cosine", dim=wpca2.shape[1])

    p1.init_index(max_elements=wpca1.shape[0], ef_construction=ef, M=M)
    p2.init_index(max_elements=wpca2.shape[0], ef_construction=ef, M=M)

    p1.add_items(wpca1, labels1)
    p2.add_items(wpca2, labels2)

    p1.set_ef(ef)
    p2.set_ef(ef)

    idx2, dist2 = p1.knn_query(wpca2, k=k)
    idx1, dist1 = p2.knn_query(wpca1, k=k)

    dist2 = 1 - dist2
    dist1 = 1 - dist1

    dist1[dist1 < 0] = 0
    dist2[dist2 < 0] = 0
    Dist1 = dist1  # np.exp(-1*(1-dist1)**2)
    Dist2 = dist2  # np.exp(-1*(1-dist2)**2)

    knn1v2 = sp.sparse.lil_matrix((wpca1.shape[0], wpca2.shape[0]))
    knn2v1 = sp.sparse.lil_matrix((wpca2.shape[0], wpca1.shape[0]))

    x1 = np.tile(np.arange(idx1.shape[0])[:, None], (1, idx1.shape[1])).flatten()
    x2 = np.tile(np.arange(idx2.shape[0])[:, None], (1, idx2.shape[1])).flatten()
    knn1v2[x1, idx1.flatten()] = Dist1.flatten()
    knn2v1[x2, idx2.flatten()] = Dist2.flatten()

    return knn1v2.tocsr(), knn2v1.tocsr()


def _mapping_window(sam1, sam2, gnnm, gn, K=20):

    ix = np.unique(np.sort(np.vstack((gnnm.nonzero())).T, axis=1), axis=0)
    ortholog_pairs = gn[ix]
    print("Found", ortholog_pairs.shape[0], "gene pairs")
    corr = gnnm[ix[:, 0], ix[:, 1]].A.flatten()
    corr = 0.5 + 0.5 * np.tanh(10 * (corr - 0.5))

    gns1 = ortholog_pairs[:, 0]
    gns2 = ortholog_pairs[:, 1]

    g1 = q(sam1.adata.var_names)
    g2 = q(sam2.adata.var_names)

    g1 = g1[np.in1d(g1, gns1)]
    g2 = g2[np.in1d(g2, gns2)]

    adata1 = sam1.adata[:, g1]
    adata2 = sam2.adata[:, g2]

    W1 = adata1.var["weights"].values[None,:]
    W2 = adata2.var["weights"].values[None,:]

    std = StandardScaler(with_mean=False)

    s1 = std.fit_transform(adata1.X)#.multiply(W1[None, :]).tocsr()
    s2 = std.fit_transform(adata2.X)#.multiply(W2[None, :]).tocsr()

    k = K

    A1 = pd.DataFrame(data=np.arange(g1.size)[None, :], columns=g1)
    A2 = pd.DataFrame(data=np.arange(g2.size)[None, :], columns=g2)

    G1 = A1[gns1].values.flatten()
    G2 = A2[gns2].values.flatten()

    avg = sp.sparse.lil_matrix((g1.size, g2.size))
    avg[G1, G2] = corr
    su1, su2 = avg.sum(1).A, avg.sum(0).A
    avg1 = avg.multiply(1 / su1).tocsr()
    avg2 = avg.multiply(1 / su2).tocsr()

    sp1 = s1.dot(avg2)
    sp2 = s2.dot(avg1.T)

    sp1 = std.fit_transform(sp1)
    sp2 = std.fit_transform(sp2)
    
    s1 = s1.multiply(W1).tocsr()
    sp1 = sp1.multiply(W2).tocsr()
    s2 = s2.multiply(W2).tocsr()
    sp2 = sp2.multiply(W1).tocsr()

    mu1 = s1.mean(0).A.flatten()[None, :]
    mu2 = s2.mean(0).A.flatten()[None, :]
    mu1s = sp1.mean(0).A.flatten()[None, :]
    mu2s = sp2.mean(0).A.flatten()[None, :]

    C1 = sam1.adata[:, g1].varm["PCs_SAMap"].T
    C2 = sam2.adata[:, g2].varm["PCs_SAMap"].T

    print("Recomputing PC projections with gene pair subsets...")
    ws1 = s1.dot(C1.T) - mu1.dot(C1.T)
    ws2 = s2.dot(C2.T) - mu2.dot(C2.T)
    wsp1 = sp1.dot(C2.T) - C2.dot(mu1s.T).T
    wsp2 = sp2.dot(C1.T) - C1.dot(mu2s.T).T
    wpca = np.hstack((np.vstack((ws1, wsp2)), np.vstack((wsp1, ws2))))

    wpca1 = wpca[: s1.shape[0], :]
    wpca2 = wpca[s1.shape[0] :, :]

    b1, b2 = _united_proj(wpca1, wpca2, k=k)
    
    output_dict = {}
    output_dict["translated_data1"] = sp.sparse.hstack((s1,sp1)).tocsr()
    output_dict["translated_data2"] = sp.sparse.hstack((sp2,s2)).tocsr()    
    output_dict["knn_1v2"] = b1.tocsr()
    output_dict["knn_2v1"] = b2.tocsr()
    output_dict["wPCA1"] = wpca1
    output_dict["wPCA2"] = wpca2
    output_dict["pca1"] = C1
    output_dict["pca2"] = C2
    output_dict["corr"] = corr
    output_dict["ortholog_pairs"] = ortholog_pairs
    output_dict["G_avg1"] = avg1.tocsr()
    output_dict["G_avg2"] = avg2.tocsr()
    output_dict["G_avg"] = avg.tocsr()
    output_dict["edge_weights"] = corr
    return output_dict


def _sparse_knn_ks(D, ks):
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


def _smart_expand(nnm, cl, NH=3):
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

    a, ix, c = np.unique(cl, return_counts=True, return_inverse=True)
    K = c[ix]

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
