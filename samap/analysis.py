import sklearn.utils.sparsefuncs as sf
from . import q, ut, pd, sp, np, warnings, sc
from .utils import to_vo, to_vn, substr, df_to_dict, sparse_knn, prepend_var_prefix

from scipy.stats import rankdata


def _log_factorial(n):
    return np.log(np.arange(1,n+1)).sum()
def _log_binomial(n,k):
    return _log_factorial(n) - (_log_factorial(k) + _log_factorial(n-k))

def GOEA(target_genes,GENE_SETS,df_key='GO',goterms=None,fdr_thresh=0.25,p_thresh=1e-3): 
    """Performs GO term Enrichment Analysis using the hypergeometric distribution.
    
    Parameters
    ----------
    target_genes - array-like
        List of target genes from which to find enriched GO terms.
    GENE_SETS - dictionary or pandas.DataFrame
        Dictionary where the keys are GO terms and the values are lists of genes associated with each GO term.
        Ex: {'GO:0000001': ['GENE_A','GENE_B'],
             'GO:0000002': ['GENE_A','GENE_C','GENE_D']}
        Make sure to include all available genes that have GO terms in your dataset.
        
        ---OR---
        
        Pandas DataFrame with genes as the index and GO terms values.
        Ex: 'GENE_A','GO:0000001',
            'GENE_A','GO:0000002',
            'GENE_B','GO:0000001',
            'GENE_B','GO:0000004',
            ...
        If `GENE_SETS` is a pandas DataFrame, the `df_key` parameter should be the name of the column in which
        the GO terms are stored.       
    df_key - str, optional, default 'GO'
        The name of the column in which GO terms are stored. Only used if `GENE_SETS` is a DataFrame.
    goterms - array-list, optional, default None
        If provided, only these GO terms will be tested.
    fdr_thresh - float, optional, default 0.25
        Filter out GO terms with FDR q value greater than this threshold.
    p_thresh - float, optional, default 1e-3
        Filter out GO terms with p value greater than this threshold.
        
    Returns:
    -------
    enriched_goterms - pandas.DataFrame
        A Pandas DataFrame of enriched GO terms with FDR q values, p values, and associated genes provided.
    """    
    
    # identify all genes found in `GENE_SETS`
    
    if isinstance(GENE_SETS,pd.DataFrame):
        print('Converting DataFrame into dictionary')
        genes = np.array(list(GENE_SETS.index))
        agt = np.array(list(GENE_SETS[df_key].values))
        idx = np.argsort(agt)
        genes = genes[idx]
        agt = agt[idx]
        bounds = np.where(agt[:-1]!=agt[1:])[0]+1
        bounds = np.append(np.append(0,bounds),agt.size)
        bounds_left=bounds[:-1]
        bounds_right=bounds[1:]
        genes_lists = [genes[bounds_left[i]:bounds_right[i]] for i in range(bounds_left.size)]
        GENE_SETS = dict(zip(np.unique(agt),genes_lists))
    all_genes = np.unique(np.concatenate(list(GENE_SETS.values())))
    all_genes = np.array(all_genes)
    
    # if goterms is None, use all the goterms found in `GENE_SETS`
    if goterms is None:
        goterms = np.unique(list(GENE_SETS.keys()))
    else:
        goterms = goterms[np.in1d(goterms,np.unique(list(GENE_SETS.keys())))]
    
    # ensure that target genes are all present in `all_genes`
    _,ix = np.unique(target_genes,return_index=True)
    target_genes=target_genes[np.sort(ix)]
    target_genes = target_genes[np.in1d(target_genes,all_genes)]
    
    # N -- total number of genes
    N = all_genes.size

    probs=[]
    probs_genes=[]
    counter=0
    # for each go term,
    for goterm in goterms:
        if counter%1000==0:
            pass; #print(counter)
        counter+=1
        
        # identify genes associated with this go term
        gene_set = np.array(GENE_SETS[goterm])
        
        # B -- number of genes associated with this go term
        B = gene_set.size
        
        # b -- number of genes in target associated with this go term
        gene_set_in_target = gene_set[np.in1d(gene_set,target_genes)]
        b = gene_set_in_target.size        
        if b != 0:
            # calculate the enrichment probability as the cumulative sum of the tail end of a hypergeometric distribution
            # with parameters (N,B,n,b)
            n = target_genes.size
            num_iter = min(n,B)
            rng = np.arange(b,num_iter+1)
            probs.append(sum([np.exp(_log_binomial(n,i)+_log_binomial(N-n,B-i) - _log_binomial(N,B)) for i in rng]))
        else:
            probs.append(1.0)
        
        #append associated genes to a list
        probs_genes.append(gene_set_in_target)
        
    probs = np.array(probs)    
    probs_genes = np.array([';'.join(x) for x in probs_genes])
    
    # adjust p value to correct for multiple testing
    fdr_q_probs = probs.size*probs / rankdata(probs,method='ordinal')
    
    # filter out go terms based on the FDR q value and p value thresholds
    filt = np.logical_and(fdr_q_probs<fdr_thresh,probs<p_thresh)
    enriched_goterms = goterms[filt]
    p_values = probs[filt]
    fdr_q_probs = fdr_q_probs[filt]    
    probs_genes=probs_genes[filt]
    
    # construct the Pandas DataFrame
    gns = probs_genes
    enriched_goterms = pd.DataFrame(data=fdr_q_probs,index=enriched_goterms,columns=['fdr_q_value'])
    enriched_goterms['p_value'] = p_values
    enriched_goterms['genes'] = gns
    
    # sort in ascending order by the p value
    enriched_goterms = enriched_goterms.sort_values('p_value')   
    return enriched_goterms

_KOG_TABLE = dict(A = "RNA processing and mofiication",
                 B = "Chromatin structure and dynamics",
                 C = "Energy production and conversion",
                 D = "Cell cycle control, cell division, chromosome partitioning",
                 E = "Amino acid transport and metabolism",
                 F = "Nucleotide transport and metabolism",
                 G = "Carbohydrate transport and metabolism",
                 H = "Coenzyme transport and metabolism",
                 I = "Lipid transport and metabolism",
                 J = "Translation, ribosomal structure and biogenesis",
                 K = "Transcription",
                 L = "Replication, recombination, and repair",
                 M = "Cell wall membrane/envelope biogenesis",
                 N = "Cell motility",
                 O = "Post-translational modification, protein turnover, chaperones",
                 P = "Inorganic ion transport and metabolism",
                 Q = "Secondary metabolites biosynthesis, transport and catabolism",
                 R = "General function prediction only",
                 S = "Function unknown",
                 T = "Signal transduction mechanisms",
                 U = "Intracellular trafficking, secretion, and vesicular transport",
                 V = "Defense mechanisms",
                 W = "Extracellular structures",
                 Y = "Nuclear structure",
                 Z = "Cytoskeleton")

import gc
from collections.abc import Iterable
class FunctionalEnrichment(object):    
    def __init__(self,sms, DFS, col_key, keys, delimiter = '', align_thr = 0.1, limit_reference = False, n_top = 0):
        """Performs functional enrichment analysis on gene pairs enriched
        in mapped cell types using functional annotations output by Eggnog.
        
        Parameters
        ----------
        sms - list or tuple of SAMAP objects
        
        DFS - list or tuple of pandas.DataFrame functional annotations (one per species present in the input SAMAP objects)
        
        col_key - str
            The column name with functional annotations in the annotation DataFrames.
        
        keys - list or tuple of column keys from `.adata.obs` DataFrames (one per species present in the input SAMAP objects)
            Cell type mappings will be computed between these annotation vectors.
                        
        delimiter - str, optional, default ''
            Some transcripts may have multiple functional annotations (e.g. GO terms or KOG terms) separated by
            a delimiter. For KOG terms, this is typically no delimiter (''). For GO terms, this is usually a comma
            (',').
            
        align_thr - float, optional, default 0.1
            The alignment score below which to filter out cell type mappings
            
        limit_reference - bool, optional, default False
            If True, limits the background set of genes to include only those that are enriched in any cell type mappings
            If False, the background set of genes will include all genes present in the input dataframes.
            
        n_top: int, optional, default 0
            If `n_top` is 0, average the alignment scores for all cells in a pair of clusters.
            Otherwise, average the alignment scores of the top `n_top` cells in a pair of clusters.
            Set this to non-zero if you suspect there to be subpopulations of your cell types mapping
            to distinct cell types in the other species.

        """
        # get dictionary of sam objects
        if not isinstance(sms,Iterable):
            sms = [sms]
        if not isinstance(DFS,Iterable):
            DFS = [DFS]
        if not isinstance(keys,Iterable):
            keys = [keys]

        SAMS={}
        for sm in sms:
            SAMS[sm.id1]=sm.sam1
            SAMS[sm.id2]=sm.sam2
        
        # link up SAM memories.
        for sm in sms:
            sm.sam1 = SAMS[sm.id1]
            sm.sam2 = SAMS[sm.id2]
            gc.collect()
            
        # figure out which species corresponds to which EGGNOG table
        keys2 = {}
        for i in range(len(DFS)):
            DFS[i] = DFS[i].copy()
            
            genes = q(DFS[i].index)
            overlap=[]
            ks = list(SAMS.keys())
            for k in ks:
                overlap.append(np.in1d(genes,['_'.join(x.split('_')[1:]) for x in SAMS[k].adata.var_names]).mean())
            k = ks[np.array(overlap).argmax()]
            DFS[i].index = k+'_'+DFS[i].index
            keys2[k] = keys[i]
        keys = keys2    
        # concatenate DFS
        A = pd.concat(DFS,axis=0)
        RES = pd.DataFrame(A[col_key])
        RES.columns=['GO']    
        RES = RES[(q(RES.values.flatten())!='nan')]
        
        # EXPAND RES
        data = []
        index = []
        for i in range(RES.shape[0]):
            if delimiter == '':
                l = list(RES.values[i][0])
                l = np.array([str(x) if str(x).isalpha() else '' for x in l])
                l = l[l!= '']
                l = list(l)
            else:
                l = RES.values[i][0].split(delimiter)
                
            data.extend(l)
            index.extend([RES.index[i]]*len(l))
        
        RES = pd.DataFrame(index = index,data = data,columns = ['GO'])
        
        genes = np.array(list(RES.index))
        agt = np.array(list(RES['GO'].values))
        idx = np.argsort(agt)
        genes = genes[idx]
        agt = agt[idx]
        bounds = np.where(agt[:-1]!=agt[1:])[0]+1
        bounds = np.append(np.append(0,bounds),agt.size)
        bounds_left=bounds[:-1]
        bounds_right=bounds[1:]
        genes_lists = [genes[bounds_left[i]:bounds_right[i]] for i in range(bounds_left.size)]
        GENE_SETS = dict(zip(np.unique(agt),genes_lists)) 
        for cc in GENE_SETS.keys():
            GENE_SETS[cc]=np.unique(GENE_SETS[cc])
        
        G = []
        for sm in sms:
            print(f'Finding enriched gene pairs between {sm.id1} and {sm.id2}...')
            gpf = GenePairFinder(sm,k1=keys[sm.id1],k2=keys[sm.id2])
            gene_pairs = gpf.find_all(thr=align_thr,n_top=n_top)   
            G.append(gene_pairs)
        gene_pairs = pd.concat(G,axis=1)
        
        self.DICT = {}
        for c in gene_pairs.columns:
            x = q(gene_pairs[c].values.flatten()).astype('str')
            ff = x!='nan'
            if ff.sum()>0:
                self.DICT[c] = x[ff]

        if limit_reference:
            all_genes = np.unique(np.concatenate(substr(np.concatenate(list(self.DICT.values())),';')))
        else:
            all_genes = np.unique(np.array(list(A.index)))
        
        for d in GENE_SETS.keys():
            GENE_SETS[d] = GENE_SETS[d][np.in1d(GENE_SETS[d],all_genes)]

        self.gene_pairs = gene_pairs
        self.CAT_NAMES = np.unique(q(RES['GO']))
        self.GENE_SETS = GENE_SETS
        self.RES = RES
        
    def calculate_enrichment(self,verbose=False):
        """ Calculates the functional enrichment.
        
        Parameters
        ----------
        verbose - bool, optional, default False
            If False, function does not log progress to output console.
            
        Returns
        -------
        ENRICHMENT_SCORES - pandas.DataFrame (cell types x function categories)
            Enrichment scores (-log10 p-value) for each function in each cell type.
        
        NUM_ENRICHED_GENES - pandas.DataFrame (cell types x function categories)
            Number of enriched genes for each function in each cell type.        
        
        ENRICHED_GENES - pandas.DataFrame (cell types x function categories)
            The IDs of enriched genes for each function in each cell type.
        """
        DICT = self.DICT
        RES = self.RES
        CAT_NAMES = self.CAT_NAMES
        GENE_SETS = self.GENE_SETS
        pairs = np.array(list(DICT.keys()))
        all_nodes = np.unique(np.concatenate(substr(pairs,';')))

        CCG={}
        P=[]
        for ik in range(len(all_nodes)):
            genes=[]
            nodes = all_nodes[ik]
            for j in range(len(pairs)):
                n1,n2 = pairs[j].split(';')
                if n1 == nodes or n2 == nodes:
                    g1,g2 = substr(DICT[pairs[j]],';')
                    genes.append(np.append(g1,g2))
            if len(genes) > 0:
                genes = np.concatenate(genes)
                genes = np.unique(genes)
            else:
                genes = np.array([])
            CCG[all_nodes[ik]] = genes

        HM = np.zeros((len(CAT_NAMES),len(all_nodes)))
        HMe = np.zeros((len(CAT_NAMES),len(all_nodes)))
        HMg = np.zeros((len(CAT_NAMES),len(all_nodes)),dtype='object')
        for ii,cln in enumerate(all_nodes):
            if verbose:
                print(f'Calculating functional enrichment for cell type {cln}')
                
            g = CCG[cln]    

            if g.size > 0:
                gi = g[np.in1d(g,q(RES.index))]
                ix = np.where(np.in1d(q(RES.index),gi))[0]
                res = RES.iloc[ix]
                goterms = np.unique(q(res['GO']))
                goterms = goterms[goterms!='S']  
                result = GOEA(gi,GENE_SETS,goterms=goterms,fdr_thresh=100,p_thresh=100)

                lens = np.array([len(np.unique(x.split(';'))) for x in result['genes'].values])
                F = -np.log10(result['p_value'])
                gt,vals = F.index,F.values
                Z = pd.DataFrame(data=np.arange(CAT_NAMES.size)[None,:],columns=CAT_NAMES)
                if gt.size>0:
                    HM[Z[gt].values.flatten(),ii] = vals
                    HMe[Z[gt].values.flatten(),ii] = lens
                    HMg[Z[gt].values.flatten(),ii] = [';'.join(np.unique(x.split(';'))) for x in result['genes'].values]

        #CAT_NAMES = [_KOG_TABLE[x] for x in CAT_NAMES]
        SC = pd.DataFrame(data = HM,index=CAT_NAMES,columns=all_nodes).T
        SCe = pd.DataFrame(data = HMe,index=CAT_NAMES,columns=all_nodes).T
        SCg = pd.DataFrame(data = HMg,index=CAT_NAMES,columns=all_nodes).T
        SCg.values[SCg.values==0]=''
        
        self.ENRICHMENT_SCORES = SC
        self.NUM_ENRICHED_GENES = SCe
        self.ENRICHED_GENES = SCg
        
        return self.ENRICHMENT_SCORES,self.NUM_ENRICHED_GENES,self.ENRICHED_GENES
    
    def plot_enrichment(self,cell_types = [], pval_thr=2.0,msize = 50):
        """Create a plot summarizing the functional enrichment analysis.
        
        Parameters
        ----------
        cell_types - list, default []
            A list of cell types for which enrichment scores will be plotted. If empty (default),
            all cell types will be plotted.
            
        pval_thr - float, default 2.0
            -log10 p-values < 2.0 will be filtered from the plot.
            
        msize - float, default 50
            The marker size in pixels for the dot plot.
        
        Returns
        -------
        fig - matplotlib.pyplot.Figure
        ax - matplotlib.pyplot.Axes
        """
        import colorsys
        import seaborn as sns
        import matplotlib
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Rectangle
        from matplotlib import cm,colors
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import linkage, dendrogram

        SC = self.ENRICHMENT_SCORES
        SCe = self.NUM_ENRICHED_GENES
        SCg = self.ENRICHED_GENES

        if len(cell_types) > 0:
            SC = SC.T[cell_types].T
            SCe = SCe.T[cell_types].T
            SCg = SCg.T[cell_types].T

        CAT_NAMES = self.CAT_NAMES
        gc_names = np.array(CAT_NAMES)

        SC.values[SC.values<pval_thr]=0
        SCe.values[SC.values<pval_thr]=0
        SCg.values[SC.values<pval_thr]=''
        SCg=SCg.astype('str')
        SCg.values[SCg.values=='nan']=''

        ixrow = np.array(dendrogram(linkage(SC.values.T,method='ward',metric='euclidean'),no_plot=True)['ivl']).astype('int')
        ixcol = np.array(dendrogram(linkage(SC.values,method='ward',metric='euclidean'),no_plot=True)['ivl']).astype('int')    

        SC = SC.iloc[ixcol].iloc[:,ixrow]
        SCe = SCe.iloc[ixcol].iloc[:,ixrow]
        SCg = SCg.iloc[ixcol].iloc[:,ixrow]


        SCgx = SCg.values.copy()


        for i in range(SCgx.shape[0]):
            idn = SCg.index[i].split('_')[0]

            for j in range(SCgx.shape[1]):
                genes = np.array(SCgx[i,j].split(';'))    
                SCgx[i,j] = ';'.join(genes[np.array([x.split('_')[0] for x in genes]) == idn])


        x,y=np.tile(np.arange(SC.shape[0]),SC.shape[1]),np.repeat(np.arange(SC.shape[1]),SC.shape[0])    
        co = SC.values[x,y].flatten()#**0.5
        ms = SCe.values[x,y].flatten()
        ms=ms/ms.max()
        x=x.max()-x #
        ms = ms*msize
        ms[np.logical_and(ms<0.15,ms>0)]=0.15

        fig,ax = plt.subplots();
        fig.set_size_inches((7*SC.shape[0]/SC.shape[1],7)) 

        scat=ax.scatter(x,y,c=co,s=ms,cmap='seismic',edgecolor='k',linewidth=0.5,vmin=3)
        cax = fig.colorbar(scat,pad=0.02);
        ax.set_yticks(np.arange(SC.shape[1]))
        ax.set_yticklabels(SC.columns,ha='right',rotation=0)
        ax.set_xticks(np.arange(SC.shape[0]))
        ax.set_xticklabels(SC.index[::-1],ha='right',rotation=45)
        ax.invert_yaxis()
        ax.invert_xaxis()
        #ax.figure.tight_layout()
        return fig,ax
    
def sankey_plot(M,align_thr=0.1):
    """Generate a sankey plot
    
    Parameters
    ----------
    M: pandas.DataFrame
        Mapping table output from `get_mapping_scores` (third output).

    align_thr: float, optional, default 0.1
        The alignment score threshold below which to remove cell type mappings.
    """    
    id1 = M.index[0].split('_')[0]
    id2 = M.columns[0].split('_')[0]
    d = M.values.copy()
    d[d<align_thr]=0
    x,y = d.nonzero()
    values = d[x,y]
    y = y + M.index.size
    nodes = np.append(q(M.index),q(M.columns))
    xPos = [0]*M.index.size + [1]*M.columns.size


    R = pd.DataFrame(data = nodes[np.vstack((x,y))].T,columns=['source','target'])
    R['Value'] = values
    
    try:
        from holoviews import dim
        from bokeh.models import Label
        import holoviews as hv
        hv.extension('bokeh',logo=False)
    except:
        raise ImportError('Please install holoviews with `!pip install holoviews`.')

    def f(plot,element):
        plot.handles['plot'].sizing_mode='scale_width'    
        plot.handles['plot'].x_range.start = -600    
        plot.handles['plot'].add_layout(Label(x=plot.handles['plot'].x_range.end*0.78, y=plot.handles['plot'].y_range.end*0.96, text=id2))
        plot.handles['plot'].x_range.end = 1500    
        plot.handles['plot'].add_layout(Label(x=0, y=plot.handles['plot'].y_range.end*0.96, text=id1))

    sankey1 = hv.Sankey(R, kdims=["source", "target"], vdims=["Value"])


    sankey1.opts(cmap='Colorblind',label_position='outer', edge_line_width=0, show_values=False,
                                     node_alpha=1.0, node_width=40, node_sort=True,frame_height=1000,frame_width=800,
                                     bgcolor="snow",apply_ranges = True,hooks=[f])

    return sankey1



class GenePairFinder(object):
    def __init__(self, sm, k1="leiden_clusters",
                 k2="leiden_clusters"):
        """Find enriched gene pairs in cell type mappings.
        
        sm: SAMAP object

        k1 & k2: str, optional, default 'leiden_clusers'
            Keys corresponding to the annotation vector in `s1.adata.obs` and `s2.adata.obs`.

        """
        self.sm = sm
        self.s1 = sm.sam1
        self.s2 = sm.sam2
        self.s3 = sm.samap

        self.id1 = sm.id1
        self.id2 = sm.id2

        prepend_var_prefix(self.s1, self.id1)
        prepend_var_prefix(self.s2, self.id2)

        self.s1.adata.obs[k1] = self.s1.adata.obs[k1].astype("str")
        self.s2.adata.obs[k2] = self.s2.adata.obs[k2].astype("str")

        mu1, v1, mu2, v2 = _get_mu_std(self.s3, self.s1, self.s2)
        self.mu1 = mu1
        self.v1 = v1
        self.mu2 = mu2
        self.v2 = v2
        self.k1 = k1
        self.k2 = k2

        self.find_markers()

    def find_markers(self):
        print(
            "Finding cluster-specific markers in {}:{} and {}:{}.".format(
                self.id1, self.k1, self.id2, self.k2
            )
        )        
        import gc
        if self.k1+'_scores' not in self.s1.adata.varm.keys():
            find_cluster_markers(self.s1, self.k1)
            gc.collect()
            
        if self.k2+'_scores' not in self.s2.adata.varm.keys():
            find_cluster_markers(self.s2, self.k2)
            gc.collect()
        
    def find_all(self,thr=0.1,n_top=0,**kwargs):
        """Find enriched gene pairs in all pairs of mapped cell types.
        
        Parameters
        ----------
        thr: float, optional, default 0.2
            Alignment score threshold above which to consider cell type pairs mapped.
        
        n_top: int, optional, default 0
            If `n_top` is 0, average the alignment scores for all cells in a pair of clusters.
            Otherwise, average the alignment scores of the top `n_top` cells in a pair of clusters.
            Set this to non-zero if you suspect there to be subpopulations of your cell types mapping
            to distinct cell types in the other species.        
            
        Keyword arguments
        -----------------
        Keyword arguments to `find_genes` accepted here.
        
        Returns
        -------
        Table of enriched gene pairs for each cell type pair
        """        
        _,_,M = get_mapping_scores(self.sm, self.k1, self.k2, n_top = n_top)
        M=M.T
        ax = q(M.index)
        bx = q(M.columns)
        data = M.values.copy()
        data[data<thr]=0
        x,y = data.nonzero()
        ct1,ct2 = ax[x],bx[y]
        res={}
        for i in range(ct1.size):
            a = '_'.join(ct1[i].split('_')[1:])
            b = '_'.join(ct2[i].split('_')[1:])
            print('Calculating gene pairs for the mapping: {};{} to {};{}'.format(self.id1,a,self.id2,b))
            res['{};{}'.format(ct1[i],ct2[i])] = self.find_genes(a,b,**kwargs)
            
        res = pd.DataFrame([res[k][0] for k in res.keys()],index=res.keys()).fillna(np.nan).T            
        return res
        
    def find_genes(
        self,
        n1,
        n2,
        w1t=0.2,
        w2t=0.2,
        n_genes=1000,
        thr=1e-2,
    ):
        """Find enriched gene pairs in a particular pair of cell types.
        
        n1: str, cell type ID from species 1
        
        n2: str, cell type ID from species 2
        
        w1t & w2t: float, optional, default 0.2
            SAM weight threshold for species 1 and 2. Genes with below this threshold will not be
            included in any enriched gene pairs.
        
        n_genes: int, optional, default 1000
            Takes the top 1000 ranked gene pairs before filtering based on differential expressivity and
            SAM weights.
        
        thr: float, optional, default 0.01
            Excludes genes with greater than 0.01 differential expression p-value.
            
        Returns
        -------
        G - Enriched gene pairs
        G1 - Genes from species 1 involved in enriched gene pairs
        G2 - Genes from species 2 involved in enriched gene pairs
        """
        n1 = str(n1)
        n2 = str(n2)
        
        assert n1 in q(self.s1.adata.obs[self.k1])
        assert n2 in q(self.s2.adata.obs[self.k2])
        
        m = self._find_link_genes_avg(n1, n2, w1t=w1t, w2t=w2t, expr_thr=0.05)

        self.gene_pair_scores = pd.Series(index=self.s3.adata.uns['gene_pairs'], data=m)

        G = q(self.s3.adata.uns['gene_pairs'][np.argsort(-m)[:n_genes]])
        G1 = substr(G, ";", 0)
        G2 = substr(G, ";", 1)
        G = q(
            G[
                np.logical_and(
                    q(self.s1.adata.varm[self.k1 + "_pvals"][n1][G1] < thr),
                    q(self.s2.adata.varm[self.k2 + "_pvals"][n2][G2] < thr),
                )
            ]
        )
        G1 = substr(G, ";", 0)
        G2 = substr(G, ";", 1)
        _, ix1 = np.unique(G1, return_index=True)
        _, ix2 = np.unique(G2, return_index=True)
        G1 = G1[np.sort(ix1)]
        G2 = G2[np.sort(ix2)]
        return G, G1, G2

    def _find_link_genes_avg(self, c1, c2, w1t=0.35, w2t=0.35, expr_thr=0.05):
        mu1 = self.mu1
        std1 = self.v1
        mu2 = self.mu2
        std2 = self.v2
        sam1 = self.s1
        sam2 = self.s2
        key1 = self.k1
        key2 = self.k2
        sam3 = self.s3

        x1 = sam1.get_labels(key1)
        x2 = sam2.get_labels(key2)
        g1, g2 = (
            ut.extract_annotation(sam3.adata.uns['gene_pairs'], 0, ";"),
            ut.extract_annotation(sam3.adata.uns['gene_pairs'], 1, ";"),
        )
        X1 = _sparse_sub_standardize(sam1.adata[:, g1].X[x1 == c1, :], mu1, std1)
        X2 = _sparse_sub_standardize(sam2.adata[:, g2].X[x2 == c2, :], mu2, std2)
        a, b = sam3.adata.obsp["connectivities"][
            : sam1.adata.shape[0], sam1.adata.shape[0] :
        ][x1 == c1, :][:, x2 == c2].nonzero()
        c, d = sam3.adata.obsp["connectivities"][
            sam1.adata.shape[0] :, : sam1.adata.shape[0]
        ][x2 == c2, :][:, x1 == c1].nonzero()

        pairs = np.unique(np.vstack((np.vstack((a, b)).T, np.vstack((d, c)).T)), axis=0)

        av1 = X1[np.unique(pairs[:, 0]), :].mean(0).A.flatten()
        av2 = X2[np.unique(pairs[:, 1]), :].mean(0).A.flatten()
        sav1 = (av1 - av1.mean()) / av1.std()
        sav2 = (av2 - av2.mean()) / av2.std()
        sav1[sav1 < 0] = 0
        sav2[sav2 < 0] = 0
        val = sav1 * sav2 / sav1.size
        X1.data[:] = 1
        X2.data[:] = 1
        min_expr = (X1.mean(0).A.flatten() > expr_thr) * (
            X2.mean(0).A.flatten() > expr_thr
        )

        w1 = sam1.adata.var["weights"][g1].values.copy()
        w2 = sam2.adata.var["weights"][g2].values.copy()
        w1[w1 < 0.2] = 0
        w2[w2 < 0.2] = 0
        w1[w1 > 0] = 1
        w2[w2 > 0] = 1
        return val * w1 * w2 * min_expr

    def _find_link_genes(
        self, c1, c2, w1t=0.35, w2t=0.35, knn=False, n_pairs=250, expr_thr=0.05
    ):
        mu1 = self.mu1
        std1 = self.v1
        mu2 = self.mu2
        std2 = self.v2
        sam1 = self.s1
        sam2 = self.s2
        key1 = self.k1
        key2 = self.k2
        sam3 = self.s3

        x1 = sam1.get_labels(key1)
        x2 = sam2.get_labels(key2)
        g1, g2 = (
            ut.extract_annotation(sam3.adata.uns['gene_pairs'], 0, ";"),
            ut.extract_annotation(sam3.adata.uns['gene_pairs'], 1, ";"),
        )
        if knn:
            X1 = _sparse_sub_standardize(
                sam1.adata[:, g1].layers["X_knn_avg"][x1 == c1, :], mu1, std1
            )
            X2 = _sparse_sub_standardize(
                sam2.adata[:, g2].layers["X_knn_avg"][x2 == c2, :], mu2, std2
            )
        else:
            X1 = _sparse_sub_standardize(sam1.adata[:, g1].X[x1 == c1, :], mu1, std1)
            X2 = _sparse_sub_standardize(sam2.adata[:, g2].X[x2 == c2, :], mu2, std2)

        X1 = _sparse_sub_standardize(X1, mu1, std1, rows=True).tocsr()
        X2 = _sparse_sub_standardize(X2, mu2, std2, rows=True).tocsr()

        a, b = sam3.adata.obsp["connectivities"][
            : sam1.adata.shape[0], sam1.adata.shape[0] :
        ][x1 == c1, :][:, x2 == c2].nonzero()
        c, d = sam3.adata.obsp["connectivities"][
            sam1.adata.shape[0] :, : sam1.adata.shape[0]
        ][x2 == c2, :][:, x1 == c1].nonzero()

        pairs = np.unique(np.vstack((np.vstack((a, b)).T, np.vstack((d, c)).T)), axis=0)

        Z = X1[pairs[:, 0], :].multiply(X2[pairs[:, 1], :]).tocsr()
        Z.data[:] /= X1.shape[1]
        X1.data[:] = 1
        X2.data[:] = 1
        min_expr = (X1.mean(0).A.flatten() > expr_thr) * (
            X2.mean(0).A.flatten() > expr_thr
        )

        w1 = sam1.adata.var["weights"][g1].values.copy()
        w2 = sam2.adata.var["weights"][g2].values.copy()
        w1[w1 < w1t] = 0
        w2[w2 < w2t] = 0
        w1[w1 > 0] = 1
        w2[w2 > 0] = 1

        Z = sparse_knn(Z.T, n_pairs)
        val = _knndist(Z, n_pairs).T
        mu = val.mean(0) * w1 * w2 * min_expr
        return mu


def find_cluster_markers(sam, key, inplace=True):
    """ Finds differentially expressed genes for provided cell type labels.
    
    Parameters
    ----------
    sam - SAM object
    
    key - str
        Column in `sam.adata.obs` for which to identifying differentially expressed genes.
        
    inplace - bool, optional, default True
        If True, deposits enrichment scores in `sam.adata.varm[f'{key}_scores']`
        and p-values in `sam.adata.varm[f'{key}_pvals']`.
        
        Otherwise, returns three pandas.DataFrame objects (genes x clusters).
            NAMES - the gene names
            PVALS - the p-values
            SCORES - the enrichment scores
    """
    sam.adata.raw = sam.adata_raw
    a,c = np.unique(q(sam.adata.obs[key]),return_counts=True)
    t = a[c==1]

    adata = sam.adata[np.in1d(q(sam.adata.obs[key]),a[c==1],invert=True)].copy()
    try:
        sc.tl.rank_genes_groups(
            adata,
            key,
            method="wilcoxon",
            n_genes=sam.adata.shape[1],
            use_raw=True,
            layer=None,
        )
    except ValueError:
        sc.tl.rank_genes_groups(
            adata,
            key,
            method="wilcoxon",
            n_genes=sam.adata.shape[1],
            use_raw=False,
            layer=None,
        )        
    sam.adata.uns['rank_genes_groups'] = adata.uns['rank_genes_groups']
    
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
        pvals = q(pvals)
        scores = q(scores)
        
        dfs1.append(pd.DataFrame(
            data=scores[None, :], index = [SCORES.columns[i]], columns=names
        )[sam.adata.var_names].T)
        dfs2.append(pd.DataFrame(
            data=pvals[None, :], index = [SCORES.columns[i]], columns=names
        )[sam.adata.var_names].T)
    df1 = pd.concat(dfs1,axis=1)
    df2 = pd.concat(dfs2,axis=1)
    
    try:
        sam.adata.varm[key+'_scores'] = df1
        sam.adata.varm[key+'_pvals'] = df2
    except:
        sam.adata.varm.dim_names = sam.adata.var_names
        sam.adata.varm.dim_names = sam.adata.var_names
        sam.adata.varm[key+'_scores'] = df1
        sam.adata.varm[key+'_pvals'] = df2        
        
    for i in range(t.size):
        sam.adata.varm[key+'_scores'][t[i]]=0
        sam.adata.varm[key+'_pvals'][t[i]]=1
    


def ParalogSubstitutions(sm, ortholog_pairs, paralog_pairs=None, psub_thr = 0.3):
    """Identify paralog substitutions. 
    
    For all genes in `ortholog_pairs` and `paralog_pairs`, this function expects the genes to
    be prepended with their corresponding species IDs (i.e. `sm.id1` or `sm.id2`).
    
    Parameters
    ----------
    sm - SAMAP object
    
    ortholog_pairs - n x 2 numpy array of ortholog pairs
    
    paralog_pairs - n x 2 numpy array of paralog pairs, optional, default None
        If None, assumes every pair in the homology graph that is not an ortholog is a paralog.
        Note that this would essentially result in the more generic 'homolog substitutions' rather
        than paralog substitutions.
        
        The paralogs can be either cross-species, within-species, or a mix of both. 
        
    psub_thr - float, optional, default 0.3
        Threshold for correlation difference between paralog pairs and ortholog pairs.
        Paralog pairs that do not have greater than `psub_thr` correlation than their 
        corresponding ortholog pairs are filtered out.
        
    Returns
    -------
    RES - pandas.DataFrame
        A table of paralog substitutions.
        
    """
    if paralog_pairs is not None:
        ids1 = np.array([x.split('_')[0] for x in paralog_pairs[:,0]])
        ids2 = np.array([x.split('_')[0] for x in paralog_pairs[:,1]])
        ix = np.where(ids1==ids2)[0]
        ixnot = np.where(ids1!=ids2)[0]
        
        if ix.size > 0:
            pps = paralog_pairs[ix]
            
            ZZ1 = {}
            ZZ2 = {}
            for i in range(pps.shape[0]):    
                L = ZZ1.get(pps[i,0],[])
                L.append(pps[i,1])
                ZZ1[pps[i,0]]=L

                L = ZZ2.get(pps[i,1],[])
                L.append(pps[i,0])
                ZZ2[pps[i,1]]=L      

            keys = list(ZZ1.keys())
            for k in keys:
                L = ZZ2.get(k,[])
                L.extend(ZZ1[k])
                ZZ2[k] = list(np.unique(L))

            ZZ = ZZ2

            L1=[]
            L2=[]
            for i in range(ortholog_pairs.shape[0]):
                try:
                    x = ZZ[ortholog_pairs[i,0]]
                except:
                    x = []
                L1.extend([ortholog_pairs[i,1]]*len(x))
                L2.extend(x)

                try:
                    x = ZZ[ortholog_pairs[i,1]]
                except:
                    x = []
                L1.extend([ortholog_pairs[i,0]]*len(x))
                L2.extend(x)        

            L = np.vstack((L2,L1)).T
            pps = np.unique(np.sort(L,axis=1),axis=0)
            
            paralog_pairs = np.unique(np.sort(np.vstack((pps,paralog_pairs[ixnot])),axis=1),axis=0)
        
        
    smp = sm.samap
    
    gnnm = smp.adata.uns["homology_graph_reweighted"]
    gn = sm.gn
    
    ortholog_pairs = ortholog_pairs[np.logical_and(np.in1d(ortholog_pairs[:,0],gn),np.in1d(ortholog_pairs[:,1],gn))]
    if paralog_pairs is None:
        paralog_pairs = gn[np.vstack(smp.adata.uns["homology_graph"].nonzero()).T]
    else:
        paralog_pairs = paralog_pairs[np.logical_and(np.in1d(paralog_pairs[:,0],gn),np.in1d(paralog_pairs[:,1],gn))]
        
    paralog_pairs = paralog_pairs[
        np.in1d(to_vn(paralog_pairs), np.append(to_vn(ortholog_pairs),to_vn(ortholog_pairs[:,::-1])), invert=True)
    ]

    A = pd.DataFrame(data=np.arange(gn.size)[None, :], columns=gn)
    xp, yp = (
        A[paralog_pairs[:, 0]].values.flatten(),
        A[paralog_pairs[:, 1]].values.flatten(),
    )
    xp, yp = np.unique(
        np.vstack((np.vstack((xp, yp)).T, np.vstack((yp, xp)).T)), axis=0
    ).T

    xo, yo = (
        A[ortholog_pairs[:, 0]].values.flatten(),
        A[ortholog_pairs[:, 1]].values.flatten(),
    )
    xo, yo = np.unique(
        np.vstack((np.vstack((xo, yo)).T, np.vstack((yo, xo)).T)), axis=0
    ).T
    A = pd.DataFrame(data=np.vstack((xp, yp)).T, columns=["x", "y"])
    pairdict = df_to_dict(A, key_key="x", val_key="y")
    Xp = []
    Yp = []
    Xo = []
    Yo = []
    for i in range(xo.size):
        try:
            y = pairdict[xo[i]]
        except KeyError:
            y = np.array([])
        Yp.extend(y)
        Xp.extend([xo[i]] * y.size)
        Xo.extend([xo[i]] * y.size)
        Yo.extend([yo[i]] * y.size)

    orths = to_vn(gn[np.vstack((np.array(Xo), np.array(Yo))).T])
    paras = to_vn(gn[np.vstack((np.array(Xp), np.array(Yp))).T])
    orth_corrs = gnnm[Xo, Yo].A.flatten()
    par_corrs = gnnm[Xp, Yp].A.flatten()
    diff_corrs = par_corrs - orth_corrs

    RES = pd.DataFrame(
        data=np.vstack((orths, paras)).T, columns=["ortholog pairs", "paralog pairs"]
    )
    RES["ortholog corrs"] = orth_corrs
    RES["paralog corrs"] = par_corrs
    RES["corr diff"] = diff_corrs
    RES = RES.sort_values("corr diff", ascending=False)
    RES = RES[RES["corr diff"] > psub_thr]
    return RES


def convert_eggnog_to_homologs(sm, A, B, og_key = 'eggNOG_OGs', taxon=2759):
    """Gets an n x 2 array of homologs at some taxonomic level based on Eggnog results.
    
    Parameters
    ----------
    smp: SAMAP object
    
    A: pandas.DataFrame, Eggnog output table
    
    B: pandas.DataFrame, Eggnog output table

    og_key: str, optional, default 'eggNOG_OGs'
        The column name of the orthology group mapping results in the Eggnog output table.

    taxon: int, optional, default 2759
        Taxonomic ID corresponding to the level at which genes with overlapping orthology groups
        will be considered homologs. Defaults to the Eukaryotic level.
        
    Returns
    -------
    homolog_pairs: n x 2 numpy array of homolog pairs.
    """
    smp = sm.samap
    
    taxon = str(taxon)
    A = A.copy()
    B = B.copy()
    s = q(smp.adata.obs["species"])
    _, ix = np.unique(s, return_index=True)
    id1, id2 = s[np.sort(ix)][:2]
    A.index = id1 + "_" + A.index
    B.index = id2 + "_" + B.index
    A = pd.concat((A, B), axis=0)
    gn = q(smp.adata.uns["homology_gene_names"])
    A = A[np.in1d(q(A.index), gn)]

    orthology_groups = A[og_key]
    og = q(orthology_groups)
    x = np.unique(",".join(og).split(","))
    D = pd.DataFrame(data=np.arange(x.size)[None, :], columns=x)

    for i in range(og.size):
        n = orthology_groups[i].split(",")
        taxa = substr(substr(n, "@", 1),'|',0)
        if (taxa == "2759").sum() > 1 and taxon == '2759':
            og[i] = ""
        else:
            og[i] = "".join(np.array(n)[taxa == taxon])

    A[og_key] = og

    og = q(A[og_key][gn])
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
    gnf = q([x.split("_")[0] for x in gn])
    id1, id2 = gnf[np.sort(np.unique(gnf, return_index=True)[1])]
    B1 = B[gnf == id1]
    B2 = B[gnf == id2]
    B = B1.dot(B2.T)
    B = sp.sparse.vstack(
        (
            sp.sparse.hstack((sp.sparse.csr_matrix((B1.shape[0],) * 2), B)),
            sp.sparse.hstack((B.T, sp.sparse.csr_matrix((B2.shape[0],) * 2))),
        )
    ).tocsr()
    B.data[:] = 1
    return gn[np.vstack((B.nonzero())).T]


def CellTypeTriangles(sms,keys, align_thr=0.1):
    """Outputs a table of cell type triangles.
    
    Parameters
    ----------
    sms: list or tuple of three SAMAP objects for three different species mappings
       
    keys: list or tuple of three strings corresponding to each species annotation column
        Let `sms[0]` be the mapping for species A to B. Each element of `keys` corresponds to an
        annotation column in species `[A, B, C]`, respectively.

    align_thr: float, optional, default, 0.1
        Only keep triangles with minimum `align_thr` alignment score.        
    """
    
    sm1,sm2,sm3 = sms
    key1,key2,key3 = keys
    
    smp1 = sm1.samap
    smp2 = sm2.samap
    smp3 = sm3.samap
    
    s = q(smp1.adata.obs["species"])
    A,B=sm1.id1,sm1.id2

    s = q(smp2.adata.obs["species"])
    B1,B2=sm2.id1,sm2.id2
    C = B1 if B1 not in [A, B] else B2

    A1, A2 = A, B
    C1,C2 = sm3.id1,sm3.id2

    codes = dict(zip([A, B, C], [key1, key2, key3]))
    X = []
    W = []
    for i in [[A1, A2, smp1], [B1, B2, smp2], [C1, C2, smp3]]:
        x, y, smp = i
        k1, k2 = codes[x], codes[y]

        cl1 = q(smp.adata.obs[k1]).astype('object')
        cl2 = q(smp.adata.obs[k2]).astype('object')
        
        cl1[smp.adata.obs['species']==y] = cl2[smp.adata.obs['species']==y]
        cl2[smp.adata.obs['species']==x] = cl1[smp.adata.obs['species']==x]

        smp.adata.obs["triangle_{}{}".format(x, y)] = pd.Categorical(cl1)

        _, ax, bx, CSIMt = _compute_csim(smp, key="triangle_{}{}".format(x, y))
        pairsi = np.vstack(CSIMt.nonzero()).T
        pairs = np.vstack((ax[pairsi[:, 0]], bx[pairsi[:, 1]])).T
        X.append(pairs)
        W.append(CSIMt[pairsi[:, 0], pairsi[:, 1]])

    all_pairsf = np.vstack(X)
    alignmentf = np.concatenate(W)
    alignment = alignmentf.copy()
    all_pairs = all_pairsf.copy()
    all_pairs = all_pairs[alignment > align_thr]
    alignment = alignment[alignment > align_thr]
    all_pairs = to_vn(np.sort(all_pairs, axis=1))

    x, y = substr(all_pairs, ";")
    ctu = np.unique(np.concatenate((x, y)))
    Z = pd.DataFrame(data=np.arange(ctu.size)[None, :], columns=ctu)
    nnm = sp.sparse.lil_matrix((ctu.size,) * 2)
    p1, p2 = Z[x].values.flatten(), Z[y].values.flatten()
    nnm[Z[x].values.flatten(), Z[y].values.flatten()] = alignment
    nnm[Z[y].values.flatten(), Z[x].values.flatten()] = alignment
    nnm = nnm.tocsr()
    pairs = np.vstack((x, y)).T

    import networkx as nx

    G = nx.Graph()
    G.add_edges_from(ctu[np.vstack(nnm.nonzero()).T])
    all_cliques = nx.enumerate_all_cliques(G)
    all_triangles = [x for x in all_cliques if len(x) == 3]
    Z = np.sort(np.vstack(all_triangles), axis=1)
    DF = pd.DataFrame(data=Z, columns=[x.split("_")[0] for x in Z[0]])
    DF = DF[[A, B, C]]
    return DF


def SubstitutionTriangles(sms,orths,keys=None,compute_markers=True,corr_thr=0.3, psub_thr = 0.3, pval_thr=1e-10):
    """Outputs a table of homolog substitution triangles.
    
    Parameters
    ----------
    sms: list or tuple of three SAMAP objects for three different species mappings
    
    orths: list or tuple of three (n x 2) ortholog pairs corresponding to each species mapping
    
    keys: list or tuple of three strings corresponding to each species annotation column, optional, default None
        If you'd like to include information about where each gene is differentially expressed, you can specify the
        annotation column to compute differential expressivity from for each species.
        Let `sms[0]` be the mapping for species A to B. Each element of `keys` corresponds to an
        annotation column in species `[A, B, C]`, respectively.

    compute_markers: bool, optional, default True
        Set this to False if you already precomputed differential expression for the input keys.
        
    corr_thr: float, optional, default, 0.3
        Only keep triangles with minimum `corr_thr` correlation.
        
    pval_thr: float, optional, defaul, 1e-10
        Consider cell types as differentially expressed if their p-values are less than `pval_thr`.
    """
    sm1,sm2,sm3 = sms
    orth1,orth2,orth3 = orths
    
    smp1 = sm1.samap
    smp2 = sm2.samap
    smp3 = sm3.samap

    s = q(smp1.adata.obs["species"])
    A, B = s[np.sort(np.unique(s, return_index=True)[1])][:2]
    sam1,sam2 = sm1.sam1,sm1.sam2
    
    
    s = q(smp2.adata.obs["species"])
    B1, B2 = s[np.sort(np.unique(s, return_index=True)[1])][:2]
    C = B1 if B1 not in [A, B] else B2
    sam3 = sm2.sam1 if B1 not in [A, B] else sm2.sam2
    
    
    A1, A2 = A, B
    s = q(smp1.adata.obs["species"])
    C1, C2 = s[np.sort(np.unique(s, return_index=True)[1])][:2]

    RES = []
    for i in [[sm1, orth1], [sm2, orth2], [sm3, orth3]]:
        sm, orth = i
        RES.append(ParalogSubstitutions(sm, orth, psub_thr = psub_thr))
    RES1, RES2, RES3 = RES

    op1 = to_vo(q(RES1["ortholog pairs"]))
    op2 = to_vo(q(RES2["ortholog pairs"]))
    op3 = to_vo(q(RES3["ortholog pairs"]))
    pp1 = to_vo(q(RES1["paralog pairs"]))
    pp2 = to_vo(q(RES2["paralog pairs"]))
    pp3 = to_vo(q(RES3["paralog pairs"]))

    gnnm1 = smp1.adata.uns["homology_graph_reweighted"]
    gnnm2 = smp2.adata.uns["homology_graph_reweighted"]
    gnnm3 = smp3.adata.uns["homology_graph_reweighted"]
    gn1 = smp1.adata.uns["homology_gene_names"]
    gn2 = smp2.adata.uns["homology_gene_names"]
    gn3 = smp3.adata.uns["homology_gene_names"]

    # suppress warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T1 = pd.DataFrame(data=np.arange(gn1.size)[None, :], columns=gn1)
        x, y = T1[op1[:, 0]].values.flatten(), T1[op1[:, 1]].values.flatten()
        gnnm1[x, y] = gnnm1[x, y]
        gnnm1[y, x] = gnnm1[y, x]

        T1 = pd.DataFrame(data=np.arange(gn2.size)[None, :], columns=gn2)
        x, y = T1[op2[:, 0]].values.flatten(), T1[op2[:, 1]].values.flatten()
        gnnm2[x, y] = gnnm2[x, y]
        gnnm2[y, x] = gnnm2[y, x]

        T1 = pd.DataFrame(data=np.arange(gn3.size)[None, :], columns=gn3)
        x, y = T1[op3[:, 0]].values.flatten(), T1[op3[:, 1]].values.flatten()
        gnnm3[x, y] = gnnm3[x, y]
        gnnm3[y, x] = gnnm3[y, x]

    gnnm1.data[gnnm1.data==0]=1e-4
    gnnm2.data[gnnm2.data==0]=1e-4
    gnnm3.data[gnnm3.data==0]=1e-4
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

    import networkx as nx

    G = nx.from_scipy_sparse_matrix(GNNM, create_using=nx.Graph)
    all_cliques = nx.enumerate_all_cliques(G)
    all_triangles = [x for x in all_cliques if len(x) == 3]
    Z = all_genes[np.sort(np.vstack(all_triangles), axis=1)]
    DF = pd.DataFrame(data=Z, columns=[x.split("_")[0] for x in Z[0]])
    DF = DF[[A, B, C]]

    orth1DF = pd.DataFrame(data=orth1, columns=[x.split("_")[0] for x in orth1[0]])[
        [A, B]
    ]
    orth2DF = pd.DataFrame(data=orth2, columns=[x.split("_")[0] for x in orth2[0]])[
        [A, C]
    ]
    orth3DF = pd.DataFrame(data=orth3, columns=[x.split("_")[0] for x in orth3[0]])[
        [B, C]
    ]

    ps1DF = pd.DataFrame(
        data=np.sort(pp1, axis=1),
        columns=[x.split("_")[0] for x in np.sort(pp1, axis=1)[0]],
    )[[A, B]]
    ps2DF = pd.DataFrame(
        data=np.sort(pp2, axis=1),
        columns=[x.split("_")[0] for x in np.sort(pp2, axis=1)[0]],
    )[[A, C]]
    ps3DF = pd.DataFrame(
        data=np.sort(pp3, axis=1),
        columns=[x.split("_")[0] for x in np.sort(pp3, axis=1)[0]],
    )[[B, C]]

    A_AB = pd.DataFrame(data=to_vn(op1)[None, :], columns=to_vn(ps1DF.values))
    A_AC = pd.DataFrame(data=to_vn(op2)[None, :], columns=to_vn(ps2DF.values))
    A_BC = pd.DataFrame(data=to_vn(op3)[None, :], columns=to_vn(ps3DF.values))

    AB = to_vn(DF[[A, B]].values)
    AC = to_vn(DF[[A, C]].values)
    BC = to_vn(DF[[B, C]].values)

    AVs = []
    CATs = []
    CORRs = []
    for i, X, O, P, Z, R in zip(
        [0, 1, 2],
        [AB, AC, BC],
        [orth1DF, orth2DF, orth3DF],
        [ps1DF, ps2DF, ps3DF],
        [A_AB, A_AC, A_BC],
        [CORR1, CORR2, CORR3],
    ):
        cat = q(["homolog"] * X.size).astype("object")
        cat[np.in1d(X, to_vn(O.values))] = "ortholog"
        ff = np.in1d(X, to_vn(P.values))
        cat[ff] = "substitution"
        z = Z[X[ff]] #problem line here
        x = X[ff]
        av = np.zeros(x.size, dtype="object")
        for ai in range(x.size):
            v=pd.DataFrame(z[x[ai]]) #get ortholog pairs - paralog pairs dataframe
            vd=v.values.flatten() #get ortholog pairs
            vc=q(';'.join(v.columns).split(';')) # get paralogous genes
            temp = np.unique(q(';'.join(vd).split(';'))) #get orthologous genes
            av[ai] = ';'.join(temp[np.in1d(temp,vc,invert=True)]) #get orthologous genes not present in paralogous genes
        AV = np.zeros(X.size, dtype="object")
        AV[ff] = av
        corr = R[X].values.flatten()

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

    FINAL = pd.DataFrame(data = data, columns = [f'{A} gene',f'{B} gene',f'{C} gene',
                                                 f'{A}/{B} subbed',f'{A}/{C} subbed',f'{B}/{C} subbed',
                                                 f'{A}/{B}',f'{A}/{C}',f'{B}/{C}',
                                                 f'{A}/{B} corr',f'{A}/{C} corr',f'{B}/{C} corr',
                                                 f'{A}/{B} type',f'{A}/{C} type',f'{B}/{C} type'])
    FINAL['#orthologs'] = (cat_pairs=='ortholog').sum(1)
    FINAL['#substitutions'] = (cat_pairs=='substitution').sum(1)    
    FINAL = FINAL[(FINAL['#orthologs']+FINAL['#substitutions'])==3]
    x = FINAL[[f'{A}/{B} corr',f'{A}/{C} corr',f'{B}/{C} corr']].min(1)
    FINAL['min_corr'] = x
    FINAL = FINAL[x>corr_thr]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if keys is not None:
            for i,sam,n in zip([0,1,2],[sam1,sam2,sam3],[A,B,C]):
                if compute_markers:
                    find_cluster_markers(sam,keys[i])
                a = sam.adata.varm[keys[i]+'_scores'].T[q(FINAL[n+' gene'])].T
                p = sam.adata.varm[keys[i]+'_pvals'].T[q(FINAL[n+' gene'])].T.values
                p[p>pval_thr]=1
                p[p<1]=0
                p=1-p
                f = a.columns[a.values.argmax(1)]
                res=[]
                for i in range(p.shape[0]):
                    res.append(';'.join(np.unique(np.append(f[i],a.columns[p[i,:]==1]))))            
                FINAL[n+' cell type'] = res
    FINAL = FINAL.sort_values('min_corr',ascending=False)
    return FINAL


def _compute_csim(sam3, key, X=None, n_top = 0):
    if n_top ==0:
        n_top = 100000000
        
    cl1 = q(sam3.adata.obs[key].values[sam3.adata.obs["batch"] == "batch1"])
    clu1,cluc1 = np.unique(cl1,return_counts=True)
    cl2 = q(sam3.adata.obs[key].values[sam3.adata.obs["batch"] == "batch2"])
    clu2,cluc2 = np.unique(cl2,return_counts=True)

    clu1s = q("batch1_" + clu1.astype("str").astype("object"))
    clu2s = q("batch2_" + clu2.astype("str").astype("object"))
    cl = q(
        sam3.adata.obs["batch"].values.astype("object")
        + "_"
        + sam3.adata.obs[key].values.astype("str").astype("object")
    )

    CSIM1 = np.zeros((clu1s.size, clu2s.size))
    if X is None:
        X = sam3.adata.obsp["connectivities"].copy()

    for i, c1 in enumerate(clu1s):
        for j, c2 in enumerate(clu2s):
            CSIM1[i, j] = np.append(
                np.sort(X[cl == c1, :][:, cl == c2].sum(1).A.flatten())[::-1][:n_top],
                np.sort(X[cl == c2, :][:, cl == c1].sum(1).A.flatten())[::-1][:n_top],
            ).mean()
    CSIMth = CSIM1 / sam3.adata.uns['mdata']['knn_1v2'][0].data.size    
    s1 = CSIMth.sum(1).flatten()[:, None]
    s2 = CSIMth.sum(0).flatten()[None, :]
    s1[s1 == 0] = 1
    s2[s2 == 0] = 1
    CSIM1 = CSIMth / s1
    CSIM2 = CSIMth / s2
    CSIM = (CSIM1 * CSIM2) ** 0.5

    return CSIM, clu1, clu2, CSIMth

def transfer_annotations(sm,reference=1, keys=[],num_iters=5, inplace = True):
    """ Transfer annotations across species using label propagation along the combined manifold.
    
    Parameters
    ----------
    sm - SAMAP object
    
    reference - 1 or 2, optional, default 1
        The reference species transfers its labels to the target species.
        1 corresponds to species 1 (`sm.id1`) and 2 corresponds to species 2.
        
    keys - str or list, optional, default []
        The `obs` key or list of keys corresponding to the labels to be propagated.
        If passed an empty list, all keys in the reference species' `obs` dataframe
        will be propagated.
        
    num_iters - int, optional, default 5
        The number of steps to run the diffusion propagation.
        
    inplace - bool, optional, default True
        If True, deposit propagated labels in the target species (`sm.sam1/sm.sam2`) `obs`
        DataFrame. Otherwise, just return the soft-membership DataFrame.
        
    Returns
    -------
    A Pandas DataFrame with soft membership scores for each cluster in each cell.
    
    """
    
    sam1 = sm.sam1
    sam2 = sm.sam2
    stitched = sm.samap
    NNM = stitched.adata.obsp['connectivities'].copy()
    NNM = NNM.multiply(1/NNM.sum(1).A).tocsr()
    
    if type(keys) is str:
        keys = [keys]
    elif len(keys) == 0:
        if reference == 1:
            keys = list(sam1.adata.obs.keys())
        elif reference == 2:
            keys = list(sam2.adata.obs.keys())
        else:
            raise ValueError('`reference` must be either 1 or 2`')
    #stitched.load_obs_annotations()
    for key in keys:
        if reference == 1:
            samref=sam1
            sam=sam2
        elif reference == 2:
            samref=sam2
            sam=sam1
        else:
            raise ValueError('`reference` must be either 1 or 2`')

        ANN = samref.adata.obs
        cl = ANN[key].values.astype('object').astype('<U300')
        clu,clui = np.unique(cl,return_inverse=True)
        P = np.zeros((NNM.shape[0],clu.size))
        Pmask = np.ones((NNM.shape[0],clu.size))
        if reference == 1:
            for i in range(samref.adata.shape[0]):
                P[i,clui[i]]=1.0
            Pmask[:samref.adata.shape[0],:]=0
        elif reference == 2:
            for i in range(samref.adata.shape[0]):
                P[i+sam.adata.shape[0],clui[i]]=1.0
            Pmask[sam.adata.shape[0]:,:]=0
        Pinit = P.copy()

        for j in range(num_iters):
            P_new = NNM.dot(P)
            if np.max(np.abs(P_new - P)) < 5e-3:
                P = P_new
                s=P.sum(1)[:,None]
                s[s==0]=1
                P = P/s
                break
            else:
                P = P_new
                s=P.sum(1)[:,None]
                s[s==0]=1
                P = P/s
            P = P * Pmask + Pinit

        uncertainty = 1-P.max(1)
        labels = clu[np.argmax(P,axis=1)]
        labels[uncertainty==1.0]='NAN'
        uncertainty[np.argmax(uncertainty)] = 1
        
        if inplace:
            sam.adata.obs[key+'_t'] = pd.Series(labels,index = stitched.adata.obs_names)        
            sam.adata.obs[key+'_uncertainty'] = pd.Series(uncertainty,index=stitched.adata.obs_names)

        res = pd.DataFrame(data=P,index=stitched.adata.obs_names,columns=clu)
        res['labels'] = labels
        return res

def get_mapping_scores(sm, key1, key2, n_top = 0):
    """Calculate mapping scores
    Parameters
    ----------
    sm: SAMAP object
    
    key1 & key2: str, annotation vector keys for species 1 and 2
    
    n_top: int, optional, default 0
        If `n_top` is 0, average the alignment scores for all cells in a pair of clusters.
        Otherwise, average the alignment scores of the top `n_top` cells in a pair of clusters.
        Set this to non-zero if you suspect there to be subpopulations of your cell types mapping
        to distinct cell types in the other species.
    Returns
    -------
    D1 - table of highest mapping scores for cell types in species 1
    D2 - table of highest mapping scores for cell types in species 2
    A - pairwise table of mapping scores between cell types in species 1 (row) and 2 (columns)
    """
    sam1=sm.sam1
    sam2=sm.sam2
    samap=sm.samap

    cl1 = q(sam1.adata.obs[key1])
    cl2 = q(sam2.adata.obs[key2])
    cl = (
        q(samap.adata.obs["species"]).astype("object")
        + "_"
        + np.append(cl1, cl2).astype("str").astype("object")
    )

    samap.adata.obs["{};{}_mapping_scores".format(key1,key2)] = pd.Categorical(cl)
    _, clu1, clu2, CSIMth = _compute_csim(samap, "{};{}_mapping_scores".format(key1,key2), n_top = n_top)

    A = pd.DataFrame(data=CSIMth, index=clu1, columns=clu2)
    i = np.argsort(-A.values.max(0).flatten())
    H = []
    C = []
    for I in range(A.shape[1]):
        x = A.iloc[:, i[I]].sort_values(ascending=False)
        H.append(np.vstack((x.index, x.values)).T)
        C.append(A.columns[i[I]])
        C.append(A.columns[i[I]])
    H = np.hstack(H)
    D2 = pd.DataFrame(data=H, columns=[C, ["Cluster","Alignment score"]*(H.shape[1]//2)])

    A = pd.DataFrame(data=CSIMth, index=clu1, columns=clu2).T
    i = np.argsort(-A.values.max(0).flatten())
    H = []
    C = []
    for I in range(A.shape[1]):
        x = A.iloc[:, i[I]].sort_values(ascending=False)
        H.append(np.vstack((x.index, x.values)).T)
        C.append(A.columns[i[I]])
        C.append(A.columns[i[I]])
    H = np.hstack(H)
    D1 = pd.DataFrame(data=H, columns=[C, ["Cluster","Alignment score"]*(H.shape[1]//2)])
    return D1, D2, A


def _knndist(nnma, k):
    x, y = nnma.nonzero()
    data = nnma.data
    xc, cc = np.unique(x, return_counts=True)
    cc2 = np.zeros(nnma.shape[0], dtype="int")
    cc2[xc] = cc
    cc = cc2
    newx = []
    newdata = []
    for i in range(nnma.shape[0]):
        newx.extend([i] * k)
        newdata.extend(list(data[x == i]) + [0] * (k - cc[i]))
    data = np.array(newdata)
    val = data.reshape((nnma.shape[0], k))
    return val


def _sparse_sub_standardize(X, mu, var, rows=False):
    x, y = X.nonzero()
    if not rows:
        Xs = X.copy()
        Xs.data[:] = (X.data - mu[y]) / var[y]
    else:
        mu, var = sf.mean_variance_axis(X, axis=1)
        var = var ** 0.5
        var[var == 0] = 1
        Xs = X.copy()
        Xs.data[:] = (X.data - mu[x]) / var[x]
    Xs.data[Xs.data < 0] = 0
    Xs.eliminate_zeros()
    return Xs

def _get_mu_std(sam3, sam1, sam2, knn=False):
    g1, g2 = ut.extract_annotation(sam3.adata.uns['gene_pairs'], 0, ";"), ut.extract_annotation(
        sam3.adata.uns['gene_pairs'], 1, ";"
    )
    if knn:
        mu1, var1 = sf.mean_variance_axis(sam1.adata[:, g1].layers["X_knn_avg"], axis=0)
        mu2, var2 = sf.mean_variance_axis(sam2.adata[:, g2].layers["X_knn_avg"], axis=0)
    else:
        mu1, var1 = sf.mean_variance_axis(sam1.adata[:, g1].X, axis=0)
        mu2, var2 = sf.mean_variance_axis(sam2.adata[:, g2].X, axis=0)
    var1[var1 == 0] = 1
    var2[var2 == 0] = 1
    var1 = var1 ** 0.5
    var2 = var2 ** 0.5
    return mu1, var1, mu2, var2    
