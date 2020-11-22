import sklearn.utils.sparsefuncs as sf
import typing
import os
from os import path
import gc
import pandas as pd
import hnswlib
import scipy as sp
from samalg import SAM
import numpy as np
import samalg.utilities as ut
from sklearn.preprocessing import StandardScaler
import scanpy as sc
import warnings
from numba.core.errors import NumbaPerformanceWarning, NumbaWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
warnings.filterwarnings("ignore", category=NumbaWarning)
__version__ = '0.1.1'

# Making SAMAP a class to keep internal objects persistent after early terminations.
class SAMAP(object):
    def __init__(self, data1: typing.Union[str,SAM],
          data2: typing.Union[str,SAM],
          id1: str,
          id2: str,
          f_maps: typing.Optional[str]='maps/',
          names1: typing.Optional[typing.Union[list,np.ndarray]]=None,
          names2: typing.Optional[typing.Union[list,np.ndarray]]=None,  
          reciprocal_blast: typing.Optional[bool]=True,
          key1: typing.Optional[str]='leiden_clusters',
          key2: typing.Optional[str]='leiden_clusters',
          NUMITERS: typing.Optional[int] = 3,
          leiden_res1: typing.Optional[int] = 3,
          leiden_res2: typing.Optional[int] = 3,
          NH1: typing.Optional[int] = 3,
          NH2: typing.Optional[int] = 3,
          K: typing.Optional[int] = 20,
          NOPs1: typing.Optional[int] = 4,
          NOPs2: typing.Optional[int] = 8,
          N_GENE_CHUNKS: typing.Optional[int] = 1,
          USE_SEQ: typing.Optional[bool] = False):

        """Runs the SAMap algorithm.
    
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
            
        reciprocal_blast : bool, optional, default True
            If True, only keep reciprocal edges in the computed BLAST graph.
            
        NUMITERS : int, optional, default 3
            Runs SAMap for `NUMITERS` iterations using the mutual-nearest
            neighborhood criterion.
    
        leiden_res1 : float, optional, default 3
            The resolution parameter for the leiden clustering to be done on the manifold of organism 1.
            Each cell's neighborhood size will be capped to be the size of its leiden cluster.
    
        leiden_res2 : float, optional, default 3
            The resolution parameter for the leiden clustering to be done on the manifold of organism 2.
            Each cell's neighborhood size will be capped to be the size of its leiden cluster.
    
        NH1 : int, optional, default 3
            Cells up to `NH1` hops away from a particular cell in organism 1
            will be included in its neighborhood.
    
        NH2 : int, optional, default 3
            Cells up to `NH2` hops away from a particular cell in organism 2
            will be included in its neighborhood.
    
        K : int, optional, default 20
            The number of cross-species edges to identify per cell.
    
        NOPs1 : int, optional, default 4
            Keeps the `NOPs1` largest outgoing edges in the homology graph, pruning
            the rest.
    
        NOPs2 : int, optional, default 8
            Keeps the `NOPs2` largest incoming edges in the homology graph, pruning
            the rest. The final homology graph is the union of the outgoing- and
            incoming-edge filtered graphs.
    
        N_GENE_CHUNKS: int, optional, default 1
            When updating the edge weights in the BLAST homology graph, the operation
            will be split up into `N_GENE_CHUNKS` chunks. For large datasets
            (>50,000 cells), use more chunks (e.g. 4) to avoid running out of
            memory.
    
        USE_SEQ: bool, optional, default False
            If `USE_SEQ` is False, gene-gene correlations replace the BLAST sequence
            similarity edge weights when refining the edge weights in the homology graph.
            If `USE_SEQ` is True, gene-gene correlations scale the BLAST sequence similarity
            edge weights. `USE_SEQ` is False by default.
    
        Returns
        -------
        samap - Samap
            The Samap object.
    
        D1 - pandas.DataFrame
            A DataFrame containing the highst-scoring cross-species neighbors of
            each cell type in organism 1.
    
        D2 - pandas.DataFrame
            A DataFrame containing the highst-scoring cross-species neighbors of
            each cell type in organism 2.
    
        sam1 - SAM
            The SAM object of organism 1 used as input to SAMap.
    
        sam2 - SAM
            The SAM object of organism 2 used as input to SAMap.
    
        ITER_DATA - tuple
            GNNMS_nnm - A list of scipy.sparse.csr_matrix
                The stitched cell nearest-neighbor graphs from each iteration of
                SAMap.
            GNNMS_corr - A list of scipy.sparse.csr_matrix
                The homology graph from each iteration of SAMap.
            GNNMS_pruned - A list of scipy.sparse.csr_matrix
                The pruned homology graph from each iteration of SAMap.
            SCORES_VEC - A list of numpy.ndarray
                Flattened leiden cluster alignment score matrices from each iteration.
                These are used to calculate the difference between alignment scores
                in adjacent iterations. The largest and smallest values are printed
                for each iteration while SAMap is running.
        """
    
        if not (isinstance(data1,str) or isinstance(data1,SAM)):
            raise TypeError('Input data 1 must be either a path or a SAM object.')
    
        if not (isinstance(data2,str) or isinstance(data2,SAM)):
            raise TypeError('Input data 2 must be either a path or a SAM object.')
    
        if isinstance(data1,str):
            print('Processing data 1 from:\n{}'.format(data1))
            sam1=SAM()
            sam1.load_data(data1)
            sam1.preprocess_data(sum_norm='cell_median',norm='log',thresh_low=0.0,thresh_high=0.96,min_expression=1)
            sam1.run(preprocessing='StandardScaler',npcs=150,weight_PCs=False,k=20,n_genes=3000)
            f1n = '.'.join(data1.split('.')[:-1])+'_pr.h5ad'
            print('Saving processed data to:\n{}'.format(f1n))
            sam1.save_anndata(f1n)
        else:
            sam1 = data1
    
        if isinstance(data2,str):
            print('Processing data 2 from:\n{}'.format(data2))
            sam2=SAM()
            sam2.load_data(data2)
            sam2.preprocess_data(sum_norm='cell_median',norm='log',thresh_low=0.0,thresh_high=0.96,min_expression=1)
            sam2.run(preprocessing='StandardScaler',npcs=150,weight_PCs=False,k=20,n_genes=3000)
            f2n = '.'.join(data2.split('.')[:-1])+'_pr.h5ad'
            print('Saving processed data to:\n{}'.format(f2n))
            sam2.save_anndata(f2n)
        else:
            sam2 = data2
    
    
        print('Preparing data 1 for SAMap.')
        sam1.leiden_clustering(res=leiden_res1)
        if 'PCs_SAMap' not in sam1.adata.varm.keys():
            prepare_SAMap_loadings(sam1)
    
        print('Preparing data 2 for SAMap.')
        sam2.leiden_clustering(res=leiden_res2)
        if 'PCs_SAMap' not in sam2.adata.varm.keys():
            prepare_SAMap_loadings(sam2)
    
    
        gnnm,gn1,gn2,gn = calculate_blast_graph(id1 , id2,
                                                f_maps = f_maps,
                                                reciprocate = reciprocal_blast,
                                                namesA = names1,
                                                namesB = names2)
    
        _prepend_var_prefix(sam1,id1)
        _prepend_var_prefix(sam2,id2)    
        
        ge1 = np.array(list(sam1.adata.var_names))
        ge2 = np.array(list(sam2.adata.var_names))
        
        gn1 = gn1[np.in1d(gn1,ge1)]
        gn2 = gn2[np.in1d(gn2,ge2)]
        f = np.in1d(gn,np.append(gn1,gn2))
        gn = gn[f]
        gnnm = gnnm[f][:,f]
        A = pd.DataFrame(data = np.arange(gn.size)[None,:], columns = gn)
        ge = np.append(ge1,ge2)
        ge=ge[np.in1d(ge,gn)]
        ix = A[ge].values.flatten()
        gnnm = gnnm[ix][:,ix]
        gn = ge
        gn1 = ge[np.in1d(ge,gn1)]
        gn2 = ge[np.in1d(ge,gn2)]        
    
        print('{} `{}` genes and {} `{}` gene symbols match between the datasets and the BLAST graph.'.format(gn1.size,id1,gn2.size,id2))
        
        smap = Samap(sam1,sam2,gnnm,gn1,gn2)
        
        # assign to self
        self.smap = smap
        self.gnnm = gnnm
        self.gn1 = gn1
        self.gn2 = gn2
        self.gn = gn
        self.sam1 = sam1
        self.sam2 = sam2
        
        ITER_DATA = smap.run(NUMITERS=NUMITERS,NOPs1=NOPs1,NOPs2=NOPs2,
                             NH1=NH1,NH2=NH2,K=K,NCLUSTERS=N_GENE_CHUNKS,use_seq=USE_SEQ)
        samap=smap.final_sam
        
        self.samap = samap
        self.ITER_DATA = ITER_DATA
        
        print('Alignment score ---',avg_as(samap).mean())
    
        print('Running UMAP on the stitched manifolds.')
        sc.tl.umap(samap.adata,min_dist=0.1,init_pos='random')
    
    
        hom_graph = smap.GNNMS_corr[-1]
        samap.adata.uns['homology_graph_reweighted'] = hom_graph
        samap.adata.uns['homology_graph'] = gnnm
        samap.adata.uns['homology_gene_names'] = gn
    
        samap.adata.obs['species'] = pd.Categorical([id1]*sam1.adata.shape[0]+[id2]*sam2.adata.shape[0])    


def _prepend_var_prefix(s,pre):
    x = ['_'.join(x.split('_')[1:]) for x in s.adata.var_names]
    if not (np.unique(x).size == 1 and x[0] == pre):
        if np.unique(substr(s.adata.var_names,'_',0)).size == 1:
            y = ['_'.join(x.split('_')[1:]) for x in s.adata.var_names]
        else:
            y = list(s.adata.var_names)
        vn=[pre+'_'+x for x in y]
        s.adata.var_names = vn
        s.adata_raw.var_names = vn
        
            
def get_mapping_scores(sam1,sam2,samap,key1,key2):
    
    cl1 = np.array(list(sam1.adata.obs[key1]))
    cl2 = np.array(list(sam2.adata.obs[key2]))
    cl = np.array(list(samap.adata.obs['species'])).astype('object') +'_'+ np.append(cl1,cl2).astype('str').astype('object')
    
    samap.adata.obs['mapping_score_labels'] = pd.Categorical(cl)
    _, clu1, clu2, CSIMth = compute_csim(samap,'mapping_score_labels')

    A = pd.DataFrame(data = CSIMth,index = clu1, columns = clu2)
    i = np.argsort(-A.values.max(0).flatten())
    H=[]
    C=[]
    for I in range(A.shape[1]):
        x = A.iloc[:,i[I]].sort_values(ascending=False)
        H.append(np.vstack((x.index,x.values)).T)
        C.append(A.columns[i[I]])
        C.append(A.columns[i[I]])
    H=np.hstack(H)
    D1 = pd.DataFrame(data=H,columns=[C, np.arange(H.shape[1])])

    A = pd.DataFrame(data = CSIMth,index = clu1, columns = clu2).T
    i = np.argsort(-A.values.max(0).flatten())
    H=[]
    C=[]
    for I in range(A.shape[1]):
        x = A.iloc[:,i[I]].sort_values(ascending=False)
        H.append(np.vstack((x.index,x.values)).T)
        C.append(A.columns[i[I]])
        C.append(A.columns[i[I]])
    H=np.hstack(H)
    D2 = pd.DataFrame(data=H,columns=[C, np.arange(H.shape[1])])    
    return D1,D2

def _united_proj(wpca1,wpca2,k=20, metric='correlation', sigma=500, ef = 200, M = 48):

    print('Running hsnwlib')

    labels1 = np.arange(wpca1.shape[0])
    labels2 = np.arange(wpca2.shape[0])

    p1 = hnswlib.Index(space = 'cosine', dim = wpca1.shape[1])
    p2 = hnswlib.Index(space = 'cosine', dim = wpca2.shape[1])

    p1.init_index(max_elements = wpca1.shape[0], ef_construction = ef, M = M)
    p2.init_index(max_elements = wpca2.shape[0], ef_construction = ef, M = M)

    p1.add_items(wpca1, labels1)
    p2.add_items(wpca2, labels2)

    p1.set_ef(ef)
    p2.set_ef(ef)

    idx2, dist2 = p1.knn_query(wpca2, k = k)
    idx1, dist1 = p2.knn_query(wpca1, k = k)

    dist2 = 1-dist2
    dist1 = 1-dist1

    dist1[dist1<0]=0
    dist2[dist2<0]=0
    Dist1 = dist1#np.exp(-1*(1-dist1)**2)
    Dist2 = dist2#np.exp(-1*(1-dist2)**2)

    knn1v2 = sp.sparse.lil_matrix((wpca1.shape[0],wpca2.shape[0]))
    knn2v1 = sp.sparse.lil_matrix((wpca2.shape[0],wpca1.shape[0]))

    x1=np.tile(np.arange(idx1.shape[0])[:,None],(1,idx1.shape[1])).flatten()
    x2=np.tile(np.arange(idx2.shape[0])[:,None],(1,idx2.shape[1])).flatten()
    knn1v2[x1,idx1.flatten()]=Dist1.flatten()
    knn2v1[x2,idx2.flatten()]=Dist2.flatten()

    return knn1v2.tocsr(),knn2v1.tocsr()

def _mapping_window(sam1,sam2,gnnm,gn,K=20):

    ix=np.unique(np.sort(np.vstack((gnnm.nonzero())).T,axis=1),axis=0)
    ortholog_pairs = gn[ix]
    print('Found',ortholog_pairs.shape[0],'gene pairs')
    corr = gnnm[ix[:,0],ix[:,1]].A.flatten()
    corr = 0.5+0.5*np.tanh(10 * (corr-0.5))

    gns1 = ortholog_pairs[:,0]
    gns2 = ortholog_pairs[:,1]

    g1 = np.array(list(sam1.adata.var_names))
    g2 = np.array(list(sam2.adata.var_names))

    g1 = g1[np.in1d(g1,gns1)]
    g2 = g2[np.in1d(g2,gns2)]

    adata1 = sam1.adata[:,g1]
    adata2 = sam2.adata[:,g2]

    W1 = adata1.var['weights'].values
    W2 = adata2.var['weights'].values

    std = StandardScaler(with_mean=False)

    s1 = std.fit_transform(adata1.X).multiply(W1[None,:]).tocsr()
    s2 = std.fit_transform(adata2.X).multiply(W2[None,:]).tocsr()

    k = K
    
    A1=pd.DataFrame(data=np.arange(g1.size)[None,:],columns=g1)
    A2=pd.DataFrame(data=np.arange(g2.size)[None,:],columns=g2)

    G1 = A1[gns1].values.flatten()
    G2 = A2[gns2].values.flatten()

    avg = sp.sparse.lil_matrix((g1.size,g2.size))
    avg[G1,G2]=corr
    su1,su2 = avg.sum(1).A,avg.sum(0).A
    avg1=avg.multiply(1/su1).tocsr()
    avg2=avg.multiply(1/su2).tocsr()

    sp1 = s1.dot(avg2)
    sp2 = s2.dot(avg1.T)

    sp1 = std.fit_transform(sp1)
    sp2 = std.fit_transform(sp2)    

    mu1 = s1.mean(0).A.flatten()[None,:]
    mu2 = s2.mean(0).A.flatten()[None,:]    
    mu1s = sp1.mean(0).A.flatten()[None,:]
    mu2s = sp2.mean(0).A.flatten()[None,:]

    C1 = sam1.adata[:,g1].varm['PCs_SAMap'].T
    C2 = sam2.adata[:,g2].varm['PCs_SAMap'].T

    print('Recomputing PC projections with gene pair subsets...')
    ws1 = s1.dot(C1.T) - mu1.dot(C1.T)
    ws2 = s2.dot(C2.T) - mu2.dot(C2.T)
    wsp1 = sp1.dot(C2.T) - C2.dot(mu1s.T).T
    wsp2 = sp2.dot(C1.T) - C1.dot(mu2s.T).T
    wpca = np.hstack((np.vstack((ws1,wsp2)),np.vstack((wsp1,ws2))))

    wpca1 = wpca[:s1.shape[0],:]
    wpca2 = wpca[s1.shape[0]:,:]

    b1,b2 = _united_proj(wpca1, wpca2, k=k)

    output_dict={}
    output_dict['knn_1v2'] = b1.tocsr()
    output_dict['knn_2v1'] = b2.tocsr()
    output_dict['wPCA1'] = wpca1
    output_dict['wPCA2'] = wpca2
    output_dict['pca1'] = C1
    output_dict['pca2'] = C2
    output_dict['corr'] = corr
    output_dict['ortholog_pairs'] = ortholog_pairs
    output_dict['G_avg1'] = avg1.tocsr()
    output_dict['G_avg2'] = avg2.tocsr()
    output_dict['G_avg'] = avg.tocsr()
    output_dict['edge_weights'] = corr
    return output_dict

def _sparse_knn(D,k):
    D1=D.tocoo()
    idr = np.argsort(D1.row)
    D1.row[:]=D1.row[idr]
    D1.col[:]=D1.col[idr]
    D1.data[:]=D1.data[idr]

    _,ind = np.unique(D1.row,return_index=True)
    ind = np.append(ind,D1.data.size)
    for i in range(ind.size-1):
        idx = np.argsort(D1.data[ind[i]:ind[i+1]])
        if idx.size > k:
            idx = idx[:-k]
            D1.data[np.arange(ind[i],ind[i+1])[idx]]=0
    D1.eliminate_zeros()
    return D1

def _sparse_knn_ks(D,ks):
    D1=D.tocoo()
    idr = np.argsort(D1.row)
    D1.row[:]=D1.row[idr]
    D1.col[:]=D1.col[idr]
    D1.data[:]=D1.data[idr]

    row,ind = np.unique(D1.row,return_index=True)
    ind = np.append(ind,D1.data.size)
    for i in range(ind.size-1):
        idx = np.argsort(D1.data[ind[i]:ind[i+1]])
        k = ks[row[i]]
        if idx.size > k:
            if k != 0:
                idx = idx[:-k]
            else:
                idx = idx
            D1.data[np.arange(ind[i],ind[i+1])[idx]]=0
    D1.eliminate_zeros()
    return D1


def _smart_expand(nnm,cl,NH=3):
    stage0 = nnm.copy()
    S=[stage0]
    running = stage0
    for i in range(1,NH+1):
        stage = running.dot(stage0)
        running = stage
        stage=stage.tolil()
        for j in range(i):
            stage[S[j].nonzero()]=0
        stage=stage.tocsr()
        S.append(stage)

    a,ix,c = np.unique(cl,return_counts=True,return_inverse=True)
    K = c[ix]

    for i in range(len(S)):
        s = _sparse_knn_ks(S[i],K).tocsr()
        a,c = np.unique(s.nonzero()[0],return_counts=True)
        numnz = np.zeros(s.shape[0],dtype='int32')
        numnz[a] = c
        K = K - numnz
        K[K<0]=0
        S[i] = s
    res = S[0]
    for i in range(1,len(S)):
        res = res + S[i]
    return res

def samap(sams,gnnm,gn,NH1=3,NH2=3,umap=False,mdata=None,k=None,K=20,
                   chunksize=20000,coarsen=True,**kwargs):
    n = len(sams)
    DS = {}
    for I in range(n):
        sam = sams[I]
        for J in range(I+1,n):
            print('Stitching SAM ' + str(I) + ' and SAM ' + str(J))
            sam2 = sams[J]

            if len(list(sam2.adata.obs.keys())) > 0 and len(list(sam2.adata.obs.keys()))>0:
                key1 = ut.search_string(np.array(list(sam.adata.obs.keys())),'_clusters')[0][0]
                key2 = ut.search_string(np.array(list(sam2.adata.obs.keys())),'_clusters')[0][0]
            else:
                print('Generate clusters first')
                return;


            if mdata is None:
                mdata = _mapping_window(sam,sam2,gnnm,gn,K=K)

            if k is None:
                k1 = sam.run_args.get('k',20)
            else:
                k1 = k


            print('Using ' + key1 + ' and ' + key2 + ' cluster labels.')

            CL1 = sam.get_labels(key1)
            CL2 = sam2.get_labels(key2)

            clu1,ix1,cluc1 = np.unique(CL1,return_counts=True,return_inverse=True)
            clu2,ix2,cluc2 = np.unique(CL2,return_counts=True,return_inverse=True)

            K1 = cluc1[ix1]
            K2 = cluc2[ix2]

            h2m = mdata['knn_1v2']
            m2h = mdata['knn_2v1']

            if coarsen:
                h2m0 = h2m.copy()
                m2h0 = m2h.copy()
                h2m0.data[:]=1
                m2h0.data[:]=1

                print('Out-neighbor smart expansion 1')
                nnm = sam.adata.obsp['connectivities'].copy()
                nnm1_out = nnm
                nnm1_in = _smart_expand(nnm,K1,NH=NH1)
                nnm1_in.data[:]=1

                print('Out-neighbor smart expansion 2')
                nnm = sam2.adata.obsp['connectivities'].copy()
                nnm2_out = nnm
                nnm2_in = _smart_expand(nnm,K2,NH=NH2)
                nnm2_in.data[:]=1

                mdata['nnm1_out']=nnm1_out
                mdata['nnm1_in']=nnm1_in
                mdata['nnm2_out']=nnm2_out
                mdata['nnm2_in']=nnm2_in

                B = h2m
                B2 = m2h
                s = B.sum(1).A
                s2 = B2.sum(1).A
                s[s==0]=1
                s2[s2==0]=1

                B=B.multiply(1/s).tocsr()
                B2=B2.multiply(1/s2).tocsr()

                print('Indegree coarsening')

                numiter = max(nnm2_in.shape[0],nnm1_in.shape[0])//chunksize+1

                if nnm2_in.shape[0]<nnm1_in.shape[0]:
                    R=True
                else:
                    R=False

                D = sp.sparse.csr_matrix((0,min(nnm2_in.shape[0],nnm1_in.shape[0])))
                for bl in range(numiter):
                    print(str(bl)+'/'+str(numiter),D.shape,R)
                    if not R:
                        C = nnm2_in[bl*chunksize:(bl+1)*chunksize].dot(B.T)
                        C.data[C.data<0.1]=0
                        C.eliminate_zeros()

                        C2 = B2[bl*chunksize:(bl+1)*chunksize].dot(nnm1_in.T)
                        C2.data[C2.data<0.1]=0
                        C2.eliminate_zeros()
                    else:
                        C = B[bl*chunksize:(bl+1)*chunksize].dot(nnm2_in.T)
                        C.data[C.data<0.1]=0
                        C.eliminate_zeros()

                        C2 = nnm1_in[bl*chunksize:(bl+1)*chunksize].dot(B2.T)
                        C2.data[C2.data<0.1]=0
                        C2.eliminate_zeros()

                    X = C.multiply(C2)
                    X.data[:] = X.data**0.5
                    del C; del C2; gc.collect()
                    D = sp.sparse.vstack((D,X))
                    del X; gc.collect()

                if not R:
                    D = D.T
                    D = D.tocsr()

                mdata['xsim']=D

                D1 = _sparse_knn(D,k1).tocsr()
                D2 = _sparse_knn(D.T,k1).tocsr()

            else:
                D1 = h2m
                D2 = m2h
                if k1 < K:
                    print('Redoing sparse kNN selection...')
                    D1 = _sparse_knn(D1,k1).tocsr()
                    D2 = _sparse_knn(D2,k1).tocsr()

            try:
                DS[I][J]=D1
            except:
                DS[I]={}
                DS[I][J]=D1

            try:
                DS[J][I]=D2
            except:
                DS[J]={}
                DS[J][I]=D2

    ROWS=[]
    for I in range(n):
        ROW = []
        ROWt = []
        for J in range(n):
            if I != J:
                ROW.append(DS[I][J])
                ROWt.append(DS[J][I])



        nnm = sams[I].adata.obsp['connectivities']

        row = sp.sparse.hstack(ROW)
        rowt = sp.sparse.vstack(ROWt)
        x = 1-row.sum(1).A.flatten()/k1/(n-1)

        #onemode projection
        s = row.sum(1).A
        s[s==0]=1
        s2 = rowt.sum(1).A
        s2[s2==0]=1
        proj = row.multiply(1/s).dot(rowt.multiply(1/s2)).tocsr()

        #find rows with abnormally small # edges in projection
        z = proj.copy()
        z.data[:]=1
        idx = np.where(z.sum(1).A.flatten()>=k1)[0]

        #copy nearest neighbor graph
        omp=nnm.copy().astype('float')
        omp.data[:]=1


        #renormalize edge weights to max 1
        s=proj.max(1).A; s[s==0]=1
        proj = proj.multiply(1/s).tocsr()

        #find edges in original graph and only choose ones from nonzero rows
        X,Y=omp.nonzero()
        X2 = X[np.in1d(X,idx)]
        Y2 = Y[np.in1d(X,idx)]

        omp=omp.tolil()
        omp[X2,Y2] = np.vstack((proj[X2,Y2].A.flatten(),np.ones(X2.size)*0.3)).max(0)
        omp=omp.tocsr()
        omp = omp.multiply(x[:,None]).tocsr()
        ROW.insert(I,omp)
        ROWS.append(sp.sparse.hstack(ROW))
    NNM = sp.sparse.vstack((ROWS)).tolil()

    NNM.setdiag(0)
    #"""

    print('Concatenating SAM objects...')
    sam3 = _concatenate_sam(sams,NNM,mdata['ortholog_pairs'])

    sam3.adata.uns['mdata'] = mdata

    if umap:
        print('Computing UMAP projection...')
        sc.tl.umap(sam3.adata,min_dist=0.1)

    return sam3

def _concatenate_sam(sams,nnm,op):

    acns=[]
    obsks=[]
    for i in range(len(sams)):
        acns.append(np.array(list(sams[i].adata.obs_names)))
        obsks.append(np.array(sams[i].adata.obs_keys()))
    obsk = list(set(obsks[0]).intersection(*obsks))

    acn = np.concatenate(acns)

    gST = op[:,0].astype('object') + ';' + op[:,1].astype('object')

    xx = sp.sparse.csr_matrix((acn.size,gST.size))
    sam=SAM(counts = (xx,gST,acn))
    sam.adata.uns['neighbors'] = {}
    nnm.setdiag(0)
    nnm=nnm.tocsr()
    nnm.eliminate_zeros()
    sam.adata.obsp['connectivities'] = nnm
    sam.adata.uns['nnm'] = sam.adata.obsp['connectivities']
    sam.adata.obsp['connectivities'] = sam.adata.uns['nnm']
    sam.adata.uns['neighbors']['params'] = {'n_neighbors':15,'method':'umap','use_rep':'X','metric':'euclidean'}

    for k in obsk:
        ann = []
        for i in range(len(sams)):
            ann.append(sams[i].get_labels(k))
        sam.adata.obs[k] = pd.Categorical(np.concatenate(ann))


    a = []
    for i in range(len(sams)):
        a.extend(['batch'+str(i+1)]*sams[i].adata.shape[0])
    sam.adata.obs['batch'] = pd.Categorical(np.array(a))
    return sam


def to_vn(op):
    return np.array(list(op[:,0].astype('object')+';'+op[:,1].astype('object')))
def to_vo(op):
    return np.vstack((ut.extract_annotation(op,None,';'))).T

def _map_features_un(A,B,sam1,sam2,thr=1e-6):
    i1 = np.where(A.columns=='10')[0][0]
    i3 = np.where(A.columns=='11')[0][0]

    inA = np.array(list(A.index))
    inB = np.array(list(B.index))

    gn1 = np.array(list(sam1.adata.var_names))
    gn2 = np.array(list(sam2.adata.var_names))

    gn1 = gn1[np.in1d(gn1,inA)]
    gn2 = gn2[np.in1d(gn2,inB)]

    A = A.iloc[np.in1d(inA,gn1),:]
    B = B.iloc[np.in1d(inB,gn2),:]

    inA2 = np.array(list(A.iloc[:,0]))
    inB2 = np.array(list(B.iloc[:,0]))

    A = A.iloc[np.in1d(inA2,gn2),:]
    B = B.iloc[np.in1d(inB2,gn1),:]

    gn = np.append(gn1,gn2)
    gnind = pd.DataFrame(data = np.arange(gn.size)[None,:],columns=gn)

    A.index = pd.Index(gnind[A.index].values.flatten())
    B.index = pd.Index(gnind[B.index].values.flatten())
    A.iloc[:,0] = gnind[A.iloc[:,0].values.flatten()].values.flatten()
    B.iloc[:,0] = gnind[B.iloc[:,0].values.flatten()].values.flatten()

    Arows=np.vstack((A.index,A.iloc[:,0],A.iloc[:,i3])).T
    Arows = Arows[A.iloc[:,i1].values.flatten()<=thr,:]
    gnnm1 = sp.sparse.lil_matrix((gn.size,)*2)
    gnnm1[Arows[:,0].astype('int32'),Arows[:,1].astype('int32')] = Arows[:,2]#-np.log10(Arows[:,2]+1e-200)

    Brows=np.vstack((B.index,B.iloc[:,0],B.iloc[:,i3])).T
    Brows = Brows[B.iloc[:,i1].values.flatten()<=thr,:]
    gnnm2 = sp.sparse.lil_matrix((gn.size,)*2)
    gnnm2[Brows[:,0].astype('int32'),Brows[:,1].astype('int32')] = Brows[:,2]#-np.log10(Brows[:,2]+1e-200)

    gnnm = (gnnm1+gnnm2).tocsr()
    gnnms = (gnnm+gnnm.T)/2
    gnnm.data[:]=1
    gnnms = gnnms.multiply(gnnm).multiply(gnnm.T).tocsr()
    return gnnms,gn1,gn2

def _filter_gnnm(gnnm,thr=0.25):
    x,y = gnnm.nonzero()
    mas = gnnm.max(1).A.flatten()
    gnnm4=gnnm.copy()
    gnnm4.data[gnnm4[x,y].A.flatten()<mas[x]*thr]=0
    gnnm4.eliminate_zeros()
    x,y = gnnm4.nonzero()
    z = gnnm4.data
    gnnm4=gnnm4.tolil()
    gnnm4[y,x] = z
    gnnm4=gnnm4.tocsr()
    return gnnm4

def calculate_blast_graph(id1,id2,f_maps = 'maps/',
                          namesA = None, namesB = None,
                          eval_thr = 1e-6, ew_thr = 0.25, reciprocate=False):
    
    if os.path.exists(f_maps+'{}{}'.format(id1,id2)):
        fA = f_maps + '{}{}/{}_to_{}.txt'.format(id1,id2,id1,id2)
        fB = f_maps + '{}{}/{}_to_{}.txt'.format(id1,id2,id2,id1)    
    elif os.path.exists(f_maps+'{}{}'.format(id2,id1)):
        fA = f_maps + '{}{}/{}_to_{}.txt'.format(id2,id1,id1,id2)
        fB = f_maps + '{}{}/{}_to_{}.txt'.format(id2,id1,id2,id1)        
    else:
        raise FileExistsError('BLAST mapping tables with the input IDs ({} and {}) not found in the specified path.'.format(id1,id2))
    
    if namesA is not None:
        groupsA = {}
        for i in range(len(namesA)):
            name = namesA[i][1]                        
            L = groupsA.get(name,[])
            L.append(namesA[i][0])
            groupsA[name] = L
    else:
        groupsA = None

    if namesB is not None:
        groupsB = {}
        for i in range(len(namesB)):
            name = namesB[i][1]                        
            L = groupsB.get(name,[])
            L.append(namesB[i][0])
            groupsB[name] = L
    else:
        groupsB = None    
        
    A=pd.read_csv(fA,sep='\t',header=None,index_col=0)
    B=pd.read_csv(fB,sep='\t',header=None,index_col=0)

    A.columns=A.columns.astype('<U100')
    B.columns=B.columns.astype('<U100')

    A=A[A.index.astype('str')!='nan']
    A=A[A.iloc[:,0].astype('str')!='nan']
    B=B[B.index.astype('str')!='nan']
    B=B[B.iloc[:,0].astype('str')!='nan']


    A.index = id1+'_'+A.index.astype('str').astype('object')
    B.iloc[:,0] = id1 +'_'+B.iloc[:,0].values.flatten().astype('str').astype('object')

    B.index = id2+'_'+B.index.astype('str').astype('object')
    A.iloc[:,0] = id2+'_' + A.iloc[:,0].values.flatten().astype('str').astype('object')

    i1 = np.where(A.columns=='10')[0][0]
    i3 = np.where(A.columns=='11')[0][0]

    inA = np.array(list(A.index))
    inB = np.array(list(B.index))

    inA2 = np.array(list(A.iloc[:,0]))
    inB2 = np.array(list(B.iloc[:,0]))
    gn1 = np.unique(np.append(inB2,inA))
    gn2 = np.unique(np.append(inA2,inB))
    gn = np.append(gn1,gn2)
    gnind = pd.DataFrame(data = np.arange(gn.size)[None,:],columns=gn)

    A.index = pd.Index(gnind[A.index].values.flatten())
    B.index = pd.Index(gnind[B.index].values.flatten())
    A.iloc[:,0] = gnind[A.iloc[:,0].values.flatten()].values.flatten()
    B.iloc[:,0] = gnind[B.iloc[:,0].values.flatten()].values.flatten()

    Arows=np.vstack((A.index,A.iloc[:,0],A.iloc[:,i3])).T
    Arows = Arows[A.iloc[:,i1].values.flatten()<=eval_thr,:]
    gnnm1 = sp.sparse.lil_matrix((gn.size,)*2)
    gnnm1[Arows[:,0].astype('int32'),Arows[:,1].astype('int32')] = Arows[:,2]#-np.log10(Arows[:,2]+1e-200)

    Brows=np.vstack((B.index,B.iloc[:,0],B.iloc[:,i3])).T
    Brows = Brows[B.iloc[:,i1].values.flatten()<=eval_thr,:]
    gnnm2 = sp.sparse.lil_matrix((gn.size,)*2)
    gnnm2[Brows[:,0].astype('int32'),Brows[:,1].astype('int32')] = Brows[:,2]#-np.log10(Brows[:,2]+1e-200)

    gnnm = (gnnm1+gnnm2).tocsr()
    gnnms = (gnnm+gnnm.T)/2
    if reciprocate:
        gnnm.data[:]=1
        gnnms = gnnms.multiply(gnnm).multiply(gnnm.T).tocsr()
    gnnm=gnnms
    
    for n,groups,ids,g in zip(['A','B'],[groupsA,groupsB],[id1,id2],[gn1,gn2]):
        if groups is not None:
            if isinstance(groups,dict):
                xdim = len(list(groups.keys()))
                for I in range(2):
                    print('Coarsening gene connectivity graph using labels `{}`, round {}.'.format(ids,I))
                    X=[]
                    Y=[]
                    D=[]
                    for i,k in enumerate(groups.keys()):
                        if len(groups[k])>1:
                            f = np.in1d(gn,[ids+'_'+x for x in groups[k]])
                            if f.sum()>0:
                                z = gnnm[f].max(0).A.flatten()
                            else:
                                z = np.array([])
                        else:
                            f = gn==ids+'_'+groups[k][0]
                            if np.any(f):
                                z = gnnm[f].A.flatten()
                            else:
                                z = np.array([])
                        y = z.nonzero()[0]
                        d = z[y]                                                
                        x = np.ones(y.size)*i
                        X.extend(x); Y.extend(y); D.extend(d)
                    if n == 'A':
                        xa,ya = gnnm[gn1.size:].nonzero()
                        da = gnnm[gn1.size:].data
                        xa+=xdim
                    else:
                        xa,ya = gnnm[:gn1.size].nonzero()
                        da = gnnm[:gn1.size].data
                        X = list(np.array(X)+gn1.size)
                    X.extend(xa)
                    Y.extend(ya)
                    D.extend(da)
                    gnnm = sp.sparse.coo_matrix((D,(X,Y)),shape = (xdim+gnnm.shape[0]-g.size,gnnm.shape[1])).T.tocsr()
                g = np.array([ids+'_'+x for x in list(groups.keys())])
                if n == 'A':
                    gn1 = g
                else:
                    gn2 = g
                gn = np.append(gn1,gn2)
            else:
                raise TypeError('Gene groupings ({}) must be in dictionary form.'.format(n))
    f = gnnm.sum(1).A.flatten()!=0
    gn = gn[f]
    gn1 = gn1[np.in1d(gn1,gn)]
    gn2 = gn2[np.in1d(gn2,gn)]
    gnnm=gnnm[f,:][:,f]
    gnnm = _filter_gnnm(gnnm,thr=ew_thr)
    
    return gnnm,gn1,gn2,gn
    
    

def get_pairs(sam1,sam2,gnnm,gn1,gn2,NOPs1=2,NOPs2=5):
    #gnnm = filter_gnnm(gnnm)
    su = gnnm.max(1).A
    su[su==0]=1
    gnnm=gnnm.multiply(1/su).tocsr()
    W1 = sam1.adata.var['weights'][gn1].values
    W2 = sam2.adata.var['weights'][gn2].values
    W = np.append(W1,W2)
    W[W<0.]=0
    W[W>0.]=1

    if NOPs1 == 0 and NOPs2 == 0:
        B = gnnm.multiply(W[None,:]).multiply(W[:,None]).tocsr()
        B.eliminate_zeros()
    else:
        B = _sparse_knn(gnnm.multiply(W[None,:]).multiply(W[:,None]).tocsr(),NOPs1).tocsr()
        B = _sparse_knn(B.T,NOPs2).T.tocsr()
        B.eliminate_zeros()

        x,y = B.nonzero()
        data = np.vstack((B[x,y].A.flatten(),B[y,x].A.flatten())).max(0)
        B=sp.sparse.lil_matrix(B.shape)
        B[x,y]=data
        B[y,x]=data
        B=B.tocsr()
    return B

def compute_csim(sam3,key,X=None):
    cl1=np.array(list(sam3.adata.obs[key].values[sam3.adata.obs['batch']=='batch1']))
    clu1 = np.unique(cl1)
    cl2=np.array(list(sam3.adata.obs[key].values[sam3.adata.obs['batch']=='batch2']))
    clu2 = np.unique(cl2)

    clu1s=np.array(list('batch1_'+clu1.astype('str').astype('object')))
    clu2s=np.array(list('batch2_'+clu2.astype('str').astype('object')))
    cl = np.array(list(sam3.adata.obs['batch'].values.astype('object')+'_'+sam3.adata.obs[key].values.astype('str').astype('object')))

    CSIM1 = np.zeros((clu1s.size,clu2s.size))
    if X is None:
        X=sam3.adata.obsp['connectivities'].copy()

    for i,c1 in enumerate(clu1s):
        for j,c2 in enumerate(clu2s):
            CSIM1[i,j] = np.append(X[cl==c1,:][:,cl==c2].sum(1).A.flatten(),
                                   X[cl==c2,:][:,cl==c1].sum(1).A.flatten()).mean()
    CSIMth = CSIM1
    s1 = CSIMth.sum(1).flatten()[:,None]
    s2 = CSIMth.sum(0).flatten()[None,:]
    s1[s1==0]=1
    s2[s2==0]=1
    CSIM1 = CSIMth/s1
    CSIM2 = CSIMth/s2
    CSIM = ((CSIM1 * CSIM2)**0.5)

    return CSIM, clu1, clu2, CSIMth

def avg_as(s):
    return np.append(s.adata.obsp['connectivities'][np.array(s.adata.obs['batch'])=='batch1',:][:,np.array(s.adata.obs['batch'])=='batch2'].sum(1).A.flatten(),
              s.adata.obsp['connectivities'][np.array(s.adata.obs['batch'])=='batch2',:][:,np.array(s.adata.obs['batch'])=='batch1'].sum(1).A.flatten())


def _parallel_init(ipl1x,isc1x,ipairs,ign1O,ign2O,iT2,iCORR,icorr_mode):
    global pl1
    global sc1
    global p
    global gn1O
    global gn2O
    global T2
    global CORR
    global corr_mode
    pl1 = ipl1x
    sc1 = isc1x
    p = ipairs
    gn1O = ign1O
    gn2O = ign2O
    T2 = iT2
    CORR = iCORR
    corr_mode = icorr_mode
def _refine_corr_parallel(sam1,sam2,st,gnnm,gn1,gn2,corr_mode='pearson', THR=0, use_seq = False,
                T1=0.0,T2=0.0):

    import scipy as sp
    gn=np.append(gn1,gn2)


    w1=sam1.adata.var['weights'][gn1].values
    w2=sam2.adata.var['weights'][gn2].values
    w = np.append(w1,w2)

    w[w>T1]=1
    w[w<1]=0
    ix=np.array(['a']*gn1.size+['b']*gn2.size)
    gnO = gn[w>0]
    ix = ix[w>0]
    gn1O = gnO[ix=='a']
    gn2O = gnO[ix=='b']#
    gnnmO = gnnm[w>0,:][:,w>0]
    x,y = gnnmO.nonzero()
    pairs = np.unique(np.sort(np.vstack((x,y)).T,axis=1),axis=0)
    pairs[pairs>=gn1O.size]=pairs[pairs>=gn1O.size] - gn1O.size


    idx1 = np.where(st.adata.obs['batch']=='batch1')[0]
    idx2 = np.where(st.adata.obs['batch']=='batch2')[0]
    nnm = st.adata.obsp['connectivities']
    x1 = sam1.adata[:,gn1O].X.tocsc().astype('float16')#[:,pairs[:,0]]
    x2 = sam2.adata[:,gn2O].X.tocsc().astype('float16')#[:,pairs[:,1]]

    nnm1 = nnm[:,idx1].astype('float16')
    nnm2 = nnm[:,idx2].astype('float16')


    s1 = nnm1.sum(1).A; s1[s1<1e-3]=1; s1=s1.flatten()[:,None]
    s2 = nnm2.sum(1).A; s2[s2<1e-3]=1; s2=s2.flatten()[:,None]

    pl1x = nnm1.dot(x1).multiply(1/s1).tocsc()

    sc1x = nnm2.dot(x2).multiply(1/s2).tocsc()

    CORR={};


    from multiprocessing import Pool, Manager

    CORR = Manager().dict()
    p=pairs
    pl1 = pl1x
    sc1 = sc1x
    pc_chunksize = pl1.shape[1]//os.cpu_count()+1

    pool = Pool(os.cpu_count(),_parallel_init,[pl1,sc1,p,gn1O,gn2O,T2,CORR,corr_mode])
    try:
        pool.map(_parallel_wrapper,range(p.shape[0]),chunksize=pc_chunksize)
    finally:
        pool.close()
        pool.join()

    CORR = CORR._getvalue()
    for k in CORR.keys():
        CORR[k] = 0 if CORR[k] < THR else CORR[k]

    gnnm2 = gnnm.multiply(w[:,None]).multiply(w[None,:]).tocsr()
    x,y = gnnm2.nonzero()
    pairs = np.unique(np.sort(np.vstack((x,y)).T,axis=1),axis=0)

    CORR = np.array([CORR[x] for x in to_vn(gn[pairs])])

    gnnm3 = sp.sparse.lil_matrix(gnnm.shape)

    if use_seq:
        gnnm3[pairs[:,0],pairs[:,1]] = CORR*gnnm2[pairs[:,0],pairs[:,1]].A.flatten()
        gnnm3[pairs[:,1],pairs[:,0]] = CORR*gnnm2[pairs[:,1],pairs[:,0]].A.flatten()
    else:
        gnnm3[pairs[:,0],pairs[:,1]] = CORR#*gnnm2[x,y].A.flatten()
        gnnm3[pairs[:,1],pairs[:,0]] = CORR#*gnnm2[x,y].A.flatten()

    gnnm3=gnnm3.tocsr()
    gnnm3.eliminate_zeros()

    return gnnm3,CORR

def _parallel_wrapper(j):
    j1,j2 = p[j,0],p[j,1]

    pl1d = pl1.data[pl1.indptr[j1]:pl1.indptr[j1+1]]
    pl1i = pl1.indices[pl1.indptr[j1]:pl1.indptr[j1+1]]

    sc1d = sc1.data[sc1.indptr[j2]:sc1.indptr[j2+1]]
    sc1i = sc1.indices[sc1.indptr[j2]:sc1.indptr[j2+1]]

    x= np.zeros(pl1.shape[0])
    x[pl1i]=pl1d
    y= np.zeros(sc1.shape[0])
    y[sc1i]=sc1d


    ha = gn1O[j1]+';'+gn2O[j2]
    iz=np.logical_or(x>T2,y>T2)
    izf=np.logical_and(x>T2,y>T2)

    if izf.sum()>0:
        if corr_mode == 'pearson':
            CORR[ha] = np.corrcoef(x[iz],y[iz])[0,1]
        else:
            print('Correlation mode not recognized.');
            return;
    else:
        CORR[ha]=0

def refine_corr(sam1,sam2,st,gnnm,gn1,gn2,corr_mode='pearson',THR=0,use_seq=False,
                     T1=0.25,T2=0,NCLUSTERS = 1):
    #import networkx as nx
    gn=np.append(gn1,gn2)

    x,y=gnnm.nonzero()
    cl = sam1.leiden_clustering(gnnm,res=0.5)
    ix = np.argsort(cl)
    NGPC = gn.size//NCLUSTERS+1
    ixs = []
    for i in range(NCLUSTERS):
        ixs.append(np.sort(ix[i*NGPC : (i+1)*NGPC]))

    assert np.concatenate(ixs).size == gn.size

    GNNMSUBS=[]
    CORRSUBS=[]
    GNSUBS=[]
    for i in range(len(ixs)):
        ixs[i] = np.unique(np.append(ixs[i],gnnm[ixs[i],:].nonzero()[1]))
        gnnm_sub = gnnm[ixs[i],:][:,ixs[i]]
        gnsub = gn[ixs[i]]
        gn1_sub = gn1[np.in1d(gn1,gnsub)]
        gn2_sub = gn2[np.in1d(gn2,gnsub)]
        gnnm2_sub,CORR_sub = _refine_corr_parallel(sam1,sam2,st,gnnm_sub,gn1_sub,gn2_sub,corr_mode=corr_mode,
                                                        THR=THR,use_seq=use_seq,T1=T1,T2=T2)
        GNNMSUBS.append(gnnm2_sub)
        CORRSUBS.append(CORR_sub)
        GNSUBS.append(gnsub)
        gc.collect()
    I=[]
    P=[]
    for i in range(len(GNNMSUBS)):
        I.append(np.unique(np.sort(np.vstack((GNNMSUBS[i].nonzero())).T,axis=1),axis=0))
        P.append(GNSUBS[i][I[-1]])

    GN = pd.DataFrame(data=np.arange(gn.size)[None,:],columns=gn)
    gnnm3 = sp.sparse.lil_matrix(gnnm.shape)
    for i in range(len(I)):
        x,y = GN[P[i][:,0]].values.flatten(),GN[P[i][:,1]].values.flatten()
        gnnm3[x,y] = GNNMSUBS[i][I[i][:,0],I[i][:,1]].A.flatten()

    gnnm3 = gnnm3.tocsr()
    x,y = gnnm3.nonzero()
    #gnnm3[y,x]=gnnm3.data
    gnnm3=gnnm3.tolil()
    gnnm3[y,x]=gnnm3[x,y].A.flatten()
    return gnnm3.tocsr()

def prepare_SAMap_loadings(sam,npcs=300):
    ra = sam.adata.uns['run_args']
    preprocessing = ra.get('preprocessing','StandardScaler')
    weight_PCs = ra.get('weight_PCs',False)
    A,_ = sam.calculate_nnm(n_genes = sam.adata.shape[1],preprocessing=preprocessing,npcs=npcs,weight_PCs=weight_PCs,
                      sparse_pca=True,update_manifold=False)
    sam.adata.varm['PCs_SAMap'] = A

class Samap(object):
    def __init__(self,sam1,sam2,gnnm,gn1,gn2):
        self.sam1=sam1
        self.sam2=sam2
        self.gnnm=gnnm
        self.gn1=gn1
        self.gn2=gn2


    def run(self,NUMITERS=3,NOPs1=4,NOPs2=8,NH1=2,NH2=2,K=20,NCLUSTERS=1,use_seq=False):
        sam1=self.sam1
        sam2=self.sam2
        gnnm=self.gnnm
        gn1=self.gn1
        gn2=self.gn2
        gn=np.append(gn1,gn2)

        self.max_score = 0
        coarsen = False
        gnnm2 = get_pairs(sam1,sam2,gnnm,gn1,gn2,NOPs1=NOPs1,NOPs2=NOPs2)
        sam_def = samap([sam1,sam2],gnnm2,gn, umap=False, NH1=NH1, NH2=NH2,
                                    coarsen=coarsen,K=K)
        self.sam_def = sam_def
        sam4=sam_def

        _, _, _, CSIMth = compute_csim(sam4,'leiden_clusters')
        new = CSIMth.flatten()
        old=20

        self.SCORES = [np.abs(new-old).max()]
        self.SCORE_VEC=[new]
        self.GNNMS_corr=[None]
        self.GNNMS_pruned=[gnnm2]
        i=0
        self.GNNMS_nnm=[sam_def.adata.obsp['connectivities']]
        BURN_IN = 0
        FLAG = True
        while i < BURN_IN+1:
            print('ITERATION: ' + str(i),
                  '\nAverage alignment score (A.S.): ',avg_as(sam4).mean(),
                  '\nMax A.S. improvement:',np.max(new-old),
                  '\nMin A.S. improvement:',np.min(new-old))
            i+=1
            sam_def=sam4
            gc.collect()
            print('Calculating gene-gene correlations in the homology graph...')
            gnnmu = refine_corr(sam1,sam2,sam_def,gnnm,gn1,gn2, THR = 0, use_seq=use_seq,corr_mode='pearson',T1=0,T2=0,NCLUSTERS=NCLUSTERS)

            self.GNNMS_corr.append(gnnmu)
            self.gnnmu = gnnmu

            gnnm2  = get_pairs(sam1,sam2,gnnmu,gn1,gn2,NOPs1=NOPs1,NOPs2=NOPs2)
            self.GNNMS_pruned.append(gnnm2)

            gc.collect()

            sam4 = samap([sam1,sam2],gnnm2,gn,umap=False,K=K,NH1=NH1,NH2=NH2,coarsen=coarsen)
            self.samap = sam4
            self.GNNMS_nnm.append(sam4.adata.uns['nnm'])

            _, _, _, CSIMth = compute_csim(sam4,'leiden_clusters')
            old=new
            new=CSIMth.flatten()
            self.SCORES.append(np.abs(new-old).max())
            self.SCORE_VEC.append(new)

            self.last_score = self.SCORES[-1]
            
            if i==BURN_IN+1 and FLAG:
                FLAG=False
                BURN_IN += NUMITERS
                coarsen=True
                
            gc.collect()

        self.final_sam=sam4
        self.final_sam.adata.var['edge_weights'] = self.final_sam.adata.uns['mdata']['edge_weights']

        self.ITER_DATA = (self.GNNMS_nnm,self.GNNMS_corr,self.GNNMS_pruned,self.SCORE_VEC)

def get_mu_std(sam3,sam1,sam2,knn=False):
    g1,g2 = ut.extract_annotation(sam3.adata.var_names,0,';'),ut.extract_annotation(sam3.adata.var_names,1,';')
    if knn:
        mu1, var1 = sf.mean_variance_axis(sam1.adata[:,g1].layers['X_knn_avg'], axis=0)
        mu2, var2 = sf.mean_variance_axis(sam2.adata[:,g2].layers['X_knn_avg'], axis=0)
    else:
        mu1, var1 = sf.mean_variance_axis(sam1.adata[:,g1].X, axis=0)
        mu2, var2 = sf.mean_variance_axis(sam2.adata[:,g2].X, axis=0)
    var1[var1==0]=1
    var2[var2==0]=1
    var1=var1**0.5
    var2=var2**0.5
    return mu1,var1,mu2,var2
    
def q(x):
    return np.array(list(x))

class GenePairFinder(object):
    def __init__(self,s1,s2,s3,k1='leiden_clusters',k2='leiden_clusters'):
        self.s1=s1
        self.s2=s2
        self.s3=s3
        self.s1.dispersion_ranking_NN();
        self.s2.dispersion_ranking_NN();
        mu1,v1,mu2,v2 = get_mu_std(self.s3,self.s1,self.s2)
        self.mu1=mu1
        self.v1=v1
        self.mu2=mu2
        self.v2=v2
        self.k1 = k1
        self.k2 = k2
        self.find_markers()
    
    def find_markers(self):
        print('Finding cluster-specific markers in {} and {}.'.format(self.k1,self.k2))
        find_cluster_markers(self.s1,self.k1)
        find_cluster_markers(self.s2,self.k2)
        
    def find_genes(self,n1,n2,w1t=0.2,w2t=0.2,n_pairs=500,n_genes=1000,thr=1e-2):
        assert n1 in q(self.s1.adata.obs[self.k1])
        assert n2 in q(self.s2.adata.obs[self.k2])
        m = self.find_link_genes(n1,n2,w1t=w1t,w2t=w2t,n_pairs=n_pairs)
        G = q(self.s3.adata.var_names[np.argsort(-m)[:n_genes]])
        G1 = substr(G,';',0)
        G2 = substr(G,';',1)
        G = q(G[np.logical_and(q(self.s1.adata.var[self.k1+';;'+n1+'_pval'][G1]<thr),q(self.s2.adata.var[self.k2+';;'+n2+'_pval'][G2]<thr))])
        G1 = substr(G,';',0)
        G2 = substr(G,';',1)
        _,ix1 = np.unique(G1,return_index=True)
        _,ix2 = np.unique(G2,return_index=True)
        G1 = G1[np.sort(ix1)]
        G2 = G2[np.sort(ix2)]
        return G,G1,G2
        
        
    def find_link_genes(self,c1,c2,w1t=0.35,w2t=0.35,knn=False,n_pairs = 250,expr_thr = 0.05):
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
        g1,g2 = ut.extract_annotation(sam3.adata.var_names,0,';'),ut.extract_annotation(sam3.adata.var_names,1,';')
        if knn:
            X1 = _sparse_sub_standardize(sam1.adata[:,g1].layers['X_knn_avg'][x1==c1,:],mu1,std1)
            X2 = _sparse_sub_standardize(sam2.adata[:,g2].layers['X_knn_avg'][x2==c2,:],mu2,std2)
        else:
            X1 = _sparse_sub_standardize(sam1.adata[:,g1].X[x1==c1,:],mu1,std1)
            X2 = _sparse_sub_standardize(sam2.adata[:,g2].X[x2==c2,:],mu2,std2)
    
    
        X1 = _sparse_sub_standardize(X1,mu1,std1,rows=True).tocsr()
        X2 = _sparse_sub_standardize(X2,mu2,std2,rows=True).tocsr()
    
        a,b = sam3.adata.obsp['connectivities'][:sam1.adata.shape[0],sam1.adata.shape[0]:][x1==c1,:][:,x2==c2].nonzero()
        c,d = sam3.adata.obsp['connectivities'][sam1.adata.shape[0]:,:sam1.adata.shape[0]][x2==c2,:][:,x1==c1].nonzero()
    
        pairs = np.unique(np.vstack((np.vstack((a,b)).T,np.vstack((d,c)).T)),axis=0)
    
        Z = X1[pairs[:,0],:].multiply(X2[pairs[:,1],:]).tocsr()
        Z.data[:] /= X1.shape[1]
        X1.data[:]=1
        X2.data[:]=1
        min_expr = (X1.mean(0).A.flatten()>expr_thr)*(X2.mean(0).A.flatten()>expr_thr)
    
        w1 = sam1.adata.var['weights'][g1].values.copy()
        w2 = sam2.adata.var['weights'][g2].values.copy()
        w1[w1<w1t]=0
        w2[w2<w2t]=0
        w1[w1>0]=1
        w2[w2>0]=1
    
        Z = _sparse_knn(Z.T,n_pairs)
        val = _knndist(Z,n_pairs).T
        mu = val.mean(0)*w1*w2*min_expr
        return mu

def substr(x, s="_", ix=None,obj=False):
    m = []
    if ix is not None:
        for i in range(len(x)):
            f = x[i].split(s)
            ix = min(len(f) - 1, ix)
            m.append(f[ix])
        return np.array(m).astype('object') if obj else np.array(m)
    else:
        ms = []
        ls = []
        for i in range(len(x)):
            f = x[i].split(s)
            m = []
            for ix in range(len(f)):
                m.append(f[ix])
            ms.append(m)
            ls.append(len(m))
        ml = max(ls)
        for i in range(len(ms)):
            ms[i].extend([""] * (ml - len(ms[i])))
            if ml - len(ms[i]) > 0:
                ms[i] = np.concatenate(ms[i])
        ms = np.vstack(ms)
        if obj:
            ms=ms.astype('object')
        MS = []
        for i in range(ms.shape[1]):
            MS.append(ms[:, i])
            return MS
            
def _knndist(nnma,k):
    x, y = nnma.nonzero()
    data = nnma.data
    xc,cc = np.unique(x,return_counts=True)
    cc2=np.zeros(nnma.shape[0],dtype='int')
    cc2[xc]=cc
    cc=cc2
    newx=[]
    newdata=[]
    for i in range(nnma.shape[0]):
        newx.extend([i]*k)
        newdata.extend(list(data[x==i])+[0]*(k-cc[i]))
    data=np.array(newdata)
    val = data.reshape((nnma.shape[0], k))
    return val 
def find_cluster_markers(sam,key,layer=None,inplace=True):
    sc.tl.rank_genes_groups(sam.adata_raw,key,method='wilcoxon',n_genes=sam.adata.shape[1],use_raw=True,layer=layer)
    NAMES = pd.DataFrame(sam.adata_raw.uns['rank_genes_groups']['names'])
    PVALS = pd.DataFrame(sam.adata_raw.uns['rank_genes_groups']['pvals'])
    SCORES = pd.DataFrame(sam.adata_raw.uns['rank_genes_groups']['scores'])
    if not inplace:
        return NAMES,PVALS,SCORES
    for i in range(SCORES.shape[1]):
        names = NAMES.iloc[:,i]
        scores = SCORES.iloc[:,i]
        pvals = PVALS.iloc[:,i]
        pvals[scores<0]=1.0
        scores[scores<0]=0
        pvals = q(pvals)
        scores = q(scores)
        sam.adata.var[key+';;'+SCORES.columns[i]] = pd.DataFrame(data = scores[None,:],columns=names)[sam.adata.var_names].values.flatten()
        sam.adata.var[key+';;'+SCORES.columns[i]+'_pval'] = pd.DataFrame(data = pvals[None,:],columns=names)[sam.adata.var_names].values.flatten()
        
def _sparse_sub_standardize(X,mu,var,rows=False):
    x,y = X.nonzero()
    if not rows:
        Xs = X.copy()
        Xs.data[:] = (X.data - mu[y])/var[y]
    else:
        mu, var = sf.mean_variance_axis(X, axis=1)
        var=var**0.5
        var[var==0]=1
        Xs = X.copy()
        Xs.data[:] = (X.data - mu[x])/var[x]
    Xs.data[Xs.data<0]=0
    Xs.eliminate_zeros()
    return Xs
