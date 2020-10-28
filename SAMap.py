import gc
import pandas as pd
import hnswlib
import scipy as sp
from samalg import SAM
import numpy as np
import samalg.utilities as ut
from sklearn.preprocessing import StandardScaler
import scanpy as sc

__version__ = '0.2.0'

def united_proj(wpca1,wpca2,k=20, metric='correlation', sigma=500, ef = 200, M = 48):

    print('Running hsnwlib (2) ')

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

def mapping_window(sam1,sam2,gnnm,gn,K=20):

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

    mu1 = s1.mean(0).A.flatten()[None,:]
    mu2 = s2.mean(0).A.flatten()[None,:]

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

    b1,b2 = united_proj(wpca1, wpca2, k=k)

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
    return output_dict

def sparse_knn(D,k):
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

def sparse_knn_ks(D,ks):
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


def smart_expand(nnm,cl,NH=3):
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
        s = sparse_knn_ks(S[i],K).tocsr()
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
                mdata = mapping_window(sam,sam2,gnnm,gn,K=K)

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
                nnm1_in = smart_expand(nnm,K1,NH=NH1)
                nnm1_in.data[:]=1

                print('Out-neighbor smart expansion 2')
                nnm = sam2.adata.obsp['connectivities'].copy()
                nnm2_out = nnm
                nnm2_in = smart_expand(nnm,K2,NH=NH2)
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

                D1 = sparse_knn(D,k1).tocsr()
                D2 = sparse_knn(D.T,k1).tocsr()

            else:
                D1 = h2m
                D2 = m2h
                if k1 < K:
                    print('Redoing sparse kNN selection...')
                    D1 = sparse_knn(D1,k1).tocsr()
                    D2 = sparse_knn(D2,k1).tocsr()

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
    sam3 = concatenate_sam(sams,NNM,mdata['ortholog_pairs'])

    sam3.adata.uns['mdata'] = mdata

    if umap:
        print('Computing UMAP projection...')
        sc.tl.umap(sam3.adata,min_dist=0.1)

    return sam3

def concatenate_sam(sams,nnm,op):

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

