import SAMap as sm
from samalg import SAM
import numpy as np
import pandas as pd
import samalg.utilities as ut
import scipy as sp
import os

def to_vn(op):
    return np.array(list(op[:,0].astype('object')+';'+op[:,1].astype('object')))
def to_vo(op):
    return np.vstack((ut.extract_annotation(op,None,';'))).T

def map_features_un(A,B,sam1,sam2,thr=1e-6):
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

def filter_gnnm(gnnm,thr=0.25):
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

def get_SAMap_init(f1,fA,f2,fB,id1='A',id2='B'):
    sam1=SAM()
    sam2=SAM()
    print('Loading data 1')
    sam1.load_data(f1,calculate_avg=False)
    print('Loading data 2')
    sam2.load_data(f2,calculate_avg=False)
    print('Calculating BLAST graph')

    A=pd.read_csv(fA,sep='\t',header=None,index_col=0)
    B=pd.read_csv(fB,sep='\t',header=None,index_col=0)

    A.columns=A.columns.astype('<U100')
    B.columns=B.columns.astype('<U100')

    A.index = id1+'_'+ut.extract_annotation(A.index,1,'|').astype('object')
    B.iloc[:,0] = id1 +'_'+ut.extract_annotation(B.iloc[:,0].values.flatten(),1,'|').astype('object')
    sam1.adata.var_names = id1+'_'+sam1.adata.var_names

    B.index = id2+'_'+ut.extract_annotation(B.index,1,'|').astype('object')
    A.iloc[:,0] = id2+'_' + ut.extract_annotation(A.iloc[:,0].values.flatten(),1,'|').astype('object')
    sam2.adata.var_names = id2+'_'+sam2.adata.var_names

    gnnm,gn1,gn2 = map_features_un(A,B,sam1,sam2)
    gn=np.append(gn1,gn2)
    gnnm = filter_gnnm(gnnm,thr=0.25)
    return sam1,sam2,gnnm,gn1,gn2,gn

def get_pairs(sam1,sam2,gnnm,gn1,gn2,NOPs1=2,NOPs2=5):
    gnnm = filter_gnnm(gnnm)
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
        B = sm.sparse_knn(gnnm.multiply(W[None,:]).multiply(W[:,None]).tocsr(),NOPs1).tocsr()
        B = sm.sparse_knn(B.T,NOPs2).T.tocsr()
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


def parallel_init(ipl1x,isc1x,ipairs,ign1O,ign2O,iT2,iCORR,icorr_mode):
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
def refine_corr_parallel(sam1,sam2,st,gnnm,gn1,gn2,corr_mode='pearson', THR=0, use_seq = False,
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

    pool = Pool(os.cpu_count(),parallel_init,[pl1,sc1,p,gn1O,gn2O,T2,CORR,corr_mode])
    try:
        pool.map(parallel_wrapper,range(p.shape[0]),chunksize=pc_chunksize)
    finally:
        pool.close()
        pool.join()

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

def parallel_wrapper(j):
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

def refine_corr_part(sam1,sam2,st,gnnm,gn1,gn2,corr_mode='pearson',THR=0,use_seq=False,
                     T1=0.25,T2=0,NCLUSTERS = 1):
    #import networkx as nx
    import gc
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
        gnnm2_sub,CORR_sub = refine_corr_parallel(sam1,sam2,st,gnnm_sub,gn1_sub,gn2_sub,corr_mode=corr_mode,
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

class SAMap(object):
    def __init__(self,sam1,sam2,gnnm,gn1,gn2):
        self.sam1=sam1
        self.sam2=sam2
        self.gnnm=gnnm
        self.gn1=gn1
        self.gn2=gn2


    def SAMap(self,NUMITERS=2,NOPs1=4,NOPs2=8,NH1=2,NH2=2,K=20,NCLUSTERS=1):
        sam1=self.sam1
        sam2=self.sam2
        gnnm=self.gnnm
        gn1=self.gn1
        gn2=self.gn2
        gn=np.append(gn1,gn2)

        self.max_score = 0
        import gc
        coarsen=False

        gnnm2 = get_pairs(sam1,sam2,gnnm,gn1,gn2,NOPs1=NOPs1,NOPs2=NOPs2)
        sam_def = sm.samap([sam1,sam2],gnnm2,gn, umap=False, NH1=NH1, NH2=NH2,
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
            print('ITERATION: ' + str(i),'Alignment Score: ',avg_as(sam4).mean(),'Max A.S. improvement:',np.max(new-old),'Min A.S. improvement:',np.min(new-old))
            i+=1
            sam_def=sam4
            gc.collect()
            print('Calculating gene-gene correlations in the homology graph...')
            gnnmu = refine_corr_part(sam1,sam2,sam_def,gnnm,gn1,gn2, THR = 0, use_seq=False,corr_mode='pearson',T1=0,T2=0,NCLUSTERS=NCLUSTERS)

            self.GNNMS_corr.append(gnnmu)
            self.gnnmu = gnnmu

            gnnm2  = get_pairs(sam1,sam2,gnnmu,gn1,gn2,NOPs1=NOPs1,NOPs2=NOPs2)
            self.GNNMS_pruned.append(gnnm2)

            gc.collect()

            sam4 = sm.samap([sam1,sam2],gnnm2,gn,umap=False,K=K,NH1=NH1,NH2=NH2,coarsen=coarsen)
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
        self.ITER_DATA = (self.GNNMS_nnm,self.GNNMS_corr,self.GNNMS_pruned,self.SCORE_VEC)
