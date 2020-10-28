from SAMap_iteration import *
import scanpy as sc
import time
import pickle

if __name__ == '__main__':
    t = time.time()

    print('Running SAM on the zebrafish data.')
    d = pickle.load(open('data/ze_raw_data.p','rb'))
    g = pickle.load(open('data/ze_vn.p','rb'))
    c = pickle.load(open('data/ze_on.p','rb'))
    o = pickle.load(open('data/ze_obs.p','rb'))
    sam1=SAM(counts = (d,g,c))
    sam1.load_obs_annotations(o)
    sam1.preprocess_data(sum_norm='cell_median',norm='log',thresh_low=0.0,thresh_high=0.96,min_expression=1)
    sam1.run(preprocessing='StandardScaler',npcs=150,weight_PCs=False,k=20,n_genes=3000)
    sam1.leiden_clustering(res=3)
    print("Calculating PC loadings for all genes.")
    sam1.adata.varm['PCs_SAMap'] = sam1.calculate_nnm(sam1.adata.shape[1],'StandardScaler',300,50,False,True,update_manifold=False)
    sam1.save_anndata('data/ze_sub_run.h5ad')

    print('Running SAM on the xenopus data.')
    d = pickle.load(open('data/xe_raw_data.p','rb'))
    g = pickle.load(open('data/xe_vn.p','rb'))
    c = pickle.load(open('data/xe_on.p','rb'))
    o = pickle.load(open('data/xe_obs.p','rb'))
    sam2=SAM(counts = (d,g,c))
    sam2.load_obs_annotations(o)
    sam2.preprocess_data(sum_norm='cell_median',norm='log',thresh_low=0.0,thresh_high=0.96,min_expression=1)
    sam2.run(preprocessing='StandardScaler',npcs=150,weight_PCs=False,k=20,n_genes=3000)
    sam2.leiden_clustering(res=3)
    print("Calculating PC loadings for all genes.")
    sam2.adata.varm['PCs_SAMap'] = sam2.calculate_nnm(sam2.adata.shape[1],'StandardScaler',300,50,False,True,update_manifold=False)
    sam2.save_anndata('data/xe_sub_run.h5ad')

    print('Initializing SAMap objects.')
    sam1,sam2,gnnm,gn1,gn2,gn = get_SAMap_init('data/ze_sub_run.h5ad','transcriptomes/maps/zexe/ze_to_xe.txt',
                                               'data/xe_sub_run.h5ad','transcriptomes/maps/zexe/xe_to_ze.txt',
                                               id1 = 'Z', id2 = 'X')

    smap = SAMap(sam1,sam2,gnnm,gn1,gn2)
    ITER_DATA = smap.SAMap()
    samap=smap.final_sam
    print('Alignment score ---',avg_as(samap).mean())
    k1 = 'Cluster_ZF'
    k2 = 'Cluster_XF'
    samap.adata.obs['celltypes'] = pd.Categorical(np.append(sam1.get_labels(k1).astype('object').astype('<U100').astype('object'),sam2.get_labels(k2).astype('object').astype('<U100').astype('object')))

    print('Running UMAP on the stitched manifolds.')
    sc.tl.umap(samap.adata,min_dist=0.1,init_pos='random')
    ax = samap.scatter(c='batch',s=15,cmap='Spectral_r',colorbar=False)
    ax.figure.set_size_inches((7,7))
    ax.set_xticks([])
    ax.set_yticks([])


    print('Saving results...')
    ut.create_folder('output/zexe/')
    samap.save_anndata('output/zexe/st_zexe.h5ad')
    ax.figure.savefig('output/zexe/umap.png',dpi=1000)

    #pickle.dump(ITER_DATA,open('output/zexe/st_zexe_iter.p','wb'))

    x = smap.GNNMS_corr[-1]
    pairs = gn[np.vstack(x.nonzero()).T]
    corr = x.data
    pairs = np.sort(pairs,axis=1)
    _,ix = np.unique(pairs,axis=0,return_index=True)
    ix = np.sort(ix)
    pairs = pairs[ix]
    corr = corr[ix]
    A = pd.DataFrame(data = pairs,columns=['Zebrafish','Xenopus'])
    A['Expression correlation'] = corr
    A = A.iloc[np.argsort(-corr)]
    A.to_csv('output/zexe/gene_pairs_correlation.tsv',sep='\t')

    _, clu1, clu2, CSIMth = compute_csim(samap,'celltypes')
    CSIMth/=20
    CSIMth[CSIMth<0.1]=0
    pd.DataFrame(data = CSIMth, index = clu1, columns = clu2).to_csv('output/zexe/celltype_mapping_scores.tsv',sep='\t')

    print('SAMap ran in',np.round((time.time() - t)/60,1),'minutes.')
