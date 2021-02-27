import sklearn.utils.sparsefuncs as sf
from . import q, ut, pd, sp, np, warnings, sc
from .utils import to_vo, to_vn, substr, df_to_dict, sparse_knn, prepend_var_prefix


def sankey_plot(sm,key1,key2,align_thr=0.1):
    """Generate a sankey plot
    
    Parameters
    ----------
    sm: SAMAP object
    
    key1 & key2: str, annotation vector keys for species 1 and 2

    align_thr: float, optional, default 0.1
        The alignment score threshold below which to remove cell type mappings.
    """    
    _,_,M = get_mapping_scores(sm,key1,key2)
    
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
                 k2="leiden_clusters",compute_markers=True):
        """Find enriched gene pairs in cell type mappings.
        
        sm: SAMAP object

        k1 & k2: str, optional, default 'leiden_clusers'
            Keys corresponding to the annotation vector in `s1.adata.obs` and `s2.adata.obs`.

        compute_markers: bool, optional, default True
            If True, compute differentially expressed genes using `find_cluster_markers`. 
            If False, assumes differentially expressed genes were already computed and stored
            in `.adata.var`.
        
        """
        self.sm = sm
        self.s1 = sm.sam1
        self.s2 = sm.sam2
        self.s3 = sm.samap

        self.id1 = self.s3.adata.var_names[0].split(";")[0].split("_")[0]
        self.id2 = self.s3.adata.var_names[0].split(";")[1].split("_")[0]

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
        if compute_markers:
            self.find_markers()

    def find_markers(self):
        print(
            "Finding cluster-specific markers in {}:{} and {}:{}.".format(
                self.id1, self.k1, self.id2, self.k2
            )
        )        
        import gc
        
        find_cluster_markers(self.s1, self.k1)
        find_cluster_markers(self.s2, self.k2)
        gc.collect()
        self.s1.dispersion_ranking_NN(save_avgs=True)        
        self.s1.identify_marker_genes_sw(labels=self.k1)
        del self.s1.adata.layers['X_knn_avg']        
        gc.collect()        
        self.s2.dispersion_ranking_NN(save_avgs=True)        
        self.s2.identify_marker_genes_sw(labels=self.k2)
        del self.s2.adata.layers['X_knn_avg']
        gc.collect() # the collects are trying to deal with a weird memory leak issue
        
    def find_all(self,thr=0.1,**kwargs):
        """Find enriched gene pairs in all pairs of mapped cell types.
        
        Parameters
        ----------
        thr: float, optional, default 0.2
            Alignment score threshold above which to consider cell type pairs mapped.
        
        Keyword arguments
        -----------------
        Keyword arguments to `find_genes` accepted here.
        
        Returns
        -------
        Table of enriched gene pairs for each cell type pair
        """        
        _,_,M = get_mapping_scores(self.sm, self.k1, self.k2)
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

        if True:
            m = self.find_link_genes_avg(n1, n2, w1t=w1t, w2t=w2t, expr_thr=0.05)
        else:
            m = self.find_link_genes(
                n1, n2, w1t=w1t, w2t=w2t, n_pairs=500, expr_thr=0.05
            )

        self.gene_pair_scores = pd.Series(index=self.s3.adata.var_names, data=m)

        G = q(self.s3.adata.var_names[np.argsort(-m)[:n_genes]])
        G1 = substr(G, ";", 0)
        G2 = substr(G, ";", 1)
        G = q(
            G[
                np.logical_and(
                    q(self.s1.adata.var[self.k1 + ";;" + n1 + "_pval"][G1] < thr),
                    q(self.s2.adata.var[self.k2 + ";;" + n2 + "_pval"][G2] < thr),
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

    def find_link_genes_avg(self, c1, c2, w1t=0.35, w2t=0.35, expr_thr=0.05):
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
            ut.extract_annotation(sam3.adata.var_names, 0, ";"),
            ut.extract_annotation(sam3.adata.var_names, 1, ";"),
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

    def find_link_genes(
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
            ut.extract_annotation(sam3.adata.var_names, 0, ";"),
            ut.extract_annotation(sam3.adata.var_names, 1, ";"),
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


def find_cluster_markers(sam, key, layer=None, inplace=True):
    sam.adata.raw = sam.adata_raw
    a,c = np.unique(q(sam.adata.obs[key]),return_counts=True)
    t = a[c==1]
    for i in range(t.size):
        sam.adata.var[key+';;'+t[i]]=0
        sam.adata.var[key+';;'+t[i]+'_pval']=1
        
    adata = sam.adata[np.in1d(q(sam.adata.obs[key]),a[c==1],invert=True)].copy()
    sc.tl.rank_genes_groups(
        adata,
        key,
        method="wilcoxon",
        n_genes=sam.adata.shape[1],
        use_raw=True,
        layer=layer,
    )
    sam.adata.uns['rank_genes_groups'] = adata.uns['rank_genes_groups']
    
    NAMES = pd.DataFrame(sam.adata.uns["rank_genes_groups"]["names"])
    PVALS = pd.DataFrame(sam.adata.uns["rank_genes_groups"]["pvals"])
    SCORES = pd.DataFrame(sam.adata.uns["rank_genes_groups"]["scores"])
    if not inplace:
        return NAMES, PVALS, SCORES
    for i in range(SCORES.shape[1]):
        names = NAMES.iloc[:, i]
        scores = SCORES.iloc[:, i]
        pvals = PVALS.iloc[:, i]
        pvals[scores < 0] = 1.0
        scores[scores < 0] = 0
        pvals = q(pvals)
        scores = q(scores)
        sam.adata.var[key + ";;" + SCORES.columns[i]] = pd.DataFrame(
            data=scores[None, :], columns=names
        )[sam.adata.var_names].values.flatten()
        sam.adata.var[key + ";;" + SCORES.columns[i] + "_pval"] = pd.DataFrame(
            data=pvals[None, :], columns=names
        )[sam.adata.var_names].values.flatten()
    


def ParalogSubstitutions(sm, ortholog_pairs, paralog_pairs=None, psub_thr = 0.3):
    """Identify paralog substitutions
    
    Parameters
    ----------
    sm - SAMAP object
    
    ortholog_pairs - n x 2 numpy array of ortholog pairs
    
    paralog_pairs - n x 2 numpy array of paralog pairs, optional, default None
        If None, assumes every pair in the homology graph that is not an ortholog is a paralog.
        Note that this would essentially result in the more generic 'homolog substitutions' rather
        than paralog substitutions.
        
    Returns
    -------
    RES - pandas.DataFrame
        A table of paralog substitutions.
        
    """
    smp = sm.samap
    
    gnnm = smp.adata.uns["homology_graph_reweighted"]
    gn = sm.gn
    
    if paralog_pairs is None:
        paralog_pairs = gn[np.vstack(smp.adata.uns["homology_graph"].nonzero()).T]
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


def convert_eggnog_to_homologs(sm, A, B, taxon=2759):
    """Gets an n x 2 array of homologs at some taxonomic level based on Eggnog results.
    
    Parameters
    ----------
    smp: SAMAP object
    
    A: pandas.DataFrame, Eggnog output table
    
    B: pandas.DataFrame, Eggnog output table
    
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

    orthology_groups = A["18"]
    og = q(orthology_groups)
    x = np.unique(",".join(og).split(","))
    D = pd.DataFrame(data=np.arange(x.size)[None, :], columns=x)

    for i in range(og.size):
        n = orthology_groups[i].split(",")
        taxa = substr(n, "@", 1)
        if (taxa == "2759").sum() > 1:
            og[i] = ""
        else:
            og[i] = "".join(np.array(n)[taxa == taxon])

    A["18"] = og

    og = q(A["18"][gn])
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


def CellTypeTriangles(sms,keys, align_thr=1):
    """Outputs a table of cell type triangles.
    
    Parameters
    ----------
    sms: list or tuple of three SAMAP objects for three different species mappings
       
    keys: list or tuple of three strings corresponding to each species annotation column, optional, default None
        If you'd like to include information about where each gene is differentially expressed, you can specify the
        annotation column to compute differential expressivity from for each species.
        Let `sms[0]` be the mapping for species A to B. Each element of `keys` corresponds to an
        annotation column in species `[A, B, C]`, respectively.

    align_thr: float, optional, default, 0.1
        Only keep triangles with minimum `align_thr` alignment score.        
    """
    
    sm1,sm2,sm3 = sms
    key1,key2,key3 = keys
    
    smp1 = sm1.samap
    smp2 = smp2.samap
    smp3 = smp3.samap
    
    s = q(smp1.adata.obs["species"])
    A, B = s[np.sort(np.unique(s, return_index=True)[1])][:2]

    s = q(smp2.adata.obs["species"])
    B1, B2 = s[np.sort(np.unique(s, return_index=True)[1])][:2]
    C = B1 if B1 not in [A, B] else B2

    A1, A2 = A, B
    s = q(smp1.adata.obs["species"])
    C1, C2 = s[np.sort(np.unique(s, return_index=True)[1])][:2]

    codes = dict(zip([A, B, C], [key1, key2, key3]))
    X = []
    W = []
    for i in [[A1, A2, smp1], [B1, B2, smp2], [C1, C2, smp3]]:
        x, y, smp = i
        k1, k2 = codes[x], codes[y]

        cl1 = q(smp.adata.obs[k1])
        if k1 != k2:
            cl2 = q(smp.adata.obs[k2])
            cl1[cl1 == ""] = cl2[cl2 != ""]
        smp.adata.obs["triangle_{}{}".format(x, y)] = pd.Categorical(cl1)

        _, ax, bx, CSIMt = compute_csim(smp, key="triangle_{}{}".format(x, y))
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
                a = sam.adata.var[keys[i]+';;'+sam.adata.obs[keys[i]].cat.categories.astype('object')].T[q(FINAL[n+' gene'])].T
                p = sam.adata.var[keys[i]+';;'+sam.adata.obs[keys[i]].cat.categories.astype('object')+'_pval'].T[q(FINAL[n+' gene'])].T.values
                p[p>pval_thr]=1
                p[p<1]=0
                p=1-p
                f = substr(a.columns[a.values.argmax(1)],';;',1)
                res=[]
                for i in range(p.shape[0]):
                    res.append(';'.join(np.unique(np.append(f[i],substr(a.columns[p[i,:]==1],';;',1)))))            
                FINAL[n+' cell type'] = res
    FINAL = FINAL.sort_values('min_corr',ascending=False)
    return FINAL


def compute_csim(sam3, key, X=None, n_top = 100):
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

def get_mapping_scores(sm, key1, key2):
    """Calculate mapping scores
    Parameters
    ----------
    sm: SAMAP object
    
    key1 & key2: str, annotation vector keys for species 1 and 2

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
    _, clu1, clu2, CSIMth = compute_csim(samap, "{};{}_mapping_scores".format(key1,key2))

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
    g1, g2 = ut.extract_annotation(sam3.adata.var_names, 0, ";"), ut.extract_annotation(
        sam3.adata.var_names, 1, ";"
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
