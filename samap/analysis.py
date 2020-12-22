import sklearn.utils.sparsefuncs as sf
from . import q, ut, pd, sp, np, warnings
from .utils import to_vo, to_vn, substr, df_to_dict, sparse_knn, prepend_var_prefix


def get_mu_std(sam3, sam1, sam2, knn=False):
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


class GenePairFinder(object):
    def __init__(self, s1, s2, s3, k1="leiden_clusters", k2="leiden_clusters"):
        self.id1 = s3.adata.var_names[0].split(";")[0].split("_")[0]
        self.id2 = s3.adata.var_names[0].split(";")[1].split("_")[0]

        prepend_var_prefix(s1, self.id1)
        prepend_var_prefix(s2, self.id2)

        s1.adata.obs[k1] = s1.adata.obs[k1].astype("str")
        s2.adata.obs[k2] = s2.adata.obs[k2].astype("str")

        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.s1.dispersion_ranking_NN(save_avgs=True)
        self.s2.dispersion_ranking_NN(save_avgs=True)
        mu1, v1, mu2, v2 = get_mu_std(self.s3, self.s1, self.s2)
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
        find_cluster_markers(self.s1, self.k1)
        find_cluster_markers(self.s2, self.k2)
        self.s1.identify_marker_genes_sw(labels=self.k1)
        self.s2.identify_marker_genes_sw(labels=self.k2)

    def find_genes(
        self,
        n1,
        n2,
        w1t=0.2,
        w2t=0.2,
        n_pairs=500,
        n_genes=1000,
        thr=1e-2,
        avgmode=True,
    ):
        assert n1 in q(self.s1.adata.obs[self.k1])
        assert n2 in q(self.s2.adata.obs[self.k2])

        if avgmode:
            m = self.find_link_genes_avg(n1, n2, w1t=w1t, w2t=w2t, expr_thr=0.05)
        else:
            m = self.find_link_genes(
                n1, n2, w1t=w1t, w2t=w2t, n_pairs=n_pairs, expr_thr=0.05
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
    sc.tl.rank_genes_groups(
        sam.adata_raw,
        key,
        method="wilcoxon",
        n_genes=sam.adata.shape[1],
        use_raw=True,
        layer=layer,
    )

    NAMES = pd.DataFrame(sam.adata_raw.uns["rank_genes_groups"]["names"])
    PVALS = pd.DataFrame(sam.adata_raw.uns["rank_genes_groups"]["pvals"])
    SCORES = pd.DataFrame(sam.adata_raw.uns["rank_genes_groups"]["scores"])
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


def ParalogSubstitutions(smp, ortholog_pairs, paralog_pairs=None):

    gnnm = smp.adata.uns["homology_graph_reweighted"]
    gn = smp.adata.uns["homology_gene_names"]
    if paralog_pairs is None:
        paralog_pairs = gn[np.vstack(smp.adata.uns["homology_graph"].nonzero()).T]
    paralog_pairs = paralog_pairs[
        np.in1d(to_vn(paralog_pairs), to_vn(ortholog_pairs), invert=True)
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
    RES = RES[RES["corr diff"] > 0]
    return RES


def convert_eggnog_to_homologs(smp, A, B, taxon=2759):
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


def CellTypeTriangles(smp1, smp2, smp3, key1, key2, key3, tr=1):
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
    all_pairs = all_pairs[alignment > tr]
    alignment = alignment[alignment > tr]
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
    DF = pd.DataFrame(data=Z, columns=[x[i].split("_")[0] for x in Z[0]])
    DF = DF[[A, B, C]]
    return DF


def GeneTriangles(smp1, smp2, smp3, orth1, orth2, orth3):
    s = q(smp1.adata.obs["species"])
    A, B = s[np.sort(np.unique(s, return_index=True)[1])][:2]

    s = q(smp2.adata.obs["species"])
    B1, B2 = s[np.sort(np.unique(s, return_index=True)[1])][:2]
    C = B1 if B1 not in [A, B] else B2

    A1, A2 = A, B
    s = q(smp1.adata.obs["species"])
    C1, C2 = s[np.sort(np.unique(s, return_index=True)[1])][:2]

    RES = []
    for i in [[smp1, orth1], [smp2, orth2], [smp3, orth3]]:
        smp, orth = i
        RES.append(ParalogSubstitutions(smp, orth))
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
    Z = np.sort(np.vstack(all_triangles), axis=1)
    DF = pd.DataFrame(data=Z, columns=[x[i].split("_")[0] for x in Z[0]])
    DF = DF[[A, B, C]]

    orth1DF = pd.DataFrame(data=orth1, columns=[x[i].split("_")[0] for x in orth1[0]])[
        [A, B]
    ]
    orth2DF = pd.DataFrame(data=orth2, columns=[x[i].split("_")[0] for x in orth2[0]])[
        [A, C]
    ]
    orth3DF = pd.DataFrame(data=orth3, columns=[x[i].split("_")[0] for x in orth3[0]])[
        [B, C]
    ]

    ps1DF = pd.DataFrame(
        data=np.sort(ps1, axis=1),
        columns=[x[i].split("_")[0] for x in np.sort(ps1, axis=1)[0]],
    )[[A, B]]
    ps2DF = pd.DataFrame(
        data=np.sort(ps2, axis=1),
        columns=[x[i].split("_")[0] for x in np.sort(ps2, axis=1)[0]],
    )[[A, C]]
    ps3DF = pd.DataFrame(
        data=np.sort(ps3, axis=1),
        columns=[x[i].split("_")[0] for x in np.sort(ps3, axis=1)[0]],
    )[[B, C]]

    A_AB = pd.DataFrame(data=op1[None, :], columns=to_vn(ps1DF.values))
    A_AC = pd.DataFrame(data=op2[None, :], columns=to_vn(ps2DF.values))
    A_BC = pd.DataFrame(data=op3[None, :], columns=to_vn(ps3DF.values))

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
        z = Z[X[ff]]
        x = X[ff]
        b = z.values.flatten()
        av = np.zeros(x.size, dtype="object")
        for ai in range(x.size):
            av[ai] = ";".join(np.unique(substr(b[Z.columns == x[ai]], ";", 1)))
        AV = np.zeros(X.size, dtype="object")
        AV[ff] = av
        corr = CORR1[X].values.flatten()

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
    return data


def compute_csim(sam3, key, X=None):
    cl1 = q(sam3.adata.obs[key].values[sam3.adata.obs["batch"] == "batch1"])
    clu1 = np.unique(cl1)
    cl2 = q(sam3.adata.obs[key].values[sam3.adata.obs["batch"] == "batch2"])
    clu2 = np.unique(cl2)

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
                X[cl == c1, :][:, cl == c2].sum(1).A.flatten(),
                X[cl == c2, :][:, cl == c1].sum(1).A.flatten(),
            ).mean()
    CSIMth = CSIM1
    s1 = CSIMth.sum(1).flatten()[:, None]
    s2 = CSIMth.sum(0).flatten()[None, :]
    s1[s1 == 0] = 1
    s2[s2 == 0] = 1
    CSIM1 = CSIMth / s1
    CSIM2 = CSIMth / s2
    CSIM = (CSIM1 * CSIM2) ** 0.5

    return CSIM, clu1, clu2, CSIMth


def get_mapping_scores(sam1, sam2, samap, key1, key2):

    cl1 = q(sam1.adata.obs[key1])
    cl2 = q(sam2.adata.obs[key2])
    cl = (
        q(samap.adata.obs["species"]).astype("object")
        + "_"
        + np.append(cl1, cl2).astype("str").astype("object")
    )

    samap.adata.obs["mapping_score_labels"] = pd.Categorical(cl)
    _, clu1, clu2, CSIMth = compute_csim(samap, "mapping_score_labels")

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
    D1 = pd.DataFrame(data=H, columns=[C, np.arange(H.shape[1])])

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
    D2 = pd.DataFrame(data=H, columns=[C, np.arange(H.shape[1])])
    return D1, D2


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
