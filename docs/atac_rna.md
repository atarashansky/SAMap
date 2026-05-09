# scATAC ↔ scRNA alignment with native peak features

SAMap is a feature-graph-guided manifold aligner: given any two
datasets with disjoint feature spaces and a prior feature↔feature
graph, it iteratively projects, kNN-stitches, and reweights the graph.
For scATAC-seq the natural feature space is **peaks**, and the prior
graph is **genomic proximity to gene TSSs** rather than sequence
homology. This page shows how to run SAMap on an unpaired ATAC↔RNA
pair without collapsing peaks to a gene-activity matrix first.

The payoff over the gene-activity reduction is interpretive: the
refined homology graph (`varp['homology_graph_reweighted']`) is then a
**data-driven peak→gene regulatory map** — the same object Signac's
`LinkPeaks` and GLUE's "regulatory scores" produce — obtained as a
side-effect of alignment.

---

## Prerequisites

```bash
pip install pyranges   # GTF parsing + interval overlap
```

You need:

- a cells × peaks AnnData with raw fragment/insertion counts in `.X`
  and peak coordinates either in `var_names` (`"chr1:1000-2000"`) or in
  `var` columns `chrom` / `start` / `end`;
- a cells × genes AnnData (standard scRNA-seq);
- a GTF for the same genome build the peaks were called on, with
  `gene_id` matching the RNA `var_names`.

---

## Same-species, unpaired ATAC ↔ RNA

```python
import samap.io as sio
from samap import SAMAP
from samap.sam import SAM

# --- 1. ATAC side: TF-IDF + LSI (not log-CPM + SAM dispersion) ---------
# prepare_atac_sam returns a SAM object with the slots SAMap reads
# (var['weights'] = scaled IDF, varm['PCs_SAMap'] = LSI loadings,
# obsp['connectivities'] = kNN on the LSI embedding) already populated,
# so SAMAP.__init__ does not re-run SAM's RNA preprocessing on it.
import anndata as ad
atac = ad.read_h5ad("pbmc_atac.h5ad")          # cells × peaks, raw counts
atac_sam = sio.prepare_atac_sam(atac, n_components=50, drop_first=True)

# Optional: real Leiden clusters on the LSI embedding (otherwise a
# placeholder single cluster is used).
atac_sam.leiden_clustering(res=1.0)

# --- 2. RNA side: standard SAM ----------------------------------------
rna_sam = SAM()
rna_sam.load_data("pbmc_rna.h5ad")
rna_sam.preprocess_data()
rna_sam.run()

# --- 3. Peak↔gene prior from the GTF ----------------------------------
gnnm = sio.gnnm_from_gtf(
    atac_sam.adata, rna_sam.adata,
    gtf="gencode.v44.annotation.gtf.gz",
    ids=("at", "rn"),
    kind="powerlaw",        # or "tss_window" for the Signac/ArchR rule
    window=150_000,
    gamma=0.87,
)

# --- 4. Align ----------------------------------------------------------
sm = SAMAP({"at": atac_sam, "rn": rna_sam}, gnnm=gnnm)
sm.run(
    hom_edge_mode="xi",                 # rank-based; robust to the
                                        # saturating accessibility→expr
                                        # relationship and to ties
    neighborhood_sizes={"at": 5, "rn": 3},
)

# --- 5. Outputs --------------------------------------------------------
# Stitched cell graph + UMAP:
sm.samap.adata.obsp["connectivities"]
sm.samap.adata.obsm["X_umap"]

# Data-driven peak→gene links (the refined feature graph):
links = sm.samap.adata.varp["homology_graph_reweighted"]
```

### Choosing `kind`

| `kind` | Edge rule | Typical nnz | Use when |
|---|---|---|---|
| `tss_window` | peak overlaps gene-body ∪ [TSS−2 kb, TSS] | ~2–5× #genes | conservative; Signac default |
| `powerlaw` | peak within ±150 kb of TSS, weight ∝ d⁻⁰·⁸⁷ | ~20–40× #genes | distal enhancers matter; GLUE default |
| `coaccess` | from a Cicero / `LinkPeaks` table | as supplied | you already ran Cicero |

For `kind='coaccess'`, pass the link table directly:

```python
gnnm = sio.gnnm_from_gtf(
    atac_sam.adata, rna_sam.adata, gtf=None,
    ids=("at", "rn"), kind="coaccess",
    coaccess=cicero_df[["peak", "gene", "score"]],
)
```

### Why `hom_edge_mode='xi'`

After kNN smoothing the ATAC side is no longer binary — each cell's
peak value is the fraction of its ~20 stitched neighbours with the peak
open — so Pearson is not degenerate. Xi (Chatterjee 2021) is still the
better choice: it is rank-based, handles the heavy ties on the ATAC
side, and detects monotone-non-linear dependence (accessibility →
expression saturates). `hom_edge_mode='xi'` is already plumbed through
`SAMAP.run()`; this page simply recommends it as the default for
binary-feature modalities.

---

## Cross-species ATAC ↔ RNA

Compose the peak→gene graph (built on the ATAC species' annotation)
with a gene-ortholog graph:

```python
# Zebrafish ATAC ↔ mouse RNA.
gnnm_pg = sio.gnnm_from_gtf(
    zf_atac_sam.adata, zf_rna_ref,      # zf_rna_ref: any AnnData whose
    gtf="Danio_rerio.GRCz11.gtf.gz",    # var_names are zebrafish gene IDs
    ids=("at", "zf"), kind="powerlaw",
)

# Ortholog graph zf ↔ mm — from BLAST, eggNOG, or BioMart pairs.
gnnm_orth = sio.gnnm_from_pairs(
    biomart_df[["zf_gene", "mm_gene"]].values,
    ids={"zf": zf_gene_ids, "mm": mm_rna_sam.adata.var_names},
)

# Chain through the shared 'zf' block: at_peak → zf_gene → mm_gene.
gnnm = sio.compose_gnnm(gnnm_pg, gnnm_orth, via="zf")

sm = SAMAP({"at": zf_atac_sam, "mm": mm_rna_sam}, gnnm=gnnm)
sm.run(hom_edge_mode="xi")
```

`compose_gnnm` is one sparse matmul of the two off-diagonal blocks;
weights are the product (sum over intermediate genes), clipped to
``[0, 1]``.

---

## What `prepare_atac_sam` does

| SAM slot | RNA path (`SAM.preprocess_data` + `SAM.run`) | ATAC path (`prepare_atac_sam`) |
|---|---|---|
| `adata.X` | log-CPM | TF-IDF (TF = per-cell L1, IDF = `log(1 + N/nᵢ)`) |
| `var['weights']` | SAM dispersion / RMS weights | scaled IDF (down-weights ubiquitous promoter peaks) |
| `varm['PCs_SAMap']` | sparse PCA loadings, 300 comps | LSI right singular vectors, SV1 dropped |
| `obsp['connectivities']` | HNSW kNN on SAM PCs | HNSW kNN on LSI embedding |
| `uns['modality']` | — | `'atac'` |

If you have already run TF-IDF/LSI elsewhere (Signac `RunTFIDF` +
`RunSVD`, muon `ac.pp.tfidf`/`ac.tl.lsi`), you can populate those slots
yourself and skip `prepare_atac_sam` — `SAMAP` only reads the slots,
not how they were produced.

---

## Caveats

- **Chromosome naming.** The peak `var_names` and the GTF must agree on
  `chr1` vs `1`. `gnnm_from_gtf` raises with examples if zero edges are
  produced.
- **Gene-ID namespace.** GTF `gene_id` must match `rna.var_names`
  (Ensembl ID vs symbol). Re-index one side, or pass a `gene_id →
  symbol` table through `gnnm_from_pairs` afterwards.
- **The overlap warning** at `SAMAP.__init__` ("only X% of var_names
  matched the homology graph") is expected to fire on the ATAC side —
  most peaks are intergenic under `kind='tss_window'`. It is benign
  here as long as the *gene* side matches well.

---

## Related

- [docs/multi_reference.md](multi_reference.md) — the within-species
  analogue (geneIDs ↔ geneIDs).
- `samap.io.gnnm_from_pairs` — the primitive both `gnnm_from_gtf` and
  `compose_gnnm` reduce to.
