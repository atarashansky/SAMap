# Within-species multi-reference integration

SAMap was written for cross-species alignment, but nothing in the
algorithm assumes the input datasets come from different organisms. The
only structural requirement is that each dataset has its **own feature
namespace** and that you can supply a many-to-many feature↔feature graph
linking them. Two scRNA-seq datasets quantified against different genome
builds, different annotation sources, or a *de novo* assembly vs. a
reference satisfy this exactly — and the existing API handles them with
**no code change**.

This page works through three concrete cases.

---

## Why not just lift over gene IDs?

The usual approach — `biomaRt` / UCSC liftOver to a 1:1 gene-ID table,
rename `var_names`, then run a standard batch corrector (Harmony / scVI)
— fails precisely where annotation versions disagree most:

- **Split / merged gene models.** A GRCh37 gene that becomes two GRCh38
  genes has no 1:1 image; ID liftover drops it. SAMap keeps a 1→2 edge
  and the correlation reweighting decides which v38 model carries the
  signal — the same machinery that resolves paralog substitutions.
- **De novo assemblies.** Trinity contigs from a non-model organism have
  no stable IDs at all. The *only* feature correspondence is sequence
  alignment, which is already SAMap's onboarding path.
- **RefSeq ↔ Ensembl.** Isoform-level disagreements between annotation
  sources are the dominant cause of `var_names` mismatch.
  `_coarsen_blast_graph` already handles transcript→gene collapse.

A note on terminology: the codebase says "species" everywhere
(`gns_dict` keys, `obs['species']`, log messages). For this use case,
read "species ID" as "feature-space ID" — a short tag for one
annotation version.

---

## Case 1 — GRCh37 ↔ GRCh38 via BLAST

Two human atlases quantified against different builds. `run_blast` on
the two protein FASTAs produces the homology graph; `SAMAP` does the
rest.

```python
import samap.io as sio
from samap import SAMAP

# 1. Reciprocal protein-protein alignment of the two annotation sets.
#    DIAMOND is the default engine for protein DBs (fast, no makeblastdb).
sio.run_blast(
    {
        "h37": "gencode.v19.pc_translations.fa.gz",   # GRCh37 / GENCODE 19
        "h38": "gencode.v44.pc_translations.fa.gz",   # GRCh38 / GENCODE 44
    },
    out="maps/",
)

# 2. Map. The two h5ads keep their native var_names (ENSG.v19 / ENSG.v44
#    or gene symbols — whatever each was quantified with).
sm = SAMAP(
    {"h37": "atlas_grch37.h5ad", "h38": "atlas_grch38.h5ad"},
    f_maps="maps/",
)
sm.run()

# 3. Inspect which gene-model pairs the iteration kept.
sm.homology_pair("h37_ENSG00000142192")   # APP, for example
```

If the FASTA headers and `adata.var_names` are in different namespaces
(transcript IDs vs gene IDs, version suffixes), you'll see the
low-overlap warning at `SAMAP.__init__`. Use `samap.io.match_fasta` or
the `names=` argument to reconcile them — the same workflow as
cross-species onboarding (see [docs/io.md](io.md)).

---

## Case 2 — RefSeq ↔ Ensembl via an ID-mapping table

If you already have a gene-ID correspondence table (BioMart export,
NCBI `gene2ensembl`, or your own liftover), skip BLAST entirely and
build the graph from pairs:

```python
import pandas as pd
import samap.io as sio
from samap import SAMAP
from samap.sam import SAM

# Any (N, ≥2) table of cross-namespace gene pairs. Here: NCBI gene2ensembl
# filtered to human, columns RefSeq gene symbol → Ensembl gene ID.
tbl = pd.read_csv("refseq_to_ensembl.tsv", sep="\t")

sam_rs = SAM(); sam_rs.load_data("atlas_refseq.h5ad")
sam_rs.preprocess_data(); sam_rs.run()

sam_en = SAM(); sam_en.load_data("atlas_ensembl.h5ad")
sam_en.preprocess_data(); sam_en.run()

gnnm = sio.gnnm_from_pairs(
    tbl[["refseq_symbol", "ensembl_gene_id"]].values,
    ids={"rs": sam_rs.adata.var_names, "en": sam_en.adata.var_names},
)

sm = SAMAP({"rs": sam_rs, "en": sam_en}, gnnm=gnnm)
sm.run()
```

`gnnm_from_pairs` takes care of the species-prefix bookkeeping and
symmetrisation; it accepts an optional `weights=` column (e.g.
`%identity` from a liftover table) if you want a non-uniform prior.

---

## Case 3 — De novo Trinity assembly ↔ reference annotation

A common situation in non-model systems: dataset A was quantified
against a Trinity transcriptome (`TRINITY_DN123_c0_g1`), dataset B
against the species' reference proteome. There is no ID table —
sequence alignment is the only bridge.

```python
import samap.io as sio
from samap import SAMAP

# Trinity outputs nucleotide contigs; the reference is protein.
# run_blast auto-selects tblastn / blastx (or MMseqs2 for nucl DB).
sio.run_blast(
    {
        "tr": ("Trinity.fasta", "nucl"),
        "rf": ("reference_proteome.fa", "prot"),
    },
    out="maps/",
)

sm = SAMAP(
    {"tr": "counts_trinity.h5ad", "rf": "counts_reference.h5ad"},
    f_maps="maps/",
    # Trinity contigs are transcript-level; collapse to Trinity 'gene'
    # (the _gN component) so multiple isoforms of one locus share an edge.
    names={"tr": trinity_tx2gene_array},  # shape (N, 2): [tx_id, gene_id]
)
sm.run()
```

---

## More than two references

`SAMAP` already accepts >2 datasets. To integrate three annotation
versions at once, run all pairwise BLASTs (`run_blast` does this
automatically when given a 3-key dict) and pass all three SAM objects:

```python
sio.run_blast({"h37": fa37, "h38": fa38, "chm": fa_chm13}, out="maps/")
sm = SAMAP({"h37": s37, "h38": s38, "chm": s_chm13}, f_maps="maps/")
sm.run(pairwise=True)
```

The refined homology graph
(`sm.samap.adata.varp['homology_graph_reweighted']`) then contains all
three pairwise blocks, and `samap.analysis` functions
(`get_mapping_scores`, `GenePairFinder`, `homology_graph_delta`) work
across all of them.

---

## Related

- [docs/io.md](io.md) — `run_blast` engines, FASTA-header / var_names
  matching, `gnnm_from_pairs` reference.
- [docs/atac_rna.md](atac_rna.md) — the cross-modality analogue of this
  page (peaks ↔ genes instead of geneIDs ↔ geneIDs).
- GitHub issues #121, #143, #157 — original user requests for this
  workflow.
