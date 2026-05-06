# Preparing inputs (`samap.io`)

SAMap needs two things per species: an `AnnData` and a way to relate its
genes to other species' genes. The most common onboarding failure is a
mismatch between `adata.var_names` and the FASTA headers used for BLAST —
transcript IDs vs gene IDs, version suffixes, namespace differences. The
`samap.io` module exists to make that join obvious and, where possible,
automatic.

```text
                ┌──────────────────────────┐
 var_names ───▶ │  detect_id_flavor        │
                └──────────────────────────┘
                       │
       Ensembl / NCBI ─┼──▶ fetch_proteome ───▶ FASTA (headers == var_names)
                       │
       de-novo / other ┴──▶ match_fasta(FASTA, gtf?) ──▶ renamed FASTA + names[]
                                                              │
                                                              ▼
                                              run_blast / DIAMOND  ──▶  maps/
                                                              │
                                          (or)                ▼
   eggNOG-mapper TSVs ──▶ homology_from_eggnog ───────▶  gnnm tuple
   BioMart orthologs  ──▶ gnnm_from_pairs    ──────────▶  gnnm tuple
                                                              │
                                                              ▼
                                                   SAMAP(sams, gnnm=...)
```

## 1. Find out what your IDs are

```python
from samap.io import detect_id_flavor
import anndata as ad

a = ad.read_h5ad("species.h5ad")
print(detect_id_flavor(a.var_names))
# IdFlavorReport(flavor='ensembl_gene', confidence=1.00, n=200, ...)
```

Recognized: `ensembl_{gene,tx,protein}`, `refseq_{rna,protein}`,
`ncbi_geneid`, `uniprot`, `flybase`/`wormbase`/`zfin`/`mgi`/`hgnc`/`wbps`,
`symbol` (heuristic), `unknown`, `mixed`.

## 2a. Stable IDs → derive a FASTA

If `flavor` is `ensembl_*` or `ncbi_geneid`, don't curate a FASTA —
fetch one whose headers *are* your var_names:

```python
from samap.io import fetch_proteome

rep = fetch_proteome(a, "human.fa")          # one record per gene, longest isoform
# headers == var_names by construction (incl. version suffixes)
```

Transcript-level input also returns a `names` array (transcript → gene)
for `SAMAP(names={...})`.

## 2b. Own FASTA → reconcile headers

For PlanMine, AEP hydra, Trinity assemblies, etc.:

```python
from samap.io import match_fasta

rep = match_fasta(a, "transcriptome.fa", gtf="annotation.gtf",
                  write="renamed.fa")
print(rep.scores)      # per-transform overlap fraction — pick the winner with confidence
# rep.names → feed to SAMAP(names={sid: rep.names}) if you keep the original FASTA
```

`match_fasta` scores a transform cascade (`first_token`, `strip_version`,
`gene:=` / `gene_id=`, UniProt `sp|ACC|`, `lcl|` strip, GTF tx→gene,
explicit mapping, …) and writes a renamed FASTA whose headers are exactly
your var_names.

## 3a. Run reciprocal BLAST/DIAMOND

```bash
samap blast \
  --species hu human.fa prot \
  --species mm mouse.fa prot \
  --maps maps/ --threads 16 --cache maps/gnnm.npz
```

or in Python:

```python
from samap.io import run_blast, save_gnnm
from samap.core.homology import _calculate_blast_graph

run_blast({"hu": ("human.fa", "prot"), "mm": ("mouse.fa", "prot")},
          f_maps="maps/", engine="auto", threads=16)
g = _calculate_blast_graph(["hu", "mm"], f_maps="maps/", reciprocate=True)
save_gnnm(g, "maps/gnnm.npz")
```

**Engines** (`--engine auto` policy): DIAMOND for protein-DB targets
(fastest CPU prot↔prot), MMseqs2 for nucleotide-DB targets (only fast
option for translated nucl↔nucl; adds `--gpu 1` automatically when an
NVIDIA GPU is detected), NCBI BLAST+ as last resort. All three emit the
12-column `-outfmt 6` table that `_calculate_blast_graph` reads.

Install all three from bioconda:

```bash
conda install -c bioconda diamond mmseqs2 blast
```

In practice you only need one of `diamond` or `mmseqs2` — MMseqs2 alone
covers every mode; DIAMOND alone leaves the nucleotide quadrants on
BLAST+.

## 3b. Skip BLAST entirely

If you already have orthologs (BioMart Compara export, OrthoDB, OMA,
in-house pipeline):

```python
from samap.io import gnnm_from_pairs

gnnm = gnnm_from_pairs(pairs_df.values,
                       ids={"hu": a_hu.var_names, "mm": a_mm.var_names},
                       weights=pairs_df["perc_id"] / 100)
sm = SAMAP(sams, gnnm=gnnm)
```

Or, for any species eggNOG-mapper has been run on (works for non-model
organisms — emapper takes raw sequences):

```python
from samap.io import homology_from_eggnog

gnnm = homology_from_eggnog({"hu": "hu.emapper.annotations",
                             "mm": "mm.emapper.annotations"},
                            taxon=33208)   # Metazoa
sm = SAMAP(sams, gnnm=gnnm)
```

## 4. Cache the graph

```python
from samap.io import load_gnnm
sm = SAMAP(sams, gnnm=load_gnnm("maps/gnnm.npz"))
```

On the bundled 3-species example: parse BLAST tables 2.8 s → load cache
0.04 s.

## CLI summary

```text
samap detect-ids     H5AD [--var-key COL]
samap fetch-proteome H5AD -o OUT.fa [--source auto|ensembl|ncbi]
samap match-fasta    H5AD FASTA [--gtf GTF] [-o OUT.fa]
samap blast          --species SID FASTA prot|nucl [...] --maps DIR
                     [--engine auto|diamond|blast] [--cache OUT.npz]
```
