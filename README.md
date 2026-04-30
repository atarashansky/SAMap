# SAMap -- version 3.0.0

# Citation
Please cite the following paper if using SAMap: https://elifesciences.org/articles/66747

Tarashansky, Alexander J., et al. "Mapping single-cell atlases throughout Metazoa unravels cell type evolution." Elife 10 (2021): e66747.

> **Requirements:** Python ≥3.11. See `pyproject.toml` for the full dependency list.

## Installation

SAMap requires **Python ≥3.11**.

### From PyPI (recommended)

```bash
conda create -n SAMap -c conda-forge python=3.12 pip
conda activate SAMap
pip install sc-samap
```

### From source (development)

```bash
conda create -n SAMap -c conda-forge python=3.12 pip
conda activate SAMap
git clone https://github.com/atarashansky/SAMap.git
cd SAMap
pip install -e .
```

### NCBI BLAST

SAMap requires NCBI BLAST on your `PATH` for the homology mapping step.

Easiest via conda:
```bash
conda install -c bioconda blast
```

Or download binaries directly from [NCBI](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/).

*Installation time should take no more than 10 minutes.*

## Preparing inputs / running BLAST

See [`docs/io.md`](docs/io.md) for the full input-preparation workflow.
In brief:

```bash
# 1. find out what your gene IDs are
samap detect-ids species1.h5ad

# 2a. Ensembl / NCBI GeneID → fetch a FASTA whose headers ARE var_names
samap fetch-proteome species1.h5ad -o sp1.fa

# 2b. own FASTA → reconcile headers against var_names
samap match-fasta species1.h5ad transcriptome.fa --gtf ann.gtf -o sp1.fa

# 3. all reciprocal alignments (DIAMOND-first), one command
samap blast --species sp1 sp1.fa prot --species sp2 sp2.fa prot \
    --maps maps/ --threads 16 --cache maps/gnnm.npz
```

You can also skip BLAST entirely and feed `SAMAP(gnnm=...)` a homology
graph built from eggNOG-mapper output or a BioMart ortholog export — see
`samap.io.homology_from_eggnog` / `samap.io.gnnm_from_pairs`.

Depending on the number of cores and the size/type of the input FASTAs,
the alignment step may take up to a few hours with NCBI BLAST+; DIAMOND
on protein inputs is typically minutes.

## Running SAMap

To run SAMap, use the `SAMAP` class from `samap`:

```python
from samap import SAMAP
sm = SAMAP(sams={'sp1': 'species1.h5ad', 'sp2': 'species2.h5ad'}, f_maps='maps/')
sm.run()
```

See the function documentation for a description of the inputs and outputs. Take a look at the provided Jupyter notebook to get started (`SAMap_vignette.ipynb`).

