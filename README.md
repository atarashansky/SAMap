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

## Running BLAST

The BLAST mapping script can be run from the `SAMap_vignette.ipynb` Jupyter notebook.

Depending on the number of cores available on your machine and the size/type of the input fasta files, this step may take up to around 4 hours.

## Running SAMap

To run SAMap, use the `SAMAP` class from `samap`:

```python
from samap import SAMAP
sm = SAMAP(sams={'sp1': 'species1.h5ad', 'sp2': 'species2.h5ad'}, f_maps='maps/')
sm.run()
```

See the function documentation for a description of the inputs and outputs. Take a look at the provided Jupyter notebook to get started (`SAMap_vignette.ipynb`).

