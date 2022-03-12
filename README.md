# SAMap -- version 1.0.2

# Citation
Please cite the following paper if using SAMap: https://elifesciences.org/articles/66747

Tarashansky, Alexander J., et al. "Mapping single-cell atlases throughout Metazoa unravels cell type evolution." Elife 10 (2021): e66747.

## Installation

### Docker - Easiest
Assumes Docker is installed on your computer.

Run `bash run_image.sh` to run the Docker image. The script will ask you for the container name (e.g. `samap`), volume mount path (e.g. `~/`; this folder should contain your data to be analyzed as it will be mounted onto the Docker image filesystem), and Jupyter server port (e.g. `8888`). If this is your first time running the image, it will be downloaded from the Docker repository.

Running the Docker image will spawn a jupyter notebook server on your specified port.

### pip

`pip install samap`

### Manual installation
Download Anacodna from here:
    https://www.anaconda.com/download/

Create and activate a new environment for SAMap as follows:

```bash
# Install SAMap dependencies availabe in conda
conda create -n SAMap -c conda-forge python=3.7 pip pybind11 h5py=2.10.0 leidenalg python-igraph texttable
conda activate SAMap
```

Having activated the environment, install SAMap like so:


```bash
git clone https://github.com/atarashansky/SAMap.git samap_directory
cd samap_directory
pip install .
```

NCBI BLAST must be installed for the commandline.

```bash
# Define NCBI BLAST version.
ncbi_blast_version='2.9.0'

# Download NCBI BLAST tarball.
wget "ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/${ncbi_blast_version}/ncbi-blast-${ncbi_blast_version}+-x64-linux.tar.gz"

# Extract NCBI BLAST binaries in current conda environment bin directory.
tar -xzvf "ncbi-blast-${ncbi_blast_version}+-x64-linux.tar.gz" \
    -C "${CONDA_PREFIX}/bin/" \
    --strip-components=2 \
    "ncbi-blast-${ncbi_blast_version}+/bin/"
```

Alternatively, add the NCBI BLAST binaries manually to the path:

```bash
# Define NCBI BLAST version.
ncbi_blast_version='2.9.0'

# Download NCBI BLAST tarball.
wget "ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/${ncbi_blast_version}/ncbi-blast-${ncbi_blast_version}+-x64-linux.tar.gz"

# Extract NCBI BLAST tarball.
tar -xzvf "ncbi-blast-${ncbi_blast_version}+-x64-linux.tar.gz"

# Add NCBI BLAST programs to PATH.
echo "export PATH=\"$PATH:/your/directory/ncbi-blast-${ncbi_blast_version}+/bin\"" >> ~/.bashrc
source ~/.bashrc
```

*Installation time should take no more than 10 minutes.*

## Running BLAST

The BLAST mapping script can be run from the `SAMap_vignette.ipynb` Jupyter notebook.

Depending on the number of cores available on your machine and the size/type of the input fasta files, this step may take up to around 4 hours.

## Running SAMap

To run SAMap, use the `SAMAP` function in `samap/mapping.py`. Please see its function documentation for a description of the inputs and outputs. Take a look at the provided Jupyter notebook to get started (`SAMap_vignette.ipynb`).
