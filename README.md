# SAMap -- version 0.1.1
The SAMap algorithm.

## Requirements
SAMap was developed and tested in an Anaconda python environment with the following dependencies:
 - `sam-algorithm==0.7.6`
 - `scanpy==1.5.1`
 - `hnswlib==0.3.4`
 - `matplotlib==3.1.3`

## Installation

Download Anacodna from here:
    https://www.anaconda.com/download/

Create and activate a new environment with python3.6 as follows:
```
conda create -n environment_name python=3.6
conda activate environment_name
```
Having activated the environment, install SAMap like so:

```
cd samap_directory
pip install .
```

NCBI BLAST must be installed for the commandline. Download version 2.1.0 here for LINUX/UNIX systems (here)[ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.9.0/ncbi-blast-2.9.0+-x64-linux.tar.gz]. Unpack the tarball with:
```
tar -xzvf ncbi-blast-2.9.0+-x64-linux.tar.gz
```
Add the bin files to your path with.
```
echo "export PATH=\"$PATH:/your/directory/ncbi-blast-2.9.0+/bin\"" >> ~/.bashrc
source ~/.bashrc
```
*Installation time should take no more than 5 minutes.*

## Running SAMap

First, we need to map the transcriptomes to generate the homology graph. For convenience, you may wish to place the transcriptomes/proteomes and the `map_genes.sh` bash script in the same directory as your data (which should be `AnnData` `.h5ad` files).

Run the mapping bash script:
```
bash map_genes.sh
```
The script will ask you for the paths to your fasta files and whether they are transcriptomes (`nucl`) or proteomes (`prot`). It will also ask you to assign a two-character ID for each dataset (e.g. `ze` for Zebrafish).

Depending on the number of cores available on your machine and the size/type of the input fasta files, this step may take up to around 4 hours.

To run SAMap, use the `SAMAP` function in `SAMap.py`. Please see its function documentation for a description of the inputs and outputs.
