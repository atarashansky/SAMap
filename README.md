# SAMap -- version 0.1.2
The SAMap algorithm.

# New in version 0.1.2
Users can provide a list of name pairs ( `(Fasta header ID, Dataset gene symbol)` ) mapping the fasta header IDs to the dataset gene symbol as input to SAMap. This is useful when the annotated GTF or transcriptome has multiple transcripts or isoforms for the same gene symbol. In cases where the same dataset gene symbol maps to multiple transcripts or isoforms in the BLAST results, they are all collapsed into a supernode in the computed BLAST homology graph.

# Beta
Hello! Just a friendly reminder that if any of you have trouble getting SAMap up and running or want to know more about the various arguments, please do not hesitate to reach out to me by submitting an issue!

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
git clone https://github.com/atarashansky/SAMap.git samap_directory
cd samap_directory
pip install .
```

NCBI BLAST must be installed for the commandline. Download version 2.9.0 here for LINUX/UNIX systems (here)[ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.9.0/ncbi-blast-2.9.0+-x64-linux.tar.gz]. Unpack the tarball with:
```
tar -xzvf ncbi-blast-2.9.0+-x64-linux.tar.gz
```
Add the bin files to your path with.
```
echo "export PATH=\"$PATH:/your/directory/ncbi-blast-2.9.0+/bin\"" >> ~/.bashrc
source ~/.bashrc
```
*Installation time should take no more than 5 minutes.*

## Running BLAST

First, we need to map the transcriptomes to generate the homology graph. For convenience, you may wish to place the transcriptomes/proteomes and the `map_genes.sh` bash script in the same directory as your data (which should be `AnnData` `.h5ad` files).

Run the mapping bash script:
```
bash map_genes.sh
```
The script will ask you for the paths to your fasta files and whether they are transcriptomes (`nucl`) or proteomes (`prot`). It will also ask you to assign a two-character ID for each dataset (e.g. `ze` for Zebrafish). These IDs will be input by the user into the SAMap algorithm (see `SAMap_quickstart.ipynb`).

Depending on the number of cores available on your machine and the size/type of the input fasta files, this step may take up to around 4 hours.

## Running SAMap

To run SAMap, use the `SAMAP` function in `SAMap.py`. Please see its function documentation for a description of the inputs and outputs. Take a look at the provided Jupyter notebook to get started (`SAMap_quickstart.ipynb`).
