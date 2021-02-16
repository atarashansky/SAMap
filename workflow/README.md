# SAMap automated workflow

This is a workflow, built in [Snakemake](https://snakemake.readthedocs.io/en/stable/), to automate the running of SAMap, mapping cell gropus between species for arbitrary input data. It will:

 * Split the input transcriptomes and run the BLAST operations required prior to SAMap. This is instead of [map_genes.sh](map_genes.sh).
 * Generate the initial SAMAP object.
 * Produce the output mapping table and heatmap relating input cell types.

## Installation

The only pre-requisite is Conda (see [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) for starting instructions), and a Conda environment with Snakemake installed. All other software dependencies will be handled by Snakemake. To create that environment:

```
conda create -n snakemake snakemake
```

(substitute [mamba](https://github.com/mamba-org/mamba) for the Conda command above if you have it, which I recommend)

### Cluster cofiguration

If you have never used Snakemake before and you have access to a cluster, you will also want to set things up so that Snakemake can exploit those resources. This can be done in the Snakemake command on every run, but it's much easier to use 'profiles', which you can find [here](https://github.com/Snakemake-Profiles) for a variety of cluster types. 

## Configuration

The workflow operates from a config.yaml which looks like this:

```
blast:
    splits: 1000
    db_exts: [ 'nhr', 'nin', 'nsq', 'nhr', 'nin' ]
    type: nucl
    threads: 16

data:
    hu:
        anndata: hu.h5ad
        transcriptome: Homo_sapiens.GRCh38.cdna.all.99.fa.gz
    mu:
        anndata: mu.h5ad
        transcriptome: Mus_musculus.GRCm38.cdna.all.99.fa.gz

cell_type_field: 'inferred_cell_type_-_ontology_labels'
outdir: 'out'
```

Generate your own config.yaml based on the above, and store it in the directory from which you will run Snakemake.

The 'blast' section configures how blast will be run to generate the homology mappings used by SAMap. If you have access to a cluster, setting splits to a high number as in the above example will save you considerable time, with the BLAST operations spread over multiple nodes. 

The key section is 'data'. Create your own species prefixes in place of 'hu' and 'mu' above, and for each set a transcriptome and an input pair of anndata objects.

The 'cell_type_field' tells SAMap which of the columns in .obs from your input objects should be used to define cell types. 'outdir' just specifies where the results go.

## Running

With the above config done, you can execute the workflow:

```
snakemake -s /path/to/Snakefile
```

Where /path/to is the workflow directory.

If you want to use a cluster configuration profile as described above the command is:

```
snakemake -s /path/to/Snakefile --profile lsf
```

(in this example for an LSF cluster).


## Output

