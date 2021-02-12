configfile: "config.yaml"

wildcard_constraints:
    prefix1 = "[^\/\.]+",
    prefix2 = "[^\/\.]+",

rule all:
    input:    
        sam="{path}/hu_mu.sam.pkl"

rule unzip_transcriptome:
    input:
        fagz=lambda wildcards: config["data"][wildcards.prefix]['transcriptome']

    output:
        fa='{prefix}.fa'

    shell:
        "zcat {input.fagz} > {output.fa}"


rule split_transcriptome: 
    conda:
         'envs/fasta-splitter.yml'
    input:
        fa="{prefix}.fa"
    
    output:
        fa=temp(expand("{{prefix}}.part-{n}.fa", n=range(1, config.get('blast').get('splits')+1)))

    params:
        splits=config.get('blast').get('splits')

    shell:
        "fasta-splitter --nopad --n-parts {params.splits} --out-dir $(pwd) {input.fa}"

rule make_blast_db:
    conda:
         'envs/blast.yml'
    
    input:
        fa="{prefix}"
 
    output:
        db=temp(expand('{{prefix}}.{ext}', ext = config.get('blast').get('db_exts')))

    params:
        type=config.get("blast").get("type")

    shell:
        "makeblastdb -in {input.fa} -dbtype {params.type}"

rule tblastx:
    conda: 'envs/blast.yml'
    
    threads: config.get('blast').get('threads')

    input:
        query="{prefix1}.part-{n}.fa",
        db=expand('{{prefix2}}.fa.{ext}', ext = config.get('blast').get('db_exts')),
        dbfa="{prefix2}.fa"

    output:
        map = "{dir}/{prefix1}_to_{prefix2}.{n}.txt"

    shell:
        "tblastx -query {input.query} -db {input.dbfa} -outfmt 6 -out {output.map} -num_threads {threads} -max_hsps 1 -evalue 1e-6"

rule merge_blast:
    input:
        maps = expand("{{dir}}/{{prefix1}}_to_{{prefix2}}.{n}.txt", n=range(1, config.get('blast').get('splits')+1))

    output:
        merged=protected("{dir}/{prefix1}_to_{prefix2}.txt")
    
    shell:
        "cat {input.maps} > {output.merged}"

rule transcript_to_gene:
    input:
        fa=lambda wildcards: config["data"][wildcards.prefix]['transcriptome']
    output:
        txt="{prefix}.t2gene"

    """
    zcat {input.fa} |  grep '>' | awk '{print $1"\t"$4}' | sed 's/>//' | sed 's/gene://g' | sed -r 's/\.[0-9]+//g' > {output.txt}
    """

rule samap:
    
    conda: 'envs/samap.yml'

    input:
        transcript_to_gene1="{prefix1}.t2gene",
        transcript_to_gene2="{prefix2}.t2gene",
        maps = lambda wildards: [ "%s/maps/%s" % (config.get('outdir'), file) for file in [ '%s%s/%s_to_%s.txt' % (wildcards.prefix1, wildcars.prefix2, wildcards.prefix1, wildcards.prefix2), 'maps/%s%s/%s_to_%s.txt' % (wildcards.prefix1, wildcars.prefix2, wildcards.prefix2, wildcards.prefix1) ] ]

    output:
        pkl="{path}/{prefix1}_{prefix2}.sam.pkl"

    params:
        maps="%s/maps/" % (config.get('outdir')
        anndata1=lambda wildcards: config["data"][wildcards.prefix1]['anndata']
        anndata2=lambda wildcards: config["data"][wildcards.prefix2]['anndata']

    resources:
        mem_mb=lambda wildcards, attempt: attempt * 32000

    shell:
        "samap-run --id1={wildcards.prefix1} --id2={wildcards.prefix2} {params.anndata1} {params.anndata2} {params.maps} {input.transcript_to_gene1} {input.transcript_to_gene2} {output.pkl}"

