#!/bin/bash
mkdir -p "maps"
mkdir -p "maps/zexe"

makeblastdb -in zebrafish.fasta -dbtype nucl
makeblastdb -in xenopus.fasta -dbtype nucl
tblastx -query zebrafish.fasta -db xenopus.fasta -outfmt 6 -out "maps/zexe/ze_to_xe.txt" -num_threads 4 -max_hsps 1 -evalue 1e-6 &
tblastx -query xenopus.fasta -db zebrafish.fasta -outfmt 6 -out "maps/zexe/xe_to_ze.txt" -num_threads 4 -max_hsps 1 -evalue 1e-6 &