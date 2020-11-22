#!/bin/bash
read -e -p 'Transcriptome 1: ' tr1
read -e -p 'Transcriptome type: ' t1
read -e -p 'Transcriptome ID: ' n1
read -e -p 'Transcriptome 2: ' tr2
read -e -p 'Transcriptome type: ' t2
read -e -p 'Transcriptome ID: ' n2

n1 = "${n1:0:2}"
n2 = "${n2:0:2}"

mkdir -p "maps"
mkdir -p "maps/${n1}${n2}"

if [ ! -f "${tr1}.nhr" ] && [ ! -f "${tr1}.phr" ]
then
makeblastdb -in $tr1 -dbtype $t1
fi

if [ ! -f "${tr2}.nhr" ] && [ ! -f "${tr2}.phr" ]
then
makeblastdb -in $tr2 -dbtype $t2
fi

if [[ "$t1" == "nucl" && "$t2" == "nucl" ]]
then
tblastx -query $tr1 -db $tr2 -outfmt 6 -out "maps/${n1}${n2}/${n1}_to_${n2}.txt" -num_threads 4 -max_hsps 1 -evalue 1e-6 &
tblastx -query $tr2 -db $tr1 -outfmt 6 -out "maps/${n1}${n2}/${n2}_to_${n1}.txt" -num_threads 4 -max_hsps 1 -evalue 1e-6 &
fi

if [[ "$t1" == "nucl" && "$t2" == "prot" ]]
then
blastx -query $tr1 -db $tr2 -outfmt 6 -out "maps/${n1}${n2}/${n1}_to_${n2}.txt" -num_threads 4 -max_hsps 1 -evalue 1e-6 &
tblastn -query $tr2 -db $tr1 -outfmt 6 -out "maps/${n1}${n2}/${n2}_to_${n1}.txt" -num_threads 4 -max_hsps 1 -evalue 1e-6 &
fi

if [[ "$t1" == "prot" && "$t2" == "nucl" ]]
then
tblastn -query $tr1 -db $tr2 -outfmt 6 -out "maps/${n1}${n2}/${n1}_to_${n2}.txt" -num_threads 4 -max_hsps 1 -evalue 1e-6 &
blastx -query $tr2 -db $tr1 -outfmt 6 -out "maps/${n1}${n2}/${n2}_to_${n1}.txt" -num_threads 4 -max_hsps 1 -evalue 1e-6 &
fi

if [[ "$t1" == "prot" && "$t2" == "prot" ]]
then
blastp -query $tr1 -db $tr2 -outfmt 6 -out "maps/${n1}${n2}/${n1}_to_${n2}.txt" -num_threads 4 -max_hsps 1 -evalue 1e-6 &
blastp -query $tr2 -db $tr1 -outfmt 6 -out "maps/${n1}${n2}/${n2}_to_${n1}.txt" -num_threads 4 -max_hsps 1 -evalue 1e-6 &
fi
