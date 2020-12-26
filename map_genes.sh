#!/bin/bash

#parse arguments
die() {
	printf '%s\n' "$1" >&2
	exit 1
}

_help() {
	printf "Usage: map_genes.sh [--tr1] [--t1] [--n1] [--tr2] [--t2] [--n2]\n\t[--tr1]: path to transcriptome/proteome 1\n\t[--t1]: is 1 a transcriptome [nucl] or proteome [prot]\n\t[--n1]: two character identifier of 1\n\t[--tr2]: path to transcriptome/proteome 2\n\t[--t2]: is 2 a transcriptome [nucl] or proteome [prot]\n\t[--n2]: two character identifier of 2"
}

if [ $# -eq 0 ]; then
    _help
    exit 1
fi

while :; do
    case $1 in
        -h|-\?|--help)
            _help
	    exit
            ;;
        --tr1)       # Takes an option argument; ensure it has been specified.
            if [ "$2" ]; then
                tr1=$2
                shift
            else
                die 'ERROR: "--tr1" requires a non-empty option argument.'
            fi
            ;;
        --tr2)       # Takes an option argument; ensure it has been specified.
            if [ "$2" ]; then
                tr2=$2
                shift
            else
                die 'ERROR: "--tr2" requires a non-empty option argument.'
            fi
            ;;
        --n1)       # Takes an option argument; ensure it has been specified.
            if [ "$2" ]; then
                n1=$2
                shift
            else
                die 'ERROR: "--n1" requires a non-empty option argument.'
            fi
            ;;
        --n2)       # Takes an option argument; ensure it has been specified.
            if [ "$2" ]; then
                n2=$2
                shift
            else
                die 'ERROR: "--n2" requires a non-empty option argument.'
            fi
            ;;
        --t1)       # Takes an option argument; ensure it has been specified.
            if [ "$2" ]; then
                t1=$2
                shift
            else
                die 'ERROR: "--t1" requires a non-empty option argument.'
            fi
            ;;
        --t2)       # Takes an option argument; ensure it has been specified.
            if [ "$2" ]; then
                t2=$2
                shift
            else
                die 'ERROR: "--t2" requires a non-empty option argument.'
            fi
            ;;
        --)              # End of all options.
            shift
            break
            ;;
        -?*)
            printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
            ;;
        *)               # Default case: No more options, so break out of the loop.
            break
    esac

    shift
done



#read -e -p 'Transcriptome 1: ' tr1
#read -e -p 'Transcriptome type: ' t1
#read -e -p 'Transcriptome ID: ' n1
#read -e -p 'Transcriptome 2: ' tr2
#read -e -p 'Transcriptome type: ' t2
#read -e -p 'Transcriptome ID: ' n2

n1="${n1:0:2}"
n2="${n2:0:2}"

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
echo "Running tblastx in both directions"
tblastx -query $tr1 -db $tr2 -outfmt 6 -out "maps/${n1}${n2}/${n1}_to_${n2}.txt" -num_threads 8 -max_hsps 1 -evalue 1e-6
tblastx -query $tr2 -db $tr1 -outfmt 6 -out "maps/${n1}${n2}/${n2}_to_${n1}.txt" -num_threads 8 -max_hsps 1 -evalue 1e-6
fi

if [[ "$t1" == "nucl" && "$t2" == "prot" ]]
then
echo "Running blastx from 1 to 2 and tblastn from 2 to 1"
blastx -query $tr1 -db $tr2 -outfmt 6 -out "maps/${n1}${n2}/${n1}_to_${n2}.txt" -num_threads 8 -max_hsps 1 -evalue 1e-6
tblastn -query $tr2 -db $tr1 -outfmt 6 -out "maps/${n1}${n2}/${n2}_to_${n1}.txt" -num_threads 8 -max_hsps 1 -evalue 1e-6
fi

if [[ "$t1" == "prot" && "$t2" == "nucl" ]]
then
echo "Running tblastn from 1 to 2 and blastx from 2 to 1"
tblastn -query $tr1 -db $tr2 -outfmt 6 -out "maps/${n1}${n2}/${n1}_to_${n2}.txt" -num_threads 8 -max_hsps 1 -evalue 1e-6
blastx -query $tr2 -db $tr1 -outfmt 6 -out "maps/${n1}${n2}/${n2}_to_${n1}.txt" -num_threads 8 -max_hsps 1 -evalue 1e-6
fi

if [[ "$t1" == "prot" && "$t2" == "prot" ]]
then
echo "Running blastp in both directions"
blastp -query $tr1 -db $tr2 -outfmt 6 -out "maps/${n1}${n2}/${n1}_to_${n2}.txt" -num_threads 8 -max_hsps 1 -evalue 1e-6
blastp -query $tr2 -db $tr1 -outfmt 6 -out "maps/${n1}${n2}/${n2}_to_${n1}.txt" -num_threads 8 -max_hsps 1 -evalue 1e-6
fi
