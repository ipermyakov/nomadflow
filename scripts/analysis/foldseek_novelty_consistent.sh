#!/bin/bash
#$ -S /bin/bash
#$ -o novelty_precise.out
#$ -j y
#$ -r y
#$ -R y
#$ -cwd
#$ -pe smp 32
#$ -l mem_free=2G
#$ -l h_rt=24:00:00

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source $SCRIPT_DIR/../../env_vars.sh

conda activate foldseek

CONSISTENT_DIR=$1
OUTPUTDIR=$2
mkdir $OUTPUTDIR

# i think i also got this from jason yim, need to double check
foldseek easy-search $CONSISTENT_DIR \
    ${FOLDSEEK_PDB_DB} \
    $OUTPUTDIR/aln_PDB_consistent.tsv \
    $OUTPUTDIR/tmpFolder \
    --alignment-type 1 \
    --format-output query,target,alntmscore,lddt \
    --tmscore-threshold 0.0 \
    --exhaustive-search \
    --threads 32 \
    --max-seqs 10000000000

# # i think i also got this from jason yim, need to double check
# foldseek easy-search $CONSISTENT_DIR \
#     ${HOME}/foldseek_db/AF_proteome \
#     $OUTPUTDIR/aln_AFDB_proteome_precise.tsv \
#     $OUTPUTDIR/tmpFolder \
#     --alignment-type 1 \
#     --format-output query,target,alntmscore,lddt \
#     --tmscore-threshold 0.0 \
#     --exhaustive-search \
#     --threads 32 \
#     --max-seqs 10000000000
#
# # i think i also got this from jason yim, need to double check
# foldseek easy-search $CONSISTENT_DIR \
#     ${HOME}/foldseek_db/AF_swiss_prot \
#     $OUTPUTDIR/aln_AFDB_swiss_prot_precise.tsv \
#     $OUTPUTDIR/tmpFolder \
#     --alignment-type 1 \
#     --format-output query,target,alntmscore,lddt \
#     --tmscore-threshold 0.0 \
#     --threads 32 \
#     --exhaustive-search \
#     --max-seqs 10000000000
#
# # # i think i also got this from jason yim, need to double check
# # foldseek easy-search $CONSISTENT_DIR \
# #     ${HOME}/foldseek_db/AF_uniprot50 \
# #     $OUTPUTDIR/aln_AFDB_uniprot_precise.tsv \
# #     $OUTPUTDIR/tmpFolder \
# #     --alignment-type 1 \
# #     --format-output query,target,alntmscore,lddt \
# #     --tmscore-threshold 0.0 \
# #     --exhaustive-search \
# #     --max-seqs 10000000000  \
# #     --threads 32
#
#
# conda activate ${ENV_NAME}
# python ${REPO_ROOT}/scripts/analysis/summarize_foldseek.py
conda activate "${ENV_NAME}"
python ${REPO_ROOT}/scripts/analysis/summarize_foldseek_consistent.py