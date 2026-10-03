#!/bin/bash
#$ -S /bin/bash
#$ -o novelty.out
#$ -j y
#$ -r y
#$ -R y
#$ -pe smp 32
#$ -cwd
#$ -l mem_free=2G
#$ -l h_rt=24:00:00

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source $SCRIPT_DIR/../../env_vars.sh

conda activate foldseek

DESIGNABLE_DIR=$1
OUTPUTDIR=$2
mkdir $OUTPUTDIR

# i think i also got this from jason yim, need to double check
foldseek easy-search $DESIGNABLE_DIR \
    ${FOLDSEEK_FRAMEFLOW_DB} \
    $OUTPUTDIR/aln_frameflow.tsv \
    $OUTPUTDIR/tmpFolder \
    --alignment-type 1 \
    --format-output query,target,alntmscore,lddt \
    --tmscore-threshold 0.0 \
    --threads 32 \
    --exhaustive-search \
    --max-seqs 10000000000

# i think i also got this from jason yim, need to double check
foldseek easy-search $DESIGNABLE_DIR \
    ${FOLDSEEK_FRAMEDIFF_DB} \
    $OUTPUTDIR/aln_framediff.tsv \
    $OUTPUTDIR/tmpFolder \
    --alignment-type 1 \
    --format-output query,target,alntmscore,lddt \
    --tmscore-threshold 0.0 \
    --exhaustive-search \
    --threads 32 \
    --max-seqs 10000000000

# i think i also got this from jason yim, need to double check
foldseek easy-search $DESIGNABLE_DIR \
    ${FOLDSEEK_PDB_DB} \
    $OUTPUTDIR/aln_PDB.tsv \
    $OUTPUTDIR/tmpFolder \
    --alignment-type 1 \
    --format-output query,target,alntmscore,lddt \
    --tmscore-threshold 0.0 \
    --exhaustive-search \
    --threads 32 \
    --max-seqs 10000000000


# # i think i also got this from jason yim, need to double check
# foldseek easy-search $DESIGNABLE_DIR \
#     ${HOME}/foldseek_db/AF_proteome \
#     $OUTPUTDIR/aln_AFDB_proteome.tsv \
#     $OUTPUTDIR/tmpFolder \
#     --alignment-type 1 \
#     --format-output query,target,alntmscore,lddt \
#     --tmscore-threshold 0.0 \
#     --exhaustive-search \
#     --threads 32 \
#     --max-seqs 10000000000
#
# # i think i also got this from jason yim, need to double check
# foldseek easy-search $DESIGNABLE_DIR \
#     ${HOME}/foldseek_db/AF_swiss_prot \
#     $OUTPUTDIR/aln_AFDB_swiss_prot.tsv \
#     $OUTPUTDIR/tmpFolder \
#     --alignment-type 1 \
#     --format-output query,target,alntmscore,lddt \
#     --tmscore-threshold 0.0 \
#     --exhaustive-search \
#     --threads 32 \
#     --max-seqs 10000000000

# # i think i also got this from jason yim, need to double check
# foldseek easy-search $DESIGNABLE_DIR \
#     ${HOME}/foldseek_db/AF_uniprot50 \
#     $OUTPUTDIR/aln_AFDB_uniprot50.tsv \
#     $OUTPUTDIR/tmpFolder \
#     --alignment-type 1 \
#     --format-output query,target,alntmscore,lddt \
#     --tmscore-threshold 0.0 \
#     --exhaustive-search \
#     --threads 32 \
#     --max-seqs 10000000000


# conda activate ${ENV_NAME}
# python ${REPO_ROOT}/scripts/analysis/summarize_foldseek.py
conda activate "${ENV_NAME}"
python ${REPO_ROOT}/scripts/analysis/summarize_foldseek.py