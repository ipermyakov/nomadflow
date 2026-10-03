#!/bin/bash
#$ -S /bin/bash
#$ -o /dev/null
#$ -j y
#$ -r y
#$ -cwd
#$ -l mem_free=2G
#$ -l h_rt=2:00:00

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "${SCRIPT_DIR}/../../env_vars.sh"

cd $1/designable_samples/
ls *.pdb > usalign.list
cd -
~/software/bin/USalign -dir $1/designable_samples/ $1/designable_samples/usalign.list -outfmt 2 > $1/usalign.tsv
conda activate protienzen
python ${REPO_ROOT}/scripts/analysis/summarize_tmalign.py