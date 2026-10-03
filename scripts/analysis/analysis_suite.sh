#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source $SCRIPT_DIR/../../env_vars.sh

python ${REPO_ROOT}/scripts/analysis/extract_designable_samples.py --csv esmfold/best_sc_rmsd.csv --folded_folders esmfold/ --samples samples/
python ${REPO_ROOT}/scripts/analysis/extract_consistent_samples.py --csv esmfold/folding_rmsd.csv --folded_folders esmfold/ --samples samples/
python ${REPO_ROOT}/scripts/analysis/extract_precise_samples.py --csv esmfold/folding_rmsd.csv --folded_folders esmfold/ --samples samples/
# qsub ${REPO_ROOT}/scripts/analysis/foldseek_novelty.sh designable_samples_folded/ novelty/
# qsub ${REPO_ROOT}/scripts/analysis/foldseek_novelty_consistent.sh consistent_samples_folded/ novelty/
# qsub ${REPO_ROOT}/scripts/analysis/foldseek_novelty_precise.sh precise_samples_folded/ novelty/
python ${REPO_ROOT}/scripts/analysis/plot_sample_dssp.py
python ${REPO_ROOT}/scripts/analysis/plot_folded_dssp.py
python ${REPO_ROOT}/scripts/analysis/esmfold_over_len.py
python ${REPO_ROOT}/scripts/analysis/folding_over_len.py
python ${REPO_ROOT}/scripts/analysis/confusion_matrix.py

# TODO: there's gotta be a better way than this
# >>> conda initialize >>>
__conda_setup="$('${HOME}/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
        . "${HOME}/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="${HOME}/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate foldseek
python ${REPO_ROOT}/scripts/analysis/foldseek_diversity.py
conda activate ${ENV_NAME}
qsub ${REPO_ROOT}/scripts/analysis/tmalign.sh $PWD