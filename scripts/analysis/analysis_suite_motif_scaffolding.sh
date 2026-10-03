#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source $SCRIPT_DIR/../../env_vars.sh

python ${REPO_ROOT}/scripts/analysis/partition_motif_samples.py --folded_folders esmfold/ --samples samples/
python ${REPO_ROOT}/scripts/analysis/plot_sample_dssp.py
python ${REPO_ROOT}/scripts/analysis/plot_folded_dssp.py
python ${REPO_ROOT}/scripts/analysis/folding_over_len_motif.py
python ${REPO_ROOT}/scripts/analysis/designability_over_len_motif.py
python ${REPO_ROOT}/scripts/analysis/which_tasks_passed.py
python ${REPO_ROOT}/scripts/analysis/confusion_matrix_motif.py

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
# conda activate ${ENV_NAME}
# qsub ${REPO_ROOT}/scripts/analysis/tmalign.sh $PWD