#!/bin/bash
#$ -S /bin/bash
#$ -o esmfold.log
#$ -j y
#$ -r y
#$ -cwd
#$ -q gpu.q
#$ -pe smp 1
#$ -l mem_free=32G
#$ -l h_rt=4:00:00
#$ -l compute_cap=61,gpu_mem=40G

set -euo pipefail

if [ -n "${SGE_GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="${SGE_GPU}"
    echo "SGE_GPU=${SGE_GPU}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../../env_vars.sh"

_activate_conda() {
    local env_name="$1"
    if [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "${CONDA_DEFAULT_ENV}" = "${env_name}" ]; then
        return 0
    fi
    if [ -n "${CONDA_EXE:-}" ] && [ -f "$(dirname "${CONDA_EXE}")/../etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "$(dirname "${CONDA_EXE}")/../etc/profile.d/conda.sh"
    elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    else
        echo "Could not find conda.sh. Run 'conda init bash' once or export CONDA_EXE." >&2
        exit 1
    fi
    conda activate "${env_name}"
}

if command -v module >/dev/null 2>&1; then
    module load Sali cuda 2>/dev/null || true
fi

fasta="${PWD}/../seqs/"
outdir="${PWD}"
esmfold_args=()
if [ "${ESMFOLD_FORCE:-0}" = "1" ]; then
    esmfold_args+=(--force)
fi

run_esmfold() {
    python "${REPO_ROOT}/scripts/analysis/batched_esmfold_inference.py" \
        -i "${fasta}" \
        -o "${outdir}" \
        "${esmfold_args[@]}" \
        "$@"
}

if [ -n "${ESMFOLD_SIF:-}" ] && [ -f "${ESMFOLD_SIF}" ] && command -v apptainer >/dev/null 2>&1; then
    apptainer exec --nv "${ESMFOLD_SIF}" python \
        "${REPO_ROOT}/scripts/analysis/batched_esmfold_inference.py" \
        -i "${fasta}" \
        -o "${outdir}" \
        "${esmfold_args[@]}"
else
    _activate_conda "${ESMFOLD_ENV:-esmfold}"
    if ! python -c "from openfold.data.data_transforms import make_atom14_masks" >/dev/null 2>&1; then
        echo "Missing OpenFold in conda env '${ESMFOLD_ENV:-esmfold}'." >&2
        echo "Run: bash ${REPO_ROOT}/scripts/setup/install_esmfold_env.sh" >&2
        exit 1
    fi
    run_esmfold
fi
