#!/bin/bash
#$ -S /bin/bash
#$ -o tmp.log
#$ -j y
#$ -r y
#$ -cwd
#$ -q gpu.q
#$ -pe smp 1
#$ -l mem_free=32G
#$ -l h_rt=60:00:00
#$ -l compute_cap=61,gpu_mem=40G

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <sampling_run_dir> [analysis_out_dir]"
    exit 1
fi

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "${SCRIPT_DIR}/../../../env_vars.sh"

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

PMPNN_ENV="${PMPNN_ENV:-proteinmpnn}"

RUN_DIR="$(readlink -f "$1")"
RUN_ID="$(basename "${RUN_DIR}")"
ANALYSIS_OUT="${2:-${REPO_ROOT}/article_results/analysis_runs/${RUN_ID}}"

if [ ! -d "${RUN_DIR}/samples" ] && [ ! -d "${RUN_DIR}/samples_eval" ]; then
    echo "Missing samples/ or samples_eval/ under ${RUN_DIR}"
    exit 1
fi

if [ -d "${RUN_DIR}/samples_eval" ]; then
    folder_with_pdbs="${RUN_DIR}/samples_eval"
else
    folder_with_pdbs="${RUN_DIR}/samples"
fi

if [ ! -f "${RUN_DIR}/samples_metadata.json" ] && ! ls "${RUN_DIR}"/metadata/samples_metadata_rank*_batch*.json >/dev/null 2>&1; then
    echo "Missing samples metadata under ${RUN_DIR}"
    exit 1
fi

cd "${RUN_DIR}"

## ProteinMPNN
SKIP_PMPNN=0
_activate_conda "${ENV_NAME}"
if [ "${PMPNN_FORCE:-0}" != "1" ]; then
    if PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python "${REPO_ROOT}/scripts/analysis/check_eval_layout.py" \
        --sampling-run "${RUN_DIR}" --seqs-complete; then
        echo "Skipping ProteinMPNN; seqs/ already complete under ${RUN_DIR}"
        SKIP_PMPNN=1
    fi
fi

if [ "${SKIP_PMPNN}" -eq 0 ]; then
_activate_conda "${PMPNN_ENV}"

output_dir="${RUN_DIR}"
path_for_parsed_chains="${output_dir}/parsed_pdbs.jsonl"
path_for_fixed_pos="${output_dir}/pmpnn_fixed_pos_dict.jsonl"

python "${PROTEINMPNN_ROOT:-$HOME/software/ProteinMPNN}/helper_scripts/parse_multiple_chains.py" \
    --input_path="${folder_with_pdbs}" \
    --output_path="${path_for_parsed_chains}"

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python "${REPO_ROOT}/scripts/analysis/ensure_pmpnn_fixed_positions.py" \
    --parsed-jsonl "${path_for_parsed_chains}" \
    --fixed-positions-json "${path_for_fixed_pos}" \
    --samples-metadata "${RUN_DIR}/samples_metadata.json"

python "${PROTEINMPNN_ROOT:-$HOME/software/ProteinMPNN}/protein_mpnn_run.py" \
    --jsonl_path "${path_for_parsed_chains}" \
    --fixed_positions_jsonl "${path_for_fixed_pos}" \
    --out_folder "${output_dir}" \
    --num_seq_per_target 8 \
    --sampling_temp "0.1" \
    --batch_size 8
fi

## ESMFold
mkdir -p "${RUN_DIR}/esmfold"
cd "${RUN_DIR}/esmfold"
if [ "${ESMFOLD_FORCE:-0}" != "1" ] && [ -f esmfold.log ] && grep -q "Predicted structure for" esmfold.log; then
    echo "Reusing existing ESMFold outputs and esmfold.log"
else
    if [ "${ESMFOLD_FORCE:-0}" != "1" ] && compgen -G "${RUN_DIR}/esmfold/len_*" >/dev/null; then
        echo "Existing ESMFold folders found without a reusable esmfold.log; forcing ESMFold regeneration"
        ESMFOLD_FORCE=1 bash "${REPO_ROOT}/scripts/analysis/esmfold.sh" > esmfold.log 2>&1
    else
        bash "${REPO_ROOT}/scripts/analysis/esmfold.sh" > esmfold.log 2>&1
    fi
fi

if [ "${SKIP_ANALYSIS:-0}" = "1" ]; then
    echo "SKIP_ANALYSIS=1; stopping after ESMFold under ${RUN_DIR}"
    exit 0
fi

## RMSD tables
_activate_conda "${ENV_NAME}"

if [ ! -f "${RUN_DIR}/samples_metadata.json" ]; then
    cd "${REPO_ROOT}"
    RUN_DIR="${RUN_DIR}" python - <<'PY'
import json
import os
from pathlib import Path

from nomadflow.benchmark.analysis_inputs import load_samples_metadata

run_dir = Path(os.environ["RUN_DIR"])
merged = load_samples_metadata(run_dir)
with open(run_dir / "samples_metadata.json", "w", encoding="utf-8") as handle:
    json.dump(merged, handle)
PY
fi

cd "${RUN_DIR}/esmfold"
SAMPLES_DIR="${RUN_DIR}/samples_eval"
if [ ! -d "${SAMPLES_DIR}" ]; then
    SAMPLES_DIR="${RUN_DIR}/samples"
fi
PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" RUN_DIR="${RUN_DIR}" python - <<'PY'
import os
from pathlib import Path

from nomadflow.benchmark.layout_manifest_eval import enrich_samples_metadata_file

enrich_samples_metadata_file(Path(os.environ["RUN_DIR"]), write=True)
PY
PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python "${REPO_ROOT}/scripts/analysis/esm_compute_rmsds.py" \
    --esmlog esmfold.log \
    --folded_folders "${PWD}" \
    --samples "${SAMPLES_DIR}" \
    --samples_metadata "${RUN_DIR}/samples_metadata.json" \
    --sampling-run "${RUN_DIR}"

cd "${REPO_ROOT}"

## Benchmark analysis
ANALYSIS_ARGS=(
    --sampling-run "${RUN_DIR}"
    --analysis-out "${ANALYSIS_OUT}"
)
if [ "${WRITE_MULTIMOTIF_ALIGNMENT_PDBS:-0}" = "1" ]; then
    ANALYSIS_ARGS+=(--write-multimotif-alignment-pdbs)
fi
if [ "${ANALYSIS_OVERWRITE:-0}" = "1" ]; then
    ANALYSIS_ARGS+=(--overwrite)
fi
if [ "${SKIP_FOLDSEEK:-0}" = "1" ]; then
    ANALYSIS_ARGS+=(--skip-foldseek)
fi
PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python "${REPO_ROOT}/scripts/analysis/run_motif_benchmark_analysis.py" \
    "${ANALYSIS_ARGS[@]}"

echo "Analysis artifacts written to ${ANALYSIS_OUT}"
