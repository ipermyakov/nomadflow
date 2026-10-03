#!/usr/bin/env bash
# Baseline (vanilla proteinzen-main) motif sampling + group annotation + eval.
#
# Type-3 launch mode: sample with normal contig fixation in STAGE2_REPO_ROOT,
# keep motif_groups labels for per-group metrics, then run the article eval chain.
#
# Examples:
#   ./scripts/run_baseline_motif_pipeline.sh
#
#   ./scripts/run_baseline_motif_pipeline.sh \
#     --config configs/sampling/protpardelle_motif_scaffolding/config.yaml \
#     --only sample
#
#   ./scripts/run_baseline_motif_pipeline.sh \
#     --out-dir article_results/sampling_runs/original_proteinzen__catalytic_small__pretrained__20260711 \
#     --only inject
#
#   ./scripts/run_baseline_motif_pipeline.sh --only eval

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../env_vars.sh"

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONPATH

DEFAULT_CONFIG="${REPO_ROOT}/configs/sampling/protpardelle_motif_scaffolding/config.yaml"

TASKS_YAML="${DEFAULT_CONFIG}"
RUN_NAME=""
CHECKPOINT="${STAGE2_BASELINE_CKPT}"
STAGE2_MODEL="${STAGE2_MODEL_DIR}"
STAGE2_REPO="${STAGE2_REPO_ROOT}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-1}"
ONLY="all"
DRY_RUN=0
WRITE_MULTIMOTIF_ALIGNMENT_PDBS=0
BATCH_SIZE=1
NUM_SAMPLES_PER_TASK=""

OUT_DIR=""
ANALYSIS_OUT=""

TMPDIR="${TMPDIR:-${HOME}/tmp}"
export TMPDIR TEMP="${TMPDIR}" TMP="${TMPDIR}"

usage() {
    cat <<'EOF'
Usage: run_baseline_motif_pipeline.sh [options]

Main options:
  --config PATH            Benchmark tasks yaml with motif_groups (alias: --tasks-yaml)
                           default: config.yaml
  --run-name NAME          Run slug (default: original_proteinzen__<config_slug>__pretrained__<YYYYMMDD>)
  --checkpoint PATH        Baseline checkpoint (default: STAGE2_BASELINE_CKPT)
  --stage2-model-dir DIR   Dir with .hydra/config.yaml (default: STAGE2_MODEL_DIR)
  --stage2-repo PATH       Checkout with baseline sample.py (default: STAGE2_REPO_ROOT)
  --cuda-device N          CUDA_VISIBLE_DEVICES (default: 1 or current env).
                           Also exported for --only eval (PMPNN/ESMFold).
                           On ripper use GPU 2 or 3 (0–1 prohibited).
  --batch-size N           sampler.batch_size (default: 1)
  --num-samples-per-task N Override num_samples in every task (smoke / debug)
  --write-multimotif-alignment-pdbs
                           Write overlay PDBs under analysis_out/multimotif_alignments/

Path overrides:
  --out-dir DIR            Sampling run directory
  --analysis-out DIR       Analysis run directory

Stage control:
  --only STAGE             all | sample | inject | eval
  --dry-run                Print resolved paths/commands and exit

Notes:
  - Sampling runs in zenprotein-main with ordinary contig fixation.
  - motif_groups in the tasks YAML is annotation-only (no multi_motif_se3).
  - New main builds annotate samples_metadata at write time; inject is idempotent
    backfill for older runs / safety net.
EOF
}

config_slug_from_path() {
    local config_path="$1"
    local basename
    basename="$(basename "${config_path}")"
    basename="${basename#config_}"
    basename="${basename%.yaml}"
    basename="${basename%.yml}"
    echo "${basename}"
}

_activate_conda() {
    local env_name="$1"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${env_name}"
}

resolve_default_paths() {
    local config_slug="$1"
    local date_stamp
    date_stamp="$(date +%Y%m%d)"
    if [ -z "${RUN_NAME}" ]; then
        RUN_NAME="original_proteinzen__${config_slug}__pretrained__${date_stamp}"
    fi
    if [ -z "${OUT_DIR}" ]; then
        OUT_DIR="${REPO_ROOT}/article_results/sampling_runs/${RUN_NAME}"
    fi
    if [ -z "${ANALYSIS_OUT}" ]; then
        ANALYSIS_OUT="${REPO_ROOT}/article_results/analysis_runs/${RUN_NAME}"
    fi
}

_hydra_quote() {
    python -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"
}

_require_stage2_model_dir() {
    if [ -z "${STAGE2_MODEL}" ]; then
        echo "STAGE2_MODEL_DIR is empty. Set --stage2-model-dir or STAGE2_MODEL_DIR." >&2
        exit 1
    fi
    if [ ! -f "${STAGE2_MODEL}/.hydra/config.yaml" ]; then
        echo "Missing ${STAGE2_MODEL}/.hydra/config.yaml" >&2
        echo "Point --stage2-model-dir at configs/sampling/pretrained/motif_scaffolding_model." >&2
        exit 1
    fi
}

_run_sample() {
    echo "== Baseline sample (proteinzen-main) =="
    echo "  out_dir:      ${OUT_DIR}"
    echo "  tasks_yaml:   ${TASKS_YAML}"
    echo "  checkpoint:   ${CHECKPOINT}"
    echo "  model_dir:    ${STAGE2_MODEL}"
    echo "  stage2_repo:  ${STAGE2_REPO}"
    if [ "${DRY_RUN}" -eq 1 ]; then
        return 0
    fi
    if [ ! -f "${CHECKPOINT}" ]; then
        echo "Missing baseline checkpoint: ${CHECKPOINT}" >&2
        exit 1
    fi
    if [ ! -f "${STAGE2_REPO}/sample.py" ]; then
        echo "Missing ${STAGE2_REPO}/sample.py (set --stage2-repo or STAGE2_REPO_ROOT)." >&2
        exit 1
    fi
    _require_stage2_model_dir
    mkdir -p "${OUT_DIR}"
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    _activate_conda "${ENV_NAME}"
    local sample_yaml="${TASKS_YAML}"
    if [ -n "${NUM_SAMPLES_PER_TASK}" ]; then
        sample_yaml="${OUT_DIR}/tasks_num_samples_override.yaml"
        PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python - <<PY
from pathlib import Path
from nomadflow.benchmark.eval_layout import rewrite_tasks_num_samples
rewrite_tasks_num_samples(
    Path("${TASKS_YAML}"),
    Path("${sample_yaml}"),
    int("${NUM_SAMPLES_PER_TASK}"),
)
print("${sample_yaml}")
PY
    fi
    (
        cd "${STAGE2_REPO}"
        # Prefer this worktree over the editable zenprotein2 install.
        PYTHONPATH="${STAGE2_REPO}" python sample.py \
            model_dir="$(_hydra_quote "${STAGE2_MODEL}")" \
            checkpoint_path="$(_hydra_quote "${CHECKPOINT}")" \
            out_dir="${OUT_DIR}" \
            sampler.tasks_yaml="${sample_yaml}" \
            sampler.batch_size="${BATCH_SIZE}" \
            save_traj=false
    )
}

_run_inject() {
    echo "== Inject motif group metadata =="
    echo "  sampling_run: ${OUT_DIR}"
    echo "  tasks_yaml:   ${TASKS_YAML}"
    if [ "${DRY_RUN}" -eq 1 ]; then
        return 0
    fi
    if [ ! -f "${OUT_DIR}/samples_metadata.json" ]; then
        echo "Missing ${OUT_DIR}/samples_metadata.json. Run --only sample first." >&2
        exit 1
    fi
    _activate_conda "${ENV_NAME}"
    python "${REPO_ROOT}/scripts/analysis/inject_motif_group_metadata.py" \
        --sampling-run "${OUT_DIR}" \
        --tasks-yaml "${TASKS_YAML}" \
        --repo-root "${REPO_ROOT}"
}

_ensure_samples_eval() {
    local sampling_run="$1"
    echo "  ensuring samples_eval under ${sampling_run}"
    if [ "${DRY_RUN}" -eq 1 ]; then
        return 0
    fi
    _activate_conda "${ENV_NAME}"
    python "${REPO_ROOT}/scripts/analysis/prepare_eval_pdbs_from_samples.py" \
        --sampling-run "${sampling_run}"
}

_run_eval() {
    echo "== Eval pipeline =="
    echo "  sampling_run: ${OUT_DIR}"
    echo "  analysis_out: ${ANALYSIS_OUT}"
    if [ "${DRY_RUN}" -eq 1 ]; then
        return 0
    fi
    if [ "${WRITE_MULTIMOTIF_ALIGNMENT_PDBS}" -eq 1 ]; then
        export WRITE_MULTIMOTIF_ALIGNMENT_PDBS=1
    fi
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    _ensure_samples_eval "${OUT_DIR}"
    bash "${REPO_ROOT}/scripts/submit_scripts/eval/submit_article_motif_benchmark.sh" \
        "${OUT_DIR}" \
        "${ANALYSIS_OUT}"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --config|--tasks-yaml)
            TASKS_YAML="$2"
            shift 2
            ;;
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --stage2-model-dir)
            STAGE2_MODEL="$2"
            shift 2
            ;;
        --stage2-repo)
            STAGE2_REPO="$2"
            shift 2
            ;;
        --cuda-device)
            CUDA_DEVICE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num-samples-per-task)
            NUM_SAMPLES_PER_TASK="$2"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --analysis-out)
            ANALYSIS_OUT="$2"
            shift 2
            ;;
        --only)
            ONLY="$2"
            shift 2
            ;;
        --write-multimotif-alignment-pdbs)
            WRITE_MULTIMOTIF_ALIGNMENT_PDBS=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ "${TASKS_YAML}" != /* ]]; then
    TASKS_YAML="${REPO_ROOT}/${TASKS_YAML}"
fi
if [ ! -f "${TASKS_YAML}" ]; then
    echo "Tasks yaml not found: ${TASKS_YAML}" >&2
    exit 1
fi

CONFIG_SLUG="$(config_slug_from_path "${TASKS_YAML}")"
resolve_default_paths "${CONFIG_SLUG}"

echo "Resolved baseline pipeline paths:"
echo "  config:       ${TASKS_YAML}"
echo "  config_slug:  ${CONFIG_SLUG}"
echo "  run_name:     ${RUN_NAME}"
echo "  out_dir:      ${OUT_DIR}"
echo "  analysis_out: ${ANALYSIS_OUT}"
echo "  stage2_repo:  ${STAGE2_REPO}"
echo "  checkpoint:   ${CHECKPOINT}"
echo "  only:         ${ONLY}"

case "${ONLY}" in
    all)
        _run_sample
        _run_inject
        _run_eval
        ;;
    sample)
        _run_sample
        ;;
    inject)
        _run_inject
        ;;
    eval)
        _run_eval
        ;;
    *)
        echo "Unknown --only value: ${ONLY}" >&2
        usage >&2
        exit 1
        ;;
esac

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "Dry run complete."
else
    echo "Baseline pipeline stage '${ONLY}' finished."
fi
