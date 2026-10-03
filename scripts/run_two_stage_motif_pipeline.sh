#!/usr/bin/env bash
# Two-stage motif pipeline: Stage-1 layout (MLflow) -> Stage-2 baseline resampling -> eval.
#
# Both sampling runs land under article_results/sampling_runs/ so Stage-1 can be
# benchmarked on its own without Stage-2.
#
# Examples:
#   ./scripts/run_two_stage_motif_pipeline.sh \
#     --run-id <stage1_mlflow_run_id> \
#     --config configs/sampling/protpardelle_motif_scaffolding/config_multi_motif_se3_catalytic_tlock_0.5.yaml
#
#   ./scripts/run_two_stage_motif_pipeline.sh \
#     --run-id "${STAGE1_MLFLOW_RUN_ID}" \
#     --config configs/sampling/protpardelle_motif_scaffolding/config_multi_motif_se3_catalytic_tlock_0.5.yaml \
#     --run-name original_proteinzen_from_layout \
#     --only stage2
#
# Auto run-name from run-id + config:
#   sampling_runs/<run_id_prefix>__<config_slug>_stage_1
#   sampling_runs/<run_id_prefix>__<config_slug>_stage_2
#   analysis_runs/<run_id_prefix>__<config_slug>
#
#   ./scripts/run_two_stage_motif_pipeline.sh \
#     --config configs/sampling/protpardelle_motif_scaffolding/config_multi_motif_se3_catalytic_tlock_0.5.yaml \
#     --only eval --eval-target stage1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../env_vars.sh"

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONPATH

DEFAULT_CONFIG="${REPO_ROOT}/configs/sampling/protpardelle_motif_scaffolding/config_multi_motif_se3_catalytic_tlock_0.5.yaml"

RUN_ID="${STAGE1_MLFLOW_RUN_ID:-}"
TASKS_YAML="${DEFAULT_CONFIG}"
RUN_NAME=""
STAGE1_CHECKPOINT="best"
STAGE2_CHECKPOINT="${STAGE2_BASELINE_CKPT}"
STAGE2_MODEL_DIR="${STAGE2_MODEL_DIR}"
STAGE2_REPO="${STAGE2_REPO_ROOT}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-1}"
STAGE2_NUM_SAMPLES=1
NUM_SAMPLES_PER_TASK=""
ONLY="all"
EVAL_TARGET="stage2"
DRY_RUN=0
INCLUDE_MOTIF_GROUPS=0
WRITE_MULTIMOTIF_ALIGNMENT_PDBS=0

STAGE1_OUT=""
STAGE2_OUT=""
ANALYSIS_OUT=""

TMPDIR="${TMPDIR:-${HOME}/tmp}"
export TMPDIR TEMP="${TMPDIR}" TMP="${TMPDIR}"

usage() {
    cat <<'EOF'
Usage: run_two_stage_motif_pipeline.sh [options]

Required for stage1/tasks/stage2/all:
  --run-id ID              Stage-1 MLflow run id (default: STAGE1_MLFLOW_RUN_ID from env_vars.sh)

Main options:
  --config PATH            Benchmark tasks yaml (alias: --tasks-yaml)
  --run-name NAME          Pipeline run slug (default: <run_id_prefix>__<config_slug>)
  --checkpoint NAME        Stage-1 checkpoint selector (default: best)
  --stage2-checkpoint PATH Stage-2 baseline checkpoint (default: STAGE2_BASELINE_CKPT)
  --stage2-model-dir DIR   Dir with .hydra/config.yaml for Stage-2 (default: STAGE2_MODEL_DIR)
  --stage2-repo PATH       Checkout with baseline sample.py (default: STAGE2_REPO_ROOT)
  --cuda-device N          CUDA_VISIBLE_DEVICES (default: 1 or current env).
                           Also exported for --only eval (PMPNN/ESMFold).
                           On ripper use GPU 2 or 3 (0–1 prohibited).
  --stage2-num-samples N   Samples per Stage-2 task (default: 1)
  --num-samples-per-task N Override Stage-1 num_samples in every task (smoke / debug)
  --include-motif-groups   Forward --include-motif-groups to task builder
  --write-multimotif-alignment-pdbs
                           Write overlay PDBs (ESMFold + aligned native motifs) under
                           analysis_out/multimotif_alignments/

Path overrides (skip auto naming when set):
  --stage1-out DIR
  --stage2-out DIR
  --analysis-out DIR

Stage control:
  --only STAGE             all | stage1 | tasks | stage2 | eval
  --eval-target TARGET     stage1 | stage2 (default: stage2)
  --dry-run                Print resolved paths/commands and exit

Default dirs (run-name = explicit --run-name or <run_id_prefix>__<config_slug>):
  article_results/sampling_runs/<run-name>_stage_1/
  article_results/sampling_runs/<run-name>_stage_2/
  article_results/analysis_runs/<run-name>/
EOF
}

config_slug_from_path() {
    local config_path="$1"
    local basename
    basename="$(basename "${config_path}")"
    basename="${basename#config_}"
    basename="${basename%.yaml}"
    basename="${basename%.yml}"
    printf '%s' "${basename}"
}

resolve_run_name() {
    local config_slug="$1"
    if [ -n "${RUN_NAME}" ]; then
        return 0
    fi
    if [ -z "${RUN_ID}" ]; then
        echo "Provide --run-name or --run-id to derive the pipeline run name." >&2
        exit 1
    fi
    local run_id_prefix="${RUN_ID:0:8}"
    RUN_NAME="${run_id_prefix}__${config_slug}"
}

resolve_default_paths() {
    local config_slug="${1}"
    resolve_run_name "${config_slug}"

    if [ -z "${STAGE1_OUT}" ]; then
        STAGE1_OUT="${REPO_ROOT}/article_results/sampling_runs/${RUN_NAME}_stage_1"
    fi

    if [ -z "${STAGE2_OUT}" ]; then
        STAGE2_OUT="${REPO_ROOT}/article_results/sampling_runs/${RUN_NAME}_stage_2"
    fi

    if [ -z "${ANALYSIS_OUT}" ]; then
        ANALYSIS_OUT="${REPO_ROOT}/article_results/analysis_runs/${RUN_NAME}"
    fi

    STAGE2_TASKS_YAML="${STAGE2_OUT}/tasks_from_layouts.yaml"
    STAGE1_LAYOUT_MANIFEST="${STAGE1_OUT}/layout_manifest.jsonl"
}

_hydra_quote() {
    python -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"
}

_require_stage2_model_dir() {
    if [ -z "${STAGE2_MODEL_DIR}" ]; then
        echo "STAGE2_MODEL_DIR is empty. Set --stage2-model-dir or STAGE2_MODEL_DIR." >&2
        exit 1
    fi
    if [ ! -f "${STAGE2_MODEL_DIR}/.hydra/config.yaml" ]; then
        echo "Missing ${STAGE2_MODEL_DIR}/.hydra/config.yaml" >&2
        echo "The extracted motif_scaffolding_model weights have no sibling Hydra config;" >&2
        echo "point STAGE2_MODEL_DIR at configs/sampling/pretrained/motif_scaffolding_model." >&2
        exit 1
    fi
}

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

_run_stage1() {
    echo "== Stage-1: layout sampling =="
    echo "  out_dir: ${STAGE1_OUT}"
    if [ "${DRY_RUN}" -eq 1 ]; then
        return 0
    fi
    if [ -z "${RUN_ID}" ]; then
        echo "Stage-1 requires --run-id or STAGE1_MLFLOW_RUN_ID." >&2
        exit 1
    fi
    _activate_conda "${ENV_NAME}"
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    local extra_args=()
    if [ -n "${NUM_SAMPLES_PER_TASK}" ]; then
        extra_args+=(--num-samples-per-task "${NUM_SAMPLES_PER_TASK}")
    fi
    python "${REPO_ROOT}/scripts/sample_from_mlflow_run.py" \
        --run-id "${RUN_ID}" \
        --tasks-yaml "${TASKS_YAML}" \
        --out-dir "${STAGE1_OUT}" \
        --checkpoint "${STAGE1_CHECKPOINT}" \
        --sample-arg sampler.batch_size=6 \
        --sample-arg save_traj=false \
        --sample-arg export_layout_manifest=true \
        --sample-arg output_motif_chains=true \
        "${extra_args[@]}"
}

_run_tasks() {
    echo "== Stage-2 tasks build =="
    echo "  layout_manifest: ${STAGE1_LAYOUT_MANIFEST}"
    echo "  tasks_yaml:      ${STAGE2_TASKS_YAML}"
    if [ "${DRY_RUN}" -eq 1 ]; then
        return 0
    fi
    if [ ! -f "${STAGE1_LAYOUT_MANIFEST}" ]; then
        echo "Missing ${STAGE1_LAYOUT_MANIFEST}. Run --only stage1 first." >&2
        exit 1
    fi
    _activate_conda "${ENV_NAME}"
    local motif_groups_args=()
    if [ "${INCLUDE_MOTIF_GROUPS}" -eq 1 ]; then
        motif_groups_args+=(--include-motif-groups)
    fi
    python "${REPO_ROOT}/scripts/analysis/build_stage2_tasks_from_layout_manifest.py" \
        --layout-manifest "${STAGE1_LAYOUT_MANIFEST}" \
        --output-yaml "${STAGE2_TASKS_YAML}" \
        --stage2-num-samples "${STAGE2_NUM_SAMPLES}" \
        --total-length sample_length \
        --motif-source viz_chains \
        "${motif_groups_args[@]}"
}

_run_stage2() {
    echo "== Stage-2: baseline resampling =="
    echo "  out_dir:      ${STAGE2_OUT}"
    echo "  tasks_yaml:   ${STAGE2_TASKS_YAML}"
    echo "  checkpoint:   ${STAGE2_CHECKPOINT}"
    echo "  model_dir:    ${STAGE2_MODEL_DIR}"
    echo "  stage2_repo:  ${STAGE2_REPO}"
    if [ "${DRY_RUN}" -eq 1 ]; then
        return 0
    fi
    if [ ! -f "${STAGE2_TASKS_YAML}" ]; then
        echo "Missing ${STAGE2_TASKS_YAML}. Run --only tasks first." >&2
        exit 1
    fi
    if [ ! -f "${STAGE2_CHECKPOINT}" ]; then
        echo "Missing Stage-2 checkpoint: ${STAGE2_CHECKPOINT}" >&2
        exit 1
    fi
    if [ ! -f "${STAGE2_REPO}/sample.py" ]; then
        echo "Missing ${STAGE2_REPO}/sample.py (set --stage2-repo or STAGE2_REPO_ROOT)." >&2
        exit 1
    fi
    _require_stage2_model_dir
    _activate_conda "${ENV_NAME}"
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    (
        cd "${STAGE2_REPO}"
        # Prefer this worktree over the editable zenprotein2 install.
        PYTHONPATH="${STAGE2_REPO}" python sample.py \
            model_dir="$(_hydra_quote "${STAGE2_MODEL_DIR}")" \
            checkpoint_path="$(_hydra_quote "${STAGE2_CHECKPOINT}")" \
            out_dir="${STAGE2_OUT}" \
            sampler.tasks_yaml="${STAGE2_TASKS_YAML}" \
            sampler.batch_size=6 \
            save_traj=false
    )
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
    local sampling_run
    case "${EVAL_TARGET}" in
        stage1) sampling_run="${STAGE1_OUT}" ;;
        stage2) sampling_run="${STAGE2_OUT}" ;;
        *)
            echo "Unknown --eval-target: ${EVAL_TARGET}" >&2
            exit 1
            ;;
    esac

    echo "== Eval pipeline =="
    echo "  sampling_run: ${sampling_run}"
    echo "  analysis_out: ${ANALYSIS_OUT}"
    if [ "${DRY_RUN}" -eq 1 ]; then
        return 0
    fi
    if [ "${WRITE_MULTIMOTIF_ALIGNMENT_PDBS}" -eq 1 ]; then
        export WRITE_MULTIMOTIF_ALIGNMENT_PDBS=1
    fi
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    _ensure_samples_eval "${sampling_run}"
    bash "${REPO_ROOT}/scripts/submit_scripts/eval/submit_article_motif_benchmark.sh" \
        "${sampling_run}" \
        "${ANALYSIS_OUT}"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --run-id)
            RUN_ID="$2"
            shift 2
            ;;
        --config|--tasks-yaml)
            TASKS_YAML="$2"
            shift 2
            ;;
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        --checkpoint)
            STAGE1_CHECKPOINT="$2"
            shift 2
            ;;
        --stage2-checkpoint)
            STAGE2_CHECKPOINT="$2"
            shift 2
            ;;
        --stage2-model-dir)
            STAGE2_MODEL_DIR="$2"
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
        --stage2-num-samples)
            STAGE2_NUM_SAMPLES="$2"
            shift 2
            ;;
        --num-samples-per-task)
            NUM_SAMPLES_PER_TASK="$2"
            shift 2
            ;;
        --stage1-out)
            STAGE1_OUT="$2"
            shift 2
            ;;
        --stage2-out)
            STAGE2_OUT="$2"
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
        --eval-target)
            EVAL_TARGET="$2"
            shift 2
            ;;
        --include-motif-groups)
            INCLUDE_MOTIF_GROUPS=1
            shift
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

echo "Resolved pipeline paths:"
echo "  config:              ${TASKS_YAML}"
echo "  config_slug:         ${CONFIG_SLUG}"
echo "  run_id:              ${RUN_ID:-<unset>}"
echo "  run_name:            ${RUN_NAME}"
echo "  stage1_out:          ${STAGE1_OUT}"
echo "  stage2_out:          ${STAGE2_OUT}"
echo "  stage2_model_dir:    ${STAGE2_MODEL_DIR}"
echo "  stage2_tasks_yaml:   ${STAGE2_TASKS_YAML}"
echo "  analysis_out:        ${ANALYSIS_OUT}"
echo "  only:                ${ONLY}"
echo "  eval_target:         ${EVAL_TARGET}"
echo "  write_alignment_pdbs: ${WRITE_MULTIMOTIF_ALIGNMENT_PDBS}"

case "${ONLY}" in
    all)
        _run_stage1
        _run_tasks
        _run_stage2
        _ensure_samples_eval "${STAGE2_OUT}"
        _run_eval
        ;;
    stage1)
        _run_stage1
        ;;
    tasks)
        _run_tasks
        ;;
    stage2)
        _run_stage2
        _ensure_samples_eval "${STAGE2_OUT}"
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
    echo "Pipeline stage '${ONLY}' finished."
fi
