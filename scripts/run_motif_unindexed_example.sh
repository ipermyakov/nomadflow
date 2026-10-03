#!/usr/bin/env bash
# Example motif scaffolding with an unindexed motif (empty contigs_idx).
# Requires motif_scaffolding_model weights (README → Pretrained weights).
#
# From the repo root:
#   conda activate nomadflow
#   bash scripts/run_motif_unindexed_example.sh
#
# Optional: SAVE_TRAJ=true, OUT_DIR=..., BATCH_SIZE=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export GEOMSTATS_BACKEND="${GEOMSTATS_BACKEND:-pytorch}"

if [[ -f "${REPO_ROOT}/env_vars.sh" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/env_vars.sh"
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/weights/motif_scaffolding_model}"
# OUT_DIR="${OUT_DIR:-${REPO_ROOT}/sampling/motif_unindexed_output_tuned}"
TASKS_YAML="${TASKS_YAML:-${REPO_ROOT}/configs/sampling/motif_scaffolding/demo_motif_unindexed.yaml}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CHECKPOINT_IDX="${CHECKPOINT_IDX:--1}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
SAVE_TRAJ="${SAVE_TRAJ:-true}"
case "${SAVE_TRAJ}" in 1|true|TRUE|yes|YES) TRAJ_HYDRA=true ;; *) TRAJ_HYDRA=false ;; esac

# With an explicit CHECKPOINT_PATH, the model config is searched upward from the .ckpt; model_dir in sample.py can be left empty.
if [[ -n "${CHECKPOINT_PATH}" ]]; then
  SAMPLING_MODEL_DIR=""
else
  SAMPLING_MODEL_DIR="${MODEL_DIR}"
fi

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "model_dir directory not found: ${MODEL_DIR}"
    echo "Set MODEL_DIR (a run with lightning_logs) or CHECKPOINT_PATH to a .ckpt."
    exit 1
  fi
fi

if [[ ! -f "${TASKS_YAML}" ]]; then
  echo "YAML not found: ${TASKS_YAML}"
  exit 1
fi

if ! python -c "import lightning" 2>/dev/null; then
  echo "lightning is missing in the current python; see scripts/run_generation_example.sh"
  exit 1
fi

if ! python -c "import torch_geometric" 2>/dev/null; then
  echo "torch_geometric is missing; from the repo root: bash install_pt26.sh"
  exit 1
fi

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/sampling/motif_unindexed_50m}"

echo "MODEL_DIR=${MODEL_DIR}"
echo "sample.py model_dir=${SAMPLING_MODEL_DIR:-<empty, config from checkpoint path>}"
echo "OUT_DIR=${OUT_DIR}"
echo "TASKS_YAML=${TASKS_YAML}"
echo "CHECKPOINT_PATH=${CHECKPOINT_PATH:-<none>}"
echo "save_traj=${TRAJ_HYDRA}"

python sample.py \
  model_dir="${SAMPLING_MODEL_DIR}" \
  out_dir="${OUT_DIR}" \
  checkpoint_path="${CHECKPOINT_PATH}" \
  checkpoint_idx="${CHECKPOINT_IDX}" \
  sampler.tasks_yaml="${TASKS_YAML}" \
  sampler.batch_size="${BATCH_SIZE}" \
  "save_traj=${TRAJ_HYDRA}" \
  output_motif_chains=true

echo "Done. PDB: ${OUT_DIR}/samples/"
[[ "${TRAJ_HYDRA}" == true ]] && echo "Trajectories: ${OUT_DIR}/traj/"
