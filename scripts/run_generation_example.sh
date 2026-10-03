#!/usr/bin/env bash
# Example ProteinZen generation run (unconditional sampling).
# Before running:
#   1) conda activate <your env> and, from the repo root: bash install_pt26.sh
#      (installs PyTorch, PyG, and the nomadflow package).
#   2) Download the weights and unpack them into ${REPO_ROOT}/weights/unconditional_model
#      (see README → Pretrained weights)
#   3) If needed, fix REPO_ROOT in env_vars.sh or pass MODEL_DIR below.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export GEOMSTATS_BACKEND="${GEOMSTATS_BACKEND:-pytorch}"

if [[ -f "${REPO_ROOT}/env_vars.sh" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/env_vars.sh"
fi

# env_vars.sh may set a different REPO_ROOT; this script needs the directory that contains the repository
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Override if needed:
# Setting CHECKPOINT_PATH is enough — the directory with `.hydra/config.yaml` is found by walking up from the checkpoint.
# MODEL_DIR is needed only when CHECKPOINT_PATH is empty (then the checkpoint is taken from lightning_logs).
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/weights/unconditional_model}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/sampling_demo_output}"
TASKS_YAML="${TASKS_YAML:-${REPO_ROOT}/configs/sampling/unconditional_sampling/demo_quick.yaml}"
BATCH_SIZE="${BATCH_SIZE:-2}"
CHECKPOINT_IDX="${CHECKPOINT_IDX:--1}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
# SAVE_TRAJ=true — write multi-model PDB trajectories to ${OUT_DIR}/traj/ (noticeably slower)
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
    echo "Set MODEL_DIR (a run with lightning_logs) or set CHECKPOINT_PATH to a .ckpt."
    exit 1
  fi
fi

if [[ ! -f "${TASKS_YAML}" ]]; then
  echo "Tasks YAML not found: ${TASKS_YAML}"
  exit 1
fi

echo "REPO_ROOT=${REPO_ROOT}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "sample.py model_dir=${SAMPLING_MODEL_DIR:-<empty, config from checkpoint path>}"
echo "OUT_DIR=${OUT_DIR}"
echo "TASKS_YAML=${TASKS_YAML}"
echo "CHECKPOINT_PATH=${CHECKPOINT_PATH:-<none>}"
echo "SAVE_TRAJ=${SAVE_TRAJ} (Hydra save_traj=${TRAJ_HYDRA})"

if ! python -c "import lightning" 2>/dev/null; then
  echo "The active Python has no lightning package (often \`python\` is not from your conda env)."
  echo "Check: which python"
  echo "Install into the same interpreter: python -m pip install -e ."
  echo "or at minimum: python -m pip install lightning torchmetrics hydra-core hydra-zen omegaconf pandas"
  exit 1
fi

if ! python -c "import torch_geometric" 2>/dev/null; then
  echo "torch_geometric is missing (required for rigid_utils / graph operations)."
  echo "From the repo root, in the same env: bash install_pt26.sh"
  exit 1
fi

python sample.py \
  model_dir="${SAMPLING_MODEL_DIR}" \
  out_dir="${OUT_DIR}" \
  checkpoint_path="${CHECKPOINT_PATH}" \
  checkpoint_idx="${CHECKPOINT_IDX}" \
  sampler.tasks_yaml="${TASKS_YAML}" \
  sampler.batch_size="${BATCH_SIZE}" \
  "save_traj=${TRAJ_HYDRA}"

echo "Done. PDB: ${OUT_DIR}/samples/"
if [[ "${TRAJ_HYDRA}" == true ]]; then
  echo "Trajectories (clean / noisy): ${OUT_DIR}/traj/"
fi
