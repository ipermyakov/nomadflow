#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
[[ -f "${REPO_ROOT}/env_vars.sh" ]] && source "${REPO_ROOT}/env_vars.sh"
# Re-resolve in case env_vars.sh overwrote machine-specific paths.
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export REPO_ROOT

# Legacy stash flow used:
#   configs/scripts/dump_training_pdbs/framediff_crop256.yaml
# together with scripts/dump_training_trajectory_to_pdb.py.
#
# That path no longer exists in HEAD, so this script maps the same intent onto the
# current train-corruption visualization entrypoint.
#
# Default current equivalent:
#   DATASET_CONFIG=configs/train/data/framediff_crop256_mix.yaml
#
# Example:
#   OUT_DIR=/tmp/train_corruption_viz \
#   SAMPLES_PER_TASK=2 \
#   NUM_FRAMES=40 \
#   ./scripts/visualize_train_corruption_dump_training_pdbs_framediff_crop256.sh

DATASET_CONFIG="${DATASET_CONFIG:-configs/train/data/spatial_distal_pretr.yaml}"
OUT_DIR="${OUT_DIR:-train_corruption_viz}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-10}"
BASE_SEED="${BASE_SEED:-0}"
NUM_FRAMES="${NUM_FRAMES:-40}"
TASK_FILTER="${TASK_FILTER:-}"
OUTPUT_MOTIF_CHAINS="${OUTPUT_MOTIF_CHAINS:-1}"
ADD_INTRA_RIGID_CONECTS="${ADD_INTRA_RIGID_CONECTS:-1}"

CMD=(
  conda run -n "${ENV_NAME:-nomadflow}" python "scripts/visualize_train_corruption.py"
  "dataset_config=${DATASET_CONFIG}"
  "out_dir=${OUT_DIR}"
  "samples_per_task=${SAMPLES_PER_TASK}"
  "base_seed=${BASE_SEED}"
  "num_frames=${NUM_FRAMES}"
)

if [[ -n "${TASK_FILTER}" ]]; then
  IFS=',' read -r -a TASK_FILTER_VALUES <<< "${TASK_FILTER}"
  CLEAN_TASK_FILTER_VALUES=()
  for task_name in "${TASK_FILTER_VALUES[@]}"; do
    task_name="${task_name#"${task_name%%[![:space:]]*}"}"
    task_name="${task_name%"${task_name##*[![:space:]]}"}"
    [[ -n "${task_name}" ]] && CLEAN_TASK_FILTER_VALUES+=("${task_name}")
  done
  if [[ ${#CLEAN_TASK_FILTER_VALUES[@]} -gt 0 ]]; then
    task_filter_override="task_filter=[${CLEAN_TASK_FILTER_VALUES[0]}"
    for ((idx = 1; idx < ${#CLEAN_TASK_FILTER_VALUES[@]}; idx++)); do
      task_filter_override+=",${CLEAN_TASK_FILTER_VALUES[idx]}"
    done
    task_filter_override+="]"
    CMD+=("${task_filter_override}")
  fi
fi

if [[ "${OUTPUT_MOTIF_CHAINS}" == "1" ]]; then
  CMD+=("output_motif_chains=true")
else
  CMD+=("output_motif_chains=false")
fi

if [[ "${ADD_INTRA_RIGID_CONECTS}" == "1" ]]; then
  CMD+=("add_intra_rigid_conects=true")
else
  CMD+=("add_intra_rigid_conects=false")
fi

"${CMD[@]}"
