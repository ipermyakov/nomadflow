#!/usr/bin/env bash
# Voronoi mix long train (crop224, 4×GPU, eff=24, step LR decay).
# Env: MLFLOW_RUN_NAME USE_TASKSET=0 WARM_START EXTRA_HYDRA_ARGS
#
# Submit from the repository root. conda activate may be broken if env shebangs
# still point at an old prefix; use PATH to the env bin instead.
#
#   mkdir -p "$PWD/slurm_logs"
#   sbatch -p gpu --gres=gpu:4 -N 1 -c 16 -t 3-00:00:00 \
#     -J voronoi_c224_d4a3 \
#     -o "$PWD/slurm_logs/slurm-%j.out" -e "$PWD/slurm_logs/slurm-%j.err" \
#     --wrap "bash -lc '
#   unset TMPDIR; export USE_TASKSET=0 MLFLOW_ALLOW_FILE_STORE=true
#   export MLFLOW_RUN_NAME=voronoi_crop224_tlock05_d4a3_lr5e4
#   export PATH=\$HOME/miniconda3/envs/nomadflow/bin:\$PATH
#   module load cuda/12.8
#   cd \$REPO_ROOT && source env_vars.sh
#   bash scripts/train_voronoi.sh
#   '"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
[[ -f "${REPO_ROOT}/env_vars.sh" ]] && source "${REPO_ROOT}/env_vars.sh"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export REPO_ROOT

TASKSET_PREFIX=()
if [[ "${USE_TASKSET:-1}" == "1" ]]; then
  TASKSET_PREFIX=(taskset -c "${TASKSET_CPUS:-0-7}")
fi

# Hydra parses unquoted '=' in ckpt filenames as override syntax.
WARM_START="${WARM_START:-${REPO_ROOT}/weights/motif_scaffolding_model/lightning_logs/version_0/checkpoints/epoch-3278-step-780000.ckpt}"
WARM_START_QUOTED="$(python -c 'import json,sys; print(json.dumps(sys.argv[1]))' "${WARM_START}")"

export MLFLOW_ALLOW_FILE_STORE="${MLFLOW_ALLOW_FILE_STORE:-true}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-${REPO_ROOT}/mlruns}"
MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT:-nomadflow_phase1}"
MLFLOW_OVERRIDES=(
  experiment.mlflow.enabled=true
  "experiment.mlflow.tracking_uri=${MLFLOW_TRACKING_URI}"
  "experiment.mlflow.experiment_name=${MLFLOW_EXPERIMENT}"
  "+experiment.mlflow.tags.phase=pretrain_voronoi"
)
if [[ -n "${MLFLOW_RUN_NAME:-}" ]]; then
  MLFLOW_OVERRIDES+=("experiment.mlflow.run_name=${MLFLOW_RUN_NAME}")
fi

# shellcheck disable=SC2206
EXTRA_HYDRA=( ${EXTRA_HYDRA_ARGS:-} )

echo "=== train_voronoi host=$(hostname) bs=2 accum=3 devices=4 eff=24 lr=5e-4 steps=20000 ==="

"${TASKSET_PREFIX[@]}" python train.py \
  domain=protein \
  paradigm=multiframefm \
  datamodule.batch_size=2 \
  lmodule.strict_weight_loading=false \
  lmodule.log_train_loss_every_step=true \
  lmodule.use_lr_step_decay=true \
  lmodule.lr_step_decay_start=7000 \
  lmodule.lr_step_decay_step=3000 \
  lmodule.lr_step_decay_factor=0.5 \
  "dataset.config=${REPO_ROOT}/configs/train/data/voronoi_crop224_tlock05.yaml" \
  "dataset.val_config=${REPO_ROOT}/configs/val/data/voronoi_crop224_tlock05.yaml" \
  experiment.lightning.devices=4 \
  experiment.lightning.accelerator=gpu \
  experiment.lightning.strategy=ddp_find_unused_parameters_true \
  experiment.lightning.accumulate_grad_batches=3 \
  experiment.lightning.check_val_every_n_epoch=null \
  experiment.lightning.val_check_interval=600 \
  experiment.lightning.num_sanity_val_steps=0 \
  experiment.lightning.max_steps=20000 \
  experiment.checkpointer.train_time_interval=null \
  experiment.checkpointer.every_n_train_steps=1000 \
  experiment.checkpointer.save_top_k=3 \
  experiment.checkpointer.monitor=val/scaled_motif_intergroup_fafe \
  experiment.checkpointer.mode=min \
  experiment.optim.lr=5e-4 \
  "experiment.warm_start=${WARM_START_QUOTED}" \
  experiment.warm_start_weights_only=true \
  "hydra.run.dir=${REPO_ROOT}/outputs/pretrain/voronoi_crop224_tlock05_d4a3" \
  "${MLFLOW_OVERRIDES[@]}" \
  "${EXTRA_HYDRA[@]}"
