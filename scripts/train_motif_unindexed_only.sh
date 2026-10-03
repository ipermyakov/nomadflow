#!/usr/bin/env bash
# Like scripts/train_motif_unindexed_only.sh, but with metric logging to MLflow.
# Requires the mlflow package (included in the base `pip install -e .` install).
#
# Variables:
#   TASKSET_CPUS — cores for taskset -c (default 0-7). Do not confuse with TASKSET=.
#   USE_TASKSET=0 — disable taskset.
#   MLFLOW_TRACKING_URI — MLflow tracking backend URI (default: <REPO_ROOT>/mlruns).
#   MLFLOW_ALLOW_FILE_STORE — allow the file-store backend (default: true).
#   MLFLOW_EXPERIMENT — MLflow experiment name (default: nomadflow_motif_unindexed).
#   MLFLOW_RUN_NAME — explicit run name; if unset, basename of hydra.run.dir is used.
#
# Everything else matches train_motif_unindexed_only.sh (data, WARM_START, taskset, BATCH_SIZE, …).
#
# View: mlflow ui --backend-store-uri "$MLFLOW_TRACKING_URI"
#
# Dense per-step plots: Lightning logs to loggers every 50 steps by default
# (experiment.lightning.log_every_n_steps). For MLflow, set this to 1 to log per-step loss.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
[[ -f "${REPO_ROOT}/env_vars.sh" ]] && source "${REPO_ROOT}/env_vars.sh"

TASKSET_PREFIX=()
if [[ "${USE_TASKSET:-1}" == "1" ]]; then
  TASKSET_PREFIX=(taskset -c "${TASKSET_CPUS:-0-7}")
fi

if [[ -n "${WARM_START:-}" ]]; then
  WARM_START_ARG=(experiment.warm_start="${WARM_START}")
else
  WARM_START_ARG=(experiment.warm_start=null)
fi

MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-${REPO_ROOT}/mlruns}"
MLFLOW_ALLOW_FILE_STORE="${MLFLOW_ALLOW_FILE_STORE:-true}"
export MLFLOW_ALLOW_FILE_STORE
MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT:-nomadflow_motif_unindexed}"
MLFLOW_OVERRIDES=(
  experiment.mlflow.enabled=true
  "experiment.mlflow.tracking_uri=${MLFLOW_TRACKING_URI}"
  "experiment.mlflow.experiment_name=${MLFLOW_EXPERIMENT}"
  "+experiment.mlflow.tags.phase=finetune_motif_unindexed_only"
)
if [[ -n "${MLFLOW_RUN_NAME:-}" ]]; then
  MLFLOW_OVERRIDES+=("experiment.mlflow.run_name=${MLFLOW_RUN_NAME}")
fi

# Checkpoint selection:
# - Keep best K checkpoints by monitored metric (default: train/loss_step, minimize).
# - Change via env, e.g. CKPT_SAVE_TOP_K=3 CKPT_MONITOR=grad_norm CKPT_MODE=min.
CKPT_SAVE_TOP_K="${CKPT_SAVE_TOP_K:-3}"
CKPT_MONITOR="${CKPT_MONITOR:-train/loss_step}"
CKPT_MODE="${CKPT_MODE:-min}"

DATASET_CONFIG="${DATASET_CONFIG:-${REPO_ROOT}/configs/train/data/framediff_motif_crop320_unindexed_only.yaml}"

"${TASKSET_PREFIX[@]}" python train.py \
    domain=protein \
    paradigm=multiframefm \
    datamodule.batch_size=2  \
    datamodule.num_workers=8 \
    model.c_s=384 \
    model.c_cond=384 \
    model.c_frame=128 \
    model.c_framepair=32 \
    model.z_broadcast=true \
    model.rigid_transformer_num_blocks=1 \
    model.rigid_transformer_rigid_updates=true \
    model.use_embedder_sc_rigid_transformer=true \
    model.use_ipa_gating=true \
    model.use_qk_norm=true \
    model.use_amp=true \
    model.rot_preconditioning=true \
    model.num_blocks=5 \
    lmodule.use_ema=true \
    corrupter.use_stochastic_centering=true \
    corrupter.center_on_motif=true \
    corrupter.sig_perturb=4 \
    dataset.config="${DATASET_CONFIG}" \
    experiment.lightning.devices=1 \
    experiment.lightning.strategy=ddp_find_unused_parameters_true \
    experiment.checkpointer.train_time_interval=null \
    experiment.checkpointer.every_n_train_steps=2000 \
    experiment.checkpointer.save_top_k="${CKPT_SAVE_TOP_K}" \
    experiment.checkpointer.monitor="${CKPT_MONITOR}" \
    experiment.checkpointer.mode="${CKPT_MODE}" \
    experiment.lightning.gradient_clip_val=5.0 \
    experiment.lightning.accumulate_grad_batches=2 \
    experiment.lightning.log_every_n_steps=100 \
    lmodule.log_train_loss_every_step=true \
    hydra.run.dir="${REPO_ROOT}/outputs/finetune/motif_unindexed_only" \
    "${WARM_START_ARG[@]}" \
    "${MLFLOW_OVERRIDES[@]}" \
    lmodule.lr_step_decay_start=720000 \
    lmodule.lr_step_decay_step=20000 \
    lmodule.lr_step_decay_factor=0.95 \
    +profile_self_condition=true
