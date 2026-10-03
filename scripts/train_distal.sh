#!/usr/bin/env bash
# Like scripts/train_motif_unindexed_only.sh, but with metric logging to MLflow.
# Requires the mlflow package (included in the base `pip install -e .` install).
#
# Variables:
#   TASKSET_CPUS — cores for taskset -c (default 0-7). Do not confuse with TASKSET=.
#   USE_TASKSET=0 — disable taskset.
#   MLFLOW_TRACKING_URI — MLflow tracking backend URI (default: <REPO_ROOT>/mlruns).
#   MLFLOW_ALLOW_FILE_STORE — allow the file-store backend (default: true).
#   MLFLOW_EXPERIMENT — MLflow experiment name (default: nomadflow_phase1).
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
# Force repo root for this script even if env_vars.sh contains stale machine-specific paths.
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export REPO_ROOT

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
MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT:-nomadflow_phase1}"
MLFLOW_OVERRIDES=(
  experiment.mlflow.enabled=true
  "experiment.mlflow.tracking_uri=${MLFLOW_TRACKING_URI}"
  "experiment.mlflow.experiment_name=${MLFLOW_EXPERIMENT}"
  "+experiment.mlflow.tags.phase=pretrain_phase1"
)
if [[ -n "${MLFLOW_RUN_NAME:-}" ]]; then
  MLFLOW_OVERRIDES+=("experiment.mlflow.run_name=${MLFLOW_RUN_NAME}")
fi

# Checkpoint selection:
# - Keep best K checkpoints by monitored metric (default: val/loss when validation is enabled).
# - Change via env, e.g. CKPT_SAVE_TOP_K=3 CKPT_MONITOR=train/loss_step CKPT_MODE=min.
CKPT_SAVE_TOP_K="${CKPT_SAVE_TOP_K:-3}"
CKPT_MONITOR="${CKPT_MONITOR:-val/loss}"
CKPT_MODE="${CKPT_MODE:-min}"

# DATASET_CONFIG="${DATASET_CONFIG:-${REPO_ROOT}/configs/train/data/framediff_crop256_mix.yaml}"
# DATASET_VAL_CONFIG="${DATASET_VAL_CONFIG:-${REPO_ROOT}/configs/val/data/framediff_crop256_mix.yaml}"

# DATASET_CONFIG="${DATASET_CONFIG:-${REPO_ROOT}/configs/train/data/framediff_crop256_mix.yaml}"
# DATASET_VAL_CONFIG="${DATASET_VAL_CONFIG:-${REPO_ROOT}/configs/val/data/framediff_crop256_mix.yaml}"
# DATASET_CONFIG="${DATASET_CONFIG:-${REPO_ROOT}/configs/train/data/spatial_frag.yaml}"
# DATASET_VAL_CONFIG="${DATASET_VAL_CONFIG:-${REPO_ROOT}/configs/val/data/spatial_frag.yaml}"
DATASET_CONFIG="${DATASET_CONFIG:-${REPO_ROOT}/configs/train/data/distal_afdb_512_clusters_crop256.yaml}"
DATASET_VAL_CONFIG="${DATASET_VAL_CONFIG:-${REPO_ROOT}/configs/val/data/distal_afdb_512_clusters_crop256.yaml}"

"${TASKSET_PREFIX[@]}" python train.py \
    domain=protein \
    paradigm=multiframefm \
    datamodule.batch_size=1  \
    datamodule.num_workers=4 \
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
    dataset.val_config="${DATASET_VAL_CONFIG}" \
    experiment.lightning.devices=2 \
    experiment.lightning.strategy=ddp_find_unused_parameters_true \
    experiment.lightning.check_val_every_n_epoch=10 \
    datamodule.val_num_workers=0 \
    experiment.checkpointer.train_time_interval=null \
    experiment.checkpointer.every_n_train_steps=2000 \
    experiment.checkpointer.save_top_k="${CKPT_SAVE_TOP_K}" \
    experiment.checkpointer.monitor="${CKPT_MONITOR}" \
    experiment.checkpointer.mode="${CKPT_MODE}" \
    experiment.lightning.accumulate_grad_batches=12 \
    experiment.optim.lr=5e-4 \
    experiment.lightning.log_every_n_steps=100 \
    experiment.lightning.max_steps=80500 \
    lmodule.log_train_loss_every_step=true \
    hydra.run.dir="${REPO_ROOT}/outputs/pretrain/phase1_0" \
    "${MLFLOW_OVERRIDES[@]}" \
