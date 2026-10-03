#!/usr/bin/env bash
# Single spatial HPO / smoke trial for bs=2 crop240 t_lock=0.15.
#
# Env overrides:
#   BATCH_SIZE=2 ACCUM=5 LR=1e-4 MAX_STEPS=800 WARMUP_STEPS=250
#   USE_COSINE=0|1 DEVICES=2 NUM_WORKERS=8
#   DATASET_CONFIG / DATASET_VAL_CONFIG / MLFLOW_RUN_NAME / HYDRA_RUN_DIR
#   ENABLE_VAL=1  — use DATASET_VAL_CONFIG (default); set 0 to skip val
#   NUM_VAL_POINTS=3  — target val/loss logs per trial (step-based; default)
#   VAL_CHECK_EVERY_N_STEPS — override interval in optimizer steps (default: MAX_STEPS/NUM_VAL_POINTS)
#   CHECK_VAL_EVERY_N_EPOCH — if set, use epoch-based val instead of step-based
#   ENABLE_MLFLOW=1
#   EXTRA_HYDRA_ARGS='lmodule.fafe_loss_weight=0.25'  — optional space-separated Hydra overrides
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
[[ -f "${REPO_ROOT}/env_vars.sh" ]] && source "${REPO_ROOT}/env_vars.sh"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export REPO_ROOT

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-nomadflow}"
module load cuda/12.8

export USE_TASKSET=0
export MLFLOW_ALLOW_FILE_STORE="${MLFLOW_ALLOW_FILE_STORE:-true}"

BATCH_SIZE="${BATCH_SIZE:-2}"
ACCUM="${ACCUM:-5}"
LR="${LR:-1e-4}"
MAX_STEPS="${MAX_STEPS:-800}"
WARMUP_STEPS="${WARMUP_STEPS:-250}"
USE_COSINE="${USE_COSINE:-0}"
DEVICES="${DEVICES:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
ENABLE_VAL="${ENABLE_VAL:-1}"
ENABLE_MLFLOW="${ENABLE_MLFLOW:-1}"
# Default: ≥3 val points via optimizer-step interval (robust to accum / short epoch counts).
# Set CHECK_VAL_EVERY_N_EPOCH to force the old epoch-based schedule instead.
NUM_VAL_POINTS="${NUM_VAL_POINTS:-3}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-}"
# Optional extra Hydra overrides (space-separated), e.g.:
#   EXTRA_HYDRA_ARGS='lmodule.fafe_loss_weight=0.25 lmodule.motif_aggregator_weight=1.0'
EXTRA_HYDRA_ARGS="${EXTRA_HYDRA_ARGS:-}"

DATASET_CONFIG="${DATASET_CONFIG:-${REPO_ROOT}/configs/train/data/spatial_afdb_512_clusters_crop240_finetune_tlock_0.15.yaml}"
DATASET_VAL_CONFIG="${DATASET_VAL_CONFIG:-${REPO_ROOT}/configs/val/data/spatial_afdb_512_clusters_crop240_finetune_tlock_0.15.yaml}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-${REPO_ROOT}/outputs/hpo/spatial_bs${BATCH_SIZE}_c240_t015}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-${REPO_ROOT}/mlruns}"
MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT:-nomadflow_phase1}"

WARM_START="${WARM_START:-${REPO_ROOT}/weights/motif_scaffolding_model/lightning_logs/version_0/checkpoints/epoch-3278-step-780000.ckpt}"
WARM_START_QUOTED="$(python -c 'import json,sys; print(json.dumps(sys.argv[1]))' "${WARM_START}")"

SCHED_ARGS=()
if [[ "${USE_COSINE}" == "1" ]]; then
  SCHED_ARGS=(
    lmodule.use_cosine_lr_sched=true
    "lmodule.cosine_warmup_steps=${WARMUP_STEPS}"
    "lmodule.cosine_total_steps=${MAX_STEPS}"
  )
elif [[ "${WARMUP_STEPS}" != "0" ]]; then
  SCHED_ARGS=(
    lmodule.use_linear_warmup=true
    "lmodule.linear_warmup_steps=${WARMUP_STEPS}"
  )
fi

VAL_ARGS=()
VAL_MODE=""
if [[ "${ENABLE_VAL}" == "1" ]]; then
  if [[ -n "${CHECK_VAL_EVERY_N_EPOCH}" ]]; then
    VAL_MODE="epoch"
    VAL_ARGS=(
      "dataset.val_config=${DATASET_VAL_CONFIG}"
      "experiment.lightning.check_val_every_n_epoch=${CHECK_VAL_EVERY_N_EPOCH}"
      experiment.lightning.num_sanity_val_steps=0
    )
  else
    VAL_MODE="steps"
    # Lightning val_check_interval counts training batches, not optimizer steps.
    # Must set check_val_every_n_epoch=null so the interval is global (not capped
    # by epoch length — short HPO epochs can be << VAL_CHECK_INTERVAL_BATCHES).
    VAL_CHECK_EVERY_N_STEPS="${VAL_CHECK_EVERY_N_STEPS:-$((MAX_STEPS / NUM_VAL_POINTS))}"
    if [[ "${VAL_CHECK_EVERY_N_STEPS}" -lt 1 ]]; then
      VAL_CHECK_EVERY_N_STEPS=1
    fi
    VAL_CHECK_INTERVAL_BATCHES=$((VAL_CHECK_EVERY_N_STEPS * ACCUM))
    if [[ "${VAL_CHECK_INTERVAL_BATCHES}" -lt 1 ]]; then
      VAL_CHECK_INTERVAL_BATCHES=1
    fi
    VAL_ARGS=(
      "dataset.val_config=${DATASET_VAL_CONFIG}"
      "experiment.lightning.check_val_every_n_epoch=null"
      "experiment.lightning.val_check_interval=${VAL_CHECK_INTERVAL_BATCHES}"
      experiment.lightning.num_sanity_val_steps=0
    )
  fi
else
  VAL_MODE="off"
  VAL_ARGS=(
    dataset.val_config=null
    experiment.lightning.check_val_every_n_epoch=100000
    experiment.lightning.num_sanity_val_steps=0
  )
fi

MLFLOW_ARGS=(experiment.mlflow.enabled=false)
if [[ "${ENABLE_MLFLOW}" == "1" ]]; then
  MLFLOW_ARGS=(
    experiment.mlflow.enabled=true
    "experiment.mlflow.tracking_uri=${MLFLOW_TRACKING_URI}"
    "experiment.mlflow.experiment_name=${MLFLOW_EXPERIMENT}"
    "+experiment.mlflow.tags.phase=hpo_spatial"
  )
  if [[ -n "${MLFLOW_RUN_NAME:-}" ]]; then
    MLFLOW_ARGS+=("experiment.mlflow.run_name=${MLFLOW_RUN_NAME}")
  fi
fi

STRATEGY=auto
if [[ "${DEVICES}" -gt 1 ]]; then
  STRATEGY=ddp_find_unused_parameters_true
fi

echo "=== HPO TRIAL host=$(hostname) bs=${BATCH_SIZE} accum=${ACCUM} lr=${LR} steps=${MAX_STEPS} cosine=${USE_COSINE} devices=${DEVICES} enable_val=${ENABLE_VAL} ==="
echo "DATASET_CONFIG=${DATASET_CONFIG}"
echo "DATASET_VAL_CONFIG=${DATASET_VAL_CONFIG}"
if [[ "${VAL_MODE}" == "steps" ]]; then
  echo "VAL_MODE=steps NUM_VAL_POINTS=${NUM_VAL_POINTS} VAL_CHECK_EVERY_N_STEPS=${VAL_CHECK_EVERY_N_STEPS} (val_check_interval batches=${VAL_CHECK_INTERVAL_BATCHES})"
elif [[ "${VAL_MODE}" == "epoch" ]]; then
  echo "VAL_MODE=epoch CHECK_VAL_EVERY_N_EPOCH=${CHECK_VAL_EVERY_N_EPOCH}"
  echo "WARNING: epoch-based val can miss metrics on short trials (few epochs); prefer unset CHECK_VAL_EVERY_N_EPOCH for ≥${NUM_VAL_POINTS} step-spaced vals"
else
  echo "VAL_MODE=off"
fi
echo "HYDRA_RUN_DIR=${HYDRA_RUN_DIR}"
if [[ -n "${EXTRA_HYDRA_ARGS}" ]]; then
  echo "EXTRA_HYDRA_ARGS=${EXTRA_HYDRA_ARGS}"
fi
mkdir -p "${HYDRA_RUN_DIR}"
TRIAL_LOG="${HYDRA_RUN_DIR}/trial.log"

# shellcheck disable=SC2206
EXTRA_HYDRA=( ${EXTRA_HYDRA_ARGS} )

set +e
python train.py \
  domain=protein \
  paradigm=multiframefm \
  "datamodule.batch_size=${BATCH_SIZE}" \
  "datamodule.num_workers=${NUM_WORKERS}" \
  datamodule.val_num_workers=0 \
  model.c_s=768 \
  model.c_cond=768 \
  model.c_frame=256 \
  model.c_framepair=64 \
  model.z_broadcast=true \
  model.rigid_transformer_num_blocks=1 \
  model.rigid_transformer_rigid_updates=true \
  model.use_embedder_sc_rigid_transformer=true \
  model.use_ipa_gating=true \
  model.use_qk_norm=true \
  model.use_amp=true \
  model.rot_preconditioning=true \
  model.num_blocks=12 \
  lmodule.use_ema=true \
  lmodule.strict_weight_loading=false \
  lmodule.log_train_loss_every_step=true \
  corrupter.use_stochastic_centering=true \
  corrupter.center_on_motif=true \
  corrupter.sig_perturb=4 \
  "dataset.config=${DATASET_CONFIG}" \
  "${VAL_ARGS[@]}" \
  "experiment.lightning.devices=${DEVICES}" \
  experiment.lightning.accelerator=gpu \
  "experiment.lightning.strategy=${STRATEGY}" \
  "experiment.lightning.accumulate_grad_batches=${ACCUM}" \
  experiment.lightning.log_every_n_steps=50 \
  "experiment.lightning.max_steps=${MAX_STEPS}" \
  experiment.checkpointer.train_time_interval=null \
  experiment.checkpointer.every_n_train_steps=100000 \
  experiment.checkpointer.save_top_k=0 \
  "experiment.optim.lr=${LR}" \
  "experiment.warm_start=${WARM_START_QUOTED}" \
  experiment.warm_start_weights_only=true \
  "hydra.run.dir=${HYDRA_RUN_DIR}" \
  "${SCHED_ARGS[@]}" \
  "${MLFLOW_ARGS[@]}" \
  "${EXTRA_HYDRA[@]}" \
  2>&1 | tee "${TRIAL_LOG}"
rc=${PIPESTATUS[0]}
set -e

# Smoke / ENABLE_VAL=0 → post-fit extract_training_trial_result(val/loss) raises; treat as OK if fit finished.
if [[ "${rc}" -ne 0 && "${ENABLE_VAL}" != "1" ]]; then
  if grep -qE "Trainer\\.fit\` stopped: \`max_steps=" "${TRIAL_LOG}" \
    || grep -q "Objective metric 'val/loss' was not produced by the training run" "${TRIAL_LOG}"; then
    if grep -q "train/loss_step=" "${TRIAL_LOG}" || grep -qE "Trainer\\.fit\` stopped: \`max_steps=" "${TRIAL_LOG}"; then
      echo "=== HPO TRIAL OK (post-fit val/loss artifact ignored) ==="
      exit 0
    fi
  fi
fi

if [[ "${rc}" -ne 0 ]]; then
  echo "=== HPO TRIAL FAILED rc=${rc} ==="
  exit "${rc}"
fi

echo "=== HPO TRIAL DONE rc=0 ==="
