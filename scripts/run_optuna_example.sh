#!/usr/bin/env bash
# Local phase1 HPO run on top of train_optuna.py.
# By default this is a more substantial multi-trial run, not a smoke test,
# but every key parameter can be overridden via the env vars below.
#
# Variables:
#   USE_TASKSET=0               — disable taskset
#   TASKSET_CPUS=0-7            — CPU mask for taskset
#   DATASET_CONFIG=...          — train dataset yaml
#   DATASET_VAL_CONFIG=...      — validation dataset yaml (required for HPO)
#   MLFLOW_TRACKING_URI=...     — tracking backend (default <REPO_ROOT>/mlruns)
#   MLFLOW_ALLOW_FILE_STORE=... — allow the file-store backend (default true)
#   MLFLOW_EXPERIMENT=...       — MLflow experiment name
#   OPTUNA_STUDY_NAME=...       — Optuna study name / parent MLflow run context
#   HPO_N_TRIALS=8              — number of trials
#   MAX_EPOCHS=2                — budget per trial
#   DEVICES=1                   — number of GPUs/devices for Lightning
#   NUM_WORKERS=4               — dataloader workers
#   VAL_NUM_WORKERS=0           — validation dataloader workers
#
# View:
#   mlflow ui --backend-store-uri "${MLFLOW_TRACKING_URI}"
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

DATASET_CONFIG="${DATASET_CONFIG:-${REPO_ROOT}/configs/train/data/framediff_crop256.yaml}"
DATASET_VAL_CONFIG="${DATASET_VAL_CONFIG:-${REPO_ROOT}/configs/val/data/framediff_crop256.yaml}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-${REPO_ROOT}/mlruns}"
MLFLOW_ALLOW_FILE_STORE="${MLFLOW_ALLOW_FILE_STORE:-true}"
export MLFLOW_ALLOW_FILE_STORE
MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT:-nomadflow_optuna}"
OPTUNA_STUDY_NAME="${OPTUNA_STUDY_NAME:-nomadflow_optuna_phase1_serious}"
HPO_N_TRIALS="${HPO_N_TRIALS:-999}"
MAX_EPOCHS="${MAX_EPOCHS:-6}"
DEVICES="${DEVICES:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-0}"

"${TASKSET_PREFIX[@]}" python scripts/train_optuna.py \
  domain=protein \
  paradigm=multiframefm \
  datamodule.batch_size=4  \
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
  datamodule.num_workers="${NUM_WORKERS}" \
  datamodule.val_num_workers="${VAL_NUM_WORKERS}" \
  experiment.lightning.devices="${DEVICES}" \
  experiment.lightning.strategy=ddp_find_unused_parameters_true \
  experiment.lightning.max_epochs="${MAX_EPOCHS}" \
  experiment.mlflow.enabled=true \
  experiment.mlflow.tracking_uri="${MLFLOW_TRACKING_URI}" \
  experiment.mlflow.experiment_name="${MLFLOW_EXPERIMENT}" \
  hpo.study_name="${OPTUNA_STUDY_NAME}" \
  hpo.n_trials="${HPO_N_TRIALS}" \
  hpo.execution_backend=local
