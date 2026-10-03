#!/usr/bin/env bash
# Short GPU smoke for framediff-only spatial train path.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-nomadflow}"
module load cuda/12.8
[[ -f "${REPO_ROOT}/env_vars.sh" ]] && source "${REPO_ROOT}/env_vars.sh"
export REPO_ROOT
export TMPDIR="${TMPDIR:-$HOME/tmp}"
export USE_TASKSET=0
export MLFLOW_ALLOW_FILE_STORE=true
mkdir -p "$TMPDIR"

echo "=== SMOKE HOST=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} ==="
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "ngpu", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device0", torch.cuda.get_device_name(0))
    x = torch.randn(2, 2, device="cuda") @ torch.randn(2, 2, device="cuda")
    print("matmul_ok", float(x.sum()))
PY

WARM_START="${REPO_ROOT}/weights/motif_scaffolding_model/lightning_logs/version_0/checkpoints/epoch-3278-step-780000.ckpt"
WARM_START_QUOTED="$(python -c 'import json,sys; print(json.dumps(sys.argv[1]))' "${WARM_START}")"
DATASET_CONFIG="${REPO_ROOT}/configs/train/data/spatial_framediff_crop256_finetune.yaml"
DATASET_VAL_CONFIG="${REPO_ROOT}/configs/val/data/spatial_framediff_crop256_finetune.yaml"
OUT_DIR="${REPO_ROOT}/outputs/smoke/spatial_framediff_gpu"

python train.py \
  domain=protein \
  paradigm=multiframefm \
  datamodule.batch_size=1 \
  datamodule.num_workers=2 \
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
  dataset.config="${DATASET_CONFIG}" \
  dataset.val_config="${DATASET_VAL_CONFIG}" \
  experiment.lightning.devices=1 \
  experiment.lightning.accelerator=gpu \
  experiment.lightning.strategy=auto \
  experiment.lightning.check_val_every_n_epoch=100000 \
  experiment.lightning.accumulate_grad_batches=1 \
  experiment.lightning.log_every_n_steps=1 \
  experiment.lightning.max_steps=2 \
  experiment.lightning.num_sanity_val_steps=0 \
  experiment.checkpointer.train_time_interval=null \
  experiment.checkpointer.every_n_train_steps=100000 \
  experiment.checkpointer.save_top_k=0 \
  experiment.optim.lr=1e-4 \
  experiment.mlflow.enabled=false \
  experiment.warm_start="${WARM_START_QUOTED}" \
  experiment.warm_start_weights_only=true \
  hydra.run.dir="${OUT_DIR}"

echo "=== SMOKE DONE rc=$? ==="
