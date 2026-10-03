#!/usr/bin/env bash
# Short Voronoi loss ablations on top of hpo_spatial_trial.sh.
#
# Default suite L0–L4: FAFE / inter-motif weights (one semantic focus).
# Legacy A0–A4 kept for one-off re-runs (t_max / agg-up / aggregator_weight).
#
# Primary watch (MLflow): val_task/motif_scaffold_unindexed_voronoi_loss
# Secondary: val/loss_frac_motif_intergroup_fafe, val/loss_frac_aggregator_fafe,
#            val/loss_frac_fafe, val/motif_real_*, val_task/unconditional_loss
#
# Usage:
#   bash scripts/hpo_voronoi_loss_ablations.sh              # print sbatch lines
#   SUBMIT=1 bash scripts/hpo_voronoi_loss_ablations.sh     # submit L0–L4
#   SUBMIT=1 ABLATIONS=L1,L4 bash scripts/hpo_voronoi_loss_ablations.sh
#   LR=5e-4 ACCUM=10 MAX_STEPS=1200 NUM_VAL_POINTS=6 SUBMIT=1 \
#     bash scripts/hpo_voronoi_loss_ablations.sh
#
# Env: LR ACCUM MAX_STEPS WARMUP_STEPS DEVICES BATCH_SIZE NUM_WORKERS
#      NUM_VAL_POINTS ABLATIONS SUBMIT TAG
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

LR="${LR:-5e-4}"
ACCUM="${ACCUM:-10}"
MAX_STEPS="${MAX_STEPS:-1200}"
WARMUP_STEPS="${WARMUP_STEPS:-250}"
DEVICES="${DEVICES:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_VAL_POINTS="${NUM_VAL_POINTS:-6}"
SUBMIT="${SUBMIT:-0}"
ABLATIONS="${ABLATIONS:-L0,L1,L2,L3,L4}"
TAG="${TAG:-loss_L_v1}"

BASE_DATASET_TRAIN="${REPO_ROOT}/configs/train/data/voronoi_crop240_tlock05.yaml"
BASE_DATASET_VAL="${REPO_ROOT}/configs/val/data/voronoi_crop240_tlock05.yaml"
CROP224_TRAIN="${REPO_ROOT}/configs/train/data/voronoi_crop224_tlock05.yaml"
CROP224_VAL="${REPO_ROOT}/configs/val/data/voronoi_crop224_tlock05.yaml"
TMAX099_TRAIN="${REPO_ROOT}/configs/train/data/voronoi_crop240_tlock05_tmax099.yaml"
TMAX099_VAL="${REPO_ROOT}/configs/val/data/voronoi_crop240_tlock05_tmax099.yaml"

mkdir -p "${REPO_ROOT}/slurm_logs"

resolve_ablation() {
  local id="$1"
  DATASET_CONFIG="${BASE_DATASET_TRAIN}"
  DATASET_VAL_CONFIG="${BASE_DATASET_VAL}"
  case "${id}" in
    # --- L suite: intergroup / FAFE weights (current HPO) ---
    L0)
      EXTRA_HYDRA_ARGS=""
      NOTE="baseline defaults: inter=0.2 aggFAFE=0.35 global=0.5"
      ;;
    L1)
      EXTRA_HYDRA_ARGS="lmodule.motif_intergroup_fafe_loss_weight=0.5"
      NOTE="intergroup FAFE 0.2 -> 0.5"
      ;;
    L2)
      EXTRA_HYDRA_ARGS="lmodule.motif_intergroup_fafe_loss_weight=0.5 lmodule.motif_aggregator_fafe_loss_weight=0.2"
      NOTE="inter=0.5, aggFAFE 0.35 -> 0.2"
      ;;
    L3)
      EXTRA_HYDRA_ARGS="lmodule.fafe_loss_weight=0.25"
      NOTE="global FAFE 0.5 -> 0.25"
      ;;
    L4)
      EXTRA_HYDRA_ARGS="lmodule.motif_intergroup_fafe_loss_weight=0.5 lmodule.fafe_loss_weight=0.25"
      NOTE="inter=0.5 + global FAFE 0.25"
      ;;
    L5)
      # Target ~30% loss_frac_motif_intergroup_fafe vs L0 (~4% at weight 0.2).
      # Single knob vs L0; compare scaled_inter/agg only (not val/loss).
      # crop224: 5-GPU DDP OOM'd at crop240 (GPU0 needed ~600MiB headroom).
      DATASET_CONFIG="${CROP224_TRAIN}"
      DATASET_VAL_CONFIG="${CROP224_VAL}"
      EXTRA_HYDRA_ARGS="lmodule.motif_intergroup_fafe_loss_weight=2.0"
      NOTE="inter=2.0 (~30% frac), crop224 for 5-GPU VRAM margin"
      ;;
    # --- legacy A suite ---
    A0)
      EXTRA_HYDRA_ARGS=""
      NOTE="legacy baseline (current weights, Voronoi t_max=0.495)"
      ;;
    A1)
      DATASET_CONFIG="${TMAX099_TRAIN}"
      DATASET_VAL_CONFIG="${TMAX099_VAL}"
      EXTRA_HYDRA_ARGS=""
      NOTE="legacy Voronoi t_max=0.99 (post-lock scaffold; t_lock=0.5)"
      ;;
    A2)
      EXTRA_HYDRA_ARGS="lmodule.fafe_loss_weight=0.25"
      NOTE="legacy global FAFE coeff 0.5 -> 0.25"
      ;;
    A3)
      EXTRA_HYDRA_ARGS="lmodule.motif_aggregator_fafe_loss_weight=0.7"
      NOTE="legacy aggregator FAFE weight 0.35 -> 0.7"
      ;;
    A4)
      EXTRA_HYDRA_ARGS="lmodule.motif_aggregator_weight=1.0"
      NOTE="legacy aggregator VF upweight 5 -> 1"
      ;;
    *)
      echo "Unknown ablation id: ${id}" >&2
      exit 1
      ;;
  esac
}

lr_tag="$(echo "${LR}" | tr '.' 'p' | tr '-' '_')"
IFS=',' read -r -a ABLATION_LIST <<< "${ABLATIONS}"
EFF=$((BATCH_SIZE * ACCUM * DEVICES))

echo "=== Voronoi loss ablations tag=${TAG} lr=${LR} accum=${ACCUM} eff=${EFF} steps=${MAX_STEPS} devices=${DEVICES} submit=${SUBMIT} ==="
echo "Watch: val_task/motif_scaffold_unindexed_voronoi_loss (+ val/loss_frac_*, motif_real_*, unc)"
echo

for id in "${ABLATION_LIST[@]}"; do
  id="$(echo "${id}" | tr -d '[:space:]')"
  [[ -z "${id}" ]] && continue
  resolve_ablation "${id}"
  # Prefer human-readable names for L-suite (match chat plan); keep compact for A/legacy.
  if [[ "${id}" == L0 ]]; then
    run_name="voronoi_loss_L0_baseline_lr${lr_tag}_${MAX_STEPS}"
  elif [[ "${id}" == L1 ]]; then
    run_name="voronoi_loss_L1_inter_strong_lr${lr_tag}_${MAX_STEPS}"
  elif [[ "${id}" == L2 ]]; then
    run_name="voronoi_loss_L2_inter_strong_agg_down_lr${lr_tag}_${MAX_STEPS}"
  elif [[ "${id}" == L3 ]]; then
    run_name="voronoi_loss_L3_global_fafe_down_lr${lr_tag}_${MAX_STEPS}"
  elif [[ "${id}" == L4 ]]; then
    run_name="voronoi_loss_L4_inter_strong_global_down_lr${lr_tag}_${MAX_STEPS}"
  elif [[ "${id}" == L5 ]]; then
    run_name="voronoi_loss_L5_inter_frac30_lr${lr_tag}_${MAX_STEPS}_d${DEVICES}a${ACCUM}_c224"
  else
    run_name="voronoi_${TAG}_${id}_lr${lr_tag}_eff${EFF}"
  fi
  hydra_dir="${REPO_ROOT}/outputs/hpo/${run_name}"
  job_name="pz_${TAG}_${id}"
  mkdir -p "${hydra_dir}"
  wrap_file="${hydra_dir}/sbatch_wrap.sh"

  # Quoting: EXTRA_HYDRA_ARGS may be empty or multi-token; @Q is bash 4.4+.
  cat > "${wrap_file}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
# Avoid NFS TMPDIR hangs seen in HPO v5; prefer node-local /tmp if present.
unset TMPDIR
export USE_TASKSET=0
export MLFLOW_ALLOW_FILE_STORE=true
export BATCH_SIZE=${BATCH_SIZE}
export ACCUM=${ACCUM}
export LR=${LR}
export MAX_STEPS=${MAX_STEPS}
export WARMUP_STEPS=${WARMUP_STEPS}
export USE_COSINE=0
export DEVICES=${DEVICES}
export NUM_WORKERS=${NUM_WORKERS}
export ENABLE_VAL=1
export NUM_VAL_POINTS=${NUM_VAL_POINTS}
export ENABLE_MLFLOW=1
export DATASET_CONFIG=${DATASET_CONFIG}
export DATASET_VAL_CONFIG=${DATASET_VAL_CONFIG}
export MLFLOW_RUN_NAME=${run_name}
export HYDRA_RUN_DIR=${hydra_dir}
export EXTRA_HYDRA_ARGS=${EXTRA_HYDRA_ARGS@Q}
source "\$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-nomadflow}"
module load cuda/12.8
cd ${REPO_ROOT}
source env_vars.sh
bash scripts/hpo_spatial_trial.sh
EOF
  chmod +x "${wrap_file}"

  echo "--- ${id}: ${NOTE}"
  echo "    MLFLOW_RUN_NAME=${run_name}"
  echo "    DATASET_CONFIG=${DATASET_CONFIG}"
  echo "    EXTRA_HYDRA_ARGS=${EXTRA_HYDRA_ARGS:-<none>}"
  echo "    wrap=${wrap_file}"

  # Match CPU budget roughly to GPUs (8 cores/GPU); keep gres in sync with DEVICES.
  CPUS=$((DEVICES * 8))
  if [[ "${CPUS}" -lt 16 ]]; then CPUS=16; fi

  if [[ "${SUBMIT}" == "1" ]]; then
    sbatch -p gpu --gres="gpu:${DEVICES}" -N 1 -c "${CPUS}" -t 12:00:00 \
      -J "${job_name}" \
      -o "${REPO_ROOT}/slurm_logs/slurm-%j.out" \
      -e "${REPO_ROOT}/slurm_logs/slurm-%j.err" \
      --wrap "bash -lc ${wrap_file@Q}"
  else
    cat <<EOF
    sbatch -p gpu --gres=gpu:${DEVICES} -N 1 -c ${CPUS} -t 12:00:00 \\
      -J ${job_name} \\
      -o ${REPO_ROOT}/slurm_logs/slurm-%j.out \\
      -e ${REPO_ROOT}/slurm_logs/slurm-%j.err \\
      --wrap "bash -lc ${wrap_file@Q}"
EOF
  fi
  echo
done

if [[ "${SUBMIT}" != "1" ]]; then
  echo "Dry run only. Re-run with SUBMIT=1 to enqueue."
fi
