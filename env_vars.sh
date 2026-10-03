GEOMSTATS_BACKEND="pytorch"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT
# Conda env name for this checkout. Python import is `nomadflow`.
export ENV_NAME="nomadflow"

# Downstream benchmark tools (ProteinMPNN / ESMFold / Foldseek).
export PROTEINMPNN_ROOT="${PROTEINMPNN_ROOT:-${HOME}/progs/ProteinMPNN}"
export PMPNN_ENV="${PMPNN_ENV:-proteinmpnn}"
export ESMFOLD_ENV="${ESMFOLD_ENV:-esmfold}"
# OpenFold main builds cleanly on newer CUDA/Torch stacks used on this machine.
export ESMFOLD_OPENFOLD_REF="${ESMFOLD_OPENFOLD_REF:-git+https://github.com/aqlaboratory/openfold.git@main}"
export ESMFOLD_MODEL_PATH="${ESMFOLD_MODEL_PATH:-${HOME}/progs/esmfold_weights/esmfold_3B_v1.pt}"
export ESM2_MODEL_PATH="${ESM2_MODEL_PATH:-${HOME}/progs/esmfold_weights/esm2_t36_3B_UR50D.pt}"
# optional Apptainer .sif
export ESMFOLD_SIF="${ESMFOLD_SIF:-}"
export FOLDSEEK_ENV="${FOLDSEEK_ENV:-foldseek}"
export FOLDSEEK_BIN="${FOLDSEEK_BIN:-${HOME}/miniconda3/envs/${FOLDSEEK_ENV}/bin}"
export PATH="${FOLDSEEK_BIN}:${PATH}"
export FOLDSEEK_FRAMEFLOW_DB="${FOLDSEEK_FRAMEFLOW_DB:-${HOME}/foldseek_db/frameflow_db}"
export FOLDSEEK_FRAMEDIFF_DB="${FOLDSEEK_FRAMEDIFF_DB:-${HOME}/foldseek_db/framediff_db}"
export FOLDSEEK_PDB_DB="${FOLDSEEK_PDB_DB:-${HOME}/foldseek_db/PDB}"

# Preprocessed Boltz datasets (directories with manifest.json). Required for train.py.
# See scripts/data/README.md and Box links in the root README.
export NOMADFLOW_DATA_FRAMEDIFF_BOLTZ="${NOMADFLOW_DATA_FRAMEDIFF_BOLTZ:-${REPO_ROOT}/data/framediff_boltz}"
export NOMADFLOW_DATA_AFDB_BOLTZ="${NOMADFLOW_DATA_AFDB_BOLTZ:-${REPO_ROOT}/data/afdb_512_clusters_boltz}"
export NOMADFLOW_DATA_AFDB_128_BOLTZ="${NOMADFLOW_DATA_AFDB_128_BOLTZ:-${REPO_ROOT}/data/afdb_128_boltz}"
export NOMADFLOW_DATA_GEOM_QM9_BOLTZ="${NOMADFLOW_DATA_GEOM_QM9_BOLTZ:-${REPO_ROOT}/data/geom_qm9_boltz}"
export NOMADFLOW_DATA_GEOM_DRUGS_BOLTZ="${NOMADFLOW_DATA_GEOM_DRUGS_BOLTZ:-${REPO_ROOT}/data/geom_drugs_boltz}"
export NOMADFLOW_DATA_RCSB_BOLTZ="${NOMADFLOW_DATA_RCSB_BOLTZ:-${REPO_ROOT}/data/rcsb_processed_targets}"

# Sequential sampling (Stage-1 custom -> Stage-2 baseline).
# Optional Stage-1 MLflow run used to generate layout manifests: set STAGE1_MLFLOW_RUN_ID in the environment (no default).
# Baseline ProteinZen checkpoint for Stage-2 resampling.
export STAGE2_BASELINE_CKPT="${STAGE2_BASELINE_CKPT:-${REPO_ROOT}/weights/motif_scaffolding_model/lightning_logs/version_0/checkpoints/epoch-3278-step-780000.ckpt}"
# Directory with `.hydra/config.yaml` for that checkpoint (extracted weights zip
# often has only lightning_logs/, so walking up from the .ckpt finds nothing).
export STAGE2_MODEL_DIR="${STAGE2_MODEL_DIR:-${REPO_ROOT}/configs/sampling/pretrained/motif_scaffolding_model}"
# Optional checkout used for Stage-2 / baseline sample.py. Override if your baseline
# tree is not a sibling worktree named zenprotein-main.
export STAGE2_REPO_ROOT="${STAGE2_REPO_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)/zenprotein-main}"
