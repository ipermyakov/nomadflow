#!/usr/bin/env bash
# Sharded Mode-2 tlock-0.5 sampling (local Lightning ckpt; empty MLflow artifacts).
#
# Suites:
#   full  12 catalytic ext tasks (default)  RUN_NAME=0c62dc8a__tlock05_full
#   se3    9 non-catalytic SE3 tasks        RUN_NAME=0c62dc8a__tlock05_se3
#
#   bash scripts/launch_two_stage_full_sharded.sh [--suite full|se3] submit-stage1
#   bash scripts/launch_two_stage_full_sharded.sh --suite se3 merge-stage1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../env_vars.sh"

SUITE="${SUITE:-full}"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --suite)
            SUITE="$2"
            shift 2
            ;;
        -h|--help)
            break
            ;;
        --*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

RUN_ID="0c62dc8a0141433895d0e2cd349a4092"
STAGE1_CKPT="${REPO_ROOT}/outputs/pretrain/voronoi_crop320_from224_lr5e5_clip5/proteinzen_phase1/${RUN_ID}/checkpoints/last.ckpt"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
YAML_STEM=""
JOB_PREFIX=""
EXPECTED_LAYOUTS=0
SHARDS=()

case "${SUITE}" in
    full)
        RUN_NAME="0c62dc8a__tlock05_full"
        YAML_STEM="_full_tlock05"
        JOB_PREFIX="pz_full"
        EXPECTED_LAYOUTS=2400
        SHARDS=(s0_1qjg s1_5aou s2_quad s3_7k4v)
        ;;
    se3)
        RUN_NAME="0c62dc8a__tlock05_se3"
        YAML_STEM="_se3_tlock05"
        JOB_PREFIX="pz_se3"
        EXPECTED_LAYOUTS=1800
        SHARDS=(s0_1prw s1_short s2_5yui s3_1bcf)
        ;;
    *)
        echo "Unknown --suite: ${SUITE} (expected full or se3)" >&2
        exit 1
        ;;
esac

TASKS_YAML="${REPO_ROOT}/article_results/sampling_runs/${YAML_STEM}.yaml"
STAGE1_OUT="${REPO_ROOT}/article_results/sampling_runs/${RUN_NAME}_stage_1"
STAGE2_OUT="${REPO_ROOT}/article_results/sampling_runs/${RUN_NAME}_stage_2"
ANALYSIS_OUT="${REPO_ROOT}/article_results/analysis_runs/${RUN_NAME}"
SLURM_LOG_DIR="${REPO_ROOT}/slurm_logs"

CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"

shard_yaml() {
    local shard="$1"
    printf '%s/article_results/sampling_runs/%s_%s.yaml' "${REPO_ROOT}" "${YAML_STEM}" "${shard}"
}

stage1_shard_out() {
    local shard="$1"
    printf '%s_%s' "${STAGE1_OUT}" "${shard}"
}

stage2_shard_out() {
    local shard="$1"
    printf '%s_%s' "${STAGE2_OUT}" "${shard}"
}

usage() {
    cat <<EOF
Usage: $0 [--suite full|se3] COMMAND [shard]

Suites:
  full   catalytic ext (12×200=2400)   default
  se3    non-catalytic SE3 (9×200=1800)

Commands:
  submit-stage1     sbatch 4× gpu:1 Stage-1 jobs (local ckpt)
  run-stage1 SHARD  Stage-1 sample.py for one shard (inside Slurm)
  merge-stage1      concat shard layout_manifest.jsonl → canonical stage_1
  tasks             build Stage-2 tasks yaml (--include-motif-groups)
  submit-stage2     sbatch 4× gpu:1 Stage-2 jobs
  run-stage2 SHARD  Stage-2 via run_two_stage_motif_pipeline.sh (inside Slurm)
  merge-stage2      prefix-copy shard PDBs/metadata into canonical stage_2
  check-eval        check_eval_layout.py on canonical stage_2
  submit-eval       sbatch 1× gpu:1 eval (PMPNN/ESMFold/metrics)
  run-eval          eval wrapper (inside Slurm)

SUITE=${SUITE}  RUN_NAME=${RUN_NAME}
EOF
}

_require_ckpt() {
    if [ ! -f "${STAGE1_CKPT}" ]; then
        echo "Missing Stage-1 checkpoint: ${STAGE1_CKPT}" >&2
        exit 1
    fi
}

_require_yaml() {
    local path="$1"
    if [ ! -f "${path}" ]; then
        echo "Missing yaml: ${path}" >&2
        exit 1
    fi
}

_refuse_slice_paths() {
    case "$1" in
        *1qjg2*)
            echo "Refusing path that looks like the two-task slice: $1" >&2
            exit 1
            ;;
    esac
}

_sbatch_gpu() {
    local job_name="$1"
    local walltime="$2"
    local inner="$3"
    mkdir -p "${SLURM_LOG_DIR}"
    sbatch -p gpu --gres=gpu:1 -N 1 -c 8 -t "${walltime}" \
        -J "${job_name}" \
        -o "${SLURM_LOG_DIR}/slurm-%j.out" \
        -e "${SLURM_LOG_DIR}/slurm-%j.err" \
        --wrap "bash -lc $(printf '%q' "${inner}")"
}

_conda_cuda_cd() {
    export TMPDIR="${TMPDIR:-${HOME}/tmp}"
    mkdir -p "${TMPDIR}"
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
    conda activate "${ENV_NAME}"
    module load cuda/12.8
    cd "${REPO_ROOT}"
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/env_vars.sh"
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    export MLFLOW_ALLOW_FILE_STORE="${MLFLOW_ALLOW_FILE_STORE:-true}"
}

run_stage1() {
    local shard="${1:?shard required}"
    local yaml out_dir
    yaml="$(shard_yaml "${shard}")"
    out_dir="$(stage1_shard_out "${shard}")"
    _refuse_slice_paths "${out_dir}"
    _require_ckpt
    _require_yaml "${yaml}"
    _conda_cuda_cd
    echo "== Stage-1 suite ${SUITE} shard ${shard} =="
    echo "  ckpt:    ${STAGE1_CKPT}"
    echo "  yaml:    ${yaml}"
    echo "  out_dir: ${out_dir}"
    python "${REPO_ROOT}/sample.py" \
        checkpoint_path="$(python -c 'import json,sys; print(json.dumps(sys.argv[1]))' "${STAGE1_CKPT}")" \
        out_dir="${out_dir}" \
        sampler.tasks_yaml="${yaml}" \
        sampler.batch_size=6 \
        save_traj=false \
        export_layout_manifest=true \
        output_motif_chains=true
}

submit_stage1() {
    local shard yaml out_dir inner
    _require_ckpt
    _require_yaml "${TASKS_YAML}"
    for shard in "${SHARDS[@]}"; do
        yaml="$(shard_yaml "${shard}")"
        out_dir="$(stage1_shard_out "${shard}")"
        _require_yaml "${yaml}"
        _refuse_slice_paths "${out_dir}"
        if [ -f "${out_dir}/layout_manifest.jsonl" ] && [ "${FORCE:-0}" != "1" ]; then
            echo "Refuse overwrite ${out_dir}/layout_manifest.jsonl (set FORCE=1)" >&2
            exit 1
        fi
        inner="cd ${REPO_ROOT} && export TMPDIR=${HOME}/tmp CUDA_VISIBLE_DEVICES=0 && mkdir -p ${HOME}/tmp && bash ${SCRIPT_DIR}/launch_two_stage_full_sharded.sh --suite ${SUITE} run-stage1 ${shard}"
        _sbatch_gpu "${JOB_PREFIX}_s1_${shard}" "2-00:00:00" "${inner}"
    done
}

merge_stage1() {
    local shard manifest n total=0
    mkdir -p "${STAGE1_OUT}"
    : > "${STAGE1_OUT}/layout_manifest.jsonl"
    for shard in "${SHARDS[@]}"; do
        manifest="$(stage1_shard_out "${shard}")/layout_manifest.jsonl"
        if [ ! -f "${manifest}" ]; then
            echo "Missing ${manifest}" >&2
            exit 1
        fi
        n="$(grep -c . "${manifest}" || true)"
        echo "  ${shard}: ${n} layouts"
        cat "${manifest}" >> "${STAGE1_OUT}/layout_manifest.jsonl"
        total=$((total + n))
    done
    echo "Merged ${total} layouts → ${STAGE1_OUT}/layout_manifest.jsonl"
    if [ "${total}" -ne "${EXPECTED_LAYOUTS}" ]; then
        echo "Expected ${EXPECTED_LAYOUTS} layouts, got ${total}" >&2
        exit 1
    fi
}

run_tasks() {
    _refuse_slice_paths "${STAGE1_OUT}"
    _refuse_slice_paths "${STAGE2_OUT}"
    if [ ! -f "${STAGE1_OUT}/layout_manifest.jsonl" ]; then
        echo "Missing ${STAGE1_OUT}/layout_manifest.jsonl. Run merge-stage1." >&2
        exit 1
    fi
    "${REPO_ROOT}/scripts/run_two_stage_motif_pipeline.sh" \
        --run-id "${RUN_ID}" \
        --config "${TASKS_YAML}" \
        --run-name "${RUN_NAME}" \
        --stage1-out "${STAGE1_OUT}" \
        --stage2-out "${STAGE2_OUT}" \
        --analysis-out "${ANALYSIS_OUT}" \
        --cuda-device "${CUDA_DEVICE}" \
        --include-motif-groups \
        --only tasks
}

_split_stage2_tasks() {
    export STAGE2_OUT PIPELINE_SUITE="${SUITE}"
    python - <<'PY'
import os
from pathlib import Path
import yaml

stage2 = Path(os.environ["STAGE2_OUT"])
suite = os.environ["PIPELINE_SUITE"]
src = stage2 / "tasks_from_layouts.yaml"
payload = yaml.safe_load(src.read_text(encoding="utf-8"))
tasks = payload["tasks"]

def task_name(task):
    return str(task.get("source_stage1_task_name") or task.get("name") or "")

if suite == "full":
    groups = {
        "s0_1qjg": lambda n: n.startswith("1QJG_"),
        "s1_5aou": lambda n: n.startswith("5AOU_") and not n.startswith("5AOU_QUAD_"),
        "s2_quad": lambda n: n.startswith("5AOU_QUAD_"),
        "s3_7k4v": lambda n: n.startswith("7K4V_"),
    }
    expected = {shard: 600 for shard in groups}
else:
    name_sets = {
        "s0_1prw": {"1PRW", "5IUS"},
        "s1_short": {"2KL8", "4JHW", "6VW1"},
        "s2_5yui": {"5YUI", "5YUI_2grp"},
        "s3_1bcf": {"1BCF", "1BCF_2grp"},
    }
    groups = {shard: (lambda names: (lambda n, names=names: n in names))(names) for shard, names in name_sets.items()}
    expected = {shard: 200 * len(names) for shard, names in name_sets.items()}

for shard, pred in groups.items():
    selected = [t for t in tasks if pred(task_name(t))]
    out_dir = Path(str(stage2) + f"_{shard}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "tasks_from_layouts.yaml"
    out.write_text(yaml.safe_dump({"tasks": selected}, sort_keys=False), encoding="utf-8")
    print(f"{shard}: {len(selected)} tasks -> {out}")
    if len(selected) != expected[shard]:
        raise SystemExit(f"{shard}: expected {expected[shard]} Stage-2 tasks, got {len(selected)}")
PY
}

run_stage2() {
    local shard="${1:?shard required}"
    local out_dir
    out_dir="$(stage2_shard_out "${shard}")"
    _refuse_slice_paths "${out_dir}"
    if [ ! -f "${out_dir}/tasks_from_layouts.yaml" ]; then
        echo "Missing ${out_dir}/tasks_from_layouts.yaml. Run tasks + submit-stage2." >&2
        exit 1
    fi
    _conda_cuda_cd
    "${REPO_ROOT}/scripts/run_two_stage_motif_pipeline.sh" \
        --run-id "${RUN_ID}" \
        --config "${TASKS_YAML}" \
        --run-name "${RUN_NAME}_${shard}" \
        --stage1-out "${STAGE1_OUT}" \
        --stage2-out "${out_dir}" \
        --analysis-out "${ANALYSIS_OUT}_${shard}" \
        --cuda-device "${CUDA_DEVICE}" \
        --include-motif-groups \
        --only stage2
}

submit_stage2() {
    local shard out_dir inner
    if [ ! -f "${STAGE2_OUT}/tasks_from_layouts.yaml" ]; then
        echo "Missing ${STAGE2_OUT}/tasks_from_layouts.yaml. Run tasks first." >&2
        exit 1
    fi
    _split_stage2_tasks
    for shard in "${SHARDS[@]}"; do
        out_dir="$(stage2_shard_out "${shard}")"
        _refuse_slice_paths "${out_dir}"
        if [ -d "${out_dir}/samples" ] && [ "${FORCE:-0}" != "1" ]; then
            echo "Refuse overwrite ${out_dir}/samples (set FORCE=1)" >&2
            exit 1
        fi
        inner="cd ${REPO_ROOT} && export TMPDIR=${HOME}/tmp CUDA_VISIBLE_DEVICES=0 && mkdir -p ${HOME}/tmp && bash ${SCRIPT_DIR}/launch_two_stage_full_sharded.sh --suite ${SUITE} run-stage2 ${shard}"
        _sbatch_gpu "${JOB_PREFIX}_s2_${shard}" "1-00:00:00" "${inner}"
    done
}

merge_stage2() {
    export STAGE2_OUT PIPELINE_SUITE="${SUITE}" EXPECTED_LAYOUTS
    python - <<'PY'
import json
import os
import shutil
from pathlib import Path

stage2 = Path(os.environ["STAGE2_OUT"])
suite = os.environ["PIPELINE_SUITE"]
expected = int(os.environ["EXPECTED_LAYOUTS"])
if suite == "full":
    shards = ["s0_1qjg", "s1_5aou", "s2_quad", "s3_7k4v"]
else:
    shards = ["s0_1prw", "s1_short", "s2_5yui", "s3_1bcf"]
samples = stage2 / "samples"
samples_eval = stage2 / "samples_eval"
metadata_dir = stage2 / "metadata"
for d in (samples, samples_eval, metadata_dir):
    d.mkdir(parents=True, exist_ok=True)

merged = {}
for shard in shards:
    shard_dir = Path(str(stage2) + f"_{shard}")
    meta_path = shard_dir / "samples_metadata.json"
    if not meta_path.is_file():
        raise SystemExit(f"Missing {meta_path}")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    for name, sample in payload.items():
        new_name = f"{shard}__{name}"
        for key in ("path", "viz_path"):
            raw = sample.get(key)
            if not raw:
                continue
            src = Path(raw)
            if not src.is_file():
                raise SystemExit(f"Missing {src}")
            if "samples_eval" in src.parts:
                dest = samples_eval / f"{new_name}.pdb"
            else:
                dest = samples / f"{new_name}.pdb"
            shutil.copy2(src, dest)
            sample[key] = str(dest)
        merged[new_name] = sample
    print(f"{shard}: {len(payload)} samples")

if len(merged) != expected:
    raise SystemExit(f"Expected {expected} merged Stage-2 samples, got {len(merged)}")
out_meta = stage2 / "samples_metadata.json"
out_meta.write_text(json.dumps(merged), encoding="utf-8")
print(f"Wrote {out_meta} ({len(merged)} samples)")
PY
}

check_eval() {
    _conda_cuda_cd
    PYTHONPATH="${REPO_ROOT}" python "${REPO_ROOT}/scripts/analysis/check_eval_layout.py" \
        --sampling-run "${STAGE2_OUT}"
}

run_eval() {
    _refuse_slice_paths "${STAGE2_OUT}"
    _refuse_slice_paths "${ANALYSIS_OUT}"
    _conda_cuda_cd
    "${REPO_ROOT}/scripts/run_two_stage_motif_pipeline.sh" \
        --run-id "${RUN_ID}" \
        --config "${TASKS_YAML}" \
        --run-name "${RUN_NAME}" \
        --stage1-out "${STAGE1_OUT}" \
        --stage2-out "${STAGE2_OUT}" \
        --analysis-out "${ANALYSIS_OUT}" \
        --cuda-device "${CUDA_DEVICE}" \
        --include-motif-groups \
        --only eval \
        --eval-target stage2
}

submit_eval() {
    local inner
    inner="cd ${REPO_ROOT} && export TMPDIR=${HOME}/tmp CUDA_VISIBLE_DEVICES=0 && mkdir -p ${HOME}/tmp && bash ${SCRIPT_DIR}/launch_two_stage_full_sharded.sh --suite ${SUITE} run-eval"
    _sbatch_gpu "${JOB_PREFIX}_eval" "12:00:00" "${inner}"
}

cmd="${1:-}"
shift || true
case "${cmd}" in
    submit-stage1) submit_stage1 ;;
    run-stage1) run_stage1 "${1:-}" ;;
    merge-stage1) merge_stage1 ;;
    tasks) run_tasks ;;
    submit-stage2) submit_stage2 ;;
    run-stage2) run_stage2 "${1:-}" ;;
    merge-stage2) merge_stage2 ;;
    check-eval) check_eval ;;
    submit-eval) submit_eval ;;
    run-eval) run_eval ;;
    -h|--help|"") usage ;;
    *)
        echo "Unknown command: ${cmd}" >&2
        usage >&2
        exit 1
        ;;
esac
