# Article Results Workspace

Workspace for sampling outputs, downstream analysis, and paper-ready summaries for
the multi-motif motif-scaffolding experiments.

The repository should track only the structure, manifests, lightweight summaries,
and notes. Heavy generated artifacts (PDBs, ESMFold folders, Foldseek databases,
raw sampling outputs) are ignored by `article_results/.gitignore`.

Public sampling API (`python sample.py`, `out_dir` layout, task YAML) lives in the
root [README.md](../README.md). This file is the lab protocol for the article
benchmark: chain-ID contract, resume rules, and the three launch modes.

## Directory Layout

- `RUN_TRACKER.md` - local (gitignored) checklist of active vs archived campaigns,
  residual merges, eval status, and where jobs ran.
- `manifests/` - model registry and benchmark matrix files that define what was
  run.
- `sampling_runs/` - raw per-run sampling outputs (`samples/`,
  `samples_metadata.json`, ProteinMPNN outputs, ESMFold outputs).
- `analysis_runs/` - per-run metric tables and derived analysis artifacts.
- `summary_tables/` - cross-run CSV/JSON summaries used for paper tables.
- `notebooks/` - comparison notebook. Tracked template:
  `compare_analysis_runs.ipynb`. Working copy with outputs:
  `compare_analysis_runs.local.ipynb` (gitignored).
- `figures/` - paper figure inputs and rendered figure artifacts.
- `external_baselines/` - manually curated prior-work numbers and notes.

## Expected Run Layout

Use one directory per benchmarked model / config pair:

```text
sampling_runs/
  <run_id>/
    manifest.yaml
    samples/
    samples_metadata.json
    pmpnn_fixed_pos_dict.jsonl
    esmfold/

analysis_runs/
  <run_id>/
    metrics.csv
    multimotif_metrics.csv
    task_success.csv
    unique_success.json
    failure_modes.csv
```

Recommended `run_id` format:

```text
<model_key>__<benchmark_key>__<checkpoint_key>__<YYYYMMDD>
```

Example:

```text
ours_multimotif__protpardelle_m2plus__last__20260701
```

## Downstream + Analysis

After sampling into `sampling_runs/<run_id>/`, run the article benchmark
downstream chain (ProteinMPNN → ESMFold → RMSD → analysis):

```bash
qsub scripts/submit_scripts/eval/submit_article_motif_benchmark.sh \
  article_results/sampling_runs/<run_id>
```

Or run analysis only (if `esmfold/folding_rmsd.csv` and `esmfold/sc_rmsd.csv`
already exist):

```bash
python scripts/analysis/run_motif_benchmark_analysis.py \
  --sampling-run article_results/sampling_runs/<run_id> \
  --analysis-out article_results/analysis_runs/<run_id>
```

Output column definitions: [`analysis_runs/metrics_schema.md`](analysis_runs/metrics_schema.md).

`env_vars.sh` defaults (`STAGE2_BASELINE_CKPT`, `STAGE2_REPO_ROOT`; optional
`STAGE1_MLFLOW_RUN_ID`) keep stage settings from getting lost between sessions.
`STAGE2_REPO_ROOT` is optional: override it to a baseline ProteinZen checkout if
Stage-2 should not use this tree. Do not patch main’s `PDBWriter` for
`samples_eval/`.

## Motif benchmark launch modes

There are three supported ways to produce motif-scaffolding samples for the article
benchmark. Downstream eval is always `scripts/submit_scripts/eval/submit_article_motif_benchmark.sh`
from **this** repo (ProteinMPNN → ESMFold → RMSD → `run_motif_benchmark_analysis.py`).
Do not add a fourth eval wrapper.

| | **1. One-shot** | **2. Two-stage** | **3. Baseline vanilla** |
|---|---|---|---|
| Sampling | `sample.py` / `sample_from_mlflow_run.py` **here** | Stage-1 here → Stage-2 `sample.py` in `STAGE2_REPO_ROOT` | `sample.py` in `STAGE2_REPO_ROOT` only |
| Wrapper | `scripts/sample_from_mlflow_run.py` | `scripts/run_two_stage_motif_pipeline.sh` | `scripts/run_baseline_motif_pipeline.sh` |
| Default tasks YAML | (your multi-motif SE3 config) | `config_multi_motif_se3_catalytic_tlock_0.5.yaml` | `config.yaml` |
| `samples_eval/` | Written at sample time (`write_eval_pdb=true`) | Stage-1: at write time. Stage-2: post-hoc `prepare_eval_pdbs_from_samples.py` if incomplete | Post-hoc prepare (main writer has no `samples_eval/`) |
| Eval chain ID | `sample_chain_name` (usually `A`; `P` 3IXT, `B` 1YCR, `F` 4JHW) | Stage-2 ≠ Mode 3. Main writes two-chain `A`+`B` viz PDBs; eval is scaffold (`sample_chain_name` / `stage2_sample_chain_name`, usually `A`), not viz motif `B` | Native/sole chain (`B` 1YCR, `P` 3IXT, `F` 4JHW, `A` else). Persist that id as `sample_chain_name` |
| Eval target dir | the sampling `out_dir` | `--eval-target stage1\|stage2` (default stage-2) | `--out-dir` sampling run |

`--cuda-device` on the wrappers is exported for **both** sampling and `--only eval`
(`CUDA_VISIBLE_DEVICES`). Restrict GPUs with `CUDA_VISIBLE_DEVICES` as appropriate
for your machine.

### Chain-ID contract

Enforced by `nomadflow/benchmark/eval_layout.py`, tests in `tests/test_eval_layout.py`:

- Single chain in `samples_eval/*.pdb` == `sample_chain_name` == `fixed_seq_chain` == PMPNN dict keys == `rmsd_reference_chain`.
- `rmsd_folded_chain` is always `A` (ESMFold output).
- Never relabel eval PDBs to `A` when that would desync metadata (`sample_chain_name=P` must stay `P`).
- `_ensure_samples_eval` skips the PDB rewrite only when eval PDB **names** match `samples_metadata.json` **and** `check_eval_layout` is empty (a leftover incomplete stub is not complete; a complete but wrong motif-`B` extract is rewritten). If skipped, metadata is synced to the chain actually in those PDBs.

### Resume

| Artifact | Skip when complete | Force |
|---|---|---|
| `samples_eval/` | PDB names == metadata keys and `check_eval_layout` empty | `prepare_eval_pdbs_from_samples.py --force` |
| `seqs/` | `seqs/<sample>.fa` for every metadata sample | `PMPNN_FORCE=1` |
| `esmfold/` | existing `esmfold.log` contains `Predicted structure for` | `ESMFOLD_FORCE=1` |

Validate a run:

```bash
PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python scripts/analysis/check_eval_layout.py \
    --sampling-run article_results/sampling_runs/<run_id>
```

## Two-stage motif pipeline (layout → baseline resampling)

For a two-stage flow where Stage-1 uses a modified model to produce motif layouts and Stage-2
resamples each layout with baseline ProteinZen:

1. Run Stage-1 sampling with manifest export enabled.
2. Build Stage-2 task YAML from `layout_manifest.jsonl`.
3. Run Stage-2 sampling on baseline (`main`) using the generated YAML.

Recommended hybrid execution:
- Stage-1 (your trained variants): `scripts/sample_from_mlflow_run.py` using only `run_id`.
- Stage-2 (fixed pretrained baseline weights): `sample.py` with `model_dir=<pretrained_folder>`.

Stage-1 (modified branch, via `sample.py`):
```bash
python sample.py \
    model_dir="${STAGE1_MODEL_DIR}" \
    out_dir="${REPO_ROOT}/article_results/sampling_runs/stage1_layout__smoke_1qjg" \
    checkpoint_idx=-1 \
    sampler.tasks_yaml="${REPO_ROOT}/configs/sampling/protpardelle_motif_scaffolding/config_multi_motif_se3_catalytic_tlock_0.5.yaml" \
    sampler.batch_size=1 \
    save_traj=false \
    export_layout_manifest=true
```

Stage-1 alternative (modified branch, via MLflow `run_id`):
```bash
python scripts/sample_from_mlflow_run.py \
    --run-id <stage1_mlflow_run_id> \
    --tasks-yaml "${REPO_ROOT}/configs/sampling/protpardelle_motif_scaffolding/config_multi_motif_se3_catalytic_tlock_0.5.yaml" \
    --out-dir "${REPO_ROOT}/article_results/sampling_runs/stage1_layout__smoke_1qjg__best" \
    --checkpoint best \
    --sample-arg export_layout_manifest=true
```

Build Stage-2 tasks (one task per layout):
```bash
python scripts/analysis/build_stage2_tasks_from_layout_manifest.py \
    --layout-manifest "${REPO_ROOT}/article_results/sampling_runs/stage1_layout__smoke_1qjg/layout_manifest.jsonl" \
    --output-yaml "${REPO_ROOT}/article_results/sampling_runs/stage2_tasks__smoke_1qjg/tasks_from_layouts.yaml" \
    --stage2-num-samples 1 \
    --total-length sample_length \
    --motif-source viz_chains
```

Stage-2 (baseline `main`, via `sample.py`):
```bash
python sample.py \
    model_dir="${NOMADFLOW_MOTIF_MODEL_DIR}" \
    out_dir="${REPO_ROOT}/article_results/sampling_runs/original_proteinzen_from_layout__smoke_1qjg__pretrained" \
    checkpoint_idx=-1 \
    sampler.tasks_yaml="${REPO_ROOT}/article_results/sampling_runs/stage2_tasks__smoke_1qjg/tasks_from_layouts.yaml" \
    sampler.batch_size=1 \
    save_traj=false
```

Notes:
- `layout_manifest.jsonl` is written automatically to Stage-1 `out_dir` when `export_layout_manifest=true`.
- Prefer `--motif-source viz_chains` when Stage-1 used `output_motif_chains=true`; this derives motifs from explicit non-scaffold chains in `samples/`.
- By default, the task builder excludes chain `V` in `viz_chains` mode (`--viz-exclude-chains V`) because it is used for visual aggregator tokens.
- Keep Stage-1 and Stage-2 in separate git worktrees (modified branch vs baseline `main`).
- The task builder validates manifest entries before writing Stage-2 YAML.

### Eval metadata vs Stage-2 conditioning metadata

Stage-1/Stage-2 runs mix two PDB views and two index schemes. Do not reuse the same fields for both.

| Category | Fields | Used by |
|----------|--------|---------|
| **Eval / downstream** | `fixed_seq_*`, `motif_reference_residues`, `motif_bb_*`, `rmsd_*`, `source_residue_keys` (scaffold `output_chain`) | ProteinMPNN, ESMFold, RMSD, `group_rmsd.csv` |
| **Stage-2 conditioning only** | `stage2_input_pdb`, `stage2_pdb_contigs`, `stage2_motif_groups`, `stage2_sample_chain_name` | `build_stage2_tasks_from_layout_manifest.py`, Stage-2 sampling input |

Invariants (enforced in `nomadflow/benchmark/pmpnn_mapping.py` and `PDBWriter` in `lmod.py`):

- `motif_idx` yields **scaffold-relative** residue numbers (as on chain `sample_chain_name`, usually `A`).
- Viz PDB chain labels from `asym_id` (e.g. motif chain `B` when `output_motif_chains=true`) are for visualization and Stage-2 **conditioning**, not for PMPNN/RMSD.
- Downstream parses `samples_eval/` (single scaffold chain). `fixed_seq_chain` and `pmpnn_fixed_pos_dict.jsonl` must use `sample_chain_name`, not viz motif chain IDs.
- Stage-2 task YAML may carry Stage-1 eval fields (`motif_reference_residues` on chain A); `source_layout_id` prevents overwriting them at write time.

Common regression symptom: `fixed_seq_chain` = `B` with indices like `85` while chain B in the PDB has only 23 residues → ProteinMPNN `IndexError`. Tests: `tests/test_pmpnn_mapping.py`, `tests/test_layout_manifest_pipeline.py`.

Quick start (recommended wrapper script):
```bash
source "${REPO_ROOT}/env_vars.sh"

./scripts/run_two_stage_motif_pipeline.sh \
    --run-id "<stage1_mlflow_run_id>" \
    --config configs/sampling/protpardelle_motif_scaffolding/config_multi_motif_se3_catalytic_tlock_0.5.yaml
```

Custom run name and Stage-1-only benchmark:
```bash
./scripts/run_two_stage_motif_pipeline.sh \
    --run-id "<stage1_mlflow_run_id>" \
    --config configs/sampling/protpardelle_motif_scaffolding/config_multi_motif_se3_catalytic_tlock_0.5.yaml \
    --only stage1

./scripts/run_two_stage_motif_pipeline.sh \
    --config configs/sampling/protpardelle_motif_scaffolding/config_multi_motif_se3_catalytic_tlock_0.5.yaml \
    --only eval --eval-target stage1
```

Manual step-by-step (same defaults as the wrapper; set `CUDA_VISIBLE_DEVICES` if you need to restrict GPUs):
```bash
source "${REPO_ROOT}/env_vars.sh"

export STAGE1_RUN="${REPO_ROOT}/article_results/sampling_runs/stage1_layout__multi_motif_se3_catalytic_tlock_0.5__best__$(date +%Y%m%d)"
export STAGE2_RUN="${REPO_ROOT}/article_results/sampling_runs/stage2_from_layout__multi_motif_se3_catalytic_tlock_0.5__pretrained__$(date +%Y%m%d)"
export STAGE2_TASKS_YAML="${STAGE2_RUN}/tasks_from_layouts.yaml"
export STAGE2_ANALYSIS="${REPO_ROOT}/article_results/analysis_runs/$(basename "${STAGE2_RUN}")"

python scripts/sample_from_mlflow_run.py \
    --run-id "<stage1_mlflow_run_id>" \
    --tasks-yaml "${REPO_ROOT}/configs/sampling/protpardelle_motif_scaffolding/config_multi_motif_se3_catalytic_tlock_0.5.yaml" \
    --out-dir "${STAGE1_RUN}" \
    --checkpoint best \
    --sample-arg sampler.batch_size=1 \
    --sample-arg save_traj=false \
    --sample-arg export_layout_manifest=true \
    --sample-arg output_motif_chains=true

python scripts/analysis/build_stage2_tasks_from_layout_manifest.py \
    --layout-manifest "${STAGE1_RUN}/layout_manifest.jsonl" \
    --output-yaml "${STAGE2_TASKS_YAML}" \
    --stage2-num-samples 1 \
    --total-length sample_length \
    --motif-source viz_chains \
    --include-motif-groups

python "${STAGE2_REPO_ROOT}/sample.py" \
    model_dir="${STAGE2_MODEL_DIR}" \
    checkpoint_path="${STAGE2_BASELINE_CKPT}" \
    out_dir="${STAGE2_RUN}" \
    sampler.tasks_yaml="${STAGE2_TASKS_YAML}" \
    sampler.batch_size=1 \
    save_traj=false

bash scripts/submit_scripts/eval/submit_article_motif_benchmark.sh \
    "${STAGE2_RUN}" \
    "${STAGE2_ANALYSIS}"
```

## Baseline vanilla sampling (ordinary fixation + group labels)

Use this when you want **pretrained ProteinZen-main** with its usual contig fixation
(no `multi_motif_se3` / `motif_rigid_mode`), but still need per-group metrics
(`group_rmsd.csv`, `multimotif_metrics.csv`).

1. Put `motif_groups` on the tasks YAML **without** multi-motif SE3 flags
   (see `configs/sampling/protpardelle_motif_scaffolding/config.yaml`).
   Sampling conditioning ignores `motif_groups`; it is annotation-only.
2. Sample from `STAGE2_REPO_ROOT`. Patched `PDBWriter` in main annotates
   `samples_metadata.json` with `source_residue_keys` / `motif_groups` / `native_pdb`
   when the copied `config.yaml` contains `motif_groups`.
3. Run the article eval chain from this repo.

Quick start:
```bash
source "${REPO_ROOT}/env_vars.sh"

./scripts/run_baseline_motif_pipeline.sh \
    --config configs/sampling/protpardelle_motif_scaffolding/config.yaml
```

Stages:
```bash
# sample only (writes under article_results/sampling_runs/original_proteinzen__...)
./scripts/run_baseline_motif_pipeline.sh --only sample

# backfill group labels on an existing / older run (idempotent)
./scripts/run_baseline_motif_pipeline.sh \
    --out-dir article_results/sampling_runs/<existing_run> \
    --only inject

# eval only
./scripts/run_baseline_motif_pipeline.sh \
    --out-dir article_results/sampling_runs/<existing_run> \
    --only eval
```

Manual equivalent (set `CUDA_VISIBLE_DEVICES` if you need to restrict GPUs):
```bash
source "${REPO_ROOT}/env_vars.sh"
export OUT_DIR="${REPO_ROOT}/article_results/sampling_runs/original_proteinzen__config__pretrained__$(date +%Y%m%d)"

(
  cd "${STAGE2_REPO_ROOT}"
  python sample.py \
      model_dir="${STAGE2_MODEL_DIR}" \
      checkpoint_path="${STAGE2_BASELINE_CKPT}" \
      out_dir="${OUT_DIR}" \
      sampler.tasks_yaml="${REPO_ROOT}/configs/sampling/protpardelle_motif_scaffolding/config.yaml" \
      sampler.batch_size=1 \
      save_traj=false
)

# Optional backfill if sampling predates main annotation, or config lacked motif_groups:
python scripts/analysis/inject_motif_group_metadata.py \
    --sampling-run "${OUT_DIR}" \
    --tasks-yaml "${REPO_ROOT}/configs/sampling/protpardelle_motif_scaffolding/config.yaml" \
    --repo-root "${REPO_ROOT}"

bash scripts/submit_scripts/eval/submit_article_motif_benchmark.sh \
    "${OUT_DIR}" \
    "${REPO_ROOT}/article_results/analysis_runs/$(basename "${OUT_DIR}")"
```

Invariant: expanding `motif_groups` in group/segment order must match the order of
`fixed_seq_res_idx` / `pdb_contigs` (length mismatch raises at annotate time).

## Benchmark Keys

- `protpardelle_m1_unindexed`
- `protpardelle_m2plus_unindexed` (registered as `protpardelle_motif_multi` in
  `mlflow_sampling.BENCHMARK_TASKS`; tasks YAML is WIP — `motif_groups` not yet
  audited)
- `protpardelle_single_rigid_group`
- `protpardelle_indexed_sanity`
- `protpardelle_tlock_0_3`
- `protpardelle_tlock_0_5`
- `protpardelle_tlock_0_7`
- `protpardelle_tlock_off`

## Tracking Rules

- Commit manifests, README files, and final figure sources.
- Do not commit raw PDB samples, ESMFold outputs, Foldseek temp folders, MLflow
  run copies, or model checkpoints.
- Every sampled run should have a `manifest.yaml` describing model checkpoint,
  tasks YAML, sampling budget, seed policy, ProteinMPNN settings, and ESMFold
  settings.
