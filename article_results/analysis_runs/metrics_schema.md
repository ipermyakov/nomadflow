# Analysis Run Metrics Schema

This document describes the CSV/JSON artifacts written by
`scripts/analysis/run_motif_benchmark_analysis.py` into
`article_results/analysis_runs/<run_id>/`.

## metrics.csv

Long-format per-sample metrics from ESMFold downstream evaluation.

| Column | Description |
| --- | --- |
| `protocol` | `one_shot` (design self-consistency) or `pmpnn_esmfold` |
| `name` | Sample id (basename of generated PDB) |
| `task` | Protpardelle task id (PDB code) |
| `sample` | PMPNN sequence index (PMPNN protocol only) |
| `motif_all_atom_rmsd` | Motif heavy-atom RMSD vs design |
| `motif_ca_rmsd` | Motif Cα RMSD vs design |
| `global_all_atom_rmsd` | Global heavy-atom RMSD (one-shot only) |
| `global_bb_rmsd` | Global backbone RMSD (PMPNN protocol) |
| `plddt`, `ptm` | ESMFold confidence metrics |
| `pass` | Unified success criteria from `nomadflow/benchmark/success_criteria.py` (ProteinZen paper §3.2.1: motif HA < 2 Å, motif Cα < 1 Å, global < 2 Å) |

## multimotif_metrics.csv

Long-format geometry metrics comparing generated samples to native PDBs via
`source_residue_keys` metadata.

| Column | Description |
| --- | --- |
| `record_type` | `group`, `intergroup`, or `angle` |
| `name`, `task` | Sample and task ids |
| `group_id` | Motif rigid group id (`record_type=group`) |
| `group_size` | Number of motif residues in the group |
| `motif_ca_rmsd`, `motif_ha_rmsd` | Per-group recovery vs native |
| `group_i`, `group_j` | Group pair (`record_type=intergroup`) |
| `native_com_distance`, `sample_com_distance`, `com_distance_error` | Inter-group COM distances |
| `apex_group_id`, `group_a`, `group_b` | Angle triple (`record_type=angle`) |
| `native_angle_deg`, `sample_angle_deg`, `angle_error_deg` | Inter-group angle errors |
| `num_motif_groups` | Number of motif groups in the task (`m`) |
| `native_min_intergroup_dist`, `native_max_intergroup_dist`, `native_mean_intergroup_dist` | Native geometry slice metadata |

## multimotif_slices.summary.csv

Binned aggregates over `multimotif_metrics.csv` group records:

| Column | Description |
| --- | --- |
| `slice_type` | `num_motif_groups`, `group_size`, or `native_dist_bin` |
| `slice_value` | Bin value |
| `num_group_records` | Number of per-group metric rows |
| `group_pass_rate` | Fraction with motif HA < 2 Å and Cα < 1 Å vs native (Cα used where BB is unavailable) |
| `mean_motif_ha_rmsd`, `mean_motif_ca_rmsd` | Mean per-group RMSDs in the bin |

## task_success.csv

Per-task success summary.

| Column | Description |
| --- | --- |
| `task` | Protpardelle task id |
| `num_samples` | Samples for the task in the run |
| `one_shot_success_count`, `one_shot_success_rate` | One-shot passes |
| `pmpnn_success_count`, `pmpnn_success_rate` | Legacy pooled PMPNN+ESMFold passes (diagnostic; **not** the paper All/rate) |
| `pmpnn_final_count`, `pmpnn_final_rate` | Publication PMPNN AND-success: **designs** with ≥1 sequence that passed pooled ∧ all groups. Rate = `pmpnn_final_count / num_samples`. Paper table `build_paper_style_table(..., protocol="pmpnn_esmfold")` uses these for All/rate. Unique is Foldseek on the same AND-success design set. |
| `any_one_shot`, `any_pmpnn` | Whether the task has at least one passing sample |

## unique_success.json

Foldseek unique-success cluster counts per task (TM threshold from
`FOLDSEEK_UNIQUE_TM_THRESHOLD`, default 0.5).

```json
{
  "foldseek_tm_threshold": 0.5,
  "one_shot": {"1PRW": 3},
  "pmpnn_esmfold": {"1PRW": 2},
  "skipped": false
}
```

## analysis_manifest.json

Provenance metadata: input paths, row counts, timestamp, git commit.

## Criteria v2 (`<run_id>__criteria_v2/`)

Publication artifacts are written to a **new** directory so legacy
`article_results/analysis_runs/<run_id>/` is never overwritten.

`CRITERIA_VERSION = v2_and_groups`. One-shot publication success is the
conjunction of pooled legacy criteria **and** every motif group
(fold vs generated design): HA < 2.0 Å, Cα < 1.0 Å, global HA < 2.0 Å
(strict `<`). Missing group rows fail `pass_all_groups` / `pass_final`.

### metrics.csv (v2 extra columns)

| Column | Description |
| --- | --- |
| `pass` / `pass_pooled_legacy` | Legacy pooled criteria (`one_shot_pass` / `pmpnn_esmfold_pass`) |
| `pass_all_groups` | Every group HA < 2.0 Å and Cα/BB < 1.0 Å vs the generated design (paper backbone cut is Cα) |
| `pass_final` | `pass_pooled_legacy AND pass_all_groups` (publication one-shot flag) |
| `group_<id>_ha`, `group_<id>_bb` | Wide per-group fold-vs-design RMSDs |
| `num_groups_scored`, `num_groups_expected` | Group rows present vs annotated count |

### task_success.csv (v2 extra columns)

| Column | Description |
| --- | --- |
| `one_shot_success_count` / `one_shot_success_rate` | Publication AND-success |
| `one_shot_pooled_legacy_count` / `one_shot_pooled_legacy_rate` | Diagnostic pooled-only counts |
| `one_shot_all_groups_count` | Samples where every group independently passes |
| `pmpnn_final_count` / `pmpnn_final_rate` | Publication PMPNN AND-success **design** counts (`nunique(name)` among `pass_final` sequences). Paper All/rate. |
| `pmpnn_success_count` / `pmpnn_success_rate` | Legacy pooled design counts (`nunique(name)` among pooled-passing sequences). Diagnostic only; do not mix with Unique. |

### unique_success_all_groups.json

Foldseek unique-success on **AND-success design backbones** (TM 0.5,
per-task, no task-sum). One-shot and PMPNN each get a protocol object.
`unique_count <= success_count` (`one_shot_success_count` /
`pmpnn_final_count`). PMPNN is one design PDB per passing `name`.

```json
{
  "criteria_version": "v2_and_groups",
  "foldseek_tm_threshold": 0.5,
  "foldseek_input": "and_success_design_backbones",
  "one_shot": {"1PRW": 3},
  "pmpnn_esmfold": {"1PRW": 2},
  "skipped": false
}
```

Legacy `unique_success.json` is not rewritten.

### group_fold_metrics.csv

Copy of `sampling_run/esmfold/group_rmsd.csv` after parent-task remap
(fold vs generated design HA/BB).

### layout_pair_metrics.csv / layout_summaries.json

Layout metrics are **mobility diagnostics**, not part of AND-success.

Each group's frame is the Kabsch rigid transform of all matched backbone N/CA/C
atoms **onto the native group** (origin = observed CoM). Native / design / fold
stay in their own lab frames; relative pose `T_ij = T_i⁻¹ ∘ T_j` is invariant to
a global SE(3) of all groups (one-rigid → relative poses unchanged).

Stored raw `T_ij` (needed for pairwise distances / MDS): `{native,design,fold}_tij_tx/ty/tz`
and scalar-first quaternion `{native,design,fold}_tij_qw/qx/qy/qz` (Hamilton, `qw >= 0`;
`q` and `-q` are the same pose). Deltas `excursion_*` / `retention_*` are kept.

Inter-sample distance between layouts A and B: per group pair `||t_gh(A)-t_gh(B)||`
and SO(3) geodesic; then **RMS over pairs** (equal pair weight, not residue count).
Task summaries: **median pairwise layout translation/rotation distance** over design
pairs **within a task**. Bootstrap resamples **designs** (stage-2: source layout if
several designs share one), never the precomputed `N(N-1)/2` pair rows.
`< 2` designs → pairwise undefined; `n < 5` → `weak_n`, show points and `n`.

Slices: all generated designs, and `pass_final` (main claim). `layout_frame`:
`kabsch_backbone_nca_c`. Stage-2 `task` remapped to parent motif before grouping.

Fig C (notebook): excursion × pairwise diversity, not boxplots of dependent pairs.
Retention is a separate per-task whisker. MDS supplement embeds **all** designs of
a task once; successes highlighted. Combined trans+rot MDS only via
`D² = (d_trans/τ_trans)² + (d_rot/τ_rot)²` with documented taus (default 2 Å, 15 deg).

### layout_pmpnn_pair_metrics.csv

Same pair geometry, but **retention is design vs each passing PMPNN ESMFold**
(`pass_final` PMPNN sequences). Sequence-level file: one row per
`(design, pmpnn_sample, group pair)` for supplement. Publication PMPNN layout unit is
**one value per design**: `aggregate_pmpnn_layout_to_design` takes median `retention_*`
over AND-passing sequences within `(run_id, name, group_i, group_j)`; excursion is
taken once. `fold_source=pmpnn_esmfold`. Fold poses in this file are the PMPNN fold.

### layout_coverage.csv

Per-design coverage of layout scoring (and, when the in-place recast job runs, group
RMSD completeness). `reason=ok` means pair rows were written. `eval_error` is a
**metadata/mapping failure**, not a model fail: stage2 vs stage1 residue-count
mismatch; missing `layout_id`; **ambiguous** `layout_id` when the same Stage-1
sample name appears in more than one task (disambiguate by `stage1_task_name`,
never last-write-wins); native residue missing. Current `pass_final` is
**not** rewritten for these ids (would change the Foldseek AND-success set); they
still lack grouped keys so AND-success treats missing groups as fail. Interpret them
via this coverage table, not as motif failure. `missing_residue_in_design` is an
honest fail (residues absent in the generated structure).

Do not silently truncate residue lists. Do not resample.

### analysis_manifest.json (v2 extra fields)

`criteria_version`, `thresholds`, `reference_frame`, `foldseek_input`,
`layout_frame`, `layout_manifest`, `git_commit`, `sampling_run`.
