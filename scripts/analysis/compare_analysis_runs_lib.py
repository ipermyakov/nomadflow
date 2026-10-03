from __future__ import annotations

import hashlib
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from nomadflow.benchmark.analysis_inputs import (
    PREFERRED_CRITERIA_VERSIONS,
    criteria_dir_name,
    criteria_version_from_run_id,
    parent_motif_task_id,
    remap_path_column,
    resolve_sampling_run_path,
    strip_criteria_version_suffix,
)
from nomadflow.benchmark.layout_pose_metrics import (
    bootstrap_median_of_designs,
    classical_mds,
    layout_distance_matrix,
    pairwise_layout_stats,
    relatives_from_pair_df,
    rms_over_values,
)

CRITERIA_V2_SUFFIX = "__criteria_v2"
CRITERIA_V3_SUFFIX = "__criteria_v3"


def _normalize_criteria_version(criteria: str | None) -> str | None:
    if criteria is None:
        return None
    ver = str(criteria).strip()
    if ver.startswith("__criteria_"):
        ver = ver.removeprefix("__criteria_")
    if ver and not ver.startswith("v"):
        ver = f"v{ver}"
    if ver not in PREFERRED_CRITERIA_VERSIONS:
        raise ValueError(
            f"Unknown criteria version {criteria!r}; expected one of "
            f"{PREFERRED_CRITERIA_VERSIONS} or None"
        )
    return ver


def _criteria_versions_to_try(criteria: str | None) -> tuple[str, ...]:
    pinned = _normalize_criteria_version(criteria)
    if pinned is None:
        return PREFERRED_CRITERIA_VERSIONS
    if pinned == "v3":
        return PREFERRED_CRITERIA_VERSIONS
    return (pinned,)


def _has_metrics(path: Path) -> bool:
    return path.is_dir() and (path / "metrics.csv").is_file()


def detect_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    if (start / "article_results").exists():
        return start
    for parent in start.parents:
        if (parent / "article_results").exists():
            return parent
    raise FileNotFoundError("Could not detect repository root containing article_results/")


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except EmptyDataError:
        return pd.DataFrame()


def discover_runs(
    analysis_root: Path,
    *,
    criteria: str | None = None,
) -> list[Path]:
    """Prefer versioned siblings (v3, then v2) over the unversioned folder."""
    analysis_root = Path(analysis_root)
    found = sorted(
        [p for p in analysis_root.iterdir() if p.is_dir() and (p / "metrics.csv").exists()]
    )
    versions = _criteria_versions_to_try(criteria)
    rank = {ver: index for index, ver in enumerate(versions)}
    best: dict[str, Path] = {}
    best_rank: dict[str, int] = {}
    fallback_rank = len(versions)
    for path in found:
        base = strip_criteria_version_suffix(path.name)
        version = criteria_version_from_run_id(path.name)
        if version is None:
            this_rank = fallback_rank
        elif version in rank:
            this_rank = rank[version]
        else:
            continue
        prev_rank = best_rank.get(base)
        if prev_rank is None or this_rank < prev_rank:
            best[base] = path
            best_rank[base] = this_rank
    return sorted(best.values(), key=lambda path: path.name)


def resolve_analysis_run_dir(
    analysis_root: Path,
    run_id: str,
    *,
    criteria: str | None = None,
) -> Path:
    """Resolve an analysis run folder, preferring ``__criteria_v3`` then ``__criteria_v2``.

    ``criteria="v2"`` skips v3. ``criteria="v3"`` still falls back to v2, then the
    unversioned sibling, when the newer folder has no ``metrics.csv``.
    """
    analysis_root = Path(analysis_root)
    base = strip_criteria_version_suffix(run_id)
    candidates: list[Path] = []
    seen: set[Path] = set()

    def _add(path: Path) -> None:
        resolved = path
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(resolved)

    for version in _criteria_versions_to_try(criteria):
        _add(analysis_root / criteria_dir_name(base, version))
    _add(analysis_root / run_id)
    _add(analysis_root / base)
    for candidate in candidates:
        if _has_metrics(candidate):
            return candidate
    raise FileNotFoundError(f"Missing analysis run (no metrics.csv): {run_id}")


def parse_run_id(run_dir: Path) -> dict[str, str]:
    parts = run_dir.name.split("__")
    if len(parts) >= 4:
        return {
            "run_id": run_dir.name,
            "model": parts[0],
            "benchmark": "__".join(parts[1:-2]),
            "checkpoint": parts[-2],
            "date": parts[-1],
        }
    return {
        "run_id": run_dir.name,
        "model": run_dir.name,
        "benchmark": "",
        "checkpoint": "",
        "date": "",
    }


def load_unique_success(path: Path) -> dict:
    if not path.exists():
        return {"skipped": True, "one_shot": {}, "pmpnn_esmfold": {}}
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    payload.setdefault("one_shot", {})
    payload.setdefault("pmpnn_esmfold", {})
    return payload


def load_unique_success_for_run(run_dir: Path) -> dict:
    """Prefer publication AND-success JSON; fall back to legacy unique_success.json."""
    and_path = run_dir / "unique_success_all_groups.json"
    if and_path.exists():
        return load_unique_success(and_path)
    return load_unique_success(run_dir / "unique_success.json")


def load_analysis_manifest(run_dir: Path) -> dict:
    path = run_dir / "analysis_manifest.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def infer_group_rmsd_protocol(path: str | Path) -> str:
    """Classify group_rmsd rows as one_shot vs pmpnn_esmfold from folded PDB path."""
    path_str = str(path)
    if "model_name=v_48_020" in path_str:
        return "one_shot"
    return "pmpnn_esmfold"


def load_group_rmsd(run_dir: Path, run_id: str, repo_root: Path | None = None) -> pd.DataFrame:
    """Load per-group ESMFold RMSD from the linked sampling run, if present."""
    manifest = load_analysis_manifest(run_dir)
    sampling_run = resolve_sampling_run_path(
        manifest_sampling_run=manifest.get("sampling_run"),
        analysis_run_dir=run_dir,
        repo_root=repo_root,
    )
    if sampling_run is None:
        return pd.DataFrame()
    path = Path(sampling_run) / "esmfold" / "group_rmsd.csv"
    local_copy = run_dir / "group_fold_metrics.csv"
    if not path.is_file() and local_copy.is_file():
        path = local_copy
    df = safe_read_csv(path)
    if df.empty:
        return df
    df = df.copy()
    df["run_id"] = run_id
    if "path" in df.columns:
        df["protocol"] = df["path"].map(infer_group_rmsd_protocol)
        if repo_root is not None:
            df["path"] = remap_path_column(df["path"], repo_root=repo_root)
    return df


def load_run_tables(run_dir: Path, repo_root: Path | None = None) -> dict:
    meta = parse_run_id(run_dir)
    if repo_root is None:
        try:
            repo_root = detect_repo_root(run_dir)
        except FileNotFoundError:
            repo_root = None
    metrics = safe_read_csv(run_dir / "metrics.csv")
    metrics["run_id"] = meta["run_id"]

    task_success = safe_read_csv(run_dir / "task_success.csv")
    if not task_success.empty:
        task_success["run_id"] = meta["run_id"]

    multimotif = safe_read_csv(run_dir / "multimotif_metrics.csv")
    if not multimotif.empty:
        multimotif["run_id"] = meta["run_id"]

    multimotif_slices = safe_read_csv(run_dir / "multimotif_slices.summary.csv")
    if not multimotif_slices.empty:
        multimotif_slices["run_id"] = meta["run_id"]

    group_rmsd = load_group_rmsd(run_dir, meta["run_id"], repo_root=repo_root)
    unique_success = load_unique_success_for_run(run_dir)

    layout_pairs = safe_read_csv(run_dir / "layout_pair_metrics.csv")
    if not layout_pairs.empty:
        layout_pairs["run_id"] = meta["run_id"]
        layout_pairs = remap_layout_pair_tasks(layout_pairs)
        layout_pairs = attach_layout_pmpnn_pass(layout_pairs, metrics)

    layout_pmpnn_pairs = safe_read_csv(run_dir / "layout_pmpnn_pair_metrics.csv")
    if not layout_pmpnn_pairs.empty:
        layout_pmpnn_pairs["run_id"] = meta["run_id"]
        layout_pmpnn_pairs = remap_layout_pair_tasks(layout_pmpnn_pairs)

    layout_coverage = safe_read_csv(run_dir / "layout_coverage.csv")
    if not layout_coverage.empty:
        layout_coverage["run_id"] = meta["run_id"]
        layout_coverage = remap_layout_pair_tasks(layout_coverage)

    layout_mobility = load_layout_mobility_table(
        run_dir, meta["run_id"], "layout_mobility.csv"
    )
    layout_pmpnn_mobility = load_layout_mobility_table(
        run_dir, meta["run_id"], "layout_pmpnn_mobility.csv"
    )

    return {
        **meta,
        "run_dir": run_dir,
        "metrics": metrics,
        "task_success": task_success,
        "multimotif": multimotif,
        "multimotif_slices": multimotif_slices,
        "group_rmsd": group_rmsd,
        "layout_pairs": layout_pairs,
        "layout_pmpnn_pairs": layout_pmpnn_pairs,
        "layout_coverage": layout_coverage,
        "layout_mobility": layout_mobility,
        "layout_pmpnn_mobility": layout_pmpnn_mobility,
        "unique_success": unique_success,
    }


def build_context(
    repo_root: Path | None = None,
    run_ids: list[str] | None = None,
    *,
    criteria: str | None = None,
) -> dict:
    """Load analysis tables for comparison.

    Args:
        repo_root: Repository root (auto-detected if omitted).
        run_ids: If set, keep only these analysis_runs/<run_id> folders.
            Order follows ``run_ids``. Missing ids raise ``FileNotFoundError``.
            If ``None``, include every run that has ``metrics.csv``.
        criteria: ``"v2"``, ``"v3"``, or ``None`` (prefer v3 then v2 then
            unversioned). Folder names may be unsuffixed; the resolver adds
            ``__criteria_v*`` when those directories exist.
    """
    repo_root = detect_repo_root(repo_root)
    analysis_root = repo_root / "article_results" / "analysis_runs"

    if run_ids is None:
        run_dirs = discover_runs(analysis_root, criteria=criteria)
    else:
        run_dirs = []
        missing = []
        for run_id in run_ids:
            try:
                run_dir = resolve_analysis_run_dir(
                    analysis_root, run_id, criteria=criteria
                )
            except FileNotFoundError:
                missing.append(run_id)
                continue
            run_dirs.append(run_dir)
        if missing:
            available = [p.name for p in discover_runs(analysis_root, criteria=criteria)]
            raise FileNotFoundError(
                "Missing analysis runs (no metrics.csv): "
                + ", ".join(missing)
                + f". Available: {available}"
            )

    runs = [load_run_tables(run_dir, repo_root=repo_root) for run_dir in run_dirs]
    return assemble_context(
        repo_root=repo_root,
        analysis_root=analysis_root,
        run_dirs=run_dirs,
        runs=runs,
    )


def load_layout_mobility_table(
    run_dir: Path,
    run_id: str,
    filename: str = "layout_mobility.csv",
) -> pd.DataFrame:
    """Load a precomputed mobility summary. Missing or empty files are empty frames."""
    df = safe_read_csv(Path(run_dir) / filename)
    if df.empty:
        return df
    out = df.copy()
    out["run_id"] = run_id
    if "task" in out.columns:
        out = remap_layout_pair_tasks(out)
    return out


def assemble_context(
    *,
    repo_root: Path,
    analysis_root: Path,
    run_dirs: list[Path],
    runs: list[dict],
) -> dict:
    """Rebuild concatenated tables and run_summary from a list of run dicts."""
    metrics = pd.concat([r["metrics"] for r in runs], ignore_index=True) if runs else pd.DataFrame()
    task_success_parts = [r["task_success"] for r in runs if not r["task_success"].empty]
    task_success = pd.concat(task_success_parts, ignore_index=True) if task_success_parts else pd.DataFrame()

    multimotif_parts = [r["multimotif"] for r in runs if not r["multimotif"].empty]
    multimotif = pd.concat(multimotif_parts, ignore_index=True) if multimotif_parts else pd.DataFrame()

    group_rmsd_parts = [r["group_rmsd"] for r in runs if not r["group_rmsd"].empty]
    group_rmsd = pd.concat(group_rmsd_parts, ignore_index=True) if group_rmsd_parts else pd.DataFrame()

    layout_parts = [
        r["layout_pairs"]
        for r in runs
        if r.get("layout_pairs") is not None and not r["layout_pairs"].empty
    ]
    layout_pairs = pd.concat(layout_parts, ignore_index=True) if layout_parts else pd.DataFrame()

    layout_pmpnn_parts = [
        r["layout_pmpnn_pairs"]
        for r in runs
        if r.get("layout_pmpnn_pairs") is not None and not r["layout_pmpnn_pairs"].empty
    ]
    layout_pmpnn_pairs = (
        pd.concat(layout_pmpnn_parts, ignore_index=True) if layout_pmpnn_parts else pd.DataFrame()
    )

    layout_coverage_parts = [
        r["layout_coverage"]
        for r in runs
        if r.get("layout_coverage") is not None and not r["layout_coverage"].empty
    ]
    layout_coverage = (
        pd.concat(layout_coverage_parts, ignore_index=True)
        if layout_coverage_parts
        else pd.DataFrame()
    )

    layout_mobility_parts = [
        r["layout_mobility"]
        for r in runs
        if r.get("layout_mobility") is not None and not r["layout_mobility"].empty
    ]
    layout_mobility = (
        pd.concat(layout_mobility_parts, ignore_index=True)
        if layout_mobility_parts
        else pd.DataFrame()
    )
    layout_pmpnn_mobility_parts = [
        r["layout_pmpnn_mobility"]
        for r in runs
        if r.get("layout_pmpnn_mobility") is not None and not r["layout_pmpnn_mobility"].empty
    ]
    layout_pmpnn_mobility = (
        pd.concat(layout_pmpnn_mobility_parts, ignore_index=True)
        if layout_pmpnn_mobility_parts
        else pd.DataFrame()
    )

    run_summary = pd.DataFrame([summarize_run(run) for run in runs]) if runs else pd.DataFrame()
    return {
        "repo_root": repo_root,
        "analysis_root": analysis_root,
        "run_dirs": run_dirs,
        "runs": runs,
        "metrics": metrics,
        "task_success": task_success,
        "multimotif": multimotif,
        "group_rmsd": group_rmsd,
        "layout_pairs": layout_pairs,
        "layout_pmpnn_pairs": layout_pmpnn_pairs,
        "layout_coverage": layout_coverage,
        "layout_mobility": layout_mobility,
        "layout_pmpnn_mobility": layout_pmpnn_mobility,
        "run_summary": run_summary,
    }


# Sampling-split partitions used for tlock05 se3 vs full. The split is only
# for job size: together they are one multi-motif campaign.
STRUCTURAL_LARGE_MOTIF_TASKS = frozenset(
    {
        "1BCF",
        "1BCF_2grp",
        "1PRW",
        "2KL8",
        "4JHW",
        "5IUS",
        "5YUI",
        "5YUI_2grp",
        "6VW1",
    }
)
CATALYTIC_PLUS_STRUCTURAL_TASKS = frozenset(
    {
        "1QJG_ext10",
        "1QJG_ext16",
        "1QJG_ext2x10",
        "5AOU_ext10",
        "5AOU_ext20",
        "5AOU_ext2x10",
        "5AOU_QUAD_ext10",
        "5AOU_QUAD_ext20",
        "5AOU_QUAD_ext2x10",
        "7K4V_ext10",
        "7K4V_ext20",
        "7K4V_ext2x10",
    }
)
PARTITION_LABELS = {
    "structural_large": "structural (large motifs)",
    "catalytic_plus_structural": "catalytic + structural",
    "other": "other",
}
PARTITION_SORT = {
    "structural (large motifs)": 0,
    "catalytic + structural": 1,
    "other": 2,
    "TOTAL": 3,
}
TLOCK05_SE3_FULL_SOURCE_IDS = (
    "0c62dc8a__tlock05_se3_stage1",
    "0c62dc8a__tlock05_full_stage1",
    "0c62dc8a__tlock05_se3_stage2",
    "0c62dc8a__tlock05_full_stage2",
)
TLOCK05_CAMPAIGNS = {
    "tlock05_stage1": [
        "0c62dc8a__tlock05_se3_stage1",
        "0c62dc8a__tlock05_full_stage1",
    ],
    "tlock05_stage2": [
        "0c62dc8a__tlock05_se3_stage2",
        "0c62dc8a__tlock05_full_stage2",
    ],
    "one_motif_proteinzen": ["one_motif_proteinzen"],
}
LAYOUT_MDS_MATCHED_FILENAME = "layout_mds_matched.csv"
MDS_TAU_TRANS_DEFAULT = 2.0
MDS_TAU_ROT_DEFAULT = 15.0


def task_partition(task: str) -> str:
    name = str(task)
    if name in STRUCTURAL_LARGE_MOTIF_TASKS:
        return "structural_large"
    if name in CATALYTIC_PLUS_STRUCTURAL_TASKS:
        return "catalytic_plus_structural"
    return "other"


def _filter_task_frame(df: pd.DataFrame, tasks: set[str]) -> pd.DataFrame:
    if df is None or df.empty or "task" not in df.columns:
        return df if df is not None else pd.DataFrame()
    return df[df["task"].astype(str).isin(tasks)].copy()


def filter_unique_success(unique: dict, tasks: set[str]) -> dict:
    tasks = set(tasks)
    filtered = dict(unique)
    for protocol in ("one_shot", "pmpnn_esmfold"):
        counts = unique.get(protocol, {}) or {}
        filtered[protocol] = {k: v for k, v in counts.items() if str(k) in tasks}
    return filtered


def merge_unique_success(parts: list[dict]) -> dict:
    if not parts:
        return {"skipped": True, "one_shot": {}, "pmpnn_esmfold": {}}
    merged = {
        "skipped": all(bool(part.get("skipped")) for part in parts),
        "foldseek_tm_threshold": next(
            (part.get("foldseek_tm_threshold") for part in parts if "foldseek_tm_threshold" in part),
            None,
        ),
        "one_shot": {},
        "pmpnn_esmfold": {},
    }
    for part in parts:
        for protocol in ("one_shot", "pmpnn_esmfold"):
            for task, count in (part.get(protocol, {}) or {}).items():
                key = str(task)
                merged[protocol][key] = int(merged[protocol].get(key, 0)) + int(count)
    return merged


def _run_id_aliases(run_id: str) -> set[str]:
    return {str(run_id), strip_criteria_version_suffix(str(run_id))}


def _lookup_run(by_id: dict[str, dict], run_id: str) -> dict | None:
    if run_id in by_id:
        return by_id[run_id]
    return by_id.get(strip_criteria_version_suffix(run_id))


def tasks_in_runs(runs: list[dict], run_ids: list[str] | None = None) -> set[str]:
    wanted: set[str] | None = None
    if run_ids is not None:
        wanted = set()
        for run_id in run_ids:
            wanted.update(_run_id_aliases(run_id))
    tasks: set[str] = set()
    for run in runs:
        if wanted is not None and _run_id_aliases(run["run_id"]).isdisjoint(wanted):
            continue
        ts = run["task_success"]
        if ts is not None and not ts.empty and "task" in ts.columns:
            tasks.update(ts["task"].astype(str))
            continue
        metrics = run["metrics"]
        if metrics is not None and not metrics.empty and "task" in metrics.columns:
            tasks.update(metrics["task"].astype(str))
    return tasks


def coverage_table(runs: list[dict], tasks: set[str]) -> pd.DataFrame:
    rows = []
    for task in sorted(
        tasks,
        key=lambda name: (PARTITION_SORT.get(PARTITION_LABELS[task_partition(name)], 99), name),
    ):
        row = {
            "task": task,
            "partition": PARTITION_LABELS[task_partition(task)],
        }
        for run in runs:
            present = False
            ts = run["task_success"]
            if ts is not None and not ts.empty and "task" in ts.columns:
                present = task in set(ts["task"].astype(str))
            row[run["run_id"]] = present
        rows.append(row)
    return pd.DataFrame(rows)


_LAYOUT_OK_REASON = "ok"
_LAYOUT_EVAL_ERROR_REASON = "eval_error"
_LAYOUT_MISSING_RESIDUE_REASON = "missing_residue_in_design"
_LAYOUT_NOT_MULTIGROUP_REASON = "not_multigroup"


def _task_success_row(run: dict, task: str) -> pd.Series | None:
    ts = run.get("task_success")
    if ts is None or ts.empty or "task" not in ts.columns:
        return None
    match = ts.loc[ts["task"].astype(str) == str(task)]
    if match.empty:
        return None
    return match.iloc[0]


def _pass_final_count(run: dict, task: str) -> int | None:
    ts_row = _task_success_row(run, task)
    if ts_row is not None and "one_shot_success_count" in ts_row.index:
        value = ts_row["one_shot_success_count"]
        if pd.notna(value):
            return int(value)
    metrics = run.get("metrics")
    if metrics is None or metrics.empty or "pass_final" not in metrics.columns:
        return None
    sub = metrics.loc[metrics["task"].astype(str) == str(task)]
    if "protocol" in sub.columns:
        sub = sub.loc[sub["protocol"].astype(str) == "one_shot"]
    if sub.empty:
        return 0
    return int(sub["pass_final"].fillna(False).astype(bool).sum())


def _group_rmsd_complete_count(coverage: pd.DataFrame) -> int | None:
    if coverage.empty:
        return 0
    if "has_all_group_rmsds" in coverage.columns:
        return int(coverage["has_all_group_rmsds"].fillna(False).astype(bool).sum())
    if "n_group_rmsd_ok" in coverage.columns and "num_groups_expected" in coverage.columns:
        expected = pd.to_numeric(coverage["num_groups_expected"], errors="coerce")
        got = pd.to_numeric(coverage["n_group_rmsd_ok"], errors="coerce")
        return int((got >= expected).fillna(False).sum())
    return None


def _eval_error_detail(coverage: pd.DataFrame) -> str:
    if coverage.empty:
        return ""
    subset = coverage
    if "reason" in coverage.columns:
        subset = coverage.loc[coverage["reason"].astype(str) == _LAYOUT_EVAL_ERROR_REASON]
    if subset.empty:
        return ""
    for column in ("eval_error", "detail"):
        if column not in subset.columns:
            continue
        values = (
            subset[column]
            .dropna()
            .astype(str)
            .map(str.strip)
        )
        values = values[values != ""]
        if values.empty:
            continue
        return str(values.value_counts().index[0])
    return _LAYOUT_EVAL_ERROR_REASON


def _coverage_note(
    *,
    task: str,
    in_campaign: bool,
    n_eval_error: int,
    n_missing_residue: int,
    eval_error_detail: str,
) -> str:
    notes: list[str] = []
    if not in_campaign:
        notes.append("tlock-only" if str(task).endswith("_2grp") else "absent")
    if n_eval_error:
        detail = eval_error_detail or "eval_error"
        notes.append(f"eval_error:{detail} (mapping, not model fail)")
    if n_missing_residue:
        notes.append("missing_residue_in_design (honest fail)")
    if str(task) == "4JHW" and (n_missing_residue or n_eval_error or not in_campaign):
        notes.append("4JHW")
    return "; ".join(notes)


def layout_design_coverage_table(runs: list[dict], tasks: set[str]) -> pd.DataFrame:
    """Per campaign × task counts of evaluable designs (layout / group RMSD).

    ``eval_error`` is a metadata/mapping failure, not a model fail.
    ``missing_residue_in_design`` is an honest generated-structure fail.
    """
    tasks = {str(t) for t in tasks}
    ordered_tasks = sorted(
        tasks,
        key=lambda name: (PARTITION_SORT.get(PARTITION_LABELS[task_partition(name)], 99), name),
    )
    rows = []
    for task in ordered_tasks:
        partition = PARTITION_LABELS[task_partition(task)]
        for run in runs:
            ts_row = _task_success_row(run, task)
            in_campaign = ts_row is not None
            n_expected = 0
            if in_campaign and "num_samples" in ts_row.index and pd.notna(ts_row["num_samples"]):
                n_expected = int(ts_row["num_samples"])

            coverage = run.get("layout_coverage", pd.DataFrame())
            if coverage is None or coverage.empty or "task" not in coverage.columns:
                coverage = pd.DataFrame()
            else:
                coverage = remap_layout_pair_tasks(coverage)
                coverage = coverage.loc[coverage["task"].astype(str) == task].copy()

            n_coverage = int(len(coverage))
            if not in_campaign:
                n_expected = 0
                n_coverage = 0
                coverage = pd.DataFrame()

            if n_expected == 0 and n_coverage and in_campaign:
                n_expected = n_coverage

            reasons = (
                coverage["reason"].fillna("").astype(str)
                if not coverage.empty and "reason" in coverage.columns
                else pd.Series(dtype=str)
            )
            n_layout_ok = int((reasons == _LAYOUT_OK_REASON).sum())
            n_eval_error = int((reasons == _LAYOUT_EVAL_ERROR_REASON).sum())
            n_missing_residue = int((reasons == _LAYOUT_MISSING_RESIDUE_REASON).sum())
            n_not_multigroup = int((reasons == _LAYOUT_NOT_MULTIGROUP_REASON).sum())
            n_other_skip = int(
                n_coverage
                - n_layout_ok
                - n_eval_error
                - n_missing_residue
                - n_not_multigroup
            )
            n_group_rmsd = _group_rmsd_complete_count(coverage) if in_campaign else 0
            n_pass_final = _pass_final_count(run, task) if in_campaign else None
            num_groups = None
            if not coverage.empty and "num_groups_expected" in coverage.columns:
                groups = pd.to_numeric(coverage["num_groups_expected"], errors="coerce")
                if groups.notna().any():
                    num_groups = int(groups.max())

            eval_detail = _eval_error_detail(coverage) if n_eval_error else ""
            layout_incomplete = bool(
                in_campaign
                and n_expected > 0
                and n_not_multigroup == 0
                and n_layout_ok < n_expected
            )
            rows.append(
                {
                    "task": task,
                    "partition": partition,
                    "run_id": run["run_id"],
                    "in_campaign": in_campaign,
                    "n_expected": n_expected,
                    "n_coverage": n_coverage,
                    "n_layout_ok": n_layout_ok,
                    "n_eval_error": n_eval_error,
                    "n_missing_residue": n_missing_residue,
                    "n_not_multigroup": n_not_multigroup,
                    "n_other_skip": n_other_skip,
                    "n_group_rmsd": n_group_rmsd if n_group_rmsd is not None else pd.NA,
                    "n_pass_final": n_pass_final if n_pass_final is not None else pd.NA,
                    "num_groups": num_groups if num_groups is not None else pd.NA,
                    "eval_error_detail": eval_detail,
                    "layout_incomplete": layout_incomplete,
                    "note": _coverage_note(
                        task=task,
                        in_campaign=in_campaign,
                        n_eval_error=n_eval_error,
                        n_missing_residue=n_missing_residue,
                        eval_error_detail=eval_detail,
                    ),
                }
            )
    return pd.DataFrame(rows)


def filter_runs_to_tasks(runs: list[dict], tasks: set[str]) -> list[dict]:
    tasks = {str(t) for t in tasks}
    filtered = []
    for run in runs:
        item = dict(run)
        item["metrics"] = _filter_task_frame(run["metrics"], tasks)
        item["task_success"] = _filter_task_frame(run["task_success"], tasks)
        item["multimotif"] = _filter_task_frame(run["multimotif"], tasks)
        item["group_rmsd"] = _filter_task_frame(run["group_rmsd"], tasks)
        slices = run.get("multimotif_slices", pd.DataFrame())
        item["multimotif_slices"] = _filter_task_frame(slices, tasks)
        item["layout_pairs"] = _filter_task_frame(run.get("layout_pairs", pd.DataFrame()), tasks)
        item["layout_pmpnn_pairs"] = _filter_task_frame(
            run.get("layout_pmpnn_pairs", pd.DataFrame()), tasks
        )
        item["layout_coverage"] = _filter_task_frame(
            run.get("layout_coverage", pd.DataFrame()), tasks
        )
        item["layout_mobility"] = _filter_task_frame(
            run.get("layout_mobility", pd.DataFrame()), tasks
        )
        item["layout_pmpnn_mobility"] = _filter_task_frame(
            run.get("layout_pmpnn_mobility", pd.DataFrame()), tasks
        )
        item["unique_success"] = filter_unique_success(run["unique_success"], tasks)
        filtered.append(item)
    return filtered


def _set_run_id(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    out = df.copy()
    out["run_id"] = run_id
    return out


def _concat_run_frames(parts: list[dict], key: str, run_id: str) -> pd.DataFrame:
    frames = [part[key] for part in parts if part.get(key) is not None and not part[key].empty]
    if not frames:
        return pd.DataFrame()
    return _set_run_id(pd.concat(frames, ignore_index=True), run_id)


def merge_runs_by_campaign(runs: list[dict], campaigns: dict[str, list[str]]) -> list[dict]:
    """Concatenate source analysis runs that were split only for sampling throughput.

    Campaign member ids may be unsuffixed or ``__criteria_v*``; lookup ignores the
    version suffix so v2/v3/legacy folders can be merged with the same map.
    """
    by_id: dict[str, dict] = {}
    for run in runs:
        by_id[run["run_id"]] = run
        by_id[strip_criteria_version_suffix(run["run_id"])] = run
    merged: list[dict] = []
    for campaign_id, members in campaigns.items():
        missing = [name for name in members if _lookup_run(by_id, name) is None]
        if missing:
            raise KeyError(f"Campaign {campaign_id} missing runs: {missing}")
        parts = [run for name in members if (run := _lookup_run(by_id, name)) is not None]
        task_success = _concat_run_frames(parts, "task_success", campaign_id)
        if not task_success.empty and task_success["task"].duplicated().any():
            dupes = sorted(task_success.loc[task_success["task"].duplicated(), "task"].astype(str).unique())
            raise ValueError(f"Overlapping tasks in campaign {campaign_id}: {dupes}")
        merged.append(
            {
                "run_id": campaign_id,
                "model": campaign_id,
                "benchmark": "",
                "checkpoint": "",
                "date": "",
                "run_dir": parts[0]["run_dir"],
                "metrics": _concat_run_frames(parts, "metrics", campaign_id),
                "task_success": task_success,
                "multimotif": _concat_run_frames(parts, "multimotif", campaign_id),
                "multimotif_slices": _concat_run_frames(parts, "multimotif_slices", campaign_id),
                "group_rmsd": _concat_run_frames(parts, "group_rmsd", campaign_id),
                "layout_pairs": _concat_run_frames(parts, "layout_pairs", campaign_id),
                "layout_pmpnn_pairs": _concat_run_frames(parts, "layout_pmpnn_pairs", campaign_id),
                "layout_coverage": _concat_run_frames(parts, "layout_coverage", campaign_id),
                "layout_mobility": _concat_run_frames(parts, "layout_mobility", campaign_id),
                "layout_pmpnn_mobility": _concat_run_frames(
                    parts, "layout_pmpnn_mobility", campaign_id
                ),
                "unique_success": merge_unique_success([part["unique_success"] for part in parts]),
            }
        )
    return merged


def attach_unique_counts(task_success: pd.DataFrame, runs: list[dict]) -> pd.DataFrame:
    if task_success.empty:
        return task_success
    lookup: dict[tuple[str, str, str], int] = {}
    skipped = set()
    for run in runs:
        unique = run["unique_success"]
        run_id = run["run_id"]
        if unique.get("skipped"):
            skipped.add(run_id)
            continue
        for protocol in ("one_shot", "pmpnn_esmfold"):
            for task, count in (unique.get(protocol, {}) or {}).items():
                lookup[(run_id, str(task), protocol)] = int(count)
    out = task_success.copy()
    out["one_shot_unique"] = [
        lookup.get((run_id, str(task), "one_shot"), pd.NA)
        for run_id, task in zip(out["run_id"], out["task"])
    ]
    out["pmpnn_unique"] = [
        lookup.get((run_id, str(task), "pmpnn_esmfold"), pd.NA)
        for run_id, task in zip(out["run_id"], out["task"])
    ]
    if skipped:
        out.loc[out["run_id"].isin(skipped), ["one_shot_unique", "pmpnn_unique"]] = pd.NA
    out["partition"] = out["task"].map(lambda name: PARTITION_LABELS[task_partition(name)])
    return out


def _ensure_pmpnn_paper_columns(task_success: pd.DataFrame) -> pd.DataFrame:
    """All/rate for PMPNN paper tables use AND-success design counts.

    Older ``task_success.csv`` files omit ``pmpnn_final_rate``. Concatenating
    them with a refreshed run that has the column leaves NaNs; fill those from
    ``pmpnn_final_count / num_samples`` so plots do not drop the points.
    """
    ts = task_success.copy()
    if "pmpnn_final_count" not in ts.columns or "num_samples" not in ts.columns:
        return ts
    computed = pd.to_numeric(ts["pmpnn_final_count"], errors="coerce") / pd.to_numeric(
        ts["num_samples"], errors="coerce"
    )
    if "pmpnn_final_rate" not in ts.columns:
        ts["pmpnn_final_rate"] = computed
    else:
        ts["pmpnn_final_rate"] = pd.to_numeric(ts["pmpnn_final_rate"], errors="coerce").fillna(
            computed
        )
    return ts


def build_paper_style_table(
    task_success: pd.DataFrame,
    *,
    protocol: str = "one_shot",
    run_order: list[str] | None = None,
    add_total: bool = True,
) -> pd.DataFrame:
    """Per-task All / Unique / rate table in the ProteinZen paper layout."""
    if task_success.empty:
        return pd.DataFrame()
    ts = _ensure_pmpnn_paper_columns(task_success)
    count_col, unique_col, rate_col = _protocol_success_columns(protocol)
    if "partition" not in ts.columns:
        ts["partition"] = ts["task"].map(lambda name: PARTITION_LABELS[task_partition(name)])
    run_order = run_order or list(dict.fromkeys(ts["run_id"].tolist()))

    def _pivot(column: str) -> pd.DataFrame:
        if column not in ts.columns:
            return pd.DataFrame(index=pd.Index([], name="task"))
        return ts.pivot_table(index="task", columns="run_id", values=column, aggfunc="first")

    pieces = []
    for column, metric in ((count_col, "All"), (unique_col, "Unique"), (rate_col, "rate")):
        frame = _pivot(column)
        frame.columns = pd.MultiIndex.from_product([frame.columns, [metric]])
        pieces.append(frame)
    table = pd.concat(pieces, axis=1)
    columns = [(run_id, metric) for run_id in run_order for metric in ("All", "Unique", "rate")]
    table = table.reindex(columns=pd.MultiIndex.from_tuples(columns))

    partition_by_task = (
        ts.drop_duplicates("task").set_index("task")["partition"]
        if "task" in ts.columns
        else pd.Series(dtype=str)
    )
    table.insert(0, ("", "partition"), table.index.map(lambda task: partition_by_task.get(task, "")))
    table[("", "_sort_part")] = table[("", "partition")].map(lambda name: PARTITION_SORT.get(name, 99))
    table[("", "_sort_task")] = table.index.astype(str)
    table = table.sort_values(by=[("", "_sort_part"), ("", "_sort_task")], kind="mergesort")
    table = table.drop(columns=[("", "_sort_part"), ("", "_sort_task")])
    table.index.name = "task"
    if not add_total:
        return _integerize_count_columns(table, run_order)

    totals = {("", "partition"): "TOTAL"}
    for run_id in run_order:
        sub = ts[ts["run_id"] == run_id]
        n_samples = float(sub["num_samples"].sum()) if "num_samples" in sub.columns and len(sub) else 0.0
        all_count = float(sub[count_col].sum()) if count_col in sub.columns and len(sub) else 0.0
        unique_series = (
            pd.to_numeric(sub[unique_col], errors="coerce")
            if unique_col in sub.columns and len(sub)
            else pd.Series(dtype=float)
        )
        unique_sum = (
            float("nan")
            if unique_series.empty or bool(unique_series.isna().all())
            else float(unique_series.sum())
        )
        totals[(run_id, "All")] = all_count
        totals[(run_id, "Unique")] = unique_sum
        totals[(run_id, "rate")] = (all_count / n_samples) if n_samples else float("nan")
    table.loc["TOTAL"] = pd.Series(totals)
    return _integerize_count_columns(table, run_order)


def _integerize_count_columns(table: pd.DataFrame, run_order: list[str]) -> pd.DataFrame:
    out = table.copy()
    for run_id in run_order:
        for metric in ("All", "Unique"):
            column = (run_id, metric)
            if column in out.columns:
                out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")
    return out


def summarize_partitions(task_success: pd.DataFrame) -> pd.DataFrame:
    if task_success.empty:
        return pd.DataFrame()
    ts = task_success.copy()
    if "partition" not in ts.columns:
        ts["partition"] = ts["task"].map(lambda name: PARTITION_LABELS[task_partition(name)])
    grouped = ts.groupby(["partition", "run_id"], as_index=False).agg(
        num_tasks=("task", "nunique"),
        num_samples=("num_samples", "sum"),
        one_shot_success_count=("one_shot_success_count", "sum"),
        pmpnn_success_count=("pmpnn_success_count", "sum"),
        one_shot_unique=("one_shot_unique", "sum"),
        pmpnn_unique=("pmpnn_unique", "sum"),
    )
    if "pmpnn_final_count" in ts.columns:
        extra = ts.groupby(["partition", "run_id"], as_index=False)["pmpnn_final_count"].sum()
        grouped = grouped.merge(extra, on=["partition", "run_id"], how="left")
    grouped["one_shot_success_rate"] = grouped["one_shot_success_count"] / grouped["num_samples"]
    grouped["pmpnn_success_rate"] = grouped["pmpnn_success_count"] / grouped["num_samples"]
    if "pmpnn_final_count" in grouped.columns:
        grouped["pmpnn_final_rate"] = grouped["pmpnn_final_count"] / grouped["num_samples"]
    grouped["_sort_part"] = grouped["partition"].map(lambda name: PARTITION_SORT.get(name, 99))
    grouped = grouped.sort_values(["_sort_part", "run_id"]).drop(columns="_sort_part")
    return grouped.reset_index(drop=True)


def _protocol_success_columns(protocol: str) -> tuple[str, str, str]:
    if protocol == "one_shot":
        return "one_shot_success_count", "one_shot_unique", "one_shot_success_rate"
    if protocol == "pmpnn_esmfold":
        return "pmpnn_final_count", "pmpnn_unique", "pmpnn_final_rate"
    raise ValueError(f"Unknown protocol: {protocol}")


def ordered_task_labels(task_success: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Task names and partition labels in paper-table order."""
    if task_success.empty:
        return [], []
    tasks = task_success.drop_duplicates("task").copy()
    tasks = tasks[tasks["task"].astype(str) != "TOTAL"]
    if "partition" not in tasks.columns:
        tasks["partition"] = tasks["task"].map(lambda name: PARTITION_LABELS[task_partition(name)])
    tasks["_sort_part"] = tasks["partition"].map(lambda name: PARTITION_SORT.get(name, 99))
    tasks["_sort_task"] = tasks["task"].astype(str)
    tasks = tasks.sort_values(["_sort_part", "_sort_task"], kind="mergesort")
    return tasks["task"].astype(str).tolist(), tasks["partition"].astype(str).tolist()


def plot_task_success_points(
    task_success: pd.DataFrame,
    *,
    protocol: str = "pmpnn_esmfold",
    run_order: list[str] | None = None,
    title: str | None = None,
) -> None:
    """Per-task success as points: x=task, y=rate and unique count."""
    if task_success.empty:
        print("No task_success for the scatter plot.")
        return
    ts = _ensure_pmpnn_paper_columns(task_success)
    ts = ts[ts["task"].astype(str) != "TOTAL"].copy()
    _count_col, unique_col, rate_col = _protocol_success_columns(protocol)
    if "partition" not in ts.columns:
        ts["partition"] = ts["task"].map(lambda name: PARTITION_LABELS[task_partition(name)])
    run_order = run_order or list(dict.fromkeys(ts["run_id"].tolist()))
    tasks, partitions = ordered_task_labels(ts)
    if not tasks:
        print("No tasks for the scatter plot.")
        return

    x = np.arange(len(tasks))
    markers = ["o", "s", "D", "^", "v"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    n_runs = len(run_order)
    dodge = 0.16 if n_runs > 1 else 0.0

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(max(11, 0.55 * len(tasks)), 8),
        sharex=True,
    )
    series = [
        (axes[0], rate_col, "success rate", (0.0, 1.05)),
        (axes[1], unique_col, "unique successes", None),
    ]
    for ax, column, ylabel, ylim in series:
        if column not in ts.columns:
            ax.set_ylabel(ylabel)
            continue
        for i, run_id in enumerate(run_order):
            sub = ts.loc[ts["run_id"] == run_id].drop_duplicates("task").set_index("task")
            y = np.array(
                [
                    float(sub.loc[task, column])
                    if task in sub.index and pd.notna(sub.loc[task, column])
                    else np.nan
                    for task in tasks
                ],
                dtype=float,
            )
            dx = (i - (n_runs - 1) / 2) * dodge
            ax.plot(
                x + dx,
                y,
                linestyle="none",
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=0.4,
                label=short_label(run_id),
            )
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        _shade_task_partitions(ax, partitions)

    axes[0].legend(fontsize=8, loc="upper right")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tasks, rotation=55, ha="right")
    axes[1].set_xlabel("task")
    fig.suptitle(title or f"Per-task success - {protocol}", y=0.98)
    plt.tight_layout()
    plt.show()


def _shade_task_partitions(ax, partitions: list[str]) -> None:
    if not partitions:
        return
    start = 0
    part_i = 0
    for i, name in enumerate(partitions):
        last = i == len(partitions) - 1
        if last or partitions[i + 1] != name:
            ax.axvspan(
                start - 0.5,
                i + 0.5,
                color="#f4f4f4" if part_i % 2 == 0 else "#ffffff",
                zorder=0,
            )
            mid = 0.5 * (start + i)
            ax.text(
                mid,
                0.98,
                name,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                color="#666666",
            )
            part_i += 1
            start = i + 1



LAYOUT_LAYERS = (
    ("excursion_trans_A", "excursion translation (Å)"),
    ("excursion_rot_deg", "excursion rotation (deg)"),
    ("retention_trans_A", "retention translation (Å)"),
    ("retention_rot_deg", "retention rotation (deg)"),
)


def remap_layout_pair_tasks(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse Stage-2 sample-level layout ``task`` ids to the parent motif."""
    if df is None or df.empty or "task" not in df.columns:
        return df if df is not None else pd.DataFrame()
    out = df.copy()
    out["task"] = out["task"].map(lambda value: parent_motif_task_id(str(value)))
    return out


_PMPNN_LAYOUT_MEDIAN_COLS = ("retention_trans_A", "retention_rot_deg")
_PMPNN_LAYOUT_FIRST_COLS = (
    "excursion_trans_A",
    "excursion_rot_deg",
    "num_motif_groups",
    "pass_final",
    "fold_source",
    "native_tij_tx",
    "native_tij_ty",
    "native_tij_tz",
    "native_tij_qw",
    "native_tij_qx",
    "native_tij_qy",
    "native_tij_qz",
    "design_tij_tx",
    "design_tij_ty",
    "design_tij_tz",
    "design_tij_qw",
    "design_tij_qx",
    "design_tij_qy",
    "design_tij_qz",
    "source_layout_id",
)


def aggregate_pmpnn_layout_to_design(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (design, group pair): median retention over AND-passing sequences.

    Excursion is design-vs-native and is taken once. Sequence-level CSV stays for
    supplement plots; this table is the publication PMPNN layout unit.
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    if "pmpnn_sample" not in df.columns:
        return df
    keys = [c for c in ("run_id", "name", "task", "group_i", "group_j") if c in df.columns]
    if not keys:
        return df
    agg: dict[str, str] = {}
    for column in _PMPNN_LAYOUT_MEDIAN_COLS:
        if column in df.columns:
            agg[column] = "median"
    for column in _PMPNN_LAYOUT_FIRST_COLS:
        if column in df.columns:
            agg[column] = "first"
    agg["pmpnn_sample"] = "nunique"
    out = df.groupby(keys, as_index=False).agg(agg)
    return out.rename(columns={"pmpnn_sample": "n_pmpnn_sequences"})


def _maybe_aggregate_pmpnn_layout(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "pmpnn_sample" not in df.columns:
        return df if df is not None else pd.DataFrame()
    if df["pmpnn_sample"].notna().any():
        return aggregate_pmpnn_layout_to_design(df)
    return df


def summarize_layout_by_task(layout_pairs: pd.DataFrame) -> pd.DataFrame:
    if layout_pairs is None or layout_pairs.empty:
        return pd.DataFrame()
    layout_pairs = _maybe_aggregate_pmpnn_layout(layout_pairs)
    rows = []
    for (run_id, task), sub in layout_pairs.groupby(["run_id", "task"], sort=False):
        row = {
            "run_id": run_id,
            "task": task,
            "n_pairs": int(len(sub)),
            "n_samples": int(sub["name"].nunique()) if "name" in sub.columns else int(len(sub)),
        }
        if "pass_final" in sub.columns:
            row["n_pass_final_pairs"] = int(sub["pass_final"].fillna(False).astype(bool).sum())
        if "pmpnn_pass_final" in sub.columns:
            row["n_pmpnn_pass_final_pairs"] = int(
                sub["pmpnn_pass_final"].fillna(False).astype(bool).sum()
            )
        for column, _label in LAYOUT_LAYERS:
            if column in sub.columns:
                row[f"{column}_median"] = float(sub[column].median())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_layout_task_medians(
    layout_pairs_or_summary: pd.DataFrame,
    *,
    run_order: list[str] | None = None,
    columns: list[tuple[str, str]] | None = None,
    title: str | None = None,
):
    """Per-task median layout deltas as points."""
    df = layout_pairs_or_summary
    if df is None or df.empty:
        print("No layout summary for the scatter plot.")
        return None
    columns = columns or [
        ("excursion_trans_A_median", "median excursion (Å)"),
        ("excursion_rot_deg_median", "median excursion (deg)"),
    ]
    if not all(column in df.columns for column, _label in columns):
        df = summarize_layout_by_task(df)
    if df.empty:
        print("No layout summary for the scatter plot.")
        return None
    tasks, partitions = ordered_task_labels(df)
    if not tasks:
        print("No tasks for the layout plot.")
        return None
    run_order = run_order or list(dict.fromkeys(df["run_id"].tolist()))
    x = np.arange(len(tasks))
    markers = ["o", "s", "D", "^", "v"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    n_runs = len(run_order)
    dodge = 0.16 if n_runs > 1 else 0.0
    fig, axes = plt.subplots(
        len(columns),
        1,
        figsize=(max(11, 0.55 * len(tasks)), 3.2 * len(columns)),
        sharex=True,
        squeeze=False,
    )
    for ax, (column, ylabel) in zip(axes[:, 0], columns):
        if column not in df.columns:
            ax.set_ylabel(ylabel)
            continue
        for i, run_id in enumerate(run_order):
            sub = df.loc[df["run_id"] == run_id].drop_duplicates("task").set_index("task")
            y = np.array(
                [
                    float(sub.loc[task, column])
                    if task in sub.index and pd.notna(sub.loc[task, column])
                    else np.nan
                    for task in tasks
                ],
                dtype=float,
            )
            ax.plot(
                x + (i - (n_runs - 1) / 2) * dodge,
                y,
                linestyle="none",
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=0.4,
                label=short_label(run_id),
            )
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        _shade_task_partitions(ax, partitions)
    axes[0, 0].legend(fontsize=8, loc="upper right")
    axes[-1, 0].set_xticks(x)
    axes[-1, 0].set_xticklabels(tasks, rotation=55, ha="right")
    axes[-1, 0].set_xlabel("task")
    fig.suptitle(title or "Per-task median layout deltas", y=0.99)
    plt.tight_layout()
    plt.show()
    return fig


def plot_layout_boxes(
    layout_pairs: pd.DataFrame,
    *,
    column: str,
    ylabel: str | None = None,
    title: str | None = None,
    run_order: list[str] | None = None,
):
    if layout_pairs is None or layout_pairs.empty or column not in layout_pairs.columns:
        print(f"No column {column} for the boxplot.")
        return None
    tasks, partitions = ordered_task_labels(layout_pairs)
    if not tasks:
        print("No tasks for the layout boxplot.")
        return None
    run_order = run_order or list(dict.fromkeys(layout_pairs["run_id"].tolist()))
    n_runs = len(run_order)
    x = np.arange(len(tasks))
    width = 0.22 if n_runs > 1 else 0.45
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    fig, ax = plt.subplots(figsize=(max(11, 0.55 * len(tasks)), 5.2))
    _shade_task_partitions(ax, partitions)
    for i, run_id in enumerate(run_order):
        data = []
        for task in tasks:
            vals = (
                layout_pairs.loc[
                    (layout_pairs["run_id"] == run_id)
                    & (layout_pairs["task"].astype(str) == task),
                    column,
                ]
                .dropna()
                .to_numpy()
            )
            data.append(vals if len(vals) else np.array([np.nan]))
        positions = x + (i - (n_runs - 1) / 2) * width
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.85,
            showfliers=False,
            patch_artist=True,
            manage_ticks=False,
        )
        color = colors[i % len(colors)]
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        for key in ("whiskers", "caps", "medians"):
            for line in bp[key]:
                line.set_color("#333333")
        ax.plot([], [], color=color, linewidth=8, alpha=0.65, label=short_label(run_id))
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=55, ha="right")
    ax.set_ylabel(ylabel or column)
    ax.set_xlabel("task")
    ax.set_title(title or column)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.show()
    return fig


def plot_layout_histograms(
    layout_pairs: pd.DataFrame,
    *,
    run_order: list[str] | None = None,
    title: str | None = None,
    pass_final_only: bool = False,
):
    if layout_pairs is None or layout_pairs.empty:
        print("No layout for the histograms.")
        return None
    df = layout_pairs
    if pass_final_only:
        if "pass_final" not in df.columns:
            print("No pass_final column.")
            return None
        df = df[df["pass_final"].fillna(False).astype(bool)].copy()
        if df.empty:
            print("No pass_final layout rows.")
            return None
    run_order = run_order or list(dict.fromkeys(df["run_id"].tolist()))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (column, ylabel) in zip(axes.ravel(), LAYOUT_LAYERS):
        if column not in df.columns:
            ax.set_title(ylabel)
            continue
        vals = df[column].dropna()
        if vals.empty:
            ax.set_title(ylabel)
            continue
        bins = np.histogram_bin_edges(vals.to_numpy(), bins=24)
        for run_id in run_order:
            sub = df.loc[df["run_id"] == run_id, column].dropna()
            if len(sub):
                ax.hist(
                    sub,
                    bins=bins,
                    alpha=0.45,
                    label=f"{short_label(run_id)} (n={len(sub)})",
                )
        ax.set_xlabel(ylabel)
        ax.set_ylabel("pair count")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    axes[0, 1].legend(fontsize=8, loc="upper right")
    fig.suptitle(title or "Layout pair deltas", y=0.99)
    plt.tight_layout()
    plt.show()
    return fig


def attach_layout_pmpnn_pass(layout_pairs: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Mark layout rows whose design has any PMPNN AND-success sequence."""
    if layout_pairs is None or layout_pairs.empty:
        return layout_pairs if layout_pairs is not None else pd.DataFrame()
    out = layout_pairs.copy()
    out["pmpnn_pass_final"] = False
    if metrics is None or metrics.empty or "name" not in out.columns:
        return out
    pmpnn = metrics
    if "protocol" in metrics.columns:
        pmpnn = metrics[metrics["protocol"] == "pmpnn_esmfold"]
    if pmpnn.empty or "pass_final" not in pmpnn.columns or "name" not in pmpnn.columns:
        return out
    passed = set(
        pmpnn.loc[pmpnn["pass_final"].fillna(False).astype(bool), "name"].astype(str)
    )
    out["pmpnn_pass_final"] = out["name"].astype(str).isin(passed)
    return out


def filter_layout_pass_final(
    layout_pairs: pd.DataFrame,
    *,
    protocol: str = "one_shot",
) -> pd.DataFrame:
    """Keep layout rows that passed one-shot or PMPNN AND-success."""
    if layout_pairs is None or layout_pairs.empty:
        return pd.DataFrame()
    column = "pmpnn_pass_final" if protocol in {"pmpnn_esmfold", "pmpnn"} else "pass_final"
    if column not in layout_pairs.columns:
        print(f"No column {column}.")
        return pd.DataFrame()
    return layout_pairs[layout_pairs[column].fillna(False).astype(bool)].copy()


def plot_matched_layout_metrics(
    layout_pairs: pd.DataFrame,
    *,
    run_order: list[str] | None = None,
    pass_final_only: bool = False,
    passed_only: str | None = None,
) -> pd.DataFrame:
    """Plot stored layout CSV (excursion/retention) and return per-task medians.

    ``passed_only``: ``None`` (all samples), ``"one_shot"``, or ``"pmpnn_esmfold"``.
    ``pass_final_only=True`` is an alias for ``passed_only="one_shot"``.
    """
    if layout_pairs is None or layout_pairs.empty:
        print("No layout_pair_metrics for the plots.")
        return pd.DataFrame()
    if passed_only is None and pass_final_only:
        passed_only = "one_shot"
    df = layout_pairs
    suffix = ""
    if passed_only in {"one_shot", "pmpnn_esmfold", "pmpnn"}:
        df = filter_layout_pass_final(df, protocol=passed_only)
        suffix = (
            " (PMPNN pass_final)"
            if passed_only in {"pmpnn_esmfold", "pmpnn"}
            else " (pass_final only)"
        )
        if df.empty:
            print(f"No layout rows after the {passed_only} pass_final filter.")
            return pd.DataFrame()
    df = _maybe_aggregate_pmpnn_layout(df)
    run_order = run_order or list(dict.fromkeys(df["run_id"].tolist()))
    summary = summarize_layout_by_task(df)
    plot_layout_task_medians(
        summary,
        run_order=run_order,
        columns=[
            ("excursion_trans_A_median", "median excursion (Å)"),
            ("excursion_rot_deg_median", "median excursion (deg)"),
        ],
        title=f"Per-task median excursion (design vs native){suffix}",
    )
    plot_layout_task_medians(
        summary,
        run_order=run_order,
        columns=[
            ("retention_trans_A_median", "median retention (Å)"),
            ("retention_rot_deg_median", "median retention (deg)"),
        ],
        title=f"Per-task median retention (design vs ESMFold){suffix}",
    )
    plot_layout_boxes(
        df,
        column="excursion_trans_A",
        ylabel="excursion translation (Å)",
        title=f"Excursion translation spread per task (no outliers){suffix}",
        run_order=run_order,
    )
    plot_layout_boxes(
        df,
        column="excursion_rot_deg",
        ylabel="excursion rotation (deg)",
        title=f"Excursion rotation spread per task (no outliers){suffix}",
        run_order=run_order,
    )
    plot_layout_histograms(
        df,
        run_order=run_order,
        title=f"Layout pair deltas{suffix or ' (all samples)'}",
    )
    if passed_only is None:
        plot_layout_histograms(
            layout_pairs,
            run_order=run_order,
            title="Layout pair deltas (pass_final only)",
            pass_final_only=True,
        )
    return summary


def _cluster_ids_from_layout(df: pd.DataFrame) -> dict[str, str] | None:
    if df is None or df.empty or "source_layout_id" not in df.columns:
        return None
    mapping: dict[str, str] = {}
    for name, source in zip(df["name"].astype(str), df["source_layout_id"].astype(str)):
        if source and source.lower() not in {"nan", "<na>", "none"}:
            mapping[name] = source
    return mapping or None


def _mobility_group_rng(random_state: int, run_id: object, task: object) -> np.random.Generator:
    payload = f"{run_id}\0{task}".encode()
    hashed = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    return np.random.default_rng([int(random_state) & 0xFFFFFFFF, hashed & 0xFFFFFFFF])


def _summarize_layout_mobility_group(
    sub: pd.DataFrame,
    n_boot: int,
    rng: np.random.Generator,
    subsets: tuple[str, ...],
) -> list[dict]:
    run_id = sub["run_id"].iloc[0]
    task = sub["task"].iloc[0]
    relatives = relatives_from_pair_df(sub, prefix="design")
    cluster_ids = _cluster_ids_from_layout(sub)
    names_all = list(dict.fromkeys(sub["name"].astype(str)))
    names_final = names_all
    if "pass_final" in sub.columns:
        names_final = list(
            dict.fromkeys(
                sub.loc[sub["pass_final"].fillna(False).astype(bool), "name"].astype(str)
            )
        )
    rows: list[dict] = []
    for subset, names in (("all", names_all), ("pass_final", names_final)):
        if subset not in subsets:
            continue
        slice_df = sub[sub["name"].astype(str).isin(names)]
        exc_t = slice_df.groupby("name")["excursion_trans_A"].apply(rms_over_values)
        exc_r = slice_df.groupby("name")["excursion_rot_deg"].apply(rms_over_values)
        ret_t = (
            slice_df.groupby("name")["retention_trans_A"].apply(rms_over_values)
            if "retention_trans_A" in slice_df.columns
            else pd.Series(dtype=float)
        )
        ret_r = (
            slice_df.groupby("name")["retention_rot_deg"].apply(rms_over_values)
            if "retention_rot_deg" in slice_df.columns
            else pd.Series(dtype=float)
        )
        pw = pairwise_layout_stats(
            relatives, names, cluster_ids=cluster_ids, n_boot=n_boot, rng=rng
        )
        exc_t_lo, exc_t_hi = bootstrap_median_of_designs(
            exc_t.to_dict(), names, cluster_ids=cluster_ids, n_boot=n_boot, rng=rng
        )
        exc_r_lo, exc_r_hi = bootstrap_median_of_designs(
            exc_r.to_dict(), names, cluster_ids=cluster_ids, n_boot=n_boot, rng=rng
        )
        ret_t_lo, ret_t_hi = bootstrap_median_of_designs(
            ret_t.to_dict(), names, cluster_ids=cluster_ids, n_boot=n_boot, rng=rng
        )
        ret_r_lo, ret_r_hi = bootstrap_median_of_designs(
            ret_r.to_dict(), names, cluster_ids=cluster_ids, n_boot=n_boot, rng=rng
        )
        n_des = int(len(names))
        n_eval = int(sum(name in relatives for name in names))
        rows.append(
            {
                "run_id": run_id,
                "task": task,
                "subset": subset,
                "n_designs": n_des,
                "n_evaluable": n_eval,
                "weak_n": n_des < 5,
                "pairwise_defined": bool(pw["pairwise_defined"]),
                "excursion_trans_median": float(np.nanmedian(exc_t.to_numpy()))
                if len(exc_t)
                else float("nan"),
                "excursion_trans_q25": float(np.nanpercentile(exc_t.to_numpy(), 25))
                if len(exc_t)
                else float("nan"),
                "excursion_trans_q75": float(np.nanpercentile(exc_t.to_numpy(), 75))
                if len(exc_t)
                else float("nan"),
                "excursion_trans_ci_lo": exc_t_lo,
                "excursion_trans_ci_hi": exc_t_hi,
                "excursion_rot_median": float(np.nanmedian(exc_r.to_numpy()))
                if len(exc_r)
                else float("nan"),
                "excursion_rot_q25": float(np.nanpercentile(exc_r.to_numpy(), 25))
                if len(exc_r)
                else float("nan"),
                "excursion_rot_q75": float(np.nanpercentile(exc_r.to_numpy(), 75))
                if len(exc_r)
                else float("nan"),
                "excursion_rot_ci_lo": exc_r_lo,
                "excursion_rot_ci_hi": exc_r_hi,
                "pairwise_trans_median": pw["median_pairwise_layout_translation_distance"],
                "pairwise_trans_ci_lo": pw["bootstrap_trans_ci_lo"],
                "pairwise_trans_ci_hi": pw["bootstrap_trans_ci_hi"],
                "pairwise_rot_median": pw["median_pairwise_layout_rotation_distance"],
                "pairwise_rot_ci_lo": pw["bootstrap_rot_ci_lo"],
                "pairwise_rot_ci_hi": pw["bootstrap_rot_ci_hi"],
                "retention_trans_median": float(np.nanmedian(ret_t.to_numpy()))
                if len(ret_t)
                else float("nan"),
                "retention_trans_q25": float(np.nanpercentile(ret_t.to_numpy(), 25))
                if len(ret_t)
                else float("nan"),
                "retention_trans_q75": float(np.nanpercentile(ret_t.to_numpy(), 75))
                if len(ret_t)
                else float("nan"),
                "retention_trans_ci_lo": ret_t_lo,
                "retention_trans_ci_hi": ret_t_hi,
                "retention_rot_median": float(np.nanmedian(ret_r.to_numpy()))
                if len(ret_r)
                else float("nan"),
                "retention_rot_q25": float(np.nanpercentile(ret_r.to_numpy(), 25))
                if len(ret_r)
                else float("nan"),
                "retention_rot_q75": float(np.nanpercentile(ret_r.to_numpy(), 75))
                if len(ret_r)
                else float("nan"),
                "retention_rot_ci_lo": ret_r_lo,
                "retention_rot_ci_hi": ret_r_hi,
                "bootstrap_unit": pw["bootstrap_unit"],
                "num_design_pairs": pw["num_design_pairs"],
            }
        )
    return rows


def _summarize_layout_mobility_group_payload(
    payload: tuple[pd.DataFrame, int, int, tuple[str, ...]],
) -> list[dict]:
    sub, n_boot, random_state, subsets = payload
    rng = _mobility_group_rng(random_state, sub["run_id"].iloc[0], sub["task"].iloc[0])
    return _summarize_layout_mobility_group(sub, n_boot, rng, subsets)


def summarize_layout_mobility(
    layout_pairs: pd.DataFrame,
    *,
    n_boot: int = 200,
    random_state: int = 0,
    subsets: tuple[str, ...] = ("all", "pass_final"),
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Per task × campaign × subset: median excursion, pairwise distance, retention.

    Pairwise distances resample **designs** (or stage-2 source layouts), never
    precomputed pair rows. ``n_success < 2`` → pairwise undefined; ``< 5`` → weak_n.
    Independent ``(run_id, task)`` groups can run in parallel via ``n_jobs``.
    """
    if layout_pairs is None or layout_pairs.empty:
        return pd.DataFrame()
    df = _maybe_aggregate_pmpnn_layout(layout_pairs)
    groups = [sub for _, sub in df.groupby(["run_id", "task"], sort=False)]
    if not groups:
        return pd.DataFrame()
    workers = max(1, int(n_jobs))
    if workers > 1 and len(groups) > 1:
        payloads = [(sub, int(n_boot), int(random_state), subsets) for sub in groups]
        with ThreadPoolExecutor(max_workers=min(workers, len(groups))) as pool:
            parts = list(pool.map(_summarize_layout_mobility_group_payload, payloads))
        rows = [row for part in parts for row in part]
    else:
        rows = []
        for sub in groups:
            rng = _mobility_group_rng(random_state, sub["run_id"].iloc[0], sub["task"].iloc[0])
            rows.extend(_summarize_layout_mobility_group(sub, int(n_boot), rng, subsets))
    return pd.DataFrame(rows)


def resolve_layout_mobility(
    layout_pairs: pd.DataFrame,
    precomputed: pd.DataFrame | None = None,
    *,
    n_boot: int = 200,
    random_state: int = 0,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, str]:
    """Prefer a precomputed mobility table; otherwise summarize pair rows.

    Returns ``(frame, source)`` where source is ``precomputed`` or ``recomputed``.
    """
    if precomputed is not None and not precomputed.empty:
        return precomputed.copy(), "precomputed"
    return (
        summarize_layout_mobility(
            layout_pairs,
            n_boot=n_boot,
            random_state=random_state,
            n_jobs=n_jobs,
        ),
        "recomputed",
    )


def plot_layout_fig_c(
    mobility: pd.DataFrame,
    *,
    run_order: list[str] | None = None,
    title: str | None = None,
    connect_run_pairs: tuple[tuple[str, str], ...] = (
        ("one_motif_proteinzen", "tlock05_stage1"),
    ),
    probe_run_ids: tuple[str, ...] = ("tlock05_stage2",),
):
    """Fig C: excursion × inter-sample diversity. Not boxplots of dependent pairs.

    ``mobility`` is a summary DataFrame from ``summarize_layout_mobility`` or a
    precomputed ``layout_mobility.csv`` (not raw pair rows).
    Stage2 is a layout-probe marker, not a third equal method vs baseline/stage1.
    """
    if mobility is None or mobility.empty:
        print("No layout mobility summary for Fig C.")
        return None
    run_order = run_order or list(dict.fromkeys(mobility["run_id"].tolist()))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=False)
    panels = (
        (
            "excursion_trans_median",
            "pairwise_trans_median",
            "excursion_trans_ci_lo",
            "excursion_trans_ci_hi",
            "pairwise_trans_ci_lo",
            "pairwise_trans_ci_hi",
            "median excursion (Å)",
            "median pairwise layout translation distance (Å)",
        ),
        (
            "excursion_rot_median",
            "pairwise_rot_median",
            "excursion_rot_ci_lo",
            "excursion_rot_ci_hi",
            "pairwise_rot_ci_lo",
            "pairwise_rot_ci_hi",
            "median excursion (deg)",
            "median pairwise layout rotation distance (deg)",
        ),
    )
    probe = set(probe_run_ids)
    for ax, (xcol, ycol, xlo, xhi, ylo, yhi, xlabel, ylabel) in zip(axes, panels):
        for i, run_id in enumerate(run_order):
            color = colors[i % len(colors)]
            marker = "D" if run_id in probe else "o"
            for subset, filled, alpha in (
                ("all", False, 0.55),
                ("pass_final", True, 0.95),
            ):
                sub = mobility[
                    (mobility["run_id"] == run_id) & (mobility["subset"] == subset)
                ]
                if sub.empty:
                    continue
                x = sub[xcol].to_numpy(dtype=float)
                y = sub[ycol].to_numpy(dtype=float)
                ax.errorbar(
                    x,
                    y,
                    xerr=np.vstack(
                        [
                            np.clip(x - sub[xlo].to_numpy(dtype=float), 0, None),
                            np.clip(sub[xhi].to_numpy(dtype=float) - x, 0, None),
                        ]
                    )
                    if xlo in sub.columns
                    else None,
                    yerr=np.vstack(
                        [
                            np.clip(y - sub[ylo].to_numpy(dtype=float), 0, None),
                            np.clip(sub[yhi].to_numpy(dtype=float) - y, 0, None),
                        ]
                    )
                    if ylo in sub.columns
                    else None,
                    fmt=marker,
                    color=color,
                    mfc=color if filled else "white",
                    mec=color,
                    alpha=alpha,
                    markersize=8 if filled else 7,
                    linestyle="none",
                    label=f"{short_label(run_id)} / {subset}",
                    zorder=3 if filled else 2,
                )
        for left_id, right_id in connect_run_pairs:
            if left_id not in run_order or right_id not in run_order:
                continue
            for subset in ("all", "pass_final"):
                left = mobility[
                    (mobility["run_id"] == left_id) & (mobility["subset"] == subset)
                ].set_index("task")
                right = mobility[
                    (mobility["run_id"] == right_id) & (mobility["subset"] == subset)
                ].set_index("task")
                for task in left.index.intersection(right.index):
                    ax.plot(
                        [left.loc[task, xcol], right.loc[task, xcol]],
                        [left.loc[task, ycol], right.loc[task, ycol]],
                        color="#888888",
                        alpha=0.35,
                        linewidth=0.8,
                        zorder=1,
                    )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle(title or "Fig C: excursion × inter-sample layout diversity", y=0.99)
    plt.tight_layout()
    plt.show()
    return fig


def plot_layout_retention_whiskers(
    mobility: pd.DataFrame,
    *,
    run_order: list[str] | None = None,
    subset: str = "pass_final",
    title: str | None = None,
):
    """Per-task median design→fold retention (lower is better)."""
    if mobility is None or mobility.empty:
        print("No retention summary.")
        return None
    df = mobility[mobility["subset"] == subset].copy() if "subset" in mobility.columns else mobility
    if df.empty:
        print(f"No retention rows for subset={subset}.")
        return None
    tasks, partitions = ordered_task_labels(df)
    if not tasks:
        return None
    run_order = run_order or list(dict.fromkeys(df["run_id"].tolist()))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    x = np.arange(len(tasks))
    n_runs = len(run_order)
    dodge = 0.18 if n_runs > 1 else 0.0
    fig, axes = plt.subplots(2, 1, figsize=(max(11, 0.55 * len(tasks)), 7.2), sharex=True)
    panels = (
        ("retention_trans_median", "retention_trans_ci_lo", "retention_trans_ci_hi", "median retention (Å)"),
        ("retention_rot_median", "retention_rot_ci_lo", "retention_rot_ci_hi", "median retention (deg)"),
    )
    for ax, (col, lo, hi, ylabel) in zip(axes, panels):
        _shade_task_partitions(ax, partitions)
        for i, run_id in enumerate(run_order):
            sub = df.loc[df["run_id"] == run_id].drop_duplicates("task").set_index("task")
            y = np.array(
                [
                    float(sub.loc[task, col])
                    if task in sub.index and pd.notna(sub.loc[task, col])
                    else np.nan
                    for task in tasks
                ]
            )
            ylo = np.array(
                [
                    float(sub.loc[task, lo])
                    if lo in sub.columns and task in sub.index and pd.notna(sub.loc[task, lo])
                    else np.nan
                    for task in tasks
                ]
            )
            yhi = np.array(
                [
                    float(sub.loc[task, hi])
                    if hi in sub.columns and task in sub.index and pd.notna(sub.loc[task, hi])
                    else np.nan
                    for task in tasks
                ]
            )
            dx = x + (i - (n_runs - 1) / 2) * dodge
            ax.errorbar(
                dx,
                y,
                yerr=np.vstack(
                    [
                        np.clip(y - ylo, 0, None),
                        np.clip(yhi - y, 0, None),
                    ]
                ),
                fmt="o",
                color=colors[i % len(colors)],
                markersize=7,
                label=short_label(run_id),
            )
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(tasks, rotation=55, ha="right")
    axes[-1].set_xlabel("task")
    fig.suptitle(title or f"Retention (design → fold), subset={subset}", y=0.99)
    plt.tight_layout()
    plt.show()
    return fig


def _mds_kruskal_stress(dist: np.ndarray, xy: np.ndarray) -> float:
    """Kruskal stress-1 of a 2D embedding vs the input distances (upper triangle)."""
    n = dist.shape[0]
    if n < 2:
        return float("nan")
    delta = xy[:, None, :] - xy[None, :, :]
    embedded = np.sqrt(np.sum(delta * delta, axis=-1))
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    dhat = embedded[iu]
    denom = float(np.sum(d * d))
    if denom <= 0.0:
        return float("nan")
    return float(np.sqrt(np.sum((d - dhat) ** 2) / denom))


def layout_mds_coordinates(
    layout_pairs: pd.DataFrame,
    task: str,
    *,
    kind: str = "trans",
    tau_trans: float = MDS_TAU_TRANS_DEFAULT,
    tau_rot: float = MDS_TAU_ROT_DEFAULT,
) -> pd.DataFrame:
    """Per-task deterministic MDS of **all** designs. Do not embed successes alone.

    Joint embedding across campaigns is allowed only when group correspondence
    matches; ``_2grp`` vs full splits must not share one feature space (they are
    different ``task`` ids here). Combined trans+rot uses documented taus.
    Classical Torgerson MDS (no sklearn RNG).
    """
    if layout_pairs is None or layout_pairs.empty:
        return pd.DataFrame()
    df = layout_pairs[layout_pairs["task"].astype(str) == str(task)].copy()
    if df.empty:
        return pd.DataFrame()
    orig_name = df["name"].astype(str)
    if "run_id" in df.columns:
        df = df.assign(name=df["run_id"].astype(str) + "::" + orig_name)
        id_to_name = dict(zip(df["name"].astype(str), orig_name))
    else:
        id_to_name = {n: n for n in orig_name}
    relatives = relatives_from_pair_df(df, prefix="design")
    names = list(dict.fromkeys(df["name"].astype(str)))
    names, dist = layout_distance_matrix(
        relatives, names, kind=kind, tau_trans=tau_trans, tau_rot=tau_rot
    )
    if len(names) < 2:
        return pd.DataFrame()
    xy = classical_mds(dist, n_components=2)
    stress = _mds_kruskal_stress(dist, xy)
    meta = df.drop_duplicates("name").set_index("name")
    rows = []
    for i, mds_id in enumerate(names):
        row = {
            "name": id_to_name.get(mds_id, mds_id),
            "mds_id": mds_id,
            "task": task,
            "mds_x": float(xy[i, 0]),
            "mds_y": float(xy[i, 1]),
            "kind": kind,
            "tau_trans": float(tau_trans),
            "tau_rot": float(tau_rot),
            "stress": stress,
        }
        if mds_id in meta.index:
            rec = meta.loc[mds_id]
            if "run_id" in meta.columns:
                row["run_id"] = rec["run_id"]
            if "pass_final" in meta.columns:
                row["pass_final"] = bool(rec["pass_final"])
        rows.append(row)
    return pd.DataFrame(rows)


def _filter_layout_mds_table(
    df: pd.DataFrame,
    task: str | None = None,
    *,
    kind: str | None = None,
    tau_trans: float | None = None,
    tau_rot: float | None = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df
    if task is not None and "task" in out.columns:
        out = out[out["task"].astype(str) == str(task)]
    if kind is not None and "kind" in out.columns:
        out = out[out["kind"].astype(str) == str(kind)]
    if tau_trans is not None and "tau_trans" in out.columns:
        out = out[np.isclose(pd.to_numeric(out["tau_trans"], errors="coerce"), float(tau_trans))]
    if tau_rot is not None and "tau_rot" in out.columns:
        out = out[np.isclose(pd.to_numeric(out["tau_rot"], errors="coerce"), float(tau_rot))]
    return out.copy()


def load_layout_mds_coordinates(
    path: Path | str | None,
    task: str | None = None,
    *,
    kind: str | None = None,
    tau_trans: float | None = None,
    tau_rot: float | None = None,
) -> pd.DataFrame:
    """Load a precomputed layout MDS table. Missing/empty files are empty frames."""
    if path is None:
        return pd.DataFrame()
    df = safe_read_csv(Path(path))
    if df.empty:
        return df
    return _filter_layout_mds_table(
        df, task, kind=kind, tau_trans=tau_trans, tau_rot=tau_rot
    )


def resolve_layout_mds_coordinates(
    layout_pairs: pd.DataFrame,
    task: str,
    *,
    kind: str = "trans",
    tau_trans: float = MDS_TAU_TRANS_DEFAULT,
    tau_rot: float = MDS_TAU_ROT_DEFAULT,
    precomputed: pd.DataFrame | Path | str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Prefer a precomputed MDS table; otherwise embed from pair rows.

    Returns ``(frame, source)`` where source is ``precomputed`` or ``recomputed``.
    """
    if isinstance(precomputed, (str, Path)):
        loaded = load_layout_mds_coordinates(
            precomputed, task, kind=kind, tau_trans=tau_trans, tau_rot=tau_rot
        )
    elif precomputed is not None:
        loaded = _filter_layout_mds_table(
            precomputed, task, kind=kind, tau_trans=tau_trans, tau_rot=tau_rot
        )
    else:
        loaded = pd.DataFrame()
    if loaded is not None and not loaded.empty:
        return loaded.copy(), "precomputed"
    return (
        layout_mds_coordinates(
            layout_pairs, task, kind=kind, tau_trans=tau_trans, tau_rot=tau_rot
        ),
        "recomputed",
    )


def plot_layout_mds(
    coords: pd.DataFrame,
    *,
    title: str | None = None,
):
    """Grey = all designs; pass_final outlined/colored; campaigns by color."""
    if coords is None or coords.empty:
        print("No MDS coordinates.")
        return None
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    markers = ["o", "s", "D", "^", "P"]
    run_ids = list(dict.fromkeys(coords["run_id"].tolist())) if "run_id" in coords.columns else [""]
    for i, run_id in enumerate(run_ids):
        sub = coords[coords["run_id"] == run_id] if run_id != "" else coords
        marker = markers[i % len(markers)]
        ax.scatter(
            sub["mds_x"],
            sub["mds_y"],
            s=22,
            color="#bbbbbb",
            marker=marker,
            alpha=0.55,
            zorder=1,
        )
        if "pass_final" in sub.columns:
            passed = sub[sub["pass_final"].fillna(False).astype(bool)]
            ax.scatter(
                passed["mds_x"],
                passed["mds_y"],
                s=42,
                marker=marker,
                facecolors=colors[i % len(colors)],
                edgecolors="black",
                linewidths=0.6,
                label=short_label(run_id) if run_id else "pass_final",
                zorder=3,
            )
        else:
            ax.scatter(
                sub["mds_x"],
                sub["mds_y"],
                s=36,
                marker=marker,
                color=colors[i % len(colors)],
                label=short_label(run_id) if run_id else "designs",
                zorder=2,
            )
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title(title or "Layout MDS (all designs; successes highlighted)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.show()
    return fig


def summarize_run(run: dict) -> dict:
    metrics = run["metrics"]
    unique = run["unique_success"]
    ts = run["task_success"]

    row = {
        "run_id": run["run_id"],
        "model": run["model"],
        "benchmark": run["benchmark"],
        "checkpoint": run["checkpoint"],
        "date": run["date"],
        "num_designs": metrics.loc[metrics["protocol"] == "one_shot", "name"].nunique(),
        "num_tasks": metrics["task"].nunique(),
    }

    if not ts.empty:
        total_samples = int(ts["num_samples"].sum())
        row["one_shot_pass_count"] = int(ts["one_shot_success_count"].sum())
        if "one_shot_pooled_legacy_count" in ts.columns:
            row["one_shot_pooled_legacy_count"] = int(ts["one_shot_pooled_legacy_count"].sum())
        else:
            row["one_shot_pooled_legacy_count"] = row["one_shot_pass_count"]
        if "pmpnn_final_count" in ts.columns:
            row["pmpnn_esmfold_pass_count"] = int(ts["pmpnn_final_count"].sum())
        else:
            row["pmpnn_esmfold_pass_count"] = int(ts["pmpnn_success_count"].sum())
        row["one_shot_pass_rate"] = (
            row["one_shot_pass_count"] / total_samples if total_samples else np.nan
        )
        row["pmpnn_esmfold_pass_rate"] = (
            row["pmpnn_esmfold_pass_count"] / total_samples if total_samples else np.nan
        )
        row["tasks_with_any_one_shot"] = int(ts["any_one_shot"].sum())
        row["tasks_with_any_pmpnn"] = int(ts["any_pmpnn"].sum())
    else:
        for protocol in ("one_shot", "pmpnn_esmfold"):
            sub = metrics[metrics["protocol"] == protocol]
            if protocol == "pmpnn_esmfold" and not sub.empty:
                sub = sub.groupby(["task", "name"], as_index=False)["pass"].max()
            row[f"{protocol}_pass_count"] = int(sub["pass"].sum())
            row[f"{protocol}_pass_rate"] = float(sub["pass"].mean()) if len(sub) else np.nan
        if "pass_final" in metrics.columns:
            one_shot = metrics[metrics["protocol"] == "one_shot"]
            row["one_shot_pooled_legacy_count"] = int(one_shot["pass"].sum()) if not one_shot.empty else 0
            row["one_shot_pass_count"] = int(one_shot["pass_final"].sum()) if not one_shot.empty else 0
        else:
            row["one_shot_pooled_legacy_count"] = row.get("one_shot_pass_count", 0)

    for protocol in ("one_shot", "pmpnn_esmfold"):
        sub = metrics[metrics["protocol"] == protocol]
        row[f"{protocol}_mean_motif_ha"] = (
            float(sub["motif_all_atom_rmsd"].mean()) if len(sub) else np.nan
        )
        row[f"{protocol}_mean_motif_bb"] = float(sub["motif_bb_rmsd"].mean()) if len(sub) else np.nan
        row[f"{protocol}_unique_total"] = (
            np.nan
            if unique.get("skipped")
            else int(sum(unique.get(protocol, {}).values()))
        )
    return row


def display_columns() -> list[str]:
    return [
        "run_id",
        "num_designs",
        "one_shot_pass_rate",
        "one_shot_pooled_legacy_count",
        "pmpnn_esmfold_pass_rate",
        "one_shot_unique_total",
        "pmpnn_esmfold_unique_total",
        "tasks_with_any_one_shot",
        "tasks_with_any_pmpnn",
    ]


def build_task_pivot(task_success: pd.DataFrame) -> pd.DataFrame:
    if task_success.empty:
        return pd.DataFrame()
    return task_success.pivot_table(
        index="task",
        columns="run_id",
        values=["one_shot_success_rate", "pmpnn_success_rate", "any_one_shot", "any_pmpnn"],
        aggfunc="first",
    )


def short_label(run_id: str, max_len: int = 32) -> str:
    return run_id if len(run_id) <= max_len else run_id[: max_len - 1] + "..."


def plot_success_rates(run_summary: pd.DataFrame) -> None:
    if run_summary.empty:
        print("No run_summary for visualization.")
        return
    labels = [short_label(r) for r in run_summary["run_id"]]
    y = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, 0.45 * len(run_summary))), sharex=True)
    for ax, col, title in [
        (axes[0], "one_shot_pass_rate", "One-shot success rate"),
        (axes[1], "pmpnn_esmfold_pass_rate", "PMPNN+ESMFold success rate"),
    ]:
        values = run_summary[col].fillna(0.0).to_numpy()
        ax.barh(y, values, color="#4C72B0")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 1)
        ax.set_xlabel("fraction passing")
        ax.set_title(title)
        for i, value in enumerate(values):
            ax.text(min(value + 0.02, 0.95), i, f"{value:.1%}", va="center", fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_unique_success(run_summary: pd.DataFrame) -> None:
    if run_summary.empty:
        print("No run_summary for visualization.")
        return
    unique_cols = ["one_shot_unique_total", "pmpnn_esmfold_unique_total"]
    if not run_summary[unique_cols].notna().any().any():
        print("unique_success.json is missing or Foldseek was skipped.")
        return

    labels = [short_label(r) for r in run_summary["run_id"]]
    width = 0.35
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(run_summary))))
    ax.bar(
        x - width / 2,
        run_summary["one_shot_unique_total"].fillna(0),
        width,
        label="one-shot",
    )
    ax.bar(
        x + width / 2,
        run_summary["pmpnn_esmfold_unique_total"].fillna(0),
        width,
        label="pmpnn_esmfold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("unique success count (Foldseek)")
    ax.set_title("Unique successful designs per run")
    ax.legend()
    plt.tight_layout()
    plt.show()


def _plot_rmsd_distribution_by_column(
    metrics: pd.DataFrame,
    run_summary: pd.DataFrame,
    *,
    protocol: str,
    rmsd_column: str,
    threshold: float | None,
    title_prefix: str,
) -> None:
    if metrics.empty or run_summary.empty:
        print("No data for RMSD visualization.")
        return
    if rmsd_column not in metrics.columns:
        print(f"Column {rmsd_column} was not found in metrics.")
        return

    sub = metrics[metrics["protocol"] == protocol].copy()
    sub = sub[sub[rmsd_column].notna()]
    if sub.empty:
        print(f"No {rmsd_column} values for protocol {protocol}.")
        return

    run_order = run_summary["run_id"].tolist()
    sub["run_id"] = pd.Categorical(sub["run_id"], categories=run_order, ordered=True)
    shared_bins = np.histogram_bin_edges(sub[rmsd_column].to_numpy(), bins=20)

    fig, ax = plt.subplots(figsize=(12, 5))
    for run_id in run_order:
        vals = sub.loc[sub["run_id"] == run_id, rmsd_column]
        if len(vals):
            ax.hist(
                vals,
                bins=shared_bins,
                alpha=0.45,
                label=f"{short_label(run_id)} (n={len(vals)})",
            )
    if threshold is not None:
        ax.axvline(
            threshold,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"threshold ({threshold} A)",
        )
    ax.set_xlabel(f"{rmsd_column} (A)")
    ax.set_ylabel("count")
    ax.set_title(f"{title_prefix} RMSD distribution - {protocol}")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_rmsd_distribution(
    metrics: pd.DataFrame, run_summary: pd.DataFrame, protocol: str = "pmpnn_esmfold"
) -> None:
    from nomadflow.benchmark.success_criteria import MOTIF_HA_RMSD_MAX

    _plot_rmsd_distribution_by_column(
        metrics,
        run_summary,
        protocol=protocol,
        rmsd_column="motif_all_atom_rmsd",
        threshold=MOTIF_HA_RMSD_MAX,
        title_prefix="Motif HA",
    )


def plot_global_rmsd_distribution(
    metrics: pd.DataFrame, run_summary: pd.DataFrame, protocol: str = "pmpnn_esmfold"
) -> None:
    global_column = "global_bb_rmsd" if protocol == "pmpnn_esmfold" else "global_all_atom_rmsd"
    _plot_rmsd_distribution_by_column(
        metrics,
        run_summary,
        protocol=protocol,
        rmsd_column=global_column,
        threshold=2.0,
        title_prefix="Global",
    )


def build_multimotif_summary(runs: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for run in runs:
        df = run["multimotif_slices"]
        if df.empty:
            continue
        overall = df[(df["slice_type"] == "num_motif_groups") & df["group_pass_rate"].notna()]
        if overall.empty:
            overall = df[df["group_pass_rate"].notna()]
        if overall.empty:
            continue
        rows.append(
            {
                "run_id": run["run_id"],
                "group_pass_rate": overall["group_pass_rate"].mean(),
                "mean_motif_ha_rmsd": overall["mean_motif_ha_rmsd"].mean(),
            }
        )
    return pd.DataFrame(rows)


def build_independent_motif_success(
    multimotif: pd.DataFrame,
    *,
    ha_threshold: float | None = None,
    ca_threshold: float | None = None,
) -> pd.DataFrame:
    """Per-sample success when every motif group independently passes vs native.

    Uses the same per-group thresholds as ``aggregate_multimotif_slices``
    (ProteinZen paper §3.2.1: motif HA < 2 Å and motif Cα < 1 Å). Relative
    inter-group geometry and Protpardelle foldability criteria are intentionally
    ignored.
    """
    from nomadflow.benchmark.success_criteria import MOTIF_CA_RMSD_MAX, MOTIF_HA_RMSD_MAX

    if ha_threshold is None:
        ha_threshold = MOTIF_HA_RMSD_MAX
    if ca_threshold is None:
        ca_threshold = MOTIF_CA_RMSD_MAX

    if multimotif.empty or "record_type" not in multimotif.columns:
        return pd.DataFrame()

    groups = multimotif[multimotif["record_type"] == "group"].copy()
    required = {"run_id", "name", "task", "motif_ha_rmsd", "motif_ca_rmsd"}
    if groups.empty or not required.issubset(groups.columns):
        return pd.DataFrame()

    groups["group_pass"] = (
        (groups["motif_ha_rmsd"] < ha_threshold) & (groups["motif_ca_rmsd"] < ca_threshold)
    )
    sample = (
        groups.groupby(["run_id", "task", "name"], as_index=False)
        .agg(
            num_motif_groups=("group_pass", "size"),
            num_groups_pass=("group_pass", "sum"),
            mean_motif_ha_rmsd=("motif_ha_rmsd", "mean"),
            max_motif_ha_rmsd=("motif_ha_rmsd", "max"),
            mean_motif_ca_rmsd=("motif_ca_rmsd", "mean"),
            max_motif_ca_rmsd=("motif_ca_rmsd", "max"),
        )
    )
    sample["independent_motif_pass"] = sample["num_groups_pass"] == sample["num_motif_groups"]
    return sample


def summarize_independent_motif_success(sample_success: pd.DataFrame) -> pd.DataFrame:
    """Aggregate independent per-motif success to run and task level."""
    if sample_success.empty:
        return pd.DataFrame()

    run_summary = (
        sample_success.groupby("run_id", as_index=False)
        .agg(
            num_samples=("independent_motif_pass", "size"),
            independent_motif_pass_count=("independent_motif_pass", "sum"),
            independent_motif_pass_rate=("independent_motif_pass", "mean"),
            mean_max_group_ha_rmsd=("max_motif_ha_rmsd", "mean"),
            mean_max_group_ca_rmsd=("max_motif_ca_rmsd", "mean"),
        )
    )
    task_any = (
        sample_success.groupby(["run_id", "task"], as_index=False)
        .agg(any_independent_motif_pass=("independent_motif_pass", "any"))
    )
    tasks_with_any = (
        task_any.groupby("run_id", as_index=False)
        .agg(
            num_tasks=("task", "nunique"),
            tasks_with_any_independent_motif=("any_independent_motif_pass", "sum"),
        )
    )
    return run_summary.merge(tasks_with_any, on="run_id", how="left")


def build_independent_motif_task_pivot(sample_success: pd.DataFrame) -> pd.DataFrame:
    """Pivot per-task independent-motif pass rate across runs."""
    if sample_success.empty:
        return pd.DataFrame()
    task = (
        sample_success.groupby(["run_id", "task"], as_index=False)
        .agg(
            num_samples=("independent_motif_pass", "size"),
            independent_motif_pass_rate=("independent_motif_pass", "mean"),
            any_independent_motif_pass=("independent_motif_pass", "any"),
        )
    )
    return task.pivot(
        index="task",
        columns="run_id",
        values="independent_motif_pass_rate",
    )


def plot_independent_motif_success(run_summary: pd.DataFrame) -> None:
    if run_summary.empty or "independent_motif_pass_rate" not in run_summary.columns:
        print("No independent-motif success data for visualization.")
        return

    labels = [short_label(r) for r in run_summary["run_id"]]
    y = np.arange(len(labels))
    values = run_summary["independent_motif_pass_rate"].fillna(0.0).to_numpy()

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.45 * len(run_summary))))
    ax.barh(y, values, color="#55A868")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of designs with all groups passing")
    ax.set_title("Independent per-motif success (design vs native)")
    for i, value in enumerate(values):
        ax.text(min(value + 0.02, 0.95), i, f"{value:.1%}", va="center", fontsize=9)
    plt.tight_layout()
    plt.show()


def build_folded_independent_motif_success(
    group_rmsd: pd.DataFrame,
    *,
    ha_threshold: float | None = None,
    bb_threshold: float | None = None,
) -> pd.DataFrame:
    """Per folded sequence: success if every motif group independently passes.

    Uses ``esmfold/group_rmsd.csv`` (folded vs design), not design-vs-native
    multimotif metrics. Default thresholds match ProteinZen paper §3.2.1:
    HA < 2 Å and Cα/BB < 1 Å.
    """
    from nomadflow.benchmark.success_criteria import MOTIF_BB_RMSD_MAX, MOTIF_HA_RMSD_MAX

    if ha_threshold is None:
        ha_threshold = MOTIF_HA_RMSD_MAX
    if bb_threshold is None:
        bb_threshold = MOTIF_BB_RMSD_MAX

    required = {
        "run_id",
        "name",
        "task",
        "sample",
        "path",
        "protocol",
        "motif_group_all_atom_rmsd",
        "motif_group_bb_rmsd",
    }
    if group_rmsd.empty or not required.issubset(group_rmsd.columns):
        return pd.DataFrame()

    df = group_rmsd.copy()
    if "rmsd_status" in df.columns:
        df = df[df["rmsd_status"] == "ok"]
    if df.empty:
        return pd.DataFrame()

    df["group_pass"] = (
        (df["motif_group_all_atom_rmsd"] < ha_threshold)
        & (df["motif_group_bb_rmsd"] < bb_threshold)
    )
    sample = (
        df.groupby(["run_id", "protocol", "task", "name", "sample", "path"], as_index=False)
        .agg(
            num_motif_groups=("group_pass", "size"),
            num_groups_pass=("group_pass", "sum"),
            mean_motif_ha_rmsd=("motif_group_all_atom_rmsd", "mean"),
            max_motif_ha_rmsd=("motif_group_all_atom_rmsd", "max"),
            mean_motif_bb_rmsd=("motif_group_bb_rmsd", "mean"),
            max_motif_bb_rmsd=("motif_group_bb_rmsd", "max"),
        )
    )
    sample["folded_independent_motif_pass"] = (
        sample["num_groups_pass"] == sample["num_motif_groups"]
    )
    return sample


def summarize_folded_independent_motif_success(sample_success: pd.DataFrame) -> pd.DataFrame:
    """Aggregate folded independent-motif success by run and protocol."""
    if sample_success.empty:
        return pd.DataFrame()

    run_proto = (
        sample_success.groupby(["run_id", "protocol"], as_index=False)
        .agg(
            num_sequences=("folded_independent_motif_pass", "size"),
            folded_indep_pass_count=("folded_independent_motif_pass", "sum"),
            folded_indep_pass_rate=("folded_independent_motif_pass", "mean"),
            mean_max_group_ha_rmsd=("max_motif_ha_rmsd", "mean"),
            mean_max_group_bb_rmsd=("max_motif_bb_rmsd", "mean"),
        )
    )
    # Fraction of designs with at least one passing sequence (useful for PMPNN).
    any_pass = (
        sample_success.groupby(["run_id", "protocol", "name"], as_index=False)
        .agg(any_seq_pass=("folded_independent_motif_pass", "any"))
    )
    design_rate = (
        any_pass.groupby(["run_id", "protocol"], as_index=False)
        .agg(
            num_designs=("name", "nunique"),
            designs_with_any_seq_pass=("any_seq_pass", "sum"),
            design_any_seq_pass_rate=("any_seq_pass", "mean"),
        )
    )
    return run_proto.merge(design_rate, on=["run_id", "protocol"], how="left")


def plot_folded_independent_motif_success(run_summary: pd.DataFrame) -> None:
    if run_summary.empty or "folded_indep_pass_rate" not in run_summary.columns:
        print("No folded independent-motif success data for visualization.")
        return

    protocols = [p for p in ("one_shot", "pmpnn_esmfold") if p in set(run_summary["protocol"])]
    if not protocols:
        protocols = sorted(run_summary["protocol"].unique())

    fig, axes = plt.subplots(
        1,
        len(protocols),
        figsize=(6 * len(protocols), max(3.5, 0.45 * run_summary["run_id"].nunique())),
        sharey=True,
    )
    if len(protocols) == 1:
        axes = [axes]

    run_order = list(dict.fromkeys(run_summary["run_id"].tolist()))
    labels = [short_label(r) for r in run_order]
    y = np.arange(len(labels))

    for ax, protocol in zip(axes, protocols):
        sub = run_summary[run_summary["protocol"] == protocol].set_index("run_id")
        values = np.array(
            [float(sub.loc[r, "folded_indep_pass_rate"]) if r in sub.index else 0.0 for r in run_order]
        )
        ax.barh(y, values, color="#C44E52")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 1)
        ax.set_xlabel("fraction of sequences with all groups passing")
        ax.set_title(f"Folded independent motifs - {protocol}")
        for i, value in enumerate(values):
            ax.text(min(value + 0.02, 0.95), i, f"{value:.1%}", va="center", fontsize=9)
    plt.tight_layout()
    plt.show()


def _safe_filename(text: str) -> str:
    cleaned = re.sub(r"[^\w.\-]+", "_", str(text))
    return cleaned.strip("_") or "sample"


def _resolve_design_pdb(metrics: pd.DataFrame, run_id: str, name: str) -> Path | None:
    if metrics.empty:
        return None
    sub = metrics[(metrics["run_id"] == run_id) & (metrics["name"] == name)]
    if sub.empty or "path" not in sub.columns:
        return None
    if "protocol" in sub.columns and (sub["protocol"] == "one_shot").any():
        path = Path(str(sub.loc[sub["protocol"] == "one_shot", "path"].iloc[0]))
    else:
        path = Path(str(sub["path"].iloc[0]))
    return path if path.is_file() else None


def export_folded_independent_successes(
    sample_success: pd.DataFrame,
    *,
    repo_root: Path,
    metrics: pd.DataFrame | None = None,
    out_dir: Path | None = None,
    protocols: tuple[str, ...] = ("one_shot", "pmpnn_esmfold"),
    copy_design: bool = True,
) -> tuple[Path, pd.DataFrame]:
    """Copy ESMFold PDBs (and optional design PDBs) for independently passing motifs.

    Layout:
      <out_dir>/<run_id>/<protocol>/<name>__sample_<k>__folded.pdb
      <out_dir>/<run_id>/<protocol>/<name>__sample_<k>__design.pdb  (optional)
      <out_dir>/passed_index.csv
    """
    repo_root = Path(repo_root)
    if out_dir is None:
        out_dir = repo_root / "article_results" / "exports" / "folded_indep_motif_success"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if sample_success.empty or "folded_independent_motif_pass" not in sample_success.columns:
        empty = pd.DataFrame()
        empty.to_csv(out_dir / "passed_index.csv", index=False)
        return out_dir, empty

    passed = sample_success[
        sample_success["folded_independent_motif_pass"]
        & sample_success["protocol"].isin(protocols)
    ].copy()
    if passed.empty:
        empty = pd.DataFrame()
        empty.to_csv(out_dir / "passed_index.csv", index=False)
        return out_dir, empty

    metrics = metrics if metrics is not None else pd.DataFrame()
    rows: list[dict] = []
    for row in passed.itertuples(index=False):
        src = Path(str(row.path))
        run_id = str(row.run_id)
        protocol = str(row.protocol)
        name = str(row.name)
        sample = int(row.sample)
        dest_dir = out_dir / _safe_filename(run_id) / _safe_filename(protocol)
        dest_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{_safe_filename(name)}__sample_{sample}"
        folded_dest = dest_dir / f"{stem}__folded.pdb"
        design_dest = None
        copied_folded = False
        copied_design = False

        if src.is_file():
            shutil.copy2(src, folded_dest)
            copied_folded = True

        if copy_design:
            design_src = _resolve_design_pdb(metrics, run_id, name)
            if design_src is not None:
                design_dest = dest_dir / f"{stem}__design.pdb"
                shutil.copy2(design_src, design_dest)
                copied_design = True

        rows.append(
            {
                "run_id": run_id,
                "protocol": protocol,
                "task": getattr(row, "task", ""),
                "name": name,
                "sample": sample,
                "num_motif_groups": getattr(row, "num_motif_groups", None),
                "max_motif_ha_rmsd": getattr(row, "max_motif_ha_rmsd", None),
                "max_motif_bb_rmsd": getattr(row, "max_motif_bb_rmsd", None),
                "src_folded_path": str(src),
                "dest_folded_path": str(folded_dest) if copied_folded else "",
                "dest_design_path": str(design_dest) if copied_design and design_dest else "",
                "copied_folded": copied_folded,
                "copied_design": copied_design,
            }
        )

    index = pd.DataFrame(rows)
    index.to_csv(out_dir / "passed_index.csv", index=False)
    return out_dir, index


def best_run_per_task(task_success: pd.DataFrame) -> pd.DataFrame:
    if task_success.empty:
        return pd.DataFrame()
    best = (
        task_success.sort_values("pmpnn_success_rate", ascending=False)
        .groupby("task", as_index=False)
        .first()[["task", "run_id", "pmpnn_success_rate", "one_shot_success_rate"]]
    )
    return best.rename(columns={"run_id": "best_run_pmpnn"})


def export_tables(repo_root: Path, run_summary: pd.DataFrame, task_success: pd.DataFrame) -> dict[str, Path]:
    out_dir = repo_root / "article_results" / "summary_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "run_success_comparison.csv"
    run_summary.to_csv(summary_path, index=False)

    outputs = {"run_summary": summary_path}
    if not task_success.empty:
        task_path = out_dir / "task_success_comparison.csv"
        task_success.to_csv(task_path, index=False)
        outputs["task_success"] = task_path
    return outputs
