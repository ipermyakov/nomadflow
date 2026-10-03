"""Build unified benchmark analysis tables from ESMFold RMSD outputs."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from nomadflow.benchmark.success_criteria import (
    all_groups_pass,
    one_shot_pass,
    pmpnn_esmfold_pass,
)

_ONE_SHOT_PATH_MARKER = "model_name=v_48_020"
_PMPNN_SAMPLE_LABEL_RE = re.compile(r"pmpnn_sample_(\d+)$")


def infer_group_rmsd_protocol(path: str) -> str:
    if _ONE_SHOT_PATH_MARKER in str(path):
        return "one_shot"
    return "pmpnn_esmfold"


def _normalize_group_id(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            as_int = int(float(text))
            return str(as_int)
        except ValueError:
            return text
    try:
        as_int = int(float(text))
        if float(text) == float(as_int):
            return str(as_int)
    except ValueError:
        pass
    return text


def _group_protocol_frame(group_rmsd: pd.DataFrame, protocol: str) -> pd.DataFrame:
    if group_rmsd is None or group_rmsd.empty:
        return pd.DataFrame()
    df = group_rmsd.copy()
    if "protocol" not in df.columns:
        if "path" in df.columns:
            df["protocol"] = df["path"].map(infer_group_rmsd_protocol)
        else:
            return pd.DataFrame()
    df = df[df["protocol"] == protocol].copy()
    if df.empty:
        return df
    if "group_id" in df.columns:
        df["group_id"] = df["group_id"].map(_normalize_group_id)
    if (
        protocol == "pmpnn_esmfold"
        and "sample" not in df.columns
        and "label" in df.columns
    ):
        extracted = df["label"].astype(str).map(_sample_from_pmpnn_label)
        if extracted.notna().any():
            df["sample"] = extracted
    return df


def _sample_from_pmpnn_label(label: Any) -> int | None:
    match = _PMPNN_SAMPLE_LABEL_RE.search(str(label).strip())
    if match is None:
        return None
    return int(match.group(1))


def _group_ha_bb_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Prefer native sidecar HA/Cα; else fold-vs-design group RMSDs.

    Paper §3.2.1 motif backbone cut is Cα < 1 Å. Group tables without
    ``motif_ca_rmsd`` keep the existing BB column at the same 1.0 Å cut.
    """
    ha = "motif_ha_rmsd" if "motif_ha_rmsd" in df.columns else "motif_group_all_atom_rmsd"
    bb = "motif_ca_rmsd" if "motif_ca_rmsd" in df.columns else "motif_group_bb_rmsd"
    return ha, bb


def _sample_group_pass_table(
    group_rmsd: pd.DataFrame,
    *,
    protocol: str,
    expected_groups: dict[str, int] | None = None,
) -> pd.DataFrame:
    """One row per sample with wide per-group HA/BB and pass_all_groups."""
    df = _group_protocol_frame(group_rmsd, protocol)
    empty_cols = {
        "name": pd.Series(dtype=str),
        "num_groups_scored": pd.Series(dtype=int),
        "num_groups_expected": pd.Series(dtype=int),
        "pass_all_groups": pd.Series(dtype=bool),
    }
    if df.empty:
        empty = pd.DataFrame(empty_cols)
        if protocol == "pmpnn_esmfold":
            empty["sample"] = pd.Series(dtype=int)
        return empty

    if "rmsd_status" in df.columns:
        ok = df["rmsd_status"].astype(str) == "ok"
        df = df[ok].copy()
    if df.empty:
        empty = pd.DataFrame(empty_cols)
        if protocol == "pmpnn_esmfold":
            empty["sample"] = pd.Series(dtype=int)
        return empty

    group_keys = ["name"]
    if protocol == "pmpnn_esmfold" and "sample" in df.columns:
        group_keys = ["name", "sample"]

    ha_col, bb_col = _group_ha_bb_columns(df)
    wide_ha = df.pivot_table(
        index=group_keys,
        columns="group_id",
        values=ha_col,
        aggfunc="first",
    )
    wide_bb = df.pivot_table(
        index=group_keys,
        columns="group_id",
        values=bb_col,
        aggfunc="first",
    )
    wide_ha = wide_ha.rename(columns=lambda gid: f"group_{gid}_ha")
    wide_bb = wide_bb.rename(columns=lambda gid: f"group_{gid}_bb")
    wide = wide_ha.join(wide_bb, how="outer")
    wide = wide.reset_index()

    ha_cols = [c for c in wide.columns if c.startswith("group_") and c.endswith("_ha")]
    bb_cols = [c for c in wide.columns if c.startswith("group_") and c.endswith("_bb")]
    expected_groups = expected_groups or {}

    num_scored: list[int] = []
    num_expected: list[int] = []
    pass_flags: list[bool] = []
    for _, row in wide.iterrows():
        name = str(row["name"])
        expected = int(expected_groups.get(name, 0))
        ha_vals = [float(row[c]) for c in ha_cols if pd.notna(row[c])]
        bb_vals = [float(row[c]) for c in bb_cols if pd.notna(row[c])]
        scored = min(len(ha_vals), len(bb_vals))
        num_scored.append(scored)
        num_expected.append(expected)
        pass_flags.append(
            all_groups_pass(ha_vals, bb_vals, expected_n_groups=expected)
        )

    wide["num_groups_scored"] = num_scored
    wide["num_groups_expected"] = num_expected
    wide["pass_all_groups"] = pass_flags
    return wide


def attach_group_success_columns(
    df: pd.DataFrame,
    group_rmsd: pd.DataFrame | None,
    *,
    protocol: str,
    expected_groups: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Join per-group HA/BB columns and pass_all_groups onto a protocol table."""
    out = df.copy()
    if out.empty:
        out["pass_all_groups"] = pd.Series(dtype=bool)
        out["num_groups_scored"] = pd.Series(dtype=int)
        out["num_groups_expected"] = pd.Series(dtype=int)
        return out

    sample_groups = _sample_group_pass_table(
        group_rmsd if group_rmsd is not None else pd.DataFrame(),
        protocol=protocol,
        expected_groups=expected_groups,
    )
    join_cols = ["name"]
    if protocol == "pmpnn_esmfold" and "sample" in out.columns and "sample" in sample_groups.columns:
        join_cols = ["name", "sample"]
    if sample_groups.empty:
        out["pass_all_groups"] = False
        out["num_groups_scored"] = 0
        out["num_groups_expected"] = [
            int((expected_groups or {}).get(str(name), 0)) for name in out["name"]
        ]
        return out

    extra_cols = [c for c in sample_groups.columns if c not in join_cols]
    out = out.merge(sample_groups[join_cols + extra_cols], on=join_cols, how="left")
    if "pass_all_groups" not in out.columns:
        out["pass_all_groups"] = False
    else:
        out["pass_all_groups"] = out["pass_all_groups"].where(
            out["pass_all_groups"].notna(), False
        ).astype(bool)
    if "num_groups_scored" not in out.columns:
        out["num_groups_scored"] = 0
    else:
        out["num_groups_scored"] = out["num_groups_scored"].fillna(0).astype(int)
    if "num_groups_expected" not in out.columns:
        out["num_groups_expected"] = 0
    if expected_groups:
        missing_expected = out["num_groups_expected"].isna()
        if bool(missing_expected.any()):
            out.loc[missing_expected, "num_groups_expected"] = [
                int(expected_groups.get(str(name), 0))
                for name in out.loc[missing_expected, "name"]
            ]
    out["num_groups_expected"] = out["num_groups_expected"].fillna(0).astype(int)
    return out


def build_metrics_table(
    folding_df: pd.DataFrame,
    sc_df: pd.DataFrame,
    group_rmsd: pd.DataFrame | None = None,
    *,
    expected_groups: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Merge one-shot and PMPNN+ESMFold metrics into a long-format table."""
    one_shot = attach_group_success_columns(
        folding_df,
        group_rmsd,
        protocol="one_shot",
        expected_groups=expected_groups,
    )
    one_shot["protocol"] = "one_shot"
    one_shot["pass_pooled_legacy"] = one_shot.apply(one_shot_pass, axis=1)
    one_shot["pass"] = one_shot["pass_pooled_legacy"]
    one_shot["pass_final"] = one_shot["pass_all_groups"]

    pmpnn = attach_group_success_columns(
        sc_df,
        group_rmsd,
        protocol="pmpnn_esmfold",
        expected_groups=expected_groups,
    )
    pmpnn["protocol"] = "pmpnn_esmfold"
    pmpnn["pass_pooled_legacy"] = pmpnn.apply(pmpnn_esmfold_pass, axis=1)
    pmpnn["pass"] = pmpnn["pass_pooled_legacy"]
    pmpnn["pass_final"] = pmpnn["pass_all_groups"]

    flag_cols = ["pass", "pass_pooled_legacy", "pass_all_groups", "pass_final"]
    extra_cols: list[str] = []
    seen: set[str] = set()
    for col in list(one_shot.columns) + list(pmpnn.columns):
        if col in flag_cols or col in seen:
            continue
        seen.add(col)
        extra_cols.append(col)
    extra_cols = sorted(extra_cols)
    metrics = pd.concat(
        [
            one_shot.reindex(columns=extra_cols + flag_cols),
            pmpnn.reindex(columns=extra_cols + flag_cols),
        ],
        ignore_index=True,
    )
    return metrics


def build_task_success_table(
    folding_df: pd.DataFrame,
    sc_df: pd.DataFrame,
    *,
    num_samples_per_task: int,
    task_counts: dict[str, int] | None = None,
    group_rmsd: pd.DataFrame | None = None,
    expected_groups: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Aggregate per-task success counts and rates.

    ``one_shot_success_count`` / ``pmpnn_final_count`` come from ``pass_final``
    (every motif group vs native). Pooled-only counts stay diagnostic.
    """
    from nomadflow.benchmark.success_criteria import (
        filter_one_shot,
        filter_one_shot_final,
        filter_pmpnn_esmfold,
    )

    one_shot = attach_group_success_columns(
        folding_df,
        group_rmsd,
        protocol="one_shot",
        expected_groups=expected_groups,
    )
    if not one_shot.empty:
        one_shot["pass_pooled_legacy"] = one_shot.apply(one_shot_pass, axis=1)
        one_shot["pass_final"] = one_shot["pass_all_groups"]

    pmpnn = attach_group_success_columns(
        sc_df,
        group_rmsd,
        protocol="pmpnn_esmfold",
        expected_groups=expected_groups,
    )
    if not pmpnn.empty:
        pmpnn["pass_pooled_legacy"] = pmpnn.apply(pmpnn_esmfold_pass, axis=1)
        pmpnn["pass_final"] = pmpnn["pass_all_groups"]

    all_tasks = sorted(
        set(one_shot.get("task", pd.Series(dtype=str)).tolist())
        | set(pmpnn.get("task", pd.Series(dtype=str)).tolist())
    )

    one_shot_final = filter_one_shot_final(one_shot) if not one_shot.empty else one_shot
    one_shot_legacy = filter_one_shot(one_shot) if not one_shot.empty else one_shot
    one_shot_final_counts = (
        one_shot_final.groupby("task").size().to_dict() if not one_shot_final.empty else {}
    )
    one_shot_legacy_counts = (
        one_shot_legacy.groupby("task").size().to_dict() if not one_shot_legacy.empty else {}
    )
    one_shot_groups_counts = (
        one_shot[one_shot["pass_all_groups"]].groupby("task").size().to_dict()
        if not one_shot.empty
        else {}
    )

    pmpnn_passed = filter_pmpnn_esmfold(pmpnn) if not pmpnn.empty else pmpnn
    # Design-level: a design succeeds if at least one sequence passed. Do not
    # subsample sequences — Foldseek already clusters one PDB per passing name.
    pmpnn_counts = (
        pmpnn_passed.groupby("task")["name"].nunique().to_dict()
        if not pmpnn_passed.empty
        else {}
    )
    pmpnn_final = pmpnn[pmpnn["pass_final"]] if not pmpnn.empty else pmpnn
    pmpnn_final_counts = (
        pmpnn_final.groupby("task")["name"].nunique().to_dict()
        if not pmpnn_final.empty
        else {}
    )

    rows = []
    for task in all_tasks:
        if task_counts is not None and task in task_counts:
            denom = task_counts[task]
        else:
            denom = num_samples_per_task
        one_shot_success = int(one_shot_final_counts.get(task, 0))
        one_shot_legacy_count = int(one_shot_legacy_counts.get(task, 0))
        one_shot_groups = int(one_shot_groups_counts.get(task, 0))
        pmpnn_success = int(pmpnn_counts.get(task, 0))
        pmpnn_final_success = int(pmpnn_final_counts.get(task, 0))
        rows.append(
            {
                "task": task,
                "num_samples": denom,
                "one_shot_success_count": one_shot_success,
                "one_shot_success_rate": one_shot_success / denom if denom else 0.0,
                "one_shot_pooled_legacy_count": one_shot_legacy_count,
                "one_shot_pooled_legacy_rate": one_shot_legacy_count / denom if denom else 0.0,
                "one_shot_all_groups_count": one_shot_groups,
                "pmpnn_success_count": pmpnn_success,
                "pmpnn_success_rate": pmpnn_success / denom if denom else 0.0,
                "pmpnn_final_count": pmpnn_final_success,
                "pmpnn_final_rate": pmpnn_final_success / denom if denom else 0.0,
                "any_one_shot": one_shot_success > 0,
                "any_pmpnn": pmpnn_final_success > 0,
            }
        )
    return pd.DataFrame(rows)
