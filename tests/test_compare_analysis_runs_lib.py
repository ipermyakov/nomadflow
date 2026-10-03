from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.analysis.compare_analysis_runs_lib import (
    CATALYTIC_PLUS_STRUCTURAL_TASKS,
    aggregate_pmpnn_layout_to_design,
    attach_layout_pmpnn_pass,
    attach_unique_counts,
    build_paper_style_table,
    discover_runs,
    filter_runs_to_tasks,
    layout_design_coverage_table,
    layout_mds_coordinates,
    load_group_rmsd,
    load_layout_mds_coordinates,
    load_layout_mobility_table,
    load_unique_success_for_run,
    merge_runs_by_campaign,
    merge_unique_success,
    ordered_task_labels,
    plot_layout_fig_c,
    plot_matched_layout_metrics,
    plot_task_success_points,
    remap_layout_pair_tasks,
    resolve_analysis_run_dir,
    resolve_layout_mds_coordinates,
    resolve_layout_mobility,
    summarize_layout_by_task,
    summarize_layout_mobility,
    task_partition,
    tasks_in_runs,
)


def _run(run_id: str, tasks: list[str], *, unique: dict | None = None) -> dict:
    rows = []
    for task in tasks:
        rows.append(
            {
                "run_id": run_id,
                "task": task,
                "num_samples": 10,
                "one_shot_success_count": 2,
                "one_shot_success_rate": 0.2,
                "pmpnn_success_count": 4,
                "pmpnn_success_rate": 0.4,
                "pmpnn_final_count": 3,
                "pmpnn_final_rate": 0.3,
                "any_one_shot": True,
                "any_pmpnn": True,
            }
        )
    return {
        "run_id": run_id,
        "run_dir": run_id,
        "metrics": pd.DataFrame(
            [{"run_id": run_id, "task": task, "protocol": "one_shot", "pass": True} for task in tasks]
        ),
        "task_success": pd.DataFrame(rows),
        "multimotif": pd.DataFrame(),
        "multimotif_slices": pd.DataFrame(),
        "group_rmsd": pd.DataFrame(),
        "layout_pairs": pd.DataFrame(),
        "layout_pmpnn_pairs": pd.DataFrame(),
        "layout_coverage": pd.DataFrame(),
        "unique_success": unique
        or {
            "skipped": False,
            "one_shot": {task: 1 for task in tasks},
            "pmpnn_esmfold": {task: 2 for task in tasks},
        },
    }


def test_task_partition_labels_sampling_split():
    assert task_partition("1PRW") == "structural_large"
    assert task_partition("1BCF_2grp") == "structural_large"
    assert task_partition("1QJG_ext10") == "catalytic_plus_structural"
    assert task_partition("1QJG_ext16") == "catalytic_plus_structural"
    assert task_partition("1YCR") == "other"
    assert "1QJG_ext16" in CATALYTIC_PLUS_STRUCTURAL_TASKS
    assert "1QJG_ext20" not in CATALYTIC_PLUS_STRUCTURAL_TASKS


def test_merge_campaigns_concatenates_disjoint_tasks():
    se3 = _run("se3", ["1PRW", "1BCF"])
    full = _run("full", ["1QJG_ext10"])
    merged = merge_runs_by_campaign(
        [se3, full],
        {"stage1": ["se3", "full"]},
    )
    assert len(merged) == 1
    assert set(merged[0]["task_success"]["task"]) == {"1PRW", "1BCF", "1QJG_ext10"}
    assert set(merged[0]["task_success"]["run_id"]) == {"stage1"}
    assert merged[0]["unique_success"]["one_shot"]["1PRW"] == 1
    assert merged[0]["unique_success"]["one_shot"]["1QJG_ext10"] == 1


def test_merge_campaigns_rejects_overlapping_tasks():
    left = _run("se3", ["1PRW"])
    right = _run("full", ["1PRW"])
    with pytest.raises(ValueError, match="Overlapping tasks"):
        merge_runs_by_campaign([left, right], {"stage1": ["se3", "full"]})


def _coverage_rows(task: str, reasons: list[str], *, eval_error: str = "") -> list[dict]:
    rows = []
    for i, reason in enumerate(reasons):
        rows.append(
            {
                "name": f"{task}_{i}",
                "task": task,
                "reason": reason,
                "eval_error": eval_error if reason == "eval_error" else "",
                "detail": eval_error if reason == "eval_error" else "",
                "num_groups_expected": 2,
                "n_layout_pairs": 1 if reason == "ok" else 0,
                "has_all_group_rmsds": reason == "ok",
            }
        )
    return rows


def test_layout_design_coverage_separates_eval_error_from_missing_residue():
    tlock = _run("tlock05_stage2", ["4JHW", "1PRW", "1BCF_2grp"])
    tlock["task_success"]["num_samples"] = 2
    tlock["layout_coverage"] = pd.DataFrame(
        _coverage_rows("4JHW", ["missing_residue_in_design", "missing_residue_in_design"])
        + _coverage_rows(
            "1PRW",
            ["ok", "eval_error"],
            eval_error="stage2_stage1_residue_count_mismatch",
        )
        + _coverage_rows("1BCF_2grp", ["ok", "ok"])
    )
    baseline = _run("one_motif_proteinzen", ["4JHW", "1PRW", "1YCR"])
    baseline["task_success"]["num_samples"] = 2
    baseline["layout_coverage"] = pd.DataFrame(
        _coverage_rows("4JHW", ["ok", "ok"])
        + _coverage_rows("1PRW", ["ok", "ok"])
        + _coverage_rows("1YCR", ["not_multigroup", "not_multigroup"])
    )
    union = {"4JHW", "1PRW", "1BCF_2grp"}
    filtered = filter_runs_to_tasks([tlock, baseline], union)
    table = layout_design_coverage_table(filtered, union).set_index(["task", "run_id"])

    assert "1YCR" not in set(table.index.get_level_values("task"))
    jhw_s2 = table.loc[("4JHW", "tlock05_stage2")]
    assert int(jhw_s2["n_expected"]) == 2
    assert int(jhw_s2["n_layout_ok"]) == 0
    assert int(jhw_s2["n_missing_residue"]) == 2
    assert int(jhw_s2["n_eval_error"]) == 0
    assert bool(jhw_s2["layout_incomplete"])
    assert "honest fail" in str(jhw_s2["note"])
    assert "4JHW" in str(jhw_s2["note"])

    prw_s2 = table.loc[("1PRW", "tlock05_stage2")]
    assert int(prw_s2["n_layout_ok"]) == 1
    assert int(prw_s2["n_eval_error"]) == 1
    assert int(prw_s2["n_missing_residue"]) == 0
    assert "stage2_stage1_residue_count_mismatch" in str(prw_s2["note"])
    assert "not model fail" in str(prw_s2["note"])

    jhw_om = table.loc[("4JHW", "one_motif_proteinzen")]
    assert int(jhw_om["n_layout_ok"]) == 2
    assert int(jhw_om["n_missing_residue"]) == 0
    assert not bool(jhw_om["layout_incomplete"])

    grp = table.loc[("1BCF_2grp", "one_motif_proteinzen")]
    assert not bool(grp["in_campaign"])
    assert int(grp["n_expected"]) == 0
    assert "tlock-only" in str(grp["note"])


def test_layout_design_coverage_remaps_stage2_parent_task():
    run = _run("tlock05_stage2", ["1PRW"])
    run["task_success"]["num_samples"] = 1
    run["layout_coverage"] = pd.DataFrame(
        [
            {
                "name": "x",
                "task": "1PRW__len_80_protein_id0_0_0",
                "reason": "ok",
                "eval_error": "",
                "num_groups_expected": 2,
                "has_all_group_rmsds": True,
            }
        ]
    )
    table = layout_design_coverage_table([run], {"1PRW"})
    assert int(table.iloc[0]["n_layout_ok"]) == 1
    assert int(table.iloc[0]["n_expected"]) == 1


def test_filter_runs_and_paper_table_matched_tasks():
    tlock = _run("tlock", ["1PRW", "1BCF_2grp", "1QJG_ext10"])
    baseline = _run("one_motif", ["1PRW", "1QJG_ext10", "1YCR"])
    union = tasks_in_runs([tlock])
    filtered = filter_runs_to_tasks([tlock, baseline], union)
    assert tasks_in_runs(filtered, ["one_motif"]) == {"1PRW", "1QJG_ext10"}
    assert "1YCR" not in set(filtered[1]["metrics"]["task"])
    assert "1YCR" not in filtered[1]["unique_success"]["one_shot"]

    ts = attach_unique_counts(
        pd.concat([run["task_success"] for run in filtered], ignore_index=True),
        filtered,
    )
    table = build_paper_style_table(
        ts,
        protocol="one_shot",
        run_order=["tlock", "one_motif"],
        add_total=False,
    )
    assert "1BCF_2grp" in table.index
    assert pd.isna(table.loc["1BCF_2grp", ("one_motif", "All")])
    assert table.loc["1PRW", ("tlock", "All")] == 2


def test_paper_table_pmpnn_unique_from_json():
    tlock = _run("tlock", ["1PRW", "1BCF_2grp"])
    ts = attach_unique_counts(tlock["task_success"], [tlock])
    table = build_paper_style_table(
        ts,
        protocol="pmpnn_esmfold",
        run_order=["tlock"],
        add_total=False,
    )
    assert table.loc["1PRW", ("tlock", "All")] == 3
    assert table.loc["1PRW", ("tlock", "Unique")] == 2


def test_paper_table_pmpnn_unique_na_when_missing():
    tlock = _run(
        "tlock",
        ["1PRW"],
        unique={"skipped": False, "one_shot": {"1PRW": 1}, "pmpnn_esmfold": {}},
    )
    ts = attach_unique_counts(tlock["task_success"], [tlock])
    assert pd.isna(ts.loc[0, "pmpnn_unique"])
    assert ts.loc[0, "one_shot_unique"] == 1
    table = build_paper_style_table(
        ts,
        protocol="pmpnn_esmfold",
        run_order=["tlock"],
        add_total=True,
    )
    assert pd.isna(table.loc["1PRW", ("tlock", "Unique")])
    assert pd.isna(table.loc["TOTAL", ("tlock", "Unique")])
    assert table.loc["1PRW", ("tlock", "All")] == 3
    assert table.loc["TOTAL", ("tlock", "All")] == 3


def test_merge_unique_success_sums_counts():
    merged = merge_unique_success(
        [
            {"skipped": False, "one_shot": {"A": 1}, "pmpnn_esmfold": {"A": 3}},
            {"skipped": False, "one_shot": {"B": 2}, "pmpnn_esmfold": {"B": 4}},
        ]
    )
    assert merged["one_shot"] == {"A": 1, "B": 2}
    assert merged["pmpnn_esmfold"] == {"A": 3, "B": 4}
    assert merged["skipped"] is False


def test_plot_task_success_points_renders(monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    shown = []
    monkeypatch.setattr(
        "scripts.analysis.compare_analysis_runs_lib.plt.show",
        lambda: shown.append(True),
    )
    tlock = _run("tlock", ["1PRW", "1BCF_2grp", "1QJG_ext10"])
    baseline = _run("one_motif", ["1PRW", "1QJG_ext10"])
    ts = attach_unique_counts(
        pd.concat([tlock["task_success"], baseline["task_success"]], ignore_index=True),
        [tlock, baseline],
    )
    tasks, partitions = ordered_task_labels(ts)
    assert tasks == ["1BCF_2grp", "1PRW", "1QJG_ext10"]
    assert partitions[0] == "structural (large motifs)"
    plot_task_success_points(
        ts,
        protocol="pmpnn_esmfold",
        run_order=["tlock", "one_motif"],
        title="test",
    )
    assert shown == [True]


def test_discover_runs_prefers_criteria_v2(tmp_path):
    analysis_root = tmp_path / "analysis_runs"
    legacy = analysis_root / "one_motif_proteinzen"
    v2 = analysis_root / "one_motif_proteinzen__criteria_v2"
    other = analysis_root / "other_run"
    for folder in (legacy, v2, other):
        folder.mkdir(parents=True)
        (folder / "metrics.csv").write_text("task\n1PRW\n", encoding="utf-8")
    found = [p.name for p in discover_runs(analysis_root)]
    assert "one_motif_proteinzen__criteria_v2" in found
    assert "one_motif_proteinzen" not in found
    assert "other_run" in found
    assert resolve_analysis_run_dir(analysis_root, "one_motif_proteinzen").name.endswith(
        "__criteria_v2"
    )


def test_discover_runs_prefers_criteria_v3_over_v2(tmp_path):
    analysis_root = tmp_path / "analysis_runs"
    legacy = analysis_root / "one_motif_proteinzen"
    v2 = analysis_root / "one_motif_proteinzen__criteria_v2"
    v3 = analysis_root / "one_motif_proteinzen__criteria_v3"
    for folder in (legacy, v2, v3):
        folder.mkdir(parents=True)
        (folder / "metrics.csv").write_text("task\n1PRW\n", encoding="utf-8")
    found = [p.name for p in discover_runs(analysis_root)]
    assert found == ["one_motif_proteinzen__criteria_v3"]
    assert resolve_analysis_run_dir(analysis_root, "one_motif_proteinzen").name.endswith(
        "__criteria_v3"
    )
    assert resolve_analysis_run_dir(
        analysis_root, "one_motif_proteinzen", criteria="v2"
    ).name.endswith("__criteria_v2")
    assert resolve_analysis_run_dir(
        analysis_root, "one_motif_proteinzen", criteria="v3"
    ).name.endswith("__criteria_v3")


def test_resolve_analysis_run_dir_falls_back_when_v3_has_no_metrics(tmp_path):
    analysis_root = tmp_path / "analysis_runs"
    v2 = analysis_root / "one_motif_proteinzen__criteria_v2"
    v3 = analysis_root / "one_motif_proteinzen__criteria_v3"
    v2.mkdir(parents=True)
    v3.mkdir(parents=True)
    (v2 / "metrics.csv").write_text("task\n1PRW\n", encoding="utf-8")
    (v3 / "group_fold_vs_native.csv").write_text("task\n1PRW\n", encoding="utf-8")
    assert (
        resolve_analysis_run_dir(analysis_root, "one_motif_proteinzen", criteria="v3").name
        == "one_motif_proteinzen__criteria_v2"
    )


def test_merge_campaigns_matches_criteria_suffixes():
    se3 = _run("0c62dc8a__tlock05_se3_stage1__criteria_v2", ["1PRW"])
    full = _run("0c62dc8a__tlock05_full_stage1", ["1QJG_ext10"])
    merged = merge_runs_by_campaign(
        [se3, full],
        {
            "tlock05_stage1": [
                "0c62dc8a__tlock05_se3_stage1",
                "0c62dc8a__tlock05_full_stage1__criteria_v2",
            ]
        },
    )
    assert set(merged[0]["task_success"]["task"]) == {"1PRW", "1QJG_ext10"}


def test_tasks_in_runs_matches_criteria_suffixes():
    se3 = _run("0c62dc8a__tlock05_se3_stage1__criteria_v2", ["1PRW"])
    stage2 = _run("0c62dc8a__tlock05_se3_stage2", ["1BCF"])
    tasks = tasks_in_runs(
        [se3, stage2],
        ["0c62dc8a__tlock05_se3_stage1", "0c62dc8a__tlock05_se3_stage2__criteria_v2"],
    )
    assert tasks == {"1PRW", "1BCF"}


def test_load_unique_success_prefers_all_groups(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "unique_success.json").write_text(
        '{"one_shot": {"1PRW": 9}, "skipped": false}', encoding="utf-8"
    )
    (run_dir / "unique_success_all_groups.json").write_text(
        '{"one_shot": {"1PRW": 4}, "skipped": false}', encoding="utf-8"
    )
    payload = load_unique_success_for_run(run_dir)
    assert payload["one_shot"]["1PRW"] == 4
    assert payload["pmpnn_esmfold"] == {}


def _layout_rows(run_id: str, task: str, trans: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": run_id,
                "name": f"{task}_{i}",
                "task": task,
                "group_i": 1,
                "group_j": 2,
                "excursion_trans_A": value,
                "excursion_rot_deg": value * 2,
                "retention_trans_A": value / 2,
                "retention_rot_deg": value,
                "pass_final": i == 0,
            }
            for i, value in enumerate(trans)
        ]
    )


def test_remap_layout_pair_tasks_collapses_stage2_sample_ids():
    df = pd.DataFrame(
        {
            "task": ["1QJG_ext10__len_78_protein_id0_0_0", "1PRW"],
            "excursion_trans_A": [1.0, 2.0],
        }
    )
    remapped = remap_layout_pair_tasks(df)
    assert remapped["task"].tolist() == ["1QJG_ext10", "1PRW"]


def test_filter_runs_keeps_stage2_layout_after_parent_remap():
    run = _run("tlock05_stage2", ["1QJG_ext10"])
    run["layout_pairs"] = remap_layout_pair_tasks(
        pd.DataFrame(
            {
                "run_id": ["tlock05_stage2"],
                "task": ["1QJG_ext10__len_78_protein_id0_0_0"],
                "excursion_trans_A": [12.0],
            }
        )
    )
    filtered = filter_runs_to_tasks([run], {"1QJG_ext10"})
    assert not filtered[0]["layout_pairs"].empty
    assert filtered[0]["layout_pairs"]["task"].tolist() == ["1QJG_ext10"]


def test_summarize_layout_by_task_medians():
    pairs = pd.concat(
        [
            _layout_rows("tlock", "1PRW", [10.0, 20.0, 30.0]),
            _layout_rows("one_motif", "1PRW", [0.0, 0.2]),
        ],
        ignore_index=True,
    )
    summary = summarize_layout_by_task(pairs)
    tlock = summary.set_index(["run_id", "task"]).loc[("tlock", "1PRW")]
    assert tlock["n_pairs"] == 3
    assert tlock["n_samples"] == 3
    assert tlock["excursion_trans_A_median"] == 20.0
    assert tlock["n_pass_final_pairs"] == 1


def test_plot_matched_layout_metrics_renders(monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    shown = []
    monkeypatch.setattr(
        "scripts.analysis.compare_analysis_runs_lib.plt.show",
        lambda: shown.append(True),
    )
    pairs = pd.concat(
        [
            _layout_rows("tlock", "1PRW", [10.0, 20.0]),
            _layout_rows("one_motif", "1PRW", [0.1, 0.2]),
        ],
        ignore_index=True,
    )
    summary = plot_matched_layout_metrics(pairs, run_order=["tlock", "one_motif"])
    assert not summary.empty
    assert shown == [True] * 6


def test_plot_matched_layout_metrics_pass_final_only(monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    shown = []
    monkeypatch.setattr(
        "scripts.analysis.compare_analysis_runs_lib.plt.show",
        lambda: shown.append(True),
    )
    pairs = pd.concat(
        [
            _layout_rows("tlock", "1PRW", [10.0, 20.0, 30.0]),
            _layout_rows("one_motif", "1PRW", [0.1, 0.2]),
        ],
        ignore_index=True,
    )
    summary = plot_matched_layout_metrics(
        pairs, run_order=["tlock", "one_motif"], pass_final_only=True
    )
    assert list(summary["n_pairs"]) == [1, 1]
    assert shown == [True] * 5


def test_attach_layout_pmpnn_pass_any_sequence():
    layout = _layout_rows("tlock", "1PRW", [1.0, 2.0])
    metrics = pd.DataFrame(
        [
            {"protocol": "pmpnn_esmfold", "name": "1PRW_0", "pass_final": False},
            {"protocol": "pmpnn_esmfold", "name": "1PRW_0", "pass_final": True},
            {"protocol": "one_shot", "name": "1PRW_0", "pass_final": False},
            {"protocol": "pmpnn_esmfold", "name": "1PRW_1", "pass_final": False},
        ]
    )
    out = attach_layout_pmpnn_pass(layout, metrics)
    assert bool(out.loc[out["name"] == "1PRW_0", "pmpnn_pass_final"].all())
    assert not bool(out.loc[out["name"] == "1PRW_1", "pmpnn_pass_final"].any())


def test_plot_matched_layout_metrics_pmpnn_passed_only(monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    shown = []
    monkeypatch.setattr(
        "scripts.analysis.compare_analysis_runs_lib.plt.show",
        lambda: shown.append(True),
    )
    pairs = _layout_rows("tlock", "1PRW", [10.0, 20.0, 30.0])
    pairs["pmpnn_pass_final"] = [True, False, False]
    summary = plot_matched_layout_metrics(
        pairs, run_order=["tlock"], passed_only="pmpnn_esmfold"
    )
    assert list(summary["n_pairs"]) == [1]
    assert shown == [True] * 5


def test_aggregate_pmpnn_layout_to_design_medians_retention():
    rows = []
    for sample, ret in ((0, 2.0), (1, 8.0), (2, 4.0)):
        rows.append(
            {
                "run_id": "tlock",
                "name": "design_a",
                "task": "1PRW",
                "group_i": 1,
                "group_j": 2,
                "pmpnn_sample": sample,
                "excursion_trans_A": 10.0,
                "excursion_rot_deg": 20.0,
                "retention_trans_A": ret,
                "retention_rot_deg": ret * 2,
                "pass_final": True,
            }
        )
    df = pd.DataFrame(rows)
    out = aggregate_pmpnn_layout_to_design(df)
    assert len(out) == 1
    assert int(out.iloc[0]["n_pmpnn_sequences"]) == 3
    assert float(out.iloc[0]["retention_trans_A"]) == 4.0
    assert float(out.iloc[0]["excursion_trans_A"]) == 10.0
    summary = summarize_layout_by_task(df)
    assert int(summary.iloc[0]["n_pairs"]) == 1
    assert int(summary.iloc[0]["n_samples"]) == 1


def _pose_row(run_id, name, task, tx, *, pass_final=True):
    return {
        "run_id": run_id,
        "name": name,
        "task": task,
        "group_i": 1,
        "group_j": 2,
        "excursion_trans_A": abs(tx),
        "excursion_rot_deg": 0.0,
        "retention_trans_A": 0.1,
        "retention_rot_deg": 0.0,
        "pass_final": pass_final,
        "design_tij_tx": tx,
        "design_tij_ty": 0.0,
        "design_tij_tz": 0.0,
        "design_tij_qw": 1.0,
        "design_tij_qx": 0.0,
        "design_tij_qy": 0.0,
        "design_tij_qz": 0.0,
    }


def test_summarize_layout_mobility_pairwise_and_weak_n():
    rows = [
        _pose_row("tlock", "a", "1PRW", 0.0, pass_final=True),
        _pose_row("tlock", "b", "1PRW", 10.0, pass_final=True),
        _pose_row("tlock", "c", "1PRW", 10.0, pass_final=False),
    ]
    mobility = summarize_layout_mobility(pd.DataFrame(rows), n_boot=20, random_state=0)
    all_row = mobility[(mobility["subset"] == "all")].iloc[0]
    final_row = mobility[(mobility["subset"] == "pass_final")].iloc[0]
    assert int(all_row["n_designs"]) == 3
    assert bool(all_row["pairwise_defined"])
    assert bool(all_row["weak_n"])
    assert float(all_row["pairwise_trans_median"]) == pytest.approx(10.0, abs=0.05)
    assert int(final_row["n_designs"]) == 2
    assert int(final_row["num_design_pairs"]) == 1


def test_plot_layout_fig_c_renders(monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    shown = []
    monkeypatch.setattr(
        "scripts.analysis.compare_analysis_runs_lib.plt.show",
        lambda: shown.append(True),
    )
    rows = [
        _pose_row("tlock", "a", "1PRW", 0.0),
        _pose_row("tlock", "b", "1PRW", 8.0),
        _pose_row("one_motif", "a", "1PRW", 1.0),
        _pose_row("one_motif", "b", "1PRW", 1.2),
    ]
    mobility = summarize_layout_mobility(pd.DataFrame(rows), n_boot=8, random_state=1)
    fig = plot_layout_fig_c(mobility, run_order=["tlock", "one_motif"])
    assert fig is not None
    assert shown == [True]


def test_load_layout_mobility_missing_and_empty(tmp_path):
    assert load_layout_mobility_table(tmp_path, "run").empty
    (tmp_path / "layout_mobility.csv").write_text("")
    assert load_layout_mobility_table(tmp_path, "run").empty
    (tmp_path / "layout_pmpnn_mobility.csv").write_text("run_id,task,subset\n")
    df = load_layout_mobility_table(tmp_path, "run", "layout_pmpnn_mobility.csv")
    assert df.empty


def test_resolve_layout_mobility_prefers_precomputed():
    pre = pd.DataFrame(
        [{"run_id": "tlock", "task": "1PRW", "subset": "all", "n_designs": 3}]
    )
    out, source = resolve_layout_mobility(pd.DataFrame(), pre)
    assert source == "precomputed"
    assert len(out) == 1
    out, source = resolve_layout_mobility(pd.DataFrame(), pd.DataFrame())
    assert source == "recomputed"
    assert out.empty


def test_load_layout_mds_coordinates_missing_empty_and_task(tmp_path):
    missing = tmp_path / "nope.csv"
    assert load_layout_mds_coordinates(missing, "1PRW", kind="trans").empty
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    assert load_layout_mds_coordinates(empty, "1PRW", kind="trans").empty
    path = tmp_path / "layout_mds_matched.csv"
    pd.DataFrame(
        [
            {
                "task": "1PRW",
                "kind": "trans",
                "name": "a",
                "run_id": "tlock05_stage1",
                "mds_x": 0.0,
                "mds_y": 1.0,
                "tau_trans": 2.0,
                "tau_rot": 15.0,
            },
            {
                "task": "1BCF",
                "kind": "rot",
                "name": "b",
                "run_id": "tlock05_stage1",
                "mds_x": 2.0,
                "mds_y": 3.0,
                "tau_trans": 2.0,
                "tau_rot": 15.0,
            },
        ]
    ).to_csv(path, index=False)
    assert load_layout_mds_coordinates(path, "NOPE", kind="trans").empty
    out = load_layout_mds_coordinates(path, "1PRW", kind="trans")
    assert len(out) == 1
    assert out.iloc[0]["name"] == "a"
    assert load_layout_mds_coordinates(path, "1PRW", kind="rot").empty


def test_resolve_layout_mds_coordinates_precomputed_vs_recompute():
    pairs = pd.DataFrame(
        [
            _pose_row("tlock05_stage1", "a", "1PRW", 0.0),
            _pose_row("tlock05_stage1", "b", "1PRW", 4.0),
            _pose_row("one_motif_proteinzen", "a", "1PRW", 1.0),
        ]
    )
    coords = layout_mds_coordinates(pairs, "1PRW", kind="trans")
    assert set(coords["kind"]) == {"trans"}
    assert "stress" in coords.columns
    assert len(coords) == 3
    out, source = resolve_layout_mds_coordinates(
        pairs, "1PRW", kind="trans", precomputed=coords
    )
    assert source == "precomputed"
    assert len(out) == len(coords)
    out, source = resolve_layout_mds_coordinates(
        pairs, "1PRW", kind="rot", precomputed=coords
    )
    assert source == "recomputed"
    assert not out.empty
    assert set(out["kind"]) == {"rot"}


def test_merge_campaigns_concatenates_mobility():
    se3 = _run("se3", ["1PRW"])
    se3["layout_mobility"] = pd.DataFrame(
        [{"run_id": "se3", "task": "1PRW", "subset": "all", "n_designs": 4}]
    )
    full = _run("full", ["1QJG_ext10"])
    full["layout_mobility"] = pd.DataFrame(
        [{"run_id": "full", "task": "1QJG_ext10", "subset": "all", "n_designs": 5}]
    )
    merged = merge_runs_by_campaign([se3, full], {"stage1": ["se3", "full"]})
    mobility = merged[0]["layout_mobility"]
    assert set(mobility["run_id"]) == {"stage1"}
    assert set(mobility["task"]) == {"1PRW", "1QJG_ext10"}


def test_load_group_rmsd_rewrites_paths_without_stating_pdbs(tmp_path, monkeypatch):
    sampling = tmp_path / "article_results" / "sampling_runs" / "demo_stage_1"
    esmfold = sampling / "esmfold"
    esmfold.mkdir(parents=True)
    remote_pdb = (
        "/mnt/old-server/article_results/sampling_runs/demo_stage_1/"
        "esmfold/s0/model_name=v_48_020.pdb"
    )
    pd.DataFrame(
        [
            {"name": "s0", "task": "1PRW", "path": remote_pdb, "bb_rmsd": 0.4},
            {"name": "s0", "task": "1PRW", "path": remote_pdb, "bb_rmsd": 0.5},
        ]
    ).to_csv(esmfold / "group_rmsd.csv", index=False)

    run_dir = tmp_path / "article_results" / "analysis_runs" / "demo_stage1"
    run_dir.mkdir(parents=True)
    (run_dir / "analysis_manifest.json").write_text(
        json.dumps({"sampling_run": str(sampling)}),
        encoding="utf-8",
    )

    original_exists = Path.exists
    original_is_file = Path.is_file

    def guarded_exists(self):
        if self.suffix == ".pdb":
            raise AssertionError(f"unexpected exists() on {self}")
        return original_exists(self)

    def guarded_is_file(self):
        if self.suffix == ".pdb":
            raise AssertionError(f"unexpected is_file() on {self}")
        return original_is_file(self)

    monkeypatch.setattr(Path, "exists", guarded_exists)
    monkeypatch.setattr(Path, "is_file", guarded_is_file)

    df = load_group_rmsd(run_dir, "demo_stage1", repo_root=tmp_path)
    expected = str(
        tmp_path
        / "article_results"
        / "sampling_runs"
        / "demo_stage_1"
        / "esmfold"
        / "s0"
        / "model_name=v_48_020.pdb"
    )
    assert list(df["path"]) == [expected, expected]
    assert list(df["protocol"]) == ["one_shot", "one_shot"]
