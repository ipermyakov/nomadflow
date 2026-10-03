from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from nomadflow.benchmark.analysis_inputs import (
    load_samples_metadata,
    parent_motif_task_id,
    remap_dataframe_tasks_to_parent,
    remap_repo_path,
    sampling_run_id_from_analysis_id,
    strip_criteria_version_suffix,
    task_sample_counts,
)
from nomadflow.benchmark.analysis_tables import (
    build_metrics_table,
    build_task_success_table,
)
from nomadflow.benchmark.foldseek_clustering import (
    compute_and_unique_success,
    count_unique_successes_per_task,
)
from nomadflow.benchmark.group_rmsd_rows import pooled_rmsd_outputs_enabled
from scripts.analysis.ensure_pmpnn_fixed_positions import ensure_pmpnn_fixed_positions
from scripts.analysis.run_motif_benchmark_analysis import run_analysis


def _write_metadata(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_parent_motif_task_id_passthrough_and_stage2_parse():
    assert parent_motif_task_id("1QJG_ext10") == "1QJG_ext10"
    assert (
        parent_motif_task_id("1QJG_ext10__len_100_protein_id0_18_3") == "1QJG_ext10"
    )
    assert (
        parent_motif_task_id(
            "ignored__len_1",
            {"source_stage1_task_name": "5AOU_ext20"},
        )
        == "5AOU_ext20"
    )


def test_task_sample_counts_and_remap_collapse_stage2_layouts():
    samples_metadata = {
        "s0__a": {"name": "1QJG_ext10__len_100_protein_id0_0_0", "num_samples": 1},
        "s0__b": {"name": "1QJG_ext10__len_101_protein_id0_1_0", "num_samples": 1},
        "s1__a": {"name": "5AOU_ext20__len_200_protein_id0_2_0", "num_samples": 1},
    }
    counts = task_sample_counts(samples_metadata)
    assert counts == {"1QJG_ext10": 2, "5AOU_ext20": 1}

    folding_df = pd.DataFrame(
        [
            {
                "name": "s0__a",
                "task": "1QJG_ext10__len_100_protein_id0_0_0",
                "motif_all_atom_rmsd": 1.0,
                "motif_bb_rmsd": 0.5,
                "motif_ca_rmsd": 0.5,
                "global_all_atom_rmsd": 1.5,
            },
            {
                "name": "s0__b",
                "task": "1QJG_ext10__len_101_protein_id0_1_0",
                "motif_all_atom_rmsd": 1.0,
                "motif_bb_rmsd": 0.5,
                "motif_ca_rmsd": 0.5,
                "global_all_atom_rmsd": 1.5,
            },
        ]
    )
    sc_df = pd.DataFrame(
        [
            {
                "name": "s0__a",
                "task": "1QJG_ext10__len_100_protein_id0_0_0",
                "sample": 1,
                "motif_all_atom_rmsd": 1.0,
                "motif_bb_rmsd": 0.5,
                "motif_ca_rmsd": 0.5,
                "global_bb_rmsd": 1.5,
            },
            {
                "name": "s0__b",
                "task": "1QJG_ext10__len_101_protein_id0_1_0",
                "sample": 1,
                "motif_all_atom_rmsd": 1.0,
                "motif_bb_rmsd": 0.5,
                "motif_ca_rmsd": 0.5,
                "global_bb_rmsd": 1.5,
            },
        ]
    )
    folding_df = remap_dataframe_tasks_to_parent(folding_df, samples_metadata)
    sc_df = remap_dataframe_tasks_to_parent(sc_df, samples_metadata)
    assert set(folding_df["task"]) == {"1QJG_ext10"}
    task_success = build_task_success_table(
        folding_df,
        sc_df,
        num_samples_per_task=1,
        task_counts=counts,
    )
    assert len(task_success) == 1
    assert int(task_success.iloc[0]["num_samples"]) == 2
    assert int(task_success.iloc[0]["one_shot_pooled_legacy_count"]) == 2
    assert int(task_success.iloc[0]["one_shot_success_count"]) == 0
    assert int(task_success.iloc[0]["pmpnn_success_count"]) == 2


@mock.patch("nomadflow.benchmark.foldseek_clustering.run_foldseek_easy_cluster")
def test_foldseek_groups_stage2_samples_by_parent_task(mock_cluster, tmp_path):
    mock_cluster.return_value = 2
    pdb_a = tmp_path / "a.pdb"
    pdb_b = tmp_path / "b.pdb"
    pdb_a.write_text("ATOM\n")
    pdb_b.write_text("ATOM\n")
    samples_metadata = {
        "s0__a": {"name": "1QJG_ext10__len_100_protein_id0_0_0", "path": str(pdb_a)},
        "s0__b": {"name": "1QJG_ext10__len_101_protein_id0_1_0", "path": str(pdb_b)},
    }
    passed_df = pd.DataFrame(
        [
            {"name": "s0__a", "task": "1QJG_ext10"},
            {"name": "s0__b", "task": "1QJG_ext10"},
        ]
    )

    def resolver(sample_name: str, _row: pd.Series) -> Path:
        return Path(samples_metadata[sample_name]["path"])

    counts = count_unique_successes_per_task(
        passed_df,
        samples_metadata,
        resolver,
        cluster_root=tmp_path / "clusters",
        all_tasks=["1QJG_ext10"],
    )
    assert counts == {"1QJG_ext10": 2}
    mock_cluster.assert_called_once()
    passed_dir = tmp_path / "clusters" / "passed_pdbs" / "1QJG_ext10"
    assert (passed_dir / "s0__a.pdb").is_file()
    assert (passed_dir / "s0__b.pdb").is_file()
    assert not (tmp_path / "clusters" / "passed_pdbs" / "1QJG_ext10__len_100_protein_id0_0_0").exists()


def _group_rmsd_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": "sample_a",
                "task": "task1",
                "path": "/tmp/esmfold/sample_a/model_name=v_48_020.pdb",
                "sample": 0,
                "group_id": 1,
                "num_group_res": 3,
                "motif_group_all_atom_rmsd": 1.0,
                "motif_group_ca_rmsd": 0.5,
                "motif_group_bb_rmsd": 0.5,
                "rmsd_status": "ok",
            },
            {
                "name": "sample_a",
                "task": "task1",
                "path": "/tmp/esmfold/sample_a/sample=1.pdb",
                "sample": 1,
                "group_id": 1,
                "num_group_res": 3,
                "motif_group_all_atom_rmsd": 1.0,
                "motif_group_ca_rmsd": 0.5,
                "motif_group_bb_rmsd": 0.5,
                "rmsd_status": "ok",
            },
            {
                "name": "sample_b",
                "task": "task1",
                "path": "/tmp/esmfold/sample_b/model_name=v_48_020.pdb",
                "sample": 0,
                "group_id": 1,
                "num_group_res": 3,
                "motif_group_all_atom_rmsd": 3.0,
                "motif_group_ca_rmsd": 0.5,
                "motif_group_bb_rmsd": 0.5,
                "rmsd_status": "ok",
            },
        ]
    )


def _write_esmfold_tables(esmfold_dir: Path) -> None:
    esmfold_dir.mkdir(parents=True, exist_ok=True)
    folding_df = pd.DataFrame(
        [
            {
                "name": "sample_a",
                "task": "task1",
                "motif_all_atom_rmsd": 1.0,
                "motif_bb_rmsd": 0.5,
                "motif_ca_rmsd": 0.5,
                "global_all_atom_rmsd": 1.5,
                "plddt": 80.0,
                "ptm": 0.7,
            },
            {
                "name": "sample_b",
                "task": "task1",
                "motif_all_atom_rmsd": 3.0,
                "motif_bb_rmsd": 0.5,
                "motif_ca_rmsd": 0.5,
                "global_all_atom_rmsd": 1.5,
                "plddt": 70.0,
                "ptm": 0.6,
            },
        ]
    )
    sc_df = pd.DataFrame(
        [
            {
                "name": "sample_a",
                "task": "task1",
                "sample": 1,
                "motif_all_atom_rmsd": 1.0,
                "motif_bb_rmsd": 0.5,
                "motif_ca_rmsd": 0.5,
                "global_bb_rmsd": 1.5,
                "plddt": 75.0,
                "ptm": 0.65,
            },
            {
                "name": "sample_b",
                "task": "task1",
                "sample": 1,
                "motif_all_atom_rmsd": 1.0,
                "motif_bb_rmsd": 0.5,
                "motif_ca_rmsd": 0.5,
                "global_bb_rmsd": 2.5,
                "plddt": 65.0,
                "ptm": 0.55,
            },
        ]
    )
    folding_df.to_csv(esmfold_dir / "folding_rmsd.csv", index=False)
    sc_df.to_csv(esmfold_dir / "sc_rmsd.csv", index=False)
    _group_rmsd_rows().to_csv(esmfold_dir / "group_rmsd.csv", index=False)


def test_ensure_pmpnn_fixed_positions_rebuilds_from_samples_metadata_on_mismatch(tmp_path):
    parsed_jsonl = tmp_path / "parsed_pdbs.jsonl"
    parsed_jsonl.write_text(
        json.dumps(
            {
                "name": "len_109_protein_id0_0_0",
                "seq_chain_A": "A" * 86,
                "seq_chain_B": "B" * 23,
                "coords_chain_A": {},
                "coords_chain_B": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fixed_positions_json = tmp_path / "pmpnn_fixed_pos_dict.jsonl"
    fixed_positions_json.write_text(
        json.dumps({"len_109_protein_id0_0_0": {"B": [85, 24, 2], "A": []}}),
        encoding="utf-8",
    )
    samples_metadata_path = tmp_path / "samples_metadata.json"
    samples_metadata_path.write_text(
        json.dumps(
            {
                "len_109_protein_id0_0_0": {
                    "sample_chain_name": "A",
                    "fixed_seq_res_idx": [85, 24, 2],
                    "fixed_seq_chain": ["B", "B", "B"],
                }
            }
        ),
        encoding="utf-8",
    )

    ensure_pmpnn_fixed_positions(
        parsed_jsonl,
        fixed_positions_json,
        samples_metadata_path=samples_metadata_path,
    )

    with open(fixed_positions_json, "r", encoding="utf-8") as handle:
        fixed_positions = json.load(handle)

    assert fixed_positions["len_109_protein_id0_0_0"]["A"] == [85, 24, 2]
    assert fixed_positions["len_109_protein_id0_0_0"]["B"] == []


def test_ensure_pmpnn_fixed_positions_rebuilds_empty_dict_from_metadata(tmp_path):
    parsed_jsonl = tmp_path / "parsed_pdbs.jsonl"
    parsed_jsonl.write_text(
        json.dumps(
            {
                "name": "len_109_protein_id0_0_0",
                "seq_chain_A": "A" * 86,
                "coords_chain_A": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    samples_metadata_path = tmp_path / "samples_metadata.json"
    samples_metadata_path.write_text(
        json.dumps(
            {
                "len_109_protein_id0_0_0": {
                    "sample_chain_name": "A",
                    "fixed_seq_res_idx": [85, 24, 2],
                    "fixed_seq_chain": ["A", "A", "A"],
                }
            }
        ),
        encoding="utf-8",
    )

    for initial in ({}, {"len_109_protein_id0_0_0": {"A": []}}):
        fixed_positions_json = tmp_path / "pmpnn_fixed_pos_dict.jsonl"
        if initial:
            fixed_positions_json.write_text(json.dumps(initial), encoding="utf-8")
        elif fixed_positions_json.exists():
            fixed_positions_json.unlink()

        ensure_pmpnn_fixed_positions(
            parsed_jsonl,
            fixed_positions_json,
            samples_metadata_path=samples_metadata_path,
        )

        with open(fixed_positions_json, "r", encoding="utf-8") as handle:
            fixed_positions = json.load(handle)

        assert fixed_positions["len_109_protein_id0_0_0"]["A"] == [85, 24, 2]


def test_ensure_pmpnn_fixed_positions_adds_empty_entries_for_all_parsed_chains(tmp_path):
    parsed_jsonl = tmp_path / "parsed_pdbs.jsonl"
    parsed_jsonl.write_text(
        json.dumps(
            {
                "name": "sample_a",
                "seq_chain_A": "GG",
                "seq_chain_0": "G",
                "coords_chain_A": {},
                "coords_chain_0": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fixed_positions_json = tmp_path / "pmpnn_fixed_pos_dict.jsonl"
    fixed_positions_json.write_text(
        json.dumps({"sample_a": {"A": [1, 2]}}),
        encoding="utf-8",
    )

    ensure_pmpnn_fixed_positions(parsed_jsonl, fixed_positions_json)

    with open(fixed_positions_json, "r", encoding="utf-8") as handle:
        fixed_positions = json.load(handle)

    assert fixed_positions == {"sample_a": {"A": [1, 2], "0": []}}


def test_load_samples_metadata_merges_shards(tmp_path):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    shard0 = {
        "sample_a": {"name": "task1", "path": "/tmp/a.pdb"},
    }
    shard1 = {
        "sample_b": {"name": "task1", "path": "/tmp/b.pdb"},
    }
    _write_metadata(metadata_dir / "samples_metadata_rank0_batch0.json", shard0)
    _write_metadata(metadata_dir / "samples_metadata_rank1_batch0.json", shard1)

    merged = load_samples_metadata(tmp_path)
    assert set(merged) == {"sample_a", "sample_b"}


def test_build_metrics_and_task_success_tables():
    folding_df = pd.DataFrame(
        [
            {
                "name": "sample_a",
                "task": "task1",
                "motif_all_atom_rmsd": 1.0,
                "motif_bb_rmsd": 0.5,
                "motif_ca_rmsd": 0.5,
                "global_all_atom_rmsd": 1.5,
            }
        ]
    )
    sc_df = pd.DataFrame(
        [
            {
                "name": "sample_a",
                "task": "task1",
                "sample": 1,
                "motif_all_atom_rmsd": 1.0,
                "motif_bb_rmsd": 0.5,
                "motif_ca_rmsd": 0.5,
                "global_bb_rmsd": 1.5,
            }
        ]
    )
    metrics = build_metrics_table(folding_df, sc_df)
    assert set(metrics["protocol"]) == {"one_shot", "pmpnn_esmfold"}
    assert metrics["pass"].sum() == 2
    assert int(metrics["pass_final"].sum()) == 0
    assert bool((~metrics["pass_all_groups"]).all())

    group_rmsd = _group_rmsd_rows()
    metrics = build_metrics_table(
        folding_df,
        sc_df,
        group_rmsd,
        expected_groups={"sample_a": 1},
    )
    one_shot = metrics[metrics["protocol"] == "one_shot"].iloc[0]
    assert bool(one_shot["pass_pooled_legacy"])
    assert bool(one_shot["pass_all_groups"])
    assert bool(one_shot["pass_final"])
    assert float(one_shot["group_1_ha"]) == 1.0
    assert float(one_shot["group_1_bb"]) == 0.5
    pmpnn_row = metrics[metrics["protocol"] == "pmpnn_esmfold"].iloc[0]
    assert "group_1_ha" in pmpnn_row.index

    task_success = build_task_success_table(
        folding_df,
        sc_df,
        num_samples_per_task=2,
        group_rmsd=group_rmsd,
        expected_groups={"sample_a": 1},
    )
    assert task_success.iloc[0]["one_shot_success_count"] == 1
    assert task_success.iloc[0]["one_shot_pooled_legacy_count"] == 1
    assert task_success.iloc[0]["pmpnn_success_count"] == 1
    assert task_success.iloc[0]["pmpnn_final_count"] == 1
    assert bool(task_success.iloc[0]["any_pmpnn"])


def test_native_ha_ca_rows_drive_pass_final_without_pooled():
    folding_df = pd.DataFrame(
        [
            {
                "name": "sample_a",
                "task": "task1",
                "motif_all_atom_rmsd": 3.0,
                "motif_bb_rmsd": 2.0,
                "motif_ca_rmsd": 2.0,
                "global_all_atom_rmsd": 3.0,
            }
        ]
    )
    sc_df = pd.DataFrame(
        [
            {
                "name": "sample_a",
                "task": "task1",
                "sample": 1,
                "motif_all_atom_rmsd": 3.0,
                "motif_bb_rmsd": 2.0,
                "motif_ca_rmsd": 2.0,
                "global_bb_rmsd": 3.0,
            }
        ]
    )
    native = pd.DataFrame(
        [
            {
                "name": "sample_a",
                "task": "task1",
                "label": "one_shot",
                "protocol": "one_shot",
                "group_id": 1,
                "group_size": 3,
                "motif_ca_rmsd": 0.4,
                "motif_ha_rmsd": 1.0,
                "rmsd_status": "ok",
            },
            {
                "name": "sample_a",
                "task": "task1",
                "label": "pmpnn_sample_1",
                "protocol": "pmpnn_esmfold",
                "group_id": 1,
                "group_size": 3,
                "motif_ca_rmsd": 0.4,
                "motif_ha_rmsd": 1.0,
                "rmsd_status": "ok",
            },
        ]
    )
    metrics = build_metrics_table(
        folding_df,
        sc_df,
        native,
        expected_groups={"sample_a": 1},
    )
    one_shot = metrics[metrics["protocol"] == "one_shot"].iloc[0]
    assert not bool(one_shot["pass_pooled_legacy"])
    assert bool(one_shot["pass_all_groups"])
    assert bool(one_shot["pass_final"])
    assert float(one_shot["group_1_ha"]) == 1.0
    assert float(one_shot["group_1_bb"]) == 0.4
    pmpnn_row = metrics[metrics["protocol"] == "pmpnn_esmfold"].iloc[0]
    assert not bool(pmpnn_row["pass_pooled_legacy"])
    assert bool(pmpnn_row["pass_all_groups"])
    assert bool(pmpnn_row["pass_final"])
    assert float(pmpnn_row["group_1_ha"]) == 1.0
    assert float(pmpnn_row["group_1_bb"]) == 0.4


@mock.patch("scripts.analysis.run_motif_benchmark_analysis.compute_and_unique_success")
@mock.patch("scripts.analysis.run_motif_benchmark_analysis.compute_run_multimotif_metrics")
def test_run_analysis_writes_artifacts(mock_multimotif, mock_unique, tmp_path):
    sampling_run = tmp_path / "sampling_run"
    analysis_out = tmp_path / "analysis_out"
    esmfold_dir = sampling_run / "esmfold"
    _write_esmfold_tables(esmfold_dir)
    _write_metadata(
        sampling_run / "samples_metadata.json",
        {
            "sample_a": {
                "name": "task1",
                "path": str(tmp_path / "a.pdb"),
                "num_samples": 2,
                "motif_groups": [["A1"]],
            },
            "sample_b": {
                "name": "task1",
                "path": str(tmp_path / "b.pdb"),
                "num_samples": 2,
                "motif_groups": [["A1"]],
            },
        },
    )

    mock_multimotif.return_value = pd.DataFrame(
        [{"record_type": "group", "name": "sample_a", "task": "task1"}]
    )
    mock_unique.return_value = {
        "foldseek_tm_threshold": 0.5,
        "one_shot": {"task1": 1},
        "pmpnn_esmfold": {"task1": 1},
        "skipped": False,
    }

    analysis_out.mkdir()
    pd.DataFrame(
        [
            {
                "name": "sample_a",
                "task": "task1",
                "label": "one_shot",
                "protocol": "one_shot",
                "group_id": 1,
                "group_size": 3,
                "motif_ca_rmsd": 0.5,
                "motif_ha_rmsd": 1.0,
                "rmsd_status": "ok",
            },
            {
                "name": "sample_b",
                "task": "task1",
                "label": "one_shot",
                "protocol": "one_shot",
                "group_id": 1,
                "group_size": 3,
                "motif_ca_rmsd": 0.5,
                "motif_ha_rmsd": 3.0,
                "rmsd_status": "ok",
            },
            {
                "name": "sample_a",
                "task": "task1",
                "label": "pmpnn_sample_1",
                "protocol": "pmpnn_esmfold",
                "group_id": 1,
                "group_size": 3,
                "motif_ca_rmsd": 0.5,
                "motif_ha_rmsd": 1.0,
                "rmsd_status": "ok",
            },
            {
                "name": "sample_b",
                "task": "task1",
                "label": "pmpnn_sample_1",
                "protocol": "pmpnn_esmfold",
                "group_id": 1,
                "group_size": 3,
                "motif_ca_rmsd": 0.5,
                "motif_ha_rmsd": 3.0,
                "rmsd_status": "ok",
            },
        ]
    ).to_csv(analysis_out / "group_fold_vs_native.csv", index=False)

    run_analysis(
        sampling_run=sampling_run,
        analysis_out=analysis_out,
        skip_foldseek=False,
        skip_multimotif=False,
        skip_layout=True,
        overwrite=True,
    )

    assert (analysis_out / "unique_success_all_groups.json").is_file()
    assert not (analysis_out / "unique_success.json").exists()
    assert (analysis_out / "metrics.csv").is_file()
    assert (analysis_out / "task_success.csv").is_file()
    assert (analysis_out / "multimotif_metrics.csv").is_file()
    assert (analysis_out / "group_fold_metrics.csv").is_file()
    assert (analysis_out / "analysis_manifest.json").is_file()

    task_success = pd.read_csv(analysis_out / "task_success.csv")
    assert task_success.iloc[0]["one_shot_success_count"] == 1
    assert task_success.iloc[0]["one_shot_pooled_legacy_count"] == 1
    assert task_success.iloc[0]["pmpnn_success_count"] == 1

    unique_success = json.loads((analysis_out / "unique_success_all_groups.json").read_text())
    assert unique_success["one_shot"]["task1"] == 1
    assert unique_success["pmpnn_esmfold"]["task1"] == 1
    mock_unique.assert_called_once()
    assert mock_unique.call_args.kwargs["pmpnn_df"] is not None
    assert set(mock_unique.call_args.kwargs["pmpnn_df"]["protocol"]) == {"pmpnn_esmfold"}
    manifest = json.loads((analysis_out / "analysis_manifest.json").read_text())
    assert manifest["criteria_version"] == "v3_groups_vs_native"
    assert manifest["reference_frame"] == "fold_vs_native_groups"
    assert manifest["foldseek_input"] == "and_success_design_backbones"
    assert (analysis_out / "group_fold_vs_native.csv").is_file()

    metrics = pd.read_csv(analysis_out / "metrics.csv")
    one_shot = metrics[metrics["protocol"] == "one_shot"]
    assert bool(one_shot.loc[one_shot["name"] == "sample_a", "pass_final"].iloc[0])
    assert not bool(one_shot.loc[one_shot["name"] == "sample_b", "pass_final"].iloc[0])


@mock.patch("scripts.analysis.run_motif_benchmark_analysis.compute_and_unique_success")
def test_run_analysis_can_skip_foldseek(mock_unique, tmp_path):
    sampling_run = tmp_path / "sampling_run"
    analysis_out = tmp_path / "analysis_out"
    _write_esmfold_tables(sampling_run / "esmfold")
    _write_metadata(
        sampling_run / "samples_metadata.json",
        {
            "sample_a": {
                "name": "task1",
                "path": "/tmp/a.pdb",
                "num_samples": 1,
                "motif_groups": [["A1"]],
            }
        },
    )

    run_analysis(
        sampling_run=sampling_run,
        analysis_out=analysis_out,
        skip_foldseek=True,
        skip_multimotif=True,
        skip_layout=True,
    )

    mock_unique.assert_not_called()
    unique_success = json.loads((analysis_out / "unique_success_all_groups.json").read_text())
    assert unique_success["skipped"] is True
    assert unique_success["pmpnn_esmfold"] == {}
    assert not (analysis_out / "multimotif_metrics.csv").exists()
    assert not (analysis_out / "unique_success.json").exists()


def test_run_analysis_refuses_nonempty_out_without_overwrite(tmp_path):
    sampling_run = tmp_path / "sampling_run"
    analysis_out = tmp_path / "analysis_out"
    _write_esmfold_tables(sampling_run / "esmfold")
    _write_metadata(
        sampling_run / "samples_metadata.json",
        {"sample_a": {"name": "task1", "path": "/tmp/a.pdb", "num_samples": 1}},
    )
    analysis_out.mkdir()
    (analysis_out / "stale.txt").write_text("nope", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        run_analysis(
            sampling_run=sampling_run,
            analysis_out=analysis_out,
            skip_foldseek=True,
            skip_multimotif=True,
            skip_layout=True,
        )
    assert (analysis_out / "stale.txt").read_text() == "nope"


def test_run_analysis_overwrite_replaces_nonempty(tmp_path):
    sampling_run = tmp_path / "sampling_run"
    analysis_out = tmp_path / "analysis_out"
    _write_esmfold_tables(sampling_run / "esmfold")
    _write_metadata(
        sampling_run / "samples_metadata.json",
        {
            "sample_a": {
                "name": "task1",
                "path": "/tmp/a.pdb",
                "num_samples": 1,
                "motif_groups": [["A1"]],
            }
        },
    )
    analysis_out.mkdir()
    (analysis_out / "stale.txt").write_text("nope", encoding="utf-8")
    (analysis_out / "unique_success.json").write_text("{}", encoding="utf-8")
    sidecar = analysis_out / "group_fold_vs_native.csv"
    sidecar.write_text("name,task,protocol,group_id,motif_ha_rmsd,motif_ca_rmsd,rmsd_status\n", encoding="utf-8")
    design_sidecar = analysis_out / "group_design_vs_native.csv"
    design_sidecar.write_text("name\n", encoding="utf-8")
    run_analysis(
        sampling_run=sampling_run,
        analysis_out=analysis_out,
        skip_foldseek=True,
        skip_multimotif=True,
        skip_layout=True,
        overwrite=True,
    )
    assert not (analysis_out / "stale.txt").exists()
    assert not (analysis_out / "unique_success.json").exists()
    assert (analysis_out / "unique_success_all_groups.json").is_file()
    assert sidecar.is_file()
    assert design_sidecar.is_file()


@mock.patch("nomadflow.benchmark.foldseek_clustering.run_foldseek_easy_cluster")
def test_foldseek_wipes_leftovers_and_clamps_unique(mock_cluster, tmp_path):
    mock_cluster.return_value = 99
    pdb_a = tmp_path / "a.pdb"
    pdb_b = tmp_path / "b.pdb"
    pdb_a.write_text("ATOM A\n")
    pdb_b.write_text("ATOM B\n")
    leftover_dir = tmp_path / "clusters" / "passed_pdbs" / "1QJG_ext10"
    leftover_dir.mkdir(parents=True)
    (leftover_dir / "stale_extra.pdb").write_text("LEFTOVER\n")
    samples_metadata = {
        "s0__a": {"name": "1QJG_ext10__len_100_protein_id0_0_0", "path": str(pdb_a)},
        "s0__b": {"name": "1QJG_ext10__len_101_protein_id0_1_0", "path": str(pdb_b)},
    }
    passed_df = pd.DataFrame(
        [
            {"name": "s0__a", "task": "1QJG_ext10"},
            {"name": "s0__b", "task": "1QJG_ext10"},
        ]
    )

    def resolver(sample_name: str, _row: pd.Series) -> Path:
        return Path(samples_metadata[sample_name]["path"])

    counts = count_unique_successes_per_task(
        passed_df,
        samples_metadata,
        resolver,
        cluster_root=tmp_path / "clusters",
        all_tasks=["1QJG_ext10"],
    )
    assert counts == {"1QJG_ext10": 2}
    passed_dir = tmp_path / "clusters" / "passed_pdbs" / "1QJG_ext10"
    assert not (passed_dir / "stale_extra.pdb").exists()
    assert (passed_dir / "s0__a.pdb").is_file()
    assert (passed_dir / "s0__b.pdb").is_file()


def test_strip_criteria_version_suffix_drops_v2_and_v3():
    assert strip_criteria_version_suffix("one_motif_proteinzen") == "one_motif_proteinzen"
    assert (
        strip_criteria_version_suffix("one_motif_proteinzen__criteria_v2")
        == "one_motif_proteinzen"
    )
    assert (
        strip_criteria_version_suffix("one_motif_proteinzen__criteria_v3")
        == "one_motif_proteinzen"
    )


def test_sampling_run_id_and_path_remap(tmp_path):
    assert (
        sampling_run_id_from_analysis_id("0c62dc8a__tlock05_se3_stage2__criteria_v2")
        == "0c62dc8a__tlock05_se3_stage_2"
    )
    assert (
        sampling_run_id_from_analysis_id("0c62dc8a__tlock05_se3_stage2__criteria_v3")
        == "0c62dc8a__tlock05_se3_stage_2"
    )
    assert sampling_run_id_from_analysis_id("one_motif_proteinzen") == "one_motif_proteinzen"
    existing = tmp_path / "article_results" / "sampling_runs" / "demo"
    existing.mkdir(parents=True)
    fake_server = Path("/mnt/old-server") / "article_results" / "sampling_runs" / "demo"
    remapped = remap_repo_path(fake_server, repo_root=tmp_path)
    assert remapped == existing

    eval_dir = tmp_path / "article_results" / "sampling_runs" / "demo" / "samples_eval"
    eval_dir.mkdir(parents=True)
    eval_pdb = eval_dir / "s0.pdb"
    eval_pdb.write_text("ATOM\n", encoding="utf-8")
    remote_samples = Path(
        "/mnt/old-server/article_results/sampling_runs/demo/samples/s0.pdb"
    )
    assert remap_repo_path(remote_samples, repo_root=tmp_path) == eval_pdb


def test_only_group_rmsd_flag_skips_pooled_csvs():
    assert pooled_rmsd_outputs_enabled(True) is False
    assert pooled_rmsd_outputs_enabled(False) is True
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "analysis"
        / "esm_compute_rmsds.py"
    ).read_text(encoding="utf-8")
    assert "--only-group-rmsd" in source
    assert "--layout-manifest" in source


@mock.patch("nomadflow.benchmark.foldseek_clustering.run_foldseek_easy_cluster")
def test_foldseek_clusters_only_pass_final_designs(mock_cluster, tmp_path):
    mock_cluster.return_value = 1
    pdb_pass = tmp_path / "pass.pdb"
    pdb_pooled_only = tmp_path / "pooled_only.pdb"
    pdb_pass.write_text("ATOM PASS\n")
    pdb_pooled_only.write_text("ATOM POOLED\n")
    samples_metadata = {
        "sample_pass": {"name": "1QJG_ext10", "path": str(pdb_pass)},
        "sample_pooled": {"name": "1QJG_ext10", "path": str(pdb_pooled_only)},
    }
    folding_df = pd.DataFrame(
        [
            {
                "name": "sample_pass",
                "task": "1QJG_ext10",
                "pass_pooled_legacy": True,
                "pass_all_groups": True,
                "pass_final": True,
            },
            {
                "name": "sample_pooled",
                "task": "1QJG_ext10",
                "pass_pooled_legacy": True,
                "pass_all_groups": False,
                "pass_final": False,
            },
        ]
    )
    payload = compute_and_unique_success(
        folding_df,
        samples_metadata,
        cluster_root=tmp_path / "clusters",
        all_tasks=["1QJG_ext10"],
    )
    passed_dir = tmp_path / "clusters" / "one_shot" / "passed_pdbs" / "1QJG_ext10"
    assert (passed_dir / "sample_pass.pdb").is_file()
    assert not (passed_dir / "sample_pooled.pdb").exists()
    assert payload["one_shot"]["1QJG_ext10"] == 1
    assert payload["pmpnn_esmfold"] == {}
    assert payload["foldseek_input"] == "and_success_design_backbones"
    mock_cluster.assert_not_called()


@mock.patch("nomadflow.benchmark.foldseek_clustering.run_foldseek_easy_cluster")
def test_foldseek_clusters_pmpnn_and_success_designs(mock_cluster, tmp_path):
    mock_cluster.return_value = 1
    pdb_pass = tmp_path / "pass.pdb"
    pdb_pooled_only = tmp_path / "pooled_only.pdb"
    pdb_pass.write_text("ATOM PASS\n")
    pdb_pooled_only.write_text("ATOM POOLED\n")
    samples_metadata = {
        "sample_pass": {"name": "1QJG_ext10", "path": str(pdb_pass)},
        "sample_pooled": {"name": "1QJG_ext10", "path": str(pdb_pooled_only)},
    }
    folding_df = pd.DataFrame(
        [
            {
                "name": "sample_pass",
                "task": "1QJG_ext10",
                "pass_final": True,
            }
        ]
    )
    pmpnn_df = pd.DataFrame(
        [
            {
                "name": "sample_pass",
                "task": "1QJG_ext10",
                "sample": 0,
                "pass_pooled_legacy": True,
                "pass_all_groups": True,
                "pass_final": True,
            },
            {
                "name": "sample_pass",
                "task": "1QJG_ext10",
                "sample": 1,
                "pass_pooled_legacy": True,
                "pass_all_groups": True,
                "pass_final": True,
            },
            {
                "name": "sample_pooled",
                "task": "1QJG_ext10",
                "sample": 0,
                "pass_pooled_legacy": True,
                "pass_all_groups": False,
                "pass_final": False,
            },
        ]
    )
    payload = compute_and_unique_success(
        folding_df,
        samples_metadata,
        cluster_root=tmp_path / "clusters",
        all_tasks=["1QJG_ext10"],
        pmpnn_df=pmpnn_df,
    )
    pmpnn_dir = tmp_path / "clusters" / "pmpnn_esmfold" / "passed_pdbs" / "1QJG_ext10"
    assert (pmpnn_dir / "sample_pass.pdb").is_file()
    assert not (pmpnn_dir / "sample_pooled.pdb").exists()
    assert payload["pmpnn_esmfold"]["1QJG_ext10"] == 1
    mock_cluster.assert_not_called()
