from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from nomadflow.benchmark.motif_group_annotation import (
    annotate_sample_metadata,
    annotate_samples_metadata,
    apply_source_residue_key_repair,
    expand_motif_groups,
    repair_source_residue_keys,
)
from nomadflow.benchmark.rmsd_mapping import build_motif_group_mappings, build_rmsd_mapping


TASK = {
    "task": "motif_scaffolding",
    "name": "1QJG_ext2x10",
    "pdb": "configs/sampling/protpardelle_motif_scaffolding/pdbs/1QJG.pdb",
    "pdb_contigs": "A38,A14,A99,A22-31,A103-112",
    "motif_groups": [["A38", "A14", "A99"], ["A22-31"], ["A103-112"]],
}


def _baseline_sample(*, n_fixed: int | None = None) -> dict:
    expanded = expand_motif_groups(TASK["motif_groups"])
    if n_fixed is None:
        n_fixed = len(expanded)
    return {
        "path": "/tmp/fake.pdb",
        "name": "1QJG_ext2x10",
        "length": 100,
        "fixed_bb_res_idx": list(range(1, n_fixed + 1)),
        "fixed_bb_chain": ["A"] * n_fixed,
        "fixed_seq_res_idx": list(range(10, 10 + n_fixed)),
        "fixed_seq_chain": ["A"] * n_fixed,
        "rmsd_reference_chain": "A",
        "rmsd_folded_chain": "A",
        "motif_reference_residues": list(range(10, 10 + n_fixed)),
        "motif_folded_residues": list(range(10, 10 + n_fixed)),
        "motif_bb_reference_residues": list(range(10, 10 + n_fixed)),
        "motif_bb_folded_residues": list(range(10, 10 + n_fixed)),
    }


def test_expand_motif_groups_matches_catalytic_small_order():
    expanded = expand_motif_groups(TASK["motif_groups"])
    keys = [key for key, _ in expanded]
    assert keys[:3] == ["A38", "A14", "A99"]
    assert keys[3:13] == [f"A{i}" for i in range(22, 32)]
    assert keys[13:] == [f"A{i}" for i in range(103, 113)]
    assert [gid for _, gid in expanded[:3]] == [1, 1, 1]
    assert set(gid for _, gid in expanded[3:13]) == {2}
    assert set(gid for _, gid in expanded[13:]) == {3}
    assert len(expanded) == 23


def test_annotate_sample_metadata_builds_source_residue_keys():
    sample = _baseline_sample()
    annotated = annotate_sample_metadata(sample, TASK, repo_root=None)
    assert annotated["motif_groups"] == TASK["motif_groups"]
    assert len(annotated["source_residue_keys"]) == 23
    assert annotated["source_residue_keys"][0]["native_key"] == "A38"
    assert annotated["source_residue_keys"][0]["group_id"] == 1
    assert annotated["source_residue_keys"][0]["output_res_idx"] == 10
    assert annotated["source_residue_keys"][3]["group_id"] == 2
    assert "A:10" in annotated["motif_group_assignments"]

    mapping = build_rmsd_mapping(annotated)
    groups = build_motif_group_mappings(annotated, mapping)
    assert len(groups) == 3
    assert groups[0]["group_id"] == "1"
    assert len(groups[0]["native_keys"]) == 3


def test_annotate_sample_metadata_rejects_length_mismatch():
    sample = _baseline_sample(n_fixed=5)
    with pytest.raises(ValueError, match="does not match fixed_seq_res_idx"):
        annotate_sample_metadata(sample, TASK)


def test_annotate_samples_metadata_matches_by_name():
    samples = {
        "len_100_protein_id0_0_0": _baseline_sample(),
        "other": {"name": "unknown_task", "fixed_seq_res_idx": [1]},
    }
    annotated = annotate_samples_metadata(
        samples,
        [TASK],
        skip_missing_groups=True,
    )
    assert "source_residue_keys" in annotated["len_100_protein_id0_0_0"]
    assert "source_residue_keys" not in annotated["other"]


def test_inject_cli_writes_metadata(tmp_path: Path):
    from scripts.analysis.inject_motif_group_metadata import main as inject_main

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sample = _baseline_sample()
    metadata = {"len_100_protein_id0_0_0": sample}
    (run_dir / "samples_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    tasks_yaml = run_dir / "config.yaml"
    tasks_yaml.write_text(yaml.safe_dump({"tasks": [TASK]}), encoding="utf-8")

    # Ensure scripts.analysis is importable when running as module path hack.
    # The CLI imports nomadflow; inject_main is imported above.
    rc = inject_main(
        [
            "--sampling-run",
            str(run_dir),
            "--tasks-yaml",
            str(tasks_yaml),
        ]
    )
    assert rc == 0
    written = json.loads((run_dir / "samples_metadata.json").read_text(encoding="utf-8"))
    assert written["len_100_protein_id0_0_0"]["source_residue_keys"][0]["group_id"] == 1


def _helix_keys(outputs, *, native_start=61, group_id=2):
    return [
        {
            "native_key": f"A{native_start + offset}",
            "native_chain": "A",
            "native_res_idx": native_start + offset,
            "output_chain": "A",
            "output_res_idx": output,
            "group_id": group_id,
            "is_unindexed": True,
        }
        for offset, output in enumerate(outputs)
    ]


def test_repair_7k4v_ext20_far_outlier_to_helix_hole():
    outputs = list(range(193, 203)) + [110] + list(range(204, 213))
    keys = (
        [
            {
                "native_key": native,
                "native_chain": "A",
                "native_res_idx": int(native[1:]),
                "output_chain": "A",
                "output_res_idx": output,
                "group_id": 1,
                "is_unindexed": True,
            }
            for native, output in (("A44", 180), ("A50", 217), ("A127", 247))
        ]
        + _helix_keys(outputs)
    )
    repaired, notes = repair_source_residue_keys(keys)
    helix = [entry["output_res_idx"] for entry in repaired if entry["group_id"] == 2]
    assert helix == list(range(193, 213))
    assert notes == [
        {"native_key": "A71", "group_id": 2, "from": 110, "to": 203}
    ]
    metadata = apply_source_residue_key_repair(
        {
            "name": "7K4V_ext20",
            "source_residue_keys": keys,
            "fixed_seq_res_idx": [entry["output_res_idx"] for entry in keys],
        }
    )
    assert metadata["fixed_seq_res_idx"][13] == 203
    assert "A:203" in metadata["motif_group_assignments"]


def test_repair_leaves_adjacent_gap_and_catalytic_core():
    helix = _helix_keys([54] + list(range(56, 75)))
    core = [
        {
            "native_key": "A44",
            "native_chain": "A",
            "native_res_idx": 44,
            "output_chain": "A",
            "output_res_idx": 53,
            "group_id": 1,
            "is_unindexed": True,
        }
    ]
    repaired, notes = repair_source_residue_keys(core + helix)
    assert notes == []
    assert [entry["output_res_idx"] for entry in repaired if entry["group_id"] == 2][0] == 54
    assert repaired[0]["output_res_idx"] == 53


def test_repair_is_idempotent_on_contiguous_helix():
    keys = _helix_keys(list(range(193, 213)))
    repaired, notes = repair_source_residue_keys(keys)
    assert notes == []
    assert [entry["output_res_idx"] for entry in repaired] == list(range(193, 213))


def test_repair_does_not_steal_occupied_hole():
    outputs = list(range(193, 203)) + [110] + list(range(204, 213))
    keys = _helix_keys(outputs) + [
        {
            "native_key": "A44",
            "native_chain": "A",
            "native_res_idx": 44,
            "output_chain": "A",
            "output_res_idx": 203,
            "group_id": 1,
            "is_unindexed": True,
        }
    ]
    repaired, notes = repair_source_residue_keys(keys)
    assert notes == []
    assert repaired[10]["output_res_idx"] == 110

