from __future__ import annotations

import json
from pathlib import Path

from nomadflow.runtime.sampling.layout_manifest import export_layout_manifest
from scripts.analysis.build_stage2_tasks_from_layout_manifest import build_stage2_tasks


def _write_minimal_pdb(path: Path, residues: list[tuple[str, int]]) -> None:
    lines: list[str] = []
    for atom_num, (chain, res_idx) in enumerate(residues, start=1):
        lines.append(
            f"ATOM  {atom_num:5d}  CA  ALA {chain}{res_idx:4d}      "
            f"0.000   0.000   0.000  1.00  0.00           C"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _motif_sample_metadata(sample_path: Path) -> dict:
    return {
        "path": str(sample_path),
        "viz_path": str(sample_path.with_name(f"{sample_path.stem}_viz.pdb")),
        "name": "1QJG_ext10",
        "length": 100,
        "sample_chain_name": "A",
        "native_pdb": "/tmp/native/1QJG.pdb",
        "pdb_contigs": "A38,A14,A99,A22-31",
        "source_residue_keys": [
            {
                "native_key": "A38",
                "native_chain": "A",
                "native_res_idx": 38,
                "output_chain": "A",
                "output_res_idx": 12,
                "is_unindexed": False,
                "group_id": 1,
            },
            {
                "native_key": "A22",
                "native_chain": "A",
                "native_res_idx": 22,
                "output_chain": "A",
                "output_res_idx": 45,
                "is_unindexed": False,
                "group_id": 2,
            },
        ],
        "rmsd_reference_chain": "A",
        "rmsd_folded_chain": "A",
        "motif_reference_residues": [12, 45],
        "motif_folded_residues": [12, 45],
        "motif_bb_reference_residues": [12, 45],
        "motif_bb_folded_residues": [12, 45],
        "motif_native_keys": ["A38", "A22"],
    }


def test_export_layout_manifest_and_build_stage2_tasks(tmp_path: Path):
    sample_path = tmp_path / "sample_0.pdb"
    viz_path = tmp_path / "sample_0_viz.pdb"
    _write_minimal_pdb(sample_path, [("A", 12), ("A", 45)])
    _write_minimal_pdb(
        viz_path,
        [("A", 12), ("A", 45), ("B", 1), ("B", 2)],
    )

    metadata = _motif_sample_metadata(sample_path)
    metadata["viz_path"] = str(viz_path)
    samples_metadata = {
        "len_100_protein_id0_0_0": metadata,
    }
    samples_metadata_path = tmp_path / "samples_metadata.json"
    samples_metadata_path.write_text(json.dumps(samples_metadata), encoding="utf-8")

    manifest_path = tmp_path / "layout_manifest.jsonl"
    written = export_layout_manifest(
        samples_metadata_path=samples_metadata_path,
        output_path=manifest_path,
    )
    assert written == 1

    entries = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    assert len(entries) == 1
    assert entries[0]["layout_id"] == "len_100_protein_id0_0_0"
    assert entries[0]["output_pdb_contigs"] == "A12,A45"
    assert entries[0]["output_motif_groups"] == [["A12"], ["A45"]]
    assert entries[0]["rmsd_reference_chain"] == "A"
    assert entries[0]["motif_reference_residues"] == [12, 45]
    assert entries[0]["stage2_input_pdb"] == str(viz_path)
    assert entries[0]["stage2_pdb_contigs"] == "B1,B2"
    assert entries[0]["stage2_motif_groups"] == [["B1", "B2"]]

    stage2 = build_stage2_tasks(
        entries,
        stage2_num_samples=1,
        default_total_length_mode="sample_length",
        include_motif_groups=True,
        motif_source="metadata",
    )
    tasks = stage2["tasks"]
    assert len(tasks) == 1
    assert tasks[0]["pdb"] == str(sample_path)
    assert tasks[0]["pdb_contigs"] == "A12,A45"
    assert tasks[0]["num_samples"] == 1
    assert tasks[0]["total_length"] == "100"
    assert tasks[0]["motif_groups"] == [["A12"], ["A45"]]
    assert tasks[0]["rmsd_reference_chain"] == "A"
    assert tasks[0]["motif_folded_residues"] == [12, 45]


def test_export_layout_manifest_skips_samples_without_motif_metadata(tmp_path: Path):
    samples_metadata = {
        "plain_sample": {
            "path": str(tmp_path / "plain.pdb"),
            "name": "unconditional",
            "length": 80,
        },
        "motif_sample": _motif_sample_metadata(tmp_path / "motif.pdb"),
    }
    _write_minimal_pdb(tmp_path / "motif.pdb", [("A", 12), ("A", 45)])

    samples_metadata_path = tmp_path / "samples_metadata.json"
    samples_metadata_path.write_text(json.dumps(samples_metadata), encoding="utf-8")

    manifest_path = tmp_path / "layout_manifest.jsonl"
    written = export_layout_manifest(
        samples_metadata_path=samples_metadata_path,
        output_path=manifest_path,
    )
    assert written == 1

    entries = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    assert len(entries) == 1
    assert entries[0]["layout_id"] == "motif_sample"


def test_build_stage2_tasks_from_viz_chains(tmp_path: Path):
    eval_path = tmp_path / "sample_eval.pdb"
    viz_path = tmp_path / "sample_viz.pdb"
    _write_minimal_pdb(eval_path, [("A", 1), ("A", 2), ("A", 3), ("A", 4), ("A", 5)])
    _write_minimal_pdb(
        viz_path,
        [("A", 1), ("A", 2), ("A", 3), ("A", 4), ("A", 5), ("B", 1), ("B", 2), ("V", 1), ("V", 2)],
    )

    entries = [
        {
            "layout_id": "layout_0",
            "stage1_task_name": "motif_task",
            "stage1_sample_path": str(eval_path),
            "stage1_viz_path": str(viz_path),
            "sample_chain_name": "A",
            "sample_length": 5,
            "output_pdb_contigs": "A1,A2",
            "stage2_input_pdb": str(viz_path),
            "stage2_pdb_contigs": "B1,B2",
            "stage2_motif_groups": [["B1", "B2"]],
            "motif_reference_residues": [1, 2],
            "source_residue_keys": [
                {
                    "native_chain": "A",
                    "native_res_idx": 1,
                    "output_chain": "A",
                    "output_res_idx": 1,
                }
            ],
        }
    ]

    stage2 = build_stage2_tasks(
        entries,
        stage2_num_samples=1,
        default_total_length_mode="sample_length",
        include_motif_groups=True,
        motif_source="viz_chains",
    )
    task = stage2["tasks"][0]
    assert task["pdb"] == str(viz_path)
    assert task["pdb_contigs"] == "B1,B2"
    assert task["stage2_input_pdb"] == str(viz_path)
    assert task["stage2_pdb_contigs"] == "B1,B2"
    assert task["motif_reference_residues"] == [1, 2]
    assert task["total_length"] == "5"
    assert task["motif_groups"] == [["B1", "B2"]]
