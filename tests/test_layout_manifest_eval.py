from __future__ import annotations

import json
from pathlib import Path

import yaml

from nomadflow.benchmark.layout_manifest_eval import (
    enrich_sample_metadata_from_layout_entry,
    enrich_samples_metadata_from_layout_manifest,
    infer_layout_id,
    load_layout_manifest,
    resolve_layout_manifest_path,
    select_layout_entry,
    stage1_task_name_for_sample,
)


def _layout_entry() -> dict:
    return {
        "layout_id": "len_99_protein_id0_0_0",
        "sample_chain_name": "A",
        "native_pdb": "/tmp/native/1QJG.pdb",
        "motif_native_keys": ["A38", "A14", "A99", "A22"],
        "source_residue_keys": [
            {
                "group_id": 1,
                "native_chain": "A",
                "native_key": "A38",
                "native_res_idx": 38,
                "output_chain": "A",
                "output_res_idx": 41,
                "is_unindexed": True,
            },
            {
                "group_id": 1,
                "native_chain": "A",
                "native_key": "A14",
                "native_res_idx": 14,
                "output_chain": "A",
                "output_res_idx": 72,
                "is_unindexed": True,
            },
            {
                "group_id": 1,
                "native_chain": "A",
                "native_key": "A99",
                "native_res_idx": 99,
                "output_chain": "A",
                "output_res_idx": 51,
                "is_unindexed": True,
            },
            {
                "group_id": 2,
                "native_chain": "A",
                "native_key": "A22",
                "native_res_idx": 22,
                "output_chain": "A",
                "output_res_idx": 2,
                "is_unindexed": True,
            },
        ],
    }


def test_infer_layout_id_from_task_name_suffix():
    layout_id = infer_layout_id({"name": "1QJG_ext2x10__len_99_protein_id0_0_0"})
    assert layout_id == "len_99_protein_id0_0_0"


def test_enrich_sample_metadata_maps_stage2_indices_by_order():
    stage2_metadata = {
        "name": "1QJG_ext2x10__len_99_protein_id0_0_0",
        "sample_chain_name": "A",
        "fixed_bb_res_idx": [73, 14, 28, 39],
    }
    enriched = enrich_sample_metadata_from_layout_entry(stage2_metadata, _layout_entry())

    assert enriched["native_pdb"] == "/tmp/native/1QJG.pdb"
    assert enriched["source_layout_id"] == "len_99_protein_id0_0_0"
    assert len(enriched["source_residue_keys"]) == 4
    assert enriched["source_residue_keys"][0]["output_res_idx"] == 73
    assert enriched["source_residue_keys"][0]["group_id"] == 1
    assert enriched["source_residue_keys"][3]["output_res_idx"] == 39
    assert enriched["source_residue_keys"][3]["group_id"] == 2


def test_enrich_without_manifest_still_repairs_far_outlier_keys():
    helix = [
        {
            "native_key": f"A{61 + i}",
            "native_chain": "A",
            "native_res_idx": 61 + i,
            "output_chain": "A",
            "output_res_idx": (list(range(193, 203)) + [110] + list(range(204, 213)))[i],
            "group_id": 2,
            "is_unindexed": True,
        }
        for i in range(20)
    ]
    samples = {"s0": {"name": "7K4V_ext20", "source_residue_keys": helix}}
    enriched = enrich_samples_metadata_from_layout_manifest(samples)
    out = [e["output_res_idx"] for e in enriched["s0"]["source_residue_keys"]]
    assert out == list(range(193, 213))

    stage2_metadata = {
        "name": "1QJG_ext2x10__len_99_protein_id0_0_0",
        "sample_chain_name": "A",
        "fixed_bb_res_idx": [73, 14, 28],
    }
    enriched = enrich_sample_metadata_from_layout_entry(stage2_metadata, _layout_entry())
    assert "source_residue_keys" not in enriched
    assert enriched.get("eval_error") == "stage2_stage1_residue_count_mismatch"
    assert enriched["eval_error_detail"]["stage2_n_residues"] == 3


def test_enrich_samples_metadata_from_manifest_file(tmp_path: Path):
    manifest_path = tmp_path / "layout_manifest.jsonl"
    manifest_path.write_text(json.dumps(_layout_entry()) + "\n", encoding="utf-8")

    samples_metadata = {
        "len_96_protein_id0_0_0": {
            "name": "1QJG_ext2x10__len_99_protein_id0_0_0",
            "fixed_bb_res_idx": [73, 14, 28, 39],
        }
    }
    enriched = enrich_samples_metadata_from_layout_manifest(
        samples_metadata,
        layout_manifest_path=manifest_path,
    )
    assert "source_residue_keys" in enriched["len_96_protein_id0_0_0"]
    assert enriched["len_96_protein_id0_0_0"]["source_residue_keys"][1]["group_id"] == 1


def test_load_layout_manifest_keeps_duplicate_layout_ids(tmp_path: Path):
    path = tmp_path / "layout_manifest.jsonl"
    first = dict(_layout_entry())
    first["stage1_task_name"] = "1PRW"
    second = dict(_layout_entry())
    second["stage1_task_name"] = "1BCF"
    second["source_residue_keys"] = first["source_residue_keys"][:2]
    path.write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8"
    )
    grouped = load_layout_manifest(path)
    assert len(grouped["len_99_protein_id0_0_0"]) == 2
    assert [entry["stage1_task_name"] for entry in grouped["len_99_protein_id0_0_0"]] == [
        "1PRW",
        "1BCF",
    ]


def test_select_layout_entry_disambiguates_by_stage1_task_name():
    entries = [
        {**_layout_entry(), "stage1_task_name": "1PRW"},
        {**_layout_entry(), "stage1_task_name": "1BCF"},
    ]
    picked = select_layout_entry(entries, stage1_task_name="1PRW")
    assert picked is not None
    assert picked["stage1_task_name"] == "1PRW"
    assert select_layout_entry(entries, stage1_task_name=None) is None
    assert select_layout_entry(entries, stage1_task_name="5IUS") is None
    assert select_layout_entry(entries[:1], stage1_task_name=None) is entries[0]


def test_stage1_task_name_from_stage2_sample_name():
    assert (
        stage1_task_name_for_sample({"name": "5AOU_ext10__len_256_protein_id0_0_3"})
        == "5AOU_ext10"
    )
    assert (
        stage1_task_name_for_sample(
            {"name": "1PRW__len_179_protein_id0_57_4", "source_stage1_task_name": "1PRW"}
        )
        == "1PRW"
    )


def test_enrich_picks_matching_task_when_layout_id_collides(tmp_path: Path):
    prw_entry = dict(_layout_entry())
    prw_entry["stage1_task_name"] = "1PRW"
    bcf_entry = dict(_layout_entry())
    bcf_entry["stage1_task_name"] = "1BCF"
    bcf_entry["source_residue_keys"] = prw_entry["source_residue_keys"][:2]
    manifest_path = tmp_path / "layout_manifest.jsonl"
    manifest_path.write_text(
        json.dumps(prw_entry) + "\n" + json.dumps(bcf_entry) + "\n", encoding="utf-8"
    )

    samples_metadata = {
        "s0_1prw__len_97_protein_id0_0_0": {
            "name": "1PRW__len_99_protein_id0_0_0",
            "fixed_bb_res_idx": [73, 14, 28, 39],
        }
    }
    enriched = enrich_samples_metadata_from_layout_manifest(
        samples_metadata,
        layout_manifest_path=manifest_path,
    )
    sample = enriched["s0_1prw__len_97_protein_id0_0_0"]
    assert "eval_error" not in sample
    assert len(sample["source_residue_keys"]) == 4
    assert sample["source_residue_keys"][3]["group_id"] == 2


def test_enrich_marks_ambiguous_duplicate_layout_id(tmp_path: Path):
    first = dict(_layout_entry())
    first["stage1_task_name"] = "1PRW"
    second = dict(_layout_entry())
    second["stage1_task_name"] = "1BCF"
    manifest_path = tmp_path / "layout_manifest.jsonl"
    manifest_path.write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8"
    )
    samples_metadata = {
        "s0_unknown__len_97_protein_id0_0_0": {
            "name": "mystery__len_99_protein_id0_0_0",
            "fixed_bb_res_idx": [73, 14, 28, 39],
        }
    }
    enriched = enrich_samples_metadata_from_layout_manifest(
        samples_metadata,
        layout_manifest_path=manifest_path,
    )
    sample = enriched["s0_unknown__len_97_protein_id0_0_0"]
    assert sample.get("eval_error") == "layout_id_ambiguous"
    assert "source_residue_keys" not in sample


def test_resolve_layout_manifest_path_from_stage2_config(tmp_path: Path):
    stage1_run = tmp_path / "stage1_layout__demo"
    samples_dir = stage1_run / "samples"
    samples_dir.mkdir(parents=True)
    sample_pdb = samples_dir / "len_99_protein_id0_0_0.pdb"
    sample_pdb.write_text("END\n", encoding="utf-8")
    manifest_path = stage1_run / "layout_manifest.jsonl"
    manifest_path.write_text(json.dumps(_layout_entry()) + "\n", encoding="utf-8")

    stage2_run = tmp_path / "stage2_run"
    stage2_run.mkdir()
    config = {
        "tasks": [
            {
                "name": "1QJG_ext2x10__len_99_protein_id0_0_0",
                "pdb": str(sample_pdb),
                "source_layout_id": "len_99_protein_id0_0_0",
            }
        ]
    }
    (stage2_run / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    resolved = resolve_layout_manifest_path(stage2_run)
    assert resolved == manifest_path
