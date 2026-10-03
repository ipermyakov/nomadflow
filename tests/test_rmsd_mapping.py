from __future__ import annotations

from nomadflow.benchmark.rmsd_mapping import (
    build_motif_group_mappings,
    build_rmsd_mapping,
    pose_indices_from_residue_pairs,
)


def test_build_rmsd_mapping_prefers_explicit_fields():
    mapping = build_rmsd_mapping(
        {
            "rmsd_reference_chain": "C",
            "rmsd_folded_chain": "A",
            "motif_reference_residues": [8, 9, 9, 10],
            "motif_folded_residues": [1, 2, 2, 3],
            "motif_bb_reference_residues": [20, 21],
            "motif_bb_folded_residues": [4, 5],
        }
    )
    assert mapping["reference_chain"] == "C"
    assert mapping["folded_chain"] == "A"
    assert mapping["motif_reference_residues"] == [8, 9, 10]
    assert mapping["motif_folded_residues"] == [1, 2, 3]
    assert mapping["motif_bb_reference_residues"] == [20, 21]
    assert mapping["motif_bb_folded_residues"] == [4, 5]


def test_build_rmsd_mapping_falls_back_to_fixed_residues():
    mapping = build_rmsd_mapping(
        {
            "sample_chain_name": "A",
            "fixed_seq_res_idx": [69, 44, 2],
            "fixed_bb_res_idx": [50, 51, 52],
        }
    )
    assert mapping["reference_chain"] == "A"
    assert mapping["folded_chain"] == "A"
    assert mapping["motif_reference_residues"] == [69, 44, 2]
    assert mapping["motif_folded_residues"] == [69, 44, 2]
    assert mapping["motif_bb_reference_residues"] == [50, 51, 52]
    assert mapping["motif_bb_folded_residues"] == [50, 51, 52]


def test_build_rmsd_mapping_prefers_source_output_chain_when_available():
    mapping = build_rmsd_mapping(
        {
            "sample_chain_name": "B",
            "fixed_seq_chain": "B",
            "fixed_seq_res_idx": [12, 45],
            "fixed_bb_res_idx": [12, 45],
            "source_residue_keys": [
                {"output_chain": "A", "output_res_idx": 12, "group_id": 1},
                {"output_chain": "A", "output_res_idx": 45, "group_id": 2},
            ],
        }
    )
    assert mapping["reference_chain"] == "A"
    assert mapping["folded_chain"] == "A"


def test_build_motif_group_mappings_from_source_residue_keys():
    sample_metadata = {
        "source_residue_keys": [
            {"native_key": "A38", "output_res_idx": 12, "group_id": 1},
            {"native_key": "A14", "output_res_idx": 23, "group_id": 1},
            {"native_key": "A22", "output_res_idx": 45, "group_id": 2},
        ],
        "motif_native_keys": ["A38", "A14", "A22"],
    }
    mapping = {
        "motif_reference_residues": [12, 23, 45],
        "motif_folded_residues": [2, 7, 50],
        "motif_bb_reference_residues": [12, 45],
        "motif_bb_folded_residues": [2, 50],
    }
    groups = build_motif_group_mappings(sample_metadata, mapping)
    assert len(groups) == 2
    assert groups[0]["group_id"] == "1"
    assert groups[0]["reference_residues"] == [12, 23]
    assert groups[0]["folded_residues"] == [2, 7]
    assert groups[0]["bb_reference_residues"] == [12]
    assert groups[0]["bb_folded_residues"] == [2]


def test_build_motif_group_mappings_falls_back_to_motif_groups():
    sample_metadata = {
        "motif_groups": [["A38", "A14"], ["A22-23"]],
        "motif_native_keys": ["A38", "A14", "A22", "A23"],
    }
    mapping = {
        "motif_reference_residues": [12, 23, 45, 46],
        "motif_folded_residues": [2, 7, 50, 51],
        "motif_bb_reference_residues": [12, 45],
        "motif_bb_folded_residues": [2, 50],
    }
    groups = build_motif_group_mappings(sample_metadata, mapping)
    assert len(groups) == 2
    assert groups[1]["group_id"] == "2"
    assert groups[1]["reference_residues"] == [45, 46]
    assert groups[1]["folded_residues"] == [50, 51]
    assert groups[1]["bb_reference_residues"] == [45]
    assert groups[1]["bb_folded_residues"] == [50]


def test_build_motif_group_mappings_missing_group_metadata_returns_empty():
    groups = build_motif_group_mappings(
        sample_metadata={"motif_native_keys": ["A38"]},
        mapping={
            "motif_reference_residues": [12],
            "motif_folded_residues": [2],
            "motif_bb_reference_residues": [12],
            "motif_bb_folded_residues": [2],
        },
    )
    assert groups == []


def test_pose_indices_from_residue_pairs_dedupes_and_filters_invalid():
    def pdb2pose(chain, resid):
        mapping = {
            ("A", 48): 10,
            ("A", 49): 11,
            ("A", 99): 0,
        }
        return mapping.get((chain, resid), 0)

    pairs = [(48, "A"), (49, "A"), (48, "A"), (49, "A"), (99, "A")]
    assert pose_indices_from_residue_pairs(pairs, pdb2pose) == [10, 11]
