from __future__ import annotations

from nomadflow.benchmark.pmpnn_mapping import (
    apply_eval_sample_metadata,
    build_pmpnn_fixed_pos_dict,
    eval_scaffold_chain,
    normalize_eval_fixed_positions,
    rebuild_pmpnn_from_eval_metadata,
)


def test_normalize_eval_fixed_positions_remaps_viz_chain_to_scaffold():
    metadata = {
        "sample_chain_name": "A",
        "fixed_seq_res_idx": [85, 24, 2],
        "fixed_seq_chain": ["B", "B", "B"],
        "fixed_bb_res_idx": [85, 24],
        "fixed_bb_chain": ["B", "B"],
    }

    normalize_eval_fixed_positions(metadata)

    assert metadata["fixed_seq_chain"] == ["A", "A", "A"]
    assert metadata["fixed_bb_chain"] == ["A", "A"]
    assert metadata["fixed_seq_res_idx"] == [85, 24, 2]


def test_build_pmpnn_fixed_pos_dict_uses_scaffold_chain_only():
    samples_metadata = {
        "sample_a": {
            "sample_chain_name": "A",
            "fixed_seq_res_idx": [12, 45],
            "fixed_seq_chain": ["A", "A"],
        }
    }

    fixed = build_pmpnn_fixed_pos_dict(samples_metadata)

    assert fixed == {"sample_a": {"A": [12, 45]}}


def test_apply_eval_sample_metadata_preserves_stage2_eval_fields():
    metadata = {
        "sample_chain_name": "A",
        "source_layout_id": "layout_0",
        "fixed_seq_res_idx": [85, 24, 2],
        "fixed_seq_chain": ["B", "B", "B"],
        "fixed_bb_res_idx": [85, 24, 2],
        "fixed_bb_chain": ["B", "B", "B"],
        "motif_reference_residues": [43, 78, 3],
        "motif_folded_residues": [43, 78, 3],
        "rmsd_reference_chain": "A",
        "rmsd_folded_chain": "A",
    }

    apply_eval_sample_metadata(metadata)

    assert metadata["fixed_seq_chain"] == ["A", "A", "A"]
    assert metadata["motif_reference_residues"] == [43, 78, 3]
    assert metadata["motif_folded_residues"] == [43, 78, 3]


def test_apply_eval_sample_metadata_uses_native_fixed_seq_chain_without_sample_chain_name():
    metadata = {
        "fixed_seq_res_idx": [19, 20, 21],
        "fixed_seq_chain": ["B", "B", "B"],
        "fixed_bb_res_idx": [19, 20],
        "fixed_bb_chain": ["B", "B"],
    }

    apply_eval_sample_metadata(metadata)

    assert metadata["sample_chain_name"] == "B"
    assert metadata["fixed_seq_chain"] == ["B", "B", "B"]
    assert metadata["rmsd_reference_chain"] == "B"
    assert metadata["rmsd_folded_chain"] == "A"


def test_rebuild_pmpnn_from_eval_metadata_repairs_chain_mismatch():
    samples_metadata = {
        "len_109_protein_id0_0_0": {
            "sample_chain_name": "A",
            "fixed_seq_res_idx": [85, 24, 2],
            "fixed_seq_chain": ["B", "B", "B"],
        }
    }

    fixed = rebuild_pmpnn_from_eval_metadata(samples_metadata)

    assert fixed == {"len_109_protein_id0_0_0": {"A": [85, 24, 2]}}


def test_eval_scaffold_chain_prefers_sample_chain_name_over_fixed_seq():
    assert eval_scaffold_chain({"sample_chain_name": "P", "fixed_seq_chain": ["B"]}) == "P"


def test_eval_scaffold_chain_falls_back_to_unique_fixed_seq_chain():
    assert eval_scaffold_chain({"fixed_seq_chain": ["P", "P"]}) == "P"


def test_eval_scaffold_chain_defaults_to_a_when_unspecified():
    assert eval_scaffold_chain({}) == "A"


def test_rebuild_pmpnn_native_chain_without_sample_chain_name():
    samples_metadata = {
        "len_53_1ycr": {
            "fixed_seq_res_idx": [19, 20],
            "fixed_seq_chain": ["B", "B"],
        }
    }
    fixed = rebuild_pmpnn_from_eval_metadata(samples_metadata)
    assert fixed == {"len_53_1ycr": {"B": [19, 20]}}
