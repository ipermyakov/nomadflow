import numpy as np
import pytest

pytest.importorskip("mashumaro")

from nomadflow.boltz.data import const
from nomadflow.boltz.data.types import Atom, Chain, Interface, SamplingResidue, Structure
from nomadflow.data.featurize.sampling import sample_noise_from_struct_template
from nomadflow.data.featurize.tokenize import Token, convert_atom_str_to_tuple


def test_sampling_appends_motif_group_aggregators():
    atom_names = ["N", "CA", "C", "O", "CB"]
    atom_coords = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    atoms = np.zeros(len(atom_names), dtype=Atom)
    atoms["name"] = [convert_atom_str_to_tuple(name) for name in atom_names]
    atoms["element"] = [7, 6, 6, 8, 6]
    atoms["coords"] = atom_coords
    atoms["is_present"] = True

    residues = np.zeros(1, dtype=SamplingResidue)
    residues["name"] = "ALA"
    residues["res_type"] = const.token_ids["ALA"]
    residues["res_idx"] = 0
    residues["atom_num"] = len(atom_names)
    residues["atom_center"] = 1
    residues["atom_disto"] = 1
    residues["is_standard"] = True
    residues["is_present"] = True
    residues["is_copy"] = True

    chains = np.zeros(1, dtype=Chain)
    chains["name"] = "A"
    chains["mol_type"] = const.chain_type_ids["PROTEIN"]
    chains["res_num"] = 1

    struct = Structure(
        atoms=atoms,
        bonds=np.zeros(0, dtype=[]),
        residues=residues,
        chains=chains,
        connections=np.zeros(0, dtype=[]),
        interfaces=np.zeros(0, dtype=Interface),
        mask=np.array([True]),
    )
    task_masks = {
        "atom_noising_mask": np.zeros(len(atom_names), dtype=bool),
        "res_type_noising_mask": np.zeros(1, dtype=bool),
        "residue_is_unindexed_mask": np.zeros(1, dtype=bool),
        "motif_group_by_copy_key": {(0, 0): 1},
    }

    off_token_data, off_rigid_data, _, _ = sample_noise_from_struct_template(
        struct,
        task_masks=task_masks,
    )
    assert not np.any(off_token_data["motif_group_id"] > 0)
    assert not np.any(off_rigid_data["motif_group_id"] > 0)
    assert not np.any(off_token_data["is_motif_group_aggregator"])
    assert not np.any(off_rigid_data["is_motif_group_aggregator"])

    on_task_masks = {
        **task_masks,
        "multi_motif_se3": True,
        "multi_motif_se3_trans_std": 0.0,
        "np_rng": np.random.default_rng(0),
    }
    token_data, rigid_data, _, _ = sample_noise_from_struct_template(
        struct,
        task_masks=on_task_masks,
    )

    marker_tokens = token_data[token_data["is_motif_group_aggregator"]]
    marker_rigids = rigid_data[rigid_data["is_motif_group_aggregator"]]

    assert marker_tokens["motif_group_id"].tolist() == [1]
    assert marker_tokens["asym_id"].tolist() == [-1]
    assert marker_tokens["seq_noising_mask"].tolist() == [False]
    assert marker_rigids["motif_group_id"].tolist() == [1]
    assert marker_rigids["rigids_noising_mask"].tolist() == [True]

    augmented_task_masks = {
        **task_masks,
        "multi_motif_se3": True,
        "multi_motif_se3_trans_std": 4.0,
        "np_rng": np.random.default_rng(42),
    }
    augmented_token_data, augmented_rigid_data, _, _ = sample_noise_from_struct_template(
        struct,
        task_masks=augmented_task_masks,
    )
    motif_mask = rigid_data["motif_group_id"] == 1
    augmented_motif_mask = augmented_rigid_data["motif_group_id"] == 1
    motif_coords = rigid_data["tensor7"][motif_mask & ~rigid_data["is_motif_group_aggregator"], 4:]
    augmented_motif_coords = augmented_rigid_data["tensor7"][
        augmented_motif_mask & ~augmented_rigid_data["is_motif_group_aggregator"],
        4:,
    ]

    assert not np.allclose(augmented_motif_coords, motif_coords)
    assert np.allclose(
        np.linalg.norm(augmented_motif_coords[0] - augmented_motif_coords[1]),
        np.linalg.norm(motif_coords[0] - motif_coords[1]),
        atol=1e-4,
    )
    assert np.allclose(
        augmented_token_data[0]["center_coords"],
        augmented_motif_coords[0],
    )


def test_virtual_group_aggregators_are_excluded_from_fixed_sequence_metadata():
    pytest.importorskip("lightning")

    from nomadflow.runtime.lmod import _real_token_metadata_mask

    token_data = np.zeros(2, dtype=Token)
    token_data["asym_id"] = [0, -1]
    token_data["seq_noising_mask"] = [False, False]
    token_data["is_motif_group_aggregator"] = [False, True]

    fixed_sequence_mask = ~token_data["seq_noising_mask"] & _real_token_metadata_mask(token_data)

    assert fixed_sequence_mask.tolist() == [True, False]
