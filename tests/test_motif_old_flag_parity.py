"""Old-flag motif training parity: historical rigid_noise_mask mapping."""

import numpy as np
import pytest
import torch

from nomadflow.boltz.data import const
from nomadflow.data.featurize.tokenize import convert_atom_str_to_tuple
from nomadflow.openfold.data import residue_constants as rc
from nomadflow.runtime.training.motif_scaffold import (
    MinimalMotifScaffolding,
    MixedMotifScaffolding,
    motif_frame_mask_from_selection,
    rigid_noise_to_atom_noise,
)


def _make_arg_chain(num_residues: int = 4, *, spacing: float = 3.8):
    atom_names = [name for name in rc.restype_name_to_atom14_names["ARG"] if name]
    atoms_per_residue = len(atom_names)
    total_atoms = num_residues * atoms_per_residue

    atoms = np.zeros(
        total_atoms,
        dtype=[("coords", np.float32, (3,)), ("name", "4i1"), ("is_present", bool)],
    )
    atoms["is_present"] = True

    residues = np.zeros(
        num_residues,
        dtype=[
            ("name", "<U5"),
            ("res_type", np.int8),
            ("res_idx", np.int32),
            ("atom_idx", np.int32),
            ("atom_num", np.int32),
            ("atom_center", np.int32),
            ("atom_disto", np.int32),
            ("is_standard", bool),
            ("is_present", bool),
        ],
    )
    residues["name"] = "ARG"
    residues["res_idx"] = np.arange(num_residues, dtype=np.int32)
    residues["atom_idx"] = np.arange(num_residues, dtype=np.int32) * atoms_per_residue
    residues["atom_num"] = atoms_per_residue
    residues["atom_center"] = 1
    residues["atom_disto"] = 1
    residues["is_standard"] = True
    residues["is_present"] = True

    ca_index = atom_names.index("CA")
    for res_idx in range(num_residues):
        atom_start = int(residues["atom_idx"][res_idx])
        for local_idx, atom_name in enumerate(atom_names):
            global_idx = atom_start + local_idx
            atoms["name"][global_idx] = convert_atom_str_to_tuple(atom_name)
        atoms["coords"][atom_start + ca_index] = [spacing * res_idx, 0.0, 0.0]

    chains = np.array(
        [
            (
                "A",
                const.chain_type_ids["PROTEIN"],
                0,
                0,
                0,
                0,
                total_atoms,
                0,
                num_residues,
                0,
            )
        ],
        dtype=[
            ("name", "<U5"),
            ("mol_type", np.int8),
            ("entity_id", np.int32),
            ("sym_id", np.int32),
            ("asym_id", np.int32),
            ("atom_idx", np.int32),
            ("atom_num", np.int32),
            ("res_idx", np.int32),
            ("res_num", np.int32),
            ("cyclic_period", np.int32),
        ],
    )

    class _Struct:
        pass

    struct = _Struct()
    struct.residues = residues
    struct.chains = chains
    struct.atoms = atoms
    struct.mask = np.array([True])
    return struct


def _force_all_motif(task):
    def _all_motif(N, np_rng=None, *, data=None, resolved_mask=None, num_res=None):
        del np_rng, data, resolved_mask, num_res
        return torch.ones(N, dtype=torch.bool), None

    task.generate_motif_mask_and_groups = _all_motif
    return task


HISTORICAL_RIGID_NOISE = {
    "tip": [True, False, True],
    "bb": [False, True, True],
    "all": [False, False, False],
}


@pytest.mark.parametrize(
    "mode,p_fix_tip,p_fix_bb,p_fix_all,expected_rigid_noise",
    [
        ("tip", 1.0, 0.0, 0.0, HISTORICAL_RIGID_NOISE["tip"]),
        ("bb", 0.0, 1.0, 0.0, HISTORICAL_RIGID_NOISE["bb"]),
        ("all", 0.0, 0.0, 1.0, HISTORICAL_RIGID_NOISE["all"]),
    ],
)
def test_old_flag_mixed_motif_rigid_noise_matches_historical(
    mode, p_fix_tip, p_fix_bb, p_fix_all, expected_rigid_noise
):
    del mode
    struct = _make_arg_chain(4)
    task = _force_all_motif(
        MixedMotifScaffolding(
            max_frac_res=1.0,
            max_num_res=4,
            max_num_segments=1,
            p_is_unindexed=0.0,
            p_fix_tip=p_fix_tip,
            p_fix_bb=p_fix_bb,
            p_fix_all=p_fix_all,
            multi_motif_se3=False,
        )
    )
    out = task.sample_t_and_mask(struct, np_rng=np.random.RandomState(0))

    assert np.all(out["motif_group_id_residue"] == 0)
    assert np.all(out["motif_group_id_rigid_by_residue_frame"] == 0)

    expected_frame_mask = motif_frame_mask_from_selection(
        fix_tip=True if p_fix_tip == 1.0 else False,
        noise_sidechain_only=True if p_fix_bb == 1.0 else False,
        is_motif=True,
    )
    assert expected_frame_mask.tolist() == (~np.asarray(expected_rigid_noise)).tolist()

    atom_offset = 0
    for residue in struct.residues:
        atom_num = int(residue["atom_num"])
        atoms = struct.atoms[atom_offset : atom_offset + atom_num]
        expected_atom = rigid_noise_to_atom_noise(
            residue, atoms, expected_rigid_noise
        )
        got_atom = out["atom_noising_mask"][atom_offset : atom_offset + atom_num]
        assert np.array_equal(got_atom, expected_atom)
        atom_offset += atom_num


@pytest.mark.parametrize(
    "p_fix_tip,p_fix_bb,p_fix_all,expected_frame_mask",
    [
        (1.0, 0.0, 0.0, [False, True, False]),
        (0.0, 1.0, 0.0, [True, False, False]),
        (0.0, 0.0, 1.0, [True, True, True]),
    ],
)
def test_multi_motif_se3_keeps_all_true_rigid_noise_and_group_frames(
    p_fix_tip, p_fix_bb, p_fix_all, expected_frame_mask
):
    struct = _make_arg_chain(4)
    task = _force_all_motif(
        MixedMotifScaffolding(
            max_frac_res=1.0,
            max_num_res=4,
            max_num_segments=1,
            p_is_unindexed=0.0,
            p_fix_tip=p_fix_tip,
            p_fix_bb=p_fix_bb,
            p_fix_all=p_fix_all,
            multi_motif_se3=True,
            multi_motif_partition_mode="fragments",
            multi_motif_max_groups=4,
        )
    )
    out = task.sample_t_and_mask(struct, np_rng=np.random.RandomState(1))

    assert np.any(out["motif_group_id_residue"] > 0)
    rigid_gid = out["motif_group_id_rigid_by_residue_frame"]
    for res_pos, group_id in enumerate(out["motif_group_id_residue"]):
        if group_id <= 0:
            continue
        expected = np.asarray(expected_frame_mask, dtype=bool) * int(group_id)
        assert np.array_equal(rigid_gid[res_pos], expected)

    expected_all_noise = [True, True, True]
    atom_offset = 0
    for residue in struct.residues:
        atom_num = int(residue["atom_num"])
        atoms = struct.atoms[atom_offset : atom_offset + atom_num]
        expected_atom = rigid_noise_to_atom_noise(residue, atoms, expected_all_noise)
        got_atom = out["atom_noising_mask"][atom_offset : atom_offset + atom_num]
        assert np.array_equal(got_atom, expected_atom)
        atom_offset += atom_num


def test_old_flag_minimal_motif_matches_historical_fix_all():
    struct = _make_arg_chain(6)
    task = _force_all_motif(
        MinimalMotifScaffolding(
            max_res_islands=1,
            max_res_per_island=6,
            p_is_unindexed=0.0,
            p_fix_tip=0.0,
            p_fix_bb=0.0,
            p_fix_all=1.0,
            multi_motif_se3=False,
        )
    )
    out = task.sample_t_and_mask(struct, np_rng=np.random.RandomState(2))
    assert np.all(out["motif_group_id_residue"] == 0)
    assert not np.any(out["atom_noising_mask"])


def test_mixed_motif_smoke_copy_tokens_fixed_frames_without_flag():
    struct = _make_arg_chain(8)
    task = MixedMotifScaffolding(
        max_frac_res=1.0,
        max_num_res=8,
        max_num_segments=1,
        p_is_unindexed=0.0,
        p_fix_tip=0.0,
        p_fix_bb=0.0,
        p_fix_all=1.0,
        multi_motif_se3=False,
    )
    out = task.sample_t_and_mask(struct, np_rng=np.random.RandomState(3))
    copy_mask = out["copy_indexed_residue_mask"] | out["copy_unindexed_residue_mask"]
    assert copy_mask.any()
    assert np.all(out["motif_group_id_residue"] == 0)

    atom_offset = 0
    fixed_atom_seen = False
    for res_pos, residue in enumerate(struct.residues):
        atom_num = int(residue["atom_num"])
        if copy_mask[res_pos]:
            res_atom_mask = out["atom_noising_mask"][atom_offset : atom_offset + atom_num]
            assert not np.any(res_atom_mask)
            fixed_atom_seen = True
        atom_offset += atom_num
    assert fixed_atom_seen
