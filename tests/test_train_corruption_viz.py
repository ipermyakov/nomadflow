import copy

import numpy as np

import torch

from nomadflow.boltz.data import const
from nomadflow.boltz.data.types import Atom, Chain, SamplingResidue, Structure
from nomadflow.data.featurize.cropper import Cropper
from nomadflow.data.featurize.tokenize import Rigid, Token, TokenBond, Tokenized
from nomadflow.data.write import trajectory as traj
from nomadflow.stoch_interp.multiframe import MultiSE3Interpolant


def _make_dense_batch():
    rigids_1 = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [1.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    return {
        "rigids": {
            "rigids_1": rigids_1.clone(),
            "rigids_mask": torch.tensor([[True, True]]),
            "rigids_noising_mask": torch.tensor([[True, False]]),
        }
    }


def test_dense_corruption_endpoints_support_deterministic_replay_and_fixed_regions():
    corrupter = MultiSE3Interpolant(
        prealign_noise=False,
        use_stochastic_centering=False,
        center_on_motif=True,
        use_uniform_rot_noise=True,
    )
    batch = _make_dense_batch()

    first = corrupter.sample_dense_corruption_endpoints(copy.deepcopy(batch), rng_seed=11)
    second = corrupter.sample_dense_corruption_endpoints(copy.deepcopy(batch), rng_seed=11)

    assert torch.allclose(first["rigids_0"], second["rigids_0"])
    assert torch.allclose(first["rigids_1"], second["rigids_1"])

    t0 = corrupter.interpolate_dense_corruption(first, torch.tensor([[0.0]], dtype=torch.float32))
    t1 = corrupter.interpolate_dense_corruption(first, torch.tensor([[1.0]], dtype=torch.float32))

    assert torch.allclose(t1["rigids_t"], first["rigids_1"])
    assert torch.allclose(t0["rigids_t"][0, 1], first["rigids_1"][0, 1])
    assert not torch.allclose(t0["rigids_t"][0, 0], first["rigids_1"][0, 0])


def test_write_multimodel_pdb_wraps_each_frame_in_model_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(traj, "render_tokenized_structure_for_pdb", lambda frame, **kwargs: frame)
    monkeypatch.setattr(traj, "_needs_chain_renaming", lambda struct: False)
    monkeypatch.setattr(traj, "to_pdb", lambda struct, **kwargs: f"ATOM {struct}\nEND\n")

    output_path, total_conects, model_count = traj.write_multimodel_pdb(
        tmp_path / "traj.pdb",
        ["frame-a", "frame-b"],
        add_intra_rigid_conects=False,
    )

    contents = output_path.read_text(encoding="utf-8")

    assert total_conects == 0
    assert model_count == 2
    assert "MODEL        0\nATOM frame-a\nENDMDL" in contents
    assert "MODEL        1\nATOM frame-b\nENDMDL" in contents


def test_ensure_sampling_residues_adds_is_copy_field_for_training_structures():
    residues = np.zeros(
        1,
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
    struct = Structure(
        atoms=np.zeros(0, dtype=[]),
        bonds=np.zeros(0, dtype=[]),
        residues=residues,
        chains=np.zeros(0, dtype=[]),
        connections=np.zeros(0, dtype=[]),
        interfaces=np.zeros(0, dtype=[]),
        mask=np.zeros(0, dtype=bool),
    )

    normalized = traj._ensure_sampling_residues(struct)

    assert "is_copy" in normalized.residues.dtype.names
    assert normalized.residues["is_copy"].tolist() == [False]


def test_prune_structure_to_tokenized_residues_keeps_only_cropped_residues():
    residues = np.zeros(
        3,
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
            ("is_copy", bool),
        ],
    )
    residues["res_idx"] = [10, 11, 12]
    residues["atom_idx"] = [0, 4, 8]
    residues["atom_num"] = [4, 4, 4]

    chains = np.zeros(
        1,
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
    chains["res_idx"] = [0]
    chains["res_num"] = [3]

    struct = Structure(
        atoms=np.zeros(0, dtype=[]),
        bonds=np.zeros(0, dtype=[]),
        residues=residues,
        chains=chains,
        connections=np.zeros(0, dtype=[]),
        interfaces=np.zeros(0, dtype=[]),
        mask=np.array([True]),
    )
    tokenized = type(
        "TokenizedLike",
        (),
        {"tokens": np.array([(10,), (12,)], dtype=[("res_idx", np.int32)])},
    )()

    pruned = traj._prune_structure_to_tokenized_residues(struct, tokenized)

    assert pruned.residues["res_idx"].tolist() == [10, 12]
    assert pruned.chains["res_idx"].tolist() == [0]
    assert pruned.chains["res_num"].tolist() == [2]


def _make_train_tokenized_with_copy_tokens():
    atoms = np.zeros(12, dtype=Atom)
    atoms["element"] = 6
    atoms["is_present"] = True

    residues = np.zeros(3, dtype=SamplingResidue)
    residues["name"] = ["ALA", "GLY", "SER"]
    residues["res_type"] = [0, 1, 2]
    residues["res_idx"] = [10, 11, 12]
    residues["atom_idx"] = [0, 4, 8]
    residues["atom_num"] = [4, 4, 4]
    residues["atom_center"] = 1
    residues["atom_disto"] = 1
    residues["is_standard"] = True
    residues["is_present"] = True
    residues["is_copy"] = False

    chains = np.zeros(1, dtype=Chain)
    chains["name"] = ["A"]
    chains["mol_type"] = 0
    chains["entity_id"] = [0]
    chains["sym_id"] = [0]
    chains["asym_id"] = [0]
    chains["atom_idx"] = [0]
    chains["atom_num"] = [12]
    chains["res_idx"] = [0]
    chains["res_num"] = [3]
    chains["cyclic_period"] = [0]

    tokens = np.zeros(4, dtype=Token)
    tokens["token_idx"] = [0, 1, 2, 3]
    tokens["res_idx"] = [10, 12, 11, 12]
    tokens["res_type"] = [0, 2, 1, 2]
    tokens["res_name"] = ["ALA", "SER", "GLY", "SER"]
    tokens["rigid_idx"] = [0, 3, 6, 9]
    tokens["rigid_num"] = [3, 3, 3, 3]
    tokens["sym_id"] = [0, 0, 0, 0]
    tokens["asym_id"] = [0, 0, 0, 0]
    tokens["entity_id"] = [0, 0, 0, 0]
    tokens["mol_type"] = [0, 0, 0, 0]
    tokens["resolved_mask"] = True
    tokens["is_copy"] = [False, False, True, True]
    tokens["is_unindexed"] = [False, False, False, True]
    tokens["is_atomized"] = False
    tokens["seq_noising_mask"] = True

    rigids = np.zeros(12, dtype=Rigid)
    rigids["rigid_idx"] = np.arange(12)
    rigids["token_idx"] = np.repeat(np.arange(4), 3)
    rigids["sidechain_idx"] = np.tile(np.arange(3), 4)
    rigids["tensor7"][:, 0] = 1.0
    rigids["is_present"] = True
    rigids["rigids_noising_mask"] = True
    rigids["num_real_input_axes"] = 2

    return Tokenized(
        tokens=tokens,
        rigids=rigids,
        bonds=np.zeros(0, dtype=TokenBond),
        structure=Structure(
            atoms=atoms,
            bonds=np.zeros(0, dtype=[]),
            residues=residues,
            chains=chains,
            connections=np.zeros(0, dtype=[]),
            interfaces=np.zeros(0, dtype=[]),
            mask=np.array([True]),
        ),
    )


def test_adapt_tokenized_for_train_viz_copy_semantics_appends_copy_residues():
    tokenized = _make_train_tokenized_with_copy_tokens()

    adapted = traj.adapt_tokenized_for_train_viz_copy_semantics(tokenized)

    assert adapted.tokens["res_idx"].tolist() == [10, 12, 13, 14]
    assert adapted.structure.residues["res_idx"].tolist() == [10, 12, 13, 14]
    assert adapted.structure.residues["is_copy"].tolist() == [False, False, True, True]
    assert adapted.structure.residues["name"].tolist() == ["ALA", "SER", "GLY", "SER"]
    assert adapted.structure.chains["res_num"].tolist() == [2, 2]


def test_render_tokenized_structure_for_pdb_outputs_aggregator_markers_as_gly_chain():
    residues = np.zeros(1, dtype=SamplingResidue)
    residues["name"] = ["ALA"]
    residues["res_type"] = [const.token_ids["ALA"]]
    residues["res_idx"] = [0]
    residues["atom_num"] = [len(const.ref_atoms["ALA"])]
    residues["atom_center"] = 1
    residues["atom_disto"] = 1
    residues["is_standard"] = True
    residues["is_present"] = True
    residues["is_copy"] = False

    chains = np.zeros(1, dtype=Chain)
    chains["name"] = ["A"]
    chains["mol_type"] = const.chain_type_ids["PROTEIN"]
    chains["entity_id"] = [0]
    chains["sym_id"] = [0]
    chains["asym_id"] = [0]
    chains["res_idx"] = [0]
    chains["res_num"] = [1]

    tokens = np.zeros(2, dtype=Token)
    tokens["token_idx"] = [0, 1]
    tokens["res_idx"] = [0, -1]
    tokens["res_type"] = [const.token_ids["ALA"], const.token_ids["GLY"]]
    tokens["res_name"] = ["ALA", "VGM"]
    tokens["rigid_idx"] = [0, 3]
    tokens["rigid_num"] = [3, 1]
    tokens["sym_id"] = [0, -1]
    tokens["asym_id"] = [0, -1]
    tokens["entity_id"] = [0, -1]
    tokens["mol_type"] = [const.chain_type_ids["PROTEIN"], -1]
    tokens["resolved_mask"] = True
    tokens["center_coords"] = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    tokens["is_copy"] = False
    tokens["is_unindexed"] = [False, True]
    tokens["is_atomized"] = False
    tokens["seq_noising_mask"] = False
    tokens["motif_group_id"] = [0, 1]
    tokens["is_motif_group_aggregator"] = [False, True]

    rigids = np.zeros(4, dtype=Rigid)
    rigids["rigid_idx"] = np.arange(4)
    rigids["token_idx"] = [0, 0, 0, 1]
    rigids["sidechain_idx"] = [0, 1, 2, 0]
    rigids["tensor7"][:, 0] = 1.0
    rigids["tensor7"][3, 4:] = [10.0, 0.0, 0.0]
    rigids["is_present"] = True
    rigids["rigids_noising_mask"] = [True, True, True, True]
    rigids["num_real_input_axes"] = 2
    rigids["motif_group_id"] = [0, 0, 0, 1]
    rigids["is_motif_group_aggregator"] = [False, False, False, True]

    tokenized = Tokenized(
        tokens=tokens,
        rigids=rigids,
        bonds=np.zeros(0, dtype=TokenBond),
        structure=Structure(
            atoms=np.zeros(0, dtype=Atom),
            bonds=np.zeros(0, dtype=[]),
            residues=residues,
            chains=chains,
            connections=np.zeros(0, dtype=[]),
            interfaces=np.zeros(0, dtype=[]),
            mask=np.array([True]),
        ),
    )

    struct = traj.render_tokenized_structure_for_pdb(
        tokenized,
        output_motif_chains=True,
        take_copy_structure=False,
    )
    pdb_str = traj.to_pdb(struct, rename_chains=False)

    assert struct.chains["name"].tolist() == ["A", "V"]
    assert struct.residues["name"].tolist() == ["ALA", "GLY"]
    assert struct.residues["atom_num"].tolist() == [
        len(const.ref_atoms["ALA"]),
        len(const.ref_atoms["GLY"]),
    ]
    assert "GLY V" in pdb_str
    assert "  10.000   0.000   0.000" in pdb_str

    batched_pdb_str = traj.build_multimodel_pdb_string_batched(
        [tokenized],
        output_motif_chains=True,
        take_copy_structure=False,
        use_cuda_batch=False,
    )
    assert "GLY V" in batched_pdb_str


def test_cropper_reindexes_tokens_rigids_and_bonds_for_training_consistency():
    tokens = np.zeros(4, dtype=Token)
    tokens["token_idx"] = [0, 1, 2, 3]
    tokens["res_idx"] = [10, 11, 20, 21]
    tokens["res_type"] = [0, 1, 2, 3]
    tokens["res_name"] = ["ALA", "GLY", "SER", "THR"]
    tokens["rigid_idx"] = [0, 3, 6, 9]
    tokens["rigid_num"] = [3, 3, 3, 3]
    tokens["sym_id"] = [0, 0, 0, 0]
    tokens["asym_id"] = [0, 0, 1, 1]
    tokens["entity_id"] = [0, 0, 1, 1]
    tokens["mol_type"] = [0, 0, 0, 0]
    tokens["resolved_mask"] = True
    tokens["center_coords"] = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0], [11.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    tokens["is_copy"] = False
    tokens["is_unindexed"] = False
    tokens["is_atomized"] = False
    tokens["seq_noising_mask"] = True

    rigids = np.zeros(12, dtype=Rigid)
    rigids["rigid_idx"] = np.arange(12)
    rigids["token_idx"] = np.repeat(np.arange(4), 3)
    rigids["sidechain_idx"] = np.tile(np.arange(3), 4)
    rigids["tensor7"][:, 0] = 1.0
    rigids["tensor7"][:, 4] = np.arange(12, dtype=np.float32)
    rigids["is_present"] = True
    rigids["rigids_noising_mask"] = True
    rigids["num_real_input_axes"] = 2

    bonds = np.array([(2, 3, 1), (0, 1, 1)], dtype=TokenBond)

    chains = np.zeros(2, dtype=Chain)
    chains["name"] = ["A", "B"]
    chains["mol_type"] = 0
    chains["entity_id"] = [0, 1]
    chains["sym_id"] = [0, 0]
    chains["asym_id"] = [0, 1]
    chains["atom_idx"] = [0, 0]
    chains["atom_num"] = [0, 0]
    chains["res_idx"] = [0, 0]
    chains["res_num"] = [0, 0]
    chains["cyclic_period"] = [0, 0]

    tokenized = Tokenized(
        tokens=tokens,
        rigids=rigids,
        bonds=bonds,
        structure=Structure(
            atoms=np.zeros(0, dtype=Atom),
            bonds=np.zeros(0, dtype=[]),
            residues=np.zeros(0, dtype=SamplingResidue),
            chains=chains,
            connections=np.zeros(0, dtype=[]),
            interfaces=np.zeros(0, dtype=[("chain_1", np.int32), ("chain_2", np.int32)]),
            mask=np.array([True, True]),
        ),
    )

    cropped = Cropper(min_neighborhood=2, max_neighborhood=2).crop(
        tokenized,
        max_tokens=2,
        random=np.random.RandomState(0),
        chain_id=1,
    )

    assert cropped.tokens["token_idx"].tolist() == [0, 1]
    assert cropped.tokens["rigid_idx"].tolist() == [0, 3]
    assert cropped.tokens["res_idx"].tolist() == [20, 21]
    assert len(cropped.rigids) == 6
    assert cropped.rigids["rigid_idx"].tolist() == list(range(6))
    assert cropped.rigids["token_idx"].tolist() == [0, 0, 0, 1, 1, 1]
    assert np.array_equal(cropped.rigids["tensor7"][:, 4], np.array([6, 7, 8, 9, 10, 11], dtype=np.float32))
    assert cropped.bonds.tolist() == [(0, 1, 1)]
