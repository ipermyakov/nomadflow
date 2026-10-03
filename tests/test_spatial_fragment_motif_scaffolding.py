"""Tests for spatial-fragment motif scaffolding."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nomadflow.boltz.data import const
from nomadflow.boltz.data.types import Bond, Connection, Interface, Structure
from nomadflow.data.datasets import datamodule as dm
from nomadflow.data.featurize.cropper import Cropper
from nomadflow.data.featurize.tokenize import convert_atom_str_to_tuple
from nomadflow.openfold.data import residue_constants as rc
from nomadflow.runtime.training.motif_partition import asym_id_for_residue, ca_coord_for_residue
from nomadflow.runtime.training.motif_scaffold import (
    MotifScaffolding,
    SpatialFragmentMotifScaffolding,
    _protein_standard_residue_indices,
    _spatial_fragment_residues,
)


def _make_alanine_chain(num_residues: int, *, spacing: float = 3.8):
    atom_names = [name for name in rc.restype_name_to_atom14_names["ALA"] if name]
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
    residues["name"] = "ALA"
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


def _motif_global_indices(struct, resolved_mask, motif_compressed):
    resolved_idx = np.flatnonzero(resolved_mask)
    return resolved_idx[np.asarray(motif_compressed, dtype=bool)]


def _assert_single_spatial_fragment(struct, motif_global, pool):
    asym = asym_id_for_residue(struct, int(motif_global[0]))
    chain_pool = [int(i) for i in pool if asym_id_for_residue(struct, int(i)) == asym]
    target_size = int(motif_global.size)
    for seed in motif_global:
        seed_coord = ca_coord_for_residue(struct, int(seed))
        ranked = sorted(
            (float(np.linalg.norm(ca_coord_for_residue(struct, i) - seed_coord)), int(i))
            for i in chain_pool
            if ca_coord_for_residue(struct, i) is not None
        )
        nearest = {idx for _, idx in ranked[:target_size]}
        if set(motif_global.tolist()).issubset(nearest):
            return
    pytest.fail("motif residues are not a per-chain CA k-NN cluster from any seed")


def _as_structure(struct) -> Structure:
    return Structure(
        atoms=struct.atoms,
        bonds=np.zeros(0, dtype=Bond),
        residues=struct.residues,
        chains=struct.chains,
        connections=np.zeros(0, dtype=Connection),
        interfaces=np.zeros(0, dtype=Interface),
        mask=struct.mask,
    )


def test_base_generate_motif_mask_backward_compatible():
    task = MotifScaffolding(max_num_segments=4, max_frac_res=0.5, max_num_res=40)
    rng = np.random.RandomState(42)
    mask_legacy = task.generate_motif_mask(50, np_rng=rng)
    mask_kwargs = task.generate_motif_mask(
        50,
        np_rng=np.random.RandomState(42),
        data=None,
        resolved_mask=None,
        num_res=None,
    )
    assert mask_legacy.dtype == torch.bool
    assert torch.equal(mask_legacy, mask_kwargs)


def test_spatial_fragment_shell_sampling_uniform_in_distance_band():
    struct = _make_alanine_chain(40, spacing=3.8)
    resolved = struct.residues["is_present"]
    pool = _protein_standard_residue_indices(struct, resolved)
    used_mask = np.zeros(struct.residues.shape[0], dtype=bool)
    seed = 0
    center = 11.4
    half_width = 3.8
    picked: set[int] = set()
    for trial in range(200):
        selected = _spatial_fragment_residues(
            struct,
            seed,
            pool,
            used_mask,
            2,
            max_ca_dist=None,
            np_rng=np.random.RandomState(trial),
            shell_sampling=True,
            shell_center_dist=center,
            shell_half_width=half_width,
        )
        assert selected.size == 2
        assert seed in selected
        other = int(selected[selected != seed][0])
        dist = float(
            np.linalg.norm(
                ca_coord_for_residue(struct, other) - ca_coord_for_residue(struct, seed)
            )
        )
        assert abs(dist - center) <= half_width + 1e-5
        picked.add(other)
    assert len(picked) >= 2


def test_spatial_fragment_shell_sampling_only_for_small_fragments():
    struct = _make_alanine_chain(40)
    resolved = struct.residues["is_present"]
    task = SpatialFragmentMotifScaffolding(
        max_num_fragments=1,
        max_num_res=8,
        small_fragment_shell_sampling=True,
        small_fragment_shell_max_size=6,
        small_fragment_shell_center_dist=12.0,
        small_fragment_shell_half_width=2.0,
    )
    mask_small, _ = task.generate_motif_mask_and_groups(
        int(resolved.sum()),
        np_rng=np.random.RandomState(1),
        data=struct,
        resolved_mask=resolved,
        num_res=4,
    )
    mask_large, _ = task.generate_motif_mask_and_groups(
        int(resolved.sum()),
        np_rng=np.random.RandomState(1),
        data=struct,
        resolved_mask=resolved,
        num_res=8,
    )
    assert int(mask_small.sum()) == 4
    assert int(mask_large.sum()) == 8


def test_spatial_fragment_residues_are_ca_knn_on_chain():
    struct = _make_alanine_chain(24)
    resolved = struct.residues["is_present"]
    pool = _protein_standard_residue_indices(struct, resolved)
    np_rng = np.random.RandomState(0)
    seed = int(np_rng.choice(pool))
    used_mask = np.zeros(struct.residues.shape[0], dtype=bool)
    selected = _spatial_fragment_residues(
        struct,
        seed,
        pool,
        used_mask,
        5,
        max_ca_dist=None,
    )
    assert selected.size == 5
    assert seed in selected
    _assert_single_spatial_fragment(struct, selected, pool)


def test_spatial_fragment_motif_invariants_single_fragment():
    struct = _make_alanine_chain(40)
    resolved = struct.residues["is_present"]
    task = SpatialFragmentMotifScaffolding(
        max_frac_res=0.5,
        max_num_res=12,
        max_num_fragments=1,
        multi_motif_se3=True,
        multi_motif_max_groups=4,
    )
    np_rng = np.random.RandomState(7)
    mask, groups_full = task.generate_motif_mask_and_groups(
        int(resolved.sum()),
        np_rng=np_rng,
        data=struct,
        resolved_mask=resolved,
    )
    motif_global = _motif_global_indices(struct, resolved, mask.numpy())
    pool = _protein_standard_residue_indices(struct, resolved)

    assert motif_global.size == int(mask.sum())
    assert motif_global.size == len(np.unique(motif_global))
    assert np.all(resolved[motif_global])
    assert np.all(groups_full[motif_global] > 0)
    _assert_single_spatial_fragment(struct, motif_global, pool)


def test_spatial_fragment_motif_deterministic():
    struct = _make_alanine_chain(30)
    resolved = struct.residues["is_present"]
    task = SpatialFragmentMotifScaffolding(max_num_fragments=3, max_num_res=10)
    kwargs = dict(
        data=struct,
        resolved_mask=resolved,
        num_res=8,
    )
    mask_a, groups_a = task.generate_motif_mask_and_groups(
        int(resolved.sum()), np_rng=np.random.RandomState(11), **kwargs
    )
    mask_b, groups_b = task.generate_motif_mask_and_groups(
        int(resolved.sum()), np_rng=np.random.RandomState(11), **kwargs
    )
    assert torch.equal(mask_a, mask_b)
    assert np.array_equal(groups_a, groups_b)


def test_spatial_fragment_constant_group_per_fragment():
    struct = _make_alanine_chain(36)
    resolved = struct.residues["is_present"]
    task = SpatialFragmentMotifScaffolding(
        max_num_fragments=3,
        max_num_res=9,
        multi_motif_se3=True,
        multi_motif_max_groups=3,
    )
    mask, groups_full = task.generate_motif_mask_and_groups(
        int(resolved.sum()),
        np_rng=np.random.RandomState(3),
        data=struct,
        resolved_mask=resolved,
        num_res=9,
    )
    motif_global = _motif_global_indices(struct, resolved, mask.numpy())
    assert np.all(groups_full[motif_global] > 0)

    # Each spatial fragment is grown in one loop iteration with one group id.
    # Re-run generation and check id constancy within each k-NN ball from seeds.
    used_mask = np.zeros(struct.residues.shape[0], dtype=bool)
    pool = _protein_standard_residue_indices(struct, resolved)
    for res_idx in motif_global:
        if used_mask[res_idx]:
            continue
        fragment = _spatial_fragment_residues(
            struct,
            int(res_idx),
            pool,
            used_mask,
            len(motif_global),
            max_ca_dist=task.max_ca_dist,
        )
        overlap = [idx for idx in fragment if idx in motif_global]
        if len(overlap) >= 2:
            fragment_groups = groups_full[overlap]
            assert fragment_groups.min() == fragment_groups.max()
        used_mask[fragment] = True


def test_one_fragment_one_group_assigns_unique_ids():
    task = SpatialFragmentMotifScaffolding(
        multi_motif_max_groups=8,
        one_fragment_one_group=True,
    )
    assert np.array_equal(
        task._sample_fragment_group_ids(4, np.random.RandomState(0)),
        np.array([1, 2, 3, 4], dtype=np.int32),
    )
    assert np.array_equal(
        task._sample_fragment_group_ids(10, np.random.RandomState(0)),
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 1, 2], dtype=np.int32),
    )

    struct = _make_alanine_chain(40)
    resolved = struct.residues["is_present"]
    kwargs = dict(data=struct, resolved_mask=resolved, num_res=12)
    unique_per_fragment: list[int] = []
    unique_shared: list[int] = []
    for seed in range(40):
        mask_u, groups_u = SpatialFragmentMotifScaffolding(
            max_num_fragments=4,
            one_fragment_one_group=True,
            multi_motif_max_groups=8,
        ).generate_motif_mask_and_groups(
            int(resolved.sum()),
            np_rng=np.random.RandomState(seed),
            **kwargs,
        )
        mask_s, groups_s = SpatialFragmentMotifScaffolding(
            max_num_fragments=4,
            one_fragment_one_group=False,
            multi_motif_max_groups=8,
        ).generate_motif_mask_and_groups(
            int(resolved.sum()),
            np_rng=np.random.RandomState(seed),
            **kwargs,
        )
        if int(mask_u.sum()) < 6:
            continue
        motif_u = _motif_global_indices(struct, resolved, mask_u.numpy())
        motif_s = _motif_global_indices(struct, resolved, mask_s.numpy())
        unique_per_fragment.append(len(np.unique(groups_u[motif_u])))
        unique_shared.append(len(np.unique(groups_s[motif_s])))
    assert unique_per_fragment
    assert np.mean(unique_per_fragment) > np.mean(unique_shared)


def test_spatial_fragment_sample_t_preserves_generated_groups():
    struct = _make_alanine_chain(28)
    task = SpatialFragmentMotifScaffolding(
        max_num_fragments=2,
        max_num_res=8,
        multi_motif_se3=True,
    )
    out = task.sample_t_and_mask(struct, np_rng=np.random.RandomState(5))
    resolved = struct.residues["is_present"]
    motif_global = np.flatnonzero(~out["res_type_noising_mask"] & resolved)
    assert motif_global.size > 0
    assert np.all(out["motif_group_id_residue"][motif_global] > 0)


def test_spatial_fragment_sample_t_zeros_groups_when_se3_disabled():
    struct = _make_alanine_chain(24)
    task = SpatialFragmentMotifScaffolding(multi_motif_se3=False, max_num_res=8)
    out = task.sample_t_and_mask(struct, np_rng=np.random.RandomState(1))
    assert np.all(out["motif_group_id_residue"] == 0)


def test_prepare_keeps_all_sampled_spatial_fragments(monkeypatch):
    struct = _as_structure(_make_alanine_chain(40))
    task = SpatialFragmentMotifScaffolding(
        max_frac_res=0.5,
        max_num_res=8,
        max_num_fragments=2,
        min_fragment_size=2,
        max_fragment_size=4,
        p_is_unindexed=1.0,
        p_fix_tip=0.0,
        p_fix_bb=0.0,
        p_fix_all=1.0,
        multi_motif_se3=True,
        multi_motif_max_groups=2,
        one_fragment_one_group=True,
    )
    monkeypatch.setattr(dm, "load_input", lambda record, data_dir: struct)

    def prepare(seed):
        return dm.prepare_training_tokenized_example(
            dataset=SimpleNamespace(data_dir="unused"),
            sample=SimpleNamespace(
                record=SimpleNamespace(id="spatial"),
                chain_id=None,
                interface_id=None,
            ),
            task=task,
            cropper=Cropper(min_neighborhood=0, max_neighborhood=0),
            max_crop_residues=20,
            remove_mol_types=[],
            mask_nonstandard=False,
            np_rng=np.random.RandomState(seed),
            torch_generator=torch.Generator().manual_seed(seed),
        )

    prepared = prepare(7)
    repeated = prepare(7)

    sampled_groups = prepared.task_data["motif_group_id_residue"]
    expected = {
        (int(res_idx), int(sampled_groups[res_idx]))
        for res_idx in np.flatnonzero(sampled_groups > 0)
    }
    copy_tokens = prepared.tokenized.tokens[prepared.tokenized.tokens["is_copy"]]
    actual = {
        (int(token["res_idx"]), int(token["motif_group_id"]))
        for token in copy_tokens
    }
    marker_tokens = prepared.tokenized.tokens[
        prepared.tokenized.tokens["is_motif_group_aggregator"]
    ]

    assert prepared.tokenized.structure.residues["is_present"].sum() == 18
    assert actual == expected
    assert set(marker_tokens["motif_group_id"].tolist()) == {
        group_id for _, group_id in expected
    }
    assert len(prepared.tokenized.tokens) <= 20
    assert np.array_equal(
        prepared.task_data["motif_group_id_residue"],
        repeated.task_data["motif_group_id_residue"],
    )
    assert np.array_equal(prepared.tokenized.tokens, repeated.tokenized.tokens)
    assert np.array_equal(prepared.tokenized.rigids, repeated.tokenized.rigids)
