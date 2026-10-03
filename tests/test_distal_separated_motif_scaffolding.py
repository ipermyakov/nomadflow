"""Tests for DistalSeparatedMotifScaffolding."""

from types import SimpleNamespace

import numpy as np
import torch

from nomadflow.boltz.data import const
from nomadflow.boltz.data.types import Bond, Connection, Interface, Structure
from nomadflow.data.datasets import datamodule as dm
from nomadflow.data.featurize.cropper import Cropper
from nomadflow.data.featurize.tokenize import convert_atom_str_to_tuple
from nomadflow.openfold.data import residue_constants as rc
from nomadflow.runtime.training.motif_partition import (
    asym_id_for_residue,
    build_residue_spatial_lookup,
)
from nomadflow.runtime.training.motif_scaffold import (
    DistalSeparatedMotifScaffolding,
    _expand_contiguous_window,
    _fps_centers,
    _min_pairwise_ca_between_sets,
    _protein_standard_residue_indices,
)


def _make_structure_from_ca_coords(ca_coords: np.ndarray):
    num_residues = int(ca_coords.shape[0])
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
            atoms["coords"][global_idx] = ca_coords[res_idx]
        atoms["coords"][atom_start + ca_index] = ca_coords[res_idx]

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


def _make_alanine_chain(num_residues: int, *, spacing: float = 3.8):
    ca = np.stack(
        [
            spacing * np.arange(num_residues, dtype=np.float32),
            np.zeros(num_residues, dtype=np.float32),
            np.zeros(num_residues, dtype=np.float32),
        ],
        axis=1,
    )
    return _make_structure_from_ca_coords(ca)


def _make_two_strand_sheet(strand_len: int = 40, *, strand_spacing: float = 8.0, res_spacing: float = 3.8):
    """Two parallel strands: residues [0, L) and [L, 2L), spatially close across strands."""
    coords = []
    for i in range(strand_len):
        coords.append([res_spacing * i, 0.0, 0.0])
    for i in range(strand_len):
        coords.append([res_spacing * i, strand_spacing, 0.0])
    return _make_structure_from_ca_coords(np.asarray(coords, dtype=np.float32))


def _make_spatial_grid(n_side: int = 8, *, spacing: float = 6.0):
    """n_side^2 residues on a 2D grid (sequence is row-major)."""
    coords = []
    for i in range(n_side):
        for j in range(n_side):
            coords.append([spacing * j, spacing * i, 0.0])
    return _make_structure_from_ca_coords(np.asarray(coords, dtype=np.float32))


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


def _motif_global_indices(resolved_mask, motif_compressed):
    resolved_idx = np.flatnonzero(resolved_mask)
    return resolved_idx[np.asarray(motif_compressed, dtype=bool)]


def _group_residue_sets(groups: np.ndarray):
    out = {}
    for gid in np.unique(groups):
        gid = int(gid)
        if gid == 0:
            continue
        out[gid] = np.flatnonzero(groups == gid)
    return out


def test_fps_centers_respect_min_distance():
    struct = _make_spatial_grid(8, spacing=6.0)
    resolved = struct.residues["is_present"]
    pool = _protein_standard_residue_indices(struct, resolved)
    _, ca_coords = build_residue_spatial_lookup(struct)
    centers = _fps_centers(pool, ca_coords, num_groups=4, min_center_dist=18.0, np_rng=np.random.RandomState(0))
    assert centers.size >= 2
    for i, a in enumerate(centers):
        for b in centers[i + 1 :]:
            d = float(np.linalg.norm(ca_coords[int(a)] - ca_coords[int(b)]))
            assert d >= 18.0 - 1e-5


def test_expand_contiguous_window_symmetric_and_avoids_forbidden():
    chain = np.arange(20, dtype=np.int32)
    forbidden = np.zeros(20, dtype=bool)
    forbidden[8] = True
    window = _expand_contiguous_window(chain, anchor_idx=10, size=5, forbidden=forbidden)
    assert 10 in window
    assert 8 not in window
    assert window.size >= 3
    assert np.all(np.diff(window) == 1)


def test_inter_group_ca_separation():
    struct = _make_spatial_grid(10, spacing=5.0)
    resolved = struct.residues["is_present"]
    task = DistalSeparatedMotifScaffolding(
        max_groups=4,
        min_center_dist=20.0,
        min_inter_group_ca_dist=12.0,
        p_catalytic=0.5,
        multi_motif_se3=True,
    )
    _, ca_coords = build_residue_spatial_lookup(struct)
    for seed in range(40):
        mask, groups = task.generate_motif_mask_and_groups(
            int(resolved.sum()),
            np_rng=np.random.RandomState(seed),
            data=struct,
            resolved_mask=resolved,
        )
        group_sets = _group_residue_sets(groups)
        ids = sorted(group_sets)
        for i, gid_a in enumerate(ids):
            for gid_b in ids[i + 1 :]:
                d = _min_pairwise_ca_between_sets(
                    ca_coords, group_sets[gid_a], group_sets[gid_b]
                )
                assert d >= 12.0 - 1e-5, (seed, gid_a, gid_b, d)
        assert mask.dtype == torch.bool
        assert int(mask.sum()) == int((groups > 0).sum())


def test_structural_distal_seq_gap():
    struct = _make_two_strand_sheet(40, strand_spacing=8.0)
    resolved = struct.residues["is_present"]
    task = DistalSeparatedMotifScaffolding(
        max_groups=1,
        min_center_dist=1.0,
        min_inter_group_ca_dist=1.0,
        p_catalytic=0.0,
        max_structural_fragments=2,
        min_seq_gap=30,
        structural_anchor_max_ca_dist=12.0,
        min_structural_res=6,
        max_structural_res=12,
        min_fragment_size=3,
        multi_motif_se3=True,
    )
    found_multi_fragment = False
    for seed in range(200):
        # Force center near middle of first strand so second-strand distal anchors exist.
        rng = np.random.RandomState(seed)
        mask, groups = task.generate_motif_mask_and_groups(
            int(resolved.sum()),
            np_rng=rng,
            data=struct,
            resolved_mask=resolved,
        )
        motif = _motif_global_indices(resolved, mask.numpy())
        if motif.size == 0:
            continue
        # Contiguous runs in sequence index.
        runs = []
        start = prev = int(motif[0])
        for idx in motif[1:]:
            idx = int(idx)
            if idx == prev + 1:
                prev = idx
                continue
            runs.append((start, prev))
            start = prev = idx
        runs.append((start, prev))
        if len(runs) < 2:
            continue
        found_multi_fragment = True
        mids = [0.5 * (a0 + a1) for a0, a1 in runs]
        for i, mi in enumerate(mids):
            for mj in mids[i + 1 :]:
                # Anchors are constrained to seq_gap; windows may grow a few residues
                # toward each other, so compare run midpoints (proxy for anchors).
                assert abs(mi - mj) >= 30 - 1e-6, (seed, runs, mi, mj)
        break
    assert found_multi_fragment, "expected at least one structural multi-fragment sample"


def test_catalytic_size_and_same_chain():
    struct = _make_alanine_chain(80)
    resolved = struct.residues["is_present"]
    task = DistalSeparatedMotifScaffolding(
        max_groups=1,
        p_catalytic=1.0,
        catalytic_min_res=3,
        catalytic_max_res=5,
        min_center_dist=1.0,
        min_inter_group_ca_dist=1.0,
        multi_motif_se3=True,
    )
    for seed in range(30):
        mask, groups = task.generate_motif_mask_and_groups(
            int(resolved.sum()),
            np_rng=np.random.RandomState(seed),
            data=struct,
            resolved_mask=resolved,
        )
        motif = _motif_global_indices(resolved, mask.numpy())
        if motif.size == 0:
            continue
        assert 3 <= motif.size <= 5
        asyms = {asym_id_for_residue(struct, int(i)) for i in motif}
        assert len(asyms) == 1
        assert set(np.unique(groups[motif]).tolist()) == {1}


def test_sample_t_preserves_generated_groups():
    struct = _make_spatial_grid(8, spacing=6.0)
    task = DistalSeparatedMotifScaffolding(
        max_groups=3,
        min_center_dist=18.0,
        min_inter_group_ca_dist=12.0,
        p_catalytic=0.5,
        multi_motif_se3=True,
        motif_t_lock=0.5,
    )
    out = task.sample_t_and_mask(
        struct,
        np_rng=np.random.RandomState(7),
        torch_generator=torch.Generator().manual_seed(7),
    )
    groups = out["motif_group_id_residue"]
    assert out["multi_motif_se3"] is True
    assert out["motif_t_lock"] == 0.5
    assert groups.dtype == np.int32
    # Motif residues (non-noised copy masks) should carry nonzero group ids when present.
    motif_like = out["copy_indexed_residue_mask"] | out["copy_unindexed_residue_mask"]
    if motif_like.any():
        assert (groups[motif_like] > 0).all()


def test_deterministic_with_fixed_rng():
    struct = _make_spatial_grid(8, spacing=6.0)
    resolved = struct.residues["is_present"]
    task = DistalSeparatedMotifScaffolding(
        max_groups=3,
        min_center_dist=18.0,
        min_inter_group_ca_dist=12.0,
        p_catalytic=0.4,
        multi_motif_se3=True,
    )
    mask_a, groups_a = task.generate_motif_mask_and_groups(
        int(resolved.sum()),
        np_rng=np.random.RandomState(123),
        data=struct,
        resolved_mask=resolved,
    )
    mask_b, groups_b = task.generate_motif_mask_and_groups(
        int(resolved.sum()),
        np_rng=np.random.RandomState(123),
        data=struct,
        resolved_mask=resolved,
    )
    assert torch.equal(mask_a, mask_b)
    assert np.array_equal(groups_a, groups_b)


def test_prepare_keeps_all_sampled_distal_groups(monkeypatch):
    struct = _as_structure(_make_spatial_grid(8, spacing=6.0))
    task = DistalSeparatedMotifScaffolding(
        min_groups=2,
        max_groups=2,
        min_center_dist=12.0,
        min_inter_group_ca_dist=6.0,
        p_catalytic=1.0,
        catalytic_min_res=2,
        catalytic_max_res=2,
        small_fragment_shell_center_dist=6.0,
        small_fragment_shell_half_width=2.0,
        p_is_unindexed=1.0,
        p_fix_tip=0.0,
        p_fix_bb=0.0,
        p_fix_all=1.0,
        multi_motif_se3=True,
    )
    monkeypatch.setattr(dm, "load_input", lambda record, data_dir: struct)

    prepared = dm.prepare_training_tokenized_example(
        dataset=SimpleNamespace(data_dir="unused"),
        sample=SimpleNamespace(
            record=SimpleNamespace(id="distal"),
            chain_id=None,
            interface_id=None,
        ),
        task=task,
        cropper=Cropper(min_neighborhood=0, max_neighborhood=0),
        max_crop_residues=32,
        remove_mol_types=[],
        mask_nonstandard=False,
        np_rng=np.random.RandomState(11),
        torch_generator=torch.Generator().manual_seed(11),
    )

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

    assert {group_id for _, group_id in expected} == {1, 2}
    assert actual == expected
    assert set(marker_tokens["motif_group_id"].tolist()) == {1, 2}
    assert len(prepared.tokenized.tokens) <= 32
