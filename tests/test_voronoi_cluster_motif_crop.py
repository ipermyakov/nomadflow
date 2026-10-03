"""Unit and prepare tests for Voronoi cluster motif cropping."""

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
from nomadflow.runtime.training.motif_scaffold import (
    VoronoiClusterMotifScaffolding,
)
from nomadflow.runtime.training.voronoi_motif import (
    PotentialMotif,
    VoronoiCluster,
    VoronoiPartition,
    _sequence_fragments,
    order_clusters_near_query,
    partition_voronoi_clusters,
    sample_catalytic_potential,
    sample_potential_motifs,
    sample_structural_potential,
    select_cluster_atomic_crop,
)


def _make_structure_from_ca_coords(ca_coords: np.ndarray):
    num_residues = int(ca_coords.shape[0])
    atom_names = [name for name in rc.restype_name_to_atom14_names["ALA"] if name]
    atoms_per_residue = len(atom_names)
    atoms = np.zeros(
        num_residues * atoms_per_residue,
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
    for residue_index in range(num_residues):
        atom_start = int(residues["atom_idx"][residue_index])
        for local_index, atom_name in enumerate(atom_names):
            atom_index = atom_start + local_index
            atoms["name"][atom_index] = convert_atom_str_to_tuple(atom_name)
            atoms["coords"][atom_index] = ca_coords[residue_index]
        atoms["coords"][atom_start + ca_index] = ca_coords[residue_index]
    chains = np.array(
        [
            (
                "A",
                const.chain_type_ids["PROTEIN"],
                0,
                0,
                0,
                0,
                len(atoms),
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
    return SimpleNamespace(
        residues=residues,
        chains=chains,
        atoms=atoms,
        mask=np.array([True]),
    )


def _make_alanine_chain(num_residues: int, spacing: float = 3.8):
    coords = np.stack(
        [
            spacing * np.arange(num_residues, dtype=np.float32),
            np.zeros(num_residues, dtype=np.float32),
            np.zeros(num_residues, dtype=np.float32),
        ],
        axis=1,
    )
    return _make_structure_from_ca_coords(coords)


def _make_spatial_grid(side: int, spacing: float = 6.0):
    coords = np.asarray(
        [
            [spacing * column, spacing * row, 0.0]
            for row in range(side)
            for column in range(side)
        ],
        dtype=np.float32,
    )
    return _make_structure_from_ca_coords(coords)


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


def _split_into_two_chains(struct, split: int):
    chains = np.zeros(2, dtype=struct.chains.dtype)
    chains[0] = struct.chains[0]
    chains[1] = struct.chains[0]
    atoms_per_residue = int(struct.residues[0]["atom_num"])
    chains[0]["name"] = "A"
    chains[0]["asym_id"] = 0
    chains[0]["atom_num"] = split * atoms_per_residue
    chains[0]["res_num"] = split
    chains[1]["name"] = "B"
    chains[1]["asym_id"] = 1
    chains[1]["entity_id"] = 1
    chains[1]["atom_idx"] = split * atoms_per_residue
    chains[1]["atom_num"] = (len(struct.residues) - split) * atoms_per_residue
    chains[1]["res_idx"] = split
    chains[1]["res_num"] = len(struct.residues) - split
    struct.chains = chains
    struct.mask = np.array([True, True])
    return struct


def _partition(struct, seed=7, target=16, maximum=20):
    return partition_voronoi_clusters(
        struct,
        cluster_target_res=target,
        cluster_max_res=maximum,
        np_rng=np.random.RandomState(seed),
    )


def _potentials(partition, seed=8, p_catalytic=0.0, keep=0.5, min_segments=2):
    return sample_potential_motifs(
        partition,
        p_catalytic=p_catalytic,
        catalytic_min_res=3,
        catalytic_max_res=5,
        catalytic_max_ca_dist=12.0,
        catalytic_distance_temperature=2.5,
        catalytic_min_cluster_segments=min_segments,
        structural_keep_fraction_min=keep,
        structural_keep_fraction_max=keep,
        np_rng=np.random.RandomState(seed),
    )


def _cluster_segment_count(cluster):
    return len(_sequence_fragments(cluster.residue_indices))


def _compact_four_segment_cluster():
    residue_indices = np.asarray([0, 1, 10, 11, 20, 21, 30, 31], dtype=np.int32)
    ca_coords = np.zeros((32, 3), dtype=np.float32)
    ca_coords[residue_indices] = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.5, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.5, 3.0, 0.0],
            [3.0, 3.0, 0.0],
            [3.5, 3.0, 0.0],
        ],
        dtype=np.float32,
    )
    cluster = VoronoiCluster(
        cluster_id=0,
        asym_id=0,
        center_residue=0,
        residue_indices=residue_indices,
        center_coord=ca_coords[0].copy(),
        chain_center_coords=ca_coords[[0]].copy(),
        owner_center_position=0,
    )
    return cluster, ca_coords


def test_voronoi_partition_is_deterministic_bounded_and_complete():
    struct = _make_spatial_grid(10, spacing=4.0)
    first = _partition(struct, seed=13)
    second = _partition(struct, seed=13)

    assert np.array_equal(first.residue_cluster_id, second.residue_cluster_id)
    assert [cluster.center_residue for cluster in first.clusters] == [
        cluster.center_residue for cluster in second.clusters
    ]
    assigned = first.residue_cluster_id[first.eligible_residue_indices]
    assert np.all(assigned >= 0)
    assert np.unique(first.eligible_residue_indices).size == 100
    assert max(cluster.residue_indices.size for cluster in first.clusters) <= 20
    assert sum(cluster.residue_indices.size for cluster in first.clusters) == 100


def test_voronoi_partition_never_mixes_chains():
    struct = _split_into_two_chains(_make_alanine_chain(48), split=24)
    partition = _partition(struct, target=16, maximum=20)

    for cluster in partition.clusters:
        expected_chain = 0 if cluster.residue_indices[0] < 24 else 1
        assert cluster.asym_id == expected_chain
        assert np.all((cluster.residue_indices < 24) == (expected_chain == 0))


def test_structural_erosion_returns_exact_target_and_single_center_core():
    struct = _make_alanine_chain(15)
    partition = _partition(struct, target=32, maximum=40)
    cluster = partition.clusters[0]
    potential = sample_structural_potential(
        cluster,
        partition.ca_coords,
        keep_fraction_min=0.6,
        keep_fraction_max=0.6,
        np_rng=np.random.RandomState(4),
    )

    assert potential.kind == "structural"
    assert potential.residue_indices.size == round(0.6 * 15)
    distances = np.linalg.norm(
        partition.ca_coords[cluster.residue_indices] - cluster.center_coord,
        axis=1,
    )
    cutoff = np.partition(distances, 8)[8]
    selected_distances = np.linalg.norm(
        partition.ca_coords[potential.residue_indices] - cluster.center_coord,
        axis=1,
    )
    assert np.all(selected_distances <= cutoff)
    assert sum(fragment.size for fragment in potential.fragments) == 9


def test_catalytic_potential_represents_all_segments_and_is_compact():
    cluster, ca_coords = _compact_four_segment_cluster()
    potential = sample_catalytic_potential(
        cluster,
        ca_coords,
        min_res=5,
        max_res=5,
        max_ca_dist=12.0,
        distance_temperature=2.5,
        np_rng=np.random.RandomState(2),
    )
    repeated = sample_catalytic_potential(
        cluster,
        ca_coords,
        min_res=5,
        max_res=5,
        max_ca_dist=12.0,
        distance_temperature=2.5,
        np_rng=np.random.RandomState(2),
    )

    assert potential is not None
    assert repeated is not None
    assert np.array_equal(potential.residue_indices, repeated.residue_indices)
    assert potential.residue_indices.size == 5
    assert set(potential.residue_indices).issubset(set(cluster.residue_indices))
    full_segments = _sequence_fragments(cluster.residue_indices)
    segment_occupancy = [
        int(np.isin(segment, potential.residue_indices).sum())
        for segment in full_segments
    ]
    assert sorted(segment_occupancy) == [1, 1, 1, 2]
    selected_coords = ca_coords[potential.residue_indices]
    pairwise_distances = np.linalg.norm(
        selected_coords[:, None, :] - selected_coords[None, :, :],
        axis=-1,
    )
    assert pairwise_distances.max() <= 2 * 12.0


def test_catalytic_potential_rejects_spatially_separated_segments():
    cluster, ca_coords = _compact_four_segment_cluster()
    potential = sample_catalytic_potential(
        cluster,
        ca_coords,
        min_res=3,
        max_res=5,
        max_ca_dist=1.0,
        distance_temperature=2.5,
        np_rng=np.random.RandomState(2),
    )

    assert potential is None


def test_catalytic_potential_rejects_single_segment_cluster():
    struct = _make_alanine_chain(4)
    partition = _partition(struct, target=32, maximum=40)
    potential = sample_catalytic_potential(
        partition.clusters[0],
        partition.ca_coords,
        min_res=3,
        max_res=5,
        max_ca_dist=12.0,
        distance_temperature=2.5,
        np_rng=np.random.RandomState(2),
    )

    assert potential is None


def test_segment_ranked_catalytic_prefers_multi_fragment_clusters():
    partition = _partition(_make_spatial_grid(8, spacing=5.0), target=16, maximum=20)
    segment_counts = {
        cluster.cluster_id: _cluster_segment_count(cluster)
        for cluster in partition.clusters
    }
    eligible_ids = {
        cluster.cluster_id
        for cluster in partition.clusters
        if cluster.residue_indices.size >= 3 and segment_counts[cluster.cluster_id] >= 2
    }
    assert eligible_ids

    potentials = _potentials(partition, seed=11, p_catalytic=1.0, min_segments=2)
    catalytic_ids = {
        cluster_id
        for cluster_id, potential in potentials.items()
        if potential.kind == "catalytic"
    }
    assert catalytic_ids
    assert catalytic_ids <= eligible_ids
    assert catalytic_ids == eligible_ids
    for cluster_id in catalytic_ids:
        assert segment_counts[cluster_id] >= 2
    for cluster_id, potential in potentials.items():
        if cluster_id not in eligible_ids:
            assert potential.kind == "structural"


def test_single_segment_clusters_never_become_catalytic_when_min_segments_is_two():
    partition = _partition(_make_alanine_chain(48), target=16, maximum=20)
    assert all(_cluster_segment_count(cluster) == 1 for cluster in partition.clusters)

    potentials = _potentials(partition, seed=9, p_catalytic=1.0, min_segments=2)
    assert potentials
    assert all(potential.kind == "structural" for potential in potentials.values())


def test_cluster_atomic_budget_and_partial_fallback_have_no_motif():
    struct = _make_spatial_grid(8, spacing=5.0)
    partition = _partition(struct, target=16, maximum=20)
    potentials = _potentials(partition, keep=0.5)
    order = order_clusters_near_query(partition, 0, np.random.RandomState(3))
    selection = select_cluster_atomic_crop(
        partition,
        potentials,
        order,
        max_crop_residues=30,
        min_motif_groups=2,
        max_motif_groups=5,
        p_motif_cluster=1.0,
        min_cluster_sequence_contacts=1,
        np_rng=np.random.RandomState(3),
    )

    assert len(selection.complete_cluster_ids) == 1
    assert len(selection.activated_cluster_ids) == 1
    assert selection.group_shortfall
    assert selection.projected_token_count <= 30
    assert selection.partial_cluster_ids
    for cluster_id in selection.partial_cluster_ids:
        residues = partition.clusters[cluster_id].residue_indices
        assert not np.any(selection.motif_group_id_residue[residues] > 0)


@pytest.mark.parametrize(
    ("p_motif_cluster", "max_groups", "expected_groups"),
    [(0.0, 5, 2), (1.0, 2, 2)],
)
def test_cluster_activation_respects_minimum_when_feasible_and_maximum_cap(
    p_motif_cluster,
    max_groups,
    expected_groups,
):
    struct = _make_spatial_grid(10, spacing=5.0)
    partition = _partition(struct, target=16, maximum=20)
    potentials = _potentials(partition, keep=0.5)
    order = order_clusters_near_query(partition, 0, np.random.RandomState(9))
    selection = select_cluster_atomic_crop(
        partition,
        potentials,
        order,
        max_crop_residues=256,
        min_motif_groups=2,
        max_motif_groups=max_groups,
        p_motif_cluster=p_motif_cluster,
        min_cluster_sequence_contacts=1,
        np_rng=np.random.RandomState(9),
    )

    assert len(selection.activated_cluster_ids) == expected_groups
    assert not selection.group_shortfall
    assert set(selection.motif_group_id_residue) == {0, 1, 2}


def test_cluster_atomic_crop_rejects_sequence_disconnected_clusters():
    ca_coords = np.zeros((8, 3), dtype=np.float32)
    # Spatially adjacent, but sequence-gapped: 0-3 then 10-13 indexed as 0..7 with
    # artificial asym separation via non-adjacent residue indices.
    ca_coords[:4] = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    ca_coords[4:] = np.asarray(
        [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    near_indices = np.asarray([0, 1, 2, 3], dtype=np.int32)
    far_indices = np.asarray([10, 11, 12, 13], dtype=np.int32)
    coords_full = np.zeros((14, 3), dtype=np.float32)
    coords_full[near_indices] = ca_coords[:4]
    coords_full[far_indices] = ca_coords[4:]
    near = VoronoiCluster(
        cluster_id=0,
        asym_id=0,
        center_residue=0,
        residue_indices=near_indices,
        center_coord=coords_full[0].copy(),
        chain_center_coords=coords_full[[0, 10]].copy(),
        owner_center_position=0,
    )
    far = VoronoiCluster(
        cluster_id=1,
        asym_id=0,
        center_residue=10,
        residue_indices=far_indices,
        center_coord=coords_full[10].copy(),
        chain_center_coords=coords_full[[0, 10]].copy(),
        owner_center_position=1,
    )
    residue_cluster_id = np.full(14, -1, dtype=np.int32)
    residue_cluster_id[near_indices] = 0
    residue_cluster_id[far_indices] = 1
    partition = VoronoiPartition(
        clusters=(near, far),
        eligible_residue_indices=np.concatenate([near_indices, far_indices]),
        residue_cluster_id=residue_cluster_id,
        ca_coords=coords_full,
    )
    potentials = {
        0: PotentialMotif(
            cluster_id=0,
            kind="structural",
            residue_indices=near_indices.copy(),
            fragments=(near_indices.copy(),),
        ),
        1: PotentialMotif(
            cluster_id=1,
            kind="structural",
            residue_indices=far_indices.copy(),
            fragments=(far_indices.copy(),),
        ),
    }
    selection = select_cluster_atomic_crop(
        partition,
        potentials,
        (0, 1),
        max_crop_residues=256,
        min_motif_groups=1,
        max_motif_groups=2,
        p_motif_cluster=1.0,
        min_cluster_sequence_contacts=1,
        np_rng=np.random.RandomState(0),
    )

    assert selection.complete_cluster_ids == (0,)
    assert selection.partial_cluster_ids == ()
    assert set(selection.scaffold_residue_indices.tolist()) == {0, 1, 2, 3}


def test_cluster_atomic_crop_accepts_sequence_adjacent_cluster():
    coords = np.zeros((8, 3), dtype=np.float32)
    coords[:, 0] = np.arange(8, dtype=np.float32)
    left = VoronoiCluster(
        cluster_id=0,
        asym_id=0,
        center_residue=1,
        residue_indices=np.arange(4, dtype=np.int32),
        center_coord=coords[1].copy(),
        chain_center_coords=coords[[1, 5]].copy(),
        owner_center_position=0,
    )
    right = VoronoiCluster(
        cluster_id=1,
        asym_id=0,
        center_residue=5,
        residue_indices=np.arange(4, 8, dtype=np.int32),
        center_coord=coords[5].copy(),
        chain_center_coords=coords[[1, 5]].copy(),
        owner_center_position=1,
    )
    partition = VoronoiPartition(
        clusters=(left, right),
        eligible_residue_indices=np.arange(8, dtype=np.int32),
        residue_cluster_id=np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32),
        ca_coords=coords,
    )
    potentials = {
        0: PotentialMotif(
            cluster_id=0,
            kind="structural",
            residue_indices=left.residue_indices.copy(),
            fragments=(left.residue_indices.copy(),),
        ),
        1: PotentialMotif(
            cluster_id=1,
            kind="structural",
            residue_indices=right.residue_indices.copy(),
            fragments=(right.residue_indices.copy(),),
        ),
    }
    selection = select_cluster_atomic_crop(
        partition,
        potentials,
        (0, 1),
        max_crop_residues=256,
        min_motif_groups=1,
        max_motif_groups=2,
        p_motif_cluster=1.0,
        min_cluster_sequence_contacts=1,
        np_rng=np.random.RandomState(0),
    )

    assert selection.complete_cluster_ids == (0, 1)
    assert set(selection.scaffold_residue_indices.tolist()) == set(range(8))


def _prepare(monkeypatch, struct, task, seed, max_crop_residues):
    struct = _as_structure(struct)
    monkeypatch.setattr(dm, "load_input", lambda record, data_dir: struct)
    return dm.prepare_training_tokenized_example(
        dataset=SimpleNamespace(data_dir="unused"),
        sample=SimpleNamespace(
            record=SimpleNamespace(id="voronoi"),
            chain_id=None,
            interface_id=None,
        ),
        task=task,
        cropper=Cropper(min_neighborhood=0, max_neighborhood=0),
        max_crop_residues=max_crop_residues,
        remove_mol_types=[],
        mask_nonstandard=False,
        np_rng=np.random.RandomState(seed),
        torch_generator=torch.Generator().manual_seed(seed),
    )


@pytest.mark.parametrize("p_catalytic", [0.0, 1.0])
def test_voronoi_prepare_preserves_copies_groups_aggregators_and_seeds(
    monkeypatch,
    p_catalytic,
):
    task = VoronoiClusterMotifScaffolding(
        cluster_target_res=16,
        cluster_max_res=20,
        min_motif_groups=2,
        max_motif_groups=3,
        p_motif_cluster=0.7,
        p_catalytic=p_catalytic,
        catalytic_min_res=3,
        catalytic_max_res=5,
        structural_keep_fraction_min=0.5,
        structural_keep_fraction_max=0.75,
    )
    struct = _make_spatial_grid(8, spacing=5.0)
    first = _prepare(monkeypatch, struct, task, 23, 72)
    second = _prepare(monkeypatch, struct, task, 23, 72)

    assert np.array_equal(first.task_data["motif_group_id_residue"], second.task_data["motif_group_id_residue"])
    assert np.array_equal(first.tokenized.tokens, second.tokenized.tokens)
    assert np.array_equal(first.tokenized.rigids, second.tokenized.rigids)
    groups = first.task_data["motif_group_id_residue"]
    expected = {
        (int(index), int(groups[index]))
        for index in np.flatnonzero(groups > 0)
    }
    copies = first.tokenized.tokens[first.tokenized.tokens["is_copy"]]
    actual = {
        (int(token["res_idx"]), int(token["motif_group_id"]))
        for token in copies
    }
    aggregators = first.tokenized.tokens[
        first.tokenized.tokens["is_motif_group_aggregator"]
    ]
    assert actual == expected
    assert set(aggregators["motif_group_id"]) == {group for _, group in expected}
    assert len(first.tokenized.tokens) <= 72


def test_voronoi_prepare_marks_soft_minimum_shortfall(monkeypatch):
    task = VoronoiClusterMotifScaffolding(
        cluster_target_res=32,
        cluster_max_res=40,
        min_motif_groups=2,
        max_motif_groups=5,
        p_motif_cluster=1.0,
    )
    prepared = _prepare(monkeypatch, _make_alanine_chain(12), task, 5, 64)

    assert prepared.task_data["voronoi_group_shortfall"] is True
    assert prepared.task_data["voronoi_activated_group_count"] == 1
    assert len(prepared.tokenized.tokens) <= 64


def _make_multichain_with_overlapping_local_res_idx(
    residues_per_chain: int = 32,
    *,
    with_ligand: bool = True,
):
    """Two protein chains both numbered from local res_idx=0, optional ligand@0."""
    protein = _make_alanine_chain(residues_per_chain * 2, spacing=3.8)
    # Shift chain B coords away so Voronoi/FPS still works spatially.
    mid = residues_per_chain
    atoms_per = int(protein.residues[0]["atom_num"])
    b_atom_start = mid * atoms_per
    protein.atoms["coords"][b_atom_start:] += np.asarray([0.0, 40.0, 0.0], dtype=np.float32)
    protein = _split_into_two_chains(protein, mid)
    # Overlapping local residue numbers (PDB-style), not global arange.
    protein.residues["res_idx"][:mid] = np.arange(mid, dtype=np.int32)
    protein.residues["res_idx"][mid:] = np.arange(mid, dtype=np.int32)

    if not with_ligand:
        return protein

    # Append a small nonpolymer ligand with local res_idx=0 (collides with both chains).
    ligand_atom_names = ["C1", "C2", "C3", "O1", "N1"]
    n_lig_atoms = len(ligand_atom_names)
    old_n_atoms = len(protein.atoms)
    atoms = np.zeros(
        old_n_atoms + n_lig_atoms,
        dtype=protein.atoms.dtype,
    )
    atoms[:old_n_atoms] = protein.atoms
    for i, name in enumerate(ligand_atom_names):
        atoms["name"][old_n_atoms + i] = convert_atom_str_to_tuple(name)
        atoms["coords"][old_n_atoms + i] = np.asarray(
            [100.0 + i, 100.0, 0.0],
            dtype=np.float32,
        )
        atoms["is_present"][old_n_atoms + i] = True

    old_n_res = len(protein.residues)
    residues = np.zeros(old_n_res + 1, dtype=protein.residues.dtype)
    residues[:old_n_res] = protein.residues
    residues[old_n_res]["name"] = "LIG"
    residues[old_n_res]["res_type"] = 0
    residues[old_n_res]["res_idx"] = 0  # collides with protein local 0
    residues[old_n_res]["atom_idx"] = old_n_atoms
    residues[old_n_res]["atom_num"] = n_lig_atoms
    residues[old_n_res]["atom_center"] = 0
    residues[old_n_res]["atom_disto"] = 0
    residues[old_n_res]["is_standard"] = False
    residues[old_n_res]["is_present"] = True

    chains = np.zeros(3, dtype=protein.chains.dtype)
    chains[:2] = protein.chains
    chains[2]["name"] = "L"
    chains[2]["mol_type"] = const.chain_type_ids["NONPOLYMER"]
    chains[2]["entity_id"] = 2
    chains[2]["sym_id"] = 0
    chains[2]["asym_id"] = 2
    chains[2]["atom_idx"] = old_n_atoms
    chains[2]["atom_num"] = n_lig_atoms
    chains[2]["res_idx"] = old_n_res
    chains[2]["res_num"] = 1
    chains[2]["cyclic_period"] = 0

    return SimpleNamespace(
        residues=residues,
        chains=chains,
        atoms=atoms,
        mask=np.array([True, True, True]),
    )


def test_mask_structure_uses_global_array_indices_not_local_res_idx():
    struct = _as_structure(_make_multichain_with_overlapping_local_res_idx(8))
    # Keep only chain-B residue at array index 10 (local res_idx=2 on chain B).
    # Under the old bug, local res_idx∈{2} would also keep chain-A residue 2 and ligand 0
    # if 0 were in the keep set; here we keep {10} and assert ligand+chainA stay cleared.
    keep = np.asarray([10, 11, 12], dtype=np.int32)
    cropped = dm._mask_structure_to_residue_indices(struct, keep)
    present = np.flatnonzero(cropped.residues["is_present"]).tolist()
    assert present == [10, 11, 12]
    # Ligand is last residue; must not stay present via local res_idx collision.
    assert not bool(cropped.residues[-1]["is_present"])
    # Chain A residue with local res_idx=2 (array 2) must be cleared.
    assert not bool(cropped.residues[2]["is_present"])


def test_voronoi_prepare_multichain_overlapping_local_res_idx(monkeypatch):
    task = VoronoiClusterMotifScaffolding(
        cluster_target_res=16,
        cluster_max_res=20,
        min_motif_groups=1,
        max_motif_groups=4,
        p_motif_cluster=0.8,
        p_catalytic=0.35,
        catalytic_min_res=3,
        catalytic_max_res=5,
        structural_keep_fraction_min=0.5,
        structural_keep_fraction_max=0.75,
    )
    # Ligand collision covered by test_mask_*; prepare path needs full atom schema.
    struct = _make_multichain_with_overlapping_local_res_idx(32, with_ligand=False)
    max_crop = 96
    prepared = _prepare(monkeypatch, struct, task, 11, max_crop)

    groups = prepared.task_data["motif_group_id_residue"]
    activated = set(np.flatnonzero(groups > 0).tolist())
    copy_mask = (
        prepared.task_data["copy_indexed_residue_mask"]
        | prepared.task_data["copy_unindexed_residue_mask"]
    )
    assert set(np.flatnonzero(copy_mask).tolist()) == activated

    residue_map = dm._residue_array_index_map(prepared.tokenized.structure)
    copy_tokens = prepared.tokenized.tokens[prepared.tokenized.tokens["is_copy"]]
    copy_array_idxs = {
        residue_map[(int(t["asym_id"]), int(t["res_idx"]))]
        for t in copy_tokens
    }
    assert copy_array_idxs == activated
    # After aggregators are appended, total tokens must fit the crop budget.
    assert len(prepared.tokenized.tokens) <= max_crop


def test_voronoi_prepare_tight_budget_max_groups(monkeypatch):
    task = VoronoiClusterMotifScaffolding(
        cluster_target_res=16,
        cluster_max_res=20,
        min_motif_groups=2,
        max_motif_groups=10,
        p_motif_cluster=1.0,
        p_catalytic=0.5,
        catalytic_min_res=3,
        catalytic_max_res=5,
        structural_keep_fraction_min=0.4,
        structural_keep_fraction_max=0.75,
    )
    struct = _make_spatial_grid(10, spacing=5.0)
    max_crop = 48
    for seed in range(20):
        prepared = _prepare(monkeypatch, struct, task, seed, max_crop)
        groups = prepared.task_data["motif_group_id_residue"]
        # Full prepare includes aggregators in tokenized.tokens.
        assert len(prepared.tokenized.tokens) <= max_crop
        activated = set(np.flatnonzero(groups > 0).tolist())
        copy_mask = (
            prepared.task_data["copy_indexed_residue_mask"]
            | prepared.task_data["copy_unindexed_residue_mask"]
        )
        assert set(np.flatnonzero(copy_mask).tolist()) == activated
