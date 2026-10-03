"""Voronoi-cluster motif construction and cluster-atomic crop selection."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from nomadflow.boltz.data import const

from .motif_partition import build_residue_spatial_lookup, farthest_point_centers


@dataclass(frozen=True)
class VoronoiCluster:
    cluster_id: int
    asym_id: int
    center_residue: int
    residue_indices: np.ndarray
    center_coord: np.ndarray
    chain_center_coords: np.ndarray
    owner_center_position: int


@dataclass(frozen=True)
class PotentialMotif:
    cluster_id: int
    kind: str
    residue_indices: np.ndarray
    fragments: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class VoronoiPartition:
    clusters: tuple[VoronoiCluster, ...]
    eligible_residue_indices: np.ndarray
    residue_cluster_id: np.ndarray
    ca_coords: np.ndarray


@dataclass(frozen=True)
class VoronoiCropSelection:
    scaffold_residue_indices: np.ndarray
    motif_group_id_residue: np.ndarray
    complete_cluster_ids: tuple[int, ...]
    activated_cluster_ids: tuple[int, ...]
    partial_cluster_ids: tuple[int, ...]
    copy_count: int
    projected_token_count: int
    group_shortfall: bool


def eligible_protein_residue_indices(data) -> np.ndarray:
    """Resolved standard protein residues with valid CA coordinates."""
    _, ca_coords = build_residue_spatial_lookup(data)
    protein_type = const.chain_type_ids["PROTEIN"]
    eligible: list[int] = []
    for chain in data.chains[data.mask]:
        if int(chain["mol_type"]) != int(protein_type):
            continue
        start = int(chain["res_idx"])
        stop = start + int(chain["res_num"])
        for residue_index in range(start, stop):
            residue = data.residues[residue_index]
            if (
                bool(residue["is_present"])
                and bool(residue["is_standard"])
                and residue["name"] != "UNK"
                and not np.isnan(ca_coords[residue_index]).any()
            ):
                eligible.append(residue_index)
    return np.asarray(eligible, dtype=np.int32)


def _capacity_limited_assignment(
    residue_indices: np.ndarray,
    center_indices: np.ndarray,
    ca_coords: np.ndarray,
    max_cluster_residues: int,
    np_rng,
) -> np.ndarray:
    """Assign every residue to a nearby center without exceeding capacity."""
    residue_coords = ca_coords[residue_indices]
    center_coords = ca_coords[center_indices]
    distances = np.linalg.norm(
        residue_coords[:, None, :] - center_coords[None, :, :],
        axis=-1,
    )
    assignment = np.full(residue_indices.size, -1, dtype=np.int32)
    capacity = np.full(center_indices.size, int(max_cluster_residues), dtype=np.int32)

    residue_position = {int(residue): pos for pos, residue in enumerate(residue_indices)}
    for center_position, center_residue in enumerate(center_indices):
        pos = residue_position[int(center_residue)]
        assignment[pos] = center_position
        capacity[center_position] -= 1

    unassigned = np.flatnonzero(assignment < 0)
    if unassigned.size:
        residue_pos = np.repeat(unassigned, center_indices.size)
        center_pos = np.tile(np.arange(center_indices.size, dtype=np.int32), unassigned.size)
        pair_distances = distances[residue_pos, center_pos]
        tie_order = np.asarray(np_rng.permutation(pair_distances.size), dtype=np.int64)
        order = np.lexsort((tie_order, pair_distances))
        for pair_index in order:
            res_pos = int(residue_pos[pair_index])
            ctr_pos = int(center_pos[pair_index])
            if assignment[res_pos] >= 0 or capacity[ctr_pos] <= 0:
                continue
            assignment[res_pos] = ctr_pos
            capacity[ctr_pos] -= 1

    if np.any(assignment < 0):
        raise RuntimeError("Capacity-limited Voronoi assignment left residues unassigned")
    return assignment


def partition_voronoi_clusters(
    data,
    *,
    cluster_target_res: int,
    cluster_max_res: int,
    np_rng,
) -> VoronoiPartition:
    """Partition eligible residues independently per chain."""
    cluster_target_res = int(cluster_target_res)
    cluster_max_res = int(cluster_max_res)
    if cluster_target_res < 1:
        raise ValueError("cluster_target_res must be >= 1")
    if cluster_max_res < cluster_target_res:
        raise ValueError("cluster_max_res must be >= cluster_target_res")

    asym_ids, ca_coords = build_residue_spatial_lookup(data)
    eligible = eligible_protein_residue_indices(data)
    residue_cluster_id = np.full(len(data.residues), -1, dtype=np.int32)
    clusters: list[VoronoiCluster] = []

    for asym_id in np.unique(asym_ids[eligible]) if eligible.size else []:
        chain_residues = eligible[asym_ids[eligible] == int(asym_id)]
        chain_residues = np.sort(chain_residues)
        center_count = int(math.ceil(chain_residues.size / cluster_target_res))
        centers = farthest_point_centers(
            chain_residues,
            ca_coords,
            center_count,
            np_rng=np_rng,
        )
        assignment = _capacity_limited_assignment(
            chain_residues,
            centers,
            ca_coords,
            cluster_max_res,
            np_rng,
        )
        center_coords = ca_coords[centers].copy()
        for owner_position, center_residue in enumerate(centers):
            members = np.sort(chain_residues[assignment == owner_position]).astype(np.int32)
            cluster_id = len(clusters)
            residue_cluster_id[members] = cluster_id
            clusters.append(
                VoronoiCluster(
                    cluster_id=cluster_id,
                    asym_id=int(asym_id),
                    center_residue=int(center_residue),
                    residue_indices=members,
                    center_coord=ca_coords[int(center_residue)].copy(),
                    chain_center_coords=center_coords,
                    owner_center_position=int(owner_position),
                )
            )

    if eligible.size:
        assigned = residue_cluster_id[eligible]
        if np.any(assigned < 0):
            raise RuntimeError("Voronoi partition does not cover every eligible residue")
        sizes = np.bincount(assigned, minlength=len(clusters))
        if np.any(sizes > cluster_max_res):
            raise RuntimeError("Voronoi partition exceeded cluster_max_res")

    return VoronoiPartition(
        clusters=tuple(clusters),
        eligible_residue_indices=eligible,
        residue_cluster_id=residue_cluster_id,
        ca_coords=ca_coords,
    )


def _sequence_fragments(residue_indices: np.ndarray) -> tuple[np.ndarray, ...]:
    ordered = np.sort(np.asarray(residue_indices, dtype=np.int32))
    if ordered.size == 0:
        return ()
    split_points = np.flatnonzero(np.diff(ordered) != 1) + 1
    return tuple(part.astype(np.int32, copy=False) for part in np.split(ordered, split_points))


def _cluster_segment_ids(cluster: VoronoiCluster) -> np.ndarray:
    """Return full-cell sequence-segment ids aligned with cluster residues."""
    residues = cluster.residue_indices
    if residues.size == 0:
        return np.empty(0, dtype=np.int32)
    return np.cumsum(
        np.concatenate(
            [
                np.zeros(1, dtype=np.int32),
                (np.diff(residues) != 1).astype(np.int32),
            ]
        )
    )


def _catalytic_anchor_options(
    cluster: VoronoiCluster,
    ca_coords: np.ndarray,
    *,
    target_res: int,
    max_ca_dist: float,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, float]]]:
    """Find anchors that can realize a target-sized, segment-diverse site."""
    residues = cluster.residue_indices
    segment_ids = _cluster_segment_ids(cluster)
    coords = ca_coords[residues]
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    segment_count = int(segment_ids.max()) + 1 if segment_ids.size else 0
    represented_target = min(segment_count, int(target_res))
    options: list[tuple[int, float]] = []

    for anchor_pos in range(residues.size):
        pool = np.flatnonzero(
            (np.arange(residues.size) != anchor_pos)
            & (distances[anchor_pos] <= float(max_ca_dist))
        )
        if pool.size < int(target_res) - 1:
            continue
        available_segments = np.unique(
            np.concatenate([segment_ids[[anchor_pos]], segment_ids[pool]])
        )
        if available_segments.size < represented_target:
            continue

        residue_radius = float(
            np.partition(distances[anchor_pos, pool], int(target_res) - 2)[
                int(target_res) - 2
            ]
        )
        other_segment_radii = [
            float(distances[anchor_pos, pool[segment_ids[pool] == segment_id]].min())
            for segment_id in available_segments
            if int(segment_id) != int(segment_ids[anchor_pos])
        ]
        required_other_segments = represented_target - 1
        if required_other_segments:
            segment_radius = float(
                np.partition(
                    np.asarray(other_segment_radii),
                    required_other_segments - 1,
                )[required_other_segments - 1]
            )
        else:
            segment_radius = 0.0
        options.append((anchor_pos, max(residue_radius, segment_radius)))
    return distances, segment_ids, options


def _sample_soft_index(
    indices: np.ndarray,
    distances: np.ndarray,
    temperature: float,
    np_rng,
) -> int:
    """Sample one index with an exponentially decaying distance preference."""
    logits = -np.asarray(distances, dtype=np.float64) / float(temperature)
    logits -= logits.max()
    weights = np.exp(logits)
    probabilities = weights / weights.sum()
    return int(np_rng.choice(indices, p=probabilities))


def sample_catalytic_potential(
    cluster: VoronoiCluster,
    ca_coords: np.ndarray,
    *,
    min_res: int,
    max_res: int,
    max_ca_dist: float,
    distance_temperature: float,
    np_rng,
) -> PotentialMotif | None:
    """Sample a compact site representing as many cluster segments as possible."""
    if len(_sequence_fragments(cluster.residue_indices)) < 2:
        return None
    feasible_targets: list[
        tuple[int, np.ndarray, np.ndarray, list[tuple[int, float]]]
    ] = []
    max_target_res = min(int(max_res), int(cluster.residue_indices.size))
    for target_res in range(int(min_res), max_target_res + 1):
        distances, segment_ids, options = _catalytic_anchor_options(
            cluster,
            ca_coords,
            target_res=target_res,
            max_ca_dist=max_ca_dist,
        )
        if options:
            feasible_targets.append((target_res, distances, segment_ids, options))
    if not feasible_targets:
        return None

    target_choice = int(np_rng.choice(len(feasible_targets)))
    target, distances, segment_ids, anchor_options = feasible_targets[target_choice]
    anchor_positions = np.asarray(
        [anchor_pos for anchor_pos, _ in anchor_options], dtype=np.int32
    )
    anchor_radii = np.asarray(
        [radius for _, radius in anchor_options], dtype=np.float64
    )
    anchor_pos = _sample_soft_index(
        anchor_positions,
        anchor_radii,
        distance_temperature,
        np_rng,
    )
    pool = np.flatnonzero(
        (np.arange(cluster.residue_indices.size) != anchor_pos)
        & (distances[anchor_pos] <= float(max_ca_dist))
    )
    selected_positions = [anchor_pos]
    anchor_segment = int(segment_ids[anchor_pos])
    available_segments = [
        int(segment_id)
        for segment_id in np.unique(segment_ids[pool])
        if int(segment_id) != anchor_segment
    ]
    represented_target = min(int(segment_ids.max()) + 1, target)

    for _ in range(represented_target - 1):
        segment_min_distances = np.asarray(
            [
                distances[
                    anchor_pos,
                    pool[segment_ids[pool] == segment_id],
                ].min()
                for segment_id in available_segments
            ],
            dtype=np.float64,
        )
        segment_array = np.asarray(available_segments, dtype=np.int32)
        chosen_segment = _sample_soft_index(
            segment_array,
            segment_min_distances,
            distance_temperature,
            np_rng,
        )
        segment_candidates = pool[segment_ids[pool] == chosen_segment]
        chosen_position = _sample_soft_index(
            segment_candidates,
            distances[anchor_pos, segment_candidates],
            distance_temperature,
            np_rng,
        )
        selected_positions.append(chosen_position)
        available_segments.remove(chosen_segment)

    while len(selected_positions) < target:
        remaining = np.asarray(
            [position for position in pool if int(position) not in selected_positions],
            dtype=np.int32,
        )
        selected_coords = ca_coords[
            cluster.residue_indices[np.asarray(selected_positions, dtype=np.int32)]
        ]
        site_center = selected_coords.mean(axis=0)
        site_distances = np.linalg.norm(
            ca_coords[cluster.residue_indices[remaining]] - site_center[None, :],
            axis=1,
        )
        occupancy = {
            int(segment_id): int(
                np.sum(segment_ids[np.asarray(selected_positions)] == segment_id)
            )
            for segment_id in np.unique(segment_ids[np.asarray(selected_positions)])
        }
        logits = -site_distances / float(distance_temperature)
        logits -= np.log(
            np.asarray(
                [
                    1 + occupancy.get(int(segment_ids[position]), 0)
                    for position in remaining
                ],
                dtype=np.float64,
            )
        )
        logits -= logits.max()
        weights = np.exp(logits)
        chosen_position = int(np_rng.choice(remaining, p=weights / weights.sum()))
        selected_positions.append(chosen_position)

    selected = np.sort(
        cluster.residue_indices[np.asarray(selected_positions, dtype=np.int32)]
    ).astype(np.int32)
    return PotentialMotif(
        cluster_id=cluster.cluster_id,
        kind="catalytic",
        residue_indices=selected,
        fragments=_sequence_fragments(selected),
    )


def sample_structural_potential(
    cluster: VoronoiCluster,
    ca_coords: np.ndarray,
    *,
    keep_fraction_min: float,
    keep_fraction_max: float,
    np_rng,
) -> PotentialMotif:
    """Erode a Voronoi cell to an exact target count."""
    keep_fraction = float(
        np_rng.uniform(float(keep_fraction_min), float(keep_fraction_max))
    )
    cluster_size = int(cluster.residue_indices.size)
    target_res = min(cluster_size, max(1, int(round(keep_fraction * cluster_size))))
    coords = ca_coords[cluster.residue_indices]
    owner_dist = np.linalg.norm(coords - cluster.center_coord[None, :], axis=1)
    if cluster.chain_center_coords.shape[0] == 1:
        depth = -owner_dist
    else:
        other_centers = np.delete(
            cluster.chain_center_coords,
            cluster.owner_center_position,
            axis=0,
        )
        second_dist = np.linalg.norm(
            coords[:, None, :] - other_centers[None, :, :],
            axis=-1,
        ).min(axis=1)
        depth = second_dist - owner_dist
    tie_order = np.asarray(np_rng.permutation(cluster_size), dtype=np.int64)
    order = np.lexsort((tie_order, -depth))
    selected = np.sort(cluster.residue_indices[order[:target_res]]).astype(np.int32)
    return PotentialMotif(
        cluster_id=cluster.cluster_id,
        kind="structural",
        residue_indices=selected,
        fragments=_sequence_fragments(selected),
    )


def sample_potential_motifs(
    partition: VoronoiPartition,
    *,
    p_catalytic: float,
    catalytic_min_res: int,
    catalytic_max_res: int,
    catalytic_max_ca_dist: float,
    catalytic_distance_temperature: float,
    catalytic_min_cluster_segments: int,
    structural_keep_fraction_min: float,
    structural_keep_fraction_max: float,
    np_rng,
) -> dict[int, PotentialMotif]:
    """Construct exactly one feasible potential motif per non-empty cluster.

    Catalytic assignment is segment-ranked: clusters whose full Voronoi cell has
    at least ``catalytic_min_cluster_segments`` sequence fragments are eligible.
    Geometrically feasible clusters enter a binomial quota
    ``Binomial(n_eligible, p_catalytic)``. They are sorted by descending segment
    count, the top quota receives guaranteed constructive catalytic sampling,
    and all non-chosen clusters use structural erosion.
    """
    segment_counts = {
        cluster.cluster_id: len(_sequence_fragments(cluster.residue_indices))
        for cluster in partition.clusters
    }
    eligible = []
    for cluster in partition.clusters:
        if (
            cluster.residue_indices.size < int(catalytic_min_res)
            or segment_counts[cluster.cluster_id]
            < int(catalytic_min_cluster_segments)
        ):
            continue
        _, _, anchor_options = _catalytic_anchor_options(
            cluster,
            partition.ca_coords,
            target_res=catalytic_min_res,
            max_ca_dist=catalytic_max_ca_dist,
        )
        if anchor_options:
            eligible.append(cluster)
    if eligible:
        if hasattr(np_rng, "binomial"):
            n_cat = int(np_rng.binomial(len(eligible), float(p_catalytic)))
        else:
            n_cat = int(sum(float(np_rng.random()) < float(p_catalytic) for _ in eligible))
        tie_keys = {
            cluster.cluster_id: float(np_rng.random()) for cluster in eligible
        }
        eligible_ranked = sorted(
            eligible,
            key=lambda cluster: (
                -segment_counts[cluster.cluster_id],
                tie_keys[cluster.cluster_id],
            ),
        )
        catalytic_ids = {
            cluster.cluster_id for cluster in eligible_ranked[:n_cat]
        }
    else:
        catalytic_ids = set()

    potentials: dict[int, PotentialMotif] = {}
    for cluster in partition.clusters:
        potential = None
        if cluster.cluster_id in catalytic_ids:
            potential = sample_catalytic_potential(
                cluster,
                partition.ca_coords,
                min_res=catalytic_min_res,
                max_res=catalytic_max_res,
                max_ca_dist=catalytic_max_ca_dist,
                distance_temperature=catalytic_distance_temperature,
                np_rng=np_rng,
            )
        if potential is None:
            potential = sample_structural_potential(
                cluster,
                partition.ca_coords,
                keep_fraction_min=structural_keep_fraction_min,
                keep_fraction_max=structural_keep_fraction_max,
                np_rng=np_rng,
            )
        potentials[cluster.cluster_id] = potential
    return potentials


def order_clusters_near_query(
    partition: VoronoiPartition,
    query_residue_index: int,
    np_rng,
) -> tuple[int, ...]:
    """Prim-like spatial order beginning at the query-owned cluster."""
    if not partition.clusters:
        return ()
    query_cluster = int(partition.residue_cluster_id[int(query_residue_index)])
    if query_cluster < 0:
        query_coord = partition.ca_coords[int(query_residue_index)]
        center_coords = np.stack([cluster.center_coord for cluster in partition.clusters])
        query_cluster = int(np.linalg.norm(center_coords - query_coord, axis=1).argmin())

    selected = [query_cluster]
    remaining = {cluster.cluster_id for cluster in partition.clusters}
    remaining.remove(query_cluster)
    while remaining:
        selected_residues = np.concatenate(
            [partition.clusters[cluster_id].residue_indices for cluster_id in selected]
        )
        selected_coords = partition.ca_coords[selected_residues]
        candidate_ids = np.asarray(sorted(remaining), dtype=np.int32)
        distances = []
        for cluster_id in candidate_ids:
            candidate_coords = partition.ca_coords[
                partition.clusters[int(cluster_id)].residue_indices
            ]
            pairwise = np.linalg.norm(
                candidate_coords[:, None, :] - selected_coords[None, :, :],
                axis=-1,
            )
            distances.append(float(pairwise.min()))
        distances = np.asarray(distances)
        best = np.flatnonzero(np.isclose(distances, distances.min()))
        chosen_position = int(np_rng.choice(best))
        chosen = int(candidate_ids[chosen_position])
        selected.append(chosen)
        remaining.remove(chosen)
    return tuple(selected)


def _residue_asym_ids(partition: VoronoiPartition) -> np.ndarray:
    """Map each residue index to its owning cluster chain id."""
    asym_ids = np.full(partition.residue_cluster_id.shape[0], -1, dtype=np.int32)
    for cluster in partition.clusters:
        asym_ids[cluster.residue_indices] = int(cluster.asym_id)
    return asym_ids


def _sequence_boundary_residue_count(
    candidate_residue_indices: np.ndarray,
    selected: set[int],
    residue_asym_ids: np.ndarray,
) -> int:
    """Count candidate residues with a same-chain sequence neighbor in ``selected``."""
    if not selected:
        return 0
    count = 0
    for residue_index in candidate_residue_indices:
        residue_index = int(residue_index)
        asym_id = int(residue_asym_ids[residue_index])
        for neighbor in (residue_index - 1, residue_index + 1):
            if neighbor in selected and int(residue_asym_ids[neighbor]) == asym_id:
                count += 1
                break
    return count


def select_cluster_atomic_crop(
    partition: VoronoiPartition,
    potentials: dict[int, PotentialMotif],
    cluster_order: tuple[int, ...],
    *,
    max_crop_residues: int,
    min_motif_groups: int,
    max_motif_groups: int,
    p_motif_cluster: float,
    min_cluster_sequence_contacts: int,
    np_rng,
) -> VoronoiCropSelection:
    """Select sequence-boundary-connected complete clusters, then fill residual.

    After the first cluster, a later complete cluster is accepted only when at
    least ``min_cluster_sequence_contacts`` of its residues are same-chain
    sequence neighbors of the already selected scaffold. Non-touching Prim
    candidates are skipped. The first candidate that fits the sequence-contact
    rule but exceeds the token budget ends the complete-cluster phase; residual
    fill then extends only through sequence-adjacent residues.
    """
    selected: set[int] = set()
    complete_cluster_ids: list[int] = []
    activated_cluster_ids: list[int] = []
    group_ids = np.zeros(partition.residue_cluster_id.shape[0], dtype=np.int32)
    copy_count = 0
    residue_asym_ids = _residue_asym_ids(partition)
    min_contacts = int(min_cluster_sequence_contacts)

    for cluster_id in cluster_order:
        cluster = partition.clusters[int(cluster_id)]
        if selected and (
            _sequence_boundary_residue_count(
                cluster.residue_indices,
                selected,
                residue_asym_ids,
            )
            < min_contacts
        ):
            continue
        potential = potentials.get(int(cluster_id))
        can_activate = (
            potential is not None and len(activated_cluster_ids) < int(max_motif_groups)
        )
        activate = can_activate and (
            len(activated_cluster_ids) < int(min_motif_groups)
            or float(np_rng.random()) < float(p_motif_cluster)
        )
        projected = (
            len(selected | set(cluster.residue_indices.tolist()))
            + copy_count
            + (int(potential.residue_indices.size) if activate else 0)
            + len(activated_cluster_ids)
            + (1 if activate else 0)
        )
        if projected > int(max_crop_residues):
            break
        selected.update(int(index) for index in cluster.residue_indices)
        complete_cluster_ids.append(int(cluster_id))
        if activate:
            group_id = len(activated_cluster_ids) + 1
            group_ids[potential.residue_indices] = group_id
            copy_count += int(potential.residue_indices.size)
            activated_cluster_ids.append(int(cluster_id))

    residual_scaffold_capacity = (
        int(max_crop_residues) - copy_count - len(activated_cluster_ids) - len(selected)
    )
    partial_cluster_ids: list[int] = []
    if residual_scaffold_capacity > 0 and selected:
        complete_ids = set(complete_cluster_ids)
        remaining = {
            int(index)
            for cluster_id in cluster_order
            if int(cluster_id) not in complete_ids
            for index in partition.clusters[int(cluster_id)].residue_indices
            if int(index) not in selected
        }
        taken: list[int] = []
        while remaining and len(taken) < residual_scaffold_capacity:
            adjacent = [
                index
                for index in remaining
                if _sequence_boundary_residue_count(
                    np.asarray([index], dtype=np.int32),
                    selected,
                    residue_asym_ids,
                )
                > 0
            ]
            if not adjacent:
                break
            adjacent_array = np.asarray(sorted(adjacent), dtype=np.int32)
            selected_coords = partition.ca_coords[
                np.asarray(sorted(selected), dtype=np.int32)
            ]
            distances = np.linalg.norm(
                partition.ca_coords[adjacent_array, None, :]
                - selected_coords[None, :, :],
                axis=-1,
            ).min(axis=1)
            tie_order = np.asarray(
                np_rng.permutation(adjacent_array.size),
                dtype=np.int64,
            )
            chosen = int(adjacent_array[np.lexsort((tie_order, distances))[0]])
            selected.add(chosen)
            taken.append(chosen)
            remaining.remove(chosen)
        if taken:
            partial_cluster_ids = sorted(
                {
                    int(partition.residue_cluster_id[int(index)])
                    for index in taken
                }
            )

    projected_token_count = len(selected) + copy_count + len(activated_cluster_ids)
    if projected_token_count > int(max_crop_residues):
        raise RuntimeError("Voronoi crop exceeded max_crop_residues")
    return VoronoiCropSelection(
        scaffold_residue_indices=np.asarray(sorted(selected), dtype=np.int32),
        motif_group_id_residue=group_ids,
        complete_cluster_ids=tuple(complete_cluster_ids),
        activated_cluster_ids=tuple(activated_cluster_ids),
        partial_cluster_ids=tuple(partial_cluster_ids),
        copy_count=copy_count,
        projected_token_count=projected_token_count,
        group_shortfall=len(activated_cluster_ids) < int(min_motif_groups),
    )
