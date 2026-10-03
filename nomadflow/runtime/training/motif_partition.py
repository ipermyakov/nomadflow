"""Assign motif residues to independent rigid-motion groups for multi-motif SE(3) augmentation."""

from __future__ import annotations

from typing import Any

import numpy as np

from nomadflow.openfold.data import residue_constants as rc

PARTITION_MODES = ("fragments", "mask_blocks", "iid", "spatial", "subfragment")


def _rng_integers(np_rng, low: int, high: int, size=None, dtype=np.int32):
    if hasattr(np_rng, "integers"):
        return np_rng.integers(low, high, size=size, dtype=dtype)
    out = np_rng.randint(low, high, size=size)
    return np.asarray(out, dtype=dtype)


def _resolved_order_indices(resolved_mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(resolved_mask)


def asym_id_for_residue(data: Any, res_idx: int) -> int:
    for chain in data.chains[data.mask]:
        res_start = int(chain["res_idx"])
        res_end = res_start + int(chain["res_num"])
        if res_start <= res_idx < res_end:
            return int(chain["asym_id"])
    return 0


def ca_coord_for_residue(data: Any, res_idx: int) -> np.ndarray | None:
    residue = data.residues[res_idx]
    if not residue["is_standard"] or residue["name"] == "UNK":
        return None
    atom_start = int(residue["atom_idx"])
    atom_num = int(residue["atom_num"])
    atoms = data.atoms[atom_start: atom_start + atom_num]
    atom_order = rc.restype_name_to_atom14_names[residue["name"]]
    ca_local = atom_order.index("CA")
    atom = atoms[ca_local]
    if not atom["is_present"]:
        return None
    return np.asarray(atom["coords"], dtype=np.float64)


def build_residue_spatial_lookup(data: Any) -> tuple[np.ndarray, np.ndarray]:
    """Per-residue chain asym_id and CA coords (NaN row when CA is unavailable)."""
    num_residues = len(data.residues)
    asym_ids = np.zeros(num_residues, dtype=np.int32)
    ca_coords = np.full((num_residues, 3), np.nan, dtype=np.float64)
    for chain in data.chains[data.mask]:
        res_start = int(chain["res_idx"])
        res_end = res_start + int(chain["res_num"])
        asym_ids[res_start:res_end] = int(chain["asym_id"])
        for res_idx in range(res_start, res_end):
            coord = ca_coord_for_residue(data, res_idx)
            if coord is not None:
                ca_coords[res_idx] = coord
    return asym_ids, ca_coords


def farthest_point_centers(
    pool: np.ndarray,
    ca_coords: np.ndarray,
    count: int,
    *,
    np_rng,
    min_center_dist: float | None = None,
) -> np.ndarray:
    """Choose reproducible farthest-point centers from valid CA coordinates."""
    pool = np.asarray(pool, dtype=np.int32)
    valid = pool[~np.isnan(ca_coords[pool]).any(axis=1)]
    if valid.size == 0 or int(count) <= 0:
        return np.zeros(0, dtype=np.int32)
    first = int(np_rng.choice(valid))
    centers = [first]
    while len(centers) < min(int(count), int(valid.size)):
        candidates = valid[~np.isin(valid, centers)]
        center_coords = ca_coords[np.asarray(centers, dtype=np.int32)]
        min_distances = np.linalg.norm(
            ca_coords[candidates, None, :] - center_coords[None, :, :],
            axis=-1,
        ).min(axis=1)
        best_distance = float(min_distances.max())
        if min_center_dist is not None and best_distance < float(min_center_dist):
            break
        tied = candidates[np.isclose(min_distances, best_distance)]
        centers.append(int(np_rng.choice(tied)))
    return np.asarray(centers, dtype=np.int32)


def _motif_residue_indices(
    resolved_idx: np.ndarray,
    motif_mask_compressed: np.ndarray,
) -> np.ndarray:
    positions = np.flatnonzero(np.asarray(motif_mask_compressed, dtype=bool))
    return resolved_idx[positions]


def _contiguous_runs_in_compressed(pos_sorted: np.ndarray) -> list[tuple[int, int]]:
    if pos_sorted.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(pos_sorted[0])
    prev = int(pos_sorted[0])
    for pos in pos_sorted[1:]:
        pos = int(pos)
        if pos == prev + 1:
            prev = pos
            continue
        runs.append((start, prev))
        start = pos
        prev = pos
    runs.append((start, prev))
    return runs


def _wrap_group_id(run_idx: int, max_groups: int) -> int:
    return int((run_idx % max_groups) + 1)


def _partition_mask_blocks(
    resolved_idx: np.ndarray,
    motif_mask_compressed: np.ndarray,
    out: np.ndarray,
    max_groups: int,
) -> None:
    positions = np.flatnonzero(np.asarray(motif_mask_compressed, dtype=bool))
    runs = _contiguous_runs_in_compressed(positions)
    for run_index, (start, end) in enumerate(runs):
        group_id = _wrap_group_id(run_index, max_groups)
        for comp in range(start, end + 1):
            out[int(resolved_idx[comp])] = group_id


def _seq_chain_fragments(data: Any, motif_res: np.ndarray) -> list[list[int]]:
    if motif_res.size == 0:
        return []
    asym_ids = np.array([asym_id_for_residue(data, int(res_idx)) for res_idx in motif_res], dtype=np.int32)
    order = np.lexsort((motif_res.astype(np.int32), asym_ids))
    sorted_res = motif_res.astype(np.int32)[order]
    asym_sorted = asym_ids[order]
    fragments: list[list[int]] = []
    current_fragment: list[int] = [int(sorted_res[0])]
    for index in range(1, len(sorted_res)):
        residue_index = int(sorted_res[index])
        prev_residue_index = int(sorted_res[index - 1])
        if asym_sorted[index] != asym_sorted[index - 1] or residue_index != prev_residue_index + 1:
            fragments.append(current_fragment)
            current_fragment = [residue_index]
        else:
            current_fragment.append(residue_index)
    fragments.append(current_fragment)
    return fragments


def _partition_fragments(
    data: Any,
    motif_res: np.ndarray,
    out: np.ndarray,
    max_groups: int,
) -> None:
    fragments = _seq_chain_fragments(data, motif_res)
    for fragment_index, fragment in enumerate(fragments):
        group_id = _wrap_group_id(fragment_index, max_groups)
        out[np.array(fragment, dtype=np.int32)] = group_id


def _partition_subfragment(
    data: Any,
    motif_res: np.ndarray,
    out: np.ndarray,
    np_rng,
    max_groups: int,
) -> None:
    fragments = _seq_chain_fragments(data, motif_res)
    group_counter = 0
    for fragment in fragments:
        if len(fragment) >= 2 and float(np_rng.random()) < 0.5:
            split_index = int(_rng_integers(np_rng, 1, len(fragment)))
            left, right = fragment[:split_index], fragment[split_index:]
            for part in (left, right):
                out[np.array(part, dtype=np.int32)] = _wrap_group_id(group_counter, max_groups)
                group_counter += 1
        else:
            out[np.array(fragment, dtype=np.int32)] = _wrap_group_id(group_counter, max_groups)
            group_counter += 1


def _partition_iid(
    motif_res: np.ndarray,
    out: np.ndarray,
    np_rng,
    max_groups: int,
) -> None:
    count = int(motif_res.size)
    if count == 0:
        return
    num_groups = int(min(max_groups, max(1, count)))
    labels = _rng_integers(np_rng, 1, num_groups + 1, size=count, dtype=np.int32)
    out[motif_res.astype(np.int32)] = labels


def _partition_spatial(
    data: Any,
    motif_res: np.ndarray,
    out: np.ndarray,
    np_rng,
    max_groups: int,
) -> None:
    if motif_res.size == 0:
        return
    coords = []
    valid_residue_indices: list[int] = []
    for residue_index in motif_res:
        coord = ca_coord_for_residue(data, int(residue_index))
        if coord is not None:
            coords.append(coord)
            valid_residue_indices.append(int(residue_index))
    if len(valid_residue_indices) < 2:
        if valid_residue_indices:
            out[np.array(valid_residue_indices, dtype=np.int32)] = 1
        return
    points = np.stack(coords, axis=0)
    num_groups = int(min(max_groups, points.shape[0]))
    try:
        from scipy.cluster.vq import kmeans2

        _, labels = kmeans2(points, num_groups, minit="points", rng=np_rng)
    except Exception:
        _partition_fragments(data, motif_res, out, max_groups)
        return
    for label, residue_index in zip(labels, valid_residue_indices):
        out[residue_index] = int(label) + 1


def build_motif_group_id_by_residue(
    data: Any,
    residue_noising_mask: np.ndarray,
    motif_mask_compressed: np.ndarray,
    resolved_mask: np.ndarray,
    *,
    enabled: bool,
    partition_mode: str,
    max_groups: int,
    np_rng,
) -> np.ndarray:
    """Per-residue motif group ids: 0 for scaffold, 1..G for fixed motif groups."""
    num_residues = int(residue_noising_mask.shape[0])
    out = np.zeros(num_residues, dtype=np.int32)
    if not enabled or partition_mode not in PARTITION_MODES:
        return out

    resolved_idx = _resolved_order_indices(resolved_mask)
    motif_res = _motif_residue_indices(resolved_idx, motif_mask_compressed)
    if motif_res.size == 0:
        return out

    if partition_mode == "mask_blocks":
        _partition_mask_blocks(resolved_idx, motif_mask_compressed, out, max_groups)
    elif partition_mode == "fragments":
        _partition_fragments(data, motif_res, out, max_groups)
    elif partition_mode == "subfragment":
        _partition_subfragment(data, motif_res, out, np_rng, max_groups)
    elif partition_mode == "iid":
        _partition_iid(motif_res, out, np_rng, max_groups)
    elif partition_mode == "spatial":
        _partition_spatial(data, motif_res, out, np_rng, max_groups)
    else:
        _partition_fragments(data, motif_res, out, max_groups)

    scaffold_mask = residue_noising_mask.astype(bool)
    out[scaffold_mask] = 0
    out[~resolved_mask] = 0
    return out
