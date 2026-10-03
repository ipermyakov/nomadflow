from functools import partialmethod

import math
import numpy as np
import torch

from nomadflow.data.constants import coarse_grain as cg
from nomadflow.data.featurize.tokenize import convert_atom_str_to_tuple

from nomadflow.boltz.data import const

from .motif_partition import (
    PARTITION_MODES,
    asym_id_for_residue,
    build_motif_group_id_by_residue,
    build_residue_spatial_lookup,
    ca_coord_for_residue,
    farthest_point_centers,
)
from .task import TrainingTask, resolve_np_rng, sample_task_t


def rigid_noise_to_atom_noise(residue, atoms, rigid_noising_mask):
    res_name = residue['name']

    bb_group = ['N', 'CA', 'C', 'O', 'CB']
    # bb_frame = ['C', 'CA', 'N']
    group2 = cg.coarse_grain_sidechain_groups[res_name][2]
    group3 = cg.coarse_grain_sidechain_groups[res_name][3]
    # construct dummy frames as necessary
    # use bb frame if frame2 doesn't exist
    if len(group2) == 0:
        group2 = bb_group
    # use frame2 frame if frame3 doesn't exist
    if len(group3) == 0:
        group3 = group2

    atom_noise_mapping = {}
    frame_atom_groups = [bb_group, group2, group3]
    for i, atom_groups in enumerate(frame_atom_groups):
        noise_atom = rigid_noising_mask[i]
        for atom_name in atom_groups:
            atom_id = convert_atom_str_to_tuple(atom_name)
            atom_noise_mapping[atom_id] = noise_atom

    atom_noising_mask = []
    for atom in atoms:
        atom_name_tuple = tuple(atom["name"])
        if atom_name_tuple in atom_noise_mapping:
            atom_noising_mask.append(
                atom_noise_mapping[atom_name_tuple]
            )
        else:
            atom_noising_mask.append(True)

    return np.array(atom_noising_mask)


def motif_frame_mask_from_selection(fix_tip, noise_sidechain_only, is_motif):
    """Map the historical p_fix_* sample to rigid frames that belong to a motif group."""
    if not is_motif:
        return np.array([False, False, False], dtype=bool)
    if fix_tip:
        return np.array([False, True, False], dtype=bool)
    if noise_sidechain_only:
        return np.array([True, False, False], dtype=bool)
    return np.array([True, True, True], dtype=bool)


def _rng_randint(np_rng, low: int, high: int, size=None):
    if hasattr(np_rng, "integers"):
        return np_rng.integers(low, high, size=size)
    return np_rng.randint(low, high, size=size)


def _protein_standard_residue_indices(data, resolved_mask: np.ndarray) -> np.ndarray:
    protein_type = const.chain_type_ids["PROTEIN"]
    indices: list[int] = []
    for chain in data.chains[data.mask]:
        if chain["mol_type"] != protein_type:
            continue
        res_start = int(chain["res_idx"])
        res_end = res_start + int(chain["res_num"])
        for res_idx in range(res_start, res_end):
            if not resolved_mask[res_idx]:
                continue
            residue = data.residues[res_idx]
            if residue["is_standard"] and residue["name"] != "UNK":
                indices.append(res_idx)
    return np.asarray(indices, dtype=np.int32)


def _sample_positive_partition(total: int, num_parts: int, np_rng) -> np.ndarray:
    num_parts = int(num_parts)
    total = int(total)
    if num_parts <= 0:
        return np.zeros(0, dtype=np.int32)
    if num_parts == 1:
        return np.array([total], dtype=np.int32)
    cuts = np_rng.choice(total - 1, size=num_parts - 1, replace=False) + 1
    cuts = np.sort(np.concatenate([[0], cuts, [total]], axis=0))
    return (cuts[1:] - cuts[:-1]).astype(np.int32)


def _apply_fragment_size_limits(
    sizes: np.ndarray,
    total: int,
    *,
    min_fragment_size: int | None,
    max_fragment_size: int | None,
) -> np.ndarray:
    sizes = sizes.astype(np.int32, copy=True)
    if min_fragment_size is not None:
        sizes = np.maximum(sizes, int(min_fragment_size))
    if max_fragment_size is not None:
        sizes = np.minimum(sizes, int(max_fragment_size))
    delta = int(total) - int(sizes.sum())
    if delta != 0 and sizes.size > 0:
        sizes[-1] += delta
    if sizes.size > 0:
        sizes[sizes < 1] = 1
    return sizes


def _spatial_fragment_candidate_pool(
    seed_idx: int,
    pool: np.ndarray,
    used_mask: np.ndarray,
    *,
    asym_ids: np.ndarray,
    ca_coords: np.ndarray,
    max_ca_dist: float | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    seed_coord = ca_coords[seed_idx]
    if np.isnan(seed_coord).any():
        return None
    chain_asym = int(asym_ids[seed_idx])
    candidates = pool[~used_mask[pool]]
    if candidates.size == 0:
        return None
    candidates = candidates[asym_ids[candidates] == chain_asym]
    if candidates.size == 0:
        return None
    coords = ca_coords[candidates]
    valid = ~np.isnan(coords).any(axis=1)
    candidates = candidates[valid]
    coords = coords[valid]
    if candidates.size == 0:
        return None
    dists = np.linalg.norm(coords - seed_coord, axis=1)
    if max_ca_dist is not None:
        in_range = dists <= max_ca_dist
        candidates = candidates[in_range]
        dists = dists[in_range]
        if candidates.size == 0:
            return None
    return candidates, dists


def _spatial_fragment_residues_knn(candidates: np.ndarray, dists: np.ndarray, size: int) -> np.ndarray:
    k = min(int(size), int(candidates.size))
    nearest = np.argpartition(dists, k - 1)[:k]
    return candidates[nearest].astype(np.int32, copy=False)


def _spatial_fragment_residues_shell(
    seed_idx: int,
    candidates: np.ndarray,
    dists: np.ndarray,
    size: int,
    np_rng,
    *,
    shell_center_dist: float,
    shell_half_width: float,
) -> np.ndarray:
    size = int(size)
    if size <= 0:
        return np.zeros(0, dtype=np.int32)
    if size == 1:
        return np.array([seed_idx], dtype=np.int32)

    seed_in_pool = candidates == seed_idx
    shell_mask = np.abs(dists - float(shell_center_dist)) <= float(shell_half_width)
    shell_mask &= ~seed_in_pool
    shell_candidates = candidates[shell_mask]

    selected: list[int] = [int(seed_idx)]
    need = size - 1
    if shell_candidates.size > 0 and need > 0:
        pick = min(need, int(shell_candidates.size))
        chosen = np_rng.choice(shell_candidates, size=pick, replace=False)
        selected.extend(int(i) for i in np.atleast_1d(chosen))

    if len(selected) < size:
        selected_set = set(selected)
        remaining_mask = np.array([int(idx) not in selected_set for idx in candidates], dtype=bool)
        remaining_candidates = candidates[remaining_mask]
        remaining_dists = dists[remaining_mask]
        fill = _spatial_fragment_residues_knn(
            remaining_candidates,
            remaining_dists,
            size - len(selected),
        )
        selected.extend(int(i) for i in fill)
    return np.unique(np.asarray(selected[:size], dtype=np.int32))


def _spatial_fragment_residues(
    data,
    seed_idx: int,
    pool: np.ndarray,
    used_mask: np.ndarray,
    size: int,
    *,
    max_ca_dist: float | None,
    asym_ids: np.ndarray | None = None,
    ca_coords: np.ndarray | None = None,
    np_rng=None,
    shell_sampling: bool = False,
    shell_center_dist: float = 10.0,
    shell_half_width: float = 2.0,
) -> np.ndarray:
    if asym_ids is None or ca_coords is None:
        asym_ids, ca_coords = build_residue_spatial_lookup(data)
    pooled = _spatial_fragment_candidate_pool(
        seed_idx,
        pool,
        used_mask,
        asym_ids=asym_ids,
        ca_coords=ca_coords,
        max_ca_dist=max_ca_dist,
    )
    if pooled is None:
        return np.zeros(0, dtype=np.int32)
    candidates, dists = pooled
    if shell_sampling:
        if np_rng is None:
            np_rng = resolve_np_rng(None)
        return _spatial_fragment_residues_shell(
            seed_idx,
            candidates,
            dists,
            size,
            np_rng,
            shell_center_dist=shell_center_dist,
            shell_half_width=shell_half_width,
        )
    return _spatial_fragment_residues_knn(candidates, dists, size)


def _ca_min_dist_to_set(
    ca_coords: np.ndarray,
    query_idx: int,
    ref_indices: np.ndarray,
) -> float:
    if ref_indices.size == 0:
        return float("inf")
    q = ca_coords[query_idx]
    if np.isnan(q).any():
        return float("nan")
    refs = ca_coords[ref_indices]
    valid = ~np.isnan(refs).any(axis=1)
    if not np.any(valid):
        return float("inf")
    return float(np.linalg.norm(refs[valid] - q, axis=1).min())


def _mask_within_ca_dist_of(
    ca_coords: np.ndarray,
    source_indices: np.ndarray,
    dist: float,
) -> np.ndarray:
    """Boolean mask over all residues within ``dist`` CA of any source residue."""
    n = ca_coords.shape[0]
    out = np.zeros(n, dtype=bool)
    if source_indices.size == 0 or dist <= 0:
        return out
    sources = ca_coords[source_indices]
    src_valid = ~np.isnan(sources).any(axis=1)
    sources = sources[src_valid]
    if sources.size == 0:
        return out
    for i in range(n):
        c = ca_coords[i]
        if np.isnan(c).any():
            continue
        if float(np.linalg.norm(sources - c, axis=1).min()) < float(dist):
            out[i] = True
    return out


def _fps_centers(
    pool: np.ndarray,
    ca_coords: np.ndarray,
    num_groups: int,
    min_center_dist: float,
    np_rng,
) -> np.ndarray:
    """Farthest-point centers; stop early if next point is closer than ``min_center_dist``."""
    return farthest_point_centers(
        pool,
        ca_coords,
        num_groups,
        np_rng=np_rng,
        min_center_dist=min_center_dist,
    )


def _chain_residue_order(
    pool: np.ndarray,
    asym_ids: np.ndarray,
    chain_asym: int,
) -> np.ndarray:
    on_chain = pool[asym_ids[pool] == int(chain_asym)]
    if on_chain.size == 0:
        return on_chain.astype(np.int32, copy=False)
    return np.sort(on_chain.astype(np.int32, copy=False))


def _expand_contiguous_window(
    chain_order: np.ndarray,
    anchor_idx: int,
    size: int,
    forbidden: np.ndarray,
) -> np.ndarray:
    """Grow a contiguous sequence window around ``anchor_idx`` avoiding ``forbidden``."""
    size = int(size)
    if size <= 0 or chain_order.size == 0:
        return np.zeros(0, dtype=np.int32)
    positions = np.flatnonzero(chain_order == int(anchor_idx))
    if positions.size == 0:
        return np.zeros(0, dtype=np.int32)
    pos = int(positions[0])
    if forbidden[int(anchor_idx)]:
        return np.zeros(0, dtype=np.int32)
    left = right = pos
    while (right - left + 1) < size:
        can_left = left > 0 and not forbidden[int(chain_order[left - 1])]
        can_right = right + 1 < chain_order.size and not forbidden[int(chain_order[right + 1])]
        if not can_left and not can_right:
            break
        left_extent = pos - left
        right_extent = right - pos
        if can_left and (not can_right or left_extent <= right_extent):
            left -= 1
        elif can_right:
            right += 1
        else:
            break
    return chain_order[left : right + 1].astype(np.int32, copy=False)


def _min_pairwise_ca_between_sets(
    ca_coords: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    if a.size == 0 or b.size == 0:
        return float("inf")
    ca_a = ca_coords[a]
    ca_b = ca_coords[b]
    va = ~np.isnan(ca_a).any(axis=1)
    vb = ~np.isnan(ca_b).any(axis=1)
    ca_a = ca_a[va]
    ca_b = ca_b[vb]
    if ca_a.size == 0 or ca_b.size == 0:
        return float("inf")
    # (Na, 1, 3) - (1, Nb, 3)
    diffs = ca_a[:, None, :] - ca_b[None, :, :]
    return float(np.linalg.norm(diffs, axis=-1).min())


# inspired by Genie2
class MotifScaffolding(TrainingTask):
    name: str = "motif_scaffolding"
    def __init__(self,
                 prob=0.0,
                 t_sched='lognorm',
                 mode='mixed',
                 lognorm_mu=0.0,
                 lognorm_sig=1.0,
                 beta_p1=1.9,
                 beta_p2=1.0,
                 shift_time_scale=False,
                 t_min=0.01,
                 t_max=0.99,
                 max_frac_res=0.5,
                 max_num_res=40,
                 max_num_segments=4,
                 p_is_unindexed=0.8,
                 name_override=None,
                 p_fix_tip=0.0,
                 p_fix_bb=0.5,
                 p_fix_all=0.5,
                 multi_motif_se3: bool = False,
                 multi_motif_partition_mode: str = "fragments",
                 multi_motif_max_groups: int = 8,
                 multi_motif_se3_trans_std: float = 16.0,
                 motif_t_lock: float | None = None,
    ):
        assert t_sched in ['lognorm', 'mixed_beta', 'uniform']
        assert mode in ['mixed']
        assert multi_motif_partition_mode in PARTITION_MODES
        self.prob = prob
        self.t_sched = t_sched
        self.mode = mode
        self.lognorm_mu = lognorm_mu
        self.lognorm_sig = lognorm_sig
        self.beta_p1 = beta_p1
        self.beta_p2 = beta_p2
        self.t_min = t_min
        self.t_max = t_max
        self.shift_time_scale = shift_time_scale

        self.max_frac_res = max_frac_res
        self.max_num_res = max_num_res
        self.max_num_segments = max_num_segments
        self.p_is_unindexed = p_is_unindexed

        self.p_fix_tip = p_fix_tip
        self.p_fix_bb = p_fix_bb
        self.p_fix_all = p_fix_all
        assert np.isclose(sum([p_fix_tip, p_fix_bb, p_fix_all]), 1.0)

        self.multi_motif_se3 = multi_motif_se3
        self.multi_motif_partition_mode = multi_motif_partition_mode
        self.multi_motif_max_groups = int(multi_motif_max_groups)
        self.multi_motif_se3_trans_std = float(multi_motif_se3_trans_std)
        self.motif_t_lock = motif_t_lock

        if name_override is not None:
            self.name = name_override

    def generate_motif_mask(
        self, N, np_rng=None, *, data=None, resolved_mask=None, num_res=None
    ):
        del data, resolved_mask, num_res
        np_rng = resolve_np_rng(np_rng)
        num_segments = np_rng.randint(1, self.max_num_segments + 1)
        # we switch this from Genie2
        # to increase the probability of sampling minimal motifs
        # we also give the option to lower the maximum number of residues sampled
        res_cap = min(math.ceil(self.max_frac_res * N), self.max_num_res)
        res_cap = max(res_cap, num_segments) + 1
        num_res = np_rng.randint(num_segments, res_cap) + 1
        # num_res = np.random.randint(math.floor(0.05 * N), math.ceil(0.5 * N) + 1)
        # # when N < 80 it's possible for num_segments > num_res
        # num_res = max(num_segments + 1, num_res)
        B = np_rng.choice(num_res - 1, size=num_segments, replace=False) + 1
        B = np.sort(
            np.concatenate([[0], B, [num_res]], axis=0)
        )
        L = B[1:] - B[:-1]
        permute_list = [[0] for _ in range(N - num_res)] + [np.ones((l,)) for l in L]
        permutation = np_rng.permutation(len(permute_list))
        permuted_list = [permute_list[i] for i in permutation]
        M = np.concatenate(permuted_list, axis=0)
        return torch.as_tensor(M, dtype=torch.bool)

    def generate_motif_mask_and_groups(
        self, N, np_rng=None, *, data=None, resolved_mask=None, num_res=None
    ):
        del data, resolved_mask, num_res
        return self.generate_motif_mask(N, np_rng=np_rng), None

    def _motif_group_id_from_generation(
        self,
        data,
        residue_noising_mask: np.ndarray,
        motif_mask_compressed: np.ndarray,
        resolved_mask: np.ndarray,
        np_rng,
        *,
        preset_motif_group_id_residue: np.ndarray | None,
    ) -> np.ndarray:
        if preset_motif_group_id_residue is not None:
            out = np.asarray(preset_motif_group_id_residue, dtype=np.int32).copy()
            if not self.multi_motif_se3:
                out.fill(0)
        else:
            out = build_motif_group_id_by_residue(
                data,
                residue_noising_mask,
                motif_mask_compressed,
                resolved_mask,
                enabled=self.multi_motif_se3,
                partition_mode=self.multi_motif_partition_mode,
                max_groups=self.multi_motif_max_groups,
                np_rng=np_rng,
            )
        out[residue_noising_mask.astype(bool)] = 0
        out[~resolved_mask] = 0
        return out

    def sample_t_and_mask(
        self,
        data,
        np_rng=None,
        torch_generator=None,
        *,
        preset_motif_group_id_residue=None,
    ):
        residues = data.residues
        atoms = data.atoms

        np_rng = resolve_np_rng(np_rng)
        t = sample_task_t(
            t_sched=self.t_sched,
            lognorm_mu=self.lognorm_mu,
            lognorm_sig=self.lognorm_sig,
            beta_p1=self.beta_p1,
            beta_p2=self.beta_p2,
            t_min=self.t_min,
            t_max=self.t_max,
            device='cpu',
            np_rng=np_rng if np_rng is not np.random else None,
            torch_generator=torch_generator,
        )

        resolved_mask = data.residues['is_present']
        num_resolved_residues = int(resolved_mask.sum())
        if preset_motif_group_id_residue is None:
            motif_mask, preset_motif_group_id_residue = self.generate_motif_mask_and_groups(
                num_resolved_residues,
                np_rng=np_rng,
                data=data,
                resolved_mask=resolved_mask,
            )
        else:
            preset_motif_group_id_residue = np.asarray(
                preset_motif_group_id_residue,
                dtype=np.int32,
            )
            if preset_motif_group_id_residue.shape != resolved_mask.shape:
                raise ValueError(
                    "preset_motif_group_id_residue must match the residue array shape"
                )
            motif_mask = torch.as_tensor(
                preset_motif_group_id_residue[resolved_mask] > 0,
                dtype=torch.bool,
            )
        residue_noising_mask = np.ones_like(resolved_mask)
        residue_noising_mask[resolved_mask] = ~motif_mask
        res_type_noising_mask = residue_noising_mask.copy()

        atom_noising_mask = []
        motif_frame_mask_by_residue = np.zeros((resolved_mask.shape[0], 3), dtype=bool)
        if self.mode == 'mixed':
            noise_select = torch.rand(
                torch.as_tensor(residue_noising_mask).shape,
                dtype=torch.float32,
                generator=torch_generator,
            ).numpy(force=True)
            fix_tip = noise_select < self.p_fix_tip
            noise_sidechain_only = noise_select > (1 - self.p_fix_bb)
            res_type_noising_mask[~residue_noising_mask & fix_tip] = False
            res_type_noising_mask[~residue_noising_mask & noise_sidechain_only] = True

            for chain in data.chains:
                # Get residue indices
                res_start = chain["res_idx"]
                res_end = chain["res_idx"] + chain["res_num"]
                is_protein = chain["mol_type"] == const.chain_type_ids["PROTEIN"]

                for i, residue in enumerate(data.residues[res_start:res_end]):
                    res_pos = res_start + i
                    atom_idx = residue['atom_idx']
                    atom_num = residue['atom_num']
                    atoms = data.atoms[atom_idx:atom_idx+atom_num]
                    if residue["is_standard"] and (residue['name'] != 'UNK') and is_protein:
                        is_motif = ~residue_noising_mask[res_pos]
                        frame_mask = motif_frame_mask_from_selection(
                            fix_tip[res_pos],
                            noise_sidechain_only[res_pos],
                            is_motif,
                        )
                        motif_frame_mask_by_residue[res_pos] = frame_mask
                        if self.multi_motif_se3:
                            rigid_noise_mask = [True, True, True]
                        else:
                            rigid_noise_mask = (~frame_mask).tolist()
                        atom_noising_mask.append(rigid_noise_to_atom_noise(residue, atoms, rigid_noise_mask))
                    else:
                        atom_noising_mask.append([True for _ in atoms])

        else:
            raise ValueError()
        atom_noising_mask = np.concatenate(atom_noising_mask)

        is_unindexed_residue = torch.rand(
            torch.as_tensor(residue_noising_mask).shape,
            dtype=torch.float32,
            generator=torch_generator,
        ) < self.p_is_unindexed
        is_unindexed_residue = is_unindexed_residue.numpy(force=True)

        copy_indexed_residue_mask = ~residue_noising_mask & ~is_unindexed_residue
        copy_unindexed_residue_mask = ~residue_noising_mask & is_unindexed_residue
        copy_atomized_residue_mask = np.zeros_like(res_type_noising_mask)

        motif_mask_np = motif_mask.numpy(force=True).astype(bool, copy=False)
        motif_group_id_residue = self._motif_group_id_from_generation(
            data,
            residue_noising_mask,
            motif_mask_np,
            resolved_mask,
            np_rng,
            preset_motif_group_id_residue=preset_motif_group_id_residue,
        )
        motif_group_id_rigid_by_residue_frame = (
            motif_frame_mask_by_residue.astype(np.int32)
            * motif_group_id_residue[:, None].astype(np.int32)
        )

        return {
            "t": t.numpy(force=True),
            "atom_noising_mask": atom_noising_mask,
            "res_type_noising_mask": res_type_noising_mask,
            "copy_indexed_residue_mask": copy_indexed_residue_mask,
            "copy_unindexed_residue_mask": copy_unindexed_residue_mask,
            "copy_atomized_residue_mask": copy_atomized_residue_mask,
            "motif_group_id_residue": motif_group_id_residue,
            "motif_group_id_rigid_by_residue_frame": motif_group_id_rigid_by_residue_frame,
            "multi_motif_se3": self.multi_motif_se3,
            "multi_motif_se3_trans_std": self.multi_motif_se3_trans_std,
            "motif_t_lock": self.motif_t_lock,
        }

    def max_added_tokens(self, N):
        res_cap = min(math.ceil(self.max_frac_res * N), self.max_num_res)
        return res_cap

    def max_motif_groups(self) -> int:
        """Return an upper bound for virtual motif-group aggregators."""
        if not self.multi_motif_se3:
            return 0
        return self.multi_motif_max_groups


class SpatialFragmentMotifScaffolding(MotifScaffolding):
    """Motif scaffolding via per-chain CA spatial fragments with explicit motif_group_id.

    By default each fragment is the ``size`` nearest CA neighbors on the same chain.
    When ``small_fragment_shell_sampling`` is enabled and ``target_size`` is below
    ``small_fragment_shell_max_size``, residues are drawn uniformly from the CA-distance
  shell ``[center - half_width, center + half_width]`` (seed always included).

    Hydra example::

        - _target_: nomadflow.runtime.training.motif_scaffold.SpatialFragmentMotifScaffolding
          prob: 0.25
          max_frac_res: 0.5
          max_num_res: 40
          max_num_fragments: 4
          small_fragment_shell_sampling: true
          small_fragment_shell_max_size: 6
          small_fragment_shell_center_dist: 12.0
          small_fragment_shell_half_width: 2.0
          multi_motif_se3: true
          one_fragment_one_group: true
          motif_t_lock: 0.5
    """

    name: str = "spatial_motif_scaffolding"

    def __init__(
        self,
        prob=0.0,
        t_sched='lognorm',
        mode='mixed',
        lognorm_mu=0.0,
        lognorm_sig=1.0,
        beta_p1=1.9,
        beta_p2=1.0,
        shift_time_scale=False,
        t_min=0.01,
        t_max=0.99,
        max_frac_res=0.5,
        max_num_res=40,
        max_num_fragments=4,
        min_fragment_size=None,
        max_fragment_size=None,
        max_ca_dist=None,
        small_fragment_shell_sampling: bool = False,
        small_fragment_shell_max_size: int = 6,
        small_fragment_shell_center_dist: float = 10.0,
        small_fragment_shell_half_width: float = 2.0,
        p_is_unindexed=0.8,
        name_override=None,
        p_fix_tip=0.0,
        p_fix_bb=0.5,
        p_fix_all=0.5,
        multi_motif_se3: bool = False,
        multi_motif_max_groups: int = 8,
        multi_motif_se3_trans_std: float = 16.0,
        one_fragment_one_group: bool = False,
        motif_t_lock: float | None = None,
    ):
        super().__init__(
            prob=prob,
            t_sched=t_sched,
            mode=mode,
            lognorm_mu=lognorm_mu,
            lognorm_sig=lognorm_sig,
            beta_p1=beta_p1,
            beta_p2=beta_p2,
            shift_time_scale=shift_time_scale,
            t_min=t_min,
            t_max=t_max,
            max_frac_res=max_frac_res,
            max_num_res=max_num_res,
            max_num_segments=1,
            p_is_unindexed=p_is_unindexed,
            name_override=name_override,
            p_fix_tip=p_fix_tip,
            p_fix_bb=p_fix_bb,
            p_fix_all=p_fix_all,
            multi_motif_se3=multi_motif_se3,
            multi_motif_partition_mode="fragments",
            multi_motif_max_groups=multi_motif_max_groups,
            multi_motif_se3_trans_std=multi_motif_se3_trans_std,
            motif_t_lock=motif_t_lock,
        )
        self.max_num_fragments = int(max_num_fragments)
        self.min_fragment_size = min_fragment_size
        self.max_fragment_size = max_fragment_size
        self.max_ca_dist = max_ca_dist
        self.small_fragment_shell_sampling = bool(small_fragment_shell_sampling)
        self.small_fragment_shell_max_size = int(small_fragment_shell_max_size)
        self.small_fragment_shell_center_dist = float(small_fragment_shell_center_dist)
        self.small_fragment_shell_half_width = float(small_fragment_shell_half_width)
        self.one_fragment_one_group = bool(one_fragment_one_group)

    def max_motif_groups(self) -> int:
        if not self.multi_motif_se3:
            return 0
        return min(self.max_num_fragments, self.multi_motif_max_groups)

    def _sample_fragment_group_ids(self, num_fragments: int, np_rng) -> np.ndarray:
        max_groups = min(self.multi_motif_max_groups, num_fragments)
        if self.one_fragment_one_group:
            return (np.arange(num_fragments, dtype=np.int32) % max_groups) + 1
        num_groups = int(_rng_randint(np_rng, 1, max_groups + 1))
        return _rng_randint(np_rng, 1, num_groups + 1, size=num_fragments)

    def _sample_num_res(self, N: int, num_fragments: int, np_rng, num_res=None) -> int:
        if num_res is not None:
            return min(int(num_res), int(N))
        res_cap = min(math.ceil(self.max_frac_res * N), self.max_num_res)
        res_cap = max(res_cap, num_fragments) + 1
        return int(_rng_randint(np_rng, num_fragments, res_cap)) + 1

    def generate_motif_mask(
        self, N, np_rng=None, *, data=None, resolved_mask=None, num_res=None
    ):
        mask, _ = self.generate_motif_mask_and_groups(
            N,
            np_rng=np_rng,
            data=data,
            resolved_mask=resolved_mask,
            num_res=num_res,
        )
        return mask

    def generate_motif_mask_and_groups(
        self, N, np_rng=None, *, data=None, resolved_mask=None, num_res=None
    ):
        del N
        if data is None or resolved_mask is None:
            raise ValueError("SpatialFragmentMotifScaffolding requires data and resolved_mask")
        np_rng = resolve_np_rng(np_rng)
        resolved_mask = np.asarray(resolved_mask, dtype=bool)
        num_resolved = int(resolved_mask.sum())
        pool = _protein_standard_residue_indices(data, resolved_mask)
        if pool.size == 0:
            empty = np.zeros(resolved_mask.shape[0], dtype=bool)
            return torch.as_tensor(empty[resolved_mask], dtype=torch.bool), np.zeros(
                resolved_mask.shape[0], dtype=np.int32
            )

        num_fragments = int(_rng_randint(np_rng, 1, self.max_num_fragments + 1))
        num_res = self._sample_num_res(num_resolved, num_fragments, np_rng, num_res=num_res)
        num_res = min(num_res, int(pool.size))
        num_fragments = min(num_fragments, num_res)

        fragment_sizes = _sample_positive_partition(num_res, num_fragments, np_rng)
        fragment_sizes = _apply_fragment_size_limits(
            fragment_sizes,
            num_res,
            min_fragment_size=self.min_fragment_size,
            max_fragment_size=self.max_fragment_size,
        )
        num_fragments = int(fragment_sizes.size)
        fragment_group_ids = self._sample_fragment_group_ids(num_fragments, np_rng)

        asym_ids, ca_coords = build_residue_spatial_lookup(data)
        num_residues = resolved_mask.shape[0]
        motif_full = np.zeros(num_residues, dtype=bool)
        motif_group_id_full = np.zeros(num_residues, dtype=np.int32)
        used_mask = np.zeros(num_residues, dtype=bool)
        carry = 0

        for fragment_size, group_id in zip(fragment_sizes, fragment_group_ids):
            target_size = int(fragment_size) + carry
            carry = 0
            if target_size <= 0:
                continue
            available = pool[~used_mask[pool]]
            if available.size == 0:
                break
            seed_idx = int(np_rng.choice(available))
            use_shell = (
                self.small_fragment_shell_sampling
                and target_size < self.small_fragment_shell_max_size
            )
            selected = _spatial_fragment_residues(
                data,
                seed_idx,
                pool,
                used_mask,
                target_size,
                max_ca_dist=self.max_ca_dist,
                asym_ids=asym_ids,
                ca_coords=ca_coords,
                np_rng=np_rng,
                shell_sampling=use_shell,
                shell_center_dist=self.small_fragment_shell_center_dist,
                shell_half_width=self.small_fragment_shell_half_width,
            )
            if selected.size == 0:
                continue
            if selected.size < target_size:
                carry = target_size - int(selected.size)
            motif_full[selected] = True
            motif_group_id_full[selected] = int(group_id)
            used_mask[selected] = True

        motif_compressed = motif_full[resolved_mask]
        return torch.as_tensor(motif_compressed, dtype=torch.bool), motif_group_id_full


class VoronoiClusterMotifScaffolding(MotifScaffolding):
    """Motif task whose groups are activated by cluster-aware training prepare.

    Catalytic vs structural potentials are assigned by segment-ranked quota:
    multi-fragment Voronoi cells are eligible, ``Binomial(n_eligible,
    p_catalytic)`` selects how many become catalytic, and the highest-segment
    eligible clusters are attempted first.
    """

    name: str = "voronoi_cluster_motif_scaffolding"

    def __init__(
        self,
        prob=0.0,
        t_sched="lognorm",
        mode="mixed",
        lognorm_mu=0.0,
        lognorm_sig=1.0,
        beta_p1=1.9,
        beta_p2=1.0,
        shift_time_scale=False,
        t_min=0.01,
        t_max=0.99,
        p_is_unindexed=1.0,
        name_override=None,
        p_fix_tip=0.0,
        p_fix_bb=0.0,
        p_fix_all=1.0,
        multi_motif_se3: bool = True,
        multi_motif_se3_trans_std: float = 14.0,
        motif_t_lock: float | None = 0.5,
        cluster_target_res: int = 32,
        cluster_max_res: int = 40,
        p_catalytic: float = 0.35,
        catalytic_min_res: int = 3,
        catalytic_max_res: int = 5,
        catalytic_max_ca_dist: float = 12.0,
        catalytic_distance_temperature: float = 2.5,
        catalytic_min_cluster_segments: int = 2,
        structural_keep_fraction_min: float = 0.5,
        structural_keep_fraction_max: float = 1.0,
        min_motif_groups: int = 2,
        max_motif_groups: int = 5,
        p_motif_cluster: float = 0.5,
        min_cluster_sequence_contacts: int = 1,
    ):
        cluster_target_res = int(cluster_target_res)
        cluster_max_res = int(cluster_max_res)
        catalytic_min_res = int(catalytic_min_res)
        catalytic_max_res = int(catalytic_max_res)
        catalytic_min_cluster_segments = int(catalytic_min_cluster_segments)
        min_motif_groups = int(min_motif_groups)
        max_motif_groups = int(max_motif_groups)
        min_cluster_sequence_contacts = int(min_cluster_sequence_contacts)
        if cluster_target_res < 1 or cluster_max_res < cluster_target_res:
            raise ValueError(
                "Require 1 <= cluster_target_res <= cluster_max_res"
            )
        if catalytic_min_res < 1 or catalytic_max_res < catalytic_min_res:
            raise ValueError(
                "Require 1 <= catalytic_min_res <= catalytic_max_res"
            )
        if float(catalytic_max_ca_dist) <= 0.0:
            raise ValueError("catalytic_max_ca_dist must be > 0")
        if float(catalytic_distance_temperature) <= 0.0:
            raise ValueError("catalytic_distance_temperature must be > 0")
        if catalytic_min_cluster_segments < 2:
            raise ValueError("catalytic_min_cluster_segments must be >= 2")
        if not (
            0.0
            < float(structural_keep_fraction_min)
            <= float(structural_keep_fraction_max)
            <= 1.0
        ):
            raise ValueError(
                "Require 0 < structural_keep_fraction_min "
                "<= structural_keep_fraction_max <= 1"
            )
        if min_motif_groups < 0 or max_motif_groups < min_motif_groups:
            raise ValueError(
                "Require 0 <= min_motif_groups <= max_motif_groups"
            )
        if not 0.0 <= float(p_catalytic) <= 1.0:
            raise ValueError("p_catalytic must be in [0, 1]")
        if not 0.0 <= float(p_motif_cluster) <= 1.0:
            raise ValueError("p_motif_cluster must be in [0, 1]")
        if min_cluster_sequence_contacts < 1:
            raise ValueError("min_cluster_sequence_contacts must be >= 1")
        if not multi_motif_se3:
            raise ValueError("VoronoiClusterMotifScaffolding requires multi_motif_se3")

        super().__init__(
            prob=prob,
            t_sched=t_sched,
            mode=mode,
            lognorm_mu=lognorm_mu,
            lognorm_sig=lognorm_sig,
            beta_p1=beta_p1,
            beta_p2=beta_p2,
            shift_time_scale=shift_time_scale,
            t_min=t_min,
            t_max=t_max,
            max_frac_res=1.0,
            max_num_res=cluster_max_res * max_motif_groups,
            max_num_segments=max_motif_groups,
            p_is_unindexed=p_is_unindexed,
            name_override=name_override,
            p_fix_tip=p_fix_tip,
            p_fix_bb=p_fix_bb,
            p_fix_all=p_fix_all,
            multi_motif_se3=True,
            multi_motif_partition_mode="fragments",
            multi_motif_max_groups=max_motif_groups,
            multi_motif_se3_trans_std=multi_motif_se3_trans_std,
            motif_t_lock=motif_t_lock,
        )
        self.cluster_target_res = cluster_target_res
        self.cluster_max_res = cluster_max_res
        self.p_catalytic = float(p_catalytic)
        self.catalytic_min_res = catalytic_min_res
        self.catalytic_max_res = catalytic_max_res
        self.catalytic_max_ca_dist = float(catalytic_max_ca_dist)
        self.catalytic_distance_temperature = float(catalytic_distance_temperature)
        self.catalytic_min_cluster_segments = catalytic_min_cluster_segments
        self.structural_keep_fraction_min = float(structural_keep_fraction_min)
        self.structural_keep_fraction_max = float(structural_keep_fraction_max)
        self.min_groups = min_motif_groups
        self.max_groups = max_motif_groups
        self.p_motif_cluster = float(p_motif_cluster)
        self.min_cluster_sequence_contacts = min_cluster_sequence_contacts

    def generate_motif_mask_and_groups(
        self, N, np_rng=None, *, data=None, resolved_mask=None, num_res=None
    ):
        del N, np_rng, data, resolved_mask, num_res
        raise RuntimeError(
            "VoronoiClusterMotifScaffolding requires cluster-aware training prepare"
        )

    def sample_activated_t_and_mask(
        self,
        data,
        motif_group_id_residue,
        *,
        group_shortfall: bool,
        np_rng=None,
        torch_generator=None,
    ):
        task_data = super().sample_t_and_mask(
            data,
            np_rng=np_rng,
            torch_generator=torch_generator,
            preset_motif_group_id_residue=motif_group_id_residue,
        )
        task_data["voronoi_group_shortfall"] = bool(group_shortfall)
        task_data["voronoi_activated_group_count"] = int(
            np.unique(
                np.asarray(motif_group_id_residue)[
                    np.asarray(motif_group_id_residue) > 0
                ]
            ).size
        )
        return task_data

    def max_motif_groups(self) -> int:
        return self.max_groups


class DistalSeparatedMotifScaffolding(MotifScaffolding):
    """Spatially separated motif groups: catalytic shell or distal structural segments.

    Pipeline:
      1. Sample group centers via farthest-point sampling with ``min_center_dist``.
      2. Per group, Bernoulli(``p_catalytic``) chooses catalytic vs structural.
      3. Catalytic: small shell-sampled cluster (``catalytic_min_res``..``catalytic_max_res``).
      4. Structural distal: 1..``max_structural_fragments`` contiguous windows whose
         anchors lie on the same chain with pairwise sequence gap ``>= min_seq_gap``.
      5. Residues of different groups stay at least ``min_inter_group_ca_dist`` apart.

    Hydra example::

        - _target_: nomadflow.runtime.training.motif_scaffold.DistalSeparatedMotifScaffolding
          prob: 0.8
          max_groups: 4
          min_groups: 1
          min_center_dist: 18.0
          min_inter_group_ca_dist: 12.0
          p_catalytic: 0.35
          multi_motif_se3: true
          motif_t_lock: 0.5
    """

    name: str = "distal_separated_motif_scaffolding"

    def __init__(
        self,
        prob=0.0,
        t_sched="lognorm",
        mode="mixed",
        lognorm_mu=0.0,
        lognorm_sig=1.0,
        beta_p1=1.9,
        beta_p2=1.0,
        shift_time_scale=False,
        t_min=0.01,
        t_max=0.99,
        p_is_unindexed=1.0,
        name_override=None,
        p_fix_tip=0.35,
        p_fix_bb=0.35,
        p_fix_all=0.30,
        multi_motif_se3: bool = True,
        multi_motif_se3_trans_std: float = 14.0,
        motif_t_lock: float | None = 0.5,
        min_groups: int = 1,
        max_groups: int = 4,
        min_center_dist: float = 18.0,
        min_inter_group_ca_dist: float = 12.0,
        p_catalytic: float = 0.35,
        catalytic_min_res: int = 3,
        catalytic_max_res: int = 5,
        small_fragment_shell_center_dist: float = 10.0,
        small_fragment_shell_half_width: float = 3.0,
        max_structural_fragments: int = 3,
        min_seq_gap: int = 30,
        structural_anchor_max_ca_dist: float = 20.0,
        min_fragment_size: int = 3,
        min_structural_res: int = 6,
        max_structural_res: int = 20,
        max_anchor_tries: int = 32,
    ):
        min_groups = int(min_groups)
        max_groups = int(max_groups)
        if min_groups < 1:
            raise ValueError(f"min_groups must be >= 1, got {min_groups}")
        if max_groups < min_groups:
            raise ValueError(
                f"max_groups ({max_groups}) must be >= min_groups ({min_groups})"
            )
        max_structural_res = int(max_structural_res)
        super().__init__(
            prob=prob,
            t_sched=t_sched,
            mode=mode,
            lognorm_mu=lognorm_mu,
            lognorm_sig=lognorm_sig,
            beta_p1=beta_p1,
            beta_p2=beta_p2,
            shift_time_scale=shift_time_scale,
            t_min=t_min,
            t_max=t_max,
            max_frac_res=1.0,
            max_num_res=max(max_structural_res * max_groups, catalytic_max_res * max_groups),
            max_num_segments=int(max_structural_fragments),
            p_is_unindexed=p_is_unindexed,
            name_override=name_override,
            p_fix_tip=p_fix_tip,
            p_fix_bb=p_fix_bb,
            p_fix_all=p_fix_all,
            multi_motif_se3=multi_motif_se3,
            multi_motif_partition_mode="fragments",
            multi_motif_max_groups=max_groups,
            multi_motif_se3_trans_std=multi_motif_se3_trans_std,
            motif_t_lock=motif_t_lock,
        )
        self.min_groups = min_groups
        self.max_groups = max_groups
        self.min_center_dist = float(min_center_dist)
        self.min_inter_group_ca_dist = float(min_inter_group_ca_dist)
        self.p_catalytic = float(p_catalytic)
        self.catalytic_min_res = int(catalytic_min_res)
        self.catalytic_max_res = int(catalytic_max_res)
        self.small_fragment_shell_center_dist = float(small_fragment_shell_center_dist)
        self.small_fragment_shell_half_width = float(small_fragment_shell_half_width)
        self.max_structural_fragments = int(max_structural_fragments)
        self.min_seq_gap = int(min_seq_gap)
        self.structural_anchor_max_ca_dist = float(structural_anchor_max_ca_dist)
        self.min_fragment_size = int(min_fragment_size)
        self.min_structural_res = int(min_structural_res)
        self.max_structural_res = max_structural_res
        self.max_anchor_tries = int(max_anchor_tries)

    def generate_motif_mask(
        self, N, np_rng=None, *, data=None, resolved_mask=None, num_res=None
    ):
        mask, _ = self.generate_motif_mask_and_groups(
            N,
            np_rng=np_rng,
            data=data,
            resolved_mask=resolved_mask,
            num_res=num_res,
        )
        return mask

    def _sample_catalytic_group(
        self,
        data,
        center_idx: int,
        pool: np.ndarray,
        used_mask: np.ndarray,
        blocked: np.ndarray,
        *,
        asym_ids: np.ndarray,
        ca_coords: np.ndarray,
        np_rng,
    ) -> np.ndarray:
        size = int(
            _rng_randint(np_rng, self.catalytic_min_res, self.catalytic_max_res + 1)
        )
        exclude = used_mask | blocked
        # Center must remain selectable even if it sits in a soft blocked zone from
        # planning; used residues are hard-excluded.
        if used_mask[center_idx]:
            return np.zeros(0, dtype=np.int32)
        exclude = exclude.copy()
        exclude[center_idx] = False
        selected = _spatial_fragment_residues(
            data,
            center_idx,
            pool,
            exclude,
            size,
            max_ca_dist=None,
            asym_ids=asym_ids,
            ca_coords=ca_coords,
            np_rng=np_rng,
            shell_sampling=True,
            shell_center_dist=self.small_fragment_shell_center_dist,
            shell_half_width=self.small_fragment_shell_half_width,
        )
        if selected.size < self.catalytic_min_res:
            return np.zeros(0, dtype=np.int32)
        # Drop any residue that still violates inter-group separation (except center
        # which we always keep if present).
        keep = []
        for idx in selected:
            idx = int(idx)
            if idx == int(center_idx) or not blocked[idx]:
                keep.append(idx)
        selected = np.asarray(keep, dtype=np.int32)
        if selected.size < self.catalytic_min_res:
            return np.zeros(0, dtype=np.int32)
        return selected

    def _sample_structural_anchors(
        self,
        center_idx: int,
        num_fragments: int,
        pool: np.ndarray,
        used_mask: np.ndarray,
        blocked: np.ndarray,
        *,
        asym_ids: np.ndarray,
        ca_coords: np.ndarray,
        np_rng,
    ) -> np.ndarray:
        chain_asym = int(asym_ids[center_idx])
        chain_order = _chain_residue_order(pool, asym_ids, chain_asym)
        if chain_order.size == 0 or int(center_idx) not in set(int(x) for x in chain_order):
            return np.zeros(0, dtype=np.int32)
        anchors = [int(center_idx)]
        if num_fragments <= 1:
            return np.asarray(anchors, dtype=np.int32)

        center_coord = ca_coords[center_idx]
        for _ in range(self.max_anchor_tries):
            if len(anchors) >= num_fragments:
                break
            available = []
            for idx in chain_order:
                idx = int(idx)
                if idx in anchors or used_mask[idx] or blocked[idx]:
                    continue
                if any(abs(idx - a) < self.min_seq_gap for a in anchors):
                    continue
                coord = ca_coords[idx]
                if np.isnan(coord).any() or np.isnan(center_coord).any():
                    continue
                if float(np.linalg.norm(coord - center_coord)) > self.structural_anchor_max_ca_dist:
                    continue
                available.append(idx)
            if not available:
                break
            anchors.append(int(np_rng.choice(np.asarray(available, dtype=np.int32))))
        return np.asarray(anchors, dtype=np.int32)

    def _sample_structural_group(
        self,
        center_idx: int,
        pool: np.ndarray,
        used_mask: np.ndarray,
        blocked: np.ndarray,
        *,
        asym_ids: np.ndarray,
        ca_coords: np.ndarray,
        np_rng,
        num_res: int | None = None,
    ) -> np.ndarray:
        forbidden = used_mask | blocked
        if used_mask[center_idx]:
            return np.zeros(0, dtype=np.int32)
        forbidden = forbidden.copy()
        forbidden[center_idx] = False

        num_fragments = int(
            _rng_randint(np_rng, 1, self.max_structural_fragments + 1)
        )
        if num_res is not None:
            group_budget = min(int(num_res), self.max_structural_res)
        else:
            lo = max(self.min_structural_res, self.min_fragment_size)
            hi = max(lo, self.max_structural_res)
            group_budget = int(_rng_randint(np_rng, lo, hi + 1))
        group_budget = max(group_budget, self.min_fragment_size)

        anchors = self._sample_structural_anchors(
            center_idx,
            num_fragments,
            pool,
            used_mask,
            blocked,
            asym_ids=asym_ids,
            ca_coords=ca_coords,
            np_rng=np_rng,
        )
        if anchors.size == 0:
            return np.zeros(0, dtype=np.int32)
        num_fragments = int(anchors.size)

        fragment_sizes = _sample_positive_partition(group_budget, num_fragments, np_rng)
        fragment_sizes = _apply_fragment_size_limits(
            fragment_sizes,
            group_budget,
            min_fragment_size=self.min_fragment_size,
            max_fragment_size=None,
        )

        chain_asym = int(asym_ids[center_idx])
        chain_order = _chain_residue_order(pool, asym_ids, chain_asym)
        selected_parts: list[np.ndarray] = []
        local_forbidden = forbidden.copy()

        for anchor, frag_size in zip(anchors, fragment_sizes):
            window = _expand_contiguous_window(
                chain_order,
                int(anchor),
                int(frag_size),
                local_forbidden,
            )
            if window.size < self.min_fragment_size:
                continue
            selected_parts.append(window)
            local_forbidden[window] = True

        if not selected_parts:
            return np.zeros(0, dtype=np.int32)
        return np.unique(np.concatenate(selected_parts)).astype(np.int32)

    def generate_motif_mask_and_groups(
        self, N, np_rng=None, *, data=None, resolved_mask=None, num_res=None
    ):
        del N
        if data is None or resolved_mask is None:
            raise ValueError("DistalSeparatedMotifScaffolding requires data and resolved_mask")
        np_rng = resolve_np_rng(np_rng)
        resolved_mask = np.asarray(resolved_mask, dtype=bool)
        pool = _protein_standard_residue_indices(data, resolved_mask)
        num_residues = resolved_mask.shape[0]
        empty_mask = np.zeros(num_residues, dtype=bool)
        empty_groups = np.zeros(num_residues, dtype=np.int32)
        if pool.size == 0:
            return torch.as_tensor(empty_mask[resolved_mask], dtype=torch.bool), empty_groups

        asym_ids, ca_coords = build_residue_spatial_lookup(data)
        desired_groups = int(
            _rng_randint(np_rng, self.min_groups, self.max_groups + 1)
        )
        centers = _fps_centers(
            pool,
            ca_coords,
            desired_groups,
            self.min_center_dist,
            np_rng,
        )
        if centers.size < self.min_groups:
            return torch.as_tensor(empty_mask[resolved_mask], dtype=torch.bool), empty_groups

        motif_full = np.zeros(num_residues, dtype=bool)
        motif_group_id_full = np.zeros(num_residues, dtype=np.int32)
        used_mask = np.zeros(num_residues, dtype=bool)
        blocked = np.zeros(num_residues, dtype=bool)
        accepted_group_residues: list[np.ndarray] = []

        # Shuffle center processing order for type diversity without bias to FPS order.
        center_order = np.asarray(centers, dtype=np.int32).copy()
        np_rng.shuffle(center_order)

        next_group_id = 1
        for center_idx in center_order:
            center_idx = int(center_idx)
            if used_mask[center_idx]:
                continue
            is_catalytic = bool(np_rng.random() < self.p_catalytic)
            if is_catalytic:
                selected = self._sample_catalytic_group(
                    data,
                    center_idx,
                    pool,
                    used_mask,
                    blocked,
                    asym_ids=asym_ids,
                    ca_coords=ca_coords,
                    np_rng=np_rng,
                )
            else:
                selected = self._sample_structural_group(
                    center_idx,
                    pool,
                    used_mask,
                    blocked,
                    asym_ids=asym_ids,
                    ca_coords=ca_coords,
                    np_rng=np_rng,
                    num_res=num_res,
                )
            if selected.size == 0:
                continue

            # Reject if this group violates separation against already accepted groups.
            ok = True
            for prev in accepted_group_residues:
                if (
                    _min_pairwise_ca_between_sets(ca_coords, selected, prev)
                    < self.min_inter_group_ca_dist
                ):
                    ok = False
                    break
            if not ok:
                continue

            group_id = next_group_id
            next_group_id += 1
            motif_full[selected] = True
            motif_group_id_full[selected] = group_id
            used_mask[selected] = True
            accepted_group_residues.append(selected)
            blocked |= _mask_within_ca_dist_of(
                ca_coords,
                selected,
                self.min_inter_group_ca_dist,
            )

        # Final sanity: drop later groups that somehow violate separation.
        if len(accepted_group_residues) >= 2:
            keep_flags = [True] * len(accepted_group_residues)
            for i in range(len(accepted_group_residues)):
                if not keep_flags[i]:
                    continue
                for j in range(i + 1, len(accepted_group_residues)):
                    if not keep_flags[j]:
                        continue
                    d = _min_pairwise_ca_between_sets(
                        ca_coords,
                        accepted_group_residues[i],
                        accepted_group_residues[j],
                    )
                    if d < self.min_inter_group_ca_dist:
                        keep_flags[j] = False
            if not all(keep_flags):
                motif_full[:] = False
                motif_group_id_full[:] = 0
                new_id = 1
                accepted_group_residues = [
                    res for keep, res in zip(keep_flags, accepted_group_residues) if keep
                ]
                for res in accepted_group_residues:
                    motif_full[res] = True
                    motif_group_id_full[res] = new_id
                    new_id += 1

        if len(accepted_group_residues) < self.min_groups:
            return torch.as_tensor(empty_mask[resolved_mask], dtype=torch.bool), empty_groups

        motif_compressed = motif_full[resolved_mask]
        return torch.as_tensor(motif_compressed, dtype=torch.bool), motif_group_id_full

    def max_added_tokens(self, N):
        del N
        return max(
            self.max_structural_res * self.max_groups,
            self.catalytic_max_res * self.max_groups,
        )

    def max_motif_groups(self) -> int:
        if not self.multi_motif_se3:
            return 0
        return self.max_groups


class BackboneMotifScaffolding(MotifScaffolding):
    name: str = "bb_motif_scaffolding"

    __init__ = partialmethod(MotifScaffolding.__init__, mode='backbone')


class ResidueMotifScaffolding(MotifScaffolding):
    name: str = "res_motif_scaffolding"
    __init__ = partialmethod(MotifScaffolding.__init__, mode='full_residue')


class InverseRotamerMotifScaffolding(MotifScaffolding):
    name: str = "inv_rot_motif_scaffolding"

    __init__ = partialmethod(MotifScaffolding.__init__, mode='inv_rotamer')


class MixedMotifScaffolding(MotifScaffolding):
    name: str = "mixed_motif_scaffolding"
    __init__ = partialmethod(MotifScaffolding.__init__, mode='mixed')


# inspired by Genie2
class MinimalMotifScaffolding(TrainingTask):
    name: str = "min_motif_scaffolding"
    def __init__(self,
                 prob=0.0,
                 t_sched='lognorm',
                 mode='mixed',
                 lognorm_mu=0.0,
                 lognorm_sig=1.0,
                 beta_p1=1.9,
                 beta_p2=1.0,
                 shift_time_scale=False,
                 t_min=0.01,
                 t_max=0.99,
                 max_res_islands=8,
                 max_res_per_island=3,
                 p_is_unindexed=0.8,
                 name_override=None,
                 p_fix_tip=0.35,
                 p_fix_bb=0.35,
                 p_fix_all=0.30,
                 multi_motif_se3: bool = False,
                 multi_motif_partition_mode: str = "fragments",
                 multi_motif_max_groups: int = 8,
                 multi_motif_se3_trans_std: float = 16.0,
                 motif_t_lock: float | None = None,
    ):
        assert t_sched in ['lognorm', 'mixed_beta', 'uniform']
        assert mode in ['mixed']
        assert multi_motif_partition_mode in PARTITION_MODES
        self.prob = prob
        self.t_sched = t_sched
        self.mode = mode
        self.lognorm_mu = lognorm_mu
        self.lognorm_sig = lognorm_sig
        self.beta_p1 = beta_p1
        self.beta_p2 = beta_p2
        self.t_min = t_min
        self.t_max = t_max
        self.shift_time_scale = shift_time_scale

        self.max_res_islands = max_res_islands
        self.max_res_per_island = max_res_per_island
        self.p_is_unindexed = p_is_unindexed

        self.p_fix_tip = p_fix_tip
        self.p_fix_bb = p_fix_bb
        self.p_fix_all = p_fix_all
        assert np.isclose(sum([p_fix_tip, p_fix_bb, p_fix_all]), 1.0)

        self.multi_motif_se3 = multi_motif_se3
        self.multi_motif_partition_mode = multi_motif_partition_mode
        self.multi_motif_max_groups = int(multi_motif_max_groups)
        self.multi_motif_se3_trans_std = float(multi_motif_se3_trans_std)
        self.motif_t_lock = motif_t_lock

        if name_override is not None:
            self.name = name_override

    def generate_motif_mask(
        self, N, np_rng=None, *, data=None, resolved_mask=None, num_res=None
    ):
        del data, resolved_mask, num_res
        np_rng = resolve_np_rng(np_rng)
        num_segments = np_rng.randint(1, self.max_res_islands+1)
        # we switch this from Genie2
        # to increase the probability of sampling minimal motifs
        # we also give the option to lower the maximum number of residues sampled
        # num_res = np.random.randint(math.floor(0.05 * N), math.ceil(0.5 * N) + 1)
        # # when N < 80 it's possible for num_segments > num_res
        # num_res = max(num_segments + 1, num_res)
        L = np_rng.choice(self.max_res_per_island, size=num_segments) + 1
        num_res = np.sum(L)
        permute_list = [[0] for _ in range(N - num_res)] + [np.ones((l,)) for l in L]
        permutation = np_rng.permutation(len(permute_list))
        permuted_list = [permute_list[i] for i in permutation]
        M = np.concatenate(permuted_list, axis=0)
        return torch.as_tensor(M, dtype=torch.bool)

    def generate_motif_mask_and_groups(
        self, N, np_rng=None, *, data=None, resolved_mask=None, num_res=None
    ):
        del data, resolved_mask, num_res
        return self.generate_motif_mask(N, np_rng=np_rng), None

    def _motif_group_id_from_generation(
        self,
        data,
        residue_noising_mask: np.ndarray,
        motif_mask_compressed: np.ndarray,
        resolved_mask: np.ndarray,
        np_rng,
        *,
        preset_motif_group_id_residue: np.ndarray | None,
    ) -> np.ndarray:
        if preset_motif_group_id_residue is not None:
            out = np.asarray(preset_motif_group_id_residue, dtype=np.int32).copy()
            if not self.multi_motif_se3:
                out.fill(0)
        else:
            out = build_motif_group_id_by_residue(
                data,
                residue_noising_mask,
                motif_mask_compressed,
                resolved_mask,
                enabled=self.multi_motif_se3,
                partition_mode=self.multi_motif_partition_mode,
                max_groups=self.multi_motif_max_groups,
                np_rng=np_rng,
            )
        out[residue_noising_mask.astype(bool)] = 0
        out[~resolved_mask] = 0
        return out

    def sample_t_and_mask(self, data, np_rng=None, torch_generator=None):
        residues = data.residues
        atoms = data.atoms

        np_rng = resolve_np_rng(np_rng)
        t = sample_task_t(
            t_sched=self.t_sched,
            lognorm_mu=self.lognorm_mu,
            lognorm_sig=self.lognorm_sig,
            beta_p1=self.beta_p1,
            beta_p2=self.beta_p2,
            t_min=self.t_min,
            t_max=self.t_max,
            device='cpu',
            np_rng=np_rng if np_rng is not np.random else None,
            torch_generator=torch_generator,
        )

        resolved_mask = data.residues['is_present']
        num_resolved_residues = int(resolved_mask.sum())
        motif_mask, preset_motif_group_id_residue = self.generate_motif_mask_and_groups(
            num_resolved_residues,
            np_rng=np_rng,
            data=data,
            resolved_mask=resolved_mask,
        )
        residue_noising_mask = np.ones_like(resolved_mask)
        residue_noising_mask[resolved_mask] = ~motif_mask
        res_type_noising_mask = residue_noising_mask.copy()

        atom_noising_mask = []
        motif_frame_mask_by_residue = np.zeros((resolved_mask.shape[0], 3), dtype=bool)
        if self.mode == 'mixed':
            noise_select = torch.rand(
                torch.as_tensor(residue_noising_mask).shape,
                dtype=torch.float32,
                generator=torch_generator,
            ).numpy(force=True)
            fix_tip = noise_select < self.p_fix_tip
            noise_sidechain_only = noise_select > (1 - self.p_fix_bb)
            res_type_noising_mask[~residue_noising_mask & fix_tip] = False
            res_type_noising_mask[~residue_noising_mask & noise_sidechain_only] = True

            for chain in data.chains:
                # Get residue indices
                res_start = chain["res_idx"]
                res_end = chain["res_idx"] + chain["res_num"]
                is_protein = chain["mol_type"] == const.chain_type_ids["PROTEIN"]

                for i, residue in enumerate(data.residues[res_start:res_end]):
                    res_pos = res_start + i
                    atom_idx = residue['atom_idx']
                    atom_num = residue['atom_num']
                    atoms = data.atoms[atom_idx:atom_idx+atom_num]
                    if residue["is_standard"] and (residue['name'] != 'UNK') and is_protein:
                        is_motif = ~residue_noising_mask[res_pos]
                        frame_mask = motif_frame_mask_from_selection(
                            fix_tip[res_pos],
                            noise_sidechain_only[res_pos],
                            is_motif,
                        )
                        motif_frame_mask_by_residue[res_pos] = frame_mask
                        if self.multi_motif_se3:
                            rigid_noise_mask = [True, True, True]
                        else:
                            rigid_noise_mask = (~frame_mask).tolist()
                        atom_noising_mask.append(rigid_noise_to_atom_noise(residue, atoms, rigid_noise_mask))
                    else:
                        atom_noising_mask.append([True for _ in atoms])

        else:
            raise ValueError()
        atom_noising_mask = np.concatenate(atom_noising_mask)

        is_unindexed_residue = torch.rand(
            torch.as_tensor(residue_noising_mask).shape,
            dtype=torch.float32,
            generator=torch_generator,
        ) < self.p_is_unindexed
        is_unindexed_residue = is_unindexed_residue.numpy(force=True)

        copy_indexed_residue_mask = ~residue_noising_mask & ~is_unindexed_residue
        copy_unindexed_residue_mask = ~residue_noising_mask & is_unindexed_residue
        copy_atomized_residue_mask = np.zeros_like(res_type_noising_mask)

        motif_mask_np = motif_mask.numpy(force=True).astype(bool, copy=False)
        motif_group_id_residue = self._motif_group_id_from_generation(
            data,
            residue_noising_mask,
            motif_mask_np,
            resolved_mask,
            np_rng,
            preset_motif_group_id_residue=preset_motif_group_id_residue,
        )
        motif_group_id_rigid_by_residue_frame = (
            motif_frame_mask_by_residue.astype(np.int32)
            * motif_group_id_residue[:, None].astype(np.int32)
        )

        return {
            "t": t.numpy(force=True),
            "atom_noising_mask": atom_noising_mask,
            "res_type_noising_mask": res_type_noising_mask,
            "copy_indexed_residue_mask": copy_indexed_residue_mask,
            "copy_unindexed_residue_mask": copy_unindexed_residue_mask,
            "copy_atomized_residue_mask": copy_atomized_residue_mask,
            "motif_group_id_residue": motif_group_id_residue,
            "motif_group_id_rigid_by_residue_frame": motif_group_id_rigid_by_residue_frame,
            "multi_motif_se3": self.multi_motif_se3,
            "multi_motif_se3_trans_std": self.multi_motif_se3_trans_std,
            "motif_t_lock": self.motif_t_lock,
        }

    def max_added_tokens(self, N):
        res_cap = self.max_res_per_island * self.max_res_islands
        return res_cap

    def max_motif_groups(self) -> int:
        """Return an upper bound for virtual motif-group aggregators."""
        if not self.multi_motif_se3:
            return 0
        return self.multi_motif_max_groups
