"""Apply independent random SE(3) transforms per motif group to tokenized rigid geometry."""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import nomadflow.openfold.utils.rigid_utils as ru


def _sample_se3_tensor7(np_rng, trans_std: float) -> np.ndarray:
    quat = Rotation.random(random_state=np_rng).as_quat(canonical=True).astype(np.float32)
    trans = (np.asarray(np_rng.standard_normal(3), dtype=np.float32) * trans_std).astype(np.float32)
    return np.concatenate([quat, trans], axis=-1)


def apply_multi_motif_se3_augmentation(
    token_data: np.ndarray,
    rigid_data: np.ndarray,
    *,
    enabled: bool,
    np_rng,
    trans_std: float,
) -> None:
    """In-place: apply one random SE(3) per rigid-level motif group."""
    if not enabled:
        return

    groups: dict[int, list[int]] = {}
    moved_token_ids: set[int] = set()
    for rigid_index in range(len(rigid_data)):
        rigid = rigid_data[rigid_index]
        group_id = int(rigid["motif_group_id"])
        if group_id <= 0:
            continue
        groups.setdefault(group_id, []).append(rigid_index)
        moved_token_ids.add(int(rigid["token_idx"]))

    if not groups:
        return

    device = torch.device("cpu")
    for group_id in sorted(groups.keys()):
        group_indices = groups[group_id]
        group_center = torch.as_tensor(
            rigid_data["tensor7"][group_indices, 4:].mean(axis=0),
            device=device,
            dtype=torch.float32,
        )
        group_transform = _sample_se3_tensor7(np_rng, trans_std)
        world_rigid = ru.Rigid.from_tensor_7(
            torch.as_tensor(group_transform, device=device).unsqueeze(0)
        )

        for rigid_index in group_indices:
            rigid_tensor7 = rigid_data["tensor7"][rigid_index].astype(np.float32, copy=False)
            frame_rigid = ru.Rigid.from_tensor_7(
                torch.as_tensor(rigid_tensor7, device=device).unsqueeze(0)
            )
            centered_frame = ru.Rigid(
                rots=frame_rigid.get_rots(),
                trans=frame_rigid.get_trans() - group_center,
            )
            composed = world_rigid.compose(centered_frame)
            composed_tensor7 = composed.to_tensor_7().squeeze(0)
            rigid_data["tensor7"][rigid_index] = composed_tensor7.numpy().astype(np.float32)

    for token in token_data:
        if int(token["token_idx"]) in moved_token_ids and int(token["rigid_num"]) >= 1:
            first_rigid_index = int(token["rigid_idx"])
            token["center_coords"] = rigid_data["tensor7"][first_rigid_index, 4:].copy()
