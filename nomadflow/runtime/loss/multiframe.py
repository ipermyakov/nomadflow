import torch

from nomadflow.openfold.utils import rigid_utils as ru
from nomadflow.stoch_interp import so3_utils as so3_fm_utils
from nomadflow.stoch_interp.motif_time_schedule import (
    LinearMotifTimeSchedule,
    MOTIF_T_LOCK_DISABLED,
    effective_rigids_noising_mask,
    motif_t_lock_is_enabled,
)


def angle_axis_rot_vf_loss_dense(
        pred_rot_vf,
        ref_rot_vf,
        rigids_mask,
        rigidwise_norm_scale,
        angle_loss_weight=0.5,
        weight=None,
        eps=1e-8):
    rot_loss = angle_axis_rot_vf_loss_rigid(
        pred_rot_vf,
        ref_rot_vf,
        rigidwise_norm_scale,
        angle_loss_weight=angle_loss_weight,
        weight=weight,
        eps=eps,
    )

    num_rigids_per_batch = rigids_mask.long().sum(dim=-1).clip(min=1)
    rot_loss = (rot_loss * rigids_mask).sum(dim=-1) / num_rigids_per_batch

    return rot_loss


def angle_axis_rot_vf_loss_rigid(
        pred_rot_vf,
        ref_rot_vf,
        rigidwise_norm_scale,
        angle_loss_weight=0.5,
        weight=None,
        eps=1e-8):
    pred_rot_vf = pred_rot_vf / rigidwise_norm_scale[..., None]
    ref_rot_vf = ref_rot_vf / rigidwise_norm_scale[..., None]

    gt_rot_angle = torch.norm(ref_rot_vf, dim=-1, keepdim=True)
    gt_rot_axis = ref_rot_vf / (gt_rot_angle + eps)

    pred_rot_angle = torch.norm(pred_rot_vf, dim=-1, keepdim=True)
    pred_rot_axis = pred_rot_vf / (pred_rot_angle + eps)

    # Separate loss on the axis
    axis_loss = torch.square(gt_rot_axis - pred_rot_axis).sum(dim=-1)

    # Separate loss on the angle
    angle_loss = (gt_rot_angle - pred_rot_angle)**2
    angle_loss = torch.sum(
        angle_loss,
        dim=-1
    )
    rot_loss = angle_loss * angle_loss_weight + axis_loss

    if weight is not None:
        rot_loss = rot_loss * weight

    return rot_loss


def _masked_batch_mean(values, mask, denominator_mask=None):
    if denominator_mask is None:
        denominator_mask = mask
    denom = denominator_mask.long().sum(dim=-1).clip(min=1)
    return (values * mask).sum(dim=-1) / denom


def _motif_t_lock_loss_scale(inputs, batch_size, device, dtype):
    motif_t_lock = inputs.get("motif_t_lock")
    if motif_t_lock is None:
        return torch.ones(batch_size, device=device, dtype=dtype)

    t_lock = torch.as_tensor(motif_t_lock, device=device, dtype=dtype).reshape(-1)
    if t_lock.numel() == 1:
        t_lock = t_lock.expand(batch_size)
    elif t_lock.numel() != batch_size:
        raise ValueError(
            f"motif_t_lock must be scalar or length batch_size={batch_size}, got {tuple(t_lock.shape)}"
        )
    return torch.where(t_lock > 0, t_lock.reciprocal(), torch.ones_like(t_lock))


def _motif_time_norm_scale(inputs, t_norm_clip):
    motif_t_lock = inputs.get("motif_t_lock")
    t = inputs["t"]
    if motif_t_lock_is_enabled(motif_t_lock):
        motif_t = LinearMotifTimeSchedule().motif_interp_time(t, motif_t_lock)
        t_lock = torch.as_tensor(motif_t_lock, device=t.device, dtype=t.dtype)
        while t_lock.ndim < t.ndim:
            t_lock = t_lock.unsqueeze(-1)
        t = torch.where(t_lock > MOTIF_T_LOCK_DISABLED, motif_t, t)
    return 1 - torch.min(t, torch.as_tensor(t_norm_clip, device=t.device, dtype=t.dtype))


def _motif_flow_masks_and_weights(rigids_data, total_mask, motif_aggregator_weight=5.0):
    motif_group_id = rigids_data.get("motif_group_id")
    aggregator_mask = rigids_data.get("rigids_is_motif_group_aggregator")
    if motif_group_id is None:
        zeros = torch.zeros_like(total_mask, dtype=torch.bool)
        return total_mask, zeros, zeros, torch.ones_like(total_mask, dtype=torch.float32)

    group_id = motif_group_id.to(device=total_mask.device, dtype=torch.long)
    if aggregator_mask is None:
        aggregator_mask = torch.zeros_like(total_mask, dtype=torch.bool)
    else:
        aggregator_mask = aggregator_mask.to(device=total_mask.device, dtype=torch.bool)

    motif_mask = total_mask.bool() & (group_id > 0)
    aggregator_flow_mask = motif_mask & aggregator_mask
    motif_real_mask = motif_mask & ~aggregator_mask
    flow_mask = total_mask.bool() & ~motif_real_mask
    aggregator_weight = torch.ones_like(total_mask, dtype=torch.float32)

    if motif_aggregator_weight != 1.0 and bool(aggregator_flow_mask.any()):
        aggregator_weight = torch.where(
            aggregator_flow_mask,
            torch.full((), motif_aggregator_weight, device=total_mask.device, dtype=aggregator_weight.dtype),
            aggregator_weight,
        )

    return flow_mask, aggregator_flow_mask, motif_real_mask, aggregator_weight


def _active_aggregator_fafe(
    pred_frames,
    gt_frames,
    aggregator_flow_mask,
    framepair_weight,
    block_diag_size,
):
    aggregator_fafe = fafe_loss_l2(
        pred_frames=pred_frames,
        gt_frames=gt_frames,
        frame_mask=aggregator_flow_mask,
        framepair_weight=framepair_weight,
        block_diag_size=block_diag_size,
    )
    has_pair = aggregator_flow_mask.long().sum(dim=-1) >= 2
    return torch.where(has_pair, aggregator_fafe, torch.zeros_like(aggregator_fafe))


def _motif_intergroup_pair_mask(group_id, motif_real_mask):
    """Bool pair mask: both frames are real motif residues in different groups."""
    group_id = group_id.to(device=motif_real_mask.device, dtype=torch.long)
    gi = group_id.unsqueeze(-1)
    gj = group_id.unsqueeze(-2)
    mi = motif_real_mask.unsqueeze(-1)
    mj = motif_real_mask.unsqueeze(-2)
    return mi & mj & (gi > 0) & (gj > 0) & (gi != gj)


def _active_motif_intergroup_fafe(
    pred_frames,
    gt_frames,
    motif_real_mask,
    group_id,
    framepair_weight,
):
    """FAFE on real motif frames with pairs restricted to group_i != group_j."""
    if group_id is None:
        return torch.zeros(
            motif_real_mask.shape[0],
            device=motif_real_mask.device,
            dtype=pred_frames.get_trans().dtype,
        )

    pair_mask = _motif_intergroup_pair_mask(group_id, motif_real_mask.bool())
    motif_intergroup_fafe = fafe_loss_l2(
        pred_frames=pred_frames,
        gt_frames=gt_frames,
        frame_mask=motif_real_mask.bool(),
        framepair_weight=framepair_weight,
        block_diag_size=1,
        pair_mask=pair_mask,
    )
    has_pair = pair_mask.flatten(start_dim=-2).any(dim=-1)
    return torch.where(has_pair, motif_intergroup_fafe, torch.zeros_like(motif_intergroup_fafe))


def multiframe_fm_loss_dense_batch(
    inputs,
    denoiser_outputs,
    t_norm_clip=0.9,
    sep_rot_loss=True,
    use_euclidean_for_rots=False,
    rigidwise_weight=1,
    rot_vf_angle_loss_weight=0.5,
    fafe_l2_block_mask_size=1,
    trans_rigidwise_weight=1,
    rot_rigidwise_weight=1,
    direct_rot_vf_loss=False,
    direct_rot_vf_loss_scale=1,
    upweight_atomic=False,
    motif_aggregator_weight=5.0,
):
    rigids_data = inputs['rigids']
    rigids_mask = rigids_data['rigids_mask']
    rigids_noising_mask = effective_rigids_noising_mask(
        rigids_data['rigids_noising_mask'],
        rigids_data.get('motif_group_id'),
        inputs['t'],
        inputs.get('motif_t_lock'),
        schedule=LinearMotifTimeSchedule(),
    )
    noised_rigids = ru.Rigid.from_tensor_7(rigids_data['rigids_t'])

    total_mask = rigids_mask * rigids_noising_mask
    flow_mask, aggregator_flow_mask, motif_real_flow_mask, aggregator_weight = (
        _motif_flow_masks_and_weights(
            rigids_data, total_mask, motif_aggregator_weight=motif_aggregator_weight
        )
    )

    if upweight_atomic:
        rigid_is_atomized = inputs['rigids']['rigids_is_atom_mask'].float()
        atom_rigid_upweight = (1 - rigid_is_atomized) + 10 * (rigid_is_atomized)
        rigidwise_weight = rigidwise_weight * atom_rigid_upweight
    framepair_rigidwise_weight = rigidwise_weight
    rigidwise_weight = rigidwise_weight * aggregator_weight

    denoised_rigids = denoiser_outputs['denoised_rigids']
    gt_rigids = ru.Rigid.from_tensor_7(rigids_data['rigids_1'])

    num_rigids_per_batch = rigids_mask.long().sum(dim=-1).clip(min=1)

    gt_frame_trans = gt_rigids.get_trans()
    pred_frame_trans = denoised_rigids.get_trans()
    ref_frame_trans = noised_rigids.get_trans()
    pred_frame_trans_se = torch.square(gt_frame_trans - pred_frame_trans).sum(dim=-1)
    pred_frame_trans_mse = (pred_frame_trans_se * total_mask).sum(-1) / num_rigids_per_batch
    ref_frame_trans_se = torch.square(gt_frame_trans - ref_frame_trans).sum(dim=-1)
    ref_frame_trans_mse = (ref_frame_trans_se * total_mask).sum(-1) / num_rigids_per_batch

    t = inputs['t']
    norm_scale = 1 - torch.min(
        t, torch.as_tensor(t_norm_clip)
    )
    motif_norm_scale = _motif_time_norm_scale(inputs, t_norm_clip)
    rots_t = noised_rigids.get_rots().get_rot_mats()
    rots_1_pred = denoised_rigids.get_rots().get_rot_mats()
    rots_1 = gt_rigids.get_rots().get_rot_mats()

    if use_euclidean_for_rots:
        def geodesic_dist(rots1, rots2):
            R_diff = torch.einsum("...ij,...jk->...ik", rots1.transpose(-2, -1), rots2)
            R_diff_trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            return torch.acos(
                torch.clamp(
                    (R_diff_trace - 1) / 2,
                    min=-1 + 1e-6,
                    max=1 - 1e-6
                )
            )
        # pred_rotvec_1 = so3_fm_utils.rotmat_to_rotvec(rots_1_pred)
        # gt_rotvec_1 = so3_fm_utils.rotmat_to_rotvec(rots_1)
        # rot_vf_loss = torch.square(pred_rotvec_1 - gt_rotvec_1).sum(dim=-1)
        rot_vf_loss = (geodesic_dist(rots_1, rots_1_pred) / torch.pi) ** 2
        unscaled_rot_vf_loss = rot_vf_loss
        rot_vf_loss_rigid_for_agg = unscaled_rot_vf_loss / (motif_norm_scale ** 2)[..., None]
        rot_vf_loss = rot_vf_loss * rigidwise_weight
        rot_vf_loss = rot_vf_loss / (norm_scale[..., None] ** 2)
        rot_vf_loss = (rot_vf_loss * flow_mask).sum(dim=-1) / num_rigids_per_batch
        with torch.no_grad():
            diagnostic_rot_vf_loss = unscaled_rot_vf_loss / (norm_scale ** 2)[..., None]
            unscaled_rot_vf_loss = (diagnostic_rot_vf_loss * flow_mask).sum(dim=-1) / num_rigids_per_batch
            aggregator_rot_loss = _masked_batch_mean(
                diagnostic_rot_vf_loss,
                aggregator_flow_mask,
            )
            motif_real_rot_loss = _masked_batch_mean(
                diagnostic_rot_vf_loss,
                motif_real_flow_mask,
            )

    else:
        if direct_rot_vf_loss:
            pred_rot_vf = denoiser_outputs['pred_rot_vf']
            gt_rot_vf = rigids_data['gt_rot_vf']
            # print("pred", pred_rot_vf)
            # print("gt", gt_rot_vf)
            # rot_vf_angle_loss_weight = rot_vf_angle_loss_weight / (2.4 ** 2)  # this is roughly the mean of the vector field magnitudes, squared

            unscaled_rot_vf_loss = torch.square(pred_rot_vf - gt_rot_vf).sum(dim=-1)
            rot_vf_loss_rigid_for_agg = unscaled_rot_vf_loss
            rot_vf_loss = unscaled_rot_vf_loss * rot_rigidwise_weight * rigidwise_weight
            rot_vf_loss = torch.sum(rot_vf_loss * flow_mask, dim=-1) / total_mask.sum(dim=-1).clip(min=1)
            rot_vf_loss = rot_vf_loss * direct_rot_vf_loss_scale
            with torch.no_grad():
                diagnostic_rot_vf_loss = unscaled_rot_vf_loss
                unscaled_rot_vf_loss = torch.sum(unscaled_rot_vf_loss * flow_mask, dim=-1) / total_mask.sum(dim=-1).clip(min=1)
                aggregator_rot_loss = _masked_batch_mean(
                    diagnostic_rot_vf_loss,
                    aggregator_flow_mask,
                )
                motif_real_rot_loss = _masked_batch_mean(
                    diagnostic_rot_vf_loss,
                    motif_real_flow_mask,
                )

            # rot_vf_loss = angle_axis_rot_vf_loss_dense(
            #     pred_rot_vf,
            #     gt_rot_vf,
            #     total_mask,
            #     torch.ones_like(norm_scale) / rot_rigidwise_weight,
            #     weight=rigidwise_weight ,
            #     angle_loss_weight=rot_vf_angle_loss_weight,
            # )
            # with torch.no_grad():
            #     unscaled_rot_vf_loss = angle_axis_rot_vf_loss_dense(
            #         pred_rot_vf,
            #         gt_rot_vf,
            #         total_mask,
            #         torch.ones_like(norm_scale),
            #     )
        else:
            pred_rot_vf = so3_fm_utils.calc_rot_vf(rots_t, rots_1_pred)
            gt_rot_vf = so3_fm_utils.calc_rot_vf(rots_t, rots_1)

            if sep_rot_loss:
                rot_vf_loss_rigid_for_agg = angle_axis_rot_vf_loss_rigid(
                    pred_rot_vf,
                    gt_rot_vf,
                    motif_norm_scale,
                    angle_loss_weight=rot_vf_angle_loss_weight,
                )
                rot_vf_loss = angle_axis_rot_vf_loss_dense(
                    pred_rot_vf,
                    gt_rot_vf,
                    total_mask,
                    norm_scale / rot_rigidwise_weight,
                    weight=rigidwise_weight * flow_mask.float(),
                    angle_loss_weight=rot_vf_angle_loss_weight,
                )
                with torch.no_grad():
                    unscaled_rot_vf_loss = angle_axis_rot_vf_loss_dense(
                        pred_rot_vf,
                        gt_rot_vf,
                        total_mask,
                        norm_scale,
                        weight=flow_mask.float(),
                    )
                    aggregator_rot_loss = angle_axis_rot_vf_loss_dense(
                        pred_rot_vf,
                        gt_rot_vf,
                        aggregator_flow_mask,
                        norm_scale,
                    )
                    motif_real_rot_loss = angle_axis_rot_vf_loss_dense(
                        pred_rot_vf,
                        gt_rot_vf,
                        motif_real_flow_mask,
                        norm_scale,
                    )
            else:
                rot_vf_loss = torch.square(pred_rot_vf - gt_rot_vf).sum(dim=-1)
                unscaled_rot_vf_loss = rot_vf_loss
                diagnostic_rot_vf_loss = unscaled_rot_vf_loss / (norm_scale ** 2)[..., None]
                rot_vf_loss_rigid_for_agg = unscaled_rot_vf_loss / (motif_norm_scale ** 2)[..., None]
                rot_vf_loss = rot_vf_loss * rigidwise_weight
                rot_vf_loss = rot_vf_loss / (norm_scale[..., None] ** 2)
                rot_vf_loss = (rot_vf_loss * flow_mask).sum(dim=-1) / num_rigids_per_batch
                with torch.no_grad():
                    unscaled_rot_vf_loss = (diagnostic_rot_vf_loss * flow_mask).sum(dim=-1) / num_rigids_per_batch
                    aggregator_rot_loss = _masked_batch_mean(
                        diagnostic_rot_vf_loss,
                        aggregator_flow_mask,
                    )
                    motif_real_rot_loss = _masked_batch_mean(
                        diagnostic_rot_vf_loss,
                        motif_real_flow_mask,
                    )

    trans_1_pred = denoised_rigids.get_trans()
    trans_1 = gt_rigids.get_trans()
    trans_vf_loss = torch.square(trans_1_pred - trans_1).sum(dim=-1) / (norm_scale ** 2)
    unscaled_trans_vf_loss = trans_vf_loss
    unscaled_trans_vf_loss_rigid = unscaled_trans_vf_loss
    aggregator_trans_vf_loss_rigid = (
        torch.square(trans_1_pred - trans_1).sum(dim=-1) / (motif_norm_scale ** 2)
    )
    trans_vf_loss = trans_vf_loss * rigidwise_weight * trans_rigidwise_weight
    trans_vf_loss = (trans_vf_loss * flow_mask).sum(dim=-1) / num_rigids_per_batch
    trans_vf_loss *= 0.01  # Angstroms to nm
    unscaled_trans_vf_loss = (unscaled_trans_vf_loss * flow_mask).sum(dim=-1) / num_rigids_per_batch
    unscaled_trans_vf_loss *= 0.01  # Angstroms to nm

    aggregator_trans_loss = _masked_batch_mean(
        unscaled_trans_vf_loss_rigid * 0.01,
        aggregator_flow_mask,
    )
    motif_real_trans_loss = _masked_batch_mean(
        unscaled_trans_vf_loss_rigid * 0.01,
        motif_real_flow_mask,
    )
    aggregator_pose_loss = (
        _masked_batch_mean(aggregator_trans_vf_loss_rigid * 0.01, aggregator_flow_mask)
        + _masked_batch_mean(rot_vf_loss_rigid_for_agg, aggregator_flow_mask)
    )
    aggregator_pose_loss = aggregator_pose_loss * _motif_t_lock_loss_scale(
        inputs,
        batch_size=rigids_mask.shape[0],
        device=rigids_mask.device,
        dtype=unscaled_trans_vf_loss_rigid.dtype,
    )
    # torch.set_printoptions(threshold=1000001)
    # print(trans_1_pred, trans_1)

    if isinstance(framepair_rigidwise_weight, (int, float)):
        framepair_weight = framepair_rigidwise_weight
    else:
        framepair_weight = (
            framepair_rigidwise_weight[..., None]
            * framepair_rigidwise_weight[..., None, :]
        )

    fafe = fafe_loss_l2(
        pred_frames=denoised_rigids,
        gt_frames=gt_rigids,
        frame_mask=rigids_mask,
        framepair_weight=framepair_weight,
        block_diag_size=fafe_l2_block_mask_size
    )
    scaled_fafe = fafe / norm_scale.squeeze(-1)
    aggregator_fafe = _active_aggregator_fafe(
        pred_frames=denoised_rigids,
        gt_frames=gt_rigids,
        aggregator_flow_mask=aggregator_flow_mask,
        framepair_weight=framepair_weight,
        block_diag_size=1,
    )
    scaled_aggregator_fafe = aggregator_fafe / motif_norm_scale.squeeze(-1)
    scaled_aggregator_fafe = scaled_aggregator_fafe * _motif_t_lock_loss_scale(
        inputs,
        batch_size=rigids_mask.shape[0],
        device=rigids_mask.device,
        dtype=scaled_aggregator_fafe.dtype,
    )

    # Inter-group FAFE on present real motif residues (not aggregators): group_i != group_j.
    # Uses rigids_mask (not noising mask) so relative layout is scored whenever motifs are present.
    _, _, motif_real_present_mask, _ = _motif_flow_masks_and_weights(
        rigids_data, rigids_mask.bool(), motif_aggregator_weight=1.0
    )
    motif_intergroup_fafe = _active_motif_intergroup_fafe(
        pred_frames=denoised_rigids,
        gt_frames=gt_rigids,
        motif_real_mask=motif_real_present_mask,
        group_id=rigids_data.get("motif_group_id"),
        framepair_weight=framepair_weight,
    )
    scaled_motif_intergroup_fafe = motif_intergroup_fafe / motif_norm_scale.squeeze(-1)
    scaled_motif_intergroup_fafe = scaled_motif_intergroup_fafe * _motif_t_lock_loss_scale(
        inputs,
        batch_size=rigids_mask.shape[0],
        device=rigids_mask.device,
        dtype=scaled_motif_intergroup_fafe.dtype,
    )

    ret = {
        "rot_vf_loss": rot_vf_loss,
        "trans_vf_loss": trans_vf_loss,
        "unscaled_rot_vf_loss": unscaled_rot_vf_loss,
        "unscaled_trans_vf_loss": unscaled_trans_vf_loss,
        "pred_trans_mse": pred_frame_trans_mse,
        "ref_trans_mse": ref_frame_trans_mse,
        "fafe": fafe,
        "scaled_fafe": scaled_fafe,
        "aggregator_fafe": aggregator_fafe,
        "scaled_aggregator_fafe": scaled_aggregator_fafe,
        "motif_intergroup_fafe": motif_intergroup_fafe,
        "scaled_motif_intergroup_fafe": scaled_motif_intergroup_fafe,
        "aggregator_trans_loss": aggregator_trans_loss,
        "aggregator_rot_loss": aggregator_rot_loss,
        "aggregator_pose_loss": aggregator_pose_loss,
        "motif_real_trans_loss": motif_real_trans_loss,
        "motif_real_rot_loss": motif_real_rot_loss,
    }
    return ret


# adapted in part from https://github.com/mooninrain/FAFE/blob/main/losses/fafe.py
def fafe_loss_l2(
    pred_frames,
    gt_frames,
    frame_mask,
    framepair_weight = 1.,
    rot_scale: float = 1.0,
    trans_scale: float = 20.0,
    dist_clamp: float | None = 20.,
    eps_so3: float = 1e-6,
    block_diag_size=1,
    pair_mask=None,
):
    def geodesic_dist(rots1, rots2):
        R_diff = torch.einsum("...ij,...jk->...ik", rots1.transpose(-2, -1), rots2)
        R_diff_trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        return torch.acos(
            torch.clamp(
                (R_diff_trace - 1) / 2,
                min=-1 + eps_so3,
                max=1 - eps_so3
            )
        )

    gt_framepairs = gt_frames[..., None].invert().compose(gt_frames[..., None, :])
    pred_framepairs = pred_frames[..., None].invert().compose(pred_frames[..., None, :])
    mask = frame_mask[..., None] * frame_mask[..., None, :]
    # mask diagonal
    if block_diag_size == 1:
        mask = mask * (1 - torch.eye(frame_mask.shape[-1], device=mask.device))[None]
    else:
        assert mask.shape[-1] % block_diag_size == 0
        repeats = mask.shape[-1] // block_diag_size
        block_mask = torch.block_diag(*[torch.ones(block_diag_size, block_diag_size) for _ in range(repeats)])
        mask = mask * (1 - block_mask.to(mask.device))[None]

    if pair_mask is not None:
        mask = mask * pair_mask.to(device=mask.device, dtype=mask.dtype)

    trans_dist = torch.linalg.vector_norm(pred_framepairs.get_trans() - gt_framepairs.get_trans(), dim=-1)
    clamp_mask = trans_dist > dist_clamp
    trans_dist = trans_dist.clamp(max=dist_clamp)
    rot_dist = geodesic_dist(pred_framepairs.get_rots().get_rot_mats(), gt_framepairs.get_rots().get_rot_mats())

    trans_dist = trans_dist * framepair_weight
    rot_dist = rot_dist * framepair_weight

    trans_dist_loss = torch.sum(
        trans_dist**2 * mask,
        dim=(-2, -1),
    ) / mask.sum(dim=(-2, -1)).clip(min=1)

    rotpair_mask = mask * clamp_mask
    rot_dist_loss = torch.sum(
        rot_dist**2 * rotpair_mask,
        dim=(-2, -1),
    ) / torch.clamp(mask.sum(dim=(-2, -1)), min=1)

    return torch.sqrt(trans_dist_loss / trans_scale**2 + rot_dist_loss / rot_scale**2 + eps_so3)

