from typing import Callable

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import copy


from nomadflow.model.modules.attention import FlashTransformerEncoder
from nomadflow.model.modules.motif_utils import motif_pair_state
from nomadflow.openfold.layers.layers_v2 import (
    Linear, swish, ipa_point_weights_init_, permute_final_dims, flatten_final_dims, BackboneUpdate, LayerNorm,
)
from nomadflow.openfold.utils import rigid_utils as ru


# from boltz1
def get_indexing_matrix(K, W, H, device):
    assert W % 2 == 0
    assert H % (W // 2) == 0

    h = H // (W // 2)
    assert h % 2 == 0

    arange = torch.arange(2 * K, device=device)
    index = ((arange.unsqueeze(0) - arange.unsqueeze(1)) + h // 2).clamp(
        min=0, max=h + 1
    )
    index = index.view(K, 2, 2 * K)[:, 0, :]
    onehot = F.one_hot(index, num_classes=h + 2)[..., 1:-1].transpose(1, 0)
    return onehot.reshape(2 * K, h * K).float()

# from boltz1
def single_to_keys(single, indexing_matrix, W, H):
    B, N, D = single.shape
    K = N // W
    single = single.view(B, 2 * K, W // 2, D)
    return torch.einsum("b j i d, j k -> b k i d", single, indexing_matrix).reshape(
        B, K, H, D
    )

# adapted from boltz1
def pairs_to_framepairs(pairs, indexing_matrix, W, H):
    B, N, _, D = pairs.shape
    K = N // W
    pairs = pairs.view(B, K, W, 2 * K, W // 2, D)
    indexing_matrix = indexing_matrix.view(2 * K, K, -1)
    ret = torch.einsum("b x y j i d, j x h -> b x y h i d", pairs, indexing_matrix)
    ret = ret.reshape(
        B, K, W, H, D
    )
    return ret


def _average_incremental_global_quat(ref_quats, pred_quats):
    """Average global frame increments q_pred * inverse(q_ref)."""
    delta_quats = ru.quat_multiply(pred_quats, ru.invert_quat(ref_quats))
    ref_quat = delta_quats[0]
    quat_sign = torch.where(
        (delta_quats * ref_quat).sum(dim=-1, keepdim=True) < 0,
        delta_quats.new_tensor(-1.0),
        delta_quats.new_tensor(1.0),
    )
    mean_quat = F.normalize((delta_quats * quat_sign).mean(dim=0), dim=-1)
    return mean_quat


def _apply_group_rigid_projection(
    ref_trans,
    pred_trans,
    ref_quats,
    pred_quats,
    apply_ref_trans,
    apply_ref_quats,
):
    ref_center = ref_trans.mean(dim=0)
    pred_center = pred_trans.mean(dim=0)
    group_quat = _average_incremental_global_quat(ref_quats, pred_quats)
    rot = ru.quat_to_rot(group_quat)

    projected_trans = (
        torch.einsum("ij,nj->ni", rot, apply_ref_trans - ref_center)
        + pred_center
    )
    projected_quats = ru.quat_multiply(
        group_quat[None],
        apply_ref_quats,
    )
    return projected_trans, projected_quats


def _project_rigid_groups_to_predicted_state(
    rigids_pred,
    rigids_ref,
    motif_group_id,
    rigids_mask,
    is_motif_group_aggregator,
):
    if motif_group_id is None:
        return rigids_pred

    group_id = motif_group_id.to(device=rigids_mask.device, dtype=torch.long)
    if is_motif_group_aggregator is None:
        aggregator_mask = torch.zeros_like(group_id, dtype=torch.bool)
    else:
        aggregator_mask = is_motif_group_aggregator.to(
            device=rigids_mask.device,
            dtype=torch.bool,
        )
    apply_group_mask = rigids_mask.bool() & (group_id > 0)
    real_group_mask = apply_group_mask & ~aggregator_mask
    if not bool(real_group_mask.any()):
        return rigids_pred

    pred_trans = rigids_pred.get_trans()
    pred_quats = rigids_pred.get_rots().get_cur_rot()
    ref_trans = rigids_ref.get_trans()
    ref_quats = rigids_ref.get_rots().get_cur_rot()
    if pred_quats.shape[-1] != 4 or ref_quats.shape[-1] != 4:
        raise ValueError("Motif group projection expects quaternion-backed rigids")

    out_trans = pred_trans.clone()
    out_quats = pred_quats.clone()

    batch_size = pred_trans.shape[0]
    for batch_idx in range(batch_size):
        batch_group_ids = torch.unique(group_id[batch_idx, real_group_mask[batch_idx]])
        for group_val in batch_group_ids.tolist():
            curr_real_mask = real_group_mask[batch_idx] & (group_id[batch_idx] == int(group_val))
            curr_apply_mask = apply_group_mask[batch_idx] & (group_id[batch_idx] == int(group_val))
            projected_trans, projected_quats = _apply_group_rigid_projection(
                ref_trans=ref_trans[batch_idx, curr_real_mask],
                pred_trans=pred_trans[batch_idx, curr_real_mask],
                ref_quats=ref_quats[batch_idx, curr_real_mask],
                pred_quats=pred_quats[batch_idx, curr_real_mask],
                apply_ref_trans=ref_trans[batch_idx, curr_apply_mask],
                apply_ref_quats=ref_quats[batch_idx, curr_apply_mask],
            )
            out_trans[batch_idx, curr_apply_mask] = projected_trans.to(
                dtype=out_trans.dtype
            )
            out_quats[batch_idx, curr_apply_mask] = projected_quats.to(
                dtype=out_quats.dtype
            )

    return ru.Rigid(
        rots=ru.Rotation(quats=out_quats),
        trans=out_trans,
    )


class GatherAdaLN(nn.Module):
    def __init__(self, c_s, c_cond):
        super().__init__()
        self.ln_s = LayerNorm(c_s, elementwise_affine=False, bias=False)
        self.ln_cond = LayerNorm(c_cond, bias=False)
        self.lin_cond = Linear(c_cond, c_s)
        self.lin_cond_nobias = Linear(c_cond, c_s, bias=False)

        self.c_s = c_s
        self.c_cond = c_cond

    def forward(self, s, cond, cond_to_s_idx, s_mask):
        s = self.ln_s(s)
        cond = self.ln_cond(cond)
        cond_gate = torch.gather(
            self.lin_cond(cond),
            dim=1,
            index=cond_to_s_idx[..., None].expand(-1, -1, self.c_s),
        ) * s_mask[..., None]
        cond_bias = torch.gather(
            self.lin_cond_nobias(cond),
            dim=1,
            index=cond_to_s_idx[..., None].expand(-1, -1, self.c_s)
        ) * s_mask[..., None]
        s = s * torch.sigmoid(cond_gate)
        s = s + cond_bias

        return s


class GatherConditionedTransition(nn.Module):
    def __init__(self, c_s, c_cond, n=2):
        super().__init__()
        self.adaln = GatherAdaLN(c_s, c_cond)
        self.lin_1 = Linear(c_s, c_s*n, bias=False)
        self.lin_2 = Linear(c_s, c_s*n, bias=False)
        self.lin_cond = Linear(c_cond, c_s)
        with torch.no_grad():
            self.lin_cond.bias.fill_(-2.0)
        self.lin_b = Linear(c_s*n, c_s, bias=False, init='final')

        self.c_s = c_s

    def forward(self, s, cond, cond_to_s_idx, s_mask):
        s = self.adaln(s, cond, cond_to_s_idx, s_mask)
        b = swish(self.lin_1(s)) * self.lin_2(s)
        cond_gate = torch.gather(
            self.lin_cond(cond),
            dim=1,
            index=cond_to_s_idx[..., None].expand(-1, -1, self.c_s),
        ) * s_mask[..., None]
        s = torch.sigmoid(cond_gate) * self.lin_b(b)
        return s


class GatherUpdate(nn.Module):
    def __init__(self, c_s, c_frame):
        super().__init__()
        self.lin = nn.Sequential(
            LayerNorm(c_s),
            Linear(c_s, c_frame, bias=False)
        )

    def forward(self,
                node_embed,
                rigids_embed,
                rigids_to_res_idx,
                rigids_mask):
        broadcast_embed = self.lin(node_embed)
        broadcast_embed = torch.gather(
            node_embed,
            -2,
            rigids_to_res_idx[..., None].expand([-1 for _ in range(rigids_to_res_idx.dim())] + [rigids_embed.shape[-1]])
        )
        rigids_embed = rigids_embed + broadcast_embed * rigids_mask[..., None]
        return rigids_embed


class ScatterUpdate(nn.Module):
    def __init__(self, c_s, c_frame):
        super().__init__()
        self.lin = Linear(c_frame, c_s, bias=False)

    def forward(self,
                rigids_embed,
                node_embed,
                rigids_to_res_idx,
                rigids_mask):
        rigids_update = F.relu(self.lin(rigids_embed)) * rigids_mask[..., None]
        out = torch.zeros_like(node_embed, dtype=rigids_update.dtype)
        out.scatter_reduce_(
            -2,
            rigids_to_res_idx[..., None].expand(-1, -1, node_embed.shape[-1]),
            rigids_update,
            reduce='mean'
        )
        out_denom = torch.zeros(node_embed.shape[:-1], device=out.device).float()
        denom = rigids_mask.float()
        out_denom.scatter_add_(
            -1,
            rigids_to_res_idx,
            denom,
        )
        # print(out_denom)
        out = out / out_denom[..., None].clip(min=1)
        return out + node_embed


class BlockInvariantPointAttention(nn.Module):
    """
    Implements Algorithm 22.
    """
    def __init__(
        self,
        c_s,
        c_z,
        c_hidden,
        num_heads,
        num_qk_points,
        num_v_points,
        inf: float = 1e5,
        eps: float = 1e-8,
        final_init='final',
        block_Q=32,
        block_K=128,
        use_compile_path=False,
        use_out_gating=False,
        ablate_down_z=False,
        use_qk_norm=False
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = num_heads
        self.no_qk_points = num_qk_points
        self.no_v_points = num_v_points
        self.inf = inf
        self.eps = eps
        self.block_Q = block_Q
        self.block_K = block_K
        self.use_compile_path = use_compile_path
        self.use_out_gating = use_out_gating
        self.ablate_down_z = ablate_down_z
        self.use_qk_norm = use_qk_norm

        if self.use_qk_norm:
            self.q_norm = LayerNorm(self.c_hidden)
            self.k_norm = LayerNorm(self.c_hidden)

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc, bias=False)
        self.linear_k = Linear(self.c_s, hc, bias=False)
        self.linear_v = Linear(self.c_s, hc, bias=False)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq, bias=False)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv, bias=False)

        self.linear_b = Linear(self.c_z, self.no_heads, bias=False)
        self.down_z = Linear(self.c_z, self.c_z // 4, bias=False)

        self.head_weights = nn.Parameter(torch.zeros((self.no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim =  (
            self.c_z // 4 + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s, init=final_init, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

        self.ln_s = LayerNorm(c_s)
        self.ln_z = LayerNorm(c_z)
        if use_out_gating:
            self.out_gate = Linear(self.c_s, self.no_heads * concat_out_dim, bias=False)
        else:
            self.out_gate = None

    def _pts_bias(
        self,
        q_pts,
        k_pts,
        pts_cdist=False
    ):
        if pts_cdist:
            # we can do a lot of these operations in-place which saves us a small chunk of memory
            pt_att = -2 * torch.einsum("...ahpd,...bhpd->...abh", q_pts, k_pts)
            q_pts_norm = torch.sum(q_pts ** 2, dim=(-1, -2))
            k_pts_norm = torch.sum(k_pts ** 2, dim=(-1, -2))
            pt_att += q_pts_norm[..., None, :]
            pt_att += k_pts_norm[..., None, :, :]

            head_weights = self.softplus(self.head_weights).view(
                *((1,) * len(pt_att.shape[:-1]) + (-1,))
            )
            head_weights = head_weights * math.sqrt(
                1.0 / (3 * (self.no_qk_points * 9.0 / 2))
            )
            pt_att *= head_weights
            pt_att *= -0.5
        else:
            # [*, N_res, N_res, H, P_q, 3]
            pt_displacement = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
            pt_att = pt_displacement ** 2

            # [*, N_res, N_res, H, P_q]
            pt_att = sum(torch.unbind(pt_att, dim=-1))
            head_weights = self.softplus(self.head_weights).view(
                *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
            )
            head_weights = head_weights * math.sqrt(
                1.0 / (3 * (self.no_qk_points * 9.0 / 2))
            )
            pt_att = pt_att * head_weights

            # [*, N_res, N_res, H]
            pt_att = torch.sum(pt_att, dim=-1) * (-0.5)

        return pt_att

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        r: ru.Rigid,
        s_mask: torch.Tensor,
        to_queries: Callable,
        to_keys: Callable,
        pts_cdist=True
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        s = self.ln_s(s)
        z = self.ln_z(z)

        q_in = to_queries(s)
        k_in = to_keys(s)

        # we have to do a bunch of shenanigans to collect
        # the proper rigids, esp since we might not know
        # if the rigids have mat or quat rots
        trans = r.get_trans()
        rots = r.get_rots().get_cur_rot()
        # [*, N, 4 or 3x3]
        rots_flat = rots.view(rots.shape[:2] + (-1,))
        # [*, N_block, block_Q]
        q_trans = to_queries(trans)
        # [*, N_block, block_Q, 4 or 3x3]
        # print("rots", rots)
        q_rots = to_queries(rots_flat)
        # print("q_rots", q_rots)
        if tuple(rots.shape[-2:]) == (3, 3):
            q_rots = q_rots.view(q_rots.shape[:3] + (3, 3))
            # [*, N_block, block_Q]
            q_rigids = ru.Rigid(
                trans=q_trans,
                rots=ru.Rotation(rot_mats=q_rots)
            )
        else:
            # [*, N_block, block_Q]
            q_rigids = ru.Rigid(
                trans=q_trans,
                rots=ru.Rotation(quats=q_rots)
            )
        # print("q_rigids", q_rigids.get_trans(), q_rigids.get_rots().get_cur_rot())
        # [*, N_block, block_K]
        k_trans = to_keys(trans)
        k_rots = to_keys(rots_flat.view(rots.shape[:2] + (-1,)))
        # print("k_rots", k_rots)
        # there's some more shenanigans that needs to happen because to_keys will
        # zero out some quats which will lead to weird problems
        if tuple(rots.shape[-2:]) == (3, 3):
            k_rots = k_rots.view(k_rots.shape[:3] + (3, 3))
            k_rots[(k_rots == 0).all(dim=(-2, -1))] = torch.eye(3, device=k_rots.device)[None, None, None]
            k_rigids = ru.Rigid(
                trans=k_trans,
                rots=ru.Rotation(rot_mats=k_rots)
            )
        else:
            mask = (k_rots == 0).all(dim=-1)
            k_rots = (
                k_rots * (~mask[..., None])
                + torch.tensor([1., 0., 0., 0.], device=k_rots.device)[None, None, None] * mask[..., None]
            )
            # print("k_rots", k_rots)
            k_rigids = ru.Rigid(
                trans=k_trans,
                rots=ru.Rotation(quats=k_rots)
            )
        # print("k_rigids", k_rigids.get_trans(), k_rigids.get_rots().get_cur_rot())

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_block, block_Q, H * C_hidden]
        q = self.linear_q(q_in)
        # [*, N_block, block_K, H * C_hidden]
        k = self.linear_k(k_in)
        # [*, N_block, block_K, H * C_hidden]
        v = self.linear_v(k_in)

        # [*, N_block, block_Q, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_block, block_K, H, C_hidden]
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        # [*, N_block, block_K, H, C_hidden]
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # [*, N_block, block_Q, H * P_q * 3]
        q_pts = self.linear_q_points(q_in)
        # [*, N_block, block_Q, H * P_q, 3]
        q_pts = q_pts.view(q_pts.shape[:-1] + (self.no_heads * self.no_qk_points, 3))
        q_pts = q_rigids[..., None].apply(q_pts)

        # [*, N_block, block_Q, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        )

        # [*, N_block, block_K, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(k_in)
        # [*, N_block, block_K, H * (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-1] + (-1, 3))
        kv_pts = k_rigids[..., None].apply(kv_pts)

        # [*, N_block, block_K, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_block, block_K, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_block, block_Q, block_K, H]
        b = self.linear_b(z)

        # [*, N_block, block_Q, block_K, H]
        a = torch.einsum("bnqhc,bnkhc->bnqkh", q, k)
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += math.sqrt(1.0 / 3) * b

        # # [*, N_block, block_Q, block_K, H, P_q, 3]
        # pt_displacement = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        # pt_att = pt_displacement ** 2

        # # [*, N_block, block_Q, block_K, H, P_q]
        # pt_att = sum(torch.unbind(pt_att, dim=-1))
        # head_weights = self.softplus(self.head_weights).view(
        #     *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        # )
        # head_weights = head_weights * math.sqrt(
        #     1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        # )
        # pt_att = pt_att * head_weights

        # # [*, N_block, block_Q, block_K, H]
        # pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        pt_att = self._pts_bias(q_pts, k_pts, pts_cdist=pts_cdist)
        # [*, N_block, block_Q, block_K]
        attn_mask = to_queries(s_mask[..., None].float()) * to_keys(s_mask[..., None].float()).transpose(-1, -2)
        attn_mask = self.inf * (attn_mask - 1)

        # [*, N_block, block_Q, block_K, H]
        a = a + pt_att
        a = a + attn_mask[..., None]
        # [*, N_block, block_Q, H, block_K]
        a = a.transpose(-1, -2)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_block, block_Q, H, C_hidden]
        o = torch.einsum("bnqhk,bnkhc->bnqhc", a, v)

        # [*, N_block, block_Q, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, N_block, block_Q, H, P_v, 3]
        o_pt = torch.einsum("bnqhk,bnkhvc->bnqhvc", a, v_pts)
        o_pt = q_rigids[..., None, None].invert_apply(o_pt)

        # [*, N_block, block_Q, H * P_v]
        o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(
            o_pt_dists, 2)

        # [*, N_block, block_Q, H * P_v * 3]
        o_pt = flatten_final_dims(o_pt, 3)

        # [*, N_block, block_Q, block_K, H, C_z // 4]
        pair_z = self.down_z(z)#.to(dtype=a.dtype)
        # [*, N_block, block_Q, H, C_z // 4]
        o_pair = torch.einsum("bnqhk,bnqkc->bnqhc", a, pair_z)

        # [*, N_block, block_Q, H * C_z // 4]
        o_pair = flatten_final_dims(o_pair, 2)
        if self.ablate_down_z:
            o_pair = torch.zeros_like(o_pair)

        o_feats = [o, o_pt, o_pt_norm_feats, o_pair]
        o_feats = torch.cat(
            o_feats, dim=-1
        )#.to(dtype=z.dtype)

        if self.out_gate is not None:
            gate = self.out_gate(q_in)
            o_feats = o_feats * torch.sigmoid(gate)

        # [*, N_block, block_Q, C_s]
        out = self.linear_out(o_feats)

        return out.view(s.shape)


class ConditionedBlockInvariantPointAttention(nn.Module):
    """
    Implements Algorithm 22.
    """
    def __init__(
        self,
        c_s,
        c_cond,
        c_z,
        c_hidden,
        num_heads,
        num_qk_points,
        num_v_points,
        inf: float = 1e5,
        eps: float = 1e-8,
        final_init='final',
        block_Q=32,
        block_K=128,
        use_compile_path=False,
        use_out_gating=False,
        use_cond_gating=False,
        ablate_down_z=False,
        use_qk_norm=False,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super().__init__()

        self.c_s = c_s
        self.c_cond = c_cond
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = num_heads
        self.no_qk_points = num_qk_points
        self.no_v_points = num_v_points
        self.inf = inf
        self.eps = eps
        self.block_Q = block_Q
        self.block_K = block_K
        self.use_compile_path = use_compile_path
        self.use_out_gating = use_out_gating
        self.use_cond_gating = use_cond_gating
        self.ablate_down_z = ablate_down_z
        self.use_qk_norm = use_qk_norm

        if self.use_qk_norm:
            self.q_norm = LayerNorm(self.c_hidden)
            self.k_norm = LayerNorm(self.c_hidden)

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc, bias=False)
        self.linear_k = Linear(self.c_s, hc, bias=False)
        self.linear_v = Linear(self.c_s, hc, bias=False)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq, bias=False)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv, bias=False)

        self.linear_b = Linear(self.c_z, self.no_heads, bias=False)
        self.down_z = Linear(self.c_z, self.c_z // 4, bias=False)

        self.head_weights = nn.Parameter(torch.zeros((self.no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim =  (
            self.c_z // 4 + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s, init=final_init, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

        self.adaln_s = GatherAdaLN(c_s, c_cond)
        self.ln_z = LayerNorm(c_z)
        if use_out_gating:
            self.out_gate = Linear(self.c_s, self.no_heads * concat_out_dim, bias=False)
        else:
            self.out_gate = None
        if use_cond_gating:
            self.cond_gate = Linear(self.c_cond, self.c_s)
            with torch.no_grad():
                self.cond_gate.bias.fill_(-2.0)
        else:
            self.cond_gate = None

    def _pts_bias(
        self,
        q_pts,
        k_pts,
        pts_cdist=False
    ):
        if pts_cdist:
            # we can do a lot of these operations in-place which saves us a small chunk of memory
            pt_att = -2 * torch.einsum("...ahpd,...bhpd->...abh", q_pts, k_pts)
            q_pts_norm = torch.sum(q_pts ** 2, dim=(-1, -2))
            k_pts_norm = torch.sum(k_pts ** 2, dim=(-1, -2))
            pt_att += q_pts_norm[..., None, :]
            pt_att += k_pts_norm[..., None, :, :]

            head_weights = self.softplus(self.head_weights).view(
                *((1,) * len(pt_att.shape[:-1]) + (-1,))
            )
            head_weights = head_weights * math.sqrt(
                1.0 / (3 * (self.no_qk_points * 9.0 / 2))
            )
            pt_att *= head_weights
            pt_att *= -0.5
        else:
            # [*, N_res, N_res, H, P_q, 3]
            pt_displacement = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
            pt_att = pt_displacement ** 2

            # [*, N_res, N_res, H, P_q]
            pt_att = sum(torch.unbind(pt_att, dim=-1))
            head_weights = self.softplus(self.head_weights).view(
                *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
            )
            head_weights = head_weights * math.sqrt(
                1.0 / (3 * (self.no_qk_points * 9.0 / 2))
            )
            pt_att = pt_att * head_weights

            # [*, N_res, N_res, H]
            pt_att = torch.sum(pt_att, dim=-1) * (-0.5)

        return pt_att

    def forward(
        self,
        s: torch.Tensor,
        cond: torch.Tensor,
        cond_to_s_idx: torch.Tensor,
        z: torch.Tensor,
        r: ru.Rigid,
        s_mask: torch.Tensor,
        to_queries: Callable,
        to_keys: Callable,
        pts_cdist=True
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        s = self.adaln_s(s, cond, cond_to_s_idx, s_mask)
        z = self.ln_z(z)

        q_in = to_queries(s)
        k_in = to_keys(s)

        # we have to do a bunch of shenanigans to collect
        # the proper rigids, esp since we might not know
        # if the rigids have mat or quat rots
        trans = r.get_trans()
        rots = r.get_rots().get_cur_rot()
        # [*, N, 4 or 3x3]
        rots_flat = rots.view(rots.shape[:2] + (-1,))
        # [*, N_block, block_Q]
        q_trans = to_queries(trans)
        # [*, N_block, block_Q, 4 or 3x3]
        # print("rots", rots)
        q_rots = to_queries(rots_flat)
        # print("q_rots", q_rots)
        if tuple(rots.shape[-2:]) == (3, 3):
            q_rots = q_rots.view(q_rots.shape[:3] + (3, 3))
            # [*, N_block, block_Q]
            q_rigids = ru.Rigid(
                trans=q_trans,
                rots=ru.Rotation(rot_mats=q_rots)
            )
        else:
            # [*, N_block, block_Q]
            q_rigids = ru.Rigid(
                trans=q_trans,
                rots=ru.Rotation(quats=q_rots)
            )
        # print("q_rigids", q_rigids.get_trans(), q_rigids.get_rots().get_cur_rot())
        # [*, N_block, block_K]
        k_trans = to_keys(trans)
        k_rots = to_keys(rots_flat.view(rots.shape[:2] + (-1,)))
        # print("k_rots", k_rots)
        # there's some more shenanigans that needs to happen because to_keys will
        # zero out some quats which will lead to weird problems
        if tuple(rots.shape[-2:]) == (3, 3):
            k_rots = k_rots.view(k_rots.shape[:3] + (3, 3))
            k_rots[(k_rots == 0).all(dim=(-2, -1))] = torch.eye(3, device=k_rots.device)[None, None, None]
            k_rigids = ru.Rigid(
                trans=k_trans,
                rots=ru.Rotation(rot_mats=k_rots)
            )
        else:
            mask = (k_rots == 0).all(dim=-1)
            k_rots = (
                k_rots * (~mask[..., None])
                + torch.tensor([1., 0., 0., 0.], device=k_rots.device)[None, None, None] * mask[..., None]
            )
            # print("k_rots", k_rots)
            k_rigids = ru.Rigid(
                trans=k_trans,
                rots=ru.Rotation(quats=k_rots)
            )
        # print("k_rigids", k_rigids.get_trans(), k_rigids.get_rots().get_cur_rot())

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_block, block_Q, H * C_hidden]
        q = self.linear_q(q_in)
        # [*, N_block, block_K, H * C_hidden]
        k = self.linear_k(k_in)
        # [*, N_block, block_K, H * C_hidden]
        v = self.linear_v(k_in)

        # [*, N_block, block_Q, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_block, block_K, H, C_hidden]
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        # [*, N_block, block_K, H, C_hidden]
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # [*, N_block, block_Q, H * P_q * 3]
        q_pts = self.linear_q_points(q_in)
        # [*, N_block, block_Q, H * P_q, 3]
        q_pts = q_pts.view(q_pts.shape[:-1] + (self.no_heads * self.no_qk_points, 3))
        q_pts = q_rigids[..., None].apply(q_pts)

        # [*, N_block, block_Q, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        )

        # [*, N_block, block_K, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(k_in)
        # [*, N_block, block_K, H * (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-1] + (-1, 3))
        kv_pts = k_rigids[..., None].apply(kv_pts)

        # [*, N_block, block_K, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_block, block_K, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_block, block_Q, block_K, H]
        b = self.linear_b(z)

        # [*, N_block, block_Q, block_K, H]
        a = torch.einsum("bnqhc,bnkhc->bnqkh", q, k)
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += math.sqrt(1.0 / 3) * b

        # # [*, N_block, block_Q, block_K, H, P_q, 3]
        # pt_displacement = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        # pt_att = pt_displacement ** 2

        # # [*, N_block, block_Q, block_K, H, P_q]
        # pt_att = sum(torch.unbind(pt_att, dim=-1))
        # head_weights = self.softplus(self.head_weights).view(
        #     *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        # )
        # head_weights = head_weights * math.sqrt(
        #     1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        # )
        # pt_att = pt_att * head_weights

        # # [*, N_block, block_Q, block_K, H]
        # pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        pt_att = self._pts_bias(q_pts, k_pts, pts_cdist=pts_cdist)
        # [*, N_block, block_Q, block_K]
        attn_mask = to_queries(s_mask[..., None].float()) * to_keys(s_mask[..., None].float()).transpose(-1, -2)
        attn_mask = self.inf * (attn_mask - 1)

        # [*, N_block, block_Q, block_K, H]
        a = a + pt_att
        a = a + attn_mask[..., None]
        # [*, N_block, block_Q, H, block_K]
        a = a.transpose(-1, -2)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_block, block_Q, H, C_hidden]
        o = torch.einsum("bnqhk,bnkhc->bnqhc", a, v)

        # [*, N_block, block_Q, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, N_block, block_Q, H, P_v, 3]
        o_pt = torch.einsum("bnqhk,bnkhvc->bnqhvc", a, v_pts)
        o_pt = q_rigids[..., None, None].invert_apply(o_pt)

        # [*, N_block, block_Q, H * P_v]
        o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(
            o_pt_dists, 2)

        # [*, N_block, block_Q, H * P_v * 3]
        o_pt = flatten_final_dims(o_pt, 3)

        # [*, N_block, block_Q, block_K, H, C_z // 4]
        pair_z = self.down_z(z)#.to(dtype=a.dtype)
        # [*, N_block, block_Q, H, C_z // 4]
        o_pair = torch.einsum("bnqhk,bnqkc->bnqhc", a, pair_z)

        # [*, N_block, block_Q, H * C_z // 4]
        o_pair = flatten_final_dims(o_pair, 2)
        if self.ablate_down_z:
            o_pair = torch.zeros_like(o_pair)

        o_feats = [o, o_pt, o_pt_norm_feats, o_pair]
        o_feats = torch.cat(
            o_feats, dim=-1
        )#.to(dtype=z.dtype)

        if self.out_gate is not None:
            gate = self.out_gate(q_in)
            o_feats = o_feats * torch.sigmoid(gate)

        # [*, N_block, block_Q, C_s]
        out = self.linear_out(o_feats)
        out = out.view(s.shape)

        if self.cond_gate is not None:
            cond_gate = self.cond_gate(cond)
            cond_gate = torch.gather(
                cond_gate,
                dim=1,
                index=cond_to_s_idx[..., None].expand(-1, -1, self.c_s),
            ) * s_mask[..., None]
            out = out * torch.sigmoid(cond_gate)

        return out


class BlockAttentionPairBias(nn.Module):
    def __init__(
        self,
        c_s,
        c_z,
        num_heads=4,
        inf=1e5,
        use_qk_norm=False,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.num_heads = num_heads
        self.c_hidden = c_s // num_heads
        self.inf = inf
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = LayerNorm(self.c_hidden)
            self.k_norm = LayerNorm(self.c_hidden)

        self.ln = LayerNorm(c_s)
        self.lin_q = Linear(c_s, c_s, bias=False)
        self.lin_kv = Linear(c_s, 2*c_s, bias=False)
        self.lin_bias = nn.Sequential(
            LayerNorm(c_z),
            Linear(c_z, num_heads, bias=False)
        )
        self.lin_gate = Linear(c_s, c_s, bias=False)
        self.lin_out = Linear(c_s, c_s, bias=False, init='final')

    def forward(
        self,
        node_embed,
        edge_embed,
        to_queries,
        to_keys,
        node_mask
    ):
        q_mask = to_queries(node_mask[..., None].float())
        k_mask = to_keys(node_mask[..., None].float()).transpose(-1, -2)
        edge_mask = q_mask * k_mask

        node_embed = self.ln(node_embed)

        q_in = to_queries(node_embed)
        kv_in = to_keys(node_embed)

        q = self.lin_q(q_in)
        q = q.unflatten(-1, (self.num_heads, self.c_hidden))
        kv = self.lin_kv(kv_in)
        kv = kv.unflatten(-1, (2 * self.num_heads, self.c_hidden))
        k, v = kv.split(self.num_heads, dim=-2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # n_batch x n_block x h x q x k
        a = torch.einsum("...ihk,...jhk->...hij", q, k)
        a = a / math.sqrt(self.c_hidden)
        bias = self.lin_bias(edge_embed)
        a = a + permute_final_dims(bias, (2, 0, 1))
        a = a + (edge_mask - 1)[..., None, :, :] * self.inf
        a = torch.softmax(a, dim=-1)

        out = torch.einsum("...hij,...jhk->...ihk", a, v)
        gate = torch.sigmoid(self.lin_gate(q_in))
        out = flatten_final_dims(out, 2) * gate
        out = self.lin_out(out.view(node_embed.shape))

        return out


class ConditionedBlockAttentionPairBias(nn.Module):
    def __init__(
        self,
        c_s,
        c_cond,
        c_z,
        num_heads=4,
        inf=1e5,
        use_qk_norm=False
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.num_heads = num_heads
        self.c_hidden = c_s // num_heads
        self.inf = inf
        self.use_qk_norm = use_qk_norm

        if use_qk_norm:
            self.q_norm = LayerNorm(self.c_hidden)
            self.k_norm = LayerNorm(self.c_hidden)

        self.adaln = GatherAdaLN(c_s, c_cond)
        self.lin_q = Linear(c_s, c_s, bias=False)
        self.lin_kv = Linear(c_s, 2*c_s, bias=False)
        self.lin_bias = nn.Sequential(
            LayerNorm(c_z),
            Linear(c_z, num_heads, bias=False)
        )
        self.lin_gate = Linear(c_s, c_s, bias=False)

        self.lin_cond_gate = Linear(c_cond, c_s)
        with torch.no_grad():
            self.lin_cond_gate.bias.fill_(-2.0)

        self.lin_out = Linear(c_s, c_s, bias=False, init='final')

    def forward(
        self,
        node_embed,
        cond_embed,
        cond_to_node_idx,
        edge_embed,
        to_queries,
        to_keys,
        node_mask
    ):
        q_mask = to_queries(node_mask[..., None].float())
        k_mask = to_keys(node_mask[..., None].float()).transpose(-1, -2)
        edge_mask = q_mask * k_mask

        node_embed = self.adaln(node_embed, cond_embed, cond_to_node_idx, node_mask)

        q_in = to_queries(node_embed)
        kv_in = to_keys(node_embed)

        q = self.lin_q(q_in)
        q = q.unflatten(-1, (self.num_heads, self.c_hidden))
        kv = self.lin_kv(kv_in)
        kv = kv.unflatten(-1, (2 * self.num_heads, self.c_hidden))
        k, v = kv.split(self.num_heads, dim=-2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # n_batch x n_block x h x q x k
        a = torch.einsum("...ihk,...jhk->...hij", q, k)
        a = a / math.sqrt(self.c_hidden)
        bias = self.lin_bias(edge_embed)
        a = a + permute_final_dims(bias, (2, 0, 1))
        a = a + (edge_mask - 1)[..., None, :, :] * self.inf
        a = torch.softmax(a, dim=-1)

        out = torch.einsum("...hij,...jhk->...ihk", a, v)
        gate = torch.sigmoid(self.lin_gate(q_in))
        out = flatten_final_dims(out, 2) * gate
        out = self.lin_out(out.view(node_embed.shape))

        cond_gate = torch.sigmoid(self.lin_cond_gate(cond_embed))
        cond_gate = torch.gather(
            cond_gate,
            dim=1,
            index=cond_to_node_idx[..., None].expand(-1, -1, self.c_s),
        ) * node_mask[..., None]
        out = out * cond_gate

        return out


class BlockTransformerPairBias(nn.Module):
    def __init__(
        self,
        c_s,
        c_cond,
        c_z,
        num_heads,
        use_qk_norm=False,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z

        self.attn = BlockAttentionPairBias(
            c_s,
            c_z,
            num_heads,
            use_qk_norm=use_qk_norm
        )
        self.transition = GatherConditionedTransition(c_s, c_cond)

    def forward(
        self,
        s,
        rigids_embed,
        framepair_embed,
        to_queries,
        to_keys,
        rigids_mask,
        rigids_to_res_idx
    ):
        update = self.attn(
            rigids_embed,
            framepair_embed,
            to_queries,
            to_keys,
            rigids_mask
        )
        rigids_embed = rigids_embed + update * rigids_mask[..., None]

        rigids_embed = rigids_embed + self.transition(rigids_embed, s, rigids_to_res_idx, rigids_mask) * rigids_mask[..., None]

        return rigids_embed


class ConditionedBlockTransformerPairBias(nn.Module):
    def __init__(
        self,
        c_s,
        c_cond,
        c_z,
        num_heads,
        use_qk_norm=False,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z

        self.attn = ConditionedBlockAttentionPairBias(
            c_s,
            c_cond,
            c_z,
            num_heads,
            use_qk_norm=use_qk_norm
        )
        self.transition = GatherConditionedTransition(c_s, c_cond)

    def forward(
        self,
        rigids_embed,
        s,
        framepair_embed,
        to_queries,
        to_keys,
        rigids_mask,
        rigids_to_res_idx,
    ):
        update = self.attn(
            rigids_embed,
            s,
            rigids_to_res_idx,
            framepair_embed,
            to_queries,
            to_keys,
            rigids_mask
        )
        rigids_embed = rigids_embed + update * rigids_mask[..., None]

        rigids_embed = rigids_embed + self.transition(rigids_embed, s, rigids_to_res_idx, rigids_mask) * rigids_mask[..., None]

        return rigids_embed


class FramepairEmbedder(nn.Module):
    def __init__(self,
                 c_framepair):
        super().__init__()
        self.lin_framepair_dist_vec = Linear(3, c_framepair, bias=False)
        self.lin_framepair_rel_quat = Linear(4, c_framepair, bias=False)
        self.lin_framepair_dist = Linear(1, c_framepair, bias=False)
        self.lin_same_res = Linear(1, c_framepair, bias=False)
        # Pair-state IDs: 0=both not motif, 1=one motif, 2=same motif group, 3=different motif groups.
        self.motif_pair_state_embed = nn.Embedding(4, c_framepair)
        nn.init.zeros_(self.motif_pair_state_embed.weight[0])

    def forward(self,
               rigids_flat,
               rigids_to_res_idx,
               rigids_mask,
               rigids_motif_group_id,
               to_queries,
               to_keys):
        trans = rigids_flat.get_trans()
        rots = rigids_flat.get_rots().get_cur_rot()
        assert rots.shape[-1] == 4

        # [*, N_block, block_Q]
        q_trans = to_queries(trans)
        q_rots = to_queries(rots)
        q_rigids = ru.Rigid(
            trans=q_trans,
            rots=ru.Rotation(quats=q_rots)
        )
        # [*, N_block, block_K]
        k_trans = to_keys(trans)
        k_rots = to_keys(rots)
        # there's some more shenanigans that needs to happen because to_keys will
        # zero out some quats which will lead to weird problems
        mask = (k_rots == 0).all(dim=-1)
        k_rots = (
            k_rots * (~mask[..., None])
            + torch.tensor([1., 0., 0., 0.], device=k_rots.device)[None, None, None] * mask[..., None]
        )
        k_rigids = ru.Rigid(
            trans=k_trans,
            rots=ru.Rotation(quats=k_rots)
        )

        rel_quat = k_rigids[..., None, :].get_rots().compose_q(q_rigids[..., None].invert().get_rots())
        rel_quat = rel_quat.get_quats()
        # rel_trans = k_rigids[..., None, :].get_trans() - q_rigids[..., None].get_trans()
        rel_trans = q_rigids[..., None].invert_apply(k_rigids[..., None, :].get_trans())
        rel_dist = torch.sum(rel_trans ** 2, dim=-1, keepdim=True)

        v_lm = to_queries(rigids_to_res_idx[..., None].float())[..., None, :].long() == to_keys(rigids_to_res_idx[..., None].float())[..., None, :, :].long()
        v_lm = v_lm.float()
        motif_group_q = to_queries(rigids_motif_group_id[..., None].float())[..., 0].long()
        motif_group_k = to_keys(rigids_motif_group_id[..., None].float())[..., 0].long()
        pair_state = motif_pair_state(motif_group_q, motif_group_k)
        
        framepair_embed = self.lin_framepair_dist_vec(rel_trans) * v_lm
        framepair_embed = framepair_embed + self.lin_framepair_rel_quat(rel_quat) * v_lm
        framepair_embed = framepair_embed + self.lin_framepair_dist(1 / (1 + rel_dist)) * v_lm
        framepair_embed = framepair_embed + self.lin_same_res(v_lm) * v_lm
        framepair_embed = framepair_embed + self.motif_pair_state_embed(pair_state)
        framepair_mask = to_queries(rigids_mask[..., None].float())[..., None, :].bool() & to_keys(rigids_mask[..., None].float())[..., None, :, :].bool()
        framepair_embed = framepair_embed * framepair_mask

        return framepair_embed


class FramepairUpdate(nn.Module):
    def __init__(
        self,
        c_z,
        c_frame,
        c_framepair,
        broadcast_singles=True,
        broadcast_pairs=True,
        framepair_ffn=True,
    ):
        super().__init__()
        if broadcast_pairs:
            self.nodepair_to_framepair_broadcast = nn.Sequential(
                LayerNorm(c_z),
                Linear(c_z, c_framepair, bias=False)
            )
        else:
            self.nodepair_to_framepair_broadcast = None

        self.broadcast_singles = broadcast_singles
        if broadcast_singles:
            self.frame_ln = LayerNorm(c_frame, c_frame)
            self.frame_l_to_framepair_broadcast = Linear(c_frame, c_framepair, bias=False)
            self.frame_m_to_framepair_broadcast = Linear(c_frame, c_framepair, bias=False)

        if framepair_ffn:
            self.framepair_ffn = nn.Sequential(
                LayerNorm(c_framepair),
                Linear(c_framepair, c_framepair),
                nn.ReLU(),
                Linear(c_framepair, c_framepair),
                nn.ReLU(),
                Linear(c_framepair, c_framepair),
            )
        else:
            self.framepair_ffn = None

    def _broadcast_pairs(
        self,
        z,
        rigids_to_res_idx,
        rigids_mask,
        to_pairs
    ):
        n_rigids = rigids_mask.shape[-1]
        # TODO: this is a little jenk
        # with the batch dim
        with torch.no_grad():
            rigids_pair_idx = torch.stack([
                rigids_to_res_idx[0:1, ..., None].tile(1, 1, n_rigids),
                rigids_to_res_idx[0:1, ..., None, :].tile(1, n_rigids, 1)
            ], dim=-1)
            rigids_pair_mask = (
                rigids_mask[0:1, ..., None].tile(1, 1, n_rigids)
                * rigids_mask[0:1, ..., None, :].tile(1, n_rigids, 1)
            )

            # bmm only works on floating point types
            # but we lose precision on large ints so we disable autocast
            with torch.autocast("cuda", enabled=False):
                framepairs_idx = to_pairs(rigids_pair_idx.float()).long()
                framepairs_mask = to_pairs(rigids_pair_mask[..., None].float()).bool()

        framepairs = z[:, framepairs_idx[0, ..., 0], framepairs_idx[0, ..., 1]]
        framepairs = framepairs * framepairs_mask
        return framepairs

    def forward(
        self,
        z,
        framepair_embed,
        rigids_embed_flat,
        rigids_to_res_idx,
        rigids_mask,
        to_queries,
        to_keys,
        to_pairs

    ):
        if self.broadcast_singles:
            framepair_src = self.frame_ln(rigids_embed_flat)
            framepair_l = self.frame_l_to_framepair_broadcast(framepair_src)
            framepair_m = self.frame_m_to_framepair_broadcast(framepair_src)
            framepair_l = to_queries(framepair_l).unsqueeze(-2) # .view(n_batch, -1, self.block_q, 1, self.c_framepair)
            framepair_m = to_keys(framepair_m).unsqueeze(-3) #.view(n_batch, -1, 1, self.block_k, self.c_framepair)
            framepair_embed = framepair_embed + framepair_l + framepair_m

        if self.nodepair_to_framepair_broadcast is not None:
            # z_broadcast = self._broadcast_pairs(z, rigids_to_res_idx, rigids_mask, to_pairs)
            # framepair_embed = framepair_embed + self.nodepair_to_framepair_broadcast(z_broadcast)
            z_broadcast = self._broadcast_pairs(self.nodepair_to_framepair_broadcast(z), rigids_to_res_idx, rigids_mask, to_pairs)
            framepair_embed = framepair_embed + z_broadcast

        if self.framepair_ffn is not None:
            framepair_embed = framepair_embed + self.framepair_ffn(framepair_embed)

        return framepair_embed


class SequenceFrameTransformerBlock(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_frame=128,
                 c_framepair=16,
                 num_heads=4,
                 num_qk_points=8,
                 num_v_points=12,
                 block_q=32,
                 block_k=128,
                 inf=1e8,
                 compile_ipa=False,
                 use_ipa_gating=False,
                 ablate_ipa_down_z=False,
                 use_qk_norm=False
                 ):
        super().__init__()

        self.c_frame = c_frame
        self.c_framepair = c_framepair
        self.c_s = c_s
        self.num_heads = num_heads
        self.c_hidden = c_frame // num_heads
        self.block_q = block_q
        self.block_k = block_k
        self.inf = inf

        self.block_ipa = BlockInvariantPointAttention(
            c_s=c_frame,
            c_z=c_framepair,
            c_hidden=self.c_hidden,
            num_heads=num_heads,
            num_qk_points=num_qk_points,
            num_v_points=num_v_points,
            block_Q=block_q,
            block_K=block_k,
            use_compile_path=compile_ipa,
            use_out_gating=use_ipa_gating,
            ablate_down_z=ablate_ipa_down_z,
            use_qk_norm=use_qk_norm
        )

        self.transition = GatherConditionedTransition(c_frame, c_s)

    def forward(
        self,
        s,
        rigids,
        rigids_embed,
        framepair_embed,
        to_queries,
        to_keys,
        rigids_mask,
        rigids_to_res_idx
    ):
        rigids_embed_update = self.block_ipa(
            s=rigids_embed,
            z=framepair_embed,
            r=rigids,
            s_mask=rigids_mask,
            to_queries=to_queries,
            to_keys=to_keys,
        )
        rigids_embed = rigids_embed + rigids_embed_update * rigids_mask[..., None]

        rigids_embed = rigids_embed + self.transition(rigids_embed, s, rigids_to_res_idx, rigids_mask) * rigids_mask[..., None]

        return rigids_embed


class SequenceFrameTransformerUpdate(nn.Module):
    def __init__(
        self,
        c_s=256,
        c_z=128,
        c_frame=128,
        c_framepair=16,
        num_heads=4,
        num_qk_points=8,
        num_v_points=12,
        block_q=32,
        block_k=128,
        inf=1e8,
        n_blocks=2,
        do_rigid_updates=False,
        broadcast_singles=False,
        broadcast_pairs=False,
        agg_rigid_embed=True,
        compile_ipa=False,
        framepair_init=False,
        framepair_ffn=False,
        add_vanilla_transformer=False,
        add_second_transformer=False,
        add_full_transformer=False,
        use_ipa_gating=False,
        ablate_ipa_down_z=False,
        use_qk_norm=False,

    ):
        super().__init__()
        self.c_s = c_s
        self.c_frame = c_frame
        self.c_framepair = c_framepair
        self.n_blocks = n_blocks
        self.block_q = block_q
        self.block_k = block_k
        self.do_rigid_updates = do_rigid_updates
        self.agg_rigid_embed = agg_rigid_embed
        self.framepair_init = framepair_init
        self.add_vanilla_transformer = add_vanilla_transformer
        self.add_second_transformer = add_second_transformer
        self.add_full_transformer = add_full_transformer

        self.node_to_frame_broadcast = GatherUpdate(c_s, c_frame)

        self.framepair_update = FramepairUpdate(
            c_z=c_z,
            c_frame=c_frame,
            c_framepair=c_framepair,
            broadcast_singles=broadcast_singles,
            broadcast_pairs=broadcast_pairs,
            framepair_ffn=framepair_ffn
        )

        self.trunk = nn.ModuleDict()
        for b in range(n_blocks):
            self.trunk[f'tfmr_{b}'] = SequenceFrameTransformerBlock(
                c_s=c_s,
                c_frame=c_frame,
                c_framepair=c_framepair,
                num_heads=num_heads,
                num_qk_points=num_qk_points,
                num_v_points=num_v_points,
                block_q=block_q,
                block_k=block_k,
                inf=inf,
                compile_ipa=compile_ipa,
                use_ipa_gating=use_ipa_gating,
                ablate_ipa_down_z=ablate_ipa_down_z,
                use_qk_norm=use_qk_norm
            )
            if self.add_vanilla_transformer:
                self.trunk[f'vanilla_tfmr_{b}'] = BlockTransformerPairBias(
                    c_s=c_frame,
                    c_cond=c_s,
                    c_z=c_framepair,
                    num_heads=num_heads,
                    use_qk_norm=use_qk_norm
                )
            if self.add_second_transformer:
                self.trunk[f'second_tfmr_{b}'] = BlockTransformerPairBias(
                    c_s=c_frame,
                    c_cond=c_s,
                    c_z=c_framepair,
                    num_heads=num_heads,
                    use_qk_norm=use_qk_norm
                )
            if self.add_full_transformer:
                self.trunk[f'full_tfmr_{b}'] = FlashTransformerEncoder(
                    h_dim=c_frame,
                    no_heads=num_heads,
                    h_ff=c_frame,
                    n_layers=2,
                    ln_first=True,
                    bias=False
                )

            if self.do_rigid_updates:
                self.trunk[f'frame_update_{b}'] = BackboneUpdate(c_frame)

        if self.agg_rigid_embed:
            self.frame_to_node_broadcast = ScatterUpdate(
                c_s,
                c_frame
            )


    def forward(
        self,
        s,
        z,
        framepair_embed,
        rigids_flat,
        rigids_embed_flat,
        rigids_to_res_idx,
        rigids_mask,
        rigids_noising_mask,
        motif_group_id,
        to_queries,
        to_keys,
        to_pairs,
        is_motif_group_aggregator,
    ):
        rigids_embed_flat = self.node_to_frame_broadcast(
            s,
            rigids_embed_flat,
            rigids_to_res_idx,
            rigids_mask
        )

        framepair_embed = self.framepair_update(
            z,
            framepair_embed,
            rigids_embed_flat,
            rigids_to_res_idx,
            rigids_mask,
            to_queries,
            to_keys,
            to_pairs
        )

        for b in range(self.n_blocks):
            rigids_embed_flat = self.trunk[f'tfmr_{b}'](
                s,
                rigids_flat,
                rigids_embed_flat,
                framepair_embed,
                to_queries,
                to_keys,
                rigids_mask,
                rigids_to_res_idx
            )
            if self.add_vanilla_transformer:
                rigids_embed_flat = self.trunk[f'vanilla_tfmr_{b}'](
                    s=s,
                    rigids_embed=rigids_embed_flat,
                    framepair_embed=framepair_embed,
                    to_queries=to_queries,
                    to_keys=to_keys,
                    rigids_mask=rigids_mask,
                    rigids_to_res_idx=rigids_to_res_idx
                )
            if self.add_second_transformer:
                rigids_embed_flat = self.trunk[f'second_tfmr_{b}'](
                    s=s,
                    rigids_embed=rigids_embed_flat,
                    framepair_embed=framepair_embed,
                    to_queries=to_queries,
                    to_keys=to_keys,
                    rigids_mask=rigids_mask,
                    rigids_to_res_idx=rigids_to_res_idx
                )
            if self.add_full_transformer:
                rigids_embed_flat = self.trunk[f'full_tfmr_{b}'](
                    rigids_embed_flat,
                    rigids_mask,
                )
            if self.do_rigid_updates:
                rigids_ref_flat = rigids_flat
                rigids_update = self.trunk[f'frame_update_{b}'](rigids_embed_flat) * rigids_mask[..., None]
                rigids_flat = rigids_flat.compose_q_update_vec(rigids_update * rigids_noising_mask[..., None])
                rigids_flat = _project_rigid_groups_to_predicted_state(
                    rigids_pred=rigids_flat,
                    rigids_ref=rigids_ref_flat,
                    motif_group_id=motif_group_id,
                    rigids_mask=rigids_mask,
                    is_motif_group_aggregator=is_motif_group_aggregator,
                )

        if self.agg_rigid_embed:
            # print(rigids_embed_flat.isnan().any(), s.isnan().any())
            s = self.frame_to_node_broadcast(
                rigids_embed_flat,
                s,
                rigids_to_res_idx,
                rigids_mask
            )

        return rigids_embed_flat, s, framepair_embed, rigids_flat


class ConditionedSequenceFrameTransformerBlock(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_frame=128,
                 c_framepair=16,
                 num_heads=4,
                 num_qk_points=8,
                 num_v_points=12,
                 block_q=32,
                 block_k=128,
                 inf=1e8,
                 compile_ipa=False,
                 use_ipa_gating=False,
                 ablate_ipa_down_z=False,
                 use_qk_norm=False,
                 use_conditioned_ipa=True,
                 ):
        super().__init__()

        self.c_frame = c_frame
        self.c_framepair = c_framepair
        self.c_s = c_s
        self.num_heads = num_heads
        self.c_hidden = c_frame // num_heads
        self.block_q = block_q
        self.block_k = block_k
        self.inf = inf

        self.use_conditioned_ipa = use_conditioned_ipa

        if use_conditioned_ipa:
            self.block_ipa = ConditionedBlockInvariantPointAttention(
                c_s=c_frame,
                c_cond=c_s,
                c_z=c_framepair,
                c_hidden=self.c_hidden,
                num_heads=num_heads,
                num_qk_points=num_qk_points,
                num_v_points=num_v_points,
                block_Q=block_q,
                block_K=block_k,
                use_compile_path=compile_ipa,
                use_out_gating=use_ipa_gating,
                use_cond_gating=use_ipa_gating,
                ablate_down_z=ablate_ipa_down_z,
                use_qk_norm=use_qk_norm,
            )
        else:
            self.block_ipa = BlockInvariantPointAttention(
                c_s=c_frame,
                c_z=c_framepair,
                c_hidden=self.c_hidden,
                num_heads=num_heads,
                num_qk_points=num_qk_points,
                num_v_points=num_v_points,
                block_Q=block_q,
                block_K=block_k,
                use_compile_path=compile_ipa,
                use_out_gating=use_ipa_gating,
                ablate_down_z=ablate_ipa_down_z,
                use_qk_norm=use_qk_norm,
            )

        self.transition = GatherConditionedTransition(c_frame, c_s)


    def forward(
        self,
        s,
        rigids,
        rigids_embed,
        framepair_embed,
        to_queries,
        to_keys,
        rigids_mask,
        rigids_to_res_idx,
    ):
        if self.use_conditioned_ipa:
            rigids_embed_update = self.block_ipa(
                s=rigids_embed,
                cond=s,
                cond_to_s_idx=rigids_to_res_idx,
                z=framepair_embed,
                r=rigids,
                s_mask=rigids_mask,
                to_queries=to_queries,
                to_keys=to_keys,
            )
        else:
            rigids_embed_update = self.block_ipa(
                s=rigids_embed,
                z=framepair_embed,
                r=rigids,
                s_mask=rigids_mask,
                to_queries=to_queries,
                to_keys=to_keys,
            )
        rigids_embed = rigids_embed + rigids_embed_update * rigids_mask[..., None]

        rigids_embed = rigids_embed + self.transition(
            rigids_embed,
            s,
            rigids_to_res_idx,
            rigids_mask
        ) * rigids_mask[..., None]

        return rigids_embed


class ConditionedSequenceFrameTransformerUpdate(nn.Module):
    def __init__(
        self,
        c_s=256,
        c_z=128,
        c_frame=128,
        c_framepair=16,
        num_heads=4,
        num_qk_points=8,
        num_v_points=12,
        block_q=32,
        block_k=128,
        inf=1e8,
        n_blocks=2,
        do_rigid_updates=False,
        broadcast_singles=True,
        broadcast_pairs=False,
        framepair_ffn=True,
        agg_rigid_embed=True,
        compile_ipa=False,
        framepair_init=False,
        add_vanilla_transformer=False,
        add_second_transformer=False,
        use_ipa_gating=False,
        ablate_ipa_down_z=False,
        use_qk_norm=False,
        use_conditioned_ipa=True,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_frame = c_frame
        self.c_framepair = c_framepair
        self.n_blocks = n_blocks
        self.block_q = block_q
        self.block_k = block_k
        self.do_rigid_updates = do_rigid_updates
        self.agg_rigid_embed = agg_rigid_embed
        self.framepair_init = framepair_init
        self.add_vanilla_transformer = add_vanilla_transformer
        self.add_second_transformer = add_second_transformer

        self.node_to_frame_broadcast = GatherUpdate(c_s, c_frame)

        self.framepair_update = FramepairUpdate(
            c_z=c_z,
            c_frame=c_frame,
            c_framepair=c_framepair,
            broadcast_singles=broadcast_singles,
            broadcast_pairs=broadcast_pairs,
            framepair_ffn=framepair_ffn
        )

        self.trunk = nn.ModuleDict()
        for b in range(n_blocks):
            self.trunk[f'tfmr_{b}'] = ConditionedSequenceFrameTransformerBlock(
                c_s=c_s,
                c_frame=c_frame,
                c_framepair=c_framepair,
                num_heads=num_heads,
                num_qk_points=num_qk_points,
                num_v_points=num_v_points,
                block_q=block_q,
                block_k=block_k,
                inf=inf,
                compile_ipa=compile_ipa,
                use_ipa_gating=use_ipa_gating,
                ablate_ipa_down_z=ablate_ipa_down_z,
                use_qk_norm=use_qk_norm,
                use_conditioned_ipa=use_conditioned_ipa,
            )
            if self.add_vanilla_transformer:
                self.trunk[f'vanilla_tfmr_{b}'] = ConditionedBlockTransformerPairBias(
                    c_s=c_frame,
                    c_cond=c_s,
                    c_z=c_framepair,
                    num_heads=num_heads,
                    use_qk_norm=use_qk_norm
                )
            if self.add_second_transformer:
                self.trunk[f'second_tfmr_{b}'] = ConditionedBlockTransformerPairBias(
                    c_s=c_frame,
                    c_cond=c_s,
                    c_z=c_framepair,
                    num_heads=num_heads,
                    use_qk_norm=use_qk_norm
                )

            if self.do_rigid_updates:
                self.trunk[f'frame_update_{b}'] = BackboneUpdate(c_frame)

        if self.agg_rigid_embed:
            self.frame_to_node_broadcast = ScatterUpdate(
                c_s,
                c_frame
            )

    def forward(
        self,
        s,
        z,
        framepair_embed,
        rigids_flat,
        rigids_embed_flat,
        rigids_to_res_idx,
        rigids_mask,
        rigids_noising_mask,
        motif_group_id,
        to_queries,
        to_keys,
        to_pairs,
        is_motif_group_aggregator,
    ):
        rigids_embed_flat = self.node_to_frame_broadcast(
            s,
            rigids_embed_flat,
            rigids_to_res_idx,
            rigids_mask
        )

        framepair_embed = self.framepair_update(
            z,
            framepair_embed,
            rigids_embed_flat,
            rigids_to_res_idx,
            rigids_mask,
            to_queries,
            to_keys,
            to_pairs
        )

        for b in range(self.n_blocks):
            rigids_embed_flat = self.trunk[f'tfmr_{b}'](
                s,
                rigids_flat,
                rigids_embed_flat,
                framepair_embed,
                to_queries,
                to_keys,
                rigids_mask,
                rigids_to_res_idx,
            )
            if self.add_vanilla_transformer:
                rigids_embed_flat = self.trunk[f'vanilla_tfmr_{b}'](
                    s=s,
                    rigids_embed=rigids_embed_flat,
                    framepair_embed=framepair_embed,
                    to_queries=to_queries,
                    to_keys=to_keys,
                    rigids_mask=rigids_mask,
                    rigids_to_res_idx=rigids_to_res_idx,
                )
            if self.add_second_transformer:
                rigids_embed_flat = self.trunk[f'second_tfmr_{b}'](
                    s=s,
                    rigids_embed=rigids_embed_flat,
                    framepair_embed=framepair_embed,
                    to_queries=to_queries,
                    to_keys=to_keys,
                    rigids_mask=rigids_mask,
                    rigids_to_res_idx=rigids_to_res_idx,
                )

            if self.do_rigid_updates:
                rigids_ref_flat = rigids_flat
                rigids_update = self.trunk[f'frame_update_{b}'](rigids_embed_flat) * rigids_mask[..., None]
                rigids_flat = rigids_flat.compose_q_update_vec(rigids_update * rigids_noising_mask[..., None])
                rigids_flat = _project_rigid_groups_to_predicted_state(
                    rigids_pred=rigids_flat,
                    rigids_ref=rigids_ref_flat,
                    motif_group_id=motif_group_id,
                    rigids_mask=rigids_mask,
                    is_motif_group_aggregator=is_motif_group_aggregator,
                )

        if self.agg_rigid_embed:
            s = self.frame_to_node_broadcast(
                rigids_embed_flat,
                s,
                rigids_to_res_idx,
                rigids_mask
            )

        return rigids_embed_flat, s, framepair_embed, rigids_flat