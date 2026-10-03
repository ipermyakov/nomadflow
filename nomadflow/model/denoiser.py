import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment

from nomadflow.boltz.data import const

import nomadflow.openfold.utils.rigid_utils as ru
from nomadflow.openfold.utils.rigid_utils import Rigid
from nomadflow.stoch_interp.so3_utils import rotquat_to_rotvec

from nomadflow.model.denoiser_modules.embedder import Embedder
from nomadflow.model.denoiser_modules.utils import gather_helper
from nomadflow.model.denoiser_modules.ipa_trunk import IpaDenoiser
from nomadflow.model.denoiser_modules.pairformer import Pairformer
from nomadflow.model.modules.frame_transformer import (
    _project_rigid_groups_to_predicted_state as project_rigid_groups_to_predicted_state,
)


def calc_distogram(pos, min_bin, max_bin, num_bins):
    dists_2d = torch.linalg.norm(
        pos[:, :, None, :] - pos[:, None, :, :], axis=-1)[..., None]
    lower = torch.linspace(
        min_bin,
        max_bin,
        num_bins,
        device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    dgram = ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)
    return dgram


def batched_mask_select(data, mask, no_batch_dims):
    remaining_mask_dims = [i for i in range(mask.dim()) if i < no_batch_dims]
    assert len(remaining_mask_dims) > 0
    selected_per_batch = mask.sum(dim=remaining_mask_dims)
    assert (selected_per_batch == selected_per_batch.view(-1)[0]).all(), selected_per_batch

    batched_mask_dims = mask.shape[:no_batch_dims]
    return data[mask].view(*batched_mask_dims, *data.shape[no_batch_dims:])


def scaffold_candidate_mask(token_data, rigids_data):
    """Tokens eligible as scaffold partners for motif-copy assignment."""
    mask = ~token_data["token_is_copy_mask"]
    token_mask = token_data.get("token_mask")
    if token_mask is not None:
        mask = mask & token_mask.bool()

    aggregator_mask = rigids_data.get("rigids_is_motif_group_aggregator")
    if aggregator_mask is not None:
        token_to_rigid = token_data["token_to_rep_rigid"].long()
        token_is_aggregator = torch.gather(
            aggregator_mask.bool(),
            dim=1,
            index=token_to_rigid,
        )
        mask = mask & ~token_is_aggregator
    return mask


class IpaMultiRigidDenoiser(nn.Module):
    def __init__(self,
                 c_s=768,
                 c_cond=768,
                 c_z=128,
                 c_frame=256,
                 c_framepair=64,
                 c_hidden=16,
                 c_hidden_trig=32,
                 c_hidden_trig_embedder=64,
                 num_heads=16,
                 num_qk_points=8,
                 num_v_points=12,
                 num_blocks=12,
                 trans_preconditioning=False,
                 rot_preconditioning=True,
                 block_q=16,
                 block_k=64,
                 use_conditioned_ipa=True,
                 use_conditioned_rigid_transformer=False,
                 rigid_transformer_num_blocks=1,
                 rigid_transformer_num_heads=4,
                 rigid_transformer_rigid_updates=True,
                 rigid_transformer_agg_embed=True,
                 rigid_transformer_add_vanilla_transformer=False,
                 rigid_transformer_add_second_transformer=False,
                 rigid_transformer_add_full_transformer=False,
                 use_embedder_sc_rigid_transformer=True,
                 rel_quat_pair_updates=False,
                 z_broadcast=True,
                 compile_ipa=False,
                 use_ipa_gating=True,
                 ablate_ipa_down_z=False,
                 ipa_row_dropout_r=0.,
                 tfmr_row_dropout_r=0.,
                 cg_version=1,
                 use_qk_norm=True,
                 use_amp=True,
                 predict_final_rot=False,
                 direct_rot_vf_output=False,
                 learnable_noise_schedule=False,
                 use_pairformer=False,
                 rot_vf_scaling=1,
                 self_conditioning=True,
                 self_folding=False,
                 use_residue_indexing=True,
                 num_registers=0
                 ):
        super().__init__()

        # some compatibility code
        self.self_conditioning = self_conditioning
        self.lrange_k = 10000
        self.knn_k = 10000
        self.lrange_logn_scale = 10000
        self.lrange_logn_offset = 10000

        self.use_amp = use_amp
        self.self_folding = self_folding
        self.use_residue_indexing = use_residue_indexing

        self.ipa_denoiser = IpaDenoiser(
            c_s=c_s,
            c_cond=c_s,
            c_z=c_z,
            c_frame=c_frame,
            c_framepair=c_framepair,
            c_hidden=c_hidden // num_heads,
            c_hidden_trig=c_hidden_trig,
            num_heads=num_heads,
            num_qk_points=num_qk_points,
            num_v_points=num_v_points,
            num_blocks=num_blocks,
            coordinate_scaling=1 if trans_preconditioning else 0.1,
            block_q=block_q,
            block_k=block_k,
            use_conditioned_ipa=use_conditioned_ipa,
            use_conditioned_rigid_transformer=use_conditioned_rigid_transformer,
            rigid_transformer_num_blocks=rigid_transformer_num_blocks,
            rigid_transformer_num_heads=rigid_transformer_num_heads,
            rigid_transformer_rigid_updates=rigid_transformer_rigid_updates,
            rigid_transformer_agg_embed=rigid_transformer_agg_embed,
            rigid_transformer_add_vanilla_transformer=rigid_transformer_add_vanilla_transformer,
            rigid_transformer_add_second_transformer=rigid_transformer_add_second_transformer,
            rigid_transformer_add_full_transformer=rigid_transformer_add_full_transformer,
            rel_quat_pair_updates=rel_quat_pair_updates,
            z_broadcast=z_broadcast,
            compile_ipa=compile_ipa,
            use_ipa_gating=use_ipa_gating,
            ablate_ipa_down_z=ablate_ipa_down_z,
            ipa_row_dropout_r=ipa_row_dropout_r,
            tfmr_row_dropout_r=tfmr_row_dropout_r,
            use_qk_norm=use_qk_norm,
            predict_final_rot=predict_final_rot,
            direct_rot_vf_output=direct_rot_vf_output,
            detach_grad_pre_seq_pred=learnable_noise_schedule,
            num_aa=len(const.tokens)
        )

        self.embedder = Embedder(
            c_s=c_s,
            c_z=c_z,
            c_frame=c_frame,
            c_framepair=c_framepair,
            c_hidden=c_hidden_trig_embedder,
            block_q=block_q,
            block_k=block_k,
            use_sc_rigid_transformer=use_embedder_sc_rigid_transformer,
            rigid_transformer_add_vanilla_transformer=rigid_transformer_add_vanilla_transformer,
            rigid_transformer_add_second_transformer=rigid_transformer_add_second_transformer,
            rigid_transformer_add_full_transformer=rigid_transformer_add_full_transformer,
            use_ipa_gating=use_ipa_gating,
            ablate_ipa_down_z=ablate_ipa_down_z,
            use_qk_norm=use_qk_norm,
            use_self_folding=self_folding
        )

        if use_pairformer:
            self.pairformer = Pairformer(
                c_s=c_s,
                c_cond=c_cond,
                c_z=c_z,
            )
        else:
            self.pairformer = None

        self.c_s = c_s
        self.trans_preconditioning = trans_preconditioning
        self.rot_preconditioning = rot_preconditioning
        self.cg_version = cg_version
        self.direct_rot_vf_output = direct_rot_vf_output
        self.rot_vf_scaling = rot_vf_scaling

        if learnable_noise_schedule:
            self.rot_gamma_t = MonotonicIncreasingFn()
            self.trans_gamma_t = MonotonicIncreasingFn()

        # TODO: need to actually implement this at some point
        self.num_registers = num_registers
        if num_registers > 0:
            self.registers = nn.Parameter(
                torch.randn(self.num_registers, self.c_s) / 20.0
            )
        else:
            self.registers = None

    def forward(self, data, self_condition=None, self_folding=None, sanitize_motif_idx=False):
        token_data = data['token']
        rigids_data = data['rigids']

        if self_condition is not None:
            sc_rigids = self_condition['denoised_rigids']
        else:
            sc_rigids = None

        if self.self_folding and self_folding is not None:
            sf_rigids = self_folding['denoised_rigids']
        else:
            sf_rigids = None

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
            input_feats = self.embedder(
                token_mask=token_data['token_mask'],
                token_seq_idx=(
                    token_data['residue_idx'] if self.use_residue_indexing
                    else  token_data['token_seq_idx']),
                token_seq=token_data['res_type'],
                token_seq_mask=token_data['token_mask'],
                token_seq_noising_mask=token_data['seq_noising_mask'],
                token_chain_idx=token_data['asym_id'],
                # token_is_atomized_mask=token_data['token_is_atomized_mask'],
                token_is_unindexed_mask=token_data['token_is_unindexed_mask'],
                token_gather_idx=token_data['token_to_rep_rigid'],
                t=data['t'],
                rigids=rigids_data['rigids_t'],
                rigids_element=rigids_data['rigids_ref_element'],
                rigids_charge=rigids_data['rigids_ref_charge'],
                rigids_token_uid=rigids_data['rigids_to_token'],
                rigids_idx=rigids_data['rigids_sidechain_idx'],
                rigids_motif_group_id=rigids_data["motif_group_id"],
                rigids_mask=rigids_data['rigids_mask'],
                rigids_is_atomized_mask=rigids_data['rigids_is_atom_mask'],
                rigids_noising_mask=rigids_data['rigids_noising_mask'],
                rigids_is_motif_group_aggregator=rigids_data['rigids_is_motif_group_aggregator'],
                token_bonds=token_data['token_bonds'],
                sc_rigids=sc_rigids,
                sf_rigids=sf_rigids
            )

            input_feats['condition_embed'] = input_feats['time_condition_embed']
            input_feats['token_mask'] = token_data['token_mask']
            input_feats['token_gather_idx'] = token_data['token_to_rep_rigid']
            input_feats['rigids_motif_group_id'] = rigids_data['motif_group_id']
            input_feats['rigids_is_motif_group_aggregator'] = rigids_data['rigids_is_motif_group_aggregator']
            input_feats['t'] = data['t']

            if self.pairformer is not None:
                token_embed, edge_embed = self.pairformer(
                    s=input_feats['node_embed'],
                    z=input_feats['edge_embed'],
                    s_cond=input_feats['condition_embed'],
                    node_mask=input_feats['token_mask']
                )
                input_feats['node_embed'] = token_embed
                input_feats['edge_embed'] = edge_embed

            score_dict = self.ipa_denoiser(input_feats)

        rigids_out = score_dict['final_rigids']

        if self.rot_preconditioning:
            t = data['t']
            def scale_rot(rot_in, rot_out):
                rel_rot = rot_out.compose_q(rot_in.invert())
                rel_rotquat = rel_rot.get_quats()
                rel_rotvec = rotquat_to_rotvec(rel_rotquat.view(-1, 4)).view(*rel_rotquat.shape[:-1], -1)
                angle = torch.linalg.vector_norm(rel_rotvec + 1e-8, dim=-1)
                scaled_angle = angle * (1 - t)
                axis = F.normalize(rel_rotvec, dim=-1)
                scaled_rotquat = torch.cat([
                    torch.cos(scaled_angle/2)[..., None], torch.sin(scaled_angle/2)[..., None] * axis
                ], dim=-1)
                scaled_rot = ru.Rotation(quats=scaled_rotquat)
                new_rot = scaled_rot.compose_q(rot_in)
                return new_rot

            rigids_in = ru.Rigid.from_tensor_7(data['rigids']['rigids_t'])
            rots_in = rigids_in.get_rots()
            rots_out = rigids_out.get_rots()
            rigids_out = Rigid(
                rots=scale_rot(rots_in, rots_out),
                trans=rigids_out.get_trans()
            )
            rigids_out = project_rigid_groups_to_predicted_state(
                rigids_pred=rigids_out,
                rigids_ref=rigids_in,
                motif_group_id=data['rigids'].get('motif_group_id'),
                rigids_mask=data['rigids']['rigids_mask'],
                is_motif_group_aggregator=data['rigids'].get('rigids_is_motif_group_aggregator'),
            )

        seq_logits = score_dict['seq_logits']

        pred_seq = seq_logits[..., :-1].argmax(dim=-1)
        seq_noising_mask = token_data['seq_noising_mask']
        pred_seq = pred_seq * seq_noising_mask + token_data['seq'] * (~seq_noising_mask)

        if rigids_out.to_tensor_7().isnan().any() or pred_seq.isnan().any():
            print("caught a nan in forward")
            exit()

        ret = {}
        ret['denoised_rigids'] = rigids_out
        ret['decoded_seq_logits'] = seq_logits
        ret['pred_seq'] = pred_seq

        if self.direct_rot_vf_output:
            pred_rot_vf = score_dict['pred_rot_vf'].float() * self.rot_vf_scaling
        else:
            pred_rot_vf = None
        ret['pred_rot_vf'] = pred_rot_vf

        with torch.no_grad():
            token_rigids = gather_helper(rigids_out.to_tensor_7(), token_data['token_to_rep_rigid'])
            token_rigids = ru.Rigid.from_tensor_7(token_rigids)

            motif_rigid_mask = token_data['token_is_copy_mask']
            protein_rigid_mask = scaffold_candidate_mask(token_data, rigids_data)
            dist_mask = motif_rigid_mask[..., None] & protein_rigid_mask[..., None, :]
            res_CA_pos = token_rigids.get_trans()
            trans_dist = torch.cdist(res_CA_pos, res_CA_pos)
            trans_dist[~dist_mask] = 1e6
            closest_neighbors = torch.argsort(trans_dist)
            motif_idx = closest_neighbors[..., 0]
            ret['motif_idx'] = motif_idx
            if sanitize_motif_idx and not self.training:
                # there are rare cases where
                # two motif residues will be assigned to the same residue
                # this can sometimes happen because of a bad generated structure
                # if prompted, we patch this out using linear sum assignment
                # with the distance matrix as the cost matrix
                for i, sample_motif_rigid_mask in enumerate(motif_rigid_mask):
                    motif_assignments = motif_idx[i, sample_motif_rigid_mask]
                    if motif_assignments.unique().numel() == motif_assignments.numel():
                        continue
                    # if we have duplicate motif assignments, resolve them via linear sum assignment
                    cost_mat = trans_dist[i, sample_motif_rigid_mask].numpy(force=True)
                    row_assign, col_assign = linear_sum_assignment(cost_mat)
                    motif_idx[i, sample_motif_rigid_mask] = torch.as_tensor(col_assign, device=motif_idx.device)
                    # print(motif_assignments, col_assign)

        return ret


class MonotonicIncreasingFn(nn.Module):
    def __init__(self, c_hidden=1024):
        super().__init__()
        self.c_hidden = c_hidden
        self.l1 = nn.Linear(1, 1, bias=False)
        self.l2 = nn.Linear(1, c_hidden)
        self.l3 = nn.Linear(c_hidden, 1)

    def _forward(self, l):
        l = F.linear(l, torch.abs(self.l1.weight))
        out = F.linear(l, torch.abs(self.l2.weight), self.l2.bias)
        out = torch.sigmoid(out)
        out = F.linear(out, torch.abs(self.l3.weight))
        return l + out

    def forward(self, x, l):
        shift_scale = self._forward(l)
        ret = x / (x * (1 - shift_scale) + shift_scale)
        # print(x, l, shift_scale, ret)
        return ret
