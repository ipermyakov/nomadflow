import torch
import torch.nn as nn
import torch.nn.functional as F
import functools as fn

import nomadflow.openfold.utils.rigid_utils as ru
from nomadflow.openfold.layers.layers import InvariantPointAttention, Dropout
from nomadflow.openfold.layers.layers_v2 import (
    Linear,
    ConditionedInvariantPointAttention,
    BackboneUpdate,
    LayerNorm,
    ConditionedTransition,
)
from nomadflow.model.modules.attention import ConditionedTransformerPairBias
from nomadflow.model.modules.pair_modules import ConditionedPairUpdate
from nomadflow.model.modules.frame_transformer import (
    ScatterUpdate,
    SequenceFrameTransformerUpdate,
    ConditionedSequenceFrameTransformerUpdate,
)
from nomadflow.model.denoiser_modules.utils import gather_helper


class TorsionAngles(nn.Module):
    def __init__(self, c, num_torsions, eps=1e-8):
        super(TorsionAngles, self).__init__()

        self.c = c
        self.eps = eps
        self.num_torsions = num_torsions

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        # TODO: Remove after published checkpoint is updated without these weights.
        self.linear_3 = Linear(self.c, self.c, init="final")
        self.linear_final = Linear(
            self.c, self.num_torsions * 2, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)

        s = s + s_initial
        unnormalized_s = self.linear_final(s)
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(unnormalized_s ** 2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        normalized_s = unnormalized_s / norm_denom

        return unnormalized_s, normalized_s


class SeqPredictor(nn.Module):
    def __init__(self, c_s, c_frame, n_aa=21):
        super().__init__()
        self.ln = LayerNorm(c_frame)
        self.scatter_update = ScatterUpdate(c_s, c_frame)
        self.out = Linear(c_s, n_aa)
        self.c_s = c_s

    def forward(self, rigids_embed_flat, rigids_to_res_idx, rigids_mask, out):
        rigids_embed_flat = self.ln(rigids_embed_flat)
        seq_embed = self.scatter_update(
            rigids_embed_flat,
            out,
            rigids_to_res_idx,
            rigids_mask
        )
        return self.out(seq_embed)


class IpaDenoiser(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_cond=256,
                 c_z=128,
                 c_frame=64,
                 c_framepair=64,
                 c_hidden=256,
                 c_hidden_trig=32,
                 num_heads=8,
                 num_qk_points=8,
                 num_v_points=12,
                 num_blocks=4,
                 coordinate_scaling=0.1,
                 block_q=32,
                 block_k=128,
                 use_conditioned_ipa=True,
                 use_conditioned_rigid_transformer=False,
                 rigid_transformer_num_blocks=1,
                 rigid_transformer_num_heads=4,
                 rigid_transformer_rigid_updates=False,
                 rigid_transformer_agg_embed=True,
                 rigid_transformer_add_vanilla_transformer=False,
                 rigid_transformer_add_second_transformer=False,
                 rigid_transformer_add_full_transformer=False,
                 rel_quat_pair_updates=False,
                 z_broadcast=False,
                 compile_ipa=False,
                 use_ipa_gating=False,
                 ablate_ipa_down_z=False,
                 ipa_row_dropout_r=0.,
                 tfmr_row_dropout_r=0.,
                 use_qk_norm=False,
                 predict_final_rot=False,
                 direct_rot_vf_output=False,
                 detach_grad_pre_seq_pred=False,
                 num_aa=21,
                 accumulate_rot_vf_output=False
                 ):
        super().__init__()
        # self.diffuser = diffuser
        self.use_conditioned_ipa = use_conditioned_ipa
        self.use_conditioned_rigid_transformer = use_conditioned_rigid_transformer
        self.rigid_transformer_rigid_updates = rigid_transformer_rigid_updates

        self.scale_pos = lambda x: x * coordinate_scaling
        self.scale_rigids = lambda x: x.apply_trans_fn(self.scale_pos)

        self.unscale_pos = lambda x: x / coordinate_scaling
        self.unscale_rigids = lambda x: x.apply_trans_fn(self.unscale_pos)
        self.trunk = nn.ModuleDict()

        self.num_blocks = num_blocks
        self.block_q = block_q
        self.block_k = block_k
        self.num_aa = num_aa
        self.accumulate_rot_vf_output = accumulate_rot_vf_output

        for b in range(num_blocks):
            if use_conditioned_ipa:
                self.trunk[f'ipa_{b}'] = ConditionedInvariantPointAttention(
                    c_s=c_s,
                    c_cond=c_cond,
                    c_z=c_z,
                    c_hidden=c_hidden,
                    num_heads=num_heads,
                    num_qk_points=num_qk_points,
                    num_v_points=num_v_points,
                    use_qk_norm=use_qk_norm,
                )
            else:
                self.trunk[f'ipa_{b}'] = InvariantPointAttention(
                    c_s=c_s,
                    c_z=c_z,
                    c_hidden=c_hidden,
                    num_heads=num_heads,
                    num_qk_points=num_qk_points,
                    num_v_points=num_v_points,
                    use_qk_norm=use_qk_norm
                )
            if compile_ipa:
                self.trunk[f'ipa_{b}'].compile()
            self.trunk[f'ipa_row_dropout_{b}'] = Dropout(ipa_row_dropout_r, -2)

            self.trunk[f'tfmr_{b}'] = ConditionedTransformerPairBias(
                c_s=c_s,
                c_cond=c_cond,
                c_z=c_z,
                no_heads=4,
                n_layers=2,
                row_dropout=tfmr_row_dropout_r,
                use_qk_norm=use_qk_norm,
            )
            self.trunk[f'post_tfmr_{b}'] = Linear(
                c_s, c_s, init="final", bias=False)
            self.trunk[f'transition_{b}'] = ConditionedTransition(
                c_s=c_s,
                c_cond=c_cond
            )

            if self.use_conditioned_rigid_transformer:
                self.trunk[f'rigids_tfmr_{b}'] = ConditionedSequenceFrameTransformerUpdate(
                    c_s=c_s,
                    c_z=c_z,
                    c_frame=c_frame,
                    c_framepair=c_framepair,
                    num_heads=rigid_transformer_num_heads,
                    num_qk_points=num_qk_points,
                    num_v_points=num_v_points,
                    block_q=block_q,
                    block_k=block_k,
                    n_blocks=rigid_transformer_num_blocks,
                    do_rigid_updates=rigid_transformer_rigid_updates,
                    agg_rigid_embed=rigid_transformer_agg_embed,
                    broadcast_pairs=z_broadcast,
                    add_vanilla_transformer=rigid_transformer_add_vanilla_transformer,
                    add_second_transformer=rigid_transformer_add_second_transformer,
                    use_ipa_gating=use_ipa_gating,
                    ablate_ipa_down_z=ablate_ipa_down_z,
                    use_qk_norm=use_qk_norm,
                    use_conditioned_ipa=use_conditioned_ipa,
                )
            else:
                self.trunk[f'rigids_tfmr_{b}'] = SequenceFrameTransformerUpdate(
                    c_s=c_s,
                    c_z=c_z,
                    c_frame=c_frame,
                    c_framepair=c_framepair,
                    num_heads=4, # num_heads,
                    num_qk_points=num_qk_points,
                    num_v_points=num_v_points,
                    block_q=block_q,
                    block_k=block_k,
                    n_blocks=rigid_transformer_num_blocks,
                    do_rigid_updates=rigid_transformer_rigid_updates,
                    agg_rigid_embed=rigid_transformer_agg_embed,
                    broadcast_pairs=z_broadcast,
                    add_vanilla_transformer=rigid_transformer_add_vanilla_transformer,
                    add_second_transformer=rigid_transformer_add_second_transformer,
                    add_full_transformer=rigid_transformer_add_full_transformer,
                    use_ipa_gating=use_ipa_gating,
                    ablate_ipa_down_z=ablate_ipa_down_z,
                    use_qk_norm=use_qk_norm,
                )

            if not rigid_transformer_rigid_updates:
                self.trunk[f'rigids_update_{b}'] = BackboneUpdate(c_frame)

            if direct_rot_vf_output and accumulate_rot_vf_output:
                self.trunk[f'rot_vf_update_{b}'] = nn.Sequential(
                    LayerNorm(c_frame),
                    Linear(c_frame, 3, bias=False, init='final')
                )

            if b < num_blocks-1:
                # No edge update on the last block.
                self.trunk[f'edge_transition_{b}'] = ConditionedPairUpdate(
                    c_s=c_s,
                    c_z=c_z,
                    c_hidden=c_hidden_trig,
                    no_heads=4,
                    include_rel_quat=rel_quat_pair_updates,
                    use_qk_norm=use_qk_norm
                )

        self.torsion_pred = TorsionAngles(c_s, 1)
        self.seq_pred = SeqPredictor(c_s, c_frame, n_aa=num_aa)
        self.detach_grad_pre_seq_pred = detach_grad_pre_seq_pred

        if predict_final_rot and not accumulate_rot_vf_output:
            self.final_rot_head = nn.Sequential(
                LayerNorm(c_frame),
                Linear(c_frame, c_frame, bias=False),
                nn.ReLU(),
                Linear(c_frame, 3, bias=False)
            )
        else:
            self.final_rot_head = None

        if direct_rot_vf_output:
            self.rot_vf_head = nn.Sequential(
                LayerNorm(c_frame),
                Linear(c_frame, c_frame, bias=False),
                nn.ReLU(),
                Linear(c_frame, 3, bias=False, init='final')
            )
        else:
            self.rot_vf_head = None

    def forward(self, input_feats):
        node_embed = input_feats['node_embed']
        node_mask = input_feats['token_mask'].type(torch.float32)
        condition_embed = input_feats['condition_embed']

        edge_embed = input_feats['edge_embed']
        edge_mask = node_mask[..., None] * node_mask[..., None, :]

        framepair_embed = input_feats['framepair_embed']

        init_rigids = input_feats['rigids_t']
        rigids_embed_flat = input_feats['rigids_embed']
        rigids_token_uid = input_feats['rigids_token_uid']
        rigids_mask_flat = input_feats['rigids_mask']
        rigids_noising_mask_flat = input_feats['rigids_noising_mask']
        motif_group_id_flat = input_feats.get('rigids_motif_group_id')
        motif_group_aggregator_flat = input_feats.get('rigids_is_motif_group_aggregator')

        to_queries = input_feats['to_queries']
        to_keys = input_feats['to_keys']
        to_pairs = input_feats['to_pairs']

        curr_rigids = self.scale_rigids(init_rigids)
        node_embed = node_embed * node_mask[..., None]
        rigids_to_nodes = fn.partial(gather_helper, token_gather_idx=input_feats['token_gather_idx'])

        if self.accumulate_rot_vf_output:
            pred_rot_vf = 0
        else:
            pred_rot_vf = None

        # Main trunk
        for b in range(self.num_blocks):
            curr_rigids_tensor_7 = curr_rigids.to_tensor_7()
            token_rigids = ru.Rigid.from_tensor_7(rigids_to_nodes(curr_rigids_tensor_7))
            if self.use_conditioned_ipa:
                ipa_embed = self.trunk[f'ipa_{b}'](
                    s=node_embed,
                    cond=condition_embed,
                    z=edge_embed,
                    r=token_rigids,
                    mask=node_mask)
            else:
                ipa_embed = self.trunk[f'ipa_{b}'](
                    s=node_embed,
                    z=edge_embed,
                    r=token_rigids,
                    mask=node_mask)
            node_embed = (node_embed + ipa_embed) * node_mask[..., None]

            seq_tfmr_out = self.trunk[f'tfmr_{b}'](
                node_embed, condition_embed, edge_embed, node_mask)
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = node_embed * node_mask[..., None]

            node_embed = node_embed + self.trunk[f'transition_{b}'](node_embed, condition_embed)
            node_embed = node_embed * node_mask[..., None]

            rigids_embed_flat, node_embed, framepair_embed, curr_rigids = self.trunk[f'rigids_tfmr_{b}'](
                node_embed,
                edge_embed,
                framepair_embed,
                curr_rigids,
                rigids_embed_flat,
                rigids_token_uid,
                rigids_mask_flat,
                rigids_noising_mask_flat,
                motif_group_id_flat,
                to_queries,
                to_keys,
                to_pairs,
                motif_group_aggregator_flat
            )
            if not self.rigid_transformer_rigid_updates:
                rigid_update = self.trunk[f'rigids_update_{b}'](
                    rigids_embed_flat * rigids_noising_mask_flat[..., None])
                curr_rigids = curr_rigids.compose_q_update_vec(
                    rigid_update *  rigids_noising_mask_flat[..., None])

            if self.accumulate_rot_vf_output:
                assert pred_rot_vf is not None
                rot_vf_update = self.trunk[f'rot_vf_update_{b}'](
                    rigids_embed_flat * rigids_noising_mask_flat[..., None]
                ) * rigids_noising_mask_flat[..., None]
                pred_rot_vf = pred_rot_vf + rot_vf_update

            if b < self.num_blocks-1:
                curr_rigids_tensor_7 = curr_rigids.to_tensor_7()
                token_rigids = ru.Rigid.from_tensor_7(rigids_to_nodes(curr_rigids_tensor_7))
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed, token_rigids, edge_mask)
                edge_embed *= edge_mask[..., None]

        if self.detach_grad_pre_seq_pred:
            seq_logits = self.seq_pred(
                rigids_embed_flat.detach(),
                rigids_token_uid,
                rigids_mask_flat,
                out=torch.zeros_like(node_embed)
            )
        else:
            seq_logits = self.seq_pred(
                rigids_embed_flat,
                rigids_token_uid,
                rigids_mask_flat,
                out=torch.zeros_like(node_embed)
            )

        if self.final_rot_head is not None:
            rotvec = self.final_rot_head(rigids_embed_flat)
            axis = F.normalize(rotvec, dim=-1)
            angle = F.sigmoid(torch.linalg.vector_norm(rotvec + 1e-8, dim=-1)) * 2 * torch.pi
            angle = angle * rigids_noising_mask_flat * rigids_mask_flat
            rotquat = torch.cat([
                torch.cos(angle/2)[..., None], torch.sin(angle/2)[..., None] * axis
            ], dim=-1)
            rel_rot = ru.Rotation(quats=rotquat)
            new_rot = rel_rot.compose_q(init_rigids.get_rots())
            curr_rigids = ru.Rigid(
                rots=new_rot,
                trans=curr_rigids.get_trans()
            )

        if self.rot_vf_head is not None:
            pred_rot_vf = self.rot_vf_head(rigids_embed_flat)
        else:
            pred_rot_vf = None

        if pred_rot_vf is not None:
            axis = F.normalize(pred_rot_vf, dim=-1)
            angle = torch.linalg.vector_norm(pred_rot_vf + 1e-8, dim=-1)
            angle = angle * rigids_noising_mask_flat * rigids_mask_flat
            angle = angle * (1 - input_feats['t'])
            rotquat = torch.cat([
                torch.cos(angle/2)[..., None], torch.sin(angle/2)[..., None] * axis
            ], dim=-1)
            rel_rot = ru.Rotation(quats=rotquat)
            new_rot = rel_rot.compose_q(init_rigids.get_rots())
            curr_rigids = ru.Rigid(
                rots=new_rot,
                trans=curr_rigids.get_trans()
            )

            if input_feats['n_padding'] > 0:
                pred_rot_vf = pred_rot_vf[..., :-input_feats['n_padding'], :]

        if input_feats['n_padding'] > 0:
            curr_rigids = curr_rigids[..., :-input_feats['n_padding']]
        curr_rigids = self.unscale_rigids(curr_rigids)


        _, psi_pred = self.torsion_pred(node_embed)
        model_out = {
            'psi': psi_pred,
            'final_rigids': curr_rigids,
            'pred_rot_vf': pred_rot_vf,
            'node_embed': node_embed,
            'seq_logits': seq_logits
        }
        return model_out

