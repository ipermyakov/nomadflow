import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import functools as fn
from typing import cast

from nomadflow.boltz.data import const
from nomadflow.openfold.layers.layers_v2 import AdaLN, Linear
from nomadflow.model.modules.frame_transformer import (
    ScatterUpdate, FramepairEmbedder,
    SequenceFrameTransformerUpdate, ConditionedSequenceFrameTransformerUpdate,
    get_indexing_matrix, single_to_keys, pairs_to_framepairs,
)
from nomadflow.model.modules.pair_modules import MultiRigidPairEmbedder
import nomadflow.openfold.utils.rigid_utils as ru
from nomadflow.model.denoiser_modules.utils import gather_helper


def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size//2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size)))# .to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size)))# .to(indices.device)
    pos_embedding = torch.cat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def get_timestep_embedding_flexshape(timesteps, embedding_dim, max_positions=10000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[..., None] * emb.view(*[1 for _ in timesteps.shape], -1)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb



class Embedder(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 c_frame,
                 c_framepair,
                 c_hidden=64,
                 num_heads=4,
                 block_q=32,
                 block_k=128,
                 n_tfmr_blocks=3,
                 n_pair_embed_blocks=2,
                 index_embed_size=256,
                 max_rigids_idx=3+14,
                 use_sc_rigid_transformer=False,
                 use_self_folding=False,
                 rigid_transformer_add_vanilla_transformer=False,
                 rigid_transformer_add_second_transformer=False,
                 rigid_transformer_add_full_transformer=False,
                 use_ipa_gating=False,
                 ablate_ipa_down_z=False,
                 use_qk_norm=False,
                 max_motif_group_id=32,
                 restype_dict=const.token_ids,
                 num_elements=const.num_elements,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_sidechain = c_frame
        self.block_q = block_q
        self.block_k = block_k

        self.timestep_embedder = fn.partial(
            get_timestep_embedding_flexshape,
            embedding_dim=index_embed_size
        )
        self.time_condition_embed = Linear(index_embed_size, c_s, bias=False)
        self.index_embedder = fn.partial(
            get_index_embedding,
            embed_size=index_embed_size
        )

        self.restype_dict = restype_dict
        self.num_aa = len(restype_dict)
        self.mask_token = restype_dict['UNK']
        self.node_init = Linear(index_embed_size, self.c_s, bias=False)
        self.node_seq_embed = Linear(
            self.num_aa, self.c_s, bias=False
        )
        # self.node_is_atomized_embed = Linear(1, c_s, bias=False)
        self.node_is_unindexed_embed = Linear(1, c_s, bias=False)
        self.node_motif_group_embed = nn.Embedding(max_motif_group_id + 1, c_s)
        nn.init.zeros_(self.node_motif_group_embed.weight[0])
        self.max_motif_group_id = max_motif_group_id

        self.rigid_init = Linear(self.c_s, c_frame, bias=False)
        self.rigid_time_embed = Linear(index_embed_size, c_frame, bias=False)
        self.rigid_idx_embed = nn.Embedding(max_rigids_idx, c_frame)
        self.rigid_motif_group_embed = nn.Embedding(max_motif_group_id + 1, c_frame)
        nn.init.zeros_(self.rigid_motif_group_embed.weight[0])
        self.rigid_motif_group_aggregator_embed = nn.Embedding(2, c_frame)
        nn.init.zeros_(self.rigid_motif_group_aggregator_embed.weight[0])
        self.rigid_is_atomized_embed = Linear(1, c_frame, bias=False)
        self.rigid_element_embed = nn.Embedding(num_elements, c_frame)
        self.rigid_charge_embed = Linear(1, c_frame, bias=False)
        # self.rigid_is_unindexed_embed = Linear(1, c_frame, bias=False)

        self.framepair_init = FramepairEmbedder(
            c_framepair
        )
        self.frame_tfmr = SequenceFrameTransformerUpdate(
            c_s,
            c_z,
            c_frame,
            c_framepair,
            num_heads=num_heads,
            n_blocks=n_tfmr_blocks,
            block_q=block_q,
            block_k=block_k,
            do_rigid_updates=False,
            broadcast_singles=True,
            broadcast_pairs=True,
            framepair_init=True,
            framepair_ffn=True,
            add_vanilla_transformer=rigid_transformer_add_vanilla_transformer,
            add_second_transformer=rigid_transformer_add_second_transformer,
            add_full_transformer=rigid_transformer_add_full_transformer,
            use_ipa_gating=use_ipa_gating,
            ablate_ipa_down_z=ablate_ipa_down_z,
            use_qk_norm=use_qk_norm,
        )

        self.use_sc_rigid_transformer = use_sc_rigid_transformer
        if use_sc_rigid_transformer:
            self.sc_framepair_init = FramepairEmbedder(
                c_framepair
            )
            self.sc_frame_tfmr = SequenceFrameTransformerUpdate(
                c_s,
                c_z,
                c_frame,
                c_framepair,
                num_heads=num_heads,
                n_blocks=n_tfmr_blocks,
                block_q=block_q,
                block_k=block_k,
                do_rigid_updates=False,
                broadcast_singles=True,
                broadcast_pairs=True,
                framepair_init=True,
                framepair_ffn=True,
                add_vanilla_transformer=rigid_transformer_add_vanilla_transformer,
                add_second_transformer=rigid_transformer_add_second_transformer,
                use_ipa_gating=use_ipa_gating,
                ablate_ipa_down_z=ablate_ipa_down_z,
                use_qk_norm=use_qk_norm,
            )
            self.node_adaln = AdaLN(c_s, c_s)
            self.frame_adaln = AdaLN(c_frame, c_frame)
            self.framepair_adaln = AdaLN(c_framepair, c_framepair)

        self.use_self_folding = use_self_folding
        if use_self_folding:
            self.sf_framepair_init = FramepairEmbedder(
                c_framepair
            )
            self.sf_frame_tfmr = SequenceFrameTransformerUpdate(
                c_s,
                c_z,
                c_frame,
                c_framepair,
                num_heads=num_heads,
                n_blocks=n_tfmr_blocks,
                block_q=block_q,
                block_k=block_k,
                do_rigid_updates=False,
                broadcast_singles=True,
                broadcast_pairs=True,
                framepair_init=True,
                framepair_ffn=True,
                add_vanilla_transformer=rigid_transformer_add_vanilla_transformer,
                add_second_transformer=rigid_transformer_add_second_transformer,
                use_ipa_gating=use_ipa_gating,
                ablate_ipa_down_z=ablate_ipa_down_z,
                use_qk_norm=use_qk_norm,
            )
            self.sf_node_adaln = AdaLN(c_s, c_s)
            self.sf_frame_adaln = AdaLN(c_frame, c_frame)
            self.sf_framepair_adaln = AdaLN(c_framepair, c_framepair)

        self.pair_embedder = MultiRigidPairEmbedder(
            c_z,
            c_hidden,
            no_blocks=n_pair_embed_blocks,
            use_qk_norm=use_qk_norm,
            use_self_folding=use_self_folding
        )

    def _gen_node_features(
        self,
        seq,
        seq_idx,
        seq_noising_mask,
        seq_mask,
        rigids,
        token_gather_idx,
        is_unindexed_mask,
        token_motif_group_id,
        # is_atomized_mask,
        sc_rigids=None,
        sf_rigids=None
    ):
        rigids_to_nodes = fn.partial(gather_helper, token_gather_idx=token_gather_idx)
        # print(token_gather_idx, rigids.shape)

        node_rigids = rigids_to_nodes(rigids)
        if sc_rigids is not None:
            # node_sc_rigids = rigids_to_nodes(sc_rigids)
            node_sc_rigids = ru.Rigid.from_tensor_7(rigids_to_nodes(sc_rigids.to_tensor_7()))
        else:
            node_sc_rigids = None
        if sf_rigids is not None:
            # node_sc_rigids = rigids_to_nodes(sc_rigids)
            node_sf_rigids = ru.Rigid.from_tensor_7(rigids_to_nodes(sf_rigids.to_tensor_7()))
        else:
            node_sf_rigids = None

        node_seq_idx_embed = self.index_embedder(seq_idx)
        node_init = self.node_init(node_seq_idx_embed) * (~is_unindexed_mask[..., None])
        visible_seq = seq * seq_mask + self.mask_token * (~seq_mask)
        visible_seq = visible_seq * (~seq_noising_mask) + self.mask_token * seq_noising_mask
        # print(seq_idx)
        # print(seq, seq_mask, seq_noising_mask, visible_seq)
        # print(is_unindexed_mask)
        seq_embed = F.one_hot(visible_seq, num_classes=self.num_aa).float()
        node_init = (
            node_init
            + self.node_seq_embed(seq_embed)
            # + self.node_is_atomized_embed(is_atomized_mask[..., None].float())
            + self.node_is_unindexed_embed(is_unindexed_mask[..., None].float())
            + self.node_motif_group_embed(token_motif_group_id.long().clamp(0, self.max_motif_group_id))
        )
        return {
            "node_init": node_init,
            "node_rigids": ru.Rigid.from_tensor_7(node_rigids),
            "node_sc_rigids": node_sc_rigids,
            "node_sf_rigids": node_sf_rigids,
        }

    def _gen_rigid_features(
        self,
        node_init,
        rigids_element,
        rigids_charge,
        t,
        rigids_token_uid,
        rigids_idx,
        rigids_is_atomized_mask,
        rigids_motif_group_id,
        rigids_is_motif_group_aggregator,
    ):
        nodes_to_rigids = fn.partial(gather_helper, token_gather_idx=rigids_token_uid)

        rigids_init = self.rigid_init(nodes_to_rigids(node_init))
        time_embed = self.rigid_time_embed(self.timestep_embedder(t))
        rigids_idx_embed = self.rigid_idx_embed(rigids_idx)
        rigids_motif_group_embed = self.rigid_motif_group_embed(rigids_motif_group_id.long())
        aggregator_embed = self.rigid_motif_group_aggregator_embed(rigids_is_motif_group_aggregator.long())
        is_atomized_embed = self.rigid_is_atomized_embed(rigids_is_atomized_mask[..., None].float())
        element_mask = (rigids_element != -1)
        element_embed = self.rigid_element_embed(rigids_element * element_mask) * element_mask[..., None]
        charge_embed = self.rigid_charge_embed(rigids_charge.unsqueeze(-1))

        rigids_init = (
            rigids_init
            + time_embed
            + rigids_idx_embed
            + rigids_motif_group_embed
            + aggregator_embed
            + is_atomized_embed
            + element_embed
            + charge_embed
        )
        return rigids_init


    def _pad_rigid_features(
        self,
        rigids_data,
        n_padding: int
    ):
        padded_data = {}
        for key, value in rigids_data.items():
            if key in ("rigids_t", "sc_rigids", "sf_rigids"):
                rigids = value
                if rigids is not None:
                    n_batch = int(rigids.shape[0])
                    rigids_padding = ru.Rigid.identity(
                        shape=cast(tuple[int], (n_batch, n_padding)),
                        device=rigids._trans.device,
                        fmt="quat",
                    )
                    # we don't use ru.Rigid.cat cuz this automatically converts stuff to rotmats...
                    # we need to stay in quats
                    rigids = ru.Rigid(
                        trans=torch.cat([rigids.get_trans(), rigids_padding.get_trans()], dim=-2),
                        rots=ru.Rotation(
                            quats=torch.cat([
                                rigids.get_rots().get_quats(), rigids_padding.get_rots().get_quats()
                            ], dim=-2)
                        )
                    )
                    padded_data[key] = rigids
                else:
                    padded_data[key] = None
            elif key == 'rigids_init':
                padded_data[key] = F.pad(value, (0, 0, 0, n_padding), value=False)
            elif value.dtype == torch.bool:
                padded_data[key] = F.pad(value, (0, n_padding), value=False)
            elif value.dtype == torch.long or value.dtype == torch.float:
                padded_data[key] = F.pad(value, (0, n_padding), value=0)
        return padded_data


    def forward(
            self,
            *,
            token_mask,
            token_seq,
            token_seq_idx,
            token_seq_noising_mask,
            token_seq_mask,
            token_chain_idx,
            # token_is_atomized_mask,
            token_is_unindexed_mask,
            token_gather_idx,
            t,
            rigids,
            rigids_element,
            rigids_charge,
            rigids_token_uid,
            rigids_idx,
            rigids_motif_group_id,
            rigids_mask,
            rigids_noising_mask,
            rigids_is_atomized_mask,
            token_bonds,
            rigids_is_motif_group_aggregator,
            sc_rigids=None,
            sf_rigids=None
        ):

        # build the indexing matrices
        n_batch = rigids.shape[0]
        n_rigids = rigids.shape[1]
        n_padding = (self.block_q - n_rigids % self.block_q) % self.block_q
        n_attn_blocks = (n_rigids + n_padding) // self.block_q

        indexing_matrix = get_indexing_matrix(
            n_attn_blocks,
            W=self.block_q,
            H=self.block_k,
            device=t.device
        )
        to_queries = lambda x: x.view(n_batch, n_attn_blocks, self.block_q, -1)
        to_keys = fn.partial(
            single_to_keys, indexing_matrix=indexing_matrix, W=self.block_q, H=self.block_k
        )
        to_pairs = fn.partial(
            pairs_to_framepairs, indexing_matrix=indexing_matrix, W=self.block_q, H=self.block_k
        )

        # Derive token-level motif groups from all rigids in each token.
        # We use max aggregation so any non-zero rigid motif id is preserved.
        token_motif_group_id = torch.zeros_like(token_seq_idx, dtype=torch.long).scatter_reduce(
            dim=1,
            index=rigids_token_uid.long(),
            src=rigids_motif_group_id.long(),
            reduce="amax",
            include_self=True,
        )

        # compute node embeddings
        node_data = self._gen_node_features(
            token_seq,
            token_seq_idx,
            token_seq_noising_mask,
            token_seq_mask,
            rigids,
            token_gather_idx,
            token_is_unindexed_mask,
            token_motif_group_id,
            # token_is_atomized_mask,
            sc_rigids=sc_rigids,
            sf_rigids=sf_rigids
        )
        node_init = node_data['node_init']

        edge_embed = self.pair_embedder(
            node_data['node_rigids'],
            token_mask,
            token_seq_idx,
            token_chain_idx,
            token_is_unindexed_mask,
            token_bonds,
            sc_rigids=node_data['node_sc_rigids'],
            sf_rigids=node_data['node_sf_rigids'],
            token_motif_group_id=token_motif_group_id,
        )

        rigids_init = self._gen_rigid_features(
            node_init,
            rigids_element,
            rigids_charge,
            t,
            rigids_token_uid,
            rigids_idx,
            rigids_is_atomized_mask,
            rigids_motif_group_id,
            rigids_is_motif_group_aggregator,
        )

        rigids_inputs = {
            "rigids_t": ru.Rigid.from_tensor_7(rigids),
            "sc_rigids": sc_rigids if sc_rigids is not None else None,
            "sf_rigids": sf_rigids if sf_rigids is not None else None,
            "rigids_init": rigids_init,
            "rigids_mask": rigids_mask,
            "rigids_token_uid": rigids_token_uid,
            "rigids_motif_group_id": rigids_motif_group_id.long(),
            "rigids_is_motif_group_aggregator": rigids_is_motif_group_aggregator,
            "rigids_is_atomized_mask": rigids_is_atomized_mask,
            "rigids_noising_mask": rigids_noising_mask
        }
        # we need to pad this so we can use the efficient seq-local blocks
        rigids_data = self._pad_rigid_features(
            rigids_inputs,
            n_padding
        )
        rigids = rigids_data['rigids_t']
        rigids_init = rigids_data['rigids_init']
        rigids_token_uid = rigids_data['rigids_token_uid']
        rigids_mask = rigids_data['rigids_mask']
        rigids_motif_group_id = rigids_data["rigids_motif_group_id"]
        rigids_is_motif_group_aggregator = rigids_data["rigids_is_motif_group_aggregator"]

        framepair_init = self.framepair_init(
            rigids,
            rigids_token_uid,
            rigids_mask,
            rigids_motif_group_id,
            to_queries,
            to_keys
        )

        rigids_embed, node_embed, framepair_embed, _ = self.frame_tfmr(
            node_init,
            edge_embed,
            framepair_init,
            rigids,
            rigids_init,
            rigids_token_uid,
            rigids_mask,
            None,
            None,
            to_queries,
            to_keys,
            to_pairs,
            rigids_is_motif_group_aggregator,
        )

        if self.use_sc_rigid_transformer:
            sc_rigids = rigids_data['sc_rigids']
            if sc_rigids is not None:
                sc_framepair_init = self.sc_framepair_init(
                    sc_rigids,
                    rigids_token_uid,
                    rigids_mask,
                    rigids_motif_group_id,
                    to_queries,
                    to_keys
                )
                sc_rigids_embed_flat, sc_node_embed, sc_framepair_embed, _ = self.sc_frame_tfmr(
                    node_init,
                    edge_embed,
                    sc_framepair_init,
                    sc_rigids,
                    rigids_init,
                    rigids_token_uid,
                    rigids_mask,
                    None,
                    None,
                    to_queries,
                    to_keys,
                    to_pairs,
                    rigids_is_motif_group_aggregator,
                )
            else:
                sc_rigids_embed_flat = torch.zeros_like(rigids_init)
                sc_node_embed = torch.zeros_like(node_init)
                sc_framepair_embed = torch.zeros_like(framepair_embed)

            node_embed = self.node_adaln(node_embed, sc_node_embed)
            rigids_embed = self.frame_adaln(rigids_embed, sc_rigids_embed_flat)
            framepair_embed = self.framepair_adaln(framepair_embed, sc_framepair_embed)

        if self.use_self_folding:
            sf_rigids = rigids_data['sf_rigids']
            if sf_rigids is not None:
                sf_framepair_init = self.sf_framepair_init(
                    sf_rigids,
                    rigids_token_uid,
                    rigids_mask,
                    rigids_motif_group_id,
                    to_queries,
                    to_keys
                )
                sf_rigids_embed_flat, sf_node_embed, sf_framepair_embed, _ = self.sf_frame_tfmr(
                    node_init,
                    edge_embed,
                    sf_framepair_init,
                    sf_rigids,
                    rigids_init,
                    rigids_token_uid,
                    rigids_mask,
                    None,
                    None,
                    to_queries,
                    to_keys,
                    to_pairs,
                    rigids_is_motif_group_aggregator,
                )
            else:
                sf_rigids_embed_flat = torch.zeros_like(rigids_init)
                sf_node_embed = torch.zeros_like(node_init)
                sf_framepair_embed = torch.zeros_like(framepair_embed)

            node_embed = node_embed + self.sf_node_adaln(node_embed, sf_node_embed)
            rigids_embed = rigids_embed + self.sf_frame_adaln(rigids_embed, sf_rigids_embed_flat)
            framepair_embed = framepair_embed + self.sf_framepair_adaln(framepair_embed, sf_framepair_embed)

        ret = {
            "node_init": node_init,
            "node_embed": node_embed,
            "edge_embed": edge_embed,
            "time_condition_embed": self.time_condition_embed(self.timestep_embedder(t)),
            "token_motif_group_id": token_motif_group_id,
            "rigids_embed": rigids_embed,
            "framepair_embed": framepair_embed,
            "framepair_init": framepair_init,
            "to_queries": to_queries,
            "to_keys": to_keys,
            "to_pairs": to_pairs,
            "n_padding": n_padding,
        }
        ret.update(rigids_data)

        return ret

