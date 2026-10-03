import torch
import torch.nn as nn

from nomadflow.openfold.layers.layers import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
    permute_final_dims,
)
from nomadflow.openfold.layers.layers_v2 import Linear, LayerNorm, ConditionedTransition

from nomadflow.model.modules.attention import ConditionedTransformerPairBias
from nomadflow.model.modules.pair_modules import TriangleAttentionCore, SwishTransition, cuet_supported


class Pairformer(nn.Module):
    def __init__(
        self,
        c_s,
        c_cond,
        c_z,
        num_blocks=8,
        num_heads=8,
        num_trig_heads=4,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.inf = 1e8
        c_head = c_z // num_trig_heads

        self.lin_edge_cond = nn.Sequential(
            LayerNorm(c_s),
            Linear(c_s, c_z, bias=False)
        )
        self.edge_transition = ConditionedTransition(c_z, c_z)
        self.trunk = nn.ModuleDict()
        for i in range(num_blocks):
            self.trunk[f'mult_out_{i}'] = TriangleMultiplicationOutgoing(c_z, c_z)
            self.trunk[f'mult_in_{i}'] = TriangleMultiplicationIncoming(c_z, c_z)
            self.trunk[f'edge_bias_{i}'] = nn.Sequential(
                LayerNorm(c_z),
                Linear(c_z, num_trig_heads, bias=False)
            )
            self.trunk[f'attn_start_{i}'] = TriangleAttentionCore(c_z, c_head, num_trig_heads, starting=True, use_qk_norm=True)
            self.trunk[f'attn_end_{i}'] = TriangleAttentionCore(c_z, c_head, num_trig_heads, starting=False, use_qk_norm=True)
            self.trunk[f'transition_{i}'] = SwishTransition(c_z)

            self.trunk[f'token_attn_{i}'] = ConditionedTransformerPairBias(
                c_s,
                c_cond,
                c_z,
                num_heads,
                n_layers=1,
                use_qk_norm=True
            )

    def forward(
        self,
        s,
        z,
        s_cond,
        node_mask,
    ):
        edge_cond = self.lin_edge_cond(s)
        num_res = s.shape[-2]
        cond = (
            torch.tile(edge_cond[..., None, :], (1, 1, num_res, 1))
            + torch.tile(edge_cond[..., None, :, :], (1, num_res, 1, 1))
        )
        z = z + self.edge_transition(z, cond)

        edge_mask = node_mask[..., None] & node_mask[..., None, :]
        global cuet_supported
        if cuet_supported:
            mask_bias = edge_mask[..., :, None, None, :]
        else:
            mask_bias = (edge_mask[..., :, None, None, :].float() - 1) * self.inf

        for i in range(self.num_blocks):
            z = z + self.trunk[f'mult_out_{i}'](z, edge_mask)
            z = z + self.trunk[f'mult_in_{i}'](z, edge_mask)
            edge_bias = self.trunk[f'edge_bias_{i}'](z)
            edge_bias = permute_final_dims(edge_bias.unsqueeze(-4), (2, 0, 1))
            z = z + self.trunk[f'attn_start_{i}'](z, mask_bias=mask_bias, edge_bias=edge_bias)
            z = z + self.trunk[f'attn_end_{i}'](z, mask_bias=mask_bias, edge_bias=edge_bias)
            z = z + self.trunk[f'transition_{i}'](z)

            s = self.trunk[f'token_attn_{i}'](
                s, s_cond, z, node_mask)

        return s, z
