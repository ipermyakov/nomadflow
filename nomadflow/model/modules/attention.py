
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch_geometric.utils as pygu

import math
from typing import Optional, Sequence, Tuple, Union, Callable

from nomadflow.openfold.layers.layers import Linear, flatten_final_dims, ipa_point_weights_init_, Dropout
from nomadflow.model.modules.pair_modules import DropoutRowwise
from nomadflow.openfold.layers.layers_v2 import LayerNorm, Transition, permute_final_dims, AdaLN
from nomadflow.openfold.utils.rigid_utils import Rigid

from nomadflow.openfold.utils import rigid_utils as ru


class FlashTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 h_dim,
                 no_heads,
                 h_ff=2048,
                 ln_first=False,
                 bias=False,
                 dropout=0.1,
                 dtype=None
    ):
        super().__init__()
        self.h_head = h_dim // no_heads
        self.no_heads = no_heads
        self.ln_first = ln_first
        self.dtype = dtype

        self.lin_q = Linear(h_dim, self.h_head * no_heads, bias=bias)
        self.lin_kv = Linear(h_dim, 2 * self.h_head * no_heads, bias=bias)
        self.ln1 = LayerNorm(h_dim)
        self.ln2 = LayerNorm(h_dim)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            Linear(h_dim, h_ff, bias=bias),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(h_ff, h_dim, bias=bias),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x, x_mask):
        dtype = x.dtype
        x_mask = x_mask.bool()

        if self.ln_first:
            _x = self.ln1(x)
        else:
            _x = x

        q = self.lin_q(_x)
        kv = self.lin_kv(_x)
        k, v = kv.split(self.h_head * self.no_heads, dim=-1)
        q = q.view(*q.shape[:2], self.no_heads, self.h_head).transpose(-2, -3)
        k = k.view(*k.shape[:2], self.no_heads, self.h_head).transpose(-2, -3)
        v = v.view(*v.shape[:2], self.no_heads, self.h_head).transpose(-2, -3)
        if self.dtype is not None:
            q = q.type(self.dtype)
            k = k.type(self.dtype)
            v = v.type(self.dtype)
        # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        #     update = F.scaled_dot_product_attention(
        #         q, k, v, attn_mask=x_mask
        #     )
        print(q.is_contiguous(), k.is_contiguous(), v.is_contiguous(), x_mask.is_contiguous())
        update = F.scaled_dot_product_attention(
            q, k, v, attn_mask=x_mask[..., None, :, None]
        )
        update = update.transpose(-2, -3).flatten(-2, -1)
        if self.dtype is not None:
            update = update.to(dtype)
        x = x + self.dropout(update)
        if not self.ln_first:
            x = self.ln1(x)

        if self.ln_first:
            _x = self.ln2(x)
        else:
            _x = x
        x = x + self.ffn(_x) * x_mask[..., None]
        if not self.ln_first:
            x = self.ln2(x)

        return x


class FlashTransformerEncoder(nn.Module):
    def __init__(self,
                 h_dim,
                 no_heads,
                 n_layers,
                 h_ff=2048,
                 ln_first=False,
                 bias=False,
                 dropout=0.1,
                 dtype=None
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                FlashTransformerEncoderLayer(
                    h_dim=h_dim,
                    no_heads=no_heads,
                    h_ff=h_ff,
                    ln_first=ln_first,
                    bias=bias,
                    dropout=dropout,
                    dtype=dtype
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, x_mask):
        for layer in self.layers:
            x = layer(x, x_mask)
        return x


class TransformerPairBiasLayer(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 no_heads,
                 dropout=0.1,
                 row_dropout=0.0,
                 inf=1e8,
                 use_qk_norm=False

    ):
        super().__init__()
        self.h_head = c_s // no_heads
        self.no_heads = no_heads
        self.inf = inf
        self.use_qk_norm = use_qk_norm

        if self.use_qk_norm:
            self.q_norm = LayerNorm(self.h_head)
            self.k_norm = LayerNorm(self.h_head)

        self.lin_q = Linear(c_s, self.h_head * no_heads, bias=False)
        self.lin_kv = Linear(c_s, 2 * self.h_head * no_heads, bias=False)
        self.lin_b = Linear(c_z, no_heads, bias=False)
        self.out_gate = Linear(c_s, c_s, bias=False)
        self.lin_out = Linear(c_s, c_s, bias=False)

        self.ln_s = LayerNorm(c_s)
        self.ln_z = LayerNorm(c_z)
        self.dropout = nn.Dropout(dropout)
        self.row_dropout = Dropout(row_dropout, -2)

        self.ffn = Transition(c_s, n=2)

    def forward(self, x, z, x_mask):
        x_mask = x_mask.bool()

        _x = self.ln_s(x)
        _z = self.ln_z(z)

        q = self.lin_q(_x)
        kv = self.lin_kv(_x)
        b = self.lin_b(_z)
        k, v = kv.split(self.h_head * self.no_heads, dim=-1)

        b += self.inf * (x_mask[..., None, :, None].float() - 1)
        b += self.inf * (x_mask[..., None, None].float() - 1)

        q = q.view(*q.shape[:2], self.no_heads, self.h_head).transpose(-2, -3)
        k = k.view(*k.shape[:2], self.no_heads, self.h_head).transpose(-2, -3)
        v = v.view(*v.shape[:2], self.no_heads, self.h_head).transpose(-2, -3)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # print(q.shape, k.shape, v.shape, b.shape)
        update = F.scaled_dot_product_attention(
            q, k, v, attn_mask=permute_final_dims(b, (2, 0, 1))
        )
        update = update.transpose(-2, -3).flatten(-2, -1)

        out_gate = self.out_gate(_x)
        update = self.lin_out(update * torch.sigmoid(out_gate))
        x = x + self.row_dropout(self.dropout(update))

        x = x + self.ffn(x) * x_mask[..., None]

        return x


class TransformerPairBias(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 no_heads,
                 n_layers,
                 dropout=0.1,
                 row_dropout=0.0,
                 use_qk_norm=False
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerPairBiasLayer(
                    c_s,
                    c_z,
                    no_heads=no_heads,
                    dropout=dropout,
                    row_dropout=row_dropout,
                    use_qk_norm=use_qk_norm,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, z, x_mask):
        for layer in self.layers:
            x = layer(x, z, x_mask)
        return x


class ConditionedTransformerPairBiasLayer(nn.Module):
    def __init__(self,
                 c_s,
                 c_cond,
                 c_z,
                 no_heads,
                 dropout=0.1,
                 row_dropout=0.0,
                 inf=1e8,
                 use_qk_norm=False

    ):
        super().__init__()
        self.h_head = c_s // no_heads
        self.no_heads = no_heads
        self.inf = inf
        self.use_qk_norm = use_qk_norm

        if self.use_qk_norm:
            self.q_norm = LayerNorm(self.h_head)
            self.k_norm = LayerNorm(self.h_head)

        self.lin_q = Linear(c_s, self.h_head * no_heads, bias=False)
        self.lin_kv = Linear(c_s, 2 * self.h_head * no_heads, bias=False)
        self.lin_b = Linear(c_z, no_heads, bias=False)
        self.out_gate = Linear(c_s, c_s, bias=False)
        self.lin_out = Linear(c_s, c_s, bias=False)
        self.cond_gate = Linear(c_cond, c_s)
        with torch.no_grad():
            self.cond_gate.bias.fill_(-2.0)

        self.ln_s = AdaLN(c_s, c_cond)
        self.ln_z = LayerNorm(c_z)
        self.dropout = nn.Dropout(dropout)
        self.row_dropout = Dropout(row_dropout, -2)

        self.ffn = Transition(c_s, n=2)

    def forward(self, x, cond, z, x_mask):
        x_mask = x_mask.bool()

        _x = self.ln_s(x, cond)
        _z = self.ln_z(z)

        q = self.lin_q(_x)
        kv = self.lin_kv(_x)
        b = self.lin_b(_z)
        k, v = kv.split(self.h_head * self.no_heads, dim=-1)

        b += self.inf * (x_mask[..., None, :, None].float() - 1)
        b += self.inf * (x_mask[..., None, None].float() - 1)

        if True or self.training:
            q = q.view(*q.shape[:2], self.no_heads, self.h_head).transpose(-2, -3)
            k = k.view(*k.shape[:2], self.no_heads, self.h_head).transpose(-2, -3)
            v = v.view(*v.shape[:2], self.no_heads, self.h_head).transpose(-2, -3)
        else:
            q = q.view(*q.shape[:2], self.no_heads, self.h_head)#.transpose(-2, -3)
            k = k.view(*k.shape[:2], self.no_heads, self.h_head)#.transpose(-2, -3)
            v = v.view(*v.shape[:2], self.no_heads, self.h_head)#.transpose(-2, -3)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        update = F.scaled_dot_product_attention(
            q, k, v, attn_mask=permute_final_dims(b, (2, 0, 1))
        )
        update = update.transpose(-2, -3).flatten(-2, -1)

        out_gate = self.out_gate(_x)
        cond_gate = self.cond_gate(cond)
        update = self.lin_out(update * torch.sigmoid(out_gate))
        update = update * torch.sigmoid(cond_gate)
        x = x + self.row_dropout(self.dropout(update))

        x = x + self.ffn(x) * x_mask[..., None]

        return x


class ConditionedTransformerPairBias(nn.Module):
    def __init__(self,
                 c_s,
                 c_cond,
                 c_z,
                 no_heads,
                 n_layers,
                 dropout=0.1,
                 row_dropout=0.0,
                 use_qk_norm=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ConditionedTransformerPairBiasLayer(
                    c_s,
                    c_cond,
                    c_z,
                    no_heads=no_heads,
                    dropout=dropout,
                    row_dropout=row_dropout,
                    use_qk_norm=use_qk_norm
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, cond, z, x_mask):
        for layer in self.layers:
            x = layer(x, cond, z, x_mask)
        return x



