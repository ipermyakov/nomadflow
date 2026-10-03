from abc import ABC, abstractmethod
from functools import partialmethod
import numpy as np
import torch
import math
from scipy.stats import truncnorm
import torch.nn as nn
from typing import Optional, Callable, List, Sequence, Union
import torch.nn.functional as F

from nomadflow.openfold.utils.rigid_utils import Rigid
from .layers_v2 import LayerNorm


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])

def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)

def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f

def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
        dtype = torch.float32
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias, dtype=dtype)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init_fn is not None:
            init_fn(self.weight, self.bias)
        else:
            if init == "default":
                lecun_normal_init_(self.weight)
            elif init == "relu":
                he_normal_init_(self.weight)
            elif init == "glorot":
                glorot_uniform_init_(self.weight)
            elif init == "gating":
                gating_init_(self.weight)
                if bias:
                    with torch.no_grad():
                        self.bias.fill_(1.0)
            elif init == "normal":
                normal_init_(self.weight)
            elif init == "final":
                final_init_(self.weight)
            else:
                raise ValueError("Invalid init string.")


class BackboneUpdate(nn.Module):
    """
    Implements part of Algorithm 23.
    """

    def __init__(self, c_s):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super().__init__()

        self.c_s = c_s

        self.linear_s = Linear(self.c_s, 6, init="final")

    def forward(self, s: torch.Tensor):
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector
        """
        # [*, 6]
        update_s = self.linear_s(s)

        return update_s


class StructureModuleTransition(nn.Module):
    def __init__(self, c, dropout=0.):
        super(StructureModuleTransition, self).__init__()

        self.c = c

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")
        self.relu = nn.ReLU()
        self.ln = LayerNorm(self.c)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)
        if self.dropout is not None:
            s = self.dropout(s)
        s = s + s_initial
        s = self.ln(s)

        return s

class EdgeTransition(nn.Module):
    def __init__(
            self,
            *,
            node_embed_size,
            edge_embed_in,
            edge_embed_out,
            num_layers=2,
            node_dilation=2
        ):
        super(EdgeTransition, self).__init__()

        bias_embed_size = node_embed_size // node_dilation
        self.initial_embed = Linear(
            node_embed_size, bias_embed_size, init="relu")
        hidden_size = bias_embed_size * 2 + edge_embed_in
        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers.append(Linear(hidden_size, hidden_size, init="relu"))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)
        self.final_layer = Linear(hidden_size, edge_embed_out, init="final")
        self.layer_norm = LayerNorm(edge_embed_out)

    def forward(self, node_embed, edge_embed):
        node_embed = self.initial_embed(node_embed)
        batch_size, num_res, _ = node_embed.shape
        edge_bias = torch.cat([
            torch.tile(node_embed[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(node_embed[:, None, :, :], (1, num_res, 1, 1)),
        ], axis=-1)
        edge_embed = torch.cat(
            [edge_embed, edge_bias], axis=-1).reshape(
                batch_size * num_res**2, -1)
        edge_embed = self.final_layer(self.trunk(edge_embed) + edge_embed)
        edge_embed = self.layer_norm(edge_embed)
        edge_embed = edge_embed.reshape(
            batch_size, num_res, num_res, -1
        )
        return edge_embed

class TorsionAngles(nn.Module):
    def __init__(self, c, num_torsions, eps=1e-8):
        super(TorsionAngles, self).__init__()

        self.c = c
        self.eps = eps
        self.num_torsions = num_torsions

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
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


class BaseTriangleMultiplicativeUpdate(nn.Module, ABC):
    """
    Implements Algorithms 11 and 12.
    """
    @abstractmethod
    def __init__(self, c_z, c_hidden, _outgoing):
        """
        Args:
            c_z:
                Input channel dimension
            c:
                Hidden channel dimension
        """
        super(BaseTriangleMultiplicativeUpdate, self).__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing

        self.linear_g = Linear(self.c_z, self.c_z, init="gating")
        self.linear_z = Linear(self.c_hidden, self.c_z, init="final")

        self.layer_norm_in = LayerNorm(self.c_z)
        self.layer_norm_out = LayerNorm(self.c_hidden)

        self.sigmoid = nn.Sigmoid()

    def _combine_projections(self,
        a: torch.Tensor,
        b: torch.Tensor,
        _inplace_chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        if(self._outgoing):
            a = permute_final_dims(a, (2, 0, 1))
            b = permute_final_dims(b, (2, 1, 0))
        else:
            a = permute_final_dims(a, (2, 1, 0))
            b = permute_final_dims(b,  (2, 0, 1))

        if(_inplace_chunk_size is not None):
            # To be replaced by torch vmap
            for i in range(0, a.shape[-3], _inplace_chunk_size):
                a_chunk = a[..., i: i + _inplace_chunk_size, :, :]
                b_chunk = b[..., i: i + _inplace_chunk_size, :, :]
                a[..., i: i + _inplace_chunk_size, :, :] = (
                    torch.matmul(
                        a_chunk,
                        b_chunk,
                    )
                )

            p = a
        else:
            p = torch.matmul(a, b)

        return permute_final_dims(p, (1, 2, 0))

    @abstractmethod
    def forward(self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        inplace_safe: bool = False,
        _add_with_inplace: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor
            mask:
                [*, N_res, N_res] input mask
        Returns:
            [*, N_res, N_res, C_z] output tensor
        """
        pass


class TriangleMultiplicativeUpdate(BaseTriangleMultiplicativeUpdate):
    """
    Implements Algorithms 11 and 12.
    """
    def __init__(self, c_z, c_hidden, _outgoing=True):
        """
        Args:
            c_z:
                Input channel dimension
            c:
                Hidden channel dimension
        """
        super(TriangleMultiplicativeUpdate, self).__init__(c_z=c_z,
                                                           c_hidden=c_hidden,
                                                           _outgoing=_outgoing)

        self.linear_a_p = Linear(self.c_z, self.c_hidden)
        self.linear_a_g = Linear(self.c_z, self.c_hidden, init="gating")
        self.linear_b_p = Linear(self.c_z, self.c_hidden)
        self.linear_b_g = Linear(self.c_z, self.c_hidden, init="gating")


    def forward(self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor
            mask:
                [*, N_res, N_res] input mask
        Returns:
            [*, N_res, N_res, C_z] output tensor
        """
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask = mask.unsqueeze(-1)

        z = self.layer_norm_in(z)
        a = mask
        a = a * self.sigmoid(self.linear_a_g(z))
        a = a * self.linear_a_p(z)
        b = mask
        b = b * self.sigmoid(self.linear_b_g(z))
        b = b * self.linear_b_p(z)

        x = self._combine_projections(a, b)

        del a, b
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.sigmoid(self.linear_g(z))
        x = x * g

        return x


class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 12.
    """
    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=False)


class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 11.
    """
    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=True)


class Dropout(nn.Module):
    """
    Implementation of dropout with the ability to share the dropout mask
    along a particular dimension.

    If not in training mode, this module computes the identity function.
    """

    def __init__(self, r: float, batch_dim: Union[int, List[int]]):
        """
        Args:
            r:
                Dropout rate
            batch_dim:
                Dimension(s) along which the dropout mask is shared
        """
        super(Dropout, self).__init__()

        self.r = r
        if type(batch_dim) == int:
            batch_dim = [batch_dim]
        self.batch_dim = batch_dim
        self.dropout = nn.Dropout(self.r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                Tensor to which dropout is applied. Can have any shape
                compatible with self.batch_dim
        """
        shape = list(x.shape)
        if self.batch_dim is not None:
            for bd in self.batch_dim:
                shape[bd] = 1
        mask = x.new_ones(shape)
        mask = self.dropout(mask)
        if x.is_contiguous():
            x *= mask
        else:
            x = x * mask
        return x


class DropoutColumnwise(Dropout):
    """
    Convenience class for columnwise dropout as described in subsection
    1.11.6.
    """

    __init__ = partialmethod(Dropout.__init__, batch_dim=-2)


class DropoutRowwise(Dropout):
    """
    Convenience class for rowwise dropout as described in subsection
    1.11.6.
    """

    __init__ = partialmethod(Dropout.__init__, batch_dim=-3)


class PairTransition(nn.Module):
    """
    Implements Algorithm 15.
    """

    def __init__(self, c_z, n):
        """
        Args:
            c_z:
                Pair transition channel dimension
            n:
                Factor by which c_z is multiplied to obtain hidden channel
                dimension
        """
        super(PairTransition, self).__init__()

        self.c_z = c_z
        self.n = n

        self.layer_norm = LayerNorm(self.c_z)
        self.linear_1 = Linear(self.c_z, self.n * self.c_z, init="relu")
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.n * self.c_z, c_z, init="final")

    def _transition(self, z, mask):
        # [*, N_res, N_res, C_z]
        z = self.layer_norm(z)

        # [*, N_res, N_res, C_hidden]
        z = self.linear_1(z)
        z = self.relu(z)

        # [*, N_res, N_res, C_z]
        z = self.linear_2(z)
        z = z * mask

        return z


    def forward(self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        # DISCREPANCY: DeepMind forgets to apply the mask in this module.
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        # [*, N_res, N_res, 1]
        mask = mask.unsqueeze(-1)
        z = self._transition(z=z, mask=mask)

        return z


class InvariantPointAttention(nn.Module):
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
        pre_ln=False,
        lin_bias=True,
        final_init='final',
        use_compile_path=False,
        use_fused_kernel=False,
        use_out_gating=False,
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
        super(InvariantPointAttention, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = num_heads
        self.no_qk_points = num_qk_points
        self.no_v_points = num_v_points
        self.inf = inf
        self.eps = eps
        self.use_compile_path = use_compile_path
        self.use_fused_kernel = use_fused_kernel
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
        self.linear_q = Linear(self.c_s, hc, bias=lin_bias)
        self.linear_kv = Linear(self.c_s, 2 * hc, bias=lin_bias)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq, bias=lin_bias)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv, bias=lin_bias)

        self.linear_b = Linear(self.c_z, self.no_heads, bias=lin_bias)
        self.down_z = Linear(self.c_z, self.c_z // 4, bias=lin_bias)

        self.head_weights = nn.Parameter(torch.zeros((self.no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim =  (
            self.c_z // 4 + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s, init=final_init, bias=lin_bias)

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

        self.pre_ln = pre_ln
        if pre_ln:
            self.pre_ln_s = LayerNorm(c_s)
            self.pre_ln_z = LayerNorm(c_z)

        if use_out_gating:
            self.out_gate = Linear(self.c_s, self.no_heads * concat_out_dim, bias=lin_bias)
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

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        return pt_att

    # @torch.compile()
    def _attn(
        self,
        q,
        k,
        v,
        q_pts,
        k_pts,
        v_pts,
        z,
        mask,
        pts_cdist=False
    ):
        # [*, N_res, N_res, H]
        b = self.linear_b(z)

        # [*, H, N_res, N_res]
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))

        pt_att = self._pts_bias(q_pts, k_pts, pts_cdist=pts_cdist)
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        a = a + pt_att
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(
            a, v.transpose(-2, -3)#.to(dtype=a.dtype)
        ).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, H, 3, N_res, P_v]
        o_pt = torch.sum(
            (
                a[..., None, :, :, None]
                * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
            ),
            dim=-2,
        )

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))

        # [*, N_res, H, C_z // 4]
        pair_z = self.down_z(z)#.to(dtype=a.dtype)
        o_pair = torch.matmul(a.transpose(-2, -3), pair_z)

        # [*, N_res, H * C_z // 4]
        o_pair = flatten_final_dims(o_pair, 2)

        return o, o_pt, o_pair

    def forward(
        self,
        s: torch.Tensor,
        z: Optional[torch.Tensor],
        r: Rigid,
        mask: torch.Tensor,
        _offload_inference: bool = False,
        _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
        flash_attn=False,
        pts_cdist=True,
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
        if _offload_inference:
            z = _z_reference_list
        else:
            z = [z]

        if self.pre_ln:
            s = self.pre_ln_s(s)
            z[0] = self.pre_ln_z(z[0])

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)

        # # [*, N_res, H, P_q, 3]
        # q_pts = q_pts.view(
        #     q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        # )

        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)

        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)

        # # [*, N_res, H, (P_q + P_v), 3]
        # kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # # [*, N_res, H, P_q/P_v, 3]
        # k_pts, v_pts = torch.split(
        #     kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        # )

        ##########################
        # Compute attention scores
        ##########################
        if self.use_compile_path:
            # [*, N_res, H, P_q, 3]
            q_pts = q_pts.view(
                q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
            )
            # [*, N_res, H, (P_q + P_v), 3]
            kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

            # [*, N_res, H, P_q/P_v, 3]
            k_pts, v_pts = torch.split(
                kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
            )

            o, o_pt, o_pair = self._attn(q, k, v, q_pts, k_pts, v_pts, z, mask, pts_cdist=pts_cdist)
            if self.ablate_pair_z:
                o_pair = torch.zeros_like(o_pair)
            o_pt = r[..., None, None].invert_apply(o_pt)

            # [*, N_res, H * P_v]
            o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
            o_pt_norm_feats = flatten_final_dims(
                o_pt_dists, 2)

            # [*, N_res, H * P_v, 3]
            o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

            o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair]
        elif self.use_fused_kernel:
            raise NotImplementedError
            # [*, N_res, N_res, H]
            b = permute_final_dims(self.linear_b(z[0]), (2, 0, 1))
            pair_z = self.down_z(z[0])

            # [*, N_res, H * P_q / H * P_v, 3]
            k_pts, v_pts = torch.split(
                kv_pts, [self.no_heads * self.no_qk_points, self.no_heads * self.no_v_points], dim=-2
            )
            # [*, H * P_q, N_res,  3]
            q = q.transpose(-2, -3)
            k = k.transpose(-2, -3)
            v = v.transpose(-2, -3)
            q_pts = q_pts.transpose(-2, -3)
            k_pts = k_pts.transpose(-2, -3)
            v_pts = v_pts.transpose(-2, -3)

            square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
            square_mask = self.inf * (square_mask - 1)

            head_weights = self.softplus(self.head_weights)

            # BHLD, B(HP)LD, BHLDp
            # o, o_pt, o_pair = fused_ipa_kernel(
            #     q,
            #     q_pts,
            #     k,
            #     k_pts,
            #     v,
            #     v_pts,
            #     z_attn_bias=b + square_mask.unsqueeze(-3),
            #     z_out_bias=pair_z,
            #     pts_bias_scale=head_weights,
            #     n_qk_pts=self.no_qk_points,
            #     n_v_pts=self.no_v_points
            # )
            # [*, N_res, H * P_v, 3]
            o_pt = r[..., None].invert_apply(o_pt.transpose(-2, -3))

            # [*, N_res, H * P_v]
            o_pt_norm_feats = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)

            o = flatten_final_dims(o.transpose(-2, -3), 2)
            o_pair = flatten_final_dims(o_pair.transpose(-2, -3), 2)
            if self.ablate_pair_z:
                o_pair = torch.zeros_like(o_pair)

            o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair]
        else:
            # [*, N_res, H, P_q, 3]
            q_pts = q_pts.view(
                q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
            )
            # [*, N_res, H, (P_q + P_v), 3]
            kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

            # [*, N_res, H, P_q/P_v, 3]
            k_pts, v_pts = torch.split(
                kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
            )

            # [*, N_res, N_res, H]
            b = self.linear_b(z[0])

            if(_offload_inference):
                z[0] = z[0].cpu()

            if flash_attn:
                q = permute_final_dims(q, (1, 0, 2))  # [*, H, N_res, C_hidden]
                k = permute_final_dims(k, (1, 0, 2))  # [*, H, N_res, C_hidden]
                scale = math.sqrt(1.0 / (3 * self.c_hidden))

                bias = (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))

                pt_att = self._pts_bias(q_pts, k_pts, pts_cdist=pts_cdist)
                # [*, N_res, N_res]
                square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
                square_mask = self.inf * (square_mask - 1)

                # bias += pt_att
                # bias = bias + square_mask.unsqueeze(-3)
                pt_att += bias
                pt_att += square_mask.unsqueeze(-3)
                bias = pt_att

                v_in = [
                    v.transpose(-2, -3),  # [*, H, N_res, C_hidden]
                    flatten_final_dims(v_pts, 2).transpose(-2, -3),  # [*, H, N_res, P_v * 3]
                ]

                n_res = v.shape[1]
                attn_score_collector = torch.arange(n_res, device=v.device)
                # N_res x N_res
                attn_score_collector = F.one_hot(attn_score_collector, num_classes=n_res)
                v_in.append(
                    # [*, H, N_res, N_res]
                    torch.tile(attn_score_collector[None, None, :, :], (v.shape[0], self.no_heads, 1, 1))
                )

                split_lens = [t.shape[-1] for t in v_in]
                # print([t.shape for t in v_in])
                # print(q.shape, k.shape, bias.shape)
                v_in = torch.cat(v_in, dim=-1)

                with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
                    v_out = F.scaled_dot_product_attention(
                        query=q,
                        key=k,
                        value=v_in,
                        attn_mask=bias,
                        scale=scale
                    )

                ################
                # Compute output
                ################

                o, o_pt, a = v_out.split(split_lens, dim=-1)

                # [*, N_res, H * C_hidden]
                o = flatten_final_dims(o.transpose(-2, -3), 2)

                # [*, N_res, H, P_v, 3]
                o_pt = o_pt.transpose(-2, -3).view(o_pt.shape[0], n_res, self.no_heads, -1, 3)
                o_pt = r[..., None, None].invert_apply(o_pt)

                # [*, N_res, H * P_v]
                o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
                o_pt_norm_feats = flatten_final_dims(
                    o_pt_dists, 2)

                # [*, N_res, H * P_v, 3]
                o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

                if(_offload_inference):
                    z[0] = z[0].to(o_pt.device)

                # [*, N_res, H, C_z // 4]
                pair_z = self.down_z(z[0])#.to(dtype=a.dtype)
                o_pair = torch.matmul(a.transpose(-2, -3), pair_z)

                # [*, N_res, H * C_z // 4]
                o_pair = flatten_final_dims(o_pair, 2)
                if self.ablate_pair_z:
                    o_pair = torch.zeros_like(o_pair)

                o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair]

            else:
                # [*, H, N_res, N_res]
                a = torch.matmul(
                    permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
                    permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
                )
                a *= math.sqrt(1.0 / (3 * self.c_hidden))
                a += (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))

                pt_att = self._pts_bias(q_pts, k_pts, pts_cdist=pts_cdist)
                # [*, N_res, N_res]
                square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
                square_mask = self.inf * (square_mask - 1)
                a = a + pt_att
                a = a + square_mask.unsqueeze(-3)
                a = self.softmax(a)

                ################
                # Compute output
                ################
                # [*, N_res, H, C_hidden]
                o = torch.matmul(
                    a, v.transpose(-2, -3)#.to(dtype=a.dtype)
                ).transpose(-2, -3)

                # [*, N_res, H * C_hidden]
                o = flatten_final_dims(o, 2)

                # [*, H, 3, N_res, P_v]
                o_pt = torch.sum(
                    (
                        a[..., None, :, :, None]
                        * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
                    ),
                    dim=-2,
                )

                # [*, N_res, H, P_v, 3]
                o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
                o_pt = r[..., None, None].invert_apply(o_pt)

                # [*, N_res, H * P_v]
                o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
                o_pt_norm_feats = flatten_final_dims(
                    o_pt_dists, 2)

                # [*, N_res, H * P_v, 3]
                o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

                if(_offload_inference):
                    z[0] = z[0].to(o_pt.device)

                # [*, N_res, H, C_z // 4]
                pair_z = self.down_z(z[0])# .to(dtype=a.dtype)
                o_pair = torch.matmul(a.transpose(-2, -3), pair_z)

                # [*, N_res, H * C_z // 4]
                o_pair = flatten_final_dims(o_pair, 2)
                if self.ablate_down_z:
                    o_pair = torch.zeros_like(o_pair)

                o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair]

        o_feats = torch.cat(
            o_feats, dim=-1
        )#.to(dtype=z[0].dtype)
        if self.out_gate is not None:
            gate = self.out_gate(s)
            o_feats = o_feats * torch.sigmoid(gate)

        # [*, N_res, C_s]
        s = self.linear_out(o_feats)

        return s
