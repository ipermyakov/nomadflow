"""Conditioning masks.

Where `cmask` is 0, the bridge is pinned to the masked state's value. Where it
is 1, the value is free. This matches `cmask!` in Flowfusion.jl.
"""

from __future__ import annotations

from flowfusion._array import asarray_like, broadcast_coeff, is_torch, where
from flowfusion.states import (
    CategoricalLikelihood,
    DiscreteState,
    GaussianLikelihood,
    MaskedState,
    State,
    StateLikelihood,
    onehot,
    tensor,
    unhot,
    unmask,
)


def _mask_array(cmask, like):
    """Move `cmask` onto `like`'s device without forcing it onto the state dtype."""
    if is_torch(like):
        import torch

        if is_torch(cmask):
            mask = cmask.to(device=like.device)
        else:
            mask = torch.as_tensor(cmask, device=like.device)
    else:
        import numpy as np

        mask = np.asarray(to_numpy_mask(cmask))
    if getattr(mask.dtype, "kind", None) == "b" or str(mask.dtype) == "torch.bool":
        return mask
    return mask != 0


def to_numpy_mask(cmask):
    if is_torch(cmask):
        return cmask.detach().cpu().numpy()
    return cmask


def apply_cmask(Xt_state, X1_state, cmask):
    """Return `Xt_state`, with entries replaced by `X1_state` where `cmask` is 0."""
    cond = _mask_array(cmask, Xt_state)
    cond = broadcast_coeff(cond, Xt_state) if getattr(cond, "ndim", 0) else cond
    # broadcast_coeff casts dtype. A bool expanded mask is what `where` needs.
    if is_torch(Xt_state):
        import torch

        if cond.dtype != torch.bool:
            cond = cond != 0
    else:
        import numpy as np

        if cond.dtype != np.bool_:
            cond = cond != 0
    return where(cond, Xt_state, X1_state)


def mask(Xt, X1):
    """If `X1` is masked, pin `Xt` where `X1.cmask` is 0 and inherit both masks."""
    if isinstance(Xt, tuple):
        return tuple(mask(a, b) for a, b in zip(Xt, X1))
    if not isinstance(X1, MaskedState):
        return Xt
    if isinstance(X1.S, DiscreteState) and X1.S.onehot:
        return onehot(mask(unhot(Xt), unhot(X1)))
    if isinstance(Xt, State) and isinstance(X1.S, StateLikelihood):
        raise TypeError("Cannot condition a state on a likelihood.")
    pinned = _pin(Xt, X1)
    return MaskedState(unmask(pinned), X1.cmask, X1.lmask)


def _pin(Xt, X1: MaskedState):
    left = tensor(unmask(Xt))
    right = tensor(X1.S)
    updated = apply_cmask(left, right, X1.cmask)
    inner = unmask(Xt)
    if isinstance(inner, DiscreteState):
        return DiscreteState(inner.K, updated, inner.onehot)
    copied = inner.copy()
    if hasattr(copied, "state"):
        copied.state = updated
        return copied
    if isinstance(copied, CategoricalLikelihood):
        return CategoricalLikelihood(updated, copied.log_norm_const)
    if isinstance(copied, GaussianLikelihood):
        return GaussianLikelihood(updated, copied.var, copied.log_norm_const)
    raise TypeError(f"Cannot apply a conditioning mask to {type(inner).__name__}.")
