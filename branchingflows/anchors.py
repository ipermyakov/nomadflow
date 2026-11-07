"""Anchor states for internal nodes of a coalescent forest."""

from __future__ import annotations

import numpy as np

from flowfusion._array import is_torch
from flowfusion.manifolds import _geodesic
from flowfusion.states import ContinuousState, DiscreteState, ManifoldState, MaskedState, tensor


def canonical_anchor_merge(S1, S2, w1, w2):
    """Merge two child anchors into the parent anchor.

    Continuous values are a weighted Euclidean average. Manifold values follow
    the geodesic, with weight `w2 / (w1 + w2)` on `S2`. Discrete values become
    the mask token, which is the last category `K - 1` (Julia's category `K`).
    A tuple of states is merged componentwise with the same weights.
    """
    if isinstance(S1, tuple):
        return tuple(canonical_anchor_merge(a, b, w1, w2) for a, b in zip(S1, S2))
    if isinstance(S1, MaskedState):
        return canonical_anchor_merge(S1.S, S2.S, w1, w2)
    if isinstance(S1, ContinuousState):
        w1 = float(w1)
        w2 = float(w2)
        total = w1 + w2
        if total == 0:
            return S1.copy()
        return ContinuousState((tensor(S1) * w1 + tensor(S2) * w2) / total)
    if isinstance(S1, ManifoldState):
        w1 = float(w1)
        w2 = float(w2)
        total = w1 + w2
        alpha = 0.0 if total == 0 else w2 / total
        points = _geodesic(S1.M, tensor(S1), tensor(S2), _filled(tensor(S1), alpha))
        return ManifoldState(S1.M, points)
    if isinstance(S1, DiscreteState):
        return _mask_token(S1)
    raise TypeError(f"No anchor merge for {type(S1).__name__}.")


def select_anchor_merge(S1, S2, w1, w2):
    """Copy one child instead of interpolating. Discrete components still mask."""
    w1 = float(w1)
    w2 = float(w2)
    total = w1 + w2
    take_left = total <= 0 or float(np.random.random()) < (w1 / total)
    if take_left:
        return canonical_anchor_merge(S1, S2, 1, 0)
    return canonical_anchor_merge(S2, S1, 0, 1)


def _filled(like, value):
    """An array shaped like `like`, so SO(3) geodesics can read `alpha[0, 0]`."""
    if is_torch(like):
        import torch

        return torch.full_like(like, float(value))
    return np.full_like(np.asarray(like), float(value))


def _mask_token(state: DiscreteState) -> DiscreteState:
    dest = state.copy()
    token = dest.K - 1
    if dest.onehot:
        if is_torch(dest.state):
            out = dest.state.new_zeros(dest.state.shape)
        else:
            out = np.zeros_like(dest.state)
        out[token] = 1
        dest.state = out
    else:
        if is_torch(dest.state):
            import torch

            dest.state = torch.full_like(dest.state, int(token))
        else:
            dest.state = np.full_like(dest.state, int(token))
    return dest

