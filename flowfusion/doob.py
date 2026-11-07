"""Doob h-transform matching for discrete processes."""

from __future__ import annotations

from flowfusion._array import (
    broadcast_coeff,
    clamp,
    float_ref,
    is_torch,
    softmax,
    sum_keep,
    xlogy,
)
from flowfusion.processes import DoobMatchingFlow, PiQ, UniformDiscrete, UniformUnmasking
from flowfusion.states import CategoricalLikelihood, DiscreteState, Guide, MaskedState, onehot, sample_likelihood, tensor, unhot, unmask


def onescale(process: DoobMatchingFlow, t):
    if process.onescale:
        return 1 - t
    if is_torch(t):
        import torch

        return torch.ones_like(t)
    if hasattr(t, "shape"):
        import numpy as np

        return np.ones_like(t)
    return 1.0


def _onehot_tensor(X):
    X = unmask(X)
    if isinstance(X, DiscreteState):
        return tensor(onehot(unhot(X) if X.onehot else X))
    return tensor(X)


def forward_positive_velocities(Xt: DiscreteState, process):
    oh = _onehot_tensor(Xt)
    if isinstance(process, PiQ):
        pi = process.pi
        weights = pi / pi.sum()
        # (K,) aligned to axis 0.
        import numpy as np

        w = np.asarray(weights, dtype=float)
        if is_torch(oh):
            import torch

            w = torch.as_tensor(w, device=oh.device, dtype=oh.dtype).reshape((-1,) + (1,) * (oh.ndim - 1))
        else:
            w = w.reshape((-1,) + (1,) * (oh.ndim - 1))
        return (float(process.r) * w) * (1 - oh)
    if isinstance(process, UniformUnmasking):
        return (float(process.mu) * (1 / (Xt.K - 1))) * (1 - oh)
    if isinstance(process, UniformDiscrete):
        return (float(process.mu) * (1 / (Xt.K * (1 - 1 / Xt.K)))) * (1 - oh)
    raise TypeError(f"No closed-form Doob velocity for {type(process).__name__}.")


def closed_form_doob(process, t, Xt, X1):
    from flowfusion.propagation import backward

    ten = _onehot_tensor(Xt)
    # Backward message at the remaining time. `backward` on a discrete state
    # returns a categorical likelihood.
    bk = tensor(backward(unmask(X1), process, 1 - t))
    fv = forward_positive_velocities(unhot(unmask(Xt)) if isinstance(unmask(Xt), DiscreteState) else Xt, process)
    positive = (fv * bk) / sum_keep(bk * ten, 0)
    return positive - ten * sum_keep(positive, 0)


def doob_guide(process, t, Xt, X1):
    try:
        return closed_form_doob(process, t, Xt, X1)
    except TypeError:
        return _fallback_doob(process, t, Xt, X1)


def _fallback_doob(process, t, Xt, X1, delta=1e-5):
    from flowfusion.propagation import backward, forward, likelihood_product

    Xt = unmask(Xt)
    X1 = unmask(X1)
    fwd = forward(Xt, process, delta)
    bwd = backward(X1, process, (1 - t) - delta)
    prod = likelihood_product(fwd, bwd)
    return (tensor(prod) - _onehot_tensor(Xt)) / delta


def make_guide(process: DoobMatchingFlow, t, Xt, X1):
    target = doob_guide(process.P, t, Xt, X1)
    scale = onescale(process, t if hasattr(t, "shape") else _time_like(t, target))
    scaled = broadcast_coeff(scale, target) * target
    if isinstance(X1, MaskedState):
        return Guide(scaled, X1.cmask, X1.lmask)
    return Guide(scaled)


def _time_like(t, like):
    from flowfusion._array import asarray_like

    return asarray_like(t, float_ref(like))


def rate_constraint(Xt, raw, transform):
    pos = transform(raw) * (1 - Xt)
    diag = -sum_keep(pos, 0) * Xt
    return pos + diag


def step_doob(process: DoobMatchingFlow, Xt: DiscreteState, hat, s1, s2):
    restore = Xt.onehot
    indexed = unhot(Xt) if Xt.onehot else Xt
    raw = hat.H if isinstance(hat, Guide) else tensor(hat)
    oh = _onehot_tensor(indexed)
    scale = 1 / onescale(process, _time_like(s1, oh))
    velocity = rate_constraint(oh, raw, process.transform) * broadcast_coeff(scale, oh)
    delta = broadcast_coeff(s2, oh) - broadcast_coeff(s1, oh)
    updated = clamp(oh + delta * velocity, 0, float("inf"))
    zeros = oh.new_zeros(oh.shape[1:]) if is_torch(oh) else __import__("numpy").zeros(oh.shape[1:], dtype=oh.dtype)
    sampled = sample_likelihood(CategoricalLikelihood(updated, zeros))
    return onehot(sampled) if restore else sampled


def doob_floss(process, Xt, Xhat, target: Guide, c):
    from flowfusion.loss import scaledmaskedmean

    Qt = process.transform(tensor(Xhat))
    Xt_oh = _onehot_tensor(Xt)
    per = (1 - Xt_oh) * (Qt - xlogy(target.H, Qt))
    # Sum over the category axis. Julia keeps that axis; the mean is unchanged.
    reduced = per.sum(dim=0) if is_torch(per) else per.sum(axis=0)
    return scaledmaskedmean(reduced, c, target.lmask)
