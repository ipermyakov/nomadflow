"""Endpoint losses. Continuous processes use squared error. Discrete processes use logit cross-entropy."""

from __future__ import annotations

from flowfusion._array import asarray_like, broadcast_coeff, is_torch, log_softmax, mean_all, numel, sum_drop
from flowfusion.processes import (
    BrownianMotion,
    Deterministic,
    DiscreteProcess,
    DoobMatchingFlow,
    FProcess,
    ManifoldProcess,
    OUBridgeExpVar,
    OUFlow,
    OrnsteinUhlenbeck,
    OrnsteinUhlenbeckExpVar,
    base_process,
)
from flowfusion.states import ContinuousState, DiscreteState, Guide, MaskedState, getlmask, tensor


def _tscale(process, t):
    if isinstance(process, FProcess):
        return process.F(t)
    return t


def scalefloss(process, t, power=2, eps=0.05):
    """Loss weight `1 / (1 + eps - F(t))^power`. Default `eps` is 0.05, as in Julia."""
    if isinstance(process, tuple):
        if isinstance(t, tuple):
            return tuple(scalefloss(p, ti, power, eps) for p, ti in zip(process, t))
        return tuple(scalefloss(p, t, power, eps) for p in process)
    ts = _tscale(process, t)
    if is_torch(ts):
        import torch

        one = torch.ones((), device=ts.device, dtype=ts.dtype)
        eps_t = torch.as_tensor(eps, device=ts.device, dtype=ts.dtype)
        return one / (one + eps_t - ts).pow(power)
    import numpy as np

    ts = np.asarray(ts, dtype=float if not isinstance(ts, np.ndarray) else ts.dtype)
    return 1.0 / ((1.0 + float(eps)) - ts) ** power


def scaledmaskedmean(loss, scale, mask):
    scale_b = broadcast_coeff(scale, loss)
    if mask is None:
        return mean_all(loss * scale_b)
    if is_torch(loss):
        mask_b = mask.to(device=loss.device, dtype=loss.dtype) if is_torch(mask) else asarray_like(mask, loss)
    else:
        import numpy as np

        mask_b = np.asarray(mask.detach().cpu().numpy() if is_torch(mask) else mask, dtype=loss.dtype)
    mask_b = broadcast_coeff(mask_b, loss)
    if is_torch(mask_b):
        denom = mask_b.sum() / mask_b.numel() + asarray_like(1e-6, loss)
    else:
        import numpy as np

        denom = np.sum(mask_b) / np.size(mask_b) + asarray_like(1e-6, loss)
    return mean_all(loss * scale_b * mask_b) / denom


def mse(Xhat, X1):
    a = tensor(Xhat)
    b = tensor(X1)
    return (a - b) ** 2


def logit_cross_entropy(Xhat, X1):
    logits = tensor(Xhat)
    target = tensor(X1)
    per = -(target * log_softmax(logits, dim=0))
    if is_torch(per):
        return per.sum(dim=0)
    return per.sum(axis=0)


def _continuous_family(process) -> bool:
    process = base_process(process)
    return isinstance(
        process,
        (
            Deterministic,
            BrownianMotion,
            OUFlow,
            OUBridgeExpVar,
            OrnsteinUhlenbeck,
            OrnsteinUhlenbeckExpVar,
            ManifoldProcess,
        ),
    )


def floss(process, a, b, c, d=None):
    """Compare a prediction with the endpoint.

    Usual call: `floss(P, model(t, Xt), X1, scalefloss(P, t))`.

    Doob matching: `floss(P, Xt, Xhat, guide, scalefloss(P, t))`.
    """
    if d is not None:
        Xt, Xhat, X1, scale = a, b, c, d
    else:
        Xt, Xhat, X1, scale = None, a, b, c
    if isinstance(process, tuple):
        scales = scale if isinstance(scale, tuple) else (scale,) * len(process)
        return sum(floss(p, h, x, s) for p, h, x, s in zip(process, Xhat, X1, scales))
    if isinstance(X1, Guide) and isinstance(base_process(process), DoobMatchingFlow):
        from flowfusion.doob import doob_floss

        if Xt is None:
            raise TypeError("Doob floss needs the current state: floss(P, Xt, Xhat, guide, scale).")
        return doob_floss(base_process(process), Xt, Xhat, X1, scale)
    if isinstance(X1, Guide):
        return scaledmaskedmean(mse(Xhat, X1.H), scale, getlmask(X1))
    state = X1.S if isinstance(X1, MaskedState) else X1
    if isinstance(state, ContinuousState) and _continuous_family(process):
        return scaledmaskedmean(mse(Xhat, X1), scale, getlmask(X1))
    if isinstance(base_process(process), DiscreteProcess) and isinstance(state, DiscreteState):
        if not state.onehot:
            raise TypeError("X1 needs to be one-hot encoded with onehot(X1) before floss.")
        return scaledmaskedmean(logit_cross_entropy(Xhat, X1), scale, getlmask(X1))
    raise TypeError(
        f"No floss for process {type(base_process(process)).__name__} and state {type(state).__name__}."
    )
