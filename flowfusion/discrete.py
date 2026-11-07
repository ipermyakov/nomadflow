"""Interpolating discrete flows, including the distribution-scheduled variants."""

from __future__ import annotations

from flowfusion._array import (
    asarray_like,
    broadcast_coeff,
    clamp,
    float_ref,
    is_torch,
    maximum,
    rand_uniform,
    randint,
    softmax,
    where,
)
from flowfusion.processes import (
    DistInterpolatingDiscreteFlow,
    DistNoisyInterpolatingDiscreteFlow,
    InterpolatingDiscreteFlow,
    NoisyInterpolatingDiscreteFlow,
)
from flowfusion.states import (
    CategoricalLikelihood,
    DiscreteState,
    onehot,
    sample_likelihood,
    tensor,
    unhot,
)


def _index(x: DiscreteState) -> DiscreteState:
    return unhot(x) if x.onehot else x


def _time(t, state: DiscreteState):
    """Put `t` on the state's device and align it with the label tensor's rank."""
    from flowfusion._array import _is_scalar_array, expand

    tt = asarray_like(t, float_ref(state.state))
    if _is_scalar_array(tt):
        return tt
    return expand(tt, state.state.ndim)


def bridge_interpolating(process, X0: DiscreteState, X1: DiscreteState, t, t0=None):
    restore = X0.onehot or X1.onehot
    X0 = _index(X0)
    X1 = _index(X1)
    if isinstance(process, InterpolatingDiscreteFlow):
        out = _bridge_cosine(process, X0, X1, t)
    elif isinstance(process, NoisyInterpolatingDiscreteFlow):
        out = _bridge_noisy(process, X0, X1, t)
    elif isinstance(process, DistInterpolatingDiscreteFlow):
        out = _bridge_dist(process, X0, X1, t, t0)
    elif isinstance(process, DistNoisyInterpolatingDiscreteFlow):
        out = _bridge_dist_noisy(process, X0, X1, t, t0)
    else:
        raise TypeError(type(process).__name__)
    return onehot(out) if restore else out


def _bridge_cosine(process, X0, X1, t):
    ts = _time(t, X0)
    take = process.kappa(ts) >= rand_uniform(float_ref(X0.state), tuple(X0.state.shape))
    return DiscreteState(X0.K, where(take, X1.state, X0.state), False)


def _bridge_noisy(process, X0, X1, t):
    ts = _time(t, X0)
    r = rand_uniform(float_ref(X0.state), tuple(X0.state.shape))
    k1 = process.kappa1(ts)
    k2 = process.kappa2(ts)
    take_x1 = k1 > r
    take_uniform = (k1 + k2) > r
    uniform = randint(0, X0.K, X0.state, tuple(X0.state.shape))
    state = where(take_uniform, uniform, X0.state)
    state = where(take_x1, X1.state, state)
    return DiscreteState(X0.K, state, False)


def _bridge_dist(process: DistInterpolatingDiscreteFlow, X0, X1, t, t0):
    if t0 is None or _is_zero(t0):
        ts = _time(t, X0)
        kappa = process.dist.cdf(ts)
    else:
        ts0 = _time(t0, X0)
        tst = _time(t, X0)
        k0 = process.dist.cdf(ts0)
        kt = process.dist.cdf(tst)
        tiny = _tiny(k0)
        kappa_cond = clamp((kt - k0) / maximum(1 - k0, tiny), 0, 1)
        t_eff = process.dist.icdf(kappa_cond)
        kappa = process.dist.cdf(t_eff)
    take = kappa >= rand_uniform(float_ref(X0.state), tuple(X0.state.shape))
    return DiscreteState(X0.K, where(take, X1.state, X0.state), False)


def _bridge_dist_noisy(process: DistNoisyInterpolatingDiscreteFlow, X0, X1, t, t0):
    if t0 is None or _is_zero(t0):
        ts = _time(t, X0)
        k1 = _k1(process, ts)
        k2 = _k2(process, ts)
        return _mix_three(X0, X1, k1, k2)
    ts0 = _time(t0, X0)
    tst = _time(t, X0)
    k1_0, k2_0 = _k1(process, ts0), _k2(process, ts0)
    k1_t, k2_t = _k1(process, tst), _k2(process, tst)
    k3_0 = 1 - k1_0 - k2_0
    k3_t = 1 - k1_t - k2_t
    tiny = _tiny(k3_0)
    denom = maximum(k3_0, tiny)
    w1 = maximum((k1_t * denom - k1_0 * k3_t) / denom, 0)
    w2 = maximum((k2_t * denom - k2_0 * k3_t) / denom, 0)
    w3 = maximum(k3_t / denom, 0)
    total = maximum(w1 + w2 + w3, tiny)
    return _mix_three(X0, X1, w1 / total, w2 / total)


def _mix_three(X0, X1, w1, w2):
    r = rand_uniform(float_ref(X0.state), tuple(X0.state.shape))
    uniform = randint(0, X0.K, X0.state, tuple(X0.state.shape))
    state = where((w1 + w2) > r, uniform, X0.state)
    state = where(w1 > r, X1.state, state)
    return DiscreteState(X0.K, state, False)


def _k1(process, t):
    return process.D1.cdf(t)


def _dk1(process, t):
    return process.D1.pdf(t)


def _k2(process, t):
    return process.omega_u * (1 - _k1(process, t)) * process.D2.cdf(t)


def _dk2(process, t):
    k1 = _k1(process, t)
    dk1 = _dk1(process, t)
    k2t = process.D2.cdf(t)
    dk2t = process.D2.pdf(t)
    return process.omega_u * (-(dk1 * k2t) + (1 - k1) * dk2t)


def step_interpolating(process, Xt: DiscreteState, hat, s1, s2):
    restore = Xt.onehot
    Xt = _index(Xt)
    if isinstance(process, InterpolatingDiscreteFlow):
        out = _step_cosine(process, Xt, hat, s1, s2)
    elif isinstance(process, NoisyInterpolatingDiscreteFlow):
        out = _step_noisy(process, Xt, hat, s1, s2)
    elif isinstance(process, DistInterpolatingDiscreteFlow):
        out = _step_dist(process, Xt, hat, s1, s2)
    elif isinstance(process, DistNoisyInterpolatingDiscreteFlow):
        out = _step_dist_noisy(process, Xt, hat, s1, s2)
    else:
        raise TypeError(type(process).__name__)
    return onehot(out) if restore else out


def _probs_from_hat(hat, Xt: DiscreteState, logits: bool):
    values = tensor(hat)
    ref = float_ref(Xt.state)
    if is_torch(values):
        values = values.to(device=ref.device, dtype=ref.dtype)
    else:
        import numpy as np

        values = np.asarray(values, dtype=ref.dtype)
        if is_torch(ref):
            import torch

            values = torch.as_tensor(values, device=ref.device, dtype=ref.dtype)
    if logits:
        values = softmax(values, dim=0)
    return values


def _advance(Xt: DiscreteState, velocity, delta):
    oh = tensor(onehot(Xt))
    dt = broadcast_coeff(delta, oh)
    updated = clamp(oh + dt * velocity, 0, float("inf"))
    return sample_likelihood(CategoricalLikelihood(updated, oh.new_zeros(oh.shape[1:]) if is_torch(oh) else __import__("numpy").zeros(oh.shape[1:], dtype=oh.dtype)))


def _step_cosine(process, Xt, hat, s1, s2):
    probs = _probs_from_hat(hat, Xt, logits=False)
    oh = tensor(onehot(Xt))
    s = broadcast_coeff(s1, oh)
    rate = process.kappa_dot(s) / (1 - process.kappa(s))
    velocity = broadcast_coeff(rate, oh) * (probs - oh)
    return _advance(Xt, velocity, _delta(s2, s1, oh))


def _step_noisy(process, Xt, hat, s1, s2):
    probs = _probs_from_hat(hat, Xt, logits=False)
    oh = tensor(onehot(Xt))
    ref = float_ref(oh)
    s = broadcast_coeff(s1, oh)
    k1 = process.kappa1(s)
    k2 = process.kappa2(s)
    k3 = 1 - (k1 + k2)
    dk1 = process.dkappa1(s)
    dk2 = process.dkappa2(s)
    dk3 = -(dk1 + dk2)
    eps = broadcast_coeff(1e-10, oh)
    pu = broadcast_coeff(1 / Xt.K, oh)
    if process.mask_token is None:
        beta = dk3 / (eps + k3)
        velocity = (dk1 - k1 * beta) * probs + (dk2 - k2 * beta) * pu + beta * oh
    else:
        r1 = dk1 / (eps + k1)
        r2 = dk2 / (eps + k2)
        r3 = dk3 / (eps + k3)
        beta = _min3(r1, r2, r3)
        a1 = dk1 - k1 * beta
        a2 = dk2 - k2 * beta
        a3 = dk3 - k3 * beta
        velocity = a1 * probs + a2 * pu + beta * oh
        velocity = velocity.clone() if is_torch(velocity) else velocity.copy()
        velocity[process.mask_token] = velocity[process.mask_token] + _align_slice(a3, velocity[process.mask_token])
    return _advance(Xt, velocity, _delta(s2, s1, oh))


def _step_dist(process, Xt, hat, s1, s2):
    probs = _probs_from_hat(hat, Xt, logits=True)
    oh = tensor(onehot(Xt))
    s = broadcast_coeff(s1, oh)
    k1 = process.dist.cdf(s)
    dk1 = process.dist.pdf(s)
    tiny = _tiny(k1)
    velocity = broadcast_coeff(dk1 / maximum(1 - k1, tiny), oh) * (probs - oh)
    return _advance(Xt, velocity, _delta(s2, s1, oh))


def _step_dist_noisy(process, Xt, hat, s1, s2):
    probs = _probs_from_hat(hat, Xt, logits=True)
    oh = tensor(onehot(Xt))
    s = broadcast_coeff(s1, oh)
    k1 = _k1(process, s)
    k2 = _k2(process, s)
    k3 = 1 - k1 - k2
    dk1 = _dk1(process, s)
    dk2 = _dk2(process, s)
    dk3 = -(dk1 + dk2)
    eps = _tiny(k3)
    beta = dk3 / maximum(k3, eps)
    pu = broadcast_coeff(1 / Xt.K, oh)
    velocity = (dk1 - k1 * beta) * probs + (dk2 - k2 * beta) * pu + beta * oh
    return _advance(Xt, velocity, _delta(s2, s1, oh))


def _align_slice(extra, slice_):
    """Drop leading singleton axes so `extra` matches a `selectdim` along axis 0."""
    while getattr(extra, "ndim", 0) > slice_.ndim and extra.shape[0] == 1:
        extra = extra[0]
    return extra


def _delta(s2, s1, like):
    return broadcast_coeff(s2, like) - broadcast_coeff(s1, like)


def _min3(a, b, c):
    from flowfusion._array import minimum

    return minimum(a, minimum(b, c))


def _tiny(ref):
    if is_torch(ref):
        import torch

        return torch.finfo(ref.dtype).tiny
    import numpy as np

    return np.finfo(np.asarray(ref).dtype).tiny


def _is_zero(t) -> bool:
    if isinstance(t, (int, float)) and not isinstance(t, bool):
        return float(t) == 0.0
    if is_torch(t):
        return bool((t == 0).all())
    import numpy as np

    return bool(np.all(np.asarray(t) == 0))
