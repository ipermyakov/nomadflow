"""Forward and backward messages, and endpoint-conditioned sampling."""

from __future__ import annotations

import math

from flowfusion._array import (
    abs_,
    asarray_like,
    broadcast_coeff,
    clamp,
    exp,
    float_ref,
    is_torch,
    log,
    matrix_exp,
    maximum,
    randn,
    reshape,
    sqrt,
    sum_keep,
    where,
)
from flowfusion.maths import ou_noise_Q, pointwise_gaussians_product
from flowfusion.processes import (
    BrownianMotion,
    Deterministic,
    GeneralDiscrete,
    OrnsteinUhlenbeck,
    OrnsteinUhlenbeckExpVar,
    OUBridgeDistVarSched,
    OUBridgeExpVar,
    PiQ,
    ScheduledGaussianMixture,
    UniformDiscrete,
    UniformUnmasking,
    base_process,
)
from flowfusion.states import (
    CategoricalLikelihood,
    ContinuousState,
    GaussianLikelihood,
    ManifoldState,
    sample_likelihood,
    stochastic,
    tensor,
)


def _time_on(t, like):
    return asarray_like(t, float_ref(like))


def _coeff(value, like):
    return broadcast_coeff(value, like)


def interpolate(X0: ContinuousState, X1: ContinuousState, t_forward, t_backward):
    """Linear interpolation. Weight on `X1` is `t_forward / (t_forward + t_backward)`."""
    ref = X0.state
    tF = _time_on(t_forward, ref)
    tB = _time_on(t_backward, ref)
    denom = tF + tB
    # If both intervals are zero, stay at X1 (the bridge has already arrived).
    if is_torch(denom):
        import torch

        safe = torch.where(denom == 0, torch.ones_like(denom), denom)
        w1 = torch.where(denom == 0, torch.ones_like(denom), tF / safe)
    else:
        import numpy as np

        safe = np.where(denom == 0, 1, denom)
        w1 = np.where(denom == 0, 1, tF / safe)
    w0 = 1 - w1
    w0 = broadcast_coeff(w0, ref)
    w1 = broadcast_coeff(w1, ref)
    return ContinuousState(ref * w0 + X1.state * w1)


def forward(Xt, process, t, t2=None):
    process = base_process(process)
    if isinstance(Xt, ContinuousState):
        Xt = stochastic(Xt)
    elif not isinstance(Xt, (GaussianLikelihood, CategoricalLikelihood)):
        Xt = stochastic(Xt)
    if t2 is None:
        return _forward_elapsed(Xt, process, t)
    if isinstance(process, OrnsteinUhlenbeckExpVar):
        return _forward_ou_exp(Xt, process, t, t2)
    return _forward_elapsed(Xt, process, _sub_time(t2, t, tensor(Xt)))


def backward(Xt, process, t, t2=None):
    process = base_process(process)
    if isinstance(Xt, ContinuousState):
        Xt = stochastic(Xt)
    elif not isinstance(Xt, (GaussianLikelihood, CategoricalLikelihood)):
        Xt = stochastic(Xt)
    if t2 is None:
        return _backward_elapsed(Xt, process, t)
    if isinstance(process, OrnsteinUhlenbeckExpVar):
        return _backward_ou_exp(Xt, process, t, t2)
    return _backward_elapsed(Xt, process, _sub_time(t2, t, tensor(Xt)))


def _sub_time(t2, t1, like):
    a = _time_on(t2, like)
    b = _time_on(t1, like)
    return a - b


def _forward_elapsed(Xt, process, elapsed):
    if isinstance(process, BrownianMotion):
        return _forward_brownian(Xt, process, elapsed)
    if isinstance(process, OrnsteinUhlenbeck):
        return _forward_ou(Xt, process, elapsed)
    if isinstance(process, PiQ):
        return _forward_piq(Xt, process, elapsed)
    if isinstance(process, UniformDiscrete):
        return _forward_uniform(Xt, process, elapsed)
    if isinstance(process, UniformUnmasking):
        return _forward_unmask(Xt, process, elapsed)
    if isinstance(process, GeneralDiscrete):
        return _forward_general(Xt, process, elapsed)
    raise TypeError(f"forward is not implemented for {type(process).__name__}.")


def _backward_elapsed(Xt, process, elapsed):
    if isinstance(process, BrownianMotion):
        return _backward_brownian(Xt, process, elapsed)
    if isinstance(process, OrnsteinUhlenbeck):
        return _backward_ou(Xt, process, elapsed)
    if isinstance(process, PiQ):
        return _backward_piq(Xt, process, elapsed)
    if isinstance(process, UniformDiscrete):
        return _backward_uniform(Xt, process, elapsed)
    if isinstance(process, UniformUnmasking):
        return _backward_unmask(Xt, process, elapsed)
    if isinstance(process, GeneralDiscrete):
        return _backward_general(Xt, process, elapsed)
    raise TypeError(f"backward is not implemented for {type(process).__name__}.")


def _forward_brownian(Xt: GaussianLikelihood, process: BrownianMotion, elapsed):
    t = _coeff(elapsed, Xt.mu)
    delta = _coeff(process.delta, Xt.mu)
    v = _coeff(process.v, Xt.mu)
    return GaussianLikelihood(Xt.mu + delta * t, v * t + Xt.var, Xt.log_norm_const)


def _backward_brownian(Xt: GaussianLikelihood, process: BrownianMotion, elapsed):
    t = _coeff(elapsed, Xt.mu)
    delta = _coeff(process.delta, Xt.mu)
    v = _coeff(process.v, Xt.mu)
    return GaussianLikelihood(Xt.mu - delta * t, v * t + Xt.var, Xt.log_norm_const)


def _forward_ou(Xt: GaussianLikelihood, process: OrnsteinUhlenbeck, elapsed):
    t = _coeff(elapsed, Xt.mu)
    mu = _coeff(process.mu, Xt.mu)
    v = _coeff(process.v, Xt.mu)
    theta = _coeff(process.theta, Xt.mu)
    decay = exp(-theta * t)
    return GaussianLikelihood(
        mu + decay * (Xt.mu - mu),
        decay**2 * Xt.var + (v / (2 * theta)) * (1 - exp(-2 * theta * t)),
        Xt.log_norm_const,
    )


def _backward_ou(Xt: GaussianLikelihood, process: OrnsteinUhlenbeck, elapsed):
    t = _coeff(elapsed, Xt.mu)
    mu = _coeff(process.mu, Xt.mu)
    v = _coeff(process.v, Xt.mu)
    theta = _coeff(process.theta, Xt.mu)
    growth = exp(theta * t)
    return GaussianLikelihood(
        mu + growth * (Xt.mu - mu),
        (growth**2) * (Xt.var + (v / (2 * theta)) * (1 - exp(-2 * theta * t))),
        Xt.log_norm_const,
    )


def _forward_ou_exp(Xt: GaussianLikelihood, process: OrnsteinUhlenbeckExpVar, t1, t2):
    t1e = _coeff(t1, Xt.mu)
    t2e = _coeff(t2, Xt.mu)
    delta = t2e - t1e
    Q = ou_noise_Q(t1e, t2e, process.theta, process.a0, process.w, process.beta)
    mu = process.mu
    mu = mu if _same_backend(mu, Xt.mu) else _coeff(mu, Xt.mu)
    if not hasattr(mu, "shape"):
        mu = _coeff(mu, Xt.mu)
    theta = float(process.theta)
    decay = exp(-theta * delta)
    return GaussianLikelihood(mu + decay * (Xt.mu - mu), (decay**2) * Xt.var + Q, Xt.log_norm_const)


def _backward_ou_exp(Xt: GaussianLikelihood, process: OrnsteinUhlenbeckExpVar, t1, t2):
    t1e = _coeff(t1, Xt.mu)
    t2e = _coeff(t2, Xt.mu)
    delta = t2e - t1e
    Q = ou_noise_Q(t1e, t2e, process.theta, process.a0, process.w, process.beta)
    mu = process.mu
    if not hasattr(mu, "shape") or not _same_backend(mu, Xt.mu):
        mu = _coeff(mu, Xt.mu)
    theta = float(process.theta)
    growth = exp(theta * delta)
    return GaussianLikelihood(
        mu + growth * (Xt.mu - mu),
        (growth**2) * (Xt.var + Q),
        Xt.log_norm_const,
    )


def _same_backend(a, b) -> bool:
    return is_torch(a) == is_torch(b)


def _pi_on(pi, dist):
    values = asarray_like(pi, dist)
    shape = (values.shape[0],) + (1,) * (dist.ndim - 1)
    return reshape(values, shape)


def _forward_piq(Xt: CategoricalLikelihood, process: PiQ, elapsed):
    t = _coeff(elapsed, Xt.dist)
    pi = _pi_on(process.pi, Xt.dist)
    scale = sum_keep(Xt.dist, 0)
    pow_ = exp(-float(process.beta) * float(process.r) * t)
    c1 = (1 - pow_) * pi
    c2 = pow_ + (1 - pow_) * pi
    return CategoricalLikelihood((scale - Xt.dist) * c1 + Xt.dist * c2, Xt.log_norm_const)


def _backward_piq(Xt: CategoricalLikelihood, process: PiQ, elapsed):
    t = _coeff(elapsed, Xt.dist)
    pi = _pi_on(process.pi, Xt.dist)
    pow_ = exp(-float(process.beta) * float(process.r) * t)
    c1 = (1 - pow_) * pi
    vsum = sum_keep(Xt.dist * c1, 0)
    return CategoricalLikelihood(pow_ * Xt.dist + vsum, Xt.log_norm_const)


def _forward_uniform(Xt: CategoricalLikelihood, process: UniformDiscrete, elapsed):
    t = _coeff(elapsed, Xt.dist)
    K = Xt.dist.shape[0]
    scale = sum_keep(Xt.dist, 0)
    r = float(process.mu) * (1 / (1 - 1 / K))
    p = 1 / K
    pow_ = exp(-r * t)
    c1 = (1 - pow_) * p
    c2 = pow_ + (1 - pow_) * p
    return CategoricalLikelihood((scale - Xt.dist) * c1 + Xt.dist * c2, Xt.log_norm_const)


def _backward_uniform(Xt: CategoricalLikelihood, process: UniformDiscrete, elapsed):
    t = _coeff(elapsed, Xt.dist)
    K = Xt.dist.shape[0]
    r = float(process.mu) * (1 / (1 - 1 / K))
    p = 1 / K
    pow_ = exp(-r * t)
    c1 = (1 - pow_) * p
    vsum = sum_keep(Xt.dist * c1, 0)
    return CategoricalLikelihood(pow_ * Xt.dist + vsum, Xt.log_norm_const)


def _forward_unmask(Xt: CategoricalLikelihood, process: UniformUnmasking, elapsed):
    t = _coeff(elapsed, Xt.dist)
    K = Xt.dist.shape[0]
    mask_volume = Xt.dist[K - 1 : K]
    event_p = 1 - exp(-float(process.mu) * t)
    dest = Xt.dist.clone() if is_torch(Xt.dist) else Xt.dist.copy()
    dest[: K - 1] = Xt.dist[: K - 1] + mask_volume * (1 / (K - 1)) * event_p
    dest[K - 1 : K] = mask_volume * (1 - event_p)
    return CategoricalLikelihood(dest, Xt.log_norm_const)


def _backward_unmask(Xt: CategoricalLikelihood, process: UniformUnmasking, elapsed):
    t = _coeff(elapsed, Xt.dist)
    K = Xt.dist.shape[0]
    event_p = 1 - exp(-float(process.mu) * t)
    dest = Xt.dist.clone() if is_torch(Xt.dist) else Xt.dist.copy()
    dest[: K - 1] = Xt.dist[: K - 1]
    vsum = sum_keep(Xt.dist[: K - 1] * (event_p / (K - 1)), 0)
    dest[K - 1 : K] = (1 - event_p) * Xt.dist[K - 1 : K] + vsum
    return CategoricalLikelihood(dest, Xt.log_norm_const)


def _forward_general(Xt: CategoricalLikelihood, process: GeneralDiscrete, elapsed):
    if not _is_scalar_time(elapsed):
        raise TypeError("GeneralDiscrete propagation expects a scalar time.")
    t = float(asarray_like(elapsed, float_ref(Xt.dist)))
    Q = asarray_like(process.Q, float_ref(Xt.dist))
    P = clamp(matrix_exp(Q * t), 0, 1)
    flat = reshape(Xt.dist, (Xt.dist.shape[0], -1))
    # Julia: (source' * P)' = P.T @ source, with source shaped (K, N).
    dest = P.T @ flat if is_torch(P) else P.T @ flat
    return CategoricalLikelihood(reshape(dest, Xt.dist.shape), Xt.log_norm_const)


def _backward_general(Xt: CategoricalLikelihood, process: GeneralDiscrete, elapsed):
    if not _is_scalar_time(elapsed):
        raise TypeError("GeneralDiscrete propagation expects a scalar time.")
    t = float(asarray_like(elapsed, float_ref(Xt.dist)))
    Q = asarray_like(process.Q, float_ref(Xt.dist))
    P = clamp(matrix_exp(Q * t), 0, 1)
    flat = reshape(Xt.dist, (Xt.dist.shape[0], -1))
    dest = P @ flat
    return CategoricalLikelihood(reshape(dest, Xt.dist.shape), Xt.log_norm_const)


def _is_scalar_time(t) -> bool:
    if isinstance(t, (float, int)) and not isinstance(t, bool):
        return True
    if is_torch(t):
        return t.ndim == 0
    import numpy as np

    arr = np.asarray(t)
    return arr.ndim == 0


def likelihood_product(a, b):
    """Pointwise product of two likelihoods, written `⊙` in the Julia package."""
    if isinstance(a, CategoricalLikelihood) and isinstance(b, CategoricalLikelihood):
        dist = a.dist * b.dist
        scale = sum_keep(dist, 0)
        dist = dist / scale
        log_scale = log(scale)
        if is_torch(log_scale):
            log_scale = log_scale.squeeze(0)
        else:
            log_scale = log_scale.squeeze(0)
        return CategoricalLikelihood(dist, a.log_norm_const + b.log_norm_const + log_scale)
    if isinstance(a, GaussianLikelihood) and isinstance(b, GaussianLikelihood):
        mu, var, logc = pointwise_gaussians_product(a.mu, a.var, b.mu, b.var)
        return GaussianLikelihood(mu, var, logc + a.log_norm_const + b.log_norm_const)
    raise TypeError("likelihood_product expects two likelihoods of the same family.")


def endpoint_conditioned_sample(X0, X1, process, t0, t1=None, t2=None, dt=0.05):
    """Sample the bridge.

    - one time: endpoints at 0 and 1, sample at `t0`
    - two times: forward interval `t0`, backward interval `t1` (time-homogeneous)
    - three times: sample at `t1` given endpoints at `t0` and `t2`
    """
    process = base_process(process)
    if isinstance(X0, ManifoldState) or isinstance(process, tuple):
        pass
    if t1 is None:
        t = t0
        t_back = _clamp_time(1 - _as_host_or_array(t), 0, 1)
        return endpoint_conditioned_sample(X0, X1, process, t, t_back, dt=dt)
    if t2 is None:
        return _endpoint_two_time(X0, X1, process, t0, t1, dt=dt)
    return _endpoint_three_time(X0, X1, process, t0, t1, t2, dt=dt)


def _as_host_or_array(t):
    return t


def _clamp_time(t, lo, hi):
    if isinstance(t, (float, int)) and not isinstance(t, bool):
        return min(max(t, lo), hi)
    return clamp(t, lo, hi)


def _endpoint_two_time(X0, X1, process, t_forward, t_backward, dt=0.05):
    if isinstance(process, Deterministic):
        return interpolate(X0, X1, t_forward, t_backward)
    if isinstance(X0, ManifoldState):
        from flowfusion.manifolds import endpoint_manifold

        # two-time form is forward interval then backward interval, starting at 0.
        return endpoint_manifold(X0, X1, process, 0, t_forward, _add_time(t_forward, t_backward, tensor(X0)), dt=dt)
    message = likelihood_product(forward(X0, process, t_forward), backward(X1, process, t_backward))
    return sample_likelihood(message)


def _add_time(a, b, like):
    return _time_on(a, like) + _time_on(b, like)


def _endpoint_three_time(X0, X1, process, t_a, t_b, t_c, dt=0.05):
    if isinstance(X0, ManifoldState):
        from flowfusion.manifolds import endpoint_manifold

        return endpoint_manifold(X0, X1, process, t_a, t_b, t_c, dt=dt)
    if isinstance(process, Deterministic):
        return interpolate(X0, X1, _sub_time(t_b, t_a, X0.state), _sub_time(t_c, t_b, X0.state))
    if isinstance(process, BrownianMotion) and isinstance(X0, ContinuousState):
        return _brownian_bridge(X0, X1, process, t_a, t_b, t_c)
    if isinstance(process, OUBridgeExpVar):
        return _ou_bridge_exp(X0, X1, process, t_a, t_b, t_c)
    if isinstance(process, OUBridgeDistVarSched):
        return _ou_bridge_dist(X0, X1, process, t_a, t_b, t_c)
    if isinstance(process, ScheduledGaussianMixture):
        return _scheduled_gaussian(X0, X1, process, t_a, t_b, t_c)
    if isinstance(process, OrnsteinUhlenbeckExpVar):
        message = likelihood_product(
            forward(X0, process, t_a, t_b),
            backward(X1, process, t_b, t_c),
        )
        return sample_likelihood(message)
    message = likelihood_product(
        forward(X0, process, t_a, t_b),
        backward(X1, process, t_b, t_c),
    )
    return sample_likelihood(message)


def _brownian_bridge(X0, X1, process: BrownianMotion, t_a, t_b, t_c):
    """Closed form used by Flowfusion. Drift is not part of this formula."""
    ref = X0.state
    tF = broadcast_coeff(_sub_time(t_b, t_a, ref), ref)
    tB = broadcast_coeff(_sub_time(t_c, t_b, ref), ref)
    total = broadcast_coeff(_sub_time(t_c, t_a, ref), ref)
    if is_torch(total):
        import torch

        safe = torch.where(total == 0, torch.ones_like(total), total)
        w_end = torch.where(total == 0, torch.ones_like(total), tF / safe)
    else:
        import numpy as np

        safe = np.where(total == 0, 1, total)
        w_end = np.where(total == 0, 1, tF / safe)
    mean = X0.state * (1 - w_end) + X1.state * w_end
    var = _coeff(process.v, ref) * tF * tB / where(total == 0, total * 0 + 1, total)
    var = where(total == 0, var * 0, var)
    var = where(var < 0, var * 0, var)
    return ContinuousState(mean + sqrt(var) * randn(ref, tuple(ref.shape)))


def _ou_bridge_exp(X0, X1, process: OUBridgeExpVar, t_a, t_b, t_c):
    xa = X0.state
    xc = X1.state
    ta = _coeff(t_a, xa)
    tb = _coeff(t_b, xa)
    tc = _coeff(t_c, xa)
    theta = float(process.theta)
    phi_ba = exp(-theta * (tb - ta))
    phi_ca = exp(-theta * (tc - ta))
    phi_cb = exp(-theta * (tc - tb))
    m_b = xc + phi_ba * (xa - xc)
    m_c = xc + phi_ca * (xa - xc)
    V_b = ou_noise_Q(ta, tb, theta, process.a0, process.w, process.beta)
    V_c = ou_noise_Q(ta, tc, theta, process.a0, process.w, process.beta)
    C_bc = phi_cb * V_b
    tiny = _tiny(xa)
    safe = maximum(V_c, tiny)
    mu = m_b + (C_bc / safe) * (xc - m_c)
    var = maximum(V_b - (C_bc * C_bc) / safe, 0)
    return ContinuousState(mu + sqrt(var) * randn(xa, tuple(xa.shape)))


def _tiny(ref):
    if is_torch(ref):
        import torch

        return torch.finfo(ref.dtype).tiny
    import numpy as np

    return np.finfo(ref.dtype).tiny


def _beta_survival(a, b, t, ref):
    from flowfusion.processes import BetaSchedule

    schedule = BetaSchedule(a, b)
    tt = _time_on(t, ref) if not hasattr(t, "shape") else t
    # `t` may already be an expanded array on the right device.
    if is_torch(ref) and not is_torch(t):
        tt = _time_on(t, ref)
    elif not is_torch(ref) and is_torch(t):
        tt = _time_on(t, ref)
    else:
        tt = t if hasattr(t, "shape") else _time_on(t, ref)
    return 1 - schedule.cdf(tt)


def _ou_bridge_dist(X0, X1, process: OUBridgeDistVarSched, t_a, t_b, t_c):
    xa, xc = X0.state, X1.state
    ta, tb, tc = _coeff(t_a, xa), _coeff(t_b, xa), _coeff(t_c, xa)
    theta = float(process.theta)
    phi_ba = exp(-theta * (tb - ta))
    phi_cb = exp(-theta * (tc - tb))
    Sa = _beta_survival(process.a, process.b, ta, xa)
    Sb = _beta_survival(process.a, process.b, tb, xa)
    r = Sb / Sa
    mu = xc + phi_ba * r * (xa - xc)
    var = (float(process.Rscale) * Sb * (Sa - Sb)) / (Sa * (phi_cb**2))
    var = maximum(var, 0)
    return ContinuousState(mu + sqrt(var) * randn(xa, tuple(xa.shape)))


def _scheduled_gaussian(X0, X1, process: ScheduledGaussianMixture, t_a, t_b, t_c):
    from flowfusion.processes import BetaSchedule

    xa, xc = X0.state, X1.state
    ta, tb, tc = _coeff(t_a, xa), _coeff(t_b, xa), _coeff(t_c, xa)
    kdist = BetaSchedule(process.k_a, process.k_b)
    vdist = BetaSchedule(process.v_a, process.v_b)
    K_ba = float(process.k0) * (tb - ta) + float(process.k1) * (kdist.cdf(tb) - kdist.cdf(ta))
    K_cb = float(process.k0) * (tc - tb) + float(process.k1) * (kdist.cdf(tc) - kdist.cdf(tb))
    phi_ba = exp(-K_ba)
    phi_cb = exp(-K_cb)
    Ra = float(process.Rscale) * (1 - vdist.cdf(ta))
    Rb = float(process.Rscale) * (1 - vdist.cdf(tb))
    mu = xc + phi_ba * (Rb / Ra) * (xa - xc)
    var = maximum(Rb * (Ra - Rb) / (Ra * (phi_cb**2)), 0)
    return ContinuousState(mu + sqrt(var) * randn(xa, tuple(xa.shape)))
