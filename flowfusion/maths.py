"""Closed-form Gaussian identities used by the OU and Brownian bridges."""

from __future__ import annotations

import math

from flowfusion._array import (
    abs_,
    any_,
    asarray_like,
    exp,
    is_torch,
    isfinite,
    isinf,
    log,
    where,
    zeros_like,
)


def normal_logpdf(x, mu, var):
    return -0.5 * (log(2 * math.pi * var) + (x - mu) ** 2 / var)


def pointwise_gaussians_product(m1, v1, m2, v2):
    """Product of two diagonal Gaussians, broadcast over arrays."""
    m1 = asarray_like(m1, m1 if is_torch(m1) else m2 if is_torch(m2) else m1)
    # Align dtype/device using m1 when it is an array; otherwise v1.
    ref = m1 if is_torch(m1) else v1 if is_torch(v1) else m1
    m1 = asarray_like(m1, ref)
    v1 = asarray_like(v1, ref)
    m2 = asarray_like(m2, ref)
    v2 = asarray_like(v2, ref)

    both0 = (v1 == 0) & (v2 == 0)
    if any_(both0 & (m1 != m2)):
        raise ValueError("Both Gaussians have 0 variance but different means.")

    r_mu = zeros_like(m1)
    r_var = zeros_like(m1)
    r_log = zeros_like(m1)

    only_v1_0 = (v1 == 0) & (v2 > 0) & isfinite(v2)
    r_mu = where(only_v1_0, m1, r_mu)
    r_var = where(only_v1_0, v1, r_var)
    safe_v2 = where(only_v1_0, v2, where(v2 > 0, v2, v2 * 0 + 1))
    r_log = where(only_v1_0, normal_logpdf(m1, m2, safe_v2), r_log)

    only_v2_0 = (v2 == 0) & (v1 > 0) & isfinite(v1)
    r_mu = where(only_v2_0, m2, r_mu)
    r_var = where(only_v2_0, v2, r_var)
    safe_v1 = where(only_v2_0, v1, where(v1 > 0, v1, v1 * 0 + 1))
    r_log = where(only_v2_0, normal_logpdf(m2, m1, safe_v1), r_log)

    both_inf = isinf(v1) & isinf(v2)
    r_mu = where(both_inf, (m1 + m2) / 2, r_mu)
    inf_var = where(both_inf, v1, r_var)
    r_var = inf_var
    r_log = where(both_inf, r_log * 0, r_log)

    v1_inf = isinf(v1) & isfinite(v2)
    r_mu = where(v1_inf, m2, r_mu)
    r_var = where(v1_inf, v2, r_var)
    v2_inf = isinf(v2) & isfinite(v1)
    r_mu = where(v2_inf, m1, r_mu)
    r_var = where(v2_inf, v1, r_var)

    regular = isfinite(v1) & isfinite(v2) & (v1 > 0) & (v2 > 0)
    v1s = where(regular, v1, v1 * 0 + 1)
    v2s = where(regular, v2, v2 * 0 + 1)
    r_var_reg = 1 / (1 / v1s + 1 / v2s)
    r_mu_reg = r_var_reg * (m1 / v1s + m2 / v2s)
    r_log_reg = -0.5 * (
        log(2 * math.pi * (v1s * v2s / r_var_reg)) + (m1**2) / v1s + (m2**2) / v2s - (r_mu_reg**2) / r_var_reg
    )
    r_mu = where(regular, r_mu_reg, r_mu)
    r_var = where(regular, r_var_reg, r_var)
    r_log = where(regular, r_log_reg, r_log)
    return r_mu, r_var, r_log


def ou_noise_Q(t1, t2, theta, a0, w, beta):
    """Integrated OU noise `_ou_noise_Q` from ForwardBackward.jl."""
    from flowfusion._array import abs_, exp

    delta = t2 - t1
    w = list(np_list(w))
    beta = list(np_list(beta))
    theta_is_zero = abs(float(theta)) == 0.0 if not is_torch(delta) else False
    # theta is a Python float on the process. Treat it as a host scalar.
    if abs(float(theta)) <= 0.0:
        Q = a0 * delta
        for wk, bk in zip(w, beta):
            if abs(float(bk)) == 0.0:
                Q = Q + wk * delta
            else:
                Q = Q + wk * (exp(bk * t2) - exp(bk * t1)) / bk
        return Q
    e_m2 = exp(-2 * float(theta) * delta)
    Q = a0 * (-_expm1(-2 * float(theta) * delta)) / (2 * float(theta))
    for wk, bk in zip(w, beta):
        denom = float(bk) + 2 * float(theta)
        if abs(denom) > 1e-12:
            Q = Q + (wk / denom) * (exp(bk * t2) - exp(bk * t1) * e_m2)
        else:
            Q = Q + wk * exp(bk * t2) * delta
    return Q


def np_list(w):
    import numpy as np

    return np.asarray(w, dtype=float).reshape(-1)


def _expm1(x):
    if is_torch(x):
        import torch

        return torch.expm1(x)
    import numpy as np

    return np.expm1(x)
