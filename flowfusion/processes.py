"""Stochastic processes used by Flowfusion.

Parameters stay as Python or NumPy values. Sampling moves them onto the state's
device, so the same process object works for a NumPy CPU state and a PyTorch
CUDA state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class Process:
    pass


class ContinuousProcess(Process):
    pass


class DiscreteProcess(Process):
    pass


class ConvexInterpolatingDiscreteFlow(DiscreteProcess):
    """Discrete flow whose path is a convex mixture. Generation uses probability velocities."""


@dataclass
class Deterministic(ContinuousProcess):
    """Endpoint conditioning is linear interpolation."""


@dataclass
class BrownianMotion(ContinuousProcess):
    """Brownian motion with drift `delta` and variance rate `v`.

    `BrownianMotion(v)` sets the drift to 0, matching the Julia constructor.
    The three-time bridge used by `bridge` is the standard Brownian bridge and
    does not include drift. Drift is included in `forward` / `backward`.
    """

    delta: float = 0.0
    v: float = 1.0

    def __init__(self, v=None, delta=0.0):
        # BrownianMotion(v) and BrownianMotion(delta, v) both appear in Julia.
        # Positional `BrownianMotion(0.15)` means variance, because that is the
        # one-argument Julia constructor. Use keywords for a nonzero drift.
        if v is None:
            self.delta = 0.0
            self.v = 1.0
        else:
            self.delta = float(delta)
            self.v = float(v)


@dataclass
class OrnsteinUhlenbeck(ContinuousProcess):
    mu: float = 0.0
    v: float = 1.0
    theta: float = 1.0


@dataclass
class OrnsteinUhlenbeckExpVar(ContinuousProcess):
    """OU process with variance `a0 + sum_k w[k] exp(beta[k] t)`."""

    mu: object
    theta: float
    a0: float
    w: np.ndarray
    beta: np.ndarray

    def __init__(self, mu=0.0, theta=1.0, a0=1.0, w=None, beta=None, v_at_0=None, v_at_1=None, dec=-0.1):
        self.mu = mu
        self.theta = float(theta)
        if v_at_0 is not None:
            if v_at_1 is None:
                raise TypeError("Pass both v_at_0 and v_at_1.")
            a0_v, w_v, beta_v = _variance_schedule(float(v_at_0), float(v_at_1), float(dec))
            self.a0 = a0_v
            self.w = w_v
            self.beta = beta_v
        else:
            self.a0 = float(a0)
            self.w = np.zeros(0) if w is None else np.asarray(w, dtype=float)
            self.beta = np.zeros(0) if beta is None else np.asarray(beta, dtype=float)
        if self.w.shape != self.beta.shape:
            raise ValueError("w and beta must have the same length.")


def _variance_schedule(v_at_0, v_at_1, dec):
    if v_at_0 <= 0 or v_at_1 <= 0:
        raise ValueError("v_at_0 and v_at_1 must be positive.")
    if dec >= 0:
        raise ValueError("dec must be negative.")
    a0 = v_at_1 - (((v_at_0 - v_at_1) * np.exp(dec)) / (1 - np.exp(dec)))
    w1 = (v_at_0 - v_at_1) / (1 - np.exp(dec))
    return float(a0), np.asarray([w1], dtype=float), np.asarray([dec], dtype=float)


@dataclass
class OUBridgeExpVar(ContinuousProcess):
    """Endpoint-conditioned OU bridge with the exponential-mixture noise schedule."""

    theta: float
    a0: float
    w: np.ndarray
    beta: np.ndarray

    def __init__(self, theta=1.0, a0=1.0, w=None, beta=None, v=None, v_at_0=None, v_at_1=None, dec=-0.1):
        self.theta = float(theta)
        if v is not None:
            self.a0 = float(v)
            self.w = np.zeros(0)
            self.beta = np.zeros(0)
        elif v_at_0 is not None:
            a0_v, w_v, beta_v = _variance_schedule(float(v_at_0), float(v_at_1), float(dec))
            self.a0 = a0_v
            self.w = w_v
            self.beta = beta_v
        else:
            self.a0 = float(a0)
            self.w = np.zeros(0) if w is None else np.asarray(w, dtype=float)
            self.beta = np.zeros(0) if beta is None else np.asarray(beta, dtype=float)
        if self.w.shape != self.beta.shape:
            raise ValueError("w and beta must have the same length.")


@dataclass
class OUBridgeDistVarSched(ContinuousProcess):
    theta: float
    Rscale: float
    a: float
    b: float


@dataclass
class ScheduledGaussianMixture(ContinuousProcess):
    k0: float
    k1: float
    k_a: float
    k_b: float
    Rscale: float
    v_a: float
    v_b: float


@dataclass
class UniformDiscrete(DiscreteProcess):
    mu: float = 1.0


@dataclass
class UniformUnmasking(DiscreteProcess):
    """Mutates only the last category (the mask) into any other category."""

    mu: float = 1.0


@dataclass
class GeneralDiscrete(DiscreteProcess):
    """Continuous-time chain with rate matrix `Q` (Julia column layout)."""

    Q: np.ndarray

    def __post_init__(self):
        self.Q = np.asarray(self.Q, dtype=float)


@dataclass
class PiQ(DiscreteProcess):
    r: float
    pi: np.ndarray
    beta: float

    def __init__(self, pi, r=1.0, normalize=True):
        pi = np.asarray(pi, dtype=float)
        pi = pi / pi.sum()
        self.pi = pi
        self.r = float(r)
        self.beta = float(1.0 / (1.0 - np.sum(pi**2)) if normalize else 1.0)


@dataclass
class FProcess:
    """Wrap a process with a clock `F`, where `F(0) = 0` and `F(1) = 1`.

    `F` must accept NumPy arrays and PyTorch tensors (`t ** 2` is fine).
    Convex interpolating discrete flows keep their own schedules and cannot be wrapped.
    """

    P: Process
    F: object

    def __post_init__(self):
        if isinstance(self.P, ConvexInterpolatingDiscreteFlow):
            raise TypeError(
                "ConvexInterpolatingDiscreteFlow has its own schedule. Do not wrap it in FProcess."
            )


def base_process(P):
    if isinstance(P, FProcess):
        return P.P
    return P


@dataclass
class InterpolatingDiscreteFlow(ConvexInterpolatingDiscreteFlow):
    """Discrete interpolant of Gat et al., eq. 9. Default schedule is cosine."""

    def kappa(self, t):
        from flowfusion._array import cos

        return 1 - cos((np.pi / 2) * t)

    def kappa_dot(self, t):
        from flowfusion._array import sin

        return (np.pi / 2) * sin((np.pi / 2) * t)


@dataclass
class NoisyInterpolatingDiscreteFlow(ConvexInterpolatingDiscreteFlow):
    """Convex mixture of `X0`, uniform noise, and `X1`.

    `mask_token` is a 0-based category index. Leave it as `None` unless `X0`
    is that token (required when `power > 1`).
    """

    noise: float = 0.2
    power: int = 1
    mask_token: int | None = None

    def __post_init__(self):
        self.noise = float(self.noise)
        self.power = int(self.power)
        if self.power > 1 and self.mask_token is None:
            import warnings

            warnings.warn(
                "NoisyInterpolatingDiscreteFlow: if power > 1, X0 should be the mask_token, "
                "and mask_token should be passed in.",
                stacklevel=2,
            )

    def kappa1(self, t):
        from flowfusion._array import cos

        return (1 - cos((np.pi / 2) * t)) ** self.power

    def kappa2(self, t):
        from flowfusion._array import sin

        return self.noise * sin(np.pi * t)

    def dkappa1(self, t):
        from flowfusion._array import cos, sin

        return self.power * (np.pi / 2) * sin((np.pi / 2) * t) * (1 - cos((np.pi / 2) * t)) ** (self.power - 1)

    def dkappa2(self, t):
        from flowfusion._array import cos

        return self.noise * np.pi * cos(np.pi * t)


@dataclass
class OUFlow(ContinuousProcess):
    """OU bridge that reverts toward `X1`.

    `OUFlow(theta, v_at_0)` uses `v_at_1=1e-2` and `dec=-0.1`.
    """

    theta: float
    v_at_0: float
    v_at_1: float = 1e-2
    dec: float = -0.1

    def __init__(self, theta, v_at_0, v_at_1=1e-2, dec=-0.1):
        self.theta = float(theta)
        self.v_at_0 = float(v_at_0)
        self.v_at_1 = float(v_at_1)
        self.dec = float(dec)


@dataclass
class BetaSchedule:
    """Beta CDF/PDF/quantile on [0, 1], evaluated on NumPy or PyTorch inputs."""

    a: float = 2.0
    b: float = 2.0

    def __post_init__(self):
        self.a = float(self.a)
        self.b = float(self.b)

    def cdf(self, t):
        return _beta_apply("cdf", t, self.a, self.b)

    def pdf(self, t):
        return _beta_apply("pdf", t, self.a, self.b)

    def icdf(self, u):
        return _beta_apply("icdf", u, self.a, self.b)


def _beta_apply(which, t, a, b):
    """Beta CDF, PDF, or quantile.

    PyTorch 2.5 does not implement ``Beta.cdf``, so both backends evaluate the
    schedule with SciPy and, for a tensor, place the result back on that
    tensor's device. The result is not connected to autograd. Time is an input,
    not a parameter, in the Flowfusion training step.
    """
    from flowfusion._array import is_torch

    import scipy.stats

    device = None
    dtype = None
    shape = None
    if is_torch(t):
        device, dtype, shape = t.device, t.dtype, tuple(t.shape)
        arr = t.detach().cpu().numpy()
    else:
        arr = np.asarray(t, dtype=float)
    scalar = (not is_torch(t)) and (np.isscalar(t) or np.asarray(t).ndim == 0)
    if which == "cdf":
        out = scipy.stats.beta.cdf(arr, a, b)
    elif which == "pdf":
        out = scipy.stats.beta.pdf(arr, a, b)
    else:
        out = scipy.stats.beta.ppf(np.clip(np.asarray(arr, dtype=float), 1e-12, 1 - 1e-12), a, b)
    if device is not None:
        import torch

        return torch.as_tensor(np.asarray(out, dtype=np.float64), device=device, dtype=dtype).reshape(shape)
    if isinstance(t, np.ndarray) and np.issubdtype(t.dtype, np.floating):
        out = np.asarray(out, dtype=t.dtype)
    if scalar and not isinstance(t, np.ndarray):
        return np.asarray(out).reshape(())
    return out


@dataclass
class DistInterpolatingDiscreteFlow(ConvexInterpolatingDiscreteFlow):
    """Interpolating discrete flow with `kappa = CDF(dist)`. The model outputs logits."""

    dist: BetaSchedule = field(default_factory=BetaSchedule)


@dataclass
class DistNoisyInterpolatingDiscreteFlow(ConvexInterpolatingDiscreteFlow):
    """Three-way discrete path with Beta schedules. The model outputs logits.

    `kappa2(t) = omega_u * (1 - kappa1(t)) * CDF(D2, t)`.
    """

    D1: BetaSchedule = field(default_factory=BetaSchedule)
    D2: BetaSchedule = field(default_factory=BetaSchedule)
    omega_u: float = 0.2
    mask_token: int | None = None

    def __post_init__(self):
        self.omega_u = float(self.omega_u)


@dataclass
class ManifoldProcess(Process):
    """Manifold bridge. `v` is an isotropic tangent variance, or another process."""

    v: object = 0.0


@dataclass
class DoobMatchingFlow(Process):
    """Match the Doob h-transform rate of an underlying discrete process.

    The model outputs unconstrained rates. `transform` (default softplus) maps
    them to positive off-diagonal rates.
    """

    P: DiscreteProcess
    onescale: bool = True
    transform: object = None

    def __post_init__(self):
        if self.transform is None:
            from flowfusion._array import softplus

            self.transform = softplus
