"""Euclidean, torus, and SO(3) bridges.

General `Manifolds.jl` types are not wrapped. These three cover the continuous
flow, the torus example, and the rotation frames used with protein models.
SO(3) tangent coordinates use the same `sqrt(2)` scaling as `get_coordinates`
in Manifolds.jl, which is what `so3_tangent_coordinates_stack` implements.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flowfusion._array import (
    acos,
    asarray_like,
    broadcast_coeff,
    clamp,
    cos,
    exp,
    float_ref,
    is_torch,
    maximum,
    minimum,
    randn,
    sin,
    sqrt,
    stack,
    where,
)
from flowfusion.processes import ManifoldProcess, Process
from flowfusion.states import ContinuousState, ManifoldState, tensor


@dataclass(frozen=True)
class Euclidean:
    dim: int

    def __post_init__(self):
        object.__setattr__(self, "dim", int(self.dim))

    def __repr__(self):
        return f"Euclidean({self.dim})"


@dataclass(frozen=True)
class Torus:
    """Flat torus with angles stored in radians."""

    dim: int

    def __post_init__(self):
        object.__setattr__(self, "dim", int(self.dim))

    def __repr__(self):
        return f"Torus({self.dim})"


@dataclass(frozen=True)
class SpecialOrthogonal:
    n: int = 3

    def __post_init__(self):
        if int(self.n) != 3:
            raise NotImplementedError("Only SO(3) is implemented in this port.")
        object.__setattr__(self, "n", 3)

    def __repr__(self):
        return "SpecialOrthogonal(3)"


def _wrap_pi(x):
    # Map to (-pi, pi].
    if is_torch(x):
        import torch

        return torch.remainder(x + np.pi, 2 * np.pi) - np.pi
    return np.remainder(x + np.pi, 2 * np.pi) - np.pi


def tangent_guide(Xt, X1):
    """Coordinate vector pointing from `Xt` to `X1`."""
    from flowfusion.states import MaskedState

    if isinstance(Xt, MaskedState):
        Xt = Xt.S
    if isinstance(X1, MaskedState):
        X1 = X1.S
    if not isinstance(Xt, ManifoldState):
        raise TypeError("tangent_guide expects ManifoldState.")
    M = Xt.M
    if isinstance(M, Euclidean):
        return X1.state - Xt.state
    if isinstance(M, Torus):
        return _wrap_pi(X1.state - Xt.state)
    if isinstance(M, SpecialOrthogonal):
        return so3_coordinates(Xt.state, X1.state)
    raise TypeError(f"No tangent_guide for manifold {M}.")


def endpoint_guide(Xt, X1):
    """`Guide` whose coordinates point from `Xt` to `X1`, including masks from `X1`."""
    from flowfusion.states import Guide, MaskedState

    H = tangent_guide(Xt, X1)
    if isinstance(X1, MaskedState):
        return Guide(H, X1.cmask, X1.lmask)
    return Guide(H)


def apply_tangent_coordinates(Xt: ManifoldState, xi):
    """Retract `Xt` along tangent coordinates `xi` to a predicted endpoint."""
    xi = asarray_like(xi, Xt.state)
    M = Xt.M
    if isinstance(M, Euclidean):
        return ManifoldState(M, Xt.state + xi)
    if isinstance(M, Torus):
        return ManifoldState(M, _wrap_pi(Xt.state + xi))
    if isinstance(M, SpecialOrthogonal):
        rotvec = xi / math_sqrt2(xi)
        return ManifoldState(M, so3_apply(Xt.state, rotvec))
    raise TypeError(f"No retraction for manifold {M}.")


def math_sqrt2(ref):
    return asarray_like(np.sqrt(2.0), float_ref(ref))


def so3_tangent_coordinates_stack(R1, Rt):
    """Julia `so3_tangent_coordinates_stack`, including the trace clamp to 0.99."""
    squeezed = False
    if R1.shape[-1] != 3 and R1.ndim == 4:
        # (3, 3, n, m) -> (3, 3, n*m) then reshape back. Axis order is (3, 3, *batch).
        pass
    batch_shape = R1.shape[2:]
    R1f = R1.reshape(3, 3, -1)
    Rtf = Rt.reshape(3, 3, -1)
    coords = _so3_coords_flat(R1f, Rtf, clamp_trace=True)
    return coords.reshape((3,) + batch_shape)


def so3_coordinates(Rt, R1):
    """Stable SO(3) coordinates of `log(Rt^T R1)`, scaled by `sqrt(2)`."""
    batch_shape = Rt.shape[2:]
    coords = _so3_coords_flat(R1.reshape(3, 3, -1), Rt.reshape(3, 3, -1), clamp_trace=False)
    return coords.reshape((3,) + batch_shape)


def _so3_coords_flat(R1, Rt, clamp_trace: bool):
    # R1, Rt: (3, 3, N). Compute Rt^T @ R1.
    R = batched_mul33(transpose_33(Rt), R1)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    cos_theta = (tr - 1) / 2
    if clamp_trace:
        cos_theta = clamp(cos_theta, -0.99, 0.99)
    else:
        cos_theta = clamp(cos_theta, -1, 1)
    theta = acos(cos_theta)
    # vee(R - R^T) / 2 = sin(theta) * axis.
    vx = 0.5 * (R[2, 1] - R[1, 2])
    vy = 0.5 * (R[0, 2] - R[2, 0])
    vz = 0.5 * (R[1, 0] - R[0, 1])
    vee = stack([vx, vy, vz], axis=0)
    s = sin(theta)
    if is_torch(s):
        import torch

        scale = torch.where(s.abs() < 1e-4, torch.ones_like(s), theta / torch.where(s.abs() < 1e-4, torch.ones_like(s), s))
    else:
        scale = np.where(np.abs(s) < 1e-4, 1, theta / np.where(np.abs(s) < 1e-4, 1, s))
    if clamp_trace:
        # Match the Julia helper, which uses 0.5 / (sin + eps) on the raw skew entries
        # and then multiplies by theta. Rebuild that expression.
        eps = asarray_like(1e-5, theta)
        coeff = 0.5 / (sin(theta) + eps)
        axis = stack(
            [
                coeff * (R[2, 1] - R[1, 2]),
                coeff * (R[0, 2] - R[2, 0]),
                coeff * (R[1, 0] - R[0, 1]),
            ],
            axis=0,
        )
        return math_sqrt2(theta) * (broadcast_coeff(theta, axis) * axis)
    rotvec = vee * broadcast_coeff(scale, vee)
    return math_sqrt2(rotvec) * rotvec


def transpose_33(R):
    if is_torch(R):
        return R.transpose(0, 1)
    return np.swapaxes(R, 0, 1)


def batched_mul33(A, B):
    """Multiply `(3, 3, N)` batches. The matrix axes are the first two."""
    if is_torch(A):
        import torch

        return torch.einsum("ijb,jkb->ikb", A, B)
    return np.einsum("ijb,jkb->ikb", A, B)


def torch_batched_mul(A, B):
    return batched_mul33(A, B)


def so3_apply(R, rotvec):
    """Right-multiply `R` `(3, 3, *batch)` by `exp(rotvec)` with `rotvec` shaped `(3, *batch)`."""
    batch = R.shape[2:]
    delta = rodrigues(rotvec.reshape(3, -1)).reshape((3, 3) + batch)
    return batched_mul33(R.reshape(3, 3, -1), delta.reshape(3, 3, -1)).reshape(R.shape)


def rodrigues(rotvec):
    """`rotvec` shaped `(3, N)` -> rotation matrices `(3, 3, N)`."""
    rotvec = rotvec.reshape(3, -1)
    ref = rotvec
    theta = sqrt(rotvec[0] ** 2 + rotvec[1] ** 2 + rotvec[2] ** 2)
    safe = maximum(theta, asarray_like(1e-8, theta))
    scale = where(theta < 1e-8, theta * 0 + 1, sin(theta) / safe)
    cscale = where(theta < 1e-8, theta * 0 + 0.5, (1 - cos(theta)) / (safe**2))
    kx, ky, kz = rotvec[0], rotvec[1], rotvec[2]
    O = theta * 0
    row0 = stack([O, -kz, ky], axis=0)
    row1 = stack([kz, O, -kx], axis=0)
    row2 = stack([-ky, kx, O], axis=0)
    K = stack([row0, row1, row2], axis=0)  # (3, 3, N)
    KK = batched_mul33(K, K)
    I = eye_batch(ref, theta.shape)
    return I + broadcast_coeff(scale, K) * K + broadcast_coeff(cscale, KK) * KK


def eye_batch(ref, batch_shape):
    if is_torch(ref):
        import torch

        I = torch.zeros((3, 3) + tuple(batch_shape), device=ref.device, dtype=ref.dtype)
    else:
        I = np.zeros((3, 3) + tuple(batch_shape), dtype=np.asarray(ref).dtype)
    I[0, 0] = 1
    I[1, 1] = 1
    I[2, 2] = 1
    return I


def endpoint_manifold(X0: ManifoldState, X1: ManifoldState, process, t_a, t_b, t_c, dt=0.05):
    if not isinstance(process, ManifoldProcess):
        raise TypeError("Manifold states require a ManifoldProcess.")
    if isinstance(process.v, Process):
        if isinstance(X0.M, Euclidean):
            from flowfusion.propagation import endpoint_conditioned_sample

            flat0 = ContinuousState(X0.state)
            flat1 = ContinuousState(X1.state)
            sample = endpoint_conditioned_sample(flat0, flat1, process.v, t_a, t_b, t_c)
            return ManifoldState(X0.M, sample.state)
        raise NotImplementedError(
            "Wrapping a non-Euclidean manifold around another process is not implemented. "
            "Use ManifoldProcess(variance) or an Euclidean manifold."
        )
    return _step_bridge(X0, X1, float(process.v), t_a, t_b, t_c, dt=dt)


def _step_bridge(X0, X1, variance, t_a, t_b, t_c, dt):
    """Vectorized version of `step_toward!` with isotropic variance `variance`."""
    p = X0.state.clone() if is_torch(X0.state) else X0.state.copy()
    q = X1.state
    ref = p
    ta = _expand_time(t_a, ref)
    tb = _expand_time(t_b, ref)
    tc = _expand_time(t_c, ref)
    # Times are per trailing batch element. Broadcast to the full point shape
    # by giving them the same rank as `p` with leading singleton feature axes.
    t = ta.clone() if is_torch(ta) else np.array(ta, copy=True)
    guard = 0
    limit = int(max(4, np.ceil(2 / dt))) + 5
    # A uniform upper bound on the number of microsteps. Per-element masks stop early.
    while True:
        active = t < tb
        if is_torch(active):
            if not bool(active.any()):
                break
        elif not np.any(active):
            break
        guard += 1
        if guard > limit * 4:
            raise RuntimeError("Manifold bridge failed to reach the target time.")
        inc = minimum(t + dt, tb)
        delta = inc - t
        remaining = tc - t
        can = active & (remaining > 0)
        frac = where(can, delta / where(remaining > 0, remaining, remaining * 0 + 1), delta * 0)
        step_var = asarray_like(variance, float_ref(p)) * where(can, delta, delta * 0)
        noisy = _perturb(X0.M, p, step_var) if variance > 0 else p
        p = _geodesic(X0.M, noisy, q, frac)
        t = where(active, inc, t)
    return ManifoldState(X0.M, p)


def _expand_time(t, ref):
    """Expand time to `ref`'s rank, using trailing-axis broadcasting."""
    tt = asarray_like(t, float_ref(ref))
    if tt.ndim == 0:
        if is_torch(ref):
            return tt.expand(ref.shape)
        return np.broadcast_to(tt, ref.shape).copy()
    extra = ref.ndim - tt.ndim
    if extra < 0:
        raise ValueError("Time has more dimensions than the manifold state.")
    if extra:
        tt = tt.reshape((1,) * extra + tuple(tt.shape))
    if is_torch(ref):
        return tt.expand(ref.shape).clone()
    return np.broadcast_to(tt, ref.shape).copy()


def _perturb(M, p, step_var):
    """Isotropic tangent noise. `step_var` broadcasts to `p` and is zero where the bridge is inactive."""
    if isinstance(M, Euclidean):
        return p + randn(float_ref(p), tuple(p.shape)) * sqrt(step_var)
    if isinstance(M, Torus):
        return _wrap_pi(p + randn(float_ref(p), tuple(p.shape)) * sqrt(step_var))
    if isinstance(M, SpecialOrthogonal):
        # Manifolds.jl coordinates have variance `step_var`; rotvec = coord / sqrt(2).
        v = step_var[0, 0]
        scale = sqrt(broadcast_coeff(v, _zeros_batch(p)))
        coords = randn(float_ref(p), (3,) + tuple(p.shape[2:])) * scale
        return so3_apply(p, coords / math_sqrt2(coords))
    raise TypeError(f"No perturbation for {M}.")


def _zeros_batch(p):
    if is_torch(p):
        return p.new_zeros(p.shape[2:])
    return np.zeros(p.shape[2:], dtype=p.dtype)


def _geodesic(M, p, q, alpha):
    alpha = broadcast_coeff(alpha, p)
    if isinstance(M, Euclidean):
        return p + alpha * (q - p)
    if isinstance(M, Torus):
        return _wrap_pi(p + alpha * _wrap_pi(q - p))
    if isinstance(M, SpecialOrthogonal):
        # alpha is shaped like R, (3, 3, *batch). Use one slice as the step fraction.
        step = alpha[0, 0]
        rotvec = so3_log_vec(p, q) * broadcast_coeff(step, p[0])
        return so3_apply(p, rotvec)
    raise TypeError(f"No geodesic for {M}.")


def so3_log_vec(R_from, R_to):
    """Rotation vector of `R_from^T R_to`, shaped `(3, *batch)`."""
    coords = so3_coordinates(R_from, R_to)
    return coords / math_sqrt2(coords)


def identity_so3(batch_shape, ref):
    return eye_batch(ref, batch_shape)
