"""Bridges, Euler steps, and generation."""

from __future__ import annotations

from flowfusion._array import asarray_like, concat, float_ref, is_torch, stack
from flowfusion.discrete import bridge_interpolating, step_interpolating
from flowfusion.doob import step_doob
from flowfusion.mask import mask
from flowfusion.processes import (
    ConvexInterpolatingDiscreteFlow,
    DoobMatchingFlow,
    FProcess,
    OUFlow,
    OrnsteinUhlenbeckExpVar,
    base_process,
)
from flowfusion.propagation import endpoint_conditioned_sample
from flowfusion.states import (
    ContinuousState,
    DiscreteState,
    Guide,
    ManifoldState,
    MaskedState,
    State,
    StateLikelihood,
    tensor,
    unmask,
)


def _scale_time(process, t):
    if isinstance(process, FProcess):
        return process.F(t)
    return t


def bridge(process, X0, X1, t, t0=None):
    """Sample `Xt` on the bridge from `X0` to `X1`.

    `t` is the sample time. `t0` is the start time and defaults to 0. The end
    time is 1. An `FProcess` clock is applied to all three times. If `X1` is a
    `MaskedState`, entries with `cmask == 0` are copied from `X1`.
    """
    if isinstance(process, tuple):
        return tuple(bridge(p, a, b, t, t0) for p, a, b in zip(process, X0, X1))
    if isinstance(X1, MaskedState):
        return mask(bridge(process, unmask(X0), X1.S, t, t0), X1)
    if isinstance(process, FProcess):
        return _bridge_inner(process.P, X0, X1, _scale_time(process, t), None if t0 is None else _scale_time(process, t0), _scale_time(process, 1))
    return _bridge_inner(process, X0, X1, t, t0, 1)


def _bridge_inner(process, X0, X1, t, t0, t_end):
    if isinstance(process, DoobMatchingFlow):
        return bridge(process.P, X0, X1, t, t0)
    if isinstance(process, OUFlow):
        start = 0 if t0 is None else t0
        ou = OrnsteinUhlenbeckExpVar(
            tensor(X1),
            process.theta,
            v_at_0=process.v_at_0,
            v_at_1=process.v_at_1,
            dec=process.dec,
        )
        return endpoint_conditioned_sample(X0, X1, ou, start, t, t_end)
    if isinstance(process, ConvexInterpolatingDiscreteFlow):
        return bridge_interpolating(process, X0, X1, t, t0)
    start = 0 if t0 is None else t0
    return endpoint_conditioned_sample(X0, X1, process, start, t, t_end)


def step(process, Xt, hat, s1, s2):
    """Advance from time `s1` to `s2` using the prediction `hat`."""
    if isinstance(process, tuple):
        return tuple(step(p, x, h, s1, s2) for p, x, h in zip(process, Xt, hat))
    if isinstance(Xt, MaskedState):
        return step(process, unmask(Xt), unmask(hat) if isinstance(hat, MaskedState) else hat, s1, s2)
    if isinstance(process, DoobMatchingFlow):
        return step_doob(process, Xt, hat, s1, s2)
    if isinstance(base_process(process), ConvexInterpolatingDiscreteFlow):
        return step_interpolating(base_process(process), Xt, hat, s1, s2)
    return bridge(process, Xt, hat, s2, t0=s1)


def resolve_prediction(hat, Xt):
    """Turn a model output into the object `step` expects."""
    if isinstance(hat, tuple):
        return tuple(resolve_prediction(h, x) for h, x in zip(hat, Xt))
    if isinstance(Xt, MaskedState) or isinstance(hat, MaskedState):
        return resolve_prediction(unmask(hat), unmask(Xt))
    if isinstance(hat, Guide) and isinstance(Xt, ManifoldState):
        from flowfusion.manifolds import apply_tangent_coordinates

        return apply_tangent_coordinates(Xt, hat.H)
    if isinstance(hat, (State, StateLikelihood, Guide)):
        return hat
    if isinstance(Xt, (ContinuousState, ManifoldState)):
        dest = Xt.copy()
        values = asarray_like(hat, dest.state)
        dest.state = values.reshape(dest.state.shape)
        return dest
    return hat


class Tracker:
    """Records `(t, Xt, Xhat)` during `gen`."""

    def __init__(self):
        self.t = []
        self.xt = []
        self.x1hat = []

    def __call__(self, t, xt, x1hat):
        self.t.append(t)
        self.xt.append(xt.copy() if hasattr(xt, "copy") else xt)
        self.x1hat.append(x1hat.copy() if hasattr(x1hat, "copy") else x1hat)
        return None


def stack_tracker(tracker: Tracker, field: str, tuple_index: int = 0):
    """Stack a recorded field. The new axis is last, matching Julia `stack`.

    `tuple_index` is 0-based. `field` is `"t"`, `"xt"`, or `"x1hat"` (`"x̂1"` is accepted).
    """
    field = {"x̂1": "x1hat", "x1": "x1hat"}.get(field, field)
    items = getattr(tracker, field)
    if field == "t":
        if is_torch(items[0]):
            import torch

            return torch.stack([torch.as_tensor(v) for v in items])
        import numpy as np

        return np.stack([np.asarray(v) for v in items])
    arrays = []
    for item in items:
        if isinstance(item, tuple):
            item = item[tuple_index]
        arrays.append(tensor(item))
    return stack(arrays, axis=-1)


def gen(process, X0, model, steps, tracker=None, midpoint=False):
    """Integrate from `X0` to time 1 by bridging toward the model's prediction.

    `model(t, Xt)` returns a state, a tensor of the same shape as `Xt`, or a
    `Guide`. `steps` is an increasing sequence of times, typically `linspace(0, 1, n)`.
    A conditioning mask on `X0` is reapplied after every step.
    """
    single = not isinstance(process, tuple)
    if single:
        user_model = model

        def model(t, states):
            return (user_model(_scalar_time(t), states[0]),)

        process = (process,)
        X0 = (X0,)
    state = tuple(_copy_state(x) for x in X0)
    steps = list(steps)
    for s1, s2 in zip(steps[:-1], steps[1:]):
        t = (s1 + s2) / 2 if midpoint else s1
        hat = resolve_prediction(model(t, state), state)
        state = mask(step(process, state, hat, s1, s2), X0)
        if not isinstance(state, tuple):
            state = (state,)
        if tracker is not None:
            tracker(t, state[0] if single else state, hat[0] if single else hat)
    return state[0] if single else state


def _copy_state(x):
    if isinstance(x, tuple):
        return tuple(_copy_state(v) for v in x)
    return x.copy()


def _scalar_time(t):
    """Pass a Python/NumPy/Torch scalar through. Sequences are not indexed."""
    if is_torch(t) and t.ndim == 0:
        return t
    return t


def batch(states, dims_from_end=1):
    """Concatenate states along an axis counted from the end. `1` is the last axis."""
    if not states:
        raise ValueError("batch() received an empty list.")
    first = states[0]
    axis = -int(dims_from_end)
    if isinstance(first, tuple):
        return tuple(batch([s[i] for s in states], dims_from_end) for i in range(len(first)))
    if isinstance(first, MaskedState):
        return MaskedState(
            batch([unmask(s) for s in states], dims_from_end),
            concat([s.cmask for s in states], axis=axis),
            concat([s.lmask for s in states], axis=axis),
        )
    if isinstance(first, ContinuousState):
        return ContinuousState(concat([tensor(s) for s in states], axis=axis))
    if isinstance(first, DiscreteState):
        return DiscreteState(first.K, concat([tensor(s) for s in states], axis=axis), first.onehot)
    if isinstance(first, ManifoldState):
        return ManifoldState(first.M, concat([tensor(s) for s in states], axis=axis))
    raise TypeError(f"Cannot batch {type(first).__name__}.")
