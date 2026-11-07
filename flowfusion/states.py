"""States and likelihoods.

Layout follows the Julia packages: feature axes come first and the batch axis is
last. Discrete labels are 0-based (Julia's `DiscreteState` is 1-based). A one-hot
tensor has the category axis at position 0, with length `K`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flowfusion._array import (
    copy_array,
    float_ref,
    is_torch,
    ones_like,
    to_numpy,
    to_torch,
    zeros_like,
)


class State:
    def copy(self):
        raise NotImplementedError

    def numpy(self):
        raise NotImplementedError

    def torch(self, device="cpu", dtype=None):
        raise NotImplementedError


class StateLikelihood:
    def copy(self):
        raise NotImplementedError

    def numpy(self):
        raise NotImplementedError

    def torch(self, device="cpu", dtype=None):
        raise NotImplementedError


@dataclass
class ContinuousState(State):
    state: object

    def copy(self):
        return ContinuousState(copy_array(self.state))

    def numpy(self):
        return ContinuousState(to_numpy(self.state))

    def torch(self, device="cpu", dtype=None):
        return ContinuousState(to_torch(self.state, device, dtype))

    def __repr__(self):
        return f"ContinuousState(shape={tuple(self.state.shape)}, dtype={self.state.dtype})"


@dataclass
class DiscreteState(State):
    """Integer labels in `0 .. K-1`, or a one-hot tensor with category axis 0.

    `onehot=True` means `state` has shape `(K, *labels.shape)`.
    """

    K: int
    state: object
    onehot: bool = False

    def copy(self):
        return DiscreteState(self.K, copy_array(self.state), self.onehot)

    def numpy(self):
        return DiscreteState(self.K, to_numpy(self.state), self.onehot)

    def torch(self, device="cpu", dtype=None):
        return DiscreteState(self.K, to_torch(self.state, device, dtype), self.onehot)

    def __repr__(self):
        kind = "onehot" if self.onehot else "index"
        return f"DiscreteState(K={self.K}, {kind}, shape={tuple(self.state.shape)})"


@dataclass
class ManifoldState(State):
    """Points on `M`, stored densely.

    Euclidean and torus points have shape `(dim, *batch)`. SO(3) points have
    shape `(3, 3, *batch)`.
    """

    M: object
    state: object

    def copy(self):
        return ManifoldState(self.M, copy_array(self.state))

    def numpy(self):
        return ManifoldState(self.M, to_numpy(self.state))

    def torch(self, device="cpu", dtype=None):
        return ManifoldState(self.M, to_torch(self.state, device, dtype))

    def __repr__(self):
        return f"ManifoldState({self.M}, shape={tuple(self.state.shape)})"


@dataclass
class MaskedState(State):
    """A state plus a conditioning mask and a loss mask.

    Conditioning mask, matching the Julia implementation (the struct comment in
    `types.jl` says the opposite): where `cmask` is 0, the value is pinned to
    this state's value. Where `cmask` is 1, the bridge is free to noise it.

    Loss mask: where `lmask` is 1, the entry is included in `floss`.
    """

    S: object
    cmask: object
    lmask: object

    def copy(self):
        inner = self.S.copy() if hasattr(self.S, "copy") else self.S
        return MaskedState(inner, copy_array(self.cmask), copy_array(self.lmask))

    def numpy(self):
        inner = self.S.numpy() if hasattr(self.S, "numpy") else to_numpy(self.S)
        return MaskedState(inner, to_numpy(self.cmask), to_numpy(self.lmask))

    def torch(self, device="cpu", dtype=None):
        inner = self.S.torch(device, dtype) if hasattr(self.S, "torch") else to_torch(self.S, device, dtype)
        # Masks keep their own dtype (bool or 0/1), but must share the device.
        mask_dtype = None
        return MaskedState(
            inner,
            to_torch(self.cmask, device, mask_dtype),
            to_torch(self.lmask, device, mask_dtype),
        )

    def __repr__(self):
        return f"MaskedState({self.S!r}, cmask={tuple(self.cmask.shape)}, lmask={tuple(self.lmask.shape)})"


@dataclass
class Guide:
    """A tangent (or rate) prediction, rather than a prediction of `X1` itself."""

    H: object
    cmask: object = None
    lmask: object = None

    def copy(self):
        return Guide(
            copy_array(self.H),
            None if self.cmask is None else copy_array(self.cmask),
            None if self.lmask is None else copy_array(self.lmask),
        )

    def numpy(self):
        return Guide(
            to_numpy(self.H),
            None if self.cmask is None else to_numpy(self.cmask),
            None if self.lmask is None else to_numpy(self.lmask),
        )

    def torch(self, device="cpu", dtype=None):
        return Guide(
            to_torch(self.H, device, dtype),
            None if self.cmask is None else to_torch(self.cmask, device),
            None if self.lmask is None else to_torch(self.lmask, device),
        )


@dataclass
class GaussianLikelihood(StateLikelihood):
    mu: object
    var: object
    log_norm_const: object

    def copy(self):
        return GaussianLikelihood(copy_array(self.mu), copy_array(self.var), copy_array(self.log_norm_const))

    def numpy(self):
        return GaussianLikelihood(to_numpy(self.mu), to_numpy(self.var), to_numpy(self.log_norm_const))

    def torch(self, device="cpu", dtype=None):
        return GaussianLikelihood(
            to_torch(self.mu, device, dtype),
            to_torch(self.var, device, dtype),
            to_torch(self.log_norm_const, device, dtype),
        )


@dataclass
class CategoricalLikelihood(StateLikelihood):
    """Class probabilities with the category axis at position 0."""

    dist: object
    log_norm_const: object

    def copy(self):
        return CategoricalLikelihood(copy_array(self.dist), copy_array(self.log_norm_const))

    def numpy(self):
        return CategoricalLikelihood(to_numpy(self.dist), to_numpy(self.log_norm_const))

    def torch(self, device="cpu", dtype=None):
        return CategoricalLikelihood(to_torch(self.dist, device, dtype), to_torch(self.log_norm_const, device, dtype))

    @property
    def K(self) -> int:
        return int(self.dist.shape[0])


def tensor(x):
    """Underlying array of a state, likelihood, guide, or a raw array.

    This does not move devices. A CUDA state returns a CUDA tensor.
    """
    if isinstance(x, ContinuousState):
        return x.state
    if isinstance(x, DiscreteState):
        return x.state
    if isinstance(x, ManifoldState):
        return x.state
    if isinstance(x, MaskedState):
        return tensor(x.S)
    if isinstance(x, Guide):
        return x.H
    if isinstance(x, CategoricalLikelihood):
        return x.dist
    if isinstance(x, GaussianLikelihood):
        return x.mu
    return x


def unmask(x):
    if isinstance(x, MaskedState):
        return x.S
    return x


def getlmask(x):
    if isinstance(x, (MaskedState, Guide)):
        return x.lmask
    return None


def getcmask(x):
    if isinstance(x, (MaskedState, Guide)):
        return x.cmask
    return None


def one_hot(labels, K, dtype=None):
    """One-hot encode 0-based labels. Result shape is `(K, *labels.shape)`."""
    K = int(K)
    if is_torch(labels):
        import torch

        oh = torch.nn.functional.one_hot(labels.to(dtype=torch.long), K)
        oh = oh.movedim(-1, 0)
        if dtype is None:
            dtype = torch.float32
        return oh.to(dtype=dtype)
    labels = np.asarray(labels)
    if dtype is None:
        dtype = np.float32
    flat = labels.reshape(-1).astype(np.int64, copy=False)
    out = np.zeros((K, flat.size), dtype=dtype)
    if flat.size:
        out[flat, np.arange(flat.size)] = 1
    return out.reshape((K,) + labels.shape)


def onehot(x):
    if isinstance(x, MaskedState):
        return MaskedState(onehot(x.S), x.cmask, x.lmask)
    if isinstance(x, DiscreteState):
        if x.onehot:
            return x
        return DiscreteState(x.K, one_hot(x.state, x.K), True)
    raise TypeError(f"onehot() expected a DiscreteState, got {type(x).__name__}.")


def unhot(x):
    if isinstance(x, MaskedState):
        return MaskedState(unhot(x.S), x.cmask, x.lmask)
    if isinstance(x, DiscreteState):
        if not x.onehot:
            return x
        if is_torch(x.state):
            labels = x.state.argmax(dim=0)
        else:
            labels = np.argmax(x.state, axis=0)
        return DiscreteState(x.K, labels, False)
    return x


def _index_state(x: DiscreteState) -> DiscreteState:
    return unhot(x) if x.onehot else x


def stochastic(x, dtype=None):
    """Turn a state into a likelihood: a zero-variance Gaussian, or a one-hot categorical."""
    if isinstance(x, MaskedState):
        return MaskedState(stochastic(x.S, dtype), x.cmask, x.lmask)
    if isinstance(x, ContinuousState):
        ref = float_ref(x.state, dtype)
        mu = x.state if (not is_torch(x.state) and np.issubdtype(np.asarray(x.state).dtype, np.floating) and dtype is None) else x.state
        if is_torch(x.state):
            import torch

            if dtype is not None or not x.state.is_floating_point():
                mu = x.state.to(dtype=ref.dtype)
            else:
                mu = x.state
        elif dtype is not None or not np.issubdtype(np.asarray(x.state).dtype, np.floating):
            mu = np.asarray(x.state, dtype=ref.dtype)
        else:
            mu = x.state
        z = zeros_like(mu)
        return GaussianLikelihood(mu, z, z.clone() if is_torch(z) else z.copy())
    if isinstance(x, DiscreteState):
        indexed = _index_state(x)
        ref = float_ref(indexed.state, dtype if dtype is not None else (np.float32))
        oh = one_hot(indexed.state, indexed.K, dtype=ref.dtype)
        if is_torch(oh):
            log_norm = oh.new_zeros(oh.shape[1:])
        else:
            log_norm = np.zeros(oh.shape[1:], dtype=oh.dtype)
        return CategoricalLikelihood(oh, log_norm)
    raise TypeError(f"stochastic() cannot convert {type(x).__name__}.")


def dense(x, dtype=None):
    """Convert a discrete state to a categorical likelihood."""
    if isinstance(x, DiscreteState):
        return stochastic(x, dtype=np.float32 if dtype is None else dtype)
    if isinstance(x, MaskedState):
        return MaskedState(dense(x.S, dtype), x.cmask, x.lmask)
    return x


def sample_categorical(probs) -> object:
    """Sample 0-based labels from probabilities shaped `(K, *batch)`."""
    K = int(probs.shape[0])
    flat = probs.reshape(K, -1)
    totals = flat.sum(axis=0, keepdims=True) if not is_torch(flat) else flat.sum(dim=0, keepdim=True)
    # Avoid dividing by zero if a column was clamped to all zeros.
    if is_torch(flat):
        import torch

        totals = torch.clamp(totals, min=torch.finfo(flat.dtype).tiny)
        flat = flat / totals
        cdf = torch.cumsum(flat, dim=0)
        u = torch.rand(1, flat.shape[1], device=flat.device, dtype=flat.dtype)
        labels = (cdf < u).sum(dim=0)
        labels = torch.clamp(labels, 0, K - 1)
    else:
        totals = np.maximum(totals, np.finfo(flat.dtype).tiny)
        flat = flat / totals
        cdf = np.cumsum(flat, axis=0)
        u = np.random.random((1, flat.shape[1])).astype(flat.dtype, copy=False)
        labels = np.clip((cdf < u).sum(axis=0), 0, K - 1)
    return labels.reshape(probs.shape[1:])


def sample_likelihood(dist):
    if isinstance(dist, GaussianLikelihood):
        from flowfusion._array import clamp, randn, sqrt

        std = sqrt(clamp(dist.var, 0, float("inf")))
        return ContinuousState(dist.mu + std * randn(dist.mu, tuple(dist.mu.shape)))
    if isinstance(dist, CategoricalLikelihood):
        labels = sample_categorical(dist.dist)
        return DiscreteState(dist.K, labels, False)
    raise TypeError(f"Cannot sample {type(dist).__name__}.")


def ones_mask_like(state_array):
    if is_torch(state_array):
        return ones_like(state_array).to(dtype=state_array.dtype)
    return np.ones(np.asarray(state_array).shape, dtype=np.asarray(state_array).dtype)
