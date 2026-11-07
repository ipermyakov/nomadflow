"""Slice, pad, and regroup variable-length states.

`element` keeps a length-1 axis, matching Julia `selectdim(..., i:i)`. `zerostate`
replaces that trailing axis. Category labels written into a fresh discrete state
are the last category, `K - 1` (Julia writes `K`, the last 1-based label).
"""

from __future__ import annotations

import numpy as np

from flowfusion._array import full_like, is_torch, zeros_like
from flowfusion.manifolds import Euclidean, SpecialOrthogonal, Torus
from flowfusion.states import ContinuousState, DiscreteState, ManifoldState, MaskedState, onehot, tensor, unhot


def element(state, seqindex, batchindex=None):
    if isinstance(state, tuple):
        return tuple(element(s, seqindex, batchindex) for s in state)
    if isinstance(state, MaskedState):
        return element(state.S, seqindex, batchindex)
    arr = tensor(state)
    if batchindex is not None:
        arr = arr[..., batchindex]
    arr = arr[..., seqindex : seqindex + 1]
    return _rewrap(state, arr)


def _rewrap(state, arr):
    if isinstance(state, ContinuousState):
        return ContinuousState(arr)
    if isinstance(state, DiscreteState):
        return DiscreteState(state.K, arr, state.onehot)
    if isinstance(state, ManifoldState):
        return ManifoldState(state.M, arr)
    if isinstance(state, MaskedState):
        return _rewrap(state.S, arr)
    raise TypeError(type(state).__name__)


def zerostate(example, *expandsize):
    if isinstance(example, tuple):
        return tuple(zerostate(part, *expandsize) for part in example)
    if isinstance(example, MaskedState):
        return MaskedState(
            zerostate(example.S, *expandsize),
            _zeros_mask(example.cmask, expandsize),
            _zeros_mask(example.lmask, expandsize),
        )
    if isinstance(example, ContinuousState):
        return ContinuousState(_expanded_zeros(example.state, expandsize))
    if isinstance(example, DiscreteState):
        labels = tensor(unhot(example)) if example.onehot else example.state
        filled = _expanded_fill(labels, expandsize, example.K - 1)
        state = DiscreteState(example.K, filled, False)
        return onehot(state) if example.onehot else state
    if isinstance(example, ManifoldState):
        return ManifoldState(example.M, _manifold_zeros(example, expandsize))
    raise TypeError(type(example).__name__)


def _expanded_zeros(array, expandsize):
    shape = tuple(array.shape[:-1]) + tuple(expandsize)
    if is_torch(array):
        return array.new_zeros(shape)
    return np.zeros(shape, dtype=array.dtype)


def _expanded_fill(array, expandsize, value):
    shape = tuple(array.shape[:-1]) + tuple(expandsize)
    if is_torch(array):
        return array.new_full(shape, int(value))
    return np.full(shape, int(value), dtype=array.dtype)


def _zeros_mask(mask, expandsize):
    return _expanded_zeros(mask, expandsize)


def _manifold_zeros(example: ManifoldState, expandsize):
    base = example.state
    shape = tuple(base.shape[:-1]) + tuple(expandsize)
    if isinstance(example.M, SpecialOrthogonal):
        # base is (3, 3, *old). Drop the last axis, then append expandsize.
        # Identity lives on the two matrix axes.
        if is_torch(base):
            out = base.new_zeros(shape)
        else:
            out = np.zeros(shape, dtype=base.dtype)
        out[0, 0] = 1
        out[1, 1] = 1
        out[2, 2] = 1
        return out
    if isinstance(example.M, (Euclidean, Torus)):
        return _expanded_zeros(base, expandsize)
    return _expanded_zeros(base, expandsize)


def regroup(elarray):
    """Pad a batch of variable-length tuples of states into dense tensors.

    `elarray[batch][position]` is a tuple of states, one per coupled process.
    The example element is expected to come from `element`, so it still has a
    trailing length-1 axis.
    """
    example = elarray[0][0]
    maxlen = max(len(seq) for seq in elarray)
    batch = len(elarray)
    newstates = [zerostate(example[k], maxlen, batch) for k in range(len(example))]
    for i, seq in enumerate(elarray):
        for j, item in enumerate(seq):
            for k in range(len(example)):
                dest = tensor(newstates[k])
                src = tensor(item[k])
                view = dest[..., j, i]
                while getattr(src, "ndim", 0) > view.ndim and src.shape[-1] == 1:
                    src = src[..., 0]
                dest[..., j, i] = src
    return tuple(newstates)
