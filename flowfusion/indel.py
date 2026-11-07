"""Insert and delete along the last axis.

`positions` follow the Julia convention, not 0-based "after this index":
`0` inserts before the first element, and `k >= 1` inserts after the `k`-th
element (1-based). `deletions[i]` refers to the `i`-th element in the usual
0-based sense. `selectlastdim` is 0-based.
"""

from __future__ import annotations

import numpy as np

from flowfusion._array import copy_array, is_torch
from flowfusion.states import ContinuousState, DiscreteState, ManifoldState, MaskedState, onehot, unhot


def _axis(ndim, offset):
    return ndim - 1 - offset


def combine(elements, new_elements, counts=None, positions=None, deletions=None, offset=0):
    if isinstance(elements, tuple):
        return tuple(
            combine(e, n, counts, positions, deletions, offset) for e, n in zip(elements, new_elements)
        )
    if isinstance(elements, MaskedState):
        return MaskedState(
            combine(elements.S, new_elements.S, counts, positions, deletions, offset),
            combine(elements.cmask, new_elements.cmask, counts, positions, deletions, offset),
            combine(elements.lmask, new_elements.lmask, counts, positions, deletions, offset),
        )
    if isinstance(elements, DiscreteState):
        if elements.K != new_elements.K:
            raise ValueError("Number of categories must match.")
        if elements.onehot or new_elements.onehot:
            left = unhot(elements)
            right = unhot(new_elements)
            merged = combine(left.state, right.state, counts, positions, deletions, offset)
            state = DiscreteState(elements.K, merged, False)
            return onehot(state) if elements.onehot else state
        merged = combine(elements.state, new_elements.state, counts, positions, deletions, offset)
        return DiscreteState(elements.K, merged, False)
    if isinstance(elements, ContinuousState):
        return ContinuousState(combine(elements.state, new_elements.state, counts, positions, deletions, offset))
    if isinstance(elements, ManifoldState):
        return ManifoldState(elements.M, combine(elements.state, new_elements.state, counts, positions, deletions, offset))
    return _combine_array(elements, new_elements, counts, positions, deletions, offset)


def _combine_array(elements, new_elements, counts, positions, deletions, offset):
    ndim = elements.ndim
    axis = _axis(ndim, offset)
    n = elements.shape[axis]
    if deletions is None:
        deletions = np.zeros(n, dtype=bool)
    else:
        deletions = np.asarray(deletions, dtype=bool)
    if counts is None:
        counts = [new_elements.shape[axis]]
    if positions is None:
        positions = [n]  # Julia default is size(elements)[end], 1-based length, meaning "after the last"? 
        # Julia default positions=[size(elements)[end]] which is n (1-based index of the last element),
        # so the insertion is after the last element. `n` as a 1-based index is the last element. Good.
        positions = [n]
    if len(deletions) != n:
        raise ValueError("deletions must have one entry per element along the combined axis.")
    insertions = {}
    for p, c in zip(positions, counts):
        insertions[int(p)] = insertions.get(int(p), 0) + int(c)
    num_kept = int((~deletions).sum())
    num_new = sum(insertions.values())
    combined_len = num_kept + num_new
    shape = list(elements.shape)
    shape[axis] = combined_len
    if is_torch(elements):
        out = elements.new_empty(shape)
    else:
        out = np.empty(shape, dtype=elements.dtype)
    dst = 0
    new_idx = 0

    def take_new(dst, new_idx, n_new):
        if n_new <= 0:
            return dst, new_idx
        if new_idx + n_new > new_elements.shape[axis] or dst + n_new > combined_len:
            return dst, new_idx
        _assign_range(out, axis, dst, dst + n_new, new_elements, new_idx, new_idx + n_new)
        return dst + n_new, new_idx + n_new

    if 0 in insertions:
        dst, new_idx = take_new(dst, new_idx, insertions[0])
    for src in range(n):
        if not deletions[src] and dst < combined_len:
            _assign_index(out, axis, dst, elements, src)
            dst += 1
        key = src + 1  # 1-based position of this element
        if key in insertions:
            dst, new_idx = take_new(dst, new_idx, insertions[key])
    if dst < combined_len:
        return _slice_range(out, axis, 0, dst)
    return out


def _assign_index(dest, axis, dest_i, src, src_i):
    dest[(slice(None),) * axis + (dest_i,)] = src[(slice(None),) * axis + (src_i,)]


def _assign_range(dest, axis, dest_a, dest_b, src, src_a, src_b):
    dest[(slice(None),) * axis + (slice(dest_a, dest_b),)] = src[(slice(None),) * axis + (slice(src_a, src_b),)]


def _slice_range(array, axis, start, stop):
    return array[(slice(None),) * axis + (slice(start, stop),)].clone() if is_torch(array) else array[(slice(None),) * axis + (slice(start, stop),)].copy()


def selectlastdim(x, index, offset=0):
    if isinstance(x, tuple):
        return tuple(selectlastdim(v, index, offset) for v in x)
    if isinstance(x, MaskedState):
        return MaskedState(
            selectlastdim(x.S, index, offset),
            selectlastdim(x.cmask, index, offset),
            selectlastdim(x.lmask, index, offset),
        )
    if isinstance(x, DiscreteState):
        labels = unhot(x).state if x.onehot else x.state
        chosen = _index_array(labels, index, offset)
        state = DiscreteState(x.K, chosen, False)
        return onehot(state) if x.onehot else state
    if isinstance(x, ContinuousState):
        return ContinuousState(_index_array(x.state, index, offset))
    if isinstance(x, ManifoldState):
        return ManifoldState(x.M, _index_array(x.state, index, offset))
    return _index_array(x, index, offset)


def _index_array(array, index, offset):
    axis = _axis(array.ndim, offset)
    chosen = array[(slice(None),) * axis + (index,)]
    return chosen.clone() if is_torch(chosen) else np.array(chosen, copy=True)


def lastsize(x, offset=0):
    if isinstance(x, tuple):
        sizes = {lastsize(v, offset) for v in x}
        if len(sizes) != 1:
            raise ValueError("Tuple states have different lengths.")
        return sizes.pop()
    if isinstance(x, MaskedState):
        return lastsize(x.S, offset)
    array = x.state if hasattr(x, "state") else x
    return int(array.shape[_axis(array.ndim, offset)])
