"""Batched branching state, and the deletion-augmentation of a data sequence."""

from __future__ import annotations

import numpy as np

from flowfusion._array import copy_array, is_torch, to_numpy, to_torch
from flowfusion.states import MaskedState, tensor, unmask


class BranchingState:
    """A flow state plus the bookkeeping a coalescent step needs.

    `groupings` has shape `(L,)` for one sequence or `(L, B)` for a batch.
    Elements coalesce and split only inside a group. `deletion` marks leaves
    that are scheduled to disappear by `t = 1` (Julia's field name is `del`).
    `branchmask` permits splits and deletions. `flowmask` lets the base process
    move; where it is false the value is held. `padmask` marks real positions
    in a padded batch.

    Masks use the flowfusion convention once they are stored on a `MaskedState`:
    `cmask == 0` pins a value and `cmask == 1` leaves it free. The bool masks on
    this object use `True` for "yes" (`branchable`, `flowable`, `valid`).
    """

    def __init__(
        self,
        state,
        groupings,
        deletion=None,
        ids=None,
        branchmask=None,
        flowmask=None,
        padmask=None,
    ):
        self.state = state
        self.groupings = _as_array(groupings, index=True)
        shape = tuple(self.groupings.shape)
        self.deletion = _as_bool(deletion, self.groupings, False)
        self.ids = _default_ids(self.groupings) if ids is None else _as_array(ids, index=True)
        if tuple(self.ids.shape) != shape:
            raise ValueError(f"ids shape {tuple(self.ids.shape)} does not match groupings {shape}.")
        self.branchmask = _as_bool(branchmask, self.groupings, True)
        self.flowmask = _as_bool(flowmask, self.groupings, True)
        self.padmask = _as_bool(padmask, self.groupings, True)
        for name in ("deletion", "branchmask", "flowmask", "padmask"):
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} shape {tuple(value.shape)} does not match groupings {shape}.")

    def __getattr__(self, name):
        if name == "del":
            return self.deletion
        raise AttributeError(name)

    def copy(self):
        return BranchingState(
            _copy_state(self.state),
            copy_array(self.groupings),
            deletion=copy_array(self.deletion),
            ids=copy_array(self.ids),
            branchmask=copy_array(self.branchmask),
            flowmask=copy_array(self.flowmask),
            padmask=copy_array(self.padmask),
        )

    def numpy(self):
        return BranchingState(
            _numpy_state(self.state),
            to_numpy(self.groupings),
            deletion=to_numpy(self.deletion),
            ids=to_numpy(self.ids),
            branchmask=to_numpy(self.branchmask),
            flowmask=to_numpy(self.flowmask),
            padmask=to_numpy(self.padmask),
        )

    def torch(self, device="cpu", dtype=None):
        return BranchingState(
            _torch_state(self.state, device, dtype),
            to_torch(self.groupings, device),
            deletion=to_torch(self.deletion, device),
            ids=to_torch(self.ids, device),
            branchmask=to_torch(self.branchmask, device),
            flowmask=to_torch(self.flowmask, device),
            padmask=to_torch(self.padmask, device),
        )

    def __repr__(self):
        return (
            f"BranchingState(length={self.groupings.shape[0]}, "
            f"batch={1 if self.groupings.ndim == 1 else self.groupings.shape[1]})"
        )


class BranchingBridge:
    """What `branching_bridge` returns for a batch of conditional paths."""

    def __init__(self, t, Xt, X1anchor, deletion, descendants, splits_target, prev_coalescence):
        self.t = t
        self.Xt = Xt
        self.X1anchor = X1anchor
        self.deletion = deletion
        self.descendants = descendants
        self.splits_target = splits_target
        self.prev_coalescence = prev_coalescence

    def __getattr__(self, name):
        if name == "del":
            return self.deletion
        raise AttributeError(name)

    def numpy(self):
        return BranchingBridge(
            to_numpy(self.t),
            self.Xt.numpy(),
            _numpy_state(self.X1anchor),
            to_numpy(self.deletion),
            to_numpy(self.descendants),
            to_numpy(self.splits_target),
            to_numpy(self.prev_coalescence),
        )

    def torch(self, device="cpu", dtype=None):
        return BranchingBridge(
            to_torch(self.t, device, dtype),
            self.Xt.torch(device, dtype),
            _torch_state(self.X1anchor, device, dtype),
            to_torch(self.deletion, device),
            to_torch(self.descendants, device),
            to_torch(self.splits_target, device, dtype),
            to_torch(self.prev_coalescence, device, dtype),
        )


def uniform_del_insertions(X1: BranchingState, del_p):
    """Duplicate each eligible element with probability `del_p`.

    The copy is placed before or after the original, and exactly one of the two
    is marked for deletion. Eligible positions are those with `flowmask` and
    `branchmask` set.
    """
    X1 = _as_single_sequence(X1)
    groupings, flow, branch, pad, ids = _sequence_fields(X1)
    length = int(groupings.shape[0])
    flags = (np.random.random(length) < float(del_p)) & _host_bool(flow) & _host_bool(branch)
    new_indices = []
    del_flags = []
    for i in range(length):
        if flags[i]:
            new_indices.append(i)
            new_indices.append(i)
            if np.random.random() < 0.5:
                del_flags.extend([False, True])
            else:
                del_flags.extend([True, False])
        else:
            new_indices.append(i)
            del_flags.append(False)
    return _gather_sequence(X1, new_indices, del_flags)


def fixedcount_del_insertions(X1: BranchingState, num_events):
    """Insert exactly `num_events` duplicates, sampling targets with replacement."""
    num_events = int(num_events)
    if num_events <= 0:
        return X1
    X1 = _as_single_sequence(X1)
    groupings, flow, branch, pad, ids = _sequence_fields(X1)
    eligible = [i for i in range(int(groupings.shape[0])) if _host_bool(flow)[i] and _host_bool(branch)[i]]
    if not eligible:
        return X1
    before, after, orig_del = _allocate_events(len(groupings), eligible, num_events)
    return _materialize_events(X1, before, after, orig_del)


def group_fixedcount_del_insertions(X1: BranchingState, group_num_events):
    """Insert a fixed number of duplicates in each group.

    `group_num_events` maps a group id to an event count. Targets are drawn
    with replacement from elements of that group whose flow and branch masks
    are set. Each event marks exactly one of the original or the new copy.
    """
    if not any(int(n) > 0 for n in group_num_events.values()):
        return X1
    X1 = _as_single_sequence(X1)
    groupings, flow, branch, pad, ids = _sequence_fields(X1)
    length = int(groupings.shape[0])
    host_groups = _host_ints(groupings)
    eligible = [i for i in range(length) if _host_bool(flow)[i] and _host_bool(branch)[i]]
    if not eligible:
        return X1
    before = [[] for _ in range(length)]
    after = [[] for _ in range(length)]
    orig_del = [False] * length
    actual = 0
    for group, count in group_num_events.items():
        count = int(count)
        if count <= 0:
            continue
        eligible_g = [i for i in eligible if host_groups[i] == int(group)]
        if not eligible_g:
            continue
        extra_before, extra_after, extra_orig = _allocate_events(length, eligible_g, count)
        for i in range(length):
            before[i].extend(extra_before[i])
            after[i].extend(extra_after[i])
            orig_del[i] = orig_del[i] or extra_orig[i]
        actual += count
    if actual == 0:
        return X1
    return _materialize_events(X1, before, after, orig_del)


def _allocate_events(length, eligible, num_events):
    before = [[] for _ in range(length)]
    after = [[] for _ in range(length)]
    orig_del = [False] * length
    for _ in range(num_events):
        i = eligible[int(np.random.randint(0, len(eligible)))]
        insert_before = np.random.random() < 0.5
        mark_original = bool(np.random.randint(0, 2)) and not orig_del[i]
        if mark_original:
            orig_del[i] = True
            flag = False
        else:
            flag = True
        if insert_before:
            before[i].append(flag)
        else:
            after[i].append(flag)
    return before, after, orig_del


def _materialize_events(X1, before, after, orig_del):
    new_indices = []
    del_flags = []
    for i in range(len(orig_del)):
        for flag in before[i]:
            new_indices.append(i)
            del_flags.append(bool(flag))
        new_indices.append(i)
        del_flags.append(bool(orig_del[i]))
        for flag in after[i]:
            new_indices.append(i)
            del_flags.append(bool(flag))
    return _gather_sequence(X1, new_indices, del_flags)


def _gather_sequence(X1: BranchingState, new_indices, del_flags) -> BranchingState:
    groupings, flow, branch, pad, ids = _sequence_fields(X1)
    index = _index_on(groupings, new_indices)
    state = _index_components(X1.state, index, axis=-1)
    flow_i = _take(flow, index, axis=0)
    return BranchingState(
        _mask_components(state, flow_i, flow_i),
        _take(groupings, index, axis=0),
        deletion=_bool_on(groupings, del_flags),
        ids=_take(ids, index, axis=0),
        branchmask=_take(branch, index, axis=0),
        flowmask=flow_i,
        padmask=_take(pad, index, axis=0),
    )


def _as_single_sequence(X1: BranchingState) -> BranchingState:
    groupings = X1.groupings
    if groupings.ndim == 1:
        return X1
    if groupings.ndim == 2 and int(groupings.shape[1]) == 1:
        return BranchingState(
            _drop_state_batch(X1.state),
            _column0(groupings),
            deletion=_column0(X1.deletion),
            ids=_column0(X1.ids),
            branchmask=_column0(X1.branchmask),
            flowmask=_column0(X1.flowmask),
            padmask=_column0(X1.padmask),
        )
    raise ValueError(
        "Deletion augmentation expects one sequence. "
        f"groupings has shape {tuple(groupings.shape)}."
    )


def _drop_state_batch(state):
    if isinstance(state, tuple):
        return tuple(_drop_state_batch(part) for part in state)
    if isinstance(state, MaskedState):
        return MaskedState(
            _drop_state_batch(state.S),
            _column0(state.cmask),
            _column0(state.lmask),
        )
    return _rewrap(state, tensor(state)[..., 0])


def _sequence_fields(X1: BranchingState):
    groupings = _column0(X1.groupings)
    if groupings.ndim != 1:
        raise ValueError(
            "Deletion augmentation expects one sequence. "
            f"groupings has shape {tuple(X1.groupings.shape)}."
        )
    return (
        groupings,
        _column0(X1.flowmask),
        _column0(X1.branchmask),
        _column0(X1.padmask),
        _column0(X1.ids),
    )


def _column0(array):
    if array.ndim == 2 and array.shape[1] == 1:
        return array[:, 0]
    return array


def _mask_components(state, cmask, lmask):
    if isinstance(state, tuple):
        return tuple(MaskedState(part, cmask, lmask) for part in state)
    return MaskedState(state, cmask, lmask)


def _index_components(state, index, axis):
    if isinstance(state, tuple):
        return tuple(_index_components(part, index, axis) for part in state)
    if isinstance(state, MaskedState):
        return _index_components(state.S, index, axis)
    arr = tensor(state)
    return _rewrap(state, _take(arr, index, axis))


def _rewrap(state, array):
    from flowfusion.states import ContinuousState, DiscreteState, ManifoldState

    if isinstance(state, ContinuousState):
        return ContinuousState(array)
    if isinstance(state, DiscreteState):
        return DiscreteState(state.K, array, state.onehot)
    if isinstance(state, ManifoldState):
        return ManifoldState(state.M, array)
    if isinstance(state, MaskedState):
        return _rewrap(state.S, array)
    raise TypeError(type(state).__name__)


def _take(array, index, axis):
    if is_torch(array):
        import torch

        if not is_torch(index):
            index = torch.as_tensor(np.asarray(index, dtype=np.int64), device=array.device)
        return array.index_select(axis, index.to(dtype=torch.long, device=array.device))
    if is_torch(index):
        index = index.detach().cpu().numpy()
    index = np.asarray(index, dtype=np.int64)
    return np.take(array, index, axis=axis)


def torch_long():
    import torch

    return torch.long


def _index_on(like, indices):
    if is_torch(like):
        import torch

        return torch.as_tensor(np.asarray(indices, dtype=np.int64), device=like.device, dtype=torch.long)
    return np.asarray(indices, dtype=np.int64)


def _bool_on(like, flags):
    if is_torch(like):
        import torch

        return torch.as_tensor(np.asarray(flags, dtype=bool), device=like.device, dtype=torch.bool)
    return np.asarray(flags, dtype=bool)


def _host_bool(array):
    if is_torch(array):
        return array.detach().cpu().numpy().astype(bool)
    return np.asarray(array, dtype=bool)


def _host_ints(array):
    if is_torch(array):
        return [int(v) for v in array.detach().cpu().tolist()]
    return [int(v) for v in np.asarray(array).tolist()]


def _as_array(value, index=False):
    if is_torch(value):
        if index and value.dtype not in _torch_index_dtypes():
            return value.to(dtype=torch_long())
        return value
    arr = np.asarray(value)
    if index and not np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.int64, copy=False)
    return arr


def _torch_index_dtypes():
    import torch

    return (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)


def _as_bool(value, like, fill):
    if value is None:
        return _full_bool(like, fill)
    if is_torch(like):
        import torch

        if is_torch(value):
            return value.to(device=like.device, dtype=torch.bool)
        return torch.as_tensor(np.asarray(value, dtype=bool), device=like.device)
    if is_torch(value):
        return value.detach().cpu().numpy().astype(bool)
    return np.asarray(value, dtype=bool)


def _full_bool(like, fill):
    if is_torch(like):
        import torch

        return torch.full(like.shape, bool(fill), dtype=torch.bool, device=like.device)
    return np.full(like.shape, bool(fill), dtype=bool)


def _default_ids(like):
    """1-based position ids, matching Julia. 0 remains the merged-node sentinel."""
    length = int(like.shape[0])
    if is_torch(like):
        import torch

        ids = torch.arange(1, length + 1, device=like.device, dtype=torch.long)
        if like.ndim == 2:
            ids = ids.unsqueeze(1).expand(like.shape).contiguous()
        return ids
    ids = np.arange(1, length + 1, dtype=np.int64)
    if like.ndim == 2:
        ids = np.broadcast_to(ids[:, None], like.shape).copy()
    return ids


def _copy_state(state):
    if isinstance(state, tuple):
        return tuple(_copy_state(part) for part in state)
    if hasattr(state, "copy"):
        return state.copy()
    return state


def _numpy_state(state):
    if isinstance(state, tuple):
        return tuple(_numpy_state(part) for part in state)
    if hasattr(state, "numpy"):
        return state.numpy()
    return to_numpy(state)


def _torch_state(state, device, dtype):
    if isinstance(state, tuple):
        return tuple(_torch_state(part, device, dtype) for part in state)
    if isinstance(state, MaskedState):
        return MaskedState(
            _torch_state(state.S, device, dtype),
            to_torch(state.cmask, device),
            to_torch(state.lmask, device),
        )
    from flowfusion.states import DiscreteState

    if isinstance(state, DiscreteState):
        # A floating dtype is for continuous components. Labels stay integers.
        return state.torch(device, None)
    if hasattr(state, "torch"):
        return state.torch(device, dtype)
    return to_torch(state, device, dtype)


def reference_array(state):
    """A leaf array, used to choose the backend and device of newly allocated masks."""
    if isinstance(state, tuple):
        return reference_array(state[0])
    if isinstance(state, MaskedState):
        return reference_array(unmask(state))
    return tensor(state)
