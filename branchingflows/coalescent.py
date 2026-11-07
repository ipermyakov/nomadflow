"""Coalescent flow: sample a forest, bridge along its branches, and step splits forward."""

from __future__ import annotations

import math

import numpy as np

import flowfusion as ff
from flowfusion._array import clamp, exp, is_torch
from flowfusion.states import DiscreteState, tensor, unhot, unmask

from branchingflows.anchors import canonical_anchor_merge
from branchingflows.distributions import Uniform
from branchingflows.policies import SequentialUniform
from branchingflows.states import (
    BranchingBridge,
    BranchingState,
    _bool_on,
    _host_ints,
    _index_components,
    _mask_components,
    _take,
    group_fixedcount_del_insertions,
    reference_array,
)
from branchingflows.trees import FlowNode, mergenodes


class Deletion:
    """Tag for the deletion head. The loss is logit binary cross-entropy."""

    def __repr__(self):
        return "Deletion()"


class CoalescentFlow:
    """Base flow-fusion process plus split and deletion hazards.

    `P` is a flowfusion process or a tuple of them. `branch_time_dist` and
    `deletion_time_dist` need `cdf`, `pdf`, and `icdf` (or `quantile`).
    `flowfusion.BetaSchedule` and `branchingflows.Uniform` both qualify.

    Forests are built on the host because each merge depends on the last one.
    Anchor tensors, bridges, losses, and the forward step stay on the device of
    the state: a NumPy state is the CPU-only path, and a PyTorch state runs on
    CPU or CUDA.
    """

    def __init__(
        self,
        P,
        branch_time_dist,
        policy=None,
        deletion_time_dist=None,
        split_transform=None,
    ):
        self.P = P
        self.branch_time_dist = branch_time_dist
        self.split_transform = default_split_transform if split_transform is None else split_transform
        self.coalescence_policy = SequentialUniform() if policy is None else policy
        self.deletion_policy = Deletion()
        self.deletion_time_dist = Uniform(0.0, 1.0) if deletion_time_dist is None else deletion_time_dist

    def __repr__(self):
        return (
            f"CoalescentFlow(P={self.P!r}, branch_time_dist={self.branch_time_dist!r}, "
            f"policy={self.coalescence_policy!r}, deletion_time_dist={self.deletion_time_dist!r})"
        )


def default_split_transform(logits):
    """Map split logits to nonnegative intensities. Matches Julia's clamp to `[-100, 11]`."""
    return exp(clamp(logits, -100, 11))


def branching_bridge(
    process: CoalescentFlow,
    X0sampler,
    X1s,
    times,
    dtype=None,
    use_branching_time_prob=0.0,
    maxlen=math.inf,
    coalescence_factor=1.0,
    merger=canonical_anchor_merge,
    coalescence_policy=None,
    length_mins=None,
    deletion_pad=0.0,
    X1_modifier=None,
):
    """Sample a conditional bridge at each `(X1, t)`.

    Each item of `X1s` is one sequence, a `BranchingState` with groupings of
    shape `(L,)`. The returned `Xt` is padded to shape `(maxlen, B)`.

    `length_mins` is a minimum size per group: `None` (minimum 1), an int, a
    dict of ints, a 0-based count distribution with `.sample()`, or a list of
    those with one entry per batch item. `deletion_pad` inserts extra leaves
    marked for deletion so the forest can be at least as long as the minimum.
    """
    if coalescence_policy is None:
        coalescence_policy = process.coalescence_policy
    if X1_modifier is None:
        X1_modifier = lambda state: state
    if not X1s:
        raise ValueError("branching_bridge received an empty batch.")

    ref = reference_array(X1s[0].state)
    float_dtype = dtype if dtype is not None else _float_dtype(ref)
    times = _draw_times(times, len(X1s), ref, float_dtype)

    groupings = [x.groupings for x in X1s]
    if isinstance(length_mins, (list, tuple)) and not isinstance(length_mins, dict):
        if len(length_mins) != len(X1s):
            raise ValueError("A list of length_mins must have one entry per batch item.")
        resolved = [resolve_group_mins(item, groups) for item, groups in zip(length_mins, groupings)]
    else:
        resolved = [resolve_group_mins(length_mins, groups) for groups in groupings]

    if deletion_pad > 0:
        padded = []
        for i, X1 in enumerate(X1s):
            counts = _count_groups(X1.groupings)
            requests = {}
            for group, observed in counts.items():
                floor_len = max(observed, int(resolved[i][group]))
                if deletion_pad >= 1:
                    deterministic_extra = floor_len - observed
                    stochastic_extra = int(np.random.poisson((float(deletion_pad) - 1.0) * floor_len))
                    requests[group] = deterministic_extra + stochastic_extra
                else:
                    expected = float(deletion_pad) * floor_len
                    requests[group] = int(np.random.poisson(max(0.0, expected - observed)))
            padded.append(X1_modifier(group_fixedcount_del_insertions(X1, requests)))
        X1s = padded

    batch = []
    for i, X1 in enumerate(X1s):
        batch.append(
            forest_bridge(
                process,
                X0sampler,
                X1.state,
                _time_value(times, i),
                X1.groupings,
                X1.branchmask,
                X1.flowmask,
                X1.deletion,
                X1.ids,
                use_branching_time_prob=use_branching_time_prob,
                maxlen=maxlen,
                coalescence_factor=coalescence_factor,
                merger=merger,
                coalescence_policy=coalescence_policy,
                group_mins=resolved[i],
            )
        )
    return _collate(batch, ref, float_dtype)


def forest_bridge(
    process,
    X0sampler,
    X1,
    t,
    groups,
    branchable,
    flowable,
    deleted,
    ids=None,
    use_branching_time_prob=0.0,
    maxlen=math.inf,
    coalescence_factor=1.0,
    merger=canonical_anchor_merge,
    coalescence_policy=None,
    group_mins=None,
    _tries=0,
):
    if coalescence_policy is None:
        coalescence_policy = process.coalescence_policy
    elements = _elements(X1, groups)
    if ids is None:
        ids = np.arange(1, len(elements) + 1)
    forest, coal_times = sample_forest(
        process,
        elements,
        groupings=groups,
        branchable=branchable,
        flowable=flowable,
        deleted=deleted,
        ids=ids,
        coalescence_factor=coalescence_factor,
        merger=merger,
        coalescence_policy=coalescence_policy,
        group_mins=group_mins,
    )
    t = float(t)
    if coal_times and float(np.random.random()) < float(use_branching_time_prob):
        t = float(coal_times[int(np.random.randint(0, len(coal_times)))])
    occupied = len(forest) + sum(1 for stamp in coal_times if stamp <= t)
    if occupied > maxlen:
        if _tries > 32:
            raise RuntimeError("forest_bridge could not satisfy maxlen.")
        return forest_bridge(
            process,
            X0sampler,
            X1,
            t,
            groups,
            branchable,
            flowable,
            deleted,
            ids,
            use_branching_time_prob=use_branching_time_prob,
            maxlen=maxlen,
            coalescence_factor=coalescence_factor,
            merger=merger,
            coalescence_policy=coalescence_policy,
            group_mins=group_mins,
            _tries=_tries + 1,
        )
    collection = []
    for root in forest:
        tree_bridge(process, root, X0sampler(root), t, 0.0, collection)
    if not collection:
        raise RuntimeError("The conditional bridge deleted every element before the sample time.")
    return collection


def sample_forest(
    process: CoalescentFlow,
    elements,
    groupings=None,
    branchable=None,
    flowable=None,
    deleted=None,
    ids=None,
    coalescence_factor=1.0,
    merger=canonical_anchor_merge,
    coalescence_policy=None,
    group_mins=None,
):
    """Coalesce `elements` into a forest and assign absolute split times."""
    if coalescence_policy is None:
        coalescence_policy = process.coalescence_policy
    n = len(elements)
    groups = _host_vector(groupings, n, 0)
    can_branch = _host_vector(branchable, n, True)
    can_flow = _host_vector(flowable, n, True)
    is_deleted = _host_vector(deleted, n, False)
    id_values = _host_vector(ids, n, None)
    nodes = []
    for i, element in enumerate(elements):
        leaf_id = i + 1 if id_values[i] is None else int(id_values[i])
        nodes.append(
            FlowNode(
                1.0,
                _copy_data(element),
                weight=1,
                group=int(groups[i]),
                branchable=bool(can_branch[i]),
                deletion=bool(is_deleted[i]),
                id=leaf_id,
                flowable=bool(can_flow[i]),
            )
        )
    coalescence_policy.init(nodes)
    if _has_sample(coalescence_factor):
        coalescence_factor = float(coalescence_factor.sample())
    merges = int(np.random.binomial(coalescence_policy.max_coalescences(nodes), float(coalescence_factor)))
    for _ in range(merges):
        pair = coalescence_policy.select_coalescence(nodes, group_mins)
        if pair is None:
            break
        i, j = pair
        if i > j:
            i, j = j, i
        left, right = nodes[i], nodes[j]
        if left.group != right.group or not (left.branchable and right.branchable):
            raise RuntimeError("Coalescence policy returned a pair that cannot merge.")
        merged_data = merger(left.node_data, right.node_data, left.weight, right.weight)
        merged = mergenodes(
            left,
            right,
            0.0,
            merged_data,
            left.weight + right.weight,
            left.group,
            True,
            False,
            0,
            True,
        )
        nodes[i] = merged
        del nodes[j]
        coalescence_policy.update(nodes, i, j, i)
    nodes = coalescence_policy.reorder_forest(nodes)
    coal_times = []
    for root in nodes:
        _sample_split_times(process, root, 0.0, coal_times)
    return nodes, coal_times


def tree_bridge(process: CoalescentFlow, node: FlowNode, Xs, target_t, current_t, collection):
    target_t = float(target_t)
    current_t = float(current_t)
    if not node.flowable:
        collection.append(_segment(node.node_data, target_t, node, current_t, branchable=False, flowable=False))
        return
    if float(node.time) > target_t:
        if not (node.deletion and _deletion_hits(process.deletion_time_dist, current_t, target_t)):
            Xt = ff.bridge(process.P, Xs, node.node_data, target_t, current_t)
            collection.append(_segment(Xt, target_t, node, current_t, X1anchor=node.node_data))
        return
    nxt = ff.bridge(process.P, Xs, node.node_data, float(node.time), current_t)
    for child in node.children:
        tree_bridge(process, child, nxt, target_t, float(node.time), collection)


class Lineage:
    """Genealogy of one generated sequence.

    Each id keeps a polyline of `(t, x, y)`. A split records the parent of each
    new copy, and a deletion records the last point of a removed id. Child
    tracks start at the shared birth position, so the polylines fork there.
    """

    def __init__(self):
        self.tracks = {}
        self.parent = {}
        self.born = {}
        self.died = {}
        self.birth_xy = {}

    def observe(self, t, ids, xy):
        t = float(t)
        ids = [int(node_id) for node_id in ids]
        if len(ids) != int(xy.shape[1]):
            raise ValueError(f"Lineage.observe got {len(ids)} ids and {int(xy.shape[1])} positions.")
        for i, node_id in enumerate(ids):
            point = (t, float(xy[0, i]), float(xy[1, i]))
            track = self.tracks.setdefault(node_id, [])
            if not track and node_id in self.birth_xy and self.born[node_id] < t - 1e-12:
                bx, by = self.birth_xy[node_id]
                track.append((self.born[node_id], bx, by))
            if track and abs(track[-1][0] - t) <= 1e-12:
                continue
            track.append(point)

    def note_split(self, t, parent, child, x, y):
        child = int(child)
        self.parent[child] = int(parent)
        self.born[child] = float(t)
        self.birth_xy[child] = (float(x), float(y))

    def note_delete(self, t, node_id, x, y):
        node_id = int(node_id)
        t = float(t)
        self.died[node_id] = t
        track = self.tracks.setdefault(node_id, [])
        if not track or track[-1][0] < t - 1e-12:
            track.append((t, float(x), float(y)))

    @property
    def n_splits(self):
        return len(self.parent)

    @property
    def n_deletes(self):
        return len(self.died)


def step(process, Xt, hat, s1, s2, lineage=None):
    """Advance one Euler step.

    For a `CoalescentFlow`, `hat` is `(base_prediction, split_logits, deletion_logits)`.
    The base prediction is whatever `flowfusion.step` accepts for `process.P`
    (states or raw tensors). Batch size must be 1: splits change the length, and
    the Julia step is defined on a single sequence. A length-`L` state with no
    batch axis is accepted and returned in that same layout.

    Continuing elements keep their ids. Each extra copy receives a fresh id.
    Pass a `Lineage` to record those parent links and any deletions.
    """
    if not isinstance(process, CoalescentFlow):
        return ff.step(process, Xt, hat, s1, s2)
    if not isinstance(Xt, BranchingState):
        raise TypeError("CoalescentFlow.step expects a BranchingState.")
    squeezed = Xt.groupings.ndim == 1
    if squeezed:
        Xt = _add_batch_axis(Xt)
    if int(Xt.groupings.shape[1]) != 1:
        raise ValueError(
            "CoalescentFlow.step integrates one sequence. "
            f"groupings has shape {tuple(Xt.groupings.shape)}; loop over the batch."
        )
    updated = _step_one(process, Xt, hat, s1, s2, lineage=lineage)
    if squeezed:
        return _drop_batch_axis(updated)
    return updated


def gen(process, X0, model, steps, tracker=None, midpoint=False, lineage=None):
    """Integrate `X0` toward `model(t, Xt)`.

    A `CoalescentFlow` model returns `(base_prediction, split_logits, deletion_logits)`.
    Any other process is forwarded to `flowfusion.gen`. When `lineage` is a
    `Lineage`, the returned state carries the ids stored on that genealogy.
    A fully coalesced root (id 0) is renumbered before the first observation.
    """
    if not isinstance(process, CoalescentFlow):
        return ff.gen(process, X0, model, steps, tracker=tracker, midpoint=midpoint)
    state = X0.copy()
    grid = list(steps)
    if lineage is not None:
        state = _ensure_unique_ids(state)
        if grid:
            lineage.observe(grid[0], _host_ints(_column(state.ids)), _positions(state.state))
    for left, right in zip(grid[:-1], grid[1:]):
        t = (left + right) / 2 if midpoint else left
        hat = model(t, state)
        state = step(process, state, hat, left, right, lineage=lineage)
        if lineage is not None:
            lineage.observe(right, _host_ints(_column(state.ids)), _positions(state.state))
        if tracker is not None:
            tracker(t, state.copy(), hat)
    return state


def resolve_group_mins(length_mins, groupings):
    groups = _unique_groups(groupings)
    if length_mins is None:
        return {group: 1 for group in groups}
    if _is_int(length_mins):
        return {group: int(length_mins) for group in groups}
    if isinstance(length_mins, dict):
        resolved = {}
        for key, value in length_mins.items():
            resolved[int(key)] = int(value) if _is_int(value) else 1 + int(value.sample())
        for group in groups:
            resolved.setdefault(group, 1)
        return resolved
    if _has_sample(length_mins):
        return {group: 1 + int(length_mins.sample()) for group in groups}
    raise TypeError(f"Unsupported length_mins type {type(length_mins).__name__}.")


def _step_one(process: CoalescentFlow, Xt: BranchingState, hat, s1, s2, lineage=None):
    targets, event_logits, del_logits = hat
    resolved = ff.resolve_prediction(targets, Xt.state)
    moved = ff.mask(ff.step(process.P, Xt.state, resolved, s1, s2), Xt.state)
    length = int(Xt.groupings.shape[0])
    delta = float(s2) - float(s1)
    branch = _column(Xt.branchmask)
    logits = _align_head(event_logits, length)
    deletion_logits = _align_head(del_logits, length)
    intensities = process.split_transform(logits)
    rate = delta * truncated_density(process.branch_time_dist, s1) * _detach(intensities)
    counts = _poisson(rate)
    splits = _where(branch, counts, _zeros_like(counts))
    base_p = _deletion_probability(process.deletion_time_dist, s1, delta, intensities)
    rho = _sigmoid(_detach(deletion_logits))
    dels = branch & _bernoulli(rho * base_p)
    splits = _where(dels, _zeros_like(splits), splits)
    changed = _discrete_changes(Xt.state, moved, length)
    splits = _where(changed, _zeros_like(splits), splits)
    index, copies = _survivor_index(splits, dels)
    assigned = _assign_child_ids(Xt.ids, copies, lineage, s2, _positions(moved))
    new_state = _index_components(moved, index, axis=-2)
    new_branch = _take(branch, index, axis=0).reshape(-1, 1)
    new_flow = _take(_column(Xt.flowmask), index, axis=0).reshape(-1, 1)
    new_groups = _take(_column(Xt.groupings), index, axis=0).reshape(-1, 1)
    new_ids = _ids_column(new_groups, assigned)
    wrapped = _mask_components(new_state if isinstance(moved, tuple) else new_state, new_flow, new_flow)
    if not isinstance(moved, tuple) and isinstance(wrapped, tuple):
        wrapped = wrapped[0]
    return BranchingState(wrapped, new_groups, ids=new_ids, branchmask=new_branch, flowmask=new_flow)


def _survivor_index(splits, dels):
    keep = _logical_not(dels)
    copies = _where(keep, splits + 1, _zeros_like(splits))
    total = int(_sum_int(copies))
    if total < 1:
        deleted_at = _nonzero(dels)
        if len(deleted_at) == 0:
            raise RuntimeError("coalescent step produced an empty sequence without any deletions.")
        pick = deleted_at[int(np.random.randint(0, len(deleted_at)))]
        dels = _clone(dels)
        _set_index(dels, pick, False)
        copies = _where(_logical_not(dels), splits + 1, _zeros_like(splits))
    positions = _arange(splits)
    return _repeat(positions, copies), copies


def _assign_child_ids(old_ids, copies, lineage, t, xy):
    """Keep the surviving element's id and give each extra copy a fresh one."""
    previous = _host_ints(_column(old_ids))
    counts = _host_ints(copies)
    next_id = max(previous) + 1 if previous else 1
    if next_id < 1:
        next_id = 1
    assigned = []
    for i, count in enumerate(counts):
        x = float(xy[0, i])
        y = float(xy[1, i])
        if count <= 0:
            if lineage is not None:
                lineage.note_delete(t, previous[i], x, y)
            continue
        parent = previous[i]
        assigned.append(parent)
        for _ in range(count - 1):
            child = next_id
            next_id += 1
            assigned.append(child)
            if lineage is not None:
                lineage.note_split(t, parent, child, x, y)
    return assigned


def _ids_column(like, values):
    if is_torch(like):
        import torch

        column = torch.tensor(list(values), device=like.device, dtype=torch.long)
    else:
        column = np.asarray(list(values), dtype=np.int64)
    if getattr(like, "ndim", 1) == 2:
        return column.reshape(-1, 1)
    return column


def _ensure_unique_ids(state: BranchingState) -> BranchingState:
    host = _host_ints(_column(state.ids))
    if host and min(host) > 0 and len(set(host)) == len(host):
        return state
    return BranchingState(
        state.state,
        state.groupings,
        deletion=state.deletion,
        ids=_ids_column(state.groupings, list(range(1, len(host) + 1))),
        branchmask=state.branchmask,
        flowmask=state.flowmask,
        padmask=state.padmask,
    )


def _positions(components):
    """First two coordinates of the first component, shape `(2, L)` on the host."""
    component = components[0] if isinstance(components, tuple) else components
    arr = tensor(unmask(component))
    if is_torch(arr):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    width = int(arr.shape[-1])
    xs = np.asarray(arr[0], dtype=float)
    ys = np.zeros(width, dtype=float) if arr.shape[0] < 2 else np.asarray(arr[1], dtype=float)
    return np.stack([xs, ys])


def _discrete_changes(before, after, length):
    changed = None
    left = before if isinstance(before, tuple) else (before,)
    right = after if isinstance(after, tuple) else (after,)
    for old, new in zip(left, right):
        old_labels = _labels(old)
        new_labels = _labels(new)
        if old_labels is None or new_labels is None:
            continue
        old_labels = _column(old_labels)
        new_labels = _column(new_labels)
        diff = old_labels != new_labels
        changed = diff if changed is None else (changed | diff)
    if changed is None:
        ref = _column(reference_array(before))
        return _zeros_bool(ref, length)
    return changed


def _labels(component):
    inner = unmask(component)
    if not isinstance(inner, DiscreteState):
        return None
    if inner.onehot:
        indexed = unhot(inner)
        return indexed.state
    return inner.state


def _segment(Xt, t, node, current_t, branchable=None, flowable=None, X1anchor=None):
    return {
        "Xt": Xt,
        "t": float(t),
        "X1anchor": node.node_data if X1anchor is None else X1anchor,
        "descendants": int(node.weight),
        "deletion": bool(node.deletion),
        "branchable": node.branchable if branchable is None else branchable,
        "flowable": node.flowable if flowable is None else flowable,
        "group": int(node.group),
        "last_coalescence": float(current_t),
        "id": int(node.id),
    }


def _collate(batch, ref, float_dtype):
    maxlen = max(len(item) for item in batch)
    width = len(batch)
    flowmask = _ones_bool(ref, (maxlen, width))
    branchmask = _ones_bool(ref, (maxlen, width))
    descendants = _zeros_index(ref, (maxlen, width))
    padmask = _zeros_bool(ref, (maxlen, width))
    deletion = _zeros_bool(ref, (maxlen, width))
    prev = _zeros_float(ref, (maxlen, width), float_dtype)
    groups = _zeros_index(ref, (maxlen, width))
    ids = _zeros_index(ref, (maxlen, width))
    used = []
    for b, segments in enumerate(batch):
        used.append(segments[0]["t"])
        for i, seg in enumerate(segments):
            _set_matrix(flowmask, i, b, bool(seg["flowable"]))
            _set_matrix(branchmask, i, b, bool(seg["branchable"]))
            _set_matrix(descendants, i, b, int(seg["descendants"]))
            _set_matrix(padmask, i, b, True)
            _set_matrix(deletion, i, b, bool(seg["deletion"]))
            _set_matrix(prev, i, b, float(seg["last_coalescence"]))
            _set_matrix(groups, i, b, int(seg["group"]))
            _set_matrix(ids, i, b, int(seg["id"]))
    sequences = [[seg["Xt"] for seg in segments] for segments in batch]
    anchors = [[seg["X1anchor"] for seg in segments] for segments in batch]
    lmask = padmask & flowmask
    Xt = BranchingState(
        _mask_components(_batch_segments(sequences), flowmask, lmask),
        groups,
        deletion=deletion,
        ids=ids,
        branchmask=branchmask,
        flowmask=flowmask,
        padmask=padmask,
    )
    anchor = _mask_components(_batch_segments(anchors), flowmask, lmask)
    splits = _descendant_targets(descendants, ref, float_dtype)
    times = _times_array(used, ref, float_dtype)
    return BranchingBridge(times, Xt, anchor, deletion, descendants, splits, prev)


def _batch_segments(sequences):
    first = sequences[0][0]
    if isinstance(first, tuple):
        return ff.regroup(sequences)
    wrapped = [[(item,) for item in seq] for seq in sequences]
    return ff.regroup(wrapped)[0]


def _elements(X1, groups):
    if hasattr(groups, "shape"):
        length = int(groups.shape[0])
    else:
        length = len(groups)
    return [ff.element(X1, i) for i in range(length)]


def _sample_split_times(process, node, t0, collection):
    if node.weight > 1:
        nxt = next_split_time(process.branch_time_dist, node.weight, float(t0))
        node.time = nxt
        collection.append(nxt)
        for child in node.children:
            _sample_split_times(process, child, nxt, collection)


def next_split_time(dist, weight, t0):
    """Absolute time of the next split on a lineage with `weight` descendants."""
    if not 0.0 <= float(t0) < 1.0:
        raise ValueError("t0 must be in [0, 1).")
    if int(weight) < 2:
        raise ValueError("weight must be at least 2.")
    survival = 1.0 - _scalar(dist.cdf(t0))
    if survival <= 0:
        raise ValueError("The branch-time distribution has no mass after t0.")
    wait = float(np.random.exponential())
    remaining = survival * math.exp(-wait / (int(weight) - 1))
    drawn = _scalar(_icdf(dist, 1.0 - remaining))
    return float(min(max(drawn, float(t0)), 1.0))


def truncated_density(dist, lower, upper=1.0):
    """Density of `dist` truncated to `[lower, upper]`, evaluated at `lower`."""
    lower = float(lower)
    upper = float(upper)
    if lower >= upper:
        return 0.0
    mass = _scalar(dist.cdf(upper)) - _scalar(dist.cdf(lower))
    if mass <= 0:
        return 0.0
    return _scalar(dist.pdf(lower)) / mass


def _deletion_hits(dist, current_t, target_t):
    current = max(1.0 - _scalar(dist.cdf(current_t)), 0.0)
    target = max(1.0 - _scalar(dist.cdf(target_t)), 0.0)
    ratio = target / current if current > 0 else 0.0
    return float(np.random.random()) < (1.0 - ratio)


def _deletion_probability(dist, s1, delta, like):
    survival = max(1.0 - _scalar(dist.cdf(s1)), 0.0)
    hazard = (_scalar(dist.pdf(s1)) / survival) if survival > 0 else 0.0
    base = 1.0 - math.exp(-hazard * float(delta))
    return _full_float(like, base)


def _descendant_targets(descendants, ref, float_dtype):
    if is_torch(descendants):
        import torch

        dtype = float_dtype if float_dtype is not None else torch.float32
        shifted = descendants.to(dtype=dtype) - 1
        return torch.clamp(shifted, min=0)
    shifted = np.asarray(descendants, dtype=np.int64) - 1
    return np.maximum(shifted, 0).astype(_numpy_dtype(float_dtype), copy=False)


def _draw_times(times, n, ref, float_dtype):
    if _has_sample(times):
        drawn = times.sample(n)
        return _times_array(drawn, ref, float_dtype)
    if np.isscalar(times):
        return _times_array([float(times)] * n, ref, float_dtype)
    array = _times_array(times, ref, float_dtype)
    if int(array.shape[0]) != n:
        raise ValueError(f"times has length {array.shape[0]}, but the batch has {n} items.")
    return array


def _times_array(values, ref, float_dtype):
    if is_torch(ref):
        import torch

        dtype = float_dtype if _is_torch_dtype(float_dtype) else _float_dtype(ref)
        if is_torch(values):
            return values.to(device=ref.device, dtype=dtype)
        return torch.as_tensor(np.asarray(list(values), dtype=float), device=ref.device, dtype=dtype)
    return np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=_numpy_dtype(float_dtype))


def _time_value(times, index):
    value = times[index]
    if is_torch(value):
        return float(value.detach().cpu())
    return float(value)


def _count_groups(groupings):
    counts = {}
    for group in _host_ints(_column(groupings) if getattr(groupings, "ndim", 1) == 2 else groupings):
        counts[group] = counts.get(group, 0) + 1
    return counts


def _unique_groups(groupings):
    return list(_count_groups(groupings))


def _host_vector(value, n, default):
    if value is None:
        if default is None:
            return [None] * n
        return [default] * n
    if is_torch(value):
        flat = value.detach().reshape(-1).cpu().tolist()
    else:
        flat = np.asarray(value).reshape(-1).tolist()
    if len(flat) != n:
        raise ValueError(f"Expected {n} entries, got {len(flat)}.")
    return flat


def _copy_data(data):
    if isinstance(data, tuple):
        return tuple(_copy_data(part) for part in data)
    if hasattr(data, "copy"):
        return data.copy()
    return data


def _align_head(values, length):
    values = tensor(values)
    while getattr(values, "ndim", 0) > 1 and int(values.shape[0]) == 1:
        values = values[0]
    if getattr(values, "ndim", 0) == 2 and int(values.shape[1]) == 1:
        values = values[:, 0]
    if int(values.shape[0]) != length:
        raise ValueError(f"Head has length {values.shape[0]}, but the sequence has length {length}.")
    return values


def _column(array):
    if getattr(array, "ndim", 0) == 2 and int(array.shape[-1]) == 1:
        return array[:, 0]
    return array


def _add_batch_axis(state: BranchingState) -> BranchingState:
    return BranchingState(
        _map_state(state.state, lambda array: array[..., None] if not is_torch(array) else array.unsqueeze(-1)),
        state.groupings.reshape(-1, 1),
        deletion=state.deletion.reshape(-1, 1),
        ids=state.ids.reshape(-1, 1),
        branchmask=state.branchmask.reshape(-1, 1),
        flowmask=state.flowmask.reshape(-1, 1),
        padmask=state.padmask.reshape(-1, 1),
    )


def _drop_batch_axis(state: BranchingState) -> BranchingState:
    return BranchingState(
        _map_state(state.state, lambda array: array[..., 0]),
        state.groupings[:, 0],
        deletion=state.deletion[:, 0],
        ids=state.ids[:, 0],
        branchmask=state.branchmask[:, 0],
        flowmask=state.flowmask[:, 0],
        padmask=state.padmask[:, 0],
    )


def _map_state(state, fn):
    if isinstance(state, tuple):
        return tuple(_map_state(part, fn) for part in state)
    from flowfusion.states import MaskedState as _Masked

    if isinstance(state, _Masked):
        return _Masked(_map_state(state.S, fn), fn(state.cmask), fn(state.lmask))
    from branchingflows.states import _rewrap

    return _rewrap(state, fn(tensor(state)))


def _detach(values):
    if is_torch(values):
        return values.detach()
    return values


def _poisson(rate):
    if is_torch(rate):
        import torch

        return torch.poisson(torch.clamp(rate, min=0)).to(dtype=torch.long)
    return np.random.poisson(np.maximum(np.asarray(rate), 0)).astype(np.int64)


def _bernoulli(prob):
    if is_torch(prob):
        import torch

        return torch.rand(prob.shape, device=prob.device, dtype=prob.dtype) < prob
    return np.random.random(np.asarray(prob).shape) < np.asarray(prob)


def _sigmoid(values):
    if is_torch(values):
        import torch

        return torch.sigmoid(values)
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=float)))


def _where(cond, a, b):
    if is_torch(cond) or is_torch(a) or is_torch(b):
        import torch

        return torch.where(cond, a, b)
    return np.where(cond, a, b)


def _zeros_like(array):
    if is_torch(array):
        import torch

        return torch.zeros_like(array)
    return np.zeros_like(array)


def _logical_not(array):
    if is_torch(array):
        import torch

        return torch.logical_not(array)
    return np.logical_not(array)


def _clone(array):
    if is_torch(array):
        return array.clone()
    return np.array(array, copy=True)


def _set_index(array, index, value):
    array[index] = value


def _nonzero(array):
    if is_torch(array):
        return array.detach().nonzero(as_tuple=False).reshape(-1).cpu().tolist()
    return np.flatnonzero(np.asarray(array)).tolist()


def _sum_int(array):
    if is_torch(array):
        return int(array.sum().item())
    return int(np.sum(array))


def _arange(like):
    n = int(like.shape[0])
    if is_torch(like):
        import torch

        return torch.arange(n, device=like.device, dtype=torch.long)
    return np.arange(n, dtype=np.int64)


def _repeat(positions, copies):
    if is_torch(positions):
        import torch

        return torch.repeat_interleave(positions, copies.to(dtype=torch.long))
    return np.repeat(positions, np.asarray(copies, dtype=np.int64))


def _zeros_bool(ref, shape):
    if isinstance(shape, int):
        shape = (shape,)
    if is_torch(ref):
        import torch

        return torch.zeros(shape, dtype=torch.bool, device=ref.device)
    return np.zeros(shape, dtype=bool)


def _ones_bool(ref, shape):
    if is_torch(ref):
        import torch

        return torch.ones(shape, dtype=torch.bool, device=ref.device)
    return np.ones(shape, dtype=bool)


def _zeros_index(ref, shape):
    if is_torch(ref):
        import torch

        return torch.zeros(shape, dtype=torch.long, device=ref.device)
    return np.zeros(shape, dtype=np.int64)


def _zeros_float(ref, shape, float_dtype):
    if is_torch(ref):
        import torch

        dtype = float_dtype if _is_torch_dtype(float_dtype) else _float_dtype(ref)
        return torch.zeros(shape, dtype=dtype, device=ref.device)
    return np.zeros(shape, dtype=_numpy_dtype(float_dtype))


def _full_float(like, value):
    if is_torch(like):
        import torch

        dtype = like.dtype if like.is_floating_point() else torch.float32
        return torch.full(like.shape, float(value), dtype=dtype, device=like.device)
    return np.full(np.asarray(like).shape, float(value), dtype=np.asarray(like).dtype if np.issubdtype(np.asarray(like).dtype, np.floating) else np.float32)


def _set_matrix(array, i, j, value):
    array[i, j] = value


def _float_dtype(ref):
    if is_torch(ref):
        import torch

        if ref.is_floating_point():
            return ref.dtype
        return torch.float32
    arr = np.asarray(ref)
    if np.issubdtype(arr.dtype, np.floating):
        return arr.dtype
    return np.float32


def _numpy_dtype(dtype):
    if dtype is None or _is_torch_dtype(dtype):
        return np.float32
    return np.dtype(dtype)


def _is_torch_dtype(dtype) -> bool:
    return type(dtype).__module__.startswith("torch")


def _scalar(value):
    if is_torch(value):
        return float(value.detach().cpu().reshape(()))
    return float(np.asarray(value).reshape(()))


def _icdf(dist, u):
    if hasattr(dist, "icdf"):
        return dist.icdf(u)
    if hasattr(dist, "quantile"):
        return dist.quantile(u)
    if hasattr(dist, "ppf"):
        return dist.ppf(u)
    raise TypeError(f"{type(dist).__name__} needs icdf, quantile, or ppf.")


def _has_sample(value) -> bool:
    return hasattr(value, "sample") and callable(value.sample)


def _is_int(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)
