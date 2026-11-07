"""Coalescence policies.

Sequential policies only merge elements that are adjacent in the current list
and share a group. Pair indices are 0-based. Julia's tests use 1-based indices,
so the pair Julia reports as `(1, 2)` is `(0, 1)` here.
"""

from __future__ import annotations

import numpy as np

from flowfusion.states import tensor


class CoalescencePolicy:
    def init(self, nodes):
        return None

    def update(self, nodes, i, j, new_index):
        return None

    def reorder_forest(self, nodes):
        return nodes

    def should_append_on_split(self) -> bool:
        return False

    def select_coalescence(self, nodes, group_mins=None):
        raise NotImplementedError(type(self).__name__)

    def max_coalescences(self, nodes) -> int:
        return groupwise_max_coalescences(nodes)


class SequentialCoalescencePolicy(CoalescencePolicy):
    def max_coalescences(self, nodes) -> int:
        return max_coalescences_sequential(nodes)


class SequentialUniform(SequentialCoalescencePolicy):
    """Uniform draw among sequentially adjacent branchable pairs in the same group."""

    def select_coalescence(self, nodes, group_mins=None):
        pairs = eligible_sequential_pairs(nodes, group_mins)
        if not pairs:
            return None
        return pairs[int(np.random.randint(0, len(pairs)))]

    def __repr__(self):
        return "SequentialUniform()"


class BalancedSequential(SequentialCoalescencePolicy):
    """Prefer merging smaller adjacent clusters. `alpha = 0` is uniform."""

    def __init__(self, alpha=1.0):
        alpha = float(alpha)
        if alpha < 0:
            raise ValueError("alpha must be >= 0.")
        self.alpha = alpha

    def select_coalescence(self, nodes, group_mins=None):
        if group_mins is not None:
            raise NotImplementedError("Group mins are not implemented for BalancedSequential.")
        pairs = eligible_sequential_pairs(nodes, None)
        weights = [(nodes[i].weight + nodes[j].weight) ** (-self.alpha) for i, j in pairs]
        return weighted_choice(pairs, weights)

    def __repr__(self):
        return f"BalancedSequential(alpha={self.alpha})"


class RichGetRicherSequential(SequentialCoalescencePolicy):
    """Prefer merging larger adjacent clusters. `alpha = 0` is uniform."""

    def __init__(self, alpha=1.0):
        alpha = float(alpha)
        if alpha < 0:
            raise ValueError("alpha must be >= 0.")
        self.alpha = alpha

    def select_coalescence(self, nodes, group_mins=None):
        pairs = eligible_sequential_pairs(nodes, group_mins)
        weights = [(nodes[i].weight + nodes[j].weight) ** self.alpha for i, j in pairs]
        return weighted_choice(pairs, weights)

    def __repr__(self):
        return f"RichGetRicherSequential(alpha={self.alpha})"


class SequentialProximity(SequentialCoalescencePolicy):
    """Merge the closest adjacent pair. Ties within tolerance are broken uniformly."""

    def __init__(self, extractor=None, distance=None, tie_atol=0.0, tie_rtol=0.0):
        self.extractor = (lambda x: x) if extractor is None else extractor
        self.distance = euclidean_distance if distance is None else distance
        self.tie_atol = float(tie_atol)
        self.tie_rtol = float(tie_rtol)

    def select_coalescence(self, nodes, group_mins=None):
        pairs = eligible_sequential_pairs(nodes, group_mins)
        if not pairs:
            return None
        dists = []
        for i, j in pairs:
            xi = self.extractor(nodes[i].node_data)
            xj = self.extractor(nodes[j].node_data)
            dists.append(float(self.distance(xi, xj)))
        best = min(dists)
        tied = [
            pairs[k]
            for k, dist in enumerate(dists)
            if abs(dist - best) <= self.tie_atol + self.tie_rtol * abs(best)
        ]
        return tied[int(np.random.randint(0, len(tied)))]

    def __repr__(self):
        return f"SequentialProximity(tie_atol={self.tie_atol}, tie_rtol={self.tie_rtol})"


class SequentialDeepLineage(SequentialCoalescencePolicy):
    """Let a few trunks form, then only merge pairs that touch a deep lineage."""

    def __init__(self, lambda_=1.0, min_count=2, trunk_target_sampler=None):
        if min_count < 1:
            raise ValueError("min_count must be >= 1.")
        self.lambda_ = float(lambda_)
        self.min_count = int(min_count)
        if trunk_target_sampler is None:
            lam = self.lambda_

            def trunk_target_sampler():
                return 1 + int(np.random.poisson(lam))

        self.trunk_target_sampler = trunk_target_sampler
        self.target_trunks = {}

    def init(self, nodes):
        self.target_trunks = {}
        for node in nodes:
            if node.branchable and node.group not in self.target_trunks:
                self.target_trunks[node.group] = int(self.trunk_target_sampler())
        return None

    def select_coalescence(self, nodes, group_mins=None):
        pairs = _deep_lineage_pairs(self, nodes, group_mins)
        if not pairs:
            return None
        return pairs[int(np.random.randint(0, len(pairs)))]

    def __repr__(self):
        return f"SequentialDeepLineage(lambda_={self.lambda_}, min_count={self.min_count})"


def groupwise_max_coalescences(nodes) -> int:
    counts = {}
    for node in nodes:
        if node.branchable:
            counts[node.group] = counts.get(node.group, 0) + 1
    return sum(max(0, count - 1) for count in counts.values())


def max_coalescences_sequential(nodes) -> int:
    count = 0
    for i in range(len(nodes) - 1):
        left, right = nodes[i], nodes[i + 1]
        if left.branchable and right.branchable and left.group == right.group:
            count += 1
    return count


def eligible_sequential_pairs(nodes, group_mins):
    pairs = []
    group_sizes = _group_sizes(nodes)
    block_sizes = _block_sizes(nodes) if _is_int(group_mins) else None
    for i in range(len(nodes) - 1):
        left, right = nodes[i], nodes[i + 1]
        if not (left.branchable and right.branchable and left.group == right.group):
            continue
        if group_mins is None:
            allowed = True
        elif isinstance(group_mins, dict):
            allowed = group_sizes[left.group] > group_mins[left.group]
        elif _is_int(group_mins):
            allowed = block_sizes[i] > int(group_mins)
        else:
            raise TypeError(f"Unsupported group_mins type {type(group_mins).__name__}.")
        if allowed:
            pairs.append((i, i + 1))
    return pairs


def weighted_choice(pairs, weights):
    if not pairs:
        return None
    total = float(sum(weights))
    if total <= 0:
        return pairs[int(np.random.randint(0, len(pairs)))]
    cutoff = float(np.random.random()) * total
    running = 0.0
    for pair, weight in zip(pairs, weights):
        running += float(weight)
        if running >= cutoff:
            return pair
    return pairs[-1]


def euclidean_distance(x, y):
    """Distance used while choosing a merge. Tensor features are read on the host."""
    from flowfusion._array import is_torch

    def _host(value):
        value = tensor(value)
        if is_torch(value):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=float)

    diff = _host(x) - _host(y)
    return float(np.sqrt(np.sum(diff * diff)))


def _group_sizes(nodes):
    sizes = {}
    for node in nodes:
        if node.branchable:
            sizes[node.group] = sizes.get(node.group, 0) + 1
    return sizes


def _block_sizes(nodes):
    """Length of the contiguous branchable block containing each node."""
    n = len(nodes)
    forward = [0] * n
    current = 0
    for i, node in enumerate(nodes):
        continues = i > 0 and nodes[i - 1].branchable and nodes[i - 1].group == node.group
        if node.branchable and (i == 0 or continues):
            current += 1
        elif node.branchable:
            current = 1
        else:
            current = 0
        forward[i] = current
    reverse = [0] * n
    current = 0
    for i in range(n - 1, -1, -1):
        node = nodes[i]
        continues = i < n - 1 and nodes[i + 1].branchable and nodes[i + 1].group == node.group
        if node.branchable and (i == n - 1 or continues):
            current += 1
        elif node.branchable:
            current = 1
        else:
            current = 0
        reverse[i] = current
    return [forward[i] + reverse[i] - (1 if nodes[i].branchable else 0) for i in range(n)]


def _deep_lineage_pairs(policy: SequentialDeepLineage, nodes, group_mins):
    pairs = eligible_sequential_pairs(nodes, group_mins)
    if not pairs:
        return pairs
    deep_counts = {}
    for node in nodes:
        if node.branchable and node.weight >= policy.min_count:
            deep_counts[node.group] = deep_counts.get(node.group, 0) + 1
    filtered = []
    for i, j in pairs:
        group = nodes[i].group
        deep_active = deep_counts.get(group, 0) >= policy.target_trunks.get(group, 1)
        if (not deep_active) or nodes[i].weight >= policy.min_count or nodes[j].weight >= policy.min_count:
            filtered.append((i, j))
    return filtered if filtered else pairs


def _is_int(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)
