"""Branching flows: variable-length generator matching on top of flowfusion.

NumPy states are the CPU-only path. PyTorch states run on CPU or CUDA; pass a
state through ``state.torch("cuda")`` when a GPU is available. Forest
construction is sequential and runs on the host. Anchor values, base-process
bridges, losses, and the forward split/deletion step use the arrays you pass
in, so a CUDA state stays on CUDA.

Category labels follow flowfusion: they are 0-based, and the mask token is
``K - 1``. Julia's mask token ``K`` is that same last category. Leaf ids still
start at 1, and a merged node has id 0, matching BranchingFlows.jl.
"""

from branchingflows.anchors import canonical_anchor_merge, select_anchor_merge
from branchingflows.coalescent import (
    CoalescentFlow,
    Deletion,
    Lineage,
    branching_bridge,
    default_split_transform,
    forest_bridge,
    gen,
    next_split_time,
    resolve_group_mins,
    sample_forest,
    step,
    tree_bridge,
    truncated_density,
)
from branchingflows.distributions import Uniform
from branchingflows.loss import bpl, floss, lbce, sbpl
from branchingflows.loss import scalefloss
from branchingflows.policies import (
    BalancedSequential,
    RichGetRicherSequential,
    SequentialDeepLineage,
    SequentialProximity,
    SequentialUniform,
)
from branchingflows.states import (
    BranchingBridge,
    BranchingState,
    fixedcount_del_insertions,
    group_fixedcount_del_insertions,
    uniform_del_insertions,
)
from branchingflows.trees import FlowNode

__all__ = [
    "BalancedSequential",
    "BranchingBridge",
    "BranchingState",
    "CoalescentFlow",
    "Deletion",
    "FlowNode",
    "Lineage",
    "RichGetRicherSequential",
    "SequentialDeepLineage",
    "SequentialProximity",
    "SequentialUniform",
    "Uniform",
    "bpl",
    "branching_bridge",
    "canonical_anchor_merge",
    "default_split_transform",
    "fixedcount_del_insertions",
    "floss",
    "forest_bridge",
    "gen",
    "group_fixedcount_del_insertions",
    "lbce",
    "next_split_time",
    "resolve_group_mins",
    "sample_forest",
    "sbpl",
    "scalefloss",
    "select_anchor_merge",
    "step",
    "tree_bridge",
    "truncated_density",
    "uniform_del_insertions",
]

__version__ = "0.1.0"
