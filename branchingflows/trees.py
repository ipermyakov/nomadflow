"""Binary plane trees used to encode a coalescent forest."""

from __future__ import annotations


class FlowNode:
    """One node of a branching-flow forest.

    Leaves are the (possibly deletion-augmented) data elements. An internal node
    is a coalescence: in forward time it is a split. `weight` is the number of
    descendant leaves. Merged nodes use `id == 0`, are never deleted, and are
    always flowable. `id` values on leaves follow Julia and start at 1, so 0
    stays free as the internal-node sentinel.
    """

    def __init__(
        self,
        time,
        node_data,
        weight=1,
        group=0,
        branchable=True,
        deletion=False,
        id=1,
        flowable=True,
    ):
        self.parent = None
        self.children = []
        self.time = time
        self.node_data = node_data
        self.weight = int(weight)
        self.group = int(group)
        self.branchable = bool(branchable)
        self.deletion = bool(deletion)
        self.id = int(id)
        self.flowable = bool(flowable)

    def __getattr__(self, name):
        # Julia calls this field `del`. The name is a Python keyword, so the
        # stored attribute is `deletion` and `node.del` is the alias.
        if name == "del":
            return self.deletion
        raise AttributeError(name)

    def __repr__(self):
        return (
            f"FlowNode(time={self.time}, weight={self.weight}, group={self.group}, "
            f"branchable={self.branchable}, deletion={self.deletion}, id={self.id}, "
            f"flowable={self.flowable})"
        )


def addchild(parent: FlowNode, child: FlowNode):
    if child.parent is not None:
        raise ValueError("This child already has a parent.")
    parent.children.append(child)
    child.parent = parent


def mergenodes(left: FlowNode, right: FlowNode, time, node_data, weight, group, branchable, deletion, id, flowable):
    if left.parent is not None or right.parent is not None:
        raise ValueError("Cannot merge nodes that already have parents.")
    parent = FlowNode(time, node_data, weight, group, branchable, deletion, id, flowable)
    addchild(parent, left)
    addchild(parent, right)
    return parent
