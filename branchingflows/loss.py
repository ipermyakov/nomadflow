"""Split and deletion losses.

The split head is a Bregman Poisson divergence. The deletion head is logit
binary cross-entropy. Both are masked means, using `flowfusion.scaledmaskedmean`,
so a CUDA prediction stays on CUDA. Base-process losses are forwarded to
`flowfusion.floss`.
"""

from __future__ import annotations

from flowfusion._array import asarray_like, softplus, xlogy
from flowfusion.loss import floss as _base_floss
from flowfusion.loss import scalefloss, scaledmaskedmean

from branchingflows.coalescent import CoalescentFlow, Deletion


def sbpl(mu, count):
    """Separable Bregman Poisson loss, including the count-only terms."""
    count = asarray_like(count, mu)
    return mu - xlogy(count, mu) - (count - xlogy(count, count))


def bpl(mu, count):
    """Bregman Poisson loss without the terms that depend only on the count."""
    count = asarray_like(count, mu)
    return mu - xlogy(count, mu)


def lbce(logits, target):
    """Logit binary cross-entropy, `(1 - y) * logit - log_sigmoid(logit)`."""
    target = asarray_like(target, logits)
    return (1.0 - target) * logits - log_sigmoid(logits)


def log_sigmoid(logits):
    return -softplus(-logits)


def floss(process, a, b, c, d=None):
    """Endpoint loss.

    Base process, same call as flowfusion: `floss(P, prediction, X1, scale)`.

    Splits: `floss(coalescent, split_logits, splits_target, mask, scale)`.

    Deletions: `floss(coalescent.deletion_policy, del_logits, deletion, mask, scale)`.
    """
    if isinstance(process, CoalescentFlow):
        if d is None:
            raise TypeError("floss(CoalescentFlow, split_logits, splits_target, mask, scale) needs a mask.")
        mu = process.split_transform(a)
        return scaledmaskedmean(sbpl(mu, b), d, c)
    if isinstance(process, Deletion):
        if d is None:
            raise TypeError("floss(Deletion(), del_logits, deletion, mask, scale) needs a mask.")
        return scaledmaskedmean(lbce(a, b), d, c)
    if d is None:
        return _base_floss(process, a, b, c)
    return _base_floss(process, a, b, c, d)


__all__ = ["bpl", "floss", "lbce", "log_sigmoid", "sbpl", "scalefloss"]
