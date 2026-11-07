"""Time distributions on an interval, evaluated on NumPy arrays or PyTorch tensors.

`BetaSchedule` from flowfusion is also a valid branch-time distribution: it
already implements `cdf`, `pdf`, and `icdf` for both backends.
"""

from __future__ import annotations

import numpy as np

from flowfusion._array import is_torch


class Uniform:
    """Uniform density on `[low, high]`. The Julia default deletion law is `Uniform(0, 1)`."""

    def __init__(self, low=0.0, high=1.0):
        self.low = float(low)
        self.high = float(high)
        if self.high <= self.low:
            raise ValueError("Uniform high must be greater than low.")

    def pdf(self, x):
        return _apply(x, lambda z: _uniform_pdf(z, self.low, self.high))

    def cdf(self, x):
        return _apply(x, lambda z: _uniform_cdf(z, self.low, self.high))

    def icdf(self, u):
        return _apply(u, lambda z: self.low + z * (self.high - self.low))

    def sample(self, n=None):
        if n is None:
            return float(np.random.uniform(self.low, self.high))
        return np.random.uniform(self.low, self.high, size=int(n))

    def __repr__(self):
        return f"Uniform({self.low}, {self.high})"


def _uniform_pdf(x, low, high):
    width = high - low
    inside = (x >= low) & (x <= high)
    return np.where(inside, 1.0 / width, 0.0)


def _uniform_cdf(x, low, high):
    width = high - low
    z = (x - low) / width
    return np.clip(z, 0.0, 1.0)


def _apply(x, fn):
    """Run a NumPy ufunc-style map and, for a tensor, put the result back on its device."""
    if is_torch(x):
        import torch

        device, dtype, shape = x.device, x.dtype, tuple(x.shape)
        arr = x.detach().cpu().numpy()
        out = np.asarray(fn(np.asarray(arr, dtype=float)), dtype=float)
        float_dtype = dtype if x.is_floating_point() else torch.float32
        return torch.as_tensor(out, device=device, dtype=float_dtype).reshape(shape)
    arr = np.asarray(x, dtype=float)
    out = np.asarray(fn(arr), dtype=float)
    if np.isscalar(x):
        return out.reshape(())
    if isinstance(x, np.ndarray):
        return out.astype(x.dtype, copy=False) if np.issubdtype(x.dtype, np.floating) else out
    return out
