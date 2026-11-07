"""Array helpers shared by the NumPy and PyTorch backends.

NumPy arrays always live on CPU. PyTorch tensors keep whatever device they were
created on, including CUDA. Kernels in this package allocate new arrays from a
reference tensor, so a CUDA state stays on CUDA without a separate code path.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def is_torch(x) -> bool:
    return torch is not None and isinstance(x, torch.Tensor)


def is_array(x) -> bool:
    return is_torch(x) or isinstance(x, np.ndarray)


def torch_device(device):
    if torch is None:
        raise ImportError("PyTorch is not installed.")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False. "
            "NumPy and torch.device('cpu') still work on this machine."
        )
    return dev


def numpy_dtype_to_torch(dtype):
    if torch is None:
        raise ImportError("PyTorch is not installed.")
    dt = np.dtype(dtype)
    table = {
        np.dtype(np.float16): torch.float16,
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float64): torch.float64,
        np.dtype(np.int8): torch.int8,
        np.dtype(np.int16): torch.int16,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.int64): torch.int64,
        np.dtype(np.uint8): torch.uint8,
        np.dtype(np.bool_): torch.bool,
    }
    if dt not in table:
        raise TypeError(f"Cannot map NumPy dtype {dt} to PyTorch.")
    return table[dt]


def to_numpy(array):
    if is_torch(array):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def to_torch(array, device="cpu", dtype=None):
    dev = torch_device(device)
    if is_torch(array):
        return array.to(device=dev, dtype=dtype)
    np_arr = np.asarray(array)
    if dtype is None:
        dtype = numpy_dtype_to_torch(np_arr.dtype)
    return torch.as_tensor(np_arr, dtype=dtype, device=dev)


def copy_array(array):
    if is_torch(array):
        return array.clone()
    return np.array(array, copy=True)


def float_ref(array, dtype=None):
    """A floating scalar on the same backend and device as `array`."""
    if is_torch(array):
        if dtype is None:
            dtype = array.dtype if array.is_floating_point() else torch.float32
        return array.new_empty((), dtype=dtype)
    src = np.asarray(array)
    if dtype is None:
        dtype = src.dtype if np.issubdtype(src.dtype, np.floating) else np.float32
    return np.empty((), dtype=dtype)


def asarray_like(value, like):
    """Convert `value` onto the backend, dtype, and device of `like`."""
    if is_torch(like):
        if is_torch(value):
            return value.to(device=like.device, dtype=like.dtype)
        return torch.as_tensor(np.asarray(value), device=like.device, dtype=like.dtype)
    if is_torch(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.asarray(like).dtype)


def _is_scalar_array(t) -> bool:
    if isinstance(t, (float, int)) and not isinstance(t, bool):
        return True
    if is_torch(t):
        return t.ndim == 0
    if isinstance(t, np.ndarray):
        return t.ndim == 0
    return np.isscalar(t)


def expand(t, ndim: int):
    """Prepend singleton axes so `t` broadcasts over the leading dimensions.

    This matches `ForwardBackward.expand`: a length-`B` time vector becomes
    shape `(1, ..., 1, B)` against a state of rank `ndim`. Scalars are unchanged.
    """
    if _is_scalar_array(t):
        return t
    if is_torch(t):
        extra = ndim - t.ndim
        if extra < 0:
            raise ValueError(f"Cannot expand shape {tuple(t.shape)} to {ndim} dimensions.")
        if extra == 0:
            return t
        return t.reshape((1,) * extra + tuple(t.shape))
    arr = np.asarray(t)
    extra = ndim - arr.ndim
    if extra < 0:
        raise ValueError(f"Cannot expand shape {arr.shape} to {ndim} dimensions.")
    if extra == 0:
        return arr
    return arr.reshape((1,) * extra + arr.shape)


def broadcast_coeff(coeff, like):
    """Place `coeff` on `like`'s device and expand it to `like.ndim`."""
    coeff = asarray_like(coeff, like)
    if _is_scalar_array(coeff):
        return coeff
    return expand(coeff, like.ndim)


def zeros_like(array):
    if is_torch(array):
        return torch.zeros_like(array)
    return np.zeros_like(array)


def ones_like(array):
    if is_torch(array):
        return torch.ones_like(array)
    return np.ones_like(array)


def full_like(array, value):
    if is_torch(array):
        return torch.full_like(array, value)
    return np.full_like(array, value)


def randn(ref, shape):
    ref = float_ref(ref)
    if is_torch(ref):
        return torch.randn(*shape, device=ref.device, dtype=ref.dtype)
    return np.random.standard_normal(shape).astype(ref.dtype, copy=False)


def rand_uniform(ref, shape):
    ref = float_ref(ref)
    if is_torch(ref):
        return torch.rand(*shape, device=ref.device, dtype=ref.dtype)
    return np.random.random(shape).astype(ref.dtype, copy=False)


def randint(low, high, ref, shape):
    if is_torch(ref):
        return torch.randint(int(low), int(high), shape, device=ref.device, dtype=ref.dtype)
    return np.random.randint(int(low), int(high), size=shape, dtype=ref.dtype)


def where(cond, a, b):
    if is_torch(cond) or is_torch(a) or is_torch(b):
        a, b = _pair(a, b)
        if not is_torch(a):
            ref = cond if is_torch(cond) else None
            if ref is None:
                return np.where(cond, a, b)
            dtype = ref.dtype if ref.is_floating_point() or ref.dtype == torch.bool else torch.float32
            a = torch.as_tensor(a, device=ref.device, dtype=dtype)
            b = torch.as_tensor(b, device=ref.device, dtype=dtype)
        if not is_torch(cond):
            cond = torch.as_tensor(cond, device=a.device)
        return torch.where(cond, a, b)
    return np.where(cond, a, b)


def sum_keep(x, dim=0):
    if is_torch(x):
        return torch.sum(x, dim=dim, keepdim=True)
    return np.sum(x, axis=dim, keepdims=True)


def sum_drop(x, dim=0):
    if is_torch(x):
        return torch.sum(x, dim=dim)
    return np.sum(x, axis=dim)


def mean_all(x):
    if is_torch(x):
        return torch.mean(x)
    return np.mean(x)


def clamp(x, lo, hi):
    if is_torch(x):
        return torch.clamp(x, lo, hi)
    return np.clip(x, lo, hi)


def _pair(a, b):
    """Broadcast a Python number onto the other argument's device and dtype."""
    if is_torch(a) or is_torch(b):
        if not is_torch(a):
            a = torch.as_tensor(a, device=b.device, dtype=b.dtype)
        elif not is_torch(b):
            b = torch.as_tensor(b, device=a.device, dtype=a.dtype)
    return a, b


def maximum(a, b):
    a, b = _pair(a, b)
    if is_torch(a):
        return torch.maximum(a, b)
    return np.maximum(a, b)


def minimum(a, b):
    a, b = _pair(a, b)
    if is_torch(a):
        return torch.minimum(a, b)
    return np.minimum(a, b)


def exp(x):
    if is_torch(x):
        return torch.exp(x)
    return np.exp(x)


def log(x):
    if is_torch(x):
        return torch.log(x)
    return np.log(x)


def sqrt(x):
    if is_torch(x):
        return torch.sqrt(x)
    return np.sqrt(x)


def cos(x):
    if is_torch(x):
        return torch.cos(x)
    return np.cos(x)


def sin(x):
    if is_torch(x):
        return torch.sin(x)
    return np.sin(x)


def acos(x):
    if is_torch(x):
        return torch.acos(x)
    return np.arccos(x)


def abs_(x):
    if is_torch(x):
        return torch.abs(x)
    return np.abs(x)


def isfinite(x):
    if is_torch(x):
        return torch.isfinite(x)
    return np.isfinite(x)


def isinf(x):
    if is_torch(x):
        return torch.isinf(x)
    return np.isinf(x)


def any_(x) -> bool:
    if is_torch(x):
        return bool(torch.any(x).item())
    return bool(np.any(x))


def matmul(a, b):
    if is_torch(a) or is_torch(b):
        return torch.matmul(a, b)
    return np.matmul(a, b)


def move_last_to_front(x):
    if is_torch(x):
        return torch.movedim(x, -1, 0)
    return np.moveaxis(x, -1, 0)


def log_softmax(x, dim=0):
    if is_torch(x):
        return torch.log_softmax(x, dim=dim)
    m = np.max(x, axis=dim, keepdims=True)
    shifted = x - m
    return shifted - np.log(np.exp(shifted).sum(axis=dim, keepdims=True))


def softmax(x, dim=0):
    if is_torch(x):
        return torch.softmax(x, dim=dim)
    m = np.max(x, axis=dim, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=dim, keepdims=True)


def softplus(x):
    if is_torch(x):
        return torch.nn.functional.softplus(x)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def xlogy(x, y):
    if is_torch(x) or is_torch(y):
        return torch.xlogy(x, y)
    import scipy.special

    return scipy.special.xlogy(np.asarray(x), np.asarray(y))


def norm(x, dim=0):
    if is_torch(x):
        return torch.linalg.vector_norm(x, dim=dim)
    return np.linalg.norm(x, axis=dim)


def stack(arrays, axis=-1):
    if is_torch(arrays[0]):
        return torch.stack(list(arrays), dim=axis)
    return np.stack(list(arrays), axis=axis)


def concat(arrays, axis=-1):
    if is_torch(arrays[0]):
        return torch.cat(list(arrays), dim=axis)
    return np.concatenate(list(arrays), axis=axis)


def matrix_exp(a):
    if is_torch(a):
        return torch.matrix_exp(a)
    import scipy.linalg

    return scipy.linalg.expm(np.asarray(a))


def numel(x) -> int:
    if is_torch(x):
        return int(x.numel())
    return int(np.size(x))


def reshape(x, shape):
    if is_torch(x):
        return x.reshape(shape)
    return np.reshape(x, shape)


def astype_float(x, ref):
    return asarray_like(x, float_ref(ref))
