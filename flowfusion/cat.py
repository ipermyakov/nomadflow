"""The Fourier cat used in the Flowfusion.jl examples."""

from __future__ import annotations

import numpy as np

# Coefficients of sin(k t) and cos(k t), k = 1..30. Missing harmonics are zero.
_X_SIN = {
    1: -721 / 4, 2: 196 / 3, 3: -86 / 3, 4: -131 / 2, 5: 477 / 14, 6: 27, 7: -29 / 2,
    8: 68 / 5, 9: 1 / 10, 10: 23 / 4, 12: -19 / 2, 13: -85 / 21, 14: 2 / 3, 15: 27 / 5,
    16: 7 / 4, 17: 17 / 9, 18: -4, 19: -1 / 2, 20: 1 / 6, 21: 6 / 7, 22: -1 / 8, 23: 1 / 3,
    24: 3 / 2, 25: 13 / 5, 26: 1, 27: -2, 28: 3 / 5, 29: -1 / 5, 30: 1 / 5,
}
_X_COS = {
    1: 2337 / 8, 2: -43 / 5, 3: 322 / 5, 4: -117 / 5, 5: -26 / 5, 6: -23 / 3, 7: 143 / 4,
    8: -11 / 4, 9: -31 / 3, 10: -13 / 4, 11: -9 / 2, 12: 41 / 20, 13: 8, 14: 2 / 3, 15: 6,
    16: 17 / 4, 17: -3 / 2, 18: -29 / 10, 19: 11 / 6, 20: 12 / 5, 21: 3 / 2, 22: 11 / 12,
    23: -4 / 5, 24: 1, 25: 17 / 8, 26: -7 / 2, 27: -5 / 6, 28: -11 / 10, 29: 1 / 2, 30: -1 / 5,
}
_Y_SIN = {
    1: -637 / 2, 2: -188 / 5, 3: -11 / 7, 4: -12 / 5, 5: 11 / 3, 6: -37 / 4, 7: 8 / 3,
    8: 65 / 6, 9: -32 / 5, 10: -41 / 4, 11: -38 / 3, 12: -47 / 8, 13: 5 / 4, 14: -41 / 7,
    15: -7 / 3, 16: -13 / 7, 17: 17 / 4, 18: -9 / 4, 19: 8 / 9, 20: 3 / 5, 21: -2 / 5,
    22: 4 / 3, 23: 1 / 3, 24: 3 / 5, 25: -3 / 5, 26: 6 / 5, 27: -1 / 5, 28: 10 / 9, 29: 1 / 3,
    30: -3 / 4,
}
_Y_COS = {
    1: -125 / 2, 2: -521 / 9, 3: -359 / 3, 4: 47 / 3, 5: -33 / 2, 6: -5 / 4, 7: 31 / 8,
    8: 9 / 10, 9: -119 / 4, 10: -17 / 2, 11: 22 / 3, 12: 15 / 4, 13: -5 / 2, 14: 19 / 6,
    15: 7 / 4, 16: 31 / 4, 17: -1, 18: 11 / 10, 19: -2 / 3, 20: 13 / 3, 21: -5 / 4, 22: 2 / 3,
    23: 1 / 4, 24: 5 / 6, 26: 3 / 4, 27: -1 / 2, 28: -1 / 10, 29: -1 / 3, 30: -1 / 19,
}


def _series(t, sin_coeff, cos_coeff):
    t = np.asarray(t, dtype=float)
    acc = np.zeros_like(t, dtype=float)
    for k, c in sin_coeff.items():
        acc = acc + c * np.sin(k * t)
    for k, c in cos_coeff.items():
        acc = acc + c * np.cos(k * t)
    return acc


def cat_shape(t):
    """Point on the Fourier cat. `t` is an angle in radians. Returns shape `(2,)` or `(2, *t.shape)`."""
    x = _series(t, _X_SIN, _X_COS)
    y = _series(t, _Y_SIN, _Y_COS)
    if np.isscalar(t):
        return np.stack([x, y])
    return np.stack([x, y], axis=0)


def random_literal_cat(*dims, sigma=0.05, rng=None):
    """Sample the cat silhouette used in the Julia README. Shape is `(2, *dims)`."""
    rng = np.random.default_rng(rng)
    n = int(np.prod(dims)) if dims else 1
    angles = rng.random(n) * 2 * np.pi
    pts = cat_shape(angles) / 200
    pts = pts + rng.standard_normal((2, n)) * float(sigma)
    return pts.reshape((2,) + tuple(int(d) for d in dims))


def discretize(x, d, lo, hi):
    """0-based version of Flowfusion.jl `discretize` (Julia returned 1..d)."""
    edges = np.linspace(lo, hi, int(d) - 1)
    x = np.asarray(x)
    return np.sum(x[..., None] >= edges, axis=-1).astype(np.int64)


def random_discrete_cat(*dims, d=32, lo=-2.5, hi=2.5, sigma=0.05, rng=None):
    return discretize(random_literal_cat(*dims, sigma=sigma, rng=rng), d, lo, hi)
