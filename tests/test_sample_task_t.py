import numpy as np
import pytest
import torch

from nomadflow.runtime.training.task import sample_task_t


def _sample_many(*, t_sched, t_min, t_max, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        t = sample_task_t(
            t_sched=t_sched,
            lognorm_mu=0.0,
            lognorm_sig=1.0,
            beta_p1=1.9,
            beta_p2=1.0,
            t_min=t_min,
            t_max=t_max,
            np_rng=rng,
        )
        vals.append(float(t.item()))
    return np.asarray(vals, dtype=np.float64)


def test_uniform_samples_stay_in_narrow_window():
    t_min, t_max = 0.01, 0.245
    vals = _sample_many(t_sched="uniform", t_min=t_min, t_max=t_max)
    assert vals.min() >= t_min
    assert vals.max() <= t_max
    assert vals.mean() == pytest.approx(0.5 * (t_min + t_max), abs=0.01)


def test_lognorm_affine_remap_has_no_clamp_pileup():
    t_min, t_max = 0.01, 0.245
    vals = _sample_many(t_sched="lognorm", t_min=t_min, t_max=t_max, n=8000)
    assert vals.min() >= t_min
    assert vals.max() <= t_max
    # Affine remap spreads mass; clamp would put ~half+ of samples exactly on t_max.
    frac_near_max = float(np.mean(vals >= t_max - 1e-6))
    assert frac_near_max < 0.05
    # Mode of logit-normal(mu=0) maps near the window midpoint.
    assert np.median(vals) == pytest.approx(0.5 * (t_min + t_max), abs=0.02)


def test_default_window_bounds():
    vals = _sample_many(t_sched="lognorm", t_min=0.01, t_max=0.99)
    assert vals.min() >= 0.01
    assert vals.max() <= 0.99


def test_invalid_bounds_raise():
    with pytest.raises(ValueError):
        sample_task_t(
            t_sched="uniform",
            lognorm_mu=0.0,
            lognorm_sig=1.0,
            beta_p1=1.9,
            beta_p2=1.0,
            t_min=0.5,
            t_max=0.5,
            np_rng=np.random.default_rng(0),
        )


def test_torch_branch_respects_bounds():
    gen = torch.Generator().manual_seed(0)
    for _ in range(200):
        t = sample_task_t(
            t_sched="uniform",
            lognorm_mu=0.0,
            lognorm_sig=1.0,
            beta_p1=1.9,
            beta_p2=1.0,
            t_min=0.01,
            t_max=0.245,
            torch_generator=gen,
        )
        assert 0.01 <= float(t.item()) <= 0.245
