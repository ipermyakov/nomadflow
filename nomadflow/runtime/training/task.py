import abc
from typing import Dict, List, Optional
import numpy as np
import torch
from torch import Tensor


def resolve_np_rng(np_rng=None):
    return np_rng if np_rng is not None else np.random


def _validate_t_bounds(t_min: float, t_max: float) -> None:
    if not (0.0 <= t_min < t_max <= 1.0):
        raise ValueError(f"Need 0 <= t_min < t_max <= 1, got t_min={t_min}, t_max={t_max}")


def _affine_map_t(t: Tensor, t_min: float, t_max: float) -> Tensor:
    """Map unit-interval schedule samples into ``[t_min, t_max]``."""
    return t_min + (t_max - t_min) * t


def sample_task_t(
    *,
    t_sched: str,
    lognorm_mu: float,
    lognorm_sig: float,
    beta_p1: float,
    beta_p2: float,
    t_min: float = 0.01,
    t_max: float = 0.99,
    device: str = "cpu",
    np_rng=None,
    torch_generator: Optional[torch.Generator] = None,
) -> Tensor:
    _validate_t_bounds(t_min, t_max)

    if np_rng is not None:
        if t_sched == "lognorm":
            ln_sig = np_rng.normal(loc=lognorm_mu, scale=lognorm_sig, size=(1,)).astype(np.float32)
            t = torch.sigmoid(torch.from_numpy(ln_sig)).to(device)
            return _affine_map_t(t, t_min, t_max)
        if t_sched == "mixed_beta":
            u = float(np_rng.uniform())
            if u < 0.02:
                t = np_rng.uniform(size=(1,)).astype(np.float32)
            else:
                t = np_rng.beta(beta_p1, beta_p2, size=(1,)).astype(np.float32)
            t = torch.from_numpy(t).to(device)
            return _affine_map_t(t, t_min, t_max)
        if t_sched == "uniform":
            t = np_rng.uniform(size=(1,)).astype(np.float32)
            t = torch.from_numpy(t).to(device)
            return _affine_map_t(t, t_min, t_max)
        raise ValueError(f"t_sched={t_sched} not recognized")

    if t_sched == "lognorm":
        ln_sig = lognorm_mu + torch.randn(1, device=device, generator=torch_generator).float() * lognorm_sig
        t = torch.sigmoid(ln_sig)
        return _affine_map_t(t, t_min, t_max)
    if t_sched == "mixed_beta":
        u = torch.rand(1, generator=torch_generator)
        if u < 0.02:
            t = torch.rand(1, device=device, generator=torch_generator).float()
        else:
            dist = torch.distributions.beta.Beta(beta_p1, beta_p2)
            t = dist.sample((1,)).to(device)
        return _affine_map_t(t, t_min, t_max)
    if t_sched == "uniform":
        t = torch.rand(1, device=device, generator=torch_generator).float()
        return _affine_map_t(t, t_min, t_max)
    raise ValueError(f"t_sched={t_sched} not recognized")


class TrainingTask(abc.ABC):
    name: str = "abc"  # override this
    prob: float

    def sample_t_and_mask(self, batch, np_rng=None, torch_generator: Optional[torch.Generator] = None) -> Dict[str, Tensor]:
        raise NotImplementedError


class TaskSampler:
    def __init__(
        self,
        task_list: List[TrainingTask],
    ):
        task_probs = [t.prob for t in task_list]
        assert np.isclose(sum(task_probs), 1.0), sum(task_probs)

        self.task_list = task_list
        self.task_probs = task_probs

    def sample_task(self, np_rng=None) -> TrainingTask:
        rng = resolve_np_rng(np_rng)
        choice = rng.choice(a=len(self.task_probs), p=self.task_probs)
        return self.task_list[choice]
