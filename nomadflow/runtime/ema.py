from copy import deepcopy
import itertools
from typing import List, Optional

import torch
from torch import nn
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype

Tensor = torch.Tensor

# as described by https://arxiv.org/abs/2312.02696
# and based off of pytorch AveragedModel

class EMAModel(nn.Module):
    def __init__(self,
                 model,
                 gamma,
                 device=None,
                 use_buffers=False
    ):
        super().__init__()
        self.module = deepcopy(model)
        self.gamma = gamma
        if device is not None:
            self.module = self.module.to(device)
        self.register_buffer(
            "n_averaged", torch.tensor(0, dtype=torch.long, device=device)
        )
        self.use_buffers = use_buffers

    def forward(self, *args, **kwargs):
        """Forward pass."""
        return self.module(*args, **kwargs)

    def avg_fn(self, param_old, param_new, t):
        beta_t = (1 - 1/t) ** (self.gamma + 1)
        return param_old * beta_t + (1 - beta_t) * param_new

    def update_parameters(self, model: nn.Module, batch_idx: int):
        """Update model parameters."""
        self_param = (
            itertools.chain(self.module.parameters(), self.module.buffers())
            if self.use_buffers
            else self.parameters()
        )
        model_param = (
            itertools.chain(model.parameters(), model.buffers())
            if self.use_buffers
            else model.parameters()
        )
        self_param_detached: List[Optional[Tensor]] = []
        model_param_detached: List[Optional[Tensor]] = []
        for p_averaged, p_model in zip(self_param, model_param):
            p_model_ = p_model.detach().to(p_averaged.device)
            self_param_detached.append(p_averaged.detach())
            model_param_detached.append(p_model_)
            if self.n_averaged == 0:
                p_averaged.detach().copy_(p_model_)

        if self.n_averaged > 0:
            for p_averaged, p_model in zip(  # type: ignore[assignment]
                self_param_detached, model_param_detached
            ):
                n_averaged = self.n_averaged.to(p_averaged.device)
                p_averaged.detach().copy_(
                    self.avg_fn(p_averaged.detach(), p_model, batch_idx)
                )

        if not self.use_buffers:
            # If not apply running averages to the buffers,
            # keep the buffers in sync with the source model.
            for b_swa, b_model in zip(self.module.buffers(), model.buffers()):
                b_swa.detach().copy_(b_model.detach().to(b_swa.device))
        self.n_averaged += 1