import torch
from torch.optim.lr_scheduler import LambdaLR

from nomadflow.runtime.lmod import BiomoleculeModule


class TinyModel(torch.nn.Module):
    self_conditioning = False

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(1, 1)


def _make_module(**kwargs):
    defaults = dict(
        model=TinyModel(),
        corrupter=object(),
        optim=lambda params: torch.optim.Adam(params, lr=1e-3),
        use_ema=False,
        use_posthoc_ema=False,
        use_lr_step_decay=False,
        use_cosine_lr_sched=False,
        use_linear_warmup=False,
    )
    defaults.update(kwargs)
    return BiomoleculeModule(**defaults)


def test_configure_optimizers_returns_step_decay_scheduler():
    module = _make_module(
        use_lr_step_decay=True,
        use_linear_warmup=False,
        use_cosine_lr_sched=False,
        lr_step_decay_start=7000,
        lr_step_decay_step=3000,
        lr_step_decay_factor=0.5,
    )

    cfg = module.configure_optimizers()

    assert isinstance(cfg, dict)
    assert "optimizer" in cfg
    assert "lr_scheduler" in cfg
    assert isinstance(cfg["lr_scheduler"], LambdaLR)
    assert isinstance(cfg["optimizer"], torch.optim.Adam)


def test_configure_optimizers_step_decay_beats_cosine_and_warmup():
    """Priority must stay: step > cosine > warmup when flags overlap."""
    module = _make_module(
        use_lr_step_decay=True,
        use_cosine_lr_sched=True,
        use_linear_warmup=True,
        lr_step_decay_start=100,
        lr_step_decay_step=50,
        lr_step_decay_factor=0.5,
        cosine_warmup_steps=10,
        cosine_total_steps=1000,
        linear_warmup_steps=10,
    )

    cfg = module.configure_optimizers()

    assert isinstance(cfg, dict)
    assert isinstance(cfg["lr_scheduler"], LambdaLR)
    # Step-decay signature (distinct from warmup/cosine): flat until start, then factor**k
    assert cfg["lr_scheduler"].lr_lambdas[0](0) == 1.0
    assert cfg["lr_scheduler"].lr_lambdas[0](100) == 0.5
    assert cfg["lr_scheduler"].lr_lambdas[0](150) == 0.25
