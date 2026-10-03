""" Train a model """
import logging
import os
import glob
import shutil
from typing import Dict, Any, Optional

import hydra
from hydra_zen import zen, load_from_yaml, instantiate
import omegaconf
import torch
import numpy as np
import pandas as pd

from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch import Trainer

from nomadflow.runtime.config import config_sampling_hydra_store
from nomadflow.runtime.lmod import BiomoleculeSamplingModule, PDBWriter


# A logger for this file
log = logging.getLogger(__name__)


def _find_hydra_run_dir(start_path: str) -> Optional[str]:
    """Starting from a checkpoint file or a directory, walk parents until `.hydra/config.yaml` exists."""
    cur = os.path.abspath(os.path.expanduser(start_path))
    if os.path.isfile(cur):
        cur = os.path.dirname(cur)
    while True:
        cfg = os.path.join(cur, ".hydra", "config.yaml")
        if os.path.isfile(cfg):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


class Experiment:
    def __init__(self,
                 model,
                 sampler,
                 cfg):
        self._cfg = cfg
        self._sampler: LightningDataModule = sampler
        self._model: LightningModule = model

    def predict(self):
        kwargs: Dict[str, Any] = {
            "use_distributed_sampler": False
        }
        if torch.cuda.is_available():
            devices = list(range(torch.cuda.device_count()))
            if len(devices) > 1:
                kwargs['strategy'] = 'ddp_find_unused_parameters_true'
                kwargs['use_distributed_sampler'] = True
        else:
            devices = 1

        log.info(f"Using devices: {devices}")

        pred_writer = PDBWriter(
            output_dir=self._cfg['out_dir'],
            run_cfg=self._cfg
        )
        trainer = Trainer(
            # use_distributed_sampler=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=devices,
            callbacks=[pred_writer],
            **kwargs
        )
        trainer.predict(
            model=self._model,
            datamodule=self._sampler,
            ckpt_path=self._cfg['ckpt_path'],
            return_predictions=False,
            # PyTorch 2.6+: torch.load defaults to weights_only=True; Lightning ckpts
            # contain OmegaConf etc. Pretrained weights are a trusted source.
            weights_only=False,
        )


def main(sampler,
         corrupter,
         zen_cfg):
    # change into the output directory
    # os.chdir(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    log.info(f"Experiment started in folder: {os.getcwd()}")

    # so we can add extra entries in the config
    zen_cfg = omegaconf.OmegaConf.to_container(zen_cfg)
    assert zen_cfg is not None

    run_dir = (zen_cfg.get("model_dir") or "").strip()
    explicit = zen_cfg.get("checkpoint_path") or ""
    if isinstance(explicit, str):
        explicit = explicit.strip()
    if explicit:
        ckpt_path = os.path.abspath(os.path.expanduser(explicit))
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"checkpoint_path is not a file: {ckpt_path}")
        inferred = _find_hydra_run_dir(ckpt_path)
        if inferred:
            run_dir = inferred
            log.info("Using Hydra config from checkpoint path: %s", run_dir)
        elif not run_dir or not os.path.isdir(run_dir):
            raise FileNotFoundError(
                "Could not find `.hydra/config.yaml` in any parent directory of the "
                f"checkpoint ({ckpt_path}), and model_dir is missing or not a directory. "
                "Set model_dir to the training run directory that contains `.hydra/config.yaml`, "
                "or place that config next to the checkpoint (parent folders)."
            )
    else:
        if not run_dir or not os.path.isdir(run_dir):
            raise FileNotFoundError(
                "model_dir must be a directory when checkpoint_path is not set "
                f"(got {run_dir!r})."
            )
        ckpt_list = list(glob.glob(
            os.path.join(
                run_dir,
                "lightning_logs/version_0/checkpoints/*.ckpt",
            )
        ))
        epoch_list = []
        for path in ckpt_list:
            if path.split("/")[-1] == "last.ckpt":
                epoch_list.append((path, 1e6))
            else:
                epoch = path.split("=")[1].split("-")[0]
                epoch_list.append((path, int(epoch)))

        epoch_list = sorted(epoch_list, key=lambda x: x[1])
        epoch_list, _ = zip(*epoch_list)
        ckpt_path = epoch_list[zen_cfg["checkpoint_idx"]]
    log.info("Checkpoint: %s", ckpt_path)
    zen_cfg['ckpt_path'] = ckpt_path

    config_path = os.path.join(
        run_dir,
        ".hydra",
        "config.yaml"
    )
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Missing {config_path}. If you use checkpoint_path, ensure a parent folder "
            "contains `.hydra/config.yaml` from the same training run as the weights."
        )
    model_cfg = load_from_yaml(config_path)
    # lmodule_init = instantiate(model_cfg['lmodule'])
    model = instantiate(model_cfg['model'])

    # model = lmodule_init(model, corrupter, None)
    model = BiomoleculeSamplingModule(model, corrupter, zen_cfg)

    os.makedirs(zen_cfg['out_dir'], exist_ok=True)
    zen_cfg['samples_dir'] = os.path.join(
        zen_cfg['out_dir'], "samples"
    )
    os.makedirs(zen_cfg['samples_dir'], exist_ok=True)

    with open(os.path.join(zen_cfg['out_dir'], "run.log"), 'w') as fp:
        fp.write(f"Sampling config path: {zen_cfg['sampler']['tasks_yaml']}")
    shutil.copy(
        zen_cfg['sampler']['tasks_yaml'],
        os.path.join(zen_cfg['out_dir'], "config.yaml")
    )


    exp = Experiment(
        model=model,
        sampler=sampler,
        cfg=zen_cfg)
    exp.predict()


if __name__ == '__main__':
    config_sampling_hydra_store()
    torch.set_float32_matmul_precision("medium")
    zen(main, unpack_kwargs=True).hydra_main(
        config_name="main",
        version_base="1.2",
        config_path="."
    )
