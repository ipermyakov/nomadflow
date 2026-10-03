""" Train a model """

import logging
import os
import shutil
import socket
import subprocess

import hydra
from hydra_zen import zen
import omegaconf
import torch
import optuna

from lightning import LightningDataModule, LightningModule
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from nomadflow.runtime.config import config_hydra_store, remove_zen_keys
from nomadflow.runtime.hpo import (
    OptunaPruningCallback,
    build_trial_tags,
    extract_training_trial_result,
    write_training_result,
)
from nomadflow.runtime.dataset_config import instantiate_dataset_from_config
from nomadflow.runtime.mlflow_logger import MlflowArtifactsCallback, MlflowLogger


# A logger for this file
log = logging.getLogger(__name__)


def _git_value(args):
    try:
        return subprocess.check_output(
            ["git", *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _build_mlflow_logger(exp_cfg, full_cfg):
    """Instantiate MlflowLogger from ``experiment.mlflow`` settings."""
    mlflow_cfg = omegaconf.OmegaConf.to_container(exp_cfg.mlflow, resolve=True)
    if not isinstance(mlflow_cfg, dict):
        mlflow_cfg = dict(mlflow_cfg)
    mlflow_cfg = mlflow_cfg.copy()
    mlflow_cfg.pop("enabled", False)
    mlflow_cfg.pop("log_hydra_artifacts", True)
    mlflow_cfg.pop("log_checkpoint_artifacts", True)
    log_experiment_config = mlflow_cfg.pop("log_experiment_config", True)
    nested_run = mlflow_cfg.get("nested_run", False)
    parent_run_id = mlflow_cfg.get("parent_run_id")
    tracking_uri = mlflow_cfg.get("tracking_uri")
    if not tracking_uri:
        repo_root = os.environ.get("REPO_ROOT") or os.getcwd()
        mlflow_cfg["tracking_uri"] = os.path.abspath(os.path.join(repo_root, "mlruns"))

    base_name = mlflow_cfg.get("run_name") or os.path.basename(os.getcwd())
    mlflow_cfg["run_name"] = base_name

    tags = mlflow_cfg.get("tags")
    if not isinstance(tags, dict):
        tags = {}
    dataset_config = omegaconf.OmegaConf.select(full_cfg, "dataset.config")
    val_dataset_config = omegaconf.OmegaConf.select(full_cfg, "dataset.val_config")
    tags = {
        "hostname": socket.gethostname(),
        "git_branch": _git_value(["rev-parse", "--abbrev-ref", "HEAD"]) or "unknown",
        "git_commit": _git_value(["rev-parse", "HEAD"]) or "unknown",
        "hydra_output_dir": os.getcwd(),
        "dataset_config": os.path.basename(str(dataset_config)) if dataset_config else "unknown",
        "val_dataset_config": os.path.basename(str(val_dataset_config)) if val_dataset_config else "none",
        **tags,
    }
    mlflow_cfg["tags"] = tags

    cleaned = remove_zen_keys(omegaconf.OmegaConf.create(mlflow_cfg))
    init_kwargs = omegaconf.OmegaConf.to_container(cleaned, resolve=True)
    if not isinstance(init_kwargs, dict):
        init_kwargs = {}
    init_kwargs["nested_run"] = bool(nested_run)
    init_kwargs["parent_run_id"] = parent_run_id

    logger = MlflowLogger(**init_kwargs)
    if log_experiment_config:
        cleaned_cfg = remove_zen_keys(full_cfg)
        logger.log_hyperparams(omegaconf.OmegaConf.to_container(cleaned_cfg, resolve=True))
    return logger


class Experiment:
    def __init__(self,
                 model,
                 datamodule,
                 checkpointer,
                 cfg,
                 objective_metric="val/loss",
                 optuna_trial=None):
        self._cfg = cfg
        self._exp_cfg = cfg.experiment
        self._datamodule: LightningDataModule = datamodule
        self._model: LightningModule = model
        self._checkpointer: ModelCheckpoint = checkpointer
        self._objective_metric = objective_metric
        self._optuna_trial = optuna_trial

    def _optuna_logger_context(self, logger):
        if not isinstance(logger, MlflowLogger):
            return None
        tags = logger.tags
        study_name = tags.get("optuna.study_name")
        trial_number = tags.get("optuna.trial_number")
        if study_name is None or trial_number is None:
            return None
        try:
            parsed_trial_number = int(trial_number)
        except (TypeError, ValueError):
            return None
        return {
            "study_name": str(study_name),
            "trial_number": parsed_trial_number,
            "objective_metric": str(tags.get("optuna.objective", self._objective_metric)),
        }

    def train(self):
        callbacks = []
        has_validation = getattr(self._datamodule, "val_dataset", None) is not None
        mlflow_on = bool(
            omegaconf.OmegaConf.select(self._cfg, "experiment.mlflow.enabled", default=False)
        )
        if mlflow_on and not self._cfg.debug:
            log.info("MLflow logging enabled.")
            logger = _build_mlflow_logger(self._exp_cfg, self._cfg)
            callbacks.append(
                MlflowArtifactsCallback(
                    log_hydra_artifacts=bool(
                        omegaconf.OmegaConf.select(
                            self._cfg,
                            "experiment.mlflow.log_hydra_artifacts",
                            default=True,
                        )
                    ),
                    log_checkpoint_artifacts=bool(
                        omegaconf.OmegaConf.select(
                            self._cfg,
                            "experiment.mlflow.log_checkpoint_artifacts",
                            default=True,
                        )
                    ),
                )
            )
        else:
            log.info("MLflow logging disabled.")
            logger = None

        # # Checkpoint directory.
        # ckpt_dir = self._exp_cfg.checkpointer.dirpath
        # os.makedirs(ckpt_dir, exist_ok=True)
        # log.info(f"Checkpoints saved to {ckpt_dir}")

        # Model checkpoints
        if has_validation and getattr(self._checkpointer, "monitor", None) is None:
            self._checkpointer.monitor = self._objective_metric
            self._checkpointer.mode = "min"
            self._checkpointer.save_top_k = 3
            self._checkpointer.save_on_train_epoch_end = False
            self._checkpointer.train_time_interval = None
        callbacks.append(self._checkpointer)
        if self._optuna_trial is not None:
            callbacks.append(
                OptunaPruningCallback(
                    self._optuna_trial,
                    monitor=self._objective_metric,
                )
            )

        if torch.cuda.is_available():
            devices = list(range(torch.cuda.device_count()))
        else:
            devices = 1

        log.info(f"Using devices: {devices}")

        # some kinda jenk code to override the defaults
        trainer_cfg = remove_zen_keys(self._exp_cfg.lightning)
        overrides = dict(
            callbacks=callbacks,
            logger=logger,
            enable_autolog_hparams=False,
            use_distributed_sampler=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=devices,
            # reload_dataloaders_every_n_epochs=1,
            # strategy='ddp_find_unused_parameters_true',
            # detect_anomaly=True
        )
        trainer_cfg = omegaconf.OmegaConf.to_container(trainer_cfg)
        trainer_cfg.update(overrides)

        trainer = Trainer(
            **trainer_cfg,
        )
        ckpt_path = self._exp_cfg.warm_start
        warm_start_weights_only = bool(
            omegaconf.OmegaConf.select(
                self._exp_cfg, "warm_start_weights_only", default=False
            )
        )
        if ckpt_path and warm_start_weights_only:
            log.info(
                "Loading warm-start weights only from %s "
                "(strict_loading=%s); not restoring Lightning training state.",
                ckpt_path,
                getattr(self._model, "strict_loading", True),
            )
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            incompatible = self._model.load_state_dict(
                state_dict,
                strict=bool(getattr(self._model, "strict_loading", True)),
            )
            if incompatible.missing_keys:
                log.warning(
                    "warm_start_weights_only missing %d keys (kept init): %s",
                    len(incompatible.missing_keys),
                    incompatible.missing_keys,
                )
            if incompatible.unexpected_keys:
                log.warning(
                    "warm_start_weights_only unexpected %d keys (ignored): %s",
                    len(incompatible.unexpected_keys),
                    incompatible.unexpected_keys,
                )
            ckpt_path = None
        try:
            trainer.fit(
                model=self._model,
                datamodule=self._datamodule,
                ckpt_path=ckpt_path,
                weights_only=False,
            )
        except optuna.TrialPruned:
            optuna_context = self._optuna_logger_context(logger)
            if optuna_context is not None:
                logger.set_tags(
                    build_trial_tags(
                        study_name=optuna_context["study_name"],
                        trial_number=optuna_context["trial_number"],
                        objective_metric=optuna_context["objective_metric"],
                        state="PRUNED",
                        pruned=True,
                    )
                )
                logger.finalize("pruned")
            raise
        except Exception:
            optuna_context = self._optuna_logger_context(logger)
            if optuna_context is not None:
                logger.set_tags(
                    build_trial_tags(
                        study_name=optuna_context["study_name"],
                        trial_number=optuna_context["trial_number"],
                        objective_metric=optuna_context["objective_metric"],
                        state="FAILED",
                        pruned=False,
                    )
                )
                logger.finalize("failed")
            raise

        result = extract_training_trial_result(
            trainer,
            metric_name=self._objective_metric,
            status="completed",
        )
        write_training_result(os.getcwd(), result)
        return result


def _default_output_dir():
    try:
        return hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    except ValueError:
        return os.getcwd()


def run_training_cfg(cfg, *, output_dir=None, optuna_trial=None, objective_metric="val/loss"):
    """Instantiate and execute one training trial from a full Hydra config."""
    output_dir = os.path.abspath(output_dir or _default_output_dir())
    os.makedirs(output_dir, exist_ok=True)
    prev_cwd = os.getcwd()
    os.chdir(output_dir)
    log.info(f"Experiment started in folder: {os.getcwd()}")

    try:
        dataset = cfg.dataset
        train_dataset = instantiate_dataset_from_config(dataset.config)
        shutil.copy(dataset.config, os.path.join(os.getcwd(), "dataset_config.yaml"))
        val_dataset = None
        val_config = dataset.get("val_config")
        if val_config:
            val_dataset = instantiate_dataset_from_config(val_config)
            shutil.copy(val_config, os.path.join(os.getcwd(), "val_dataset_config.yaml"))

        datamodule_ctor = hydra.utils.instantiate(cfg.datamodule)
        datamodule_inst = datamodule_ctor(train_dataset=train_dataset, val_dataset=val_dataset)
        model = hydra.utils.instantiate(cfg.model)
        corrupter = hydra.utils.instantiate(cfg.corrupter)
        lmodule = hydra.utils.instantiate(cfg.lmodule)
        optim = hydra.utils.instantiate(cfg.experiment.optim)
        checkpointer = hydra.utils.instantiate(cfg.experiment.checkpointer)

        exp = Experiment(
            model=lmodule(model, corrupter, optim),
            datamodule=datamodule_inst,
            checkpointer=checkpointer,
            cfg=cfg,
            objective_metric=objective_metric,
            optuna_trial=optuna_trial,
        )
        return exp.train()
    finally:
        os.chdir(prev_cwd)


def main(model,
         corrupter,
         lmodule,
         dataset,
         datamodule,
         experiment,
         zen_cfg):
    del model, corrupter, lmodule, dataset, datamodule, experiment
    run_training_cfg(zen_cfg)


if __name__ == '__main__':
    config_hydra_store()
    torch.set_float32_matmul_precision("medium")

    # we use 1.2 so we can use .cache at the root dir
    # and change into the output directory at main()
    zen(main).hydra_main(
        config_name="main",
        version_base="1.2",
        config_path="."
    )
