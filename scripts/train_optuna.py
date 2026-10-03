"""Hyperparameter optimization entrypoint powered by Optuna."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

from hydra_zen import zen
import omegaconf
import optuna
import torch

from nomadflow.runtime.config import config_hydra_store
from nomadflow.runtime.hpo import (
    build_trial_tags,
    read_training_result,
    log_json_artifact,
    summarize_study,
)
from nomadflow.runtime.mlflow_logger import get_or_create_mlflow_experiment_id


def _clone_cfg(cfg):
    return omegaconf.OmegaConf.create(
        omegaconf.OmegaConf.to_container(cfg, resolve=True)
    )


def _build_sampler(sampler_cfg):
    name = str(sampler_cfg.name).lower()
    if name == "tpe":
        seed = sampler_cfg.get("seed")
        return optuna.samplers.TPESampler(seed=seed)
    raise ValueError(f"Unsupported Optuna sampler: {sampler_cfg.name}")


def _build_pruner(pruner_cfg):
    name = str(pruner_cfg.name).lower()
    if name == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=int(pruner_cfg.get("n_startup_trials", 5)),
            n_warmup_steps=int(pruner_cfg.get("n_warmup_steps", 1)),
            interval_steps=int(pruner_cfg.get("interval_steps", 1)),
        )
    if name == "none":
        return optuna.pruners.NopPruner()
    raise ValueError(f"Unsupported Optuna pruner: {pruner_cfg.name}")


def _suggest_value(trial: optuna.trial.Trial, name: str, spec: dict[str, Any]) -> Any:
    kind = str(spec["type"]).lower()
    if kind == "float":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            log=bool(spec.get("log", False)),
            step=spec.get("step"),
        )
    if kind == "int":
        return trial.suggest_int(
            name,
            int(spec["low"]),
            int(spec["high"]),
            step=int(spec.get("step", 1)),
            log=bool(spec.get("log", False)),
        )
    if kind == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    raise ValueError(f"Unsupported search-space type '{kind}' for '{name}'.")


def _apply_trial_overrides(cfg, overrides):
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Trial override must look like path=value, got: {override}")
        parsed_cfg = omegaconf.OmegaConf.from_dotlist([override])
        path, _ = override.split("=", 1)
        value = omegaconf.OmegaConf.select(parsed_cfg, path)
        omegaconf.OmegaConf.update(cfg, path, value, merge=False)


def _prepare_storage_uri(storage_uri: str) -> str:
    """Create local parent directories for sqlite-backed Optuna storage."""
    if storage_uri.startswith("sqlite:///"):
        db_path = storage_uri[len("sqlite:///"):]
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    return storage_uri


def _validate_hpo_cfg(cfg) -> None:
    hpo_cfg = cfg.hpo
    if str(hpo_cfg.execution_backend).lower() != "local":
        raise NotImplementedError(
            "Optuna v1 only supports hpo.execution_backend=local. "
            "The cluster backend is reserved for a later iteration."
        )
    if not bool(omegaconf.OmegaConf.select(cfg, "experiment.mlflow.enabled", default=False)):
        raise ValueError("Optuna v1 requires experiment.mlflow.enabled=true.")
    objective_metric = str(hpo_cfg.objective_metric)
    if not objective_metric.startswith("val/"):
        raise ValueError(
            f"Optuna objective must be a validation metric, got '{objective_metric}'."
        )
    val_config = omegaconf.OmegaConf.select(cfg, "dataset.val_config")
    if not val_config:
        raise ValueError("Optuna v1 requires dataset.val_config to be configured.")
    max_epochs = omegaconf.OmegaConf.select(cfg, "experiment.lightning.max_epochs")
    max_steps = omegaconf.OmegaConf.select(cfg, "experiment.lightning.max_steps")
    has_finite_epochs = max_epochs is not None and int(max_epochs) >= 0
    has_finite_steps = max_steps is not None and int(max_steps) > 0
    if not has_finite_epochs and not has_finite_steps:
        raise ValueError(
            "Optuna v1 requires a finite trial budget. Set either "
            "experiment.lightning.max_epochs or experiment.lightning.max_steps."
        )


def _train_script_path() -> str:
    repo_root = os.environ.get("REPO_ROOT")
    if repo_root:
        return os.path.join(repo_root, "train.py")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train.py")


def _trial_result_path(trial_dir: str) -> str:
    return os.path.join(trial_dir, "metadata", "training_result.json")


def _build_trial_cfg(
    base_cfg,
    trial: optuna.trial.Trial,
    study_dir: str,
    *,
    parent_run_id: str,
):
    cfg = _clone_cfg(base_cfg)
    hpo_cfg = cfg.hpo
    params: dict[str, Any] = {}
    search_space = omegaconf.OmegaConf.to_container(hpo_cfg.search_space, resolve=True)
    if not isinstance(search_space, dict):
        raise TypeError("hpo.search_space must resolve to a mapping.")
    for path, spec in search_space.items():
        if not isinstance(spec, dict):
            raise TypeError(f"Search space for '{path}' must be a mapping.")
        value = _suggest_value(trial, path, spec)
        params[path] = value
        omegaconf.OmegaConf.update(cfg, path, value, merge=False)
    _apply_trial_overrides(cfg, hpo_cfg.get("trial_overrides"))

    tracking_uri = hpo_cfg.get("tracking_uri") or cfg.experiment.mlflow.tracking_uri
    trial_dir = os.path.join(
        study_dir,
        str(hpo_cfg.trial_dir_template).format(trial_number=trial.number),
    )
    run_name = str(hpo_cfg.trial_run_name_template).format(trial_number=trial.number)
    existing_tags = omegaconf.OmegaConf.to_container(
        omegaconf.OmegaConf.select(cfg, "experiment.mlflow.tags", default={}),
        resolve=True,
    )
    if not isinstance(existing_tags, dict):
        existing_tags = {}
    existing_tags.update(
        build_trial_tags(
            study_name=trial.study.study_name,
            trial_number=trial.number,
            objective_metric=str(hpo_cfg.objective_metric),
            state="RUNNING",
            pruned=False,
        )
    )
    omegaconf.OmegaConf.update(cfg, "experiment.mlflow.enabled", True, merge=False)
    omegaconf.OmegaConf.update(cfg, "experiment.mlflow.nested_run", True, merge=False)
    omegaconf.OmegaConf.update(cfg, "experiment.mlflow.parent_run_id", parent_run_id, merge=False)
    omegaconf.OmegaConf.update(cfg, "experiment.mlflow.tracking_uri", tracking_uri, merge=False)
    omegaconf.OmegaConf.update(cfg, "experiment.mlflow.run_name", run_name, merge=False)
    omegaconf.OmegaConf.update(cfg, "experiment.mlflow.tags", existing_tags, merge=False)
    return cfg, params, trial_dir


def _write_trial_config(trial_cfg, trial_dir: str) -> str:
    train_cfg = _clone_cfg(trial_cfg)
    if "hpo" in train_cfg:
        del train_cfg["hpo"]
    config_path = os.path.join(trial_dir, "trial_config.yaml")
    omegaconf.OmegaConf.save(train_cfg, config_path, resolve=True)
    return config_path


def _run_trial_subprocess(trial_cfg, trial_dir: str):
    os.makedirs(trial_dir, exist_ok=True)
    _write_trial_config(trial_cfg, trial_dir)
    command = [
        sys.executable,
        _train_script_path(),
        "--config-path",
        trial_dir,
        "--config-name",
        "trial_config",
        f"hydra.run.dir={trial_dir}",
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    completed = subprocess.run(
        command,
        cwd=os.environ.get("REPO_ROOT", os.getcwd()),
        env=env,
        check=False,
    )
    result_path = _trial_result_path(trial_dir)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Training subprocess failed with exit code {completed.returncode}. "
            f"Expected result file at {result_path}."
        )
    if not os.path.isfile(result_path):
        raise FileNotFoundError(
            f"Training subprocess completed without producing {result_path}."
        )
    return read_training_result(result_path)


def run_hpo_study(cfg):
    """Execute an Optuna study from a composed Hydra config."""
    _validate_hpo_cfg(cfg)
    hpo_cfg = cfg.hpo
    tracking_uri = hpo_cfg.get("tracking_uri") or cfg.experiment.mlflow.tracking_uri
    experiment_name = cfg.experiment.mlflow.experiment_name
    study_dir = os.getcwd()

    optuna.logging.set_verbosity(optuna.logging.INFO)

    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    experiment_id = get_or_create_mlflow_experiment_id(
        experiment_name,
        tracking_uri=tracking_uri,
        artifact_location=cfg.experiment.mlflow.artifact_location,
    )
    sampler = _build_sampler(hpo_cfg.sampler)
    pruner = _build_pruner(hpo_cfg.pruner)
    storage_uri = _prepare_storage_uri(str(hpo_cfg.storage))
    study = optuna.create_study(
        study_name=str(hpo_cfg.study_name),
        direction=str(hpo_cfg.direction),
        storage=storage_uri,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    parent_run_name = str(hpo_cfg.parent_run_name or hpo_cfg.study_name)

    parent_tags = {
        "optuna.study_name": str(hpo_cfg.study_name),
        "optuna.direction": str(hpo_cfg.direction),
        "optuna.objective": str(hpo_cfg.objective_metric),
        "optuna.execution_backend": str(hpo_cfg.execution_backend),
    }
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=parent_run_name,
        tags=parent_tags,
    ) as parent_run:
        parent_run_id = parent_run.info.run_id
        if str(hpo_cfg.pruner.name).lower() != "none":
            optuna.logging.get_logger("optuna").warning(
                "NomadFlow Optuna v1 now executes trials in a training subprocess. "
                "Mid-run pruning is currently disabled in this mode."
            )

        def objective(trial: optuna.trial.Trial) -> float:
            trial_cfg, params, trial_dir = _build_trial_cfg(
                cfg,
                trial,
                study_dir,
                parent_run_id=parent_run_id,
            )
            result = _run_trial_subprocess(trial_cfg, trial_dir)
            trial.set_user_attr("mlflow_run_id", result.run_id)
            trial.set_user_attr("training_status", result.status)
            trial.set_user_attr("objective_metric", result.metric_name)
            trial.set_user_attr("training_result_path", _trial_result_path(trial_dir))
            if result.best_model_score is not None:
                trial.set_user_attr("best_model_score", result.best_model_score)
            for key, value in params.items():
                trial.set_user_attr(f"param::{key}", value)
            return result.objective_value

        study.optimize(
            objective,
            n_trials=int(hpo_cfg.n_trials),
            timeout=hpo_cfg.timeout,
        )
        summary = summarize_study(study)
        if "best_value" in summary:
            mlflow.log_metric("best_value", float(summary["best_value"]))
        mlflow.log_metrics(
            {
                "n_trials": float(summary["n_trials"]),
                "n_completed": float(summary["n_completed"]),
                "n_pruned": float(summary["n_pruned"]),
                "n_failed": float(summary["n_failed"]),
            }
        )
        best_params = summary.get("best_params", {})
        if isinstance(best_params, dict) and best_params:
            mlflow.log_params({f"best::{key}": str(value) for key, value in best_params.items()})
        log_json_artifact(mlflow, summary, "optuna_study_summary.json")
        return summary


def main(model,
         corrupter,
         lmodule,
         dataset,
         datamodule,
         experiment,
         hpo,
         zen_cfg):
    del model, corrupter, lmodule, dataset, datamodule, experiment, hpo
    return run_hpo_study(zen_cfg)


if __name__ == "__main__":
    config_hydra_store()
    torch.set_float32_matmul_precision("medium")
    zen(main).hydra_main(
        config_name="hpo_main",
        version_base="1.2",
        config_path=".",
    )
