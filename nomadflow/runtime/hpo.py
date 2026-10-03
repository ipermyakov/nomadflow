"""Optuna helpers for training orchestration and MLflow metadata."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from numbers import Number
from typing import Any, Mapping, Optional

import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

import optuna


def _metric_to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().item())
    if isinstance(value, Number) and not isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class TrainingTrialResult:
    """Structured outcome of a single training run."""

    objective_value: float
    metric_name: str
    status: str
    run_id: Optional[str] = None
    best_model_score: Optional[float] = None


class OptunaPruningCallback(Callback):
    """Reports validation metrics to Optuna and prunes weak trials."""

    def __init__(self, trial: optuna.trial.Trial, monitor: str = "val/loss") -> None:
        super().__init__()
        self._trial = trial
        self._monitor = monitor
        self._last_reported_step = -1

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        del pl_module
        if not getattr(trainer, "is_global_zero", True):
            return
        metric = _metric_to_float(trainer.callback_metrics.get(self._monitor))
        if metric is None:
            return
        step = int(getattr(trainer, "current_epoch", 0))
        if step <= self._last_reported_step:
            return
        self._trial.report(metric, step=step)
        self._last_reported_step = step
        if self._trial.should_prune():
            raise optuna.TrialPruned(
                f"Trial pruned on {self._monitor}={metric:.6f} at epoch {step}."
            )


def extract_training_trial_result(
    trainer,
    metric_name: str = "val/loss",
    status: str = "completed",
) -> TrainingTrialResult:
    """Build a structured training result from a finished Trainer."""
    objective_value = _metric_to_float(trainer.callback_metrics.get(metric_name))
    checkpointer = next(
        (cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)),
        None,
    )
    best_model_score = None
    if checkpointer is not None:
        best_model_score = _metric_to_float(getattr(checkpointer, "best_model_score", None))
        if objective_value is None and getattr(checkpointer, "monitor", None) == metric_name:
            objective_value = best_model_score
    if objective_value is None:
        raise RuntimeError(
            f"Objective metric '{metric_name}' was not produced by the training run."
        )
    return TrainingTrialResult(
        objective_value=objective_value,
        metric_name=metric_name,
        status=status,
        run_id=getattr(trainer.logger, "run_id", None),
        best_model_score=best_model_score,
    )


def build_trial_tags(
    *,
    study_name: str,
    trial_number: int,
    objective_metric: str,
    state: str = "RUNNING",
    pruned: bool = False,
) -> dict[str, str]:
    """Canonical MLflow tags for Optuna child runs."""
    return {
        "optuna.study_name": study_name,
        "optuna.trial_number": str(trial_number),
        "optuna.objective": objective_metric,
        "optuna.state": state,
        "optuna.pruned": str(pruned).lower(),
    }


def summarize_study(study: optuna.study.Study) -> dict[str, Any]:
    """Return a JSON-serializable study summary."""
    state_counts: dict[str, int] = {}
    for trial in study.trials:
        key = trial.state.name.lower()
        state_counts[key] = state_counts.get(key, 0) + 1
    summary: dict[str, Any] = {
        "study_name": study.study_name,
        "direction": study.direction.name.lower(),
        "n_trials": len(study.trials),
        "n_completed": state_counts.get("complete", 0),
        "n_pruned": state_counts.get("pruned", 0),
        "n_failed": state_counts.get("fail", 0),
    }
    if study.best_trial is not None:
        summary["best_trial_number"] = study.best_trial.number
        summary["best_value"] = float(study.best_value)
        summary["best_params"] = dict(study.best_params)
    return summary


def log_json_artifact(mlflow_module, payload: Mapping[str, Any], artifact_file: str) -> None:
    """Log a dictionary as a JSON artifact in the active MLflow run."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, artifact_file)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        mlflow_module.log_artifact(path)


def training_result_to_dict(result: TrainingTrialResult) -> dict[str, Any]:
    """Convert a training result dataclass to a JSON-serializable payload."""
    return asdict(result)


def training_result_from_dict(payload: Mapping[str, Any]) -> TrainingTrialResult:
    """Parse a serialized training result payload back into a dataclass."""
    return TrainingTrialResult(
        objective_value=float(payload["objective_value"]),
        metric_name=str(payload["metric_name"]),
        status=str(payload["status"]),
        run_id=str(payload["run_id"]) if payload.get("run_id") is not None else None,
        best_model_score=(
            float(payload["best_model_score"])
            if payload.get("best_model_score") is not None
            else None
        ),
    )


def write_training_result(output_dir: str, result: TrainingTrialResult) -> str:
    """Persist a structured training result under ``metadata/training_result.json``."""
    metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    result_path = os.path.join(metadata_dir, "training_result.json")
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(training_result_to_dict(result), handle, indent=2, sort_keys=True)
    return result_path


def read_training_result(path: str) -> TrainingTrialResult:
    """Load a serialized training result from disk."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in training result file: {path}")
    return training_result_from_dict(payload)
