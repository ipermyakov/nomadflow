"""Lightning logger and callbacks for MLflow experiment tracking."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from argparse import Namespace
from numbers import Number
from typing import Any, Mapping, MutableMapping, Optional, Union

import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from nomadflow.runtime.hpo import extract_training_trial_result, write_training_result

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore


def _to_float_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            v = value.detach().cpu()
            if v.numel() != 1:
                continue
            out[key] = float(v.item())
        elif np is not None and isinstance(value, np.generic):
            x = float(value)
            if math.isfinite(x):
                out[key] = x
        elif isinstance(value, Number) and not isinstance(value, bool):
            x = float(value)
            if math.isfinite(x):
                out[key] = x
        else:
            try:
                x = float(value)  # type: ignore[arg-type]
                if math.isfinite(x):
                    out[key] = x
            except (TypeError, ValueError):
                continue
    return out


def _is_global_rank_zero() -> bool:
    """Best-effort rank detection before Trainer fully initializes DDP."""
    for key in (
        "RANK",
        "LOCAL_RANK",
        "SLURM_PROCID",
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
    ):
        raw = os.environ.get(key)
        if raw is None:
            continue
        try:
            return int(raw) == 0
        except ValueError:
            continue
    return True


def _flatten_config(
    params: Union[MutableMapping[str, Any], Namespace],
    parent: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    """Flatten nested config dictionaries into MLflow-friendly params."""
    if isinstance(params, Namespace):
        params = vars(params)
    flat: dict[str, Any] = {}
    for key, value in params.items():
        name = f"{parent}{sep}{key}" if parent else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_config(value, name, sep=sep))
        elif isinstance(value, Namespace):
            flat.update(_flatten_config(vars(value), name, sep=sep))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flat[name] = value
        elif isinstance(value, Number) and not isinstance(value, bool):
            flat[name] = float(value)
        else:
            flat[name] = str(value)
    return flat


def _truncate_param_value(value: Any, max_len: int = 500) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


_INVALID_MLFLOW_KEY_CHARS = re.compile(r"[^A-Za-z0-9_.\- /:]")
_REPEATED_UNDERSCORES = re.compile(r"_+")


def _sanitize_mlflow_key(key: str) -> str:
    """Normalize names to MLflow's allowed metric/param charset."""
    sanitized = _INVALID_MLFLOW_KEY_CHARS.sub("_", key)
    sanitized = _REPEATED_UNDERSCORES.sub("_", sanitized).strip()
    sanitized = sanitized.strip("._- /:")
    if not sanitized:
        return "param"
    return sanitized


def _maybe_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Number) and not isinstance(value, bool):
        return float(value)
    return value


def _local_file_store_root(tracking_uri: Optional[str]) -> Optional[str]:
    """Return an absolute local FileStore root, or None for remote / missing URIs."""
    if not tracking_uri:
        return None
    root = tracking_uri.strip()
    if root.startswith("file:"):
        # file:///abs/path or file:/abs/path
        from urllib.parse import urlparse

        parsed = urlparse(root)
        if parsed.scheme != "file":
            return None
        root = parsed.path
    root = os.path.abspath(os.path.expanduser(os.path.expandvars(root)))
    if not os.path.isdir(root):
        return None
    return root


def _rewrite_yaml_field(path: str, key: str, value: str) -> bool:
    """Rewrite a top-level ``key: ...`` line in an MLflow meta.yaml. Returns True if changed."""
    if not os.path.isfile(path):
        return False
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    prefix = f"{key}:"
    changed = False
    new_lines: list[str] = []
    for line in lines:
        if line.startswith(prefix):
            new_line = f"{key}: {value}\n"
            if line != new_line:
                changed = True
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    if changed:
        with open(path, "w", encoding="utf-8") as handle:
            handle.writelines(new_lines)
    return changed


def repair_file_store_artifact_paths(
    tracking_uri: Optional[str],
    experiment_id: str,
) -> bool:
    """Point experiment/run artifact paths at the current local FileStore root.

    MLflow FileStore persists absolute ``artifact_location`` / ``artifact_uri`` values.
    Cloning or moving the repo (or reusing ``mlruns/`` from another host) leaves stale
    paths like ``/home/otheruser/.../mlruns/<id>``. Repair them to
    ``{tracking_uri_root}/{experiment_id}[/run_id/artifacts]`` so artifact logging works
    on any machine that shares the same relative ``mlruns`` layout.
    """
    root = _local_file_store_root(tracking_uri)
    if root is None:
        return False
    exp_dir = os.path.join(root, str(experiment_id))
    exp_meta = os.path.join(exp_dir, "meta.yaml")
    if not os.path.isfile(exp_meta):
        return False

    expected_exp = exp_dir
    changed = _rewrite_yaml_field(exp_meta, "artifact_location", expected_exp)

    for entry in os.listdir(exp_dir):
        run_dir = os.path.join(exp_dir, entry)
        run_meta = os.path.join(run_dir, "meta.yaml")
        if not os.path.isfile(run_meta):
            continue
        expected_run = os.path.join(run_dir, "artifacts")
        if _rewrite_yaml_field(run_meta, "artifact_uri", expected_run):
            changed = True
        os.makedirs(expected_run, exist_ok=True)

    return changed


def get_or_create_mlflow_experiment_id(
    experiment_name: str,
    *,
    tracking_uri: Optional[str] = None,
    artifact_location: Optional[str] = None,
) -> str:
    """Return an existing MLflow experiment id or create it.

    For local FileStore tracking URIs, stale absolute artifact paths are rewritten to
    the current store root so the same ``mlruns/`` tree works across hosts/disks.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    existing = client.get_experiment_by_name(experiment_name)
    if existing is not None:
        if getattr(existing, "lifecycle_stage", "active") != "active":
            client.restore_experiment(existing.experiment_id)
        experiment_id = existing.experiment_id
    elif artifact_location:
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=artifact_location,
        )
    else:
        # Let FileStore allocate the id, then normalize artifact_location to this host.
        experiment_id = client.create_experiment(experiment_name)

    repair_file_store_artifact_paths(tracking_uri, experiment_id)
    return experiment_id


class MlflowLogger(Logger):
    """Logs metrics, params, and artifacts to MLflow."""

    def __init__(
        self,
        experiment_name: str = "nomadflow",
        tracking_uri: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Mapping[str, Any]] = None,
        artifact_location: Optional[str] = None,
        nested_run: bool = False,
        parent_run_id: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._is_rank_zero = _is_global_rank_zero()
        self._experiment_name = experiment_name
        self._tracking_uri = tracking_uri
        self._artifact_location = artifact_location
        self._run_name = run_name
        self._tags = dict(tags or {})
        self._nested_run = nested_run
        self._parent_run_id = parent_run_id
        self._mlflow = None
        self._client = None
        self._run = None
        self._run_id: Optional[str] = None
        self._logged_params: dict[str, str] = {}

        if not self._is_rank_zero:
            return

        import mlflow
        from mlflow.tracking import MlflowClient

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self._mlflow = mlflow
        self._client = MlflowClient(tracking_uri=tracking_uri)
        experiment_id = self._get_or_create_experiment_id()
        self._run = mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
            nested=nested_run,
            parent_run_id=parent_run_id,
            tags={k: str(v) for k, v in self._tags.items()},
        )
        self._run_id = self._run.info.run_id
        self._client.log_metric(self._run_id, "run_started", 1.0, step=0)

    def _get_or_create_experiment_id(self) -> str:
        return get_or_create_mlflow_experiment_id(
            self._experiment_name,
            tracking_uri=self._tracking_uri,
            artifact_location=self._artifact_location,
        )

    @property
    def name(self) -> Optional[str]:
        return self._experiment_name

    @property
    def version(self) -> Optional[Union[int, str]]:
        return self._run_id or "0"

    @property
    def experiment(self) -> Any:
        return self._run

    @property
    def run_id(self) -> Optional[str]:
        return self._run_id

    @property
    def tags(self) -> dict[str, str]:
        return {key: str(value) for key, value in self._tags.items()}

    @rank_zero_only
    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        if self._client is None or self._run_id is None:
            return
        payload = _to_float_metrics(metrics)
        if not payload:
            return
        for key, value in payload.items():
            self._client.log_metric(
                self._run_id,
                _sanitize_mlflow_key(key),
                value,
                step=step or 0,
            )

    @rank_zero_only
    def log_hyperparams(
        self,
        params: Union[dict[str, Any], Namespace],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if self._client is None or self._run_id is None or not params:
            return
        flat = _flatten_config(params)
        sanitized_keys: dict[str, str] = {}
        for key, value in flat.items():
            sanitized_key = _sanitize_mlflow_key(key)
            previous_source = sanitized_keys.get(sanitized_key)
            if previous_source is not None and previous_source != key:
                continue
            sanitized_keys[sanitized_key] = key
            normalized = _truncate_param_value(value)
            previous = self._logged_params.get(sanitized_key)
            if previous == normalized:
                continue
            if previous is not None and previous != normalized:
                # MLflow params are immutable within a run. Keep the first value.
                continue
            self._client.log_param(self._run_id, sanitized_key, normalized)
            self._logged_params[sanitized_key] = normalized

    @rank_zero_only
    def set_tags(self, tags: Mapping[str, Any]) -> None:
        if self._client is None or self._run_id is None:
            return
        for key, value in tags.items():
            normalized = str(value)
            self._client.set_tag(self._run_id, key, normalized)
            self._tags[key] = normalized

    @rank_zero_only
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        if self._client is None or self._run_id is None or not os.path.exists(local_path):
            return
        self._client.log_artifact(self._run_id, local_path, artifact_path=artifact_path)

    @rank_zero_only
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        if self._client is None or self._run_id is None or not os.path.isdir(local_dir):
            return
        self._client.log_artifacts(self._run_id, local_dir, artifact_path=artifact_path)

    @rank_zero_only
    def log_dict(
        self,
        payload: Mapping[str, Any],
        artifact_file: str,
        artifact_path: Optional[str] = None,
    ) -> None:
        if self._client is None or self._run_id is None:
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, artifact_file)
            with open(local_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
            self.log_artifact(local_path, artifact_path=artifact_path)

    @rank_zero_only
    def log_run_artifacts(
        self,
        output_dir: str,
        checkpointer: Optional[ModelCheckpoint] = None,
        log_hydra_artifacts: bool = True,
    ) -> None:
        if self._client is None or self._run_id is None:
            return

        hydra_dir = os.path.join(output_dir, ".hydra")
        if log_hydra_artifacts and os.path.isdir(hydra_dir):
            self.log_artifacts(hydra_dir, artifact_path="hydra")

        dataset_config = os.path.join(output_dir, "dataset_config.yaml")
        if log_hydra_artifacts and os.path.isfile(dataset_config):
            self.log_artifact(dataset_config, artifact_path="configs")

        metadata: dict[str, Any] = {
            "output_dir": output_dir,
            "checkpoints": {},
        }

        if checkpointer is not None:
            best_model_path = getattr(checkpointer, "best_model_path", None) or None
            last_model_path = getattr(checkpointer, "last_model_path", None) or None
            best_model_score = _maybe_scalar(getattr(checkpointer, "best_model_score", None))

            metadata["checkpoints"] = {
                "best_model_path": best_model_path,
                "last_model_path": last_model_path,
                "best_model_score": best_model_score,
                "monitor": getattr(checkpointer, "monitor", None),
                "mode": getattr(checkpointer, "mode", None),
                "save_top_k": getattr(checkpointer, "save_top_k", None),
            }

            if best_model_path and os.path.isfile(best_model_path):
                self.log_artifact(best_model_path, artifact_path="checkpoints")
            if (
                last_model_path
                and last_model_path != best_model_path
                and os.path.isfile(last_model_path)
            ):
                self.log_artifact(last_model_path, artifact_path="checkpoints")

        self.log_dict(metadata, artifact_file="run_metadata.json", artifact_path="metadata")

    @rank_zero_only
    def finalize(self, status: str) -> None:
        super().finalize(status)
        if self._mlflow is not None and self._run is not None:
            normalized = {
                "success": "FINISHED",
                "finished": "FINISHED",
                "failed": "FAILED",
                "failure": "FAILED",
                "killed": "KILLED",
                "pruned": "KILLED",
            }.get(status.lower(), "FINISHED")
            self._mlflow.end_run(status=normalized)
            self._run = None


class MlflowArtifactsCallback(Callback):
    """Logs run artifacts and checkpoint metadata before the MLflow run closes."""

    def __init__(
        self,
        log_hydra_artifacts: bool = True,
        log_checkpoint_artifacts: bool = True,
    ) -> None:
        super().__init__()
        self._log_hydra_artifacts = log_hydra_artifacts
        self._log_checkpoint_artifacts = log_checkpoint_artifacts

    @rank_zero_only
    def on_fit_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        del pl_module
        logger = trainer.logger
        if not isinstance(logger, MlflowLogger):
            return
        checkpointer = next(
            (cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)),
            None,
        )
        if not self._log_hydra_artifacts and not self._log_checkpoint_artifacts:
            return
        if not self._log_checkpoint_artifacts:
            checkpointer = None
        logger.log_run_artifacts(
            output_dir=os.getcwd(),
            checkpointer=checkpointer,
            log_hydra_artifacts=self._log_hydra_artifacts,
        )
        if "optuna.study_name" in logger.tags:
            objective_metric = logger.tags.get("optuna.objective", "val/loss")
            result = extract_training_trial_result(
                trainer,
                metric_name=objective_metric,
                status="completed",
            )
            logger.set_tags(
                {
                    "optuna.state": "COMPLETED",
                    "optuna.pruned": "false",
                }
            )
            result_path = write_training_result(os.getcwd(), result)
            logger.log_artifact(result_path, artifact_path="metadata")
