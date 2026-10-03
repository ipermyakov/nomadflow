from types import SimpleNamespace

import mlflow
import optuna
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from mlflow.tracking import MlflowClient

from nomadflow.runtime.hpo import (
    OptunaPruningCallback,
    TrainingTrialResult,
    extract_training_trial_result,
    read_training_result,
    summarize_study,
    write_training_result,
)
from nomadflow.runtime.mlflow_logger import (
    MlflowLogger,
    get_or_create_mlflow_experiment_id,
)
from nomadflow.stoch_interp.multiframe import _resolve_cache_dir


def test_extract_training_trial_result_uses_checkpoint_fallback(tmp_path):
    checkpointer = ModelCheckpoint(dirpath=tmp_path)
    checkpointer.monitor = "val/loss"
    checkpointer.best_model_score = torch.tensor(1.25)
    trainer = SimpleNamespace(
        callback_metrics={},
        callbacks=[checkpointer],
        logger=SimpleNamespace(run_id="run-123"),
    )

    result = extract_training_trial_result(trainer, metric_name="val/loss")

    assert result.objective_value == pytest.approx(1.25)
    assert result.best_model_score == pytest.approx(1.25)
    assert result.run_id == "run-123"
    assert result.status == "completed"


def test_optuna_pruning_callback_reports_and_prunes():
    class StubTrial:
        def __init__(self):
            self.reports = []

        def report(self, value, step):
            self.reports.append((value, step))

        def should_prune(self):
            return True

    trial = StubTrial()
    callback = OptunaPruningCallback(trial, monitor="val/loss")
    trainer = SimpleNamespace(
        callback_metrics={"val/loss": torch.tensor(0.9)},
        current_epoch=2,
        is_global_zero=True,
    )

    with pytest.raises(optuna.TrialPruned):
        callback.on_validation_epoch_end(trainer, None)

    assert trial.reports == [(pytest.approx(0.9), 2)]


def test_mlflow_logger_supports_nested_child_runs(tmp_path):
    tracking_uri = f"file://{tmp_path / 'mlruns'}"
    experiment_name = "optuna-test"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = get_or_create_mlflow_experiment_id(
        experiment_name,
        tracking_uri=tracking_uri,
    )

    with mlflow.start_run(experiment_id=experiment_id, run_name="parent") as parent_run:
        logger = MlflowLogger(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            run_name="child",
            nested_run=True,
            tags={"optuna.study_name": "study-a"},
        )
        child_run_id = logger.run_id
        logger.finalize("success")

    client = MlflowClient(tracking_uri=tracking_uri)
    child_run = client.get_run(child_run_id)

    assert child_run.data.tags["mlflow.parentRunId"] == parent_run.info.run_id
    assert child_run.data.tags["optuna.study_name"] == "study-a"


def test_mlflow_logger_supports_explicit_parent_run_id(tmp_path):
    tracking_uri = f"file://{tmp_path / 'mlruns'}"
    experiment_name = "optuna-test-explicit-parent"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = get_or_create_mlflow_experiment_id(
        experiment_name,
        tracking_uri=tracking_uri,
    )

    with mlflow.start_run(experiment_id=experiment_id, run_name="parent") as parent_run:
        parent_run_id = parent_run.info.run_id

    logger = MlflowLogger(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        run_name="child",
        nested_run=True,
        parent_run_id=parent_run_id,
        tags={"optuna.study_name": "study-b"},
    )
    child_run_id = logger.run_id
    logger.finalize("success")

    client = MlflowClient(tracking_uri=tracking_uri)
    child_run = client.get_run(child_run_id)

    assert child_run.data.tags["mlflow.parentRunId"] == parent_run_id
    assert child_run.data.tags["optuna.study_name"] == "study-b"


def test_get_or_create_mlflow_experiment_id_restores_deleted_experiment(tmp_path):
    tracking_uri = f"file://{tmp_path / 'mlruns'}"
    experiment_name = "optuna-test-restore"
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment_id = client.create_experiment(experiment_name)
    client.delete_experiment(experiment_id)

    restored_id = get_or_create_mlflow_experiment_id(
        experiment_name,
        tracking_uri=tracking_uri,
    )
    experiment = client.get_experiment(restored_id)

    assert restored_id == experiment_id
    assert experiment.lifecycle_stage == "active"


def test_training_result_round_trip(tmp_path):
    result = TrainingTrialResult(
        objective_value=0.42,
        metric_name="val/loss",
        status="completed",
        run_id="run-456",
        best_model_score=0.4,
    )

    result_path = write_training_result(tmp_path, result)
    loaded = read_training_result(result_path)

    assert result_path == str(tmp_path / "metadata" / "training_result.json")
    assert loaded == result


def test_resolve_cache_dir_uses_repo_root(monkeypatch):
    monkeypatch.setenv("REPO_ROOT", "/tmp/nomadflow-repo")

    assert _resolve_cache_dir(None) == "/tmp/nomadflow-repo/.cache"


def test_summarize_study_counts_trial_states():
    study = optuna.create_study(direction="minimize")
    study.add_trial(
        optuna.trial.create_trial(
            params={"x": 1.0},
            distributions={"x": optuna.distributions.FloatDistribution(0.0, 2.0)},
            value=0.5,
            state=optuna.trial.TrialState.COMPLETE,
        )
    )
    study.add_trial(
        optuna.trial.create_trial(
            params={"x": 1.5},
            distributions={"x": optuna.distributions.FloatDistribution(0.0, 2.0)},
            value=None,
            state=optuna.trial.TrialState.PRUNED,
        )
    )

    summary = summarize_study(study)

    assert summary["n_trials"] == 2
    assert summary["n_completed"] == 1
    assert summary["n_pruned"] == 1
    assert summary["best_value"] == pytest.approx(0.5)
