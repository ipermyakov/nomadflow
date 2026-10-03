"""FileStore artifact paths must follow the current tracking URI root."""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from nomadflow.runtime.mlflow_logger import (
    get_or_create_mlflow_experiment_id,
    repair_file_store_artifact_paths,
)


@pytest.fixture(autouse=True)
def _allow_file_store(monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")


def test_repair_rewrites_stale_absolute_artifact_paths(tmp_path):
    tracking_root = tmp_path / "mlruns"
    tracking_uri = tracking_root.resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment_id = client.create_experiment("portable-exp")
    with mlflow.start_run(experiment_id=experiment_id, run_name="r1") as run:
        run_id = run.info.run_id

    # Simulate mlruns copied from another host/user.
    stale_exp = "/home/otheruser/old_repo/mlruns/" + experiment_id
    stale_run = f"{stale_exp}/{run_id}/artifacts"
    exp_meta = tracking_root / experiment_id / "meta.yaml"
    run_meta = tracking_root / experiment_id / run_id / "meta.yaml"

    def _force_field(path: Path, key: str, value: str) -> None:
        lines = []
        for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
            if line.startswith(f"{key}:"):
                lines.append(f"{key}: {value}\n")
            else:
                lines.append(line)
        path.write_text("".join(lines), encoding="utf-8")

    _force_field(exp_meta, "artifact_location", stale_exp)
    _force_field(run_meta, "artifact_uri", stale_run)

    assert "otheruser" in exp_meta.read_text(encoding="utf-8")
    changed = repair_file_store_artifact_paths(tracking_uri, experiment_id)
    assert changed is True

    expected_exp = str((tracking_root / experiment_id).resolve())
    expected_run = str((tracking_root / experiment_id / run_id / "artifacts").resolve())
    exp = client.get_experiment(experiment_id)
    got_run = client.get_run(run_id)
    assert Path(exp.artifact_location).resolve() == Path(expected_exp)
    assert Path(got_run.info.artifact_uri).resolve() == Path(expected_run)
    assert "otheruser" not in exp.artifact_location
    assert os.path.isdir(expected_run)


def test_get_or_create_repairs_existing_experiment(tmp_path):
    tracking_root = tmp_path / "mlruns"
    tracking_uri = tracking_root.resolve().as_uri()
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.create_experiment("auto-repair-exp")

    stale = "/mnt/old-server/mlruns/" + experiment_id
    exp_meta = tracking_root / experiment_id / "meta.yaml"
    text = exp_meta.read_text(encoding="utf-8")
    # Replace whatever absolute location FileStore wrote.
    lines = []
    for line in text.splitlines(keepends=True):
        if line.startswith("artifact_location:"):
            lines.append(f"artifact_location: {stale}\n")
        else:
            lines.append(line)
    exp_meta.write_text("".join(lines), encoding="utf-8")

    got_id = get_or_create_mlflow_experiment_id(
        "auto-repair-exp",
        tracking_uri=tracking_uri,
    )
    assert got_id == experiment_id
    exp = client.get_experiment(experiment_id)
    assert Path(exp.artifact_location).resolve() == (tracking_root / experiment_id).resolve()
    assert "old-server" not in exp.artifact_location
