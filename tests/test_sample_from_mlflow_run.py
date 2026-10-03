import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomadflow.runtime import mlflow_sampling


class FakeMlflowClient:
    def __init__(
        self,
        artifact_root: Path,
        *,
        tags=None,
        experiment_id: str = "exp-1",
        download_raises: bool = False,
    ):
        self.artifact_root = artifact_root
        self.tags = tags or {}
        self.experiment_id = experiment_id
        self.download_raises = download_raises

    def download_artifacts(self, run_id, artifact_path, dst_path):
        del run_id
        if self.download_raises:
            raise FileNotFoundError(f"Simulated remote artifact miss: {artifact_path}")
        source = self.artifact_root / artifact_path
        if not source.is_file():
            raise FileNotFoundError(f"Missing fake artifact: {source}")
        destination = Path(dst_path) / artifact_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return str(destination)

    def get_run(self, run_id):
        del run_id
        return SimpleNamespace(
            info=SimpleNamespace(experiment_id=self.experiment_id),
            data=SimpleNamespace(tags=self.tags),
        )


def _write_fake_run_artifacts(root: Path, *, best_name: str = "best.ckpt", last_name: str = "last.ckpt"):
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "hydra").mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "hydra" / "config.yaml").write_text("model:\n  _target_: test.Model\n", encoding="utf-8")
    (root / "checkpoints" / best_name).write_text("best", encoding="utf-8")
    (root / "checkpoints" / last_name).write_text("last", encoding="utf-8")
    payload = {
        "output_dir": "/tmp/training-run",
        "checkpoints": {
            "best_model_path": f"/tmp/training-run/lightning_logs/version_0/checkpoints/{best_name}",
            "last_model_path": f"/tmp/training-run/lightning_logs/version_0/checkpoints/{last_name}",
        },
    }
    (root / "metadata" / "run_metadata.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_choose_checkpoint_artifact_falls_back_to_last():
    metadata = {
        "checkpoints": {
            "best_model_path": "",
            "last_model_path": "/tmp/outputs/lightning_logs/version_0/checkpoints/last.ckpt",
        }
    }

    artifact, mode = mlflow_sampling.choose_checkpoint_artifact(metadata, preferred="best")

    assert artifact == "checkpoints/last.ckpt"
    assert mode == "last"


def test_resolve_tasks_yaml_supports_benchmark_shortcuts():
    resolved = mlflow_sampling.resolve_tasks_yaml(None, "protpardelle_motif")

    assert resolved.endswith("configs/sampling/protpardelle_motif_scaffolding/config.yaml")
    assert Path(resolved).is_file()


def test_resolve_tasks_yaml_supports_protpardelle_motif_multi():
    resolved = mlflow_sampling.resolve_tasks_yaml(None, "protpardelle_motif_multi")

    assert resolved.endswith(
        "configs/sampling/protpardelle_motif_scaffolding/config_multi_motif_se3.yaml"
    )
    assert Path(resolved).is_file()


def test_protpardelle_m2plus_alias_resolves_to_same_file():
    multi = mlflow_sampling.resolve_tasks_yaml(None, "protpardelle_motif_multi")
    alias = mlflow_sampling.resolve_tasks_yaml(None, "protpardelle_m2plus_unindexed")

    assert multi == alias


def test_stage_run_artifacts_creates_sample_compatible_layout(tmp_path):
    artifact_root = tmp_path / "artifacts"
    _write_fake_run_artifacts(artifact_root)
    client = FakeMlflowClient(artifact_root)

    staged = mlflow_sampling.stage_run_artifacts(
        client,
        "run-123",
        tracking_uri="file:///tmp/mlruns",
        cache_dir=str(tmp_path / "cache"),
        checkpoint="best",
    )

    assert Path(staged.run_dir, ".hydra", "config.yaml").is_file()
    assert Path(staged.run_dir, "checkpoints", "best.ckpt").is_file()
    assert Path(staged.metadata_path).is_file()
    assert staged.selected_checkpoint_mode == "best"
    assert staged.selected_checkpoint_artifact == "checkpoints/best.ckpt"
    assert staged.resolution_source == "mlflow_artifacts"


def test_build_sample_command_includes_expected_overrides(tmp_path):
    staged = mlflow_sampling.StagedMlflowRun(
        run_id="run-1",
        tracking_uri="file:///tmp/mlruns",
        run_dir=str(tmp_path / "stage"),
        checkpoint_path=str(tmp_path / "stage" / "checkpoints" / "best.ckpt"),
        hydra_config_path=str(tmp_path / "stage" / ".hydra" / "config.yaml"),
        metadata_path=str(tmp_path / "stage" / "metadata" / "run_metadata.json"),
        selected_checkpoint_mode="best",
        selected_checkpoint_artifact="checkpoints/best.ckpt",
        resolution_source="mlflow_artifacts",
    )

    command = mlflow_sampling.build_sample_command(
        staged,
        out_dir=str(tmp_path / "outputs"),
        tasks_yaml="/tmp/tasks.yaml",
        batch_size=8,
        save_traj=True,
        center_clean_traj_com=True,
        output_motif_chains=True,
        extra_sample_args=["corrupter.num_timesteps=200"],
    )

    assert Path(command[0]).name.startswith("python")
    assert command[1].endswith("sample.py")
    assert f'model_dir="{staged.run_dir}"' in command
    assert f'checkpoint_path="{staged.checkpoint_path}"' in command
    assert 'sampler.tasks_yaml="/tmp/tasks.yaml"' in command
    assert "sampler.batch_size=8" in command
    assert "save_traj=true" in command
    assert "center_clean_traj_com=true" in command
    assert "output_motif_chains=true" in command
    assert "corrupter.num_timesteps=200" in command


def test_run_sampling_from_mlflow_writes_provenance_and_invokes_sample(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    _write_fake_run_artifacts(artifact_root)
    client = FakeMlflowClient(artifact_root)
    calls = {}

    def fake_run(command, cwd, check):
        calls["command"] = command
        calls["cwd"] = cwd
        calls["check"] = check
        return "ok"

    monkeypatch.setattr(mlflow_sampling.subprocess, "run", fake_run)

    out_dir = tmp_path / "sampling-output"
    result = mlflow_sampling.run_sampling_from_mlflow(
        run_id="run-123",
        tracking_uri="file:///tmp/mlruns",
        out_dir=str(out_dir),
        benchmark="unconditional_128",
        cache_dir=str(tmp_path / "cache"),
        checkpoint="best",
        batch_size=4,
        num_samples_per_task=3,
        save_traj=True,
        center_clean_traj_com=True,
        extra_sample_args=["corrupter.num_timesteps=100"],
        client=client,
    )

    provenance_path = out_dir / "metadata" / "mlflow_sampling_source.json"
    assert result == "ok"
    assert provenance_path.is_file()
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert payload["source_run_id"] == "run-123"
    assert payload["benchmark"] == "unconditional_128"
    assert payload["selected_checkpoint_mode"] == "best"
    assert payload["resolution_source"] == "mlflow_artifacts"
    assert payload["num_samples_per_task"] == 3
    assert payload["ode_sampling"] is False
    assert "sampler.batch_size=4" in calls["command"]
    assert "save_traj=true" in calls["command"]
    assert "center_clean_traj_com=true" in calls["command"]
    assert "corrupter.num_timesteps=100" in calls["command"]
    assert calls["cwd"] == str(Path(mlflow_sampling.__file__).resolve().parents[2])
    assert calls["check"] is True


def test_parse_extra_sample_args_rejects_invalid_syntax():
    with pytest.raises(ValueError):
        mlflow_sampling.parse_extra_sample_args(["corrupter.num_timesteps"])


def test_compose_sample_args_adds_ode_overrides():
    args = mlflow_sampling.compose_sample_args([], ode_sampling=True)

    assert list(mlflow_sampling.ODE_SAMPLE_ARGS) == args


def test_compose_sample_args_keeps_explicit_override_over_ode_default():
    args = mlflow_sampling.compose_sample_args(
        ["corrupter.num_timesteps=50"],
        ode_sampling=True,
    )

    assert "corrupter.num_timesteps=50" in args
    assert "corrupter.num_timesteps=200" not in args
    assert "corrupter.sampling_noise_mode=null" in args
    assert "corrupter.trans_step_size=1" in args
    assert "corrupter.rot_step_size=10" in args


def test_normalize_tasks_yaml_rewrites_old_repo_absolute_paths(tmp_path):
    tasks_yaml = tmp_path / "tasks.yaml"
    tasks_yaml.write_text(
        "\n".join(
            [
                "tasks:",
                "- task: motif_scaffolding",
                "  pdb: /mnt/old-server/configs/sampling/protpardelle_motif_scaffolding/pdbs/1PRW.pdb",
                '  pdb_contigs: "A16-35,A52-71"',
                "  num_samples: 1",
                '  total_length: "60-150"',
            ]
        ),
        encoding="utf-8",
    )

    normalized = mlflow_sampling.normalize_tasks_yaml(
        str(tasks_yaml), cache_dir=str(tmp_path / "cache"), run_id="run-1"
    )

    assert normalized != str(tasks_yaml)
    payload = Path(normalized).read_text(encoding="utf-8")
    assert "/mnt/old-server" not in payload
    assert str(Path(mlflow_sampling.__file__).resolve().parents[2] / "configs" / "sampling" / "protpardelle_motif_scaffolding" / "pdbs" / "1PRW.pdb") in payload


def test_normalize_tasks_yaml_overrides_num_samples_for_all_tasks(tmp_path):
    tasks_yaml = tmp_path / "tasks.yaml"
    tasks_yaml.write_text(
        "\n".join(
            [
                "tasks:",
                "- task: unconditional",
                "  sample_length: 60",
                "  num_samples: 10",
                "- task: motif_scaffolding",
                "  pdb: /mnt/old-server/configs/sampling/protpardelle_motif_scaffolding/pdbs/1PRW.pdb",
                '  pdb_contigs: "A16-35,A52-71"',
                "  num_samples: 200",
                '  total_length: "60-150"',
            ]
        ),
        encoding="utf-8",
    )

    normalized = mlflow_sampling.normalize_tasks_yaml(
        str(tasks_yaml),
        cache_dir=str(tmp_path / "cache"),
        run_id="run-1",
        num_samples_per_task=7,
    )

    payload = Path(normalized).read_text(encoding="utf-8")
    assert "num_samples: 7" in payload
    assert "num_samples: 10" not in payload
    assert "num_samples: 200" not in payload


def test_stage_run_artifacts_falls_back_to_local_hydra_output_dir(tmp_path):
    train_run_dir = tmp_path / "outputs" / "phase1_0"
    checkpoint_dir = train_run_dir / "lightning_logs" / "version_0" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (train_run_dir / ".hydra").mkdir(parents=True, exist_ok=True)
    (train_run_dir / ".hydra" / "config.yaml").write_text("model:\n  _target_: test.Model\n", encoding="utf-8")
    (checkpoint_dir / "epoch=1-step=10.ckpt").write_text("best-ish", encoding="utf-8")
    (checkpoint_dir / "last.ckpt").write_text("last", encoding="utf-8")

    client = FakeMlflowClient(
        tmp_path / "empty-artifacts",
        tags={"hydra_output_dir": str(train_run_dir)},
    )

    staged = mlflow_sampling.stage_run_artifacts(
        client,
        "run-local",
        tracking_uri="file:///tmp/mlruns",
        cache_dir=str(tmp_path / "cache"),
        checkpoint="best",
    )

    assert Path(staged.run_dir, ".hydra", "config.yaml").is_file()
    assert Path(staged.run_dir, "checkpoints", "epoch=1-step=10.ckpt").is_file()
    assert staged.selected_checkpoint_mode == "best"
    assert staged.resolution_source == "hydra_output_dir"
    assert staged.selected_checkpoint_artifact.startswith("local://")


def test_stage_run_artifacts_finds_nested_run_id_checkpoint_layout(tmp_path):
    train_run_dir = tmp_path / "outputs" / "phase1_0"
    nested_checkpoint_dir = (
        train_run_dir / "proteinzen_phase1" / "run-local" / "checkpoints"
    )
    nested_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (train_run_dir / ".hydra").mkdir(parents=True, exist_ok=True)
    (train_run_dir / ".hydra" / "config.yaml").write_text(
        "model:\n  _target_: test.Model\n", encoding="utf-8"
    )
    (nested_checkpoint_dir / "epoch=111-step=14000.ckpt").write_text("best-ish", encoding="utf-8")
    (nested_checkpoint_dir / "last.ckpt").write_text("last", encoding="utf-8")

    client = FakeMlflowClient(
        tmp_path / "empty-artifacts",
        tags={"hydra_output_dir": str(train_run_dir)},
    )

    staged = mlflow_sampling.stage_run_artifacts(
        client,
        "run-local",
        tracking_uri="file:///tmp/mlruns",
        cache_dir=str(tmp_path / "cache"),
        checkpoint="best",
    )

    assert Path(staged.checkpoint_path).name == "epoch=111-step=14000.ckpt"
    assert staged.selected_checkpoint_mode == "best"


def test_choose_local_checkpoint_uses_numeric_epoch_step_order(tmp_path):
    checkpoint_dir = tmp_path / "lightning_logs" / "version_0" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "epoch=79-step=10000.ckpt").write_text("79", encoding="utf-8")
    (checkpoint_dir / "epoch=95-step=12000.ckpt").write_text("95", encoding="utf-8")
    (checkpoint_dir / "epoch=111-step=14000.ckpt").write_text("111", encoding="utf-8")
    (checkpoint_dir / "last.ckpt").write_text("last", encoding="utf-8")

    checkpoint_path, mode = mlflow_sampling._choose_local_checkpoint(
        str(tmp_path), "unused-run-id", "best"
    )

    assert Path(checkpoint_path).name == "epoch=111-step=14000.ckpt"
    assert mode == "best"


def test_stage_run_artifacts_remaps_remote_hydra_output_dir_and_file_store(tmp_path, monkeypatch):
    monkeypatch.setattr(mlflow_sampling, "_repo_root", lambda: tmp_path)

    run_id = "run-remote"
    experiment_id = "exp-remote"
    tracking_uri = str(tmp_path / "mlruns")
    file_store_artifacts = (
        tmp_path / "mlruns" / experiment_id / run_id / "artifacts"
    )
    (file_store_artifacts / "hydra").mkdir(parents=True, exist_ok=True)
    (file_store_artifacts / "metadata").mkdir(parents=True, exist_ok=True)
    (file_store_artifacts / "hydra" / "config.yaml").write_text(
        "model:\n  _target_: test.Model\n", encoding="utf-8"
    )
    metadata_payload = {
        "output_dir": "/home/other/zenprotein/outputs/finetune/spatial_agglock_05",
        "checkpoints": {
            "best_model_path": (
                "/home/other/zenprotein/outputs/finetune/spatial_agglock_05/"
                f"proteinzen_phase1/{run_id}/checkpoints/epoch=815-step=102000.ckpt"
            ),
            "last_model_path": (
                "/home/other/zenprotein/outputs/finetune/spatial_agglock_05/"
                f"proteinzen_phase1/{run_id}/checkpoints/last.ckpt"
            ),
        },
    }
    (file_store_artifacts / "metadata" / "run_metadata.json").write_text(
        json.dumps(metadata_payload),
        encoding="utf-8",
    )

    train_run_dir = tmp_path / "outputs" / "finetune" / "spatial_agglock_05"
    nested_checkpoint_dir = train_run_dir / "proteinzen_phase1" / run_id / "checkpoints"
    nested_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (train_run_dir / ".hydra").mkdir(parents=True, exist_ok=True)
    (train_run_dir / ".hydra" / "config.yaml").write_text(
        "model:\n  _target_: test.Model\n", encoding="utf-8"
    )
    (nested_checkpoint_dir / "epoch=815-step=102000.ckpt").write_text("best-ish", encoding="utf-8")
    (nested_checkpoint_dir / "last.ckpt").write_text("last", encoding="utf-8")

    remote_hydra_dir = "/home/other/zenprotein/outputs/finetune/spatial_agglock_05"
    client = FakeMlflowClient(
        file_store_artifacts,
        tags={"hydra_output_dir": remote_hydra_dir},
        experiment_id=experiment_id,
        download_raises=True,
    )

    staged = mlflow_sampling.stage_run_artifacts(
        client,
        run_id,
        tracking_uri=tracking_uri,
        cache_dir=str(tmp_path / "cache"),
        checkpoint="best",
    )

    assert Path(staged.run_dir, ".hydra", "config.yaml").is_file()
    assert Path(staged.checkpoint_path).name == "epoch=815-step=102000.ckpt"
    assert staged.resolution_source == "hydra_output_dir"
    assert staged.selected_checkpoint_artifact.startswith("local://")


def test_run_sampling_from_mlflow_rejects_nonpositive_num_samples_per_task(tmp_path):
    with pytest.raises(ValueError):
        mlflow_sampling.run_sampling_from_mlflow(
            run_id="run-123",
            tracking_uri="file:///tmp/mlruns",
            out_dir=str(tmp_path / "sampling-output"),
            benchmark="unconditional_128",
            num_samples_per_task=0,
            client=FakeMlflowClient(tmp_path / "artifacts"),
        )


def test_run_sampling_from_mlflow_records_ode_sampling_in_provenance(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    _write_fake_run_artifacts(artifact_root)
    client = FakeMlflowClient(artifact_root)

    def fake_run(command, cwd, check):
        del cwd, check
        return command

    monkeypatch.setattr(mlflow_sampling.subprocess, "run", fake_run)

    out_dir = tmp_path / "sampling-output"
    command = mlflow_sampling.run_sampling_from_mlflow(
        run_id="run-123",
        tracking_uri="file:///tmp/mlruns",
        out_dir=str(out_dir),
        benchmark="unconditional_128",
        ode_sampling=True,
        client=client,
    )

    payload = json.loads(
        (out_dir / "metadata" / "mlflow_sampling_source.json").read_text(encoding="utf-8")
    )
    assert payload["ode_sampling"] is True
    for arg in mlflow_sampling.ODE_SAMPLE_ARGS:
        assert arg in command
