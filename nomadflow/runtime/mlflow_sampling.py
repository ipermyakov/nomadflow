from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


_MULTI_MOTIF_TASKS = (
    "configs/sampling/protpardelle_motif_scaffolding/config_multi_motif_se3.yaml"
)

BENCHMARK_TASKS = {
    "unconditional_128": "configs/sampling/unconditional_sampling/benchmark_128.yaml",
    "unconditional_300": "configs/sampling/unconditional_sampling/benchmark_300.yaml",
    "protpardelle_motif": "configs/sampling/protpardelle_motif_scaffolding/config.yaml",
    "protpardelle_motif_indexed": "configs/sampling/protpardelle_motif_scaffolding/config_indexed.yaml",
    # WIP: motif_groups derived from config_indexed.yaml; partition not yet audited.
    "protpardelle_motif_multi": _MULTI_MOTIF_TASKS,
    "protpardelle_m2plus_unindexed": _MULTI_MOTIF_TASKS,
}

ODE_SAMPLE_ARGS = (
    "corrupter.sampling_noise_mode=null",
    "corrupter.trans_step_size=1",
    "corrupter.rot_step_size=10",
    "corrupter.num_timesteps=200",
)


@dataclass(frozen=True)
class StagedMlflowRun:
    run_id: str
    tracking_uri: str
    run_dir: str
    checkpoint_path: str
    hydra_config_path: str
    metadata_path: str
    selected_checkpoint_mode: str
    selected_checkpoint_artifact: str
    resolution_source: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_cache_dir(cache_dir: str | None) -> str:
    if cache_dir:
        expanded = os.path.expanduser(os.path.expandvars(cache_dir))
        return os.path.abspath(expanded)
    repo_root = os.environ.get("REPO_ROOT")
    if repo_root:
        return os.path.join(repo_root, ".cache")
    return str(_repo_root() / ".cache")


def resolve_tracking_uri(tracking_uri: str | None) -> str:
    if tracking_uri:
        return tracking_uri
    env_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_tracking_uri:
        return env_tracking_uri
    repo_root = os.environ.get("REPO_ROOT")
    if repo_root:
        return os.path.abspath(os.path.join(repo_root, "mlruns"))
    return str(_repo_root() / "mlruns")


def resolve_tasks_yaml(tasks_yaml: str | None, benchmark: str | None) -> str:
    if tasks_yaml:
        resolved = os.path.abspath(os.path.expanduser(os.path.expandvars(tasks_yaml)))
    elif benchmark:
        rel_path = BENCHMARK_TASKS.get(benchmark)
        if rel_path is None:
            allowed = ", ".join(sorted(BENCHMARK_TASKS))
            raise KeyError(f"Unknown benchmark {benchmark!r}. Expected one of: {allowed}")
        resolved = str((_repo_root() / rel_path).resolve())
    else:
        raise ValueError("Either tasks_yaml or benchmark must be provided.")
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Tasks YAML does not exist: {resolved}")
    return resolved


def _normalize_task_path_value(value: str, *, tasks_yaml_dir: Path) -> str:
    expanded = os.path.expanduser(os.path.expandvars(value))
    if os.path.exists(expanded):
        return os.path.abspath(expanded)

    repo_root = _repo_root()
    candidates: list[Path] = []
    path_obj = Path(expanded)
    if not path_obj.is_absolute():
        candidates.append((tasks_yaml_dir / path_obj).resolve())
        candidates.append((repo_root / path_obj).resolve())

    if "/configs/sampling/" in expanded:
        suffix = expanded.split("/configs/sampling/", 1)[1]
        candidates.append((repo_root / "configs" / "sampling" / suffix).resolve())
    if "/configs/" in expanded:
        suffix = expanded.split("/configs/", 1)[1]
        candidates.append((repo_root / "configs" / suffix).resolve())

    if path_obj.name:
        candidates.append((tasks_yaml_dir / path_obj.name).resolve())
        candidates.append((tasks_yaml_dir / "pdbs" / path_obj.name).resolve())

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return value


def normalize_tasks_yaml(
    tasks_yaml: str,
    *,
    cache_dir: str,
    run_id: str,
    num_samples_per_task: int | None = None,
) -> str:
    with open(tasks_yaml, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        return tasks_yaml

    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        return tasks_yaml

    tasks_yaml_dir = Path(tasks_yaml).resolve().parent
    changed = False
    for task in tasks:
        if not isinstance(task, dict):
            continue
        if num_samples_per_task is not None and task.get("num_samples") != int(num_samples_per_task):
            task["num_samples"] = int(num_samples_per_task)
            changed = True
        for key, value in list(task.items()):
            if not isinstance(value, str):
                continue
            if "/" not in value and "\\" not in value:
                continue
            normalized = _normalize_task_path_value(value, tasks_yaml_dir=tasks_yaml_dir)
            if normalized != value:
                task[key] = normalized
                changed = True

    if not changed:
        return tasks_yaml

    normalized_dir = Path(cache_dir) / "mlflow_sampling" / "tasks" / run_id
    normalized_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = normalized_dir / Path(tasks_yaml).name
    with open(normalized_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return str(normalized_path)


def parse_extra_sample_args(values: Sequence[str]) -> list[str]:
    parsed: list[str] = []
    for value in values:
        text = value.strip()
        if not text or "=" not in text:
            raise ValueError(
                f"Extra sample overrides must use KEY=VALUE syntax (got {value!r})."
            )
        parsed.append(text)
    return parsed


def compose_sample_args(
    extra_sample_args: Sequence[str] | None = None,
    *,
    ode_sampling: bool = False,
) -> list[str]:
    parsed = parse_extra_sample_args(extra_sample_args or [])
    if ode_sampling:
        existing_keys = {arg.split("=", 1)[0] for arg in parsed}
        for arg in ODE_SAMPLE_ARGS:
            key = arg.split("=", 1)[0]
            if key not in existing_keys:
                parsed.append(arg)
    return parsed


def read_run_metadata(client: Any, run_id: str, download_dir: str) -> tuple[dict[str, Any], str]:
    metadata_path = client.download_artifacts(run_id, "metadata/run_metadata.json", download_dir)
    with open(metadata_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected metadata/run_metadata.json to contain an object for run {run_id}.")
    return payload, metadata_path


def try_read_run_metadata(
    client: Any, run_id: str, download_dir: str
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return read_run_metadata(client, run_id, download_dir)
    except Exception:
        return None, None


def _try_read_local_file_store_metadata(
    client: Any,
    run_id: str,
    download_dir: str,
    *,
    tracking_uri: str,
) -> tuple[dict[str, Any] | None, str | None]:
    local_source = _resolve_local_file_store_artifact(
        client, run_id, "metadata/run_metadata.json", tracking_uri
    )
    if local_source is None:
        return None, None
    destination = os.path.join(download_dir, "metadata", "run_metadata.json")
    staged_path = _copy_to_stage(local_source, destination)
    with open(staged_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected metadata/run_metadata.json to contain an object for run {run_id}."
        )
    return payload, staged_path


def choose_checkpoint_artifact(
    metadata: Mapping[str, Any],
    preferred: str = "best",
) -> tuple[str, str]:
    checkpoints = metadata.get("checkpoints")
    if not isinstance(checkpoints, Mapping):
        raise KeyError("Run metadata is missing the `checkpoints` mapping.")
    candidates = [preferred]
    if preferred == "best":
        candidates.append("last")
    elif preferred == "last":
        candidates.append("best")
    else:
        raise ValueError(f"Unsupported checkpoint mode: {preferred!r}")

    for mode in candidates:
        key = f"{mode}_model_path"
        value = checkpoints.get(key)
        if isinstance(value, str) and value.strip():
            artifact_name = os.path.basename(value)
            if not artifact_name:
                continue
            return f"checkpoints/{artifact_name}", mode
    raise FileNotFoundError(
        "Could not determine a checkpoint artifact from run metadata; "
        f"looked for {[f'{mode}_model_path' for mode in candidates]}."
    )


def _copy_to_stage(source: str, destination: str) -> str:
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _write_json_file(payload: Mapping[str, Any], destination: str) -> str:
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
    return destination


def _tracking_uri_root(tracking_uri: str) -> str:
    root = tracking_uri.strip()
    if root.startswith("file://"):
        root = root[7:]
    return os.path.abspath(os.path.expanduser(os.path.expandvars(root)))


def _candidate_repo_paths(path: str) -> list[str]:
    expanded = os.path.abspath(os.path.expanduser(os.path.expandvars(path.strip())))
    candidates = [expanded]
    repo_root = _repo_root()
    normalized = path.replace("\\", "/")
    for marker in ("outputs/", "configs/"):
        if marker in normalized:
            suffix = normalized.split(marker, 1)[1]
            candidates.append(str((repo_root / marker.rstrip("/") / suffix).resolve()))
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _is_local_tracking_uri(tracking_uri: str) -> bool:
    return os.path.isdir(_tracking_uri_root(tracking_uri))


def _resolve_local_file_store_artifact(
    client: Any,
    run_id: str,
    artifact_path: str,
    tracking_uri: str,
) -> str | None:
    run = client.get_run(run_id)
    experiment_id = run.info.experiment_id
    candidate = os.path.join(
        _tracking_uri_root(tracking_uri),
        experiment_id,
        run_id,
        "artifacts",
        artifact_path,
    )
    return candidate if os.path.isfile(candidate) else None


def _safe_download_artifact(
    client: Any,
    run_id: str,
    artifact_path: str,
    download_dir: str,
    *,
    tracking_uri: str | None = None,
) -> str | None:
    if tracking_uri is not None and _is_local_tracking_uri(tracking_uri):
        local_source = _resolve_local_file_store_artifact(
            client, run_id, artifact_path, tracking_uri
        )
        if local_source is not None:
            destination = os.path.join(download_dir, artifact_path)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copy2(local_source, destination)
            return destination

    try:
        return client.download_artifacts(run_id, artifact_path, download_dir)
    except Exception:
        pass

    if tracking_uri is None or _is_local_tracking_uri(tracking_uri):
        return None

    local_source = _resolve_local_file_store_artifact(
        client, run_id, artifact_path, tracking_uri
    )
    if local_source is None:
        return None

    destination = os.path.join(download_dir, artifact_path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copy2(local_source, destination)
    return destination


def _get_run_tags(client: Any, run_id: str) -> Mapping[str, str]:
    run = client.get_run(run_id)
    tags = getattr(getattr(run, "data", None), "tags", None)
    return tags if isinstance(tags, Mapping) else {}


def _resolve_hydra_output_dir(
    client: Any,
    run_id: str,
    metadata: Mapping[str, Any] | None = None,
) -> str | None:
    tags = _get_run_tags(client, run_id)
    path_candidates: list[str] = []
    for key in ("hydra_output_dir",):
        value = tags.get(key)
        if isinstance(value, str) and value.strip():
            path_candidates.extend(_candidate_repo_paths(value))
    if metadata is not None:
        output_dir = metadata.get("output_dir")
        if isinstance(output_dir, str) and output_dir.strip():
            path_candidates.extend(_candidate_repo_paths(output_dir))

    seen: set[str] = set()
    for candidate in path_candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isdir(candidate):
            return candidate
    return None


def _discover_local_checkpoint_files(run_dir: str, run_id: str) -> list[Path]:
    root = Path(run_dir)
    primary_dir = root / "lightning_logs" / "version_0" / "checkpoints"
    checkpoint_files = sorted(primary_dir.glob("*.ckpt"))
    if checkpoint_files:
        return checkpoint_files

    run_specific_pattern = f"**/{run_id}/checkpoints/*.ckpt"
    checkpoint_files = sorted(root.glob(run_specific_pattern))
    if checkpoint_files:
        return checkpoint_files

    fallback_pattern = "**/checkpoints/*.ckpt"
    checkpoint_files = sorted(root.glob(fallback_pattern))
    if checkpoint_files:
        return checkpoint_files

    raise FileNotFoundError(
        "No checkpoint files found under the expected local fallback locations. "
        f"Tried {primary_dir}, then `{run_specific_pattern}`, then `{fallback_pattern}` inside {run_dir}."
    )


_EPOCH_RE = re.compile(r"epoch=(\d+)")
_STEP_RE = re.compile(r"step=(\d+)")


def _checkpoint_sort_key(path: Path) -> tuple[int, int, str]:
    epoch_match = _EPOCH_RE.search(path.name)
    step_match = _STEP_RE.search(path.name)
    epoch = int(epoch_match.group(1)) if epoch_match else -1
    step = int(step_match.group(1)) if step_match else -1
    return epoch, step, path.name


def _hydra_string_override(key: str, value: str) -> str:
    # Hydra treats JSON-style quoted strings as atomic values, which avoids parse
    # errors for checkpoint names that contain "=" (for example epoch=111-step=14000.ckpt).
    return f"{key}={json.dumps(value)}"


def _resolve_local_checkpoint_from_metadata(
    metadata: Mapping[str, Any] | None,
    preferred: str,
) -> tuple[str, str] | None:
    if metadata is None:
        return None
    checkpoints = metadata.get("checkpoints")
    if not isinstance(checkpoints, Mapping):
        return None

    modes = [preferred]
    if preferred == "best":
        modes.append("last")
    elif preferred == "last":
        modes.append("best")
    else:
        raise ValueError(f"Unsupported checkpoint mode: {preferred!r}")

    for mode in modes:
        value = checkpoints.get(f"{mode}_model_path")
        if not isinstance(value, str) or not value.strip():
            continue
        for candidate in _candidate_repo_paths(value):
            if os.path.isfile(candidate):
                return candidate, mode
    return None


def _choose_local_checkpoint(run_dir: str, run_id: str, preferred: str) -> tuple[str, str]:
    checkpoint_files = _discover_local_checkpoint_files(run_dir, run_id)

    last_ckpt = next((path for path in checkpoint_files if path.name == "last.ckpt"), None)
    non_last = sorted(
        (path for path in checkpoint_files if path.name != "last.ckpt"),
        key=_checkpoint_sort_key,
    )

    if preferred == "last":
        if last_ckpt is not None:
            return str(last_ckpt), "last"
        if non_last:
            return str(non_last[-1]), "best"
    elif preferred == "best":
        if non_last:
            return str(non_last[-1]), "best"
        if last_ckpt is not None:
            return str(last_ckpt), "last"
    else:
        raise ValueError(f"Unsupported checkpoint mode: {preferred!r}")

    raise FileNotFoundError(
        f"Could not resolve a {preferred!r} checkpoint from local run directory {run_dir}."
    )


def _stage_from_local_run_dir(
    *,
    run_id: str,
    tracking_uri: str,
    stage_dir: str,
    hydra_output_dir: str,
    preferred_checkpoint: str,
    metadata: Mapping[str, Any] | None,
) -> StagedMlflowRun:
    hydra_source = os.path.join(hydra_output_dir, ".hydra", "config.yaml")
    if not os.path.isfile(hydra_source):
        raise FileNotFoundError(
            f"Missing Hydra config at {hydra_source}. "
            "This MLflow run has no downloadable hydra artifact and no local .hydra/config.yaml."
        )

    local_checkpoint_path = None
    selected_mode = preferred_checkpoint
    if metadata is not None:
        resolved = _resolve_local_checkpoint_from_metadata(metadata, preferred_checkpoint)
        if resolved is not None:
            local_checkpoint_path, selected_mode = resolved
    if local_checkpoint_path is None:
        local_checkpoint_path, selected_mode = _choose_local_checkpoint(
            hydra_output_dir, run_id, preferred_checkpoint
        )
    checkpoint_name = os.path.basename(local_checkpoint_path)
    hydra_stage = _copy_to_stage(hydra_source, os.path.join(stage_dir, ".hydra", "config.yaml"))
    checkpoint_stage = _copy_to_stage(
        local_checkpoint_path,
        os.path.join(stage_dir, "checkpoints", checkpoint_name),
    )
    metadata_payload = dict(metadata or {})
    metadata_payload.setdefault("source_hydra_output_dir", hydra_output_dir)
    metadata_stage = _write_json_file(
        metadata_payload,
        os.path.join(stage_dir, "metadata", "run_metadata.json"),
    )
    return StagedMlflowRun(
        run_id=run_id,
        tracking_uri=tracking_uri,
        run_dir=stage_dir,
        checkpoint_path=checkpoint_stage,
        hydra_config_path=hydra_stage,
        metadata_path=metadata_stage,
        selected_checkpoint_mode=selected_mode,
        selected_checkpoint_artifact=f"local://{local_checkpoint_path}",
        resolution_source="hydra_output_dir",
    )


def stage_run_artifacts(
    client: Any,
    run_id: str,
    *,
    tracking_uri: str,
    cache_dir: str | None = None,
    checkpoint: str = "best",
) -> StagedMlflowRun:
    root_cache_dir = _resolve_cache_dir(cache_dir)
    download_dir = os.path.join(root_cache_dir, "mlflow_sampling", "downloads", run_id)
    stage_dir = os.path.join(root_cache_dir, "mlflow_sampling", "runs", run_id)
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(stage_dir, exist_ok=True)

    metadata, metadata_download = None, None
    if _is_local_tracking_uri(tracking_uri):
        metadata, metadata_download = _try_read_local_file_store_metadata(
            client, run_id, download_dir, tracking_uri=tracking_uri
        )
    if metadata is None:
        metadata, metadata_download = try_read_run_metadata(client, run_id, download_dir)
    if metadata is None:
        metadata, metadata_download = _try_read_local_file_store_metadata(
            client, run_id, download_dir, tracking_uri=tracking_uri
        )
    hydra_download = _safe_download_artifact(
        client,
        run_id,
        "hydra/config.yaml",
        download_dir,
        tracking_uri=tracking_uri,
    )

    checkpoint_artifact = None
    selected_mode = checkpoint
    checkpoint_download = None
    if metadata is not None:
        checkpoint_artifact, selected_mode = choose_checkpoint_artifact(
            metadata, preferred=checkpoint
        )
        checkpoint_download = _safe_download_artifact(
            client,
            run_id,
            checkpoint_artifact,
            download_dir,
            tracking_uri=tracking_uri,
        )

    if hydra_download is not None and checkpoint_download is not None:
        hydra_stage = _copy_to_stage(hydra_download, os.path.join(stage_dir, ".hydra", "config.yaml"))
        checkpoint_stage = _copy_to_stage(
            checkpoint_download,
            os.path.join(stage_dir, "checkpoints", os.path.basename(checkpoint_artifact)),
        )
        if metadata_download is not None:
            metadata_stage = _copy_to_stage(
                metadata_download,
                os.path.join(stage_dir, "metadata", "run_metadata.json"),
            )
        else:
            metadata_stage = _write_json_file(
                metadata or {},
                os.path.join(stage_dir, "metadata", "run_metadata.json"),
            )

        return StagedMlflowRun(
            run_id=run_id,
            tracking_uri=tracking_uri,
            run_dir=stage_dir,
            checkpoint_path=checkpoint_stage,
            hydra_config_path=hydra_stage,
            metadata_path=metadata_stage,
            selected_checkpoint_mode=selected_mode,
            selected_checkpoint_artifact=checkpoint_artifact,
            resolution_source="mlflow_artifacts",
        )

    hydra_output_dir = _resolve_hydra_output_dir(client, run_id, metadata)
    if hydra_output_dir is not None:
        return _stage_from_local_run_dir(
            run_id=run_id,
            tracking_uri=tracking_uri,
            stage_dir=stage_dir,
            hydra_output_dir=hydra_output_dir,
            preferred_checkpoint=checkpoint,
            metadata=metadata,
        )

    raise FileNotFoundError(
        "Could not stage artifacts for MLflow run "
        f"{run_id}. Neither MLflow artifacts (`hydra/config.yaml`, checkpoints, "
        "`metadata/run_metadata.json`) nor a usable `hydra_output_dir` tag were available."
    )


def build_sample_command(
    staged_run: StagedMlflowRun,
    *,
    out_dir: str,
    tasks_yaml: str,
    batch_size: int,
    save_traj: bool = False,
    center_clean_traj_com: bool = False,
    center_prot_traj_com: bool = False,
    output_motif_chains: bool = False,
    write_eval_pdb: bool = True,
    extra_sample_args: Sequence[str] | None = None,
) -> list[str]:
    repo_root = _repo_root()
    command = [
        sys.executable,
        str(repo_root / "sample.py"),
        _hydra_string_override("model_dir", staged_run.run_dir),
        _hydra_string_override("out_dir", out_dir),
        _hydra_string_override("checkpoint_path", staged_run.checkpoint_path),
        "checkpoint_idx=-1",
        _hydra_string_override("sampler.tasks_yaml", tasks_yaml),
        f"sampler.batch_size={int(batch_size)}",
        f"save_traj={'true' if save_traj else 'false'}",
    ]
    if center_clean_traj_com:
        command.append("center_clean_traj_com=true")
    if center_prot_traj_com:
        command.append("center_prot_traj_com=true")
    if output_motif_chains:
        command.append("output_motif_chains=true")
    if not write_eval_pdb:
        command.append("write_eval_pdb=false")
    if extra_sample_args:
        command.extend(extra_sample_args)
    return command


def write_sampling_provenance(
    out_dir: str,
    *,
    staged_run: StagedMlflowRun,
    tasks_yaml: str,
    benchmark: str | None,
    num_samples_per_task: int | None,
    ode_sampling: bool,
    sample_command: Sequence[str],
) -> str:
    metadata_dir = os.path.join(out_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    provenance_path = os.path.join(metadata_dir, "mlflow_sampling_source.json")
    payload = {
        "source_run_id": staged_run.run_id,
        "tracking_uri": staged_run.tracking_uri,
        "selected_checkpoint_mode": staged_run.selected_checkpoint_mode,
        "selected_checkpoint_artifact": staged_run.selected_checkpoint_artifact,
        "resolution_source": staged_run.resolution_source,
        "tasks_yaml": tasks_yaml,
        "benchmark": benchmark,
        "num_samples_per_task": num_samples_per_task,
        "ode_sampling": ode_sampling,
        "staged_run_dir": staged_run.run_dir,
        "staged_checkpoint_path": staged_run.checkpoint_path,
        "sample_command": list(sample_command),
    }
    with open(provenance_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return provenance_path


def _ensure_mlflow_file_store_allowed() -> None:
    """Match training scripts: allow local ``mlruns`` file-store by default."""
    if os.environ.get("MLFLOW_ALLOW_FILE_STORE") is None:
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


def _build_mlflow_client(tracking_uri: str) -> Any:
    _ensure_mlflow_file_store_allowed()
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=tracking_uri)


def run_sampling_from_mlflow(
    *,
    run_id: str,
    out_dir: str,
    tracking_uri: str | None = None,
    tasks_yaml: str | None = None,
    benchmark: str | None = None,
    cache_dir: str | None = None,
    checkpoint: str = "best",
    batch_size: int = 1,
    num_samples_per_task: int | None = None,
    ode_sampling: bool = False,
    save_traj: bool = False,
    center_clean_traj_com: bool = False,
    center_prot_traj_com: bool = False,
    output_motif_chains: bool = False,
    write_eval_pdb: bool = True,
    extra_sample_args: Sequence[str] | None = None,
    client: Any | None = None,
) -> subprocess.CompletedProcess[bytes]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive (got {batch_size}).")
    if num_samples_per_task is not None and num_samples_per_task <= 0:
        raise ValueError(
            f"num_samples_per_task must be positive (got {num_samples_per_task})."
        )
    tracking_uri = resolve_tracking_uri(tracking_uri)
    cache_dir = _resolve_cache_dir(cache_dir)
    tasks_yaml_path = resolve_tasks_yaml(tasks_yaml, benchmark)
    tasks_yaml_path = normalize_tasks_yaml(
        tasks_yaml_path,
        cache_dir=cache_dir,
        run_id=run_id,
        num_samples_per_task=num_samples_per_task,
    )
    resolved_out_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(out_dir)))
    parsed_extra_args = compose_sample_args(
        extra_sample_args or [],
        ode_sampling=ode_sampling,
    )
    active_client = client or _build_mlflow_client(tracking_uri)

    staged_run = stage_run_artifacts(
        active_client,
        run_id,
        tracking_uri=tracking_uri,
        cache_dir=cache_dir,
        checkpoint=checkpoint,
    )

    os.makedirs(resolved_out_dir, exist_ok=True)
    command = build_sample_command(
        staged_run,
        out_dir=resolved_out_dir,
        tasks_yaml=tasks_yaml_path,
        batch_size=batch_size,
        save_traj=save_traj,
        center_clean_traj_com=center_clean_traj_com,
        center_prot_traj_com=center_prot_traj_com,
        output_motif_chains=output_motif_chains,
        write_eval_pdb=write_eval_pdb,
        extra_sample_args=parsed_extra_args,
    )
    write_sampling_provenance(
        resolved_out_dir,
        staged_run=staged_run,
        tasks_yaml=tasks_yaml_path,
        benchmark=benchmark,
        num_samples_per_task=num_samples_per_task,
        ode_sampling=ode_sampling,
        sample_command=command,
    )
    return subprocess.run(command, cwd=str(_repo_root()), check=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch ProteinZen sampling from an MLflow training run."
    )
    parser.add_argument("--run-id", required=True, help="MLflow run id with logged hydra/checkpoint artifacts.")
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI. Defaults to MLFLOW_TRACKING_URI or REPO_ROOT/mlruns.")
    parser.add_argument("--out-dir", required=True, help="Where to write sampling outputs.")
    parser.add_argument("--cache-dir", default=None, help="Local cache directory for downloaded MLflow artifacts.")
    parser.add_argument(
        "--checkpoint",
        choices=("best", "last"),
        default="best",
        help="Preferred checkpoint to use from run metadata.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Sampling batch size forwarded to sample.py.",
    )
    parser.add_argument(
        "--num-samples-per-task",
        type=int,
        default=None,
        help="Override `num_samples` for every task entry in the tasks YAML before sampling.",
    )
    parser.add_argument(
        "--ode-sampling",
        action="store_true",
        help="Apply the ODE sampling overrides from `submit_unconditional_ode.sh`.",
    )
    parser.add_argument(
        "--save-traj",
        action="store_true",
        help="Forward save_traj=true to sample.py.",
    )
    parser.add_argument(
        "--center-clean-traj-com",
        action="store_true",
        help="Forward center_clean_traj_com=true to sample.py.",
    )
    parser.add_argument(
        "--center-prot-traj-com",
        action="store_true",
        help="Forward center_prot_traj_com=true to sample.py.",
    )
    parser.add_argument(
        "--output-motif-chains",
        action="store_true",
        help="Forward output_motif_chains=true to sample.py.",
    )
    parser.add_argument(
        "--no-write-eval-pdb",
        action="store_true",
        help="Forward write_eval_pdb=false to sample.py (skip samples_eval/ output).",
    )
    parser.add_argument(
        "--sample-arg",
        action="append",
        default=[],
        help="Extra Hydra override forwarded to sample.py, e.g. --sample-arg corrupter.num_timesteps=200",
    )
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--tasks-yaml", help="Explicit sampling task YAML path.")
    task_group.add_argument(
        "--benchmark",
        choices=tuple(sorted(BENCHMARK_TASKS)),
        help="Named benchmark shortcut for a built-in tasks YAML.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_sampling_from_mlflow(
        run_id=args.run_id,
        tracking_uri=args.tracking_uri,
        out_dir=args.out_dir,
        tasks_yaml=args.tasks_yaml,
        benchmark=args.benchmark,
        cache_dir=args.cache_dir,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        num_samples_per_task=args.num_samples_per_task,
        ode_sampling=args.ode_sampling,
        save_traj=args.save_traj,
        center_clean_traj_com=args.center_clean_traj_com,
        center_prot_traj_com=args.center_prot_traj_com,
        output_motif_chains=args.output_motif_chains,
        write_eval_pdb=not args.no_write_eval_pdb,
        extra_sample_args=args.sample_arg,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
