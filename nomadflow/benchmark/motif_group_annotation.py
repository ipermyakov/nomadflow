"""Annotate baseline samples_metadata with motif_groups / source_residue_keys.

Used when sampling with vanilla contig fixation (no multi-motif SE3 conditioning)
but still needing per-group downstream metrics. Motif group labels come from the
tasks YAML; output residue indices come from fixed_seq_res_idx written by the
sampler.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

_MIN_SEGMENT_LEN_TO_REPAIR = 4


def _native_key(chain: str, residue: int) -> str:
    return f"{str(chain)}{int(residue)}"


def _parse_native_key(native_key: str) -> tuple[str, int]:
    if len(native_key) < 2:
        raise ValueError(f"Invalid native key: {native_key!r}")
    return native_key[0], int(native_key[1:])


def expand_group_segment(segment_spec: str) -> list[str]:
    """Expand ``A38`` or ``A22-31`` into native keys."""
    if not isinstance(segment_spec, str) or len(segment_spec) < 2:
        return []
    chain = segment_spec[0]
    tail = segment_spec[1:]
    if "-" in tail:
        start_s, end_s = tail.split("-", 1)
        start = int(start_s)
        end = int(end_s)
        step = 1 if end >= start else -1
        return [_native_key(chain, idx) for idx in range(start, end + step, step)]
    return [_native_key(chain, int(tail))]


def expand_motif_groups(
    motif_groups: Sequence[Sequence[str]],
) -> list[tuple[str, int]]:
    """Expand motif_groups into ordered ``(native_key, group_id)`` pairs (1-based)."""
    expanded: list[tuple[str, int]] = []
    for group_idx, group_segments in enumerate(motif_groups, start=1):
        if not isinstance(group_segments, (list, tuple)):
            raise ValueError(
                "motif_groups must be a list of groups; each group is a list of segment specs"
            )
        for segment_spec in group_segments:
            if not isinstance(segment_spec, str):
                raise ValueError(
                    f"Each motif_groups segment must be a string like A92-99 or A14; got {segment_spec!r}"
                )
            keys = expand_group_segment(segment_spec)
            if not keys:
                raise ValueError(f"Invalid motif_groups segment: {segment_spec!r}")
            for key in keys:
                expanded.append((key, group_idx))
    return expanded


def _resolve_native_pdb(task: Mapping[str, Any], *, repo_root: Path | None = None) -> str | None:
    pdb = task.get("pdb")
    if pdb is None:
        return None
    pdb_path = Path(str(pdb))
    candidates: list[Path] = []
    if pdb_path.is_absolute():
        candidates.append(pdb_path)
    else:
        candidates.append(Path.cwd() / pdb_path)
        if repo_root is not None:
            candidates.append(Path(repo_root) / pdb_path)
        candidates.append(pdb_path)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return str(pdb)


def _output_chain_for_index(sample: Mapping[str, Any], index: int) -> str:
    fixed_seq_chain = sample.get("fixed_seq_chain")
    if isinstance(fixed_seq_chain, list) and index < len(fixed_seq_chain):
        return str(fixed_seq_chain[index])
    for key in ("rmsd_reference_chain", "sample_chain_name", "fixed_seq_chain"):
        value = sample.get(key)
        if isinstance(value, str) and value:
            return value
    return "A"


def _tasks_by_name(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    by_name: dict[str, Mapping[str, Any]] = {}
    for task in tasks:
        name = task.get("name")
        if name is None:
            continue
        by_name[str(name)] = task
    return by_name


def load_tasks_from_yaml(tasks_yaml: str | Path) -> list[dict[str, Any]]:
    """Load task entries from a sampling tasks YAML."""
    import yaml

    path = Path(tasks_yaml)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if isinstance(payload, list):
        tasks = payload
    elif isinstance(payload, dict):
        tasks = payload.get("tasks", [])
    else:
        raise ValueError(f"Unrecognized tasks YAML format in {path}")
    if not isinstance(tasks, list):
        raise ValueError(f"tasks must be a list in {path}")
    return [dict(task) for task in tasks if isinstance(task, Mapping)]


def annotate_sample_metadata(
    sample: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
    require_motif_groups: bool = True,
) -> dict[str, Any]:
    """Return a copy of sample metadata annotated with motif group fields."""
    result = dict(sample)
    motif_groups = task.get("motif_groups")
    if motif_groups is None:
        if require_motif_groups:
            raise ValueError(
                f"Task {task.get('name')!r} has no motif_groups; cannot annotate sample"
            )
        return result
    if not isinstance(motif_groups, list):
        raise ValueError(f"motif_groups must be a list; got {type(motif_groups).__name__}")

    expanded = expand_motif_groups(motif_groups)
    fixed_seq_res_idx = sample.get("fixed_seq_res_idx")
    if not isinstance(fixed_seq_res_idx, list):
        raise ValueError(
            f"Sample {sample.get('name')!r} missing fixed_seq_res_idx list required for annotation"
        )
    if len(expanded) != len(fixed_seq_res_idx):
        raise ValueError(
            "motif_groups expansion length does not match fixed_seq_res_idx: "
            f"expanded={len(expanded)} fixed_seq={len(fixed_seq_res_idx)} "
            f"task={task.get('name')!r} sample={sample.get('name')!r}"
        )

    motif_native_keys = [key for key, _ in expanded]
    source_residue_keys: list[dict[str, Any]] = []
    motif_group_assignments: dict[str, int] = {}
    for index, ((native_key, group_id), output_res_idx) in enumerate(
        zip(expanded, fixed_seq_res_idx, strict=True)
    ):
        native_chain, native_res_idx = _parse_native_key(native_key)
        output_chain = _output_chain_for_index(sample, index)
        entry = {
            "native_key": native_key,
            "native_chain": native_chain,
            "native_res_idx": native_res_idx,
            "output_chain": output_chain,
            "output_res_idx": int(output_res_idx),
            "is_unindexed": True,
            "group_id": int(group_id),
        }
        source_residue_keys.append(entry)
        motif_group_assignments[f"{output_chain}:{int(output_res_idx)}"] = int(group_id)

    result["motif_groups"] = [list(group) for group in motif_groups]
    result["motif_native_keys"] = motif_native_keys
    result["source_residue_keys"] = source_residue_keys
    result["motif_group_assignments"] = motif_group_assignments

    native_pdb = _resolve_native_pdb(task, repo_root=repo_root)
    if native_pdb is not None:
        result["native_pdb"] = native_pdb
    return apply_source_residue_key_repair(result)


def _is_contiguous(values: Sequence[int]) -> bool:
    return all(int(values[i]) - int(values[i - 1]) == 1 for i in range(1, len(values)))


def _native_contiguous_runs(
    keys: Sequence[Mapping[str, Any]],
) -> list[list[int]]:
    """Return index runs that are native-contiguous on one chain/group/output chain."""
    runs: list[list[int]] = []
    current: list[int] = []

    def _identity(entry: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            int(entry["group_id"]) if entry.get("group_id") is not None else None,
            str(entry.get("native_chain") or ""),
            str(entry.get("output_chain") or entry.get("output_asym_id") or ""),
        )

    for index, entry in enumerate(keys):
        if not isinstance(entry, Mapping) or entry.get("native_res_idx") is None:
            if current:
                runs.append(current)
                current = []
            continue
        if not current:
            current = [index]
            continue
        prev = keys[current[-1]]
        same_run = (
            _identity(entry) == _identity(prev)
            and int(entry["native_res_idx"]) == int(prev["native_res_idx"]) + 1
        )
        if same_run:
            current.append(index)
        else:
            runs.append(current)
            current = [index]
    if current:
        runs.append(current)
    return runs


def _repair_far_outlier_output(
    outputs: list[int],
    occupied: set[int],
) -> list[int] | None:
    """Replace one far-from-block output index with the unique hole in the block.

    Typical 7K4V_ext20 failure: native A61–80 maps to 193–202, 110, 204–212.
    110 is outside the helix; 203 is the unused slot that actually holds the copy.
    Adjacent +2 gaps (real placement next to another motif residue) are left alone.
    """
    n = len(outputs)
    if n < _MIN_SEGMENT_LEN_TO_REPAIR or _is_contiguous(outputs):
        return None
    hits: list[tuple[int, int]] = []
    for index, outlier in enumerate(outputs):
        rest = outputs[:index] + outputs[index + 1 :]
        rest_min, rest_max = min(rest), max(rest)
        if rest_max - rest_min != n - 1:
            continue
        missing = set(range(rest_min, rest_max + 1)) - set(rest)
        if len(missing) != 1:
            continue
        hole = missing.pop()
        if outlier == hole or rest_min <= outlier <= rest_max:
            continue
        if hole in occupied - {outlier} or hole <= 0:
            continue
        hits.append((index, hole))
    if len(hits) != 1:
        return None
    index, hole = hits[0]
    repaired = list(outputs)
    repaired[index] = hole
    if not _is_contiguous(repaired):
        return None
    return repaired


def repair_source_residue_keys(
    keys: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return copies of ``keys`` with far-outlier output indices repaired."""
    repaired = [dict(entry) for entry in keys]
    notes: list[dict[str, Any]] = []
    occupied = {
        int(entry["output_res_idx"])
        for entry in repaired
        if entry.get("output_res_idx") is not None
    }
    for run in _native_contiguous_runs(repaired):
        if len(run) < _MIN_SEGMENT_LEN_TO_REPAIR:
            continue
        outputs = [int(repaired[index]["output_res_idx"]) for index in run]
        new_outputs = _repair_far_outlier_output(outputs, occupied)
        if new_outputs is None:
            continue
        for offset, index in enumerate(run):
            if new_outputs[offset] == outputs[offset]:
                continue
            old = outputs[offset]
            new = new_outputs[offset]
            occupied.discard(old)
            occupied.add(new)
            repaired[index]["output_res_idx"] = new
            notes.append(
                {
                    "native_key": repaired[index].get("native_key"),
                    "group_id": repaired[index].get("group_id"),
                    "from": old,
                    "to": new,
                }
            )
    return repaired, notes


def apply_source_residue_key_repair(metadata: dict[str, Any]) -> dict[str, Any]:
    """Repair far-outlier ``output_res_idx`` and keep eval fields in lockstep."""
    keys = metadata.get("source_residue_keys")
    if not isinstance(keys, list) or not keys:
        return metadata
    repaired, notes = repair_source_residue_keys(keys)
    result = dict(metadata)
    result["source_residue_keys"] = repaired
    if notes:
        result["source_residue_key_repairs"] = notes
        logger.info(
            "Repaired %s far-outlier motif output index(es) for %s",
            len(notes),
            result.get("name") or result.get("path") or "sample",
        )
    assignments: dict[str, int] = {}
    for entry in repaired:
        if entry.get("group_id") is None or entry.get("output_res_idx") is None:
            continue
        chain = entry.get("output_chain")
        if not chain:
            continue
        assignments[f"{chain}:{int(entry['output_res_idx'])}"] = int(entry["group_id"])
    if assignments:
        result["motif_group_assignments"] = assignments
    fixed_seq = result.get("fixed_seq_res_idx")
    if isinstance(fixed_seq, list) and len(fixed_seq) == len(repaired):
        result["fixed_seq_res_idx"] = [int(entry["output_res_idx"]) for entry in repaired]
        chains = [entry.get("output_chain") for entry in repaired]
        if all(isinstance(chain, str) and chain for chain in chains):
            result["fixed_seq_chain"] = chains
    for field in ("motif_reference_residues", "motif_folded_residues"):
        values = result.get(field)
        if isinstance(values, list) and len(values) == len(repaired):
            result[field] = [int(entry["output_res_idx"]) for entry in repaired]
    return result


def annotate_samples_metadata(
    samples_metadata: Mapping[str, Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path | None = None,
    skip_missing_groups: bool = False,
) -> dict[str, dict[str, Any]]:
    """Annotate all samples that match a task name with motif_groups."""
    by_name = _tasks_by_name(tasks)
    annotated: dict[str, dict[str, Any]] = {}
    for sample_id, sample in samples_metadata.items():
        if not isinstance(sample, Mapping):
            raise ValueError(f"Sample entry {sample_id!r} must be a mapping")
        task_name = sample.get("name")
        if task_name is None:
            annotated[sample_id] = dict(sample)
            continue
        task = by_name.get(str(task_name))
        if task is None:
            annotated[sample_id] = dict(sample)
            continue
        if task.get("motif_groups") is None:
            if skip_missing_groups:
                annotated[sample_id] = dict(sample)
                continue
            raise ValueError(
                f"Task {task_name!r} has no motif_groups; pass skip_missing_groups=True to leave unchanged"
            )
        annotated[sample_id] = annotate_sample_metadata(
            sample,
            task,
            repo_root=repo_root,
            require_motif_groups=True,
        )
    return annotated


def annotate_samples_metadata_from_tasks_yaml(
    samples_metadata: Mapping[str, Mapping[str, Any]],
    tasks_yaml: str | Path,
    *,
    repo_root: Path | None = None,
    skip_missing_groups: bool = True,
) -> dict[str, dict[str, Any]]:
    """Load tasks YAML and annotate samples_metadata."""
    tasks = load_tasks_from_yaml(tasks_yaml)
    return annotate_samples_metadata(
        samples_metadata,
        tasks,
        repo_root=repo_root,
        skip_missing_groups=skip_missing_groups,
    )


def tasks_have_motif_groups(tasks: Sequence[Mapping[str, Any]]) -> bool:
    return any(isinstance(task.get("motif_groups"), list) for task in tasks)
