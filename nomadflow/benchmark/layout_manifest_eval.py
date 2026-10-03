"""Enrich Stage-2 eval metadata with motif groups from Stage-1 layout manifests."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from nomadflow.benchmark.analysis_inputs import parent_motif_task_id
from nomadflow.benchmark.motif_group_annotation import apply_source_residue_key_repair

logger = logging.getLogger(__name__)


def load_layout_manifest(manifest_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load layout manifest entries grouped by ``layout_id``.

    Stage-1 sample names are not unique across tasks (same ``len_*_protein_id*``
    can appear for 1PRW and 1BCF, or 5AOU_ext10 and 5AOU_ext2x10). Keep every
    copy so enrichment can disambiguate by ``stage1_task_name``.
    """
    entries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_num, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"Manifest line {line_num} is not a JSON object")
            layout_id = str(payload["layout_id"])
            entries[layout_id].append(payload)
    return dict(entries)


def _repair_grouped_samples(
    samples_metadata: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    repaired: dict[str, dict[str, Any]] = {}
    for sample_name, sample_metadata in samples_metadata.items():
        if isinstance(sample_metadata, dict) and _has_grouped_source_residue_keys(sample_metadata):
            repaired[sample_name] = apply_source_residue_key_repair(dict(sample_metadata))
        else:
            repaired[sample_name] = sample_metadata
    return repaired


def stage1_task_name_for_sample(
    sample_metadata: dict[str, Any],
    *,
    task_by_name: dict[str, dict[str, Any]] | None = None,
) -> str | None:
    """Stage-1 parent task used to disambiguate duplicate ``layout_id``s."""
    source = sample_metadata.get("source_stage1_task_name")
    if source is not None and str(source).strip():
        return str(source).strip()

    task_name = sample_metadata.get("name")
    if isinstance(task_name, str) and task_by_name and task_name in task_by_name:
        task_source = task_by_name[task_name].get("source_stage1_task_name")
        if task_source is not None and str(task_source).strip():
            return str(task_source).strip()

    if isinstance(task_name, str) and task_name.strip():
        parent = parent_motif_task_id(task_name, sample_metadata)
        if parent.strip() and parent != task_name:
            return parent.strip()
        if parent.strip() and "__" not in parent:
            return parent.strip()
    return None


def select_layout_entry(
    entries: list[dict[str, Any]],
    *,
    stage1_task_name: str | None,
) -> dict[str, Any] | None:
    """Pick a layout entry. Unique ids need no task name; duplicates do."""
    if not entries:
        return None
    if len(entries) == 1:
        return entries[0]
    if not stage1_task_name:
        return None
    wanted = str(stage1_task_name)
    matched = [
        entry
        for entry in entries
        if str(entry.get("stage1_task_name") or "") == wanted
    ]
    if len(matched) == 1:
        return matched[0]
    return None


def _load_tasks_by_name(sampling_run: Path) -> dict[str, dict[str, Any]]:
    config_path = sampling_run / "config.yaml"
    if not config_path.is_file():
        return {}
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        return {}
    by_name: dict[str, dict[str, Any]] = {}
    for task in tasks:
        if isinstance(task, dict) and task.get("name") is not None:
            by_name[str(task["name"])] = task
    return by_name


def resolve_layout_manifest_path(sampling_run: Path) -> Path | None:
    """Find Stage-1 layout_manifest.jsonl for a Stage-2 sampling run."""
    sampling_run = Path(sampling_run)
    local_manifest = sampling_run / "layout_manifest.jsonl"
    if local_manifest.is_file():
        return local_manifest

    config_path = sampling_run / "config.yaml"
    if not config_path.is_file():
        return None

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        return None

    for task in tasks:
        if not isinstance(task, dict):
            continue
        for key in ("stage2_input_pdb", "pdb"):
            pdb_value = task.get(key)
            if not isinstance(pdb_value, str) or not pdb_value.strip():
                continue
            pdb_path = Path(pdb_value)
            if pdb_path.parent.name != "samples":
                continue
            manifest_path = pdb_path.parent.parent / "layout_manifest.jsonl"
            if manifest_path.is_file():
                return manifest_path
    return None


def infer_layout_id(
    sample_metadata: dict[str, Any],
    *,
    task_by_name: dict[str, dict[str, Any]] | None = None,
) -> str | None:
    """Resolve Stage-1 layout_id for a Stage-2 sample."""
    source_layout_id = sample_metadata.get("source_layout_id")
    if source_layout_id:
        return str(source_layout_id)

    task_name = sample_metadata.get("name")
    if isinstance(task_name, str) and task_by_name and task_name in task_by_name:
        task_layout_id = task_by_name[task_name].get("source_layout_id")
        if task_layout_id:
            return str(task_layout_id)

    if isinstance(task_name, str) and "__" in task_name:
        candidate = task_name.rsplit("__", 1)[-1]
        if candidate.startswith("len_"):
            return candidate
    return None


def _stage2_motif_residue_indices(sample_metadata: dict[str, Any]) -> list[int]:
    for key in ("motif_reference_residues", "fixed_bb_res_idx", "fixed_seq_res_idx"):
        values = sample_metadata.get(key)
        if isinstance(values, list) and values:
            return [int(value) for value in values]
    return []


def _has_grouped_source_residue_keys(sample_metadata: dict[str, Any]) -> bool:
    source_residue_keys = sample_metadata.get("source_residue_keys")
    if not isinstance(source_residue_keys, list) or not source_residue_keys:
        return False
    return any(
        isinstance(entry, dict) and entry.get("group_id") is not None
        for entry in source_residue_keys
    )


def enrich_sample_metadata_from_layout_entry(
    sample_metadata: dict[str, Any],
    layout_entry: dict[str, Any],
) -> dict[str, Any]:
    """Attach grouped source_residue_keys using Stage-2 scaffold motif indices."""
    if _has_grouped_source_residue_keys(sample_metadata):
        return apply_source_residue_key_repair(dict(sample_metadata))

    stage1_keys = layout_entry.get("source_residue_keys")
    if not isinstance(stage1_keys, list) or not stage1_keys:
        return sample_metadata

    output_residues = _stage2_motif_residue_indices(sample_metadata)
    if len(output_residues) != len(stage1_keys):
        logger.warning(
            "eval_error stage2_stage1_residue_count_mismatch "
            "(stage2=%s, stage1=%s, layout_id=%s) — not a model fail; "
            "excluded from layout, pass_final unchanged (missing groups still fail AND)",
            len(output_residues),
            len(stage1_keys),
            layout_entry.get("layout_id"),
        )
        result = dict(sample_metadata)
        result["eval_error"] = "stage2_stage1_residue_count_mismatch"
        result["eval_error_detail"] = {
            "stage2_n_residues": len(output_residues),
            "stage1_n_keys": len(stage1_keys),
            "layout_id": layout_entry.get("layout_id"),
        }
        return result

    sample_chain_name = str(
        sample_metadata.get("sample_chain_name")
        or layout_entry.get("sample_chain_name")
        or "A"
    )
    enriched_keys: list[dict[str, Any]] = []
    for stage1_key, output_res_idx in zip(stage1_keys, output_residues, strict=True):
        if not isinstance(stage1_key, dict):
            continue
        entry = {
            "native_key": stage1_key.get("native_key")
            or f"{stage1_key['native_chain']}{stage1_key['native_res_idx']}",
            "native_chain": str(stage1_key["native_chain"]),
            "native_res_idx": int(stage1_key["native_res_idx"]),
            "output_chain": sample_chain_name,
            "output_res_idx": int(output_res_idx),
            "is_unindexed": bool(stage1_key.get("is_unindexed", False)),
        }
        if stage1_key.get("group_id") is not None:
            entry["group_id"] = int(stage1_key["group_id"])
        enriched_keys.append(entry)

    result = dict(sample_metadata)
    result["source_residue_keys"] = enriched_keys
    if layout_entry.get("native_pdb"):
        result["native_pdb"] = layout_entry["native_pdb"]
    if layout_entry.get("motif_native_keys"):
        result["motif_native_keys"] = layout_entry["motif_native_keys"]
    result["source_layout_id"] = str(layout_entry["layout_id"])
    return apply_source_residue_key_repair(result)


def enrich_samples_metadata_from_layout_manifest(
    samples_metadata: dict[str, dict[str, Any]],
    *,
    layout_manifest_path: Path | str | None = None,
    sampling_run: Path | str | None = None,
) -> dict[str, dict[str, Any]]:
    """Enrich Stage-2 samples with grouped motif metadata from Stage-1 manifest."""
    if layout_manifest_path is None and sampling_run is not None:
        layout_manifest_path = resolve_layout_manifest_path(Path(sampling_run))
    if layout_manifest_path is None:
        return _repair_grouped_samples(samples_metadata)

    manifest_path = Path(layout_manifest_path)
    if not manifest_path.is_file():
        logger.warning("Layout manifest not found: %s", manifest_path)
        return _repair_grouped_samples(samples_metadata)

    layout_by_id = load_layout_manifest(manifest_path)
    task_by_name = _load_tasks_by_name(Path(sampling_run)) if sampling_run is not None else {}

    enriched: dict[str, dict[str, Any]] = {}
    for sample_name, sample_metadata in samples_metadata.items():
        if not isinstance(sample_metadata, dict):
            enriched[sample_name] = sample_metadata
            continue
        if _has_grouped_source_residue_keys(sample_metadata):
            enriched[sample_name] = apply_source_residue_key_repair(dict(sample_metadata))
            continue
        layout_id = infer_layout_id(sample_metadata, task_by_name=task_by_name)
        candidates = layout_by_id.get(layout_id or "", [])
        stage1_task = stage1_task_name_for_sample(
            sample_metadata, task_by_name=task_by_name
        )
        layout_entry = select_layout_entry(
            candidates, stage1_task_name=stage1_task
        )
        if layout_entry is None:
            looks_stage2 = (
                isinstance(sample_metadata.get("name"), str)
                and "__len_" in str(sample_metadata.get("name"))
            )
            if looks_stage2:
                error = (
                    "layout_id_ambiguous"
                    if candidates
                    else "layout_id_missing_or_unknown"
                )
                logger.warning(
                    "eval_error %s sample=%s layout_id=%s stage1_task=%s n_candidates=%s",
                    error,
                    sample_name,
                    layout_id,
                    stage1_task,
                    len(candidates),
                )
                marked = dict(sample_metadata)
                marked["eval_error"] = error
                marked["eval_error_detail"] = {
                    "layout_id": layout_id,
                    "stage1_task_name": stage1_task,
                    "n_candidates": len(candidates),
                }
                enriched[sample_name] = marked
            else:
                enriched[sample_name] = sample_metadata
            continue
        enriched[sample_name] = enrich_sample_metadata_from_layout_entry(
            sample_metadata,
            layout_entry,
        )
    return enriched


def enrich_samples_metadata_file(
    sampling_run: Path | str,
    *,
    layout_manifest_path: Path | str | None = None,
    write: bool = True,
) -> dict[str, dict[str, Any]]:
    """Load, enrich, and optionally persist samples_metadata.json for a run."""
    from nomadflow.benchmark.analysis_inputs import load_samples_metadata

    sampling_run = Path(sampling_run)
    metadata = load_samples_metadata(sampling_run)
    enriched = enrich_samples_metadata_from_layout_manifest(
        metadata,
        layout_manifest_path=layout_manifest_path,
        sampling_run=sampling_run,
    )
    if write and json.dumps(enriched, sort_keys=True) != json.dumps(metadata, sort_keys=True):
        metadata_path = sampling_run / "samples_metadata.json"
        metadata_path.write_text(json.dumps(enriched), encoding="utf-8")
        logger.info("Updated grouped motif metadata in %s", metadata_path)
    return enriched
