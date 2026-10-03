"""Build Stage-2 motif-scaffolding tasks from a Stage-1 layout manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import math
from typing import Any

import yaml

from nomadflow.runtime.sampling.layout_manifest import build_stage2_contigs_from_viz_pdb

try:
    from nomadflow.runtime.sampling.layout_manifest_validation import validate_layout_manifest
except ModuleNotFoundError:
    def _is_finite_number(value: Any) -> bool:
        if isinstance(value, bool):
            return False
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    def _validate_transform_rotation(rotation: Any, *, layout_id: str) -> None:
        if not isinstance(rotation, list) or len(rotation) != 3:
            raise ValueError(f"{layout_id}: transform.rotation must be a 3x3 matrix")
        for row in rotation:
            if not isinstance(row, list) or len(row) != 3:
                raise ValueError(f"{layout_id}: transform.rotation must be a 3x3 matrix")
            if not all(_is_finite_number(v) for v in row):
                raise ValueError(f"{layout_id}: transform.rotation contains non-finite values")

    def _validate_layout_manifest_entry(entry: dict[str, Any]) -> None:
        if not isinstance(entry, dict):
            raise TypeError("layout manifest entry must be an object")
        layout_id = str(entry.get("layout_id") or "<missing-layout-id>")
        if not isinstance(entry.get("output_pdb_contigs"), str) or not entry["output_pdb_contigs"]:
            raise ValueError(f"{layout_id}: output_pdb_contigs is required")
        source_residue_keys = entry.get("source_residue_keys")
        if not isinstance(source_residue_keys, list) or not source_residue_keys:
            raise ValueError(f"{layout_id}: source_residue_keys must be non-empty")
        for key in source_residue_keys:
            for required in ("native_chain", "native_res_idx", "output_chain", "output_res_idx"):
                if required not in key:
                    raise ValueError(f"{layout_id}: source_residue_keys missing field {required!r}")
            if int(key["output_res_idx"]) <= 0:
                raise ValueError(f"{layout_id}: output_res_idx must be positive")
        transforms = entry.get("motif_group_transforms")
        if transforms is not None:
            if not isinstance(transforms, list):
                raise ValueError(f"{layout_id}: motif_group_transforms must be a list")
            for transform in transforms:
                if not isinstance(transform, dict):
                    raise ValueError(f"{layout_id}: motif_group_transforms entries must be objects")
                _validate_transform_rotation(transform.get("rotation"), layout_id=layout_id)
                translation = transform.get("translation")
                if not isinstance(translation, list) or len(translation) != 3:
                    raise ValueError(f"{layout_id}: transform.translation must have 3 values")
                if not all(_is_finite_number(v) for v in translation):
                    raise ValueError(f"{layout_id}: transform.translation contains non-finite values")

    def validate_layout_manifest(entries: list[dict[str, Any]]) -> None:
        for entry in entries:
            _validate_layout_manifest_entry(entry)


_SPEC_RE = re.compile(r"^([A-Za-z0-9])(\d+)$")


def _load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_num, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"Manifest line {line_num} is not a JSON object")
            entries.append(payload)
    return entries


def _parse_single_residue_spec(spec: str) -> tuple[str, int]:
    match = _SPEC_RE.fullmatch(spec)
    if match is None:
        raise ValueError(f"Only single-residue specs are supported in Stage-2 builder, got {spec!r}")
    return match.group(1), int(match.group(2))


def _load_available_residues(pdb_path: Path) -> set[tuple[str, int]]:
    if not pdb_path.is_file():
        raise FileNotFoundError(f"Stage-1 sample PDB does not exist: {pdb_path}")
    residues: set[tuple[str, int]] = set()
    for line in pdb_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM"):
            continue
        chain = line[21:22].strip()
        if not chain:
            continue
        res_idx = int(line[22:26])
        residues.add((chain, res_idx))
    return residues


def _build_specs_from_viz_chains(
    pdb_path: Path,
    *,
    sample_chain_name: str,
    excluded_chain_names: set[str] | None = None,
) -> tuple[list[str], list[list[str]]]:
    stage2_contigs = build_stage2_contigs_from_viz_pdb(
        pdb_path,
        sample_chain_name=sample_chain_name,
        excluded_chain_names=excluded_chain_names,
    )
    if stage2_contigs is None:
        raise ValueError(
            f"PDB {pdb_path} has no motif chains distinct from sample_chain_name={sample_chain_name!r}."
        )
    pdb_contigs, groups = stage2_contigs
    return [item.strip() for item in pdb_contigs.split(",") if item.strip()], groups


def _sanitize_specs(specs: list[str], available_residues: set[tuple[str, int]]) -> list[str]:
    kept: list[str] = []
    seen: set[str] = set()
    for spec in specs:
        chain, res_idx = _parse_single_residue_spec(spec)
        if (chain, res_idx) not in available_residues:
            continue
        if spec in seen:
            continue
        seen.add(spec)
        kept.append(spec)
    return kept


def _sanitize_groups(
    motif_groups: list[Any],
    available_residues: set[tuple[str, int]],
) -> list[list[str]]:
    sanitized_groups: list[list[str]] = []
    globally_seen: set[str] = set()
    for group in motif_groups:
        if not isinstance(group, list):
            continue
        sanitized_group: list[str] = []
        for spec in group:
            if not isinstance(spec, str):
                continue
            chain, res_idx = _parse_single_residue_spec(spec)
            if (chain, res_idx) not in available_residues:
                continue
            if spec in globally_seen:
                continue
            globally_seen.add(spec)
            sanitized_group.append(spec)
        if sanitized_group:
            sanitized_groups.append(sanitized_group)
    return sanitized_groups


def build_stage2_tasks(
    entries: list[dict[str, Any]],
    *,
    stage2_num_samples: int,
    default_total_length_mode: str,
    include_motif_groups: bool,
    motif_source: str,
    viz_excluded_chain_names: set[str] | None = None,
) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    for entry in entries:
        layout_id = str(entry["layout_id"])
        stage1_task_name = str(entry.get("stage1_task_name") or "motif_scaffolding")
        sample_chain_name = str(entry.get("sample_chain_name") or "A")
        stage2_sample_chain_name = str(
            entry.get("stage2_sample_chain_name") or sample_chain_name
        )
        sample_path = str(entry["stage1_sample_path"])
        raw_groups: list[list[str]] | None = None
        stage2_input_pdb = entry.get("stage2_input_pdb")
        stage2_pdb_contigs = entry.get("stage2_pdb_contigs")
        stage2_motif_groups = entry.get("stage2_motif_groups")
        if motif_source == "viz_chains":
            viz_path = Path(str(stage2_input_pdb or entry.get("stage1_viz_path") or sample_path))
            if isinstance(stage2_pdb_contigs, str) and stage2_pdb_contigs.strip():
                raw_specs = [
                    item.strip()
                    for item in stage2_pdb_contigs.split(",")
                    if item.strip()
                ]
                raw_groups = stage2_motif_groups if isinstance(stage2_motif_groups, list) else None
            else:
                raw_specs, raw_groups = _build_specs_from_viz_chains(
                    viz_path,
                    sample_chain_name=stage2_sample_chain_name,
                    excluded_chain_names=viz_excluded_chain_names,
                )
            sample_path = str(viz_path)
            stage2_input_pdb = str(viz_path)
            stage2_pdb_contigs = ",".join(raw_specs)
        elif motif_source == "metadata":
            raw_specs = [item.strip() for item in str(entry["output_pdb_contigs"]).split(",") if item.strip()]
        else:
            raise ValueError(f"Unsupported motif_source: {motif_source!r}")
        sample_length = entry.get("sample_length")
        available_residues = _load_available_residues(Path(sample_path))
        sanitized_specs = _sanitize_specs(raw_specs, available_residues)
        if not sanitized_specs:
            raise ValueError(
                f"Layout {layout_id!r} produced no valid motif residues for {sample_path}. "
                "Cannot build Stage-2 task."
            )
        pdb_contigs = ",".join(sanitized_specs)

        if default_total_length_mode == "sample_length":
            if sample_length is None:
                raise ValueError(
                    f"Manifest entry {layout_id!r} has no sample_length but sample_length mode was requested."
                )
            total_length = str(int(sample_length))
        else:
            total_length = str(default_total_length_mode)

        task_name = f"{stage1_task_name}__{layout_id}"
        task: dict[str, Any] = {
            "task": "motif_scaffolding",
            "name": task_name,
            "pdb": sample_path,
            "pdb_contigs": pdb_contigs,
            "contigs_idx": "",
            "num_samples": int(stage2_num_samples),
            "total_length": total_length,
            "sample_chain_name": sample_chain_name,
            "source_layout_id": layout_id,
            "source_stage1_task_name": stage1_task_name,
        }
        if motif_source == "viz_chains":
            task["stage2_input_pdb"] = str(stage2_input_pdb or sample_path)
            task["stage2_pdb_contigs"] = pdb_contigs
            task["stage2_sample_chain_name"] = stage2_sample_chain_name
            if isinstance(raw_groups, list):
                task["stage2_motif_groups"] = raw_groups
        for key in (
            "rmsd_reference_chain",
            "rmsd_folded_chain",
            "motif_reference_residues",
            "motif_folded_residues",
            "motif_bb_reference_residues",
            "motif_bb_folded_residues",
            "motif_native_keys",
        ):
            if key in entry:
                task[key] = entry[key]
        if include_motif_groups:
            motif_groups = raw_groups if motif_source == "viz_chains" else entry.get("output_motif_groups")
            if isinstance(motif_groups, list):
                task["motif_groups"] = _sanitize_groups(motif_groups, available_residues)
        tasks.append(task)
    return {"tasks": tasks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout-manifest",
        type=Path,
        required=True,
        help="Path to Stage-1 layout_manifest.jsonl",
    )
    parser.add_argument(
        "--output-yaml",
        type=Path,
        required=True,
        help="Where to write Stage-2 tasks yaml",
    )
    parser.add_argument(
        "--stage2-num-samples",
        type=int,
        default=1,
        help="num_samples per Stage-2 task (1 for 1:1 layout->sample pass)",
    )
    parser.add_argument(
        "--total-length",
        type=str,
        default="sample_length",
        help=(
            "Total length for Stage-2 tasks. "
            "Use 'sample_length' to mirror Stage-1 sample length, "
            "or pass explicit value/range like '128' or '60-150'."
        ),
    )
    parser.add_argument(
        "--include-motif-groups",
        action="store_true",
        help=(
            "Include output_motif_groups from layout manifest in Stage-2 tasks. "
            "By default this is disabled to avoid failures on motif-index collisions."
        ),
    )
    parser.add_argument(
        "--motif-source",
        type=str,
        choices=("metadata", "viz_chains"),
        default="metadata",
        help=(
            "How to derive Stage-2 motif residue specs. "
            "'metadata' uses output_pdb_contigs from layout metadata; "
            "'viz_chains' infers motifs from non-sample chains in stage1_viz_path."
        ),
    )
    parser.add_argument(
        "--viz-exclude-chains",
        type=str,
        default="V",
        help=(
            "Comma-separated chain IDs to ignore when --motif-source=viz_chains. "
            "Default excludes V (visual aggregator chain)."
        ),
    )
    args = parser.parse_args()

    entries = _load_manifest(args.layout_manifest.resolve())
    validate_layout_manifest(entries)
    payload = build_stage2_tasks(
        entries,
        stage2_num_samples=args.stage2_num_samples,
        default_total_length_mode=args.total_length,
        include_motif_groups=args.include_motif_groups,
        motif_source=args.motif_source,
        viz_excluded_chain_names={
            item.strip() for item in args.viz_exclude_chains.split(",") if item.strip()
        },
    )
    args.output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with args.output_yaml.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    print(f"Wrote {len(payload['tasks'])} Stage-2 tasks to {args.output_yaml.resolve()}")


if __name__ == "__main__":
    main()
