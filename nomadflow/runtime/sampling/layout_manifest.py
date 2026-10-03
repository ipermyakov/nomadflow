from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _segment_specs_from_source_residue_keys(
    source_residue_keys: list[dict[str, Any]],
) -> list[str]:
    specs: list[str] = []
    for item in source_residue_keys:
        chain = str(item["output_chain"])
        res_idx = int(item["output_res_idx"])
        specs.append(f"{chain}{res_idx}")
    return specs


def _motif_groups_from_source_residue_keys(
    source_residue_keys: list[dict[str, Any]],
) -> list[list[str]] | None:
    grouped: dict[int, list[str]] = {}
    for item in source_residue_keys:
        group_id = item.get("group_id")
        if group_id is None:
            return None
        group_key = int(group_id)
        chain = str(item["output_chain"])
        res_idx = int(item["output_res_idx"])
        grouped.setdefault(group_key, []).append(f"{chain}{res_idx}")

    if not grouped:
        return None
    return [grouped[group_id] for group_id in sorted(grouped)]


def _load_chain_residues(pdb_path: Path) -> dict[str, list[int]]:
    if not pdb_path.is_file():
        return {}
    chains: dict[str, set[int]] = {}
    for line in pdb_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM"):
            continue
        chain = line[21:22].strip()
        if not chain:
            continue
        res_idx = int(line[22:26])
        chains.setdefault(chain, set()).add(res_idx)
    return {chain: sorted(residues) for chain, residues in chains.items()}


def build_stage2_contigs_from_viz_pdb(
    pdb_path: Path,
    *,
    sample_chain_name: str,
    excluded_chain_names: set[str] | None = None,
) -> tuple[str, list[list[str]]] | None:
    """Infer Stage-2 conditioning contigs from non-sample chains in a viz PDB."""
    chain_residues = _load_chain_residues(pdb_path)
    excluded = set(excluded_chain_names or set())
    motif_chains = [
        chain
        for chain in sorted(chain_residues)
        if chain != sample_chain_name and chain not in excluded
    ]
    if not motif_chains:
        return None

    specs: list[str] = []
    groups: list[list[str]] = []
    for chain in motif_chains:
        group_specs = [f"{chain}{res_idx}" for res_idx in chain_residues[chain]]
        if group_specs:
            groups.append(group_specs)
            specs.extend(group_specs)
    if not specs:
        return None
    return ",".join(specs), groups


def _build_layout_entry(sample_name: str, sample_metadata: dict[str, Any]) -> dict[str, Any] | None:
    source_residue_keys = sample_metadata.get("source_residue_keys")
    if not isinstance(source_residue_keys, list) or len(source_residue_keys) == 0:
        return None
    output_motif_specs = _segment_specs_from_source_residue_keys(source_residue_keys)
    output_motif_groups = _motif_groups_from_source_residue_keys(source_residue_keys)

    task_name = str(sample_metadata.get("name") or "motif_scaffolding")
    sample_length = sample_metadata.get("sample_length")
    if sample_length is None:
        sample_length = sample_metadata.get("length")

    stage1_viz_path = str(sample_metadata.get("viz_path") or sample_metadata["path"])
    sample_chain_name = str(sample_metadata.get("sample_chain_name") or "A")

    entry: dict[str, Any] = {
        "layout_id": sample_name,
        "stage1_task_name": task_name,
        "stage1_sample_path": str(sample_metadata["path"]),
        "stage1_viz_path": stage1_viz_path,
        "sample_chain_name": sample_chain_name,
        "sample_length": int(sample_length) if sample_length is not None else None,
        "native_pdb": sample_metadata.get("native_pdb"),
        "source_pdb_contigs": sample_metadata.get("pdb_contigs"),
        "output_motif_specs": output_motif_specs,
        "output_pdb_contigs": ",".join(output_motif_specs),
        "source_residue_keys": source_residue_keys,
        "stage2_input_pdb": stage1_viz_path,
        "stage2_sample_chain_name": sample_chain_name,
    }
    stage2_contigs = build_stage2_contigs_from_viz_pdb(
        Path(stage1_viz_path),
        sample_chain_name=sample_chain_name,
        excluded_chain_names={"V"},
    )
    if stage2_contigs is not None:
        entry["stage2_pdb_contigs"], stage2_groups = stage2_contigs
        entry["stage2_motif_groups"] = stage2_groups
    for key in (
        "rmsd_reference_chain",
        "rmsd_folded_chain",
        "motif_reference_residues",
        "motif_folded_residues",
        "motif_bb_reference_residues",
        "motif_bb_folded_residues",
        "motif_native_keys",
    ):
        if key in sample_metadata:
            entry[key] = sample_metadata[key]
    if output_motif_groups is not None:
        entry["output_motif_groups"] = output_motif_groups
    return entry


def export_layout_manifest(
    *,
    samples_metadata_path: Path,
    output_path: Path,
) -> int:
    with samples_metadata_path.open("r", encoding="utf-8") as handle:
        samples_metadata = json.load(handle)
    if not isinstance(samples_metadata, dict):
        raise TypeError(f"Expected dict in {samples_metadata_path}, got {type(samples_metadata)!r}")

    entries: list[dict[str, Any]] = []
    for sample_name in sorted(samples_metadata):
        sample_metadata = samples_metadata[sample_name]
        if not isinstance(sample_metadata, dict):
            continue
        entry = _build_layout_entry(sample_name, sample_metadata)
        if entry is not None:
            entries.append(entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return len(entries)
