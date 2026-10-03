"""Eval metadata helpers for ProteinMPNN and RMSD downstream.

Stage-1/Stage-2 motif runs produce two PDB views:
- ``samples_eval/``: single scaffold chain (``sample_chain_name``, often ``A``,
  native ``B``/``P``/``F`` for vanilla ProteinZen-main) — parsed by PMPNN/RMSD.
- ``samples/``: viz PDB; motifs may appear on separate chains (e.g. ``B``) when ``output_motif_chains=true``.

``motif_idx`` from sampling is scaffold-relative (same numbering as the eval chain), but raw
``asym_id`` chain labels reflect the viz structure. This module normalizes eval fields so
``fixed_seq_*`` and ``pmpnn_fixed_pos_dict.jsonl`` always use the scaffold chain.

Stage-2 conditioning uses separate fields (``stage2_input_pdb``, ``stage2_pdb_contigs``, …) in the
layout manifest and task YAML; do not put viz-chain contigs into eval ``motif_reference_residues``.

See README.md section "Eval metadata vs Stage-2 conditioning metadata".
"""

from __future__ import annotations

from typing import Any

EVAL_MOTIF_FIELD_KEYS = (
    "rmsd_reference_chain",
    "rmsd_folded_chain",
    "motif_reference_residues",
    "motif_folded_residues",
    "motif_bb_reference_residues",
    "motif_bb_folded_residues",
    "motif_native_keys",
)


def unique_fixed_seq_chain(metadata: dict[str, Any]) -> str | None:
    """Return the sole ``fixed_seq_chain`` id when it is unambiguous."""
    chains = metadata.get("fixed_seq_chain")
    if isinstance(chains, str) and chains:
        return chains
    if isinstance(chains, list):
        unique = [str(chain) for chain in dict.fromkeys(chains) if chain]
        if len(unique) == 1:
            return unique[0]
    return None


def eval_scaffold_chain(metadata: dict[str, Any]) -> str:
    """Eval scaffold chain: ``sample_chain_name``, else unique ``fixed_seq_chain``, else ``A``."""
    named = metadata.get("sample_chain_name")
    if isinstance(named, str) and named:
        return named
    unique = unique_fixed_seq_chain(metadata)
    if unique:
        return unique
    return "A"


def _sample_chain_name(metadata: dict[str, Any]) -> str:
    return eval_scaffold_chain(metadata)


def normalize_eval_fixed_positions(
    metadata: dict[str, Any],
    *,
    sample_chain_name: str | None = None,
) -> dict[str, Any]:
    """Force eval fixed-position chain labels onto the scaffold chain."""
    chain = sample_chain_name or _sample_chain_name(metadata)
    for key in ("fixed_bb_chain", "fixed_seq_chain"):
        positions = metadata.get(key.replace("_chain", "_res_idx"))
        if isinstance(positions, list):
            metadata[key] = [chain] * len(positions)
    return metadata


def build_eval_motif_fields(
    metadata: dict[str, Any],
    *,
    sample_chain_name: str | None = None,
) -> dict[str, Any]:
    chain = sample_chain_name or _sample_chain_name(metadata)
    fixed_seq = list(metadata.get("fixed_seq_res_idx") or [])
    fixed_bb = list(metadata.get("fixed_bb_res_idx") or [])
    return {
        "rmsd_reference_chain": chain,
        "rmsd_folded_chain": "A",
        "motif_reference_residues": fixed_seq,
        "motif_folded_residues": fixed_seq,
        "motif_bb_reference_residues": fixed_bb,
        "motif_bb_folded_residues": fixed_bb,
    }


def apply_eval_sample_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Normalize eval fixed positions and fill motif/RMSD fields for downstream."""
    chain = eval_scaffold_chain(metadata)
    metadata["sample_chain_name"] = chain
    normalize_eval_fixed_positions(metadata, sample_chain_name=chain)

    preserved: dict[str, Any] = {}
    if metadata.get("source_layout_id"):
        for key in EVAL_MOTIF_FIELD_KEYS:
            if key in metadata and metadata[key] is not None:
                preserved[key] = metadata[key]

    metadata.update(build_eval_motif_fields(metadata, sample_chain_name=chain))
    metadata.update(preserved)

    source_residue_keys = metadata.get("source_residue_keys")
    if isinstance(source_residue_keys, list) and "motif_native_keys" not in preserved:
        motif_native_keys = [
            str(item["native_key"])
            for item in source_residue_keys
            if isinstance(item, dict) and "native_key" in item
        ]
        if motif_native_keys:
            metadata["motif_native_keys"] = motif_native_keys

    return metadata


def build_pmpnn_fixed_pos_dict(
    samples_metadata: dict[str, dict[str, Any]],
) -> dict[str, dict[str, list[int]]]:
    """Build ProteinMPNN fixed-position dict using eval scaffold chains only."""
    pmpnn_fixed_pos_dict: dict[str, dict[str, list[int]]] = {}
    for name, metadata in samples_metadata.items():
        entry: dict[str, list[int]] = {}
        for pos, pos_chain in zip(
            metadata.get("fixed_seq_res_idx") or [],
            metadata.get("fixed_seq_chain") or [],
        ):
            entry.setdefault(str(pos_chain), []).append(int(pos))
        if not entry:
            entry[_sample_chain_name(metadata)] = []
        pmpnn_fixed_pos_dict[name] = entry
    return pmpnn_fixed_pos_dict


def rebuild_pmpnn_from_eval_metadata(
    samples_metadata: dict[str, dict[str, Any]],
) -> dict[str, dict[str, list[int]]]:
    """Rebuild PMPNN dict after normalizing eval metadata in-place copies."""
    normalized: dict[str, dict[str, Any]] = {}
    for name, metadata in samples_metadata.items():
        sample_metadata = dict(metadata)
        apply_eval_sample_metadata(sample_metadata)
        normalized[name] = sample_metadata
    return build_pmpnn_fixed_pos_dict(normalized)
