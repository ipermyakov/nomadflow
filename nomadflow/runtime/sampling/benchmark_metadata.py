from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import omegaconf

from nomadflow.benchmark.motif_group_annotation import apply_source_residue_key_repair

if TYPE_CHECKING:
    from nomadflow.runtime.sampling.motif_scaffolding import MotifScaffoldingTask


EVAL_TASK_METADATA_KEYS = (
    "source_layout_id",
    "source_stage1_task_name",
    "rmsd_reference_chain",
    "rmsd_folded_chain",
    "motif_reference_residues",
    "motif_folded_residues",
    "motif_bb_reference_residues",
    "motif_bb_folded_residues",
    "motif_native_keys",
    "stage2_input_pdb",
    "stage2_pdb_contigs",
    "stage2_motif_groups",
    "stage2_sample_chain_name",
)


def _to_json_safe(value: Any) -> Any:
    if omegaconf.OmegaConf.is_config(value):
        value = omegaconf.OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {key: _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def build_motif_scaffolding_task_metadata(task: MotifScaffoldingTask) -> dict[str, Any]:
    total_length = (
        str(task.total_len[0])
        if len(task.total_len) == 1
        else f"{task.total_len[0]}-{task.total_len[1]}"
    )
    metadata: dict[str, Any] = {
        "native_pdb": task.native_pdb,
        "pdb_contigs": task.pdb_contigs,
        "contigs_idx": task.contigs_idx,
        "total_length": total_length,
        "multi_motif_se3": task.multi_motif_se3,
        "multi_motif_partition_mode": task.multi_motif_partition_mode,
        "multi_motif_max_groups": task.multi_motif_max_groups,
        "multi_motif_se3_trans_std": task.multi_motif_se3_trans_std,
        "motif_t_lock": task.motif_t_lock,
        "sample_chain_name": task.sample_chain_name,
        "num_samples": task.num_samples,
        "name": task.kwargs.get("name", task.task_name),
    }
    motif_groups = task.kwargs.get("motif_groups")
    if motif_groups is not None:
        metadata["motif_groups"] = motif_groups
    for key in EVAL_TASK_METADATA_KEYS:
        if key in task.kwargs:
            metadata[key] = task.kwargs[key]
    return _to_json_safe(metadata)


def normalize_eval_source_residue_keys(
    sample_metadata: dict[str, Any],
    sample_chain_name: str,
) -> dict[str, Any]:
    """Rewrite source_residue_keys output_chain labels for eval/downstream."""
    source_residue_keys = sample_metadata.get("source_residue_keys")
    if not isinstance(source_residue_keys, list):
        return sample_metadata

    normalized_keys: list[dict[str, Any]] = []
    motif_group_assignments: dict[str, int] = {}
    for entry in source_residue_keys:
        if not isinstance(entry, dict):
            continue
        normalized = dict(entry)
        output_res_idx = normalized.get("output_res_idx")
        normalized["output_chain"] = sample_chain_name
        if output_res_idx is not None and "group_id" in normalized:
            motif_group_assignments[
                f"{sample_chain_name}:{output_res_idx}"
            ] = int(normalized["group_id"])
        normalized_keys.append(normalized)

    result = dict(sample_metadata)
    result["source_residue_keys"] = normalized_keys
    if motif_group_assignments:
        result["motif_group_assignments"] = motif_group_assignments
    return apply_source_residue_key_repair(result)


def build_motif_scaffolding_sample_metadata(
    source_residue_to_output: dict[tuple[str, int], tuple[int, int, bool]],
    motif_group_by_copy_key: dict[tuple[int, int], int] | None,
    sample_length: int,
) -> dict[str, Any]:
    """Build per-sample metadata before inference (no output chain/residue yet)."""
    source_residue_keys: list[dict[str, Any]] = []
    for (native_chain, native_res_idx), (
        asym_id,
        internal_res_idx,
        is_unindexed,
    ) in source_residue_to_output.items():
        lookup_res_idx = int(
            internal_res_idx + (sample_length if is_unindexed else 0)
        )
        lookup_key = (int(asym_id), lookup_res_idx)
        entry: dict[str, Any] = {
            "native_key": f"{native_chain}{native_res_idx}",
            "native_chain": native_chain,
            "native_res_idx": int(native_res_idx),
            "lookup_asym_id": int(asym_id),
            "lookup_res_idx": lookup_res_idx,
            "is_unindexed": bool(is_unindexed),
        }
        if motif_group_by_copy_key is not None:
            group_id = motif_group_by_copy_key.get(lookup_key)
            if group_id is not None:
                entry["group_id"] = int(group_id)
        source_residue_keys.append(entry)

    return {
        "source_residue_keys": source_residue_keys,
        "sample_length": int(sample_length),
    }


def _as_numpy(array):
    if hasattr(array, "numpy"):
        return array.numpy(force=True)
    return np.asarray(array)


def finalize_motif_sample_metadata(
    token_data,
    motif_idx,
    raw_sample_metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Resolve output residue indices from model motif_idx after forward pass."""
    if raw_sample_metadata is None:
        return None

    token_is_copy = _as_numpy(token_data["is_copy"]).astype(bool)
    asym_ids = _as_numpy(token_data["asym_id"])
    res_idxs = _as_numpy(token_data["res_idx"])
    motif_idx_arr = _as_numpy(motif_idx)

    copy_key_to_output: dict[tuple[int, int], int] = {}
    for token_idx in range(len(token_is_copy)):
        if not token_is_copy[token_idx]:
            continue
        key = (int(asym_ids[token_idx]), int(res_idxs[token_idx]))
        copy_key_to_output[key] = int(motif_idx_arr[token_idx])

    finalized_keys: list[dict[str, Any]] = []
    for entry in raw_sample_metadata.get("source_residue_keys", []):
        lookup_key = (int(entry["lookup_asym_id"]), int(entry["lookup_res_idx"]))
        if lookup_key not in copy_key_to_output:
            raise ValueError(
                "Could not resolve output residue for native key "
                f"{entry['native_key']} with lookup key {lookup_key}"
            )
        finalized: dict[str, Any] = {
            "native_key": entry["native_key"],
            "native_chain": entry["native_chain"],
            "native_res_idx": entry["native_res_idx"],
            "output_asym_id": int(entry["lookup_asym_id"]),
            "output_res_idx": copy_key_to_output[lookup_key] + 1,
            "is_unindexed": entry["is_unindexed"],
        }
        if "group_id" in entry:
            finalized["group_id"] = entry["group_id"]
        finalized_keys.append(finalized)

    return apply_source_residue_key_repair(
        {
            "source_residue_keys": finalized_keys,
            "sample_length": raw_sample_metadata["sample_length"],
        }
    )


def attach_output_chains_to_sample_metadata(
    sample_metadata: dict[str, Any],
    chain_mapping: dict[int, str],
) -> dict[str, Any]:
    """Add output chain letters and motif_group_assignments after PDB write."""
    motif_group_assignments: dict[str, int] = {}
    finalized_keys: list[dict[str, Any]] = []

    for entry in sample_metadata.get("source_residue_keys", []):
        output_chain = chain_mapping[int(entry["output_asym_id"])]
        finalized = {
            "native_key": entry["native_key"],
            "native_chain": entry["native_chain"],
            "native_res_idx": entry["native_res_idx"],
            "output_chain": output_chain,
            "output_res_idx": entry["output_res_idx"],
            "is_unindexed": entry["is_unindexed"],
        }
        if "group_id" in entry:
            finalized["group_id"] = entry["group_id"]
            motif_group_assignments[
                f"{output_chain}:{entry['output_res_idx']}"
            ] = int(entry["group_id"])
        finalized_keys.append(finalized)

    result: dict[str, Any] = {
        "source_residue_keys": finalized_keys,
        "sample_length": sample_metadata["sample_length"],
    }
    if motif_group_assignments:
        result["motif_group_assignments"] = motif_group_assignments
    return result
