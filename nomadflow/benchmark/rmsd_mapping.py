from __future__ import annotations


from nomadflow.benchmark.motif_group_annotation import apply_source_residue_key_repair


def _unique_ints(values):
    out = []
    seen = set()
    for val in values:
        intval = int(val)
        if intval in seen:
            continue
        seen.add(intval)
        out.append(intval)
    return out


def _infer_reference_chain(sample_metadata):
    explicit = sample_metadata.get("rmsd_reference_chain")
    if explicit:
        return str(explicit)
    source_keys = sample_metadata.get("source_residue_keys")
    if isinstance(source_keys, list):
        chains = {
            str(item.get("output_chain"))
            for item in source_keys
            if isinstance(item, dict) and item.get("output_chain") is not None
        }
        if len(chains) == 1:
            return next(iter(chains))
    return str(
        sample_metadata.get("sample_chain_name")
        or sample_metadata.get("fixed_seq_chain")
        or "A"
    )


def _infer_folded_chain(sample_metadata, reference_chain):
    explicit = sample_metadata.get("rmsd_folded_chain")
    if explicit:
        return str(explicit)
    return str(reference_chain or "A")


def _native_key(chain, residue):
    return f"{str(chain)}{int(residue)}"


def _expand_group_segment(segment_spec):
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


def build_rmsd_mapping(sample_metadata):
    sample_metadata = apply_source_residue_key_repair(dict(sample_metadata))
    ref_chain = _infer_reference_chain(sample_metadata)
    folded_chain = _infer_folded_chain(sample_metadata, ref_chain)
    ref_res = sample_metadata.get("motif_reference_residues")
    folded_res = sample_metadata.get("motif_folded_residues")
    bb_ref_res = sample_metadata.get("motif_bb_reference_residues")
    bb_folded_res = sample_metadata.get("motif_bb_folded_residues")
    if ref_res is None:
        ref_res = sample_metadata.get("fixed_seq_res_idx", [])
    if folded_res is None:
        folded_res = ref_res
    if bb_ref_res is None:
        bb_ref_res = sample_metadata.get("fixed_bb_res_idx", [])
    if bb_folded_res is None:
        bb_folded_res = bb_ref_res

    return {
        "reference_chain": ref_chain,
        "folded_chain": folded_chain,
        "motif_reference_residues": _unique_ints(ref_res),
        "motif_folded_residues": _unique_ints(folded_res),
        "motif_bb_reference_residues": _unique_ints(bb_ref_res),
        "motif_bb_folded_residues": _unique_ints(bb_folded_res),
    }


def build_motif_group_mappings(sample_metadata, mapping):
    motif_native_keys = sample_metadata.get("motif_native_keys") or []
    ref_res = mapping["motif_reference_residues"]
    folded_res = mapping["motif_folded_residues"]
    bb_ref_set = set(mapping["motif_bb_reference_residues"])
    bb_folded_set = set(mapping["motif_bb_folded_residues"])
    native_to_ref = {
        str(native_key): int(ref_idx)
        for native_key, ref_idx in zip(motif_native_keys, ref_res, strict=False)
    }
    native_to_folded = {
        str(native_key): int(folded_idx)
        for native_key, folded_idx in zip(motif_native_keys, folded_res, strict=False)
    }

    grouped = {}
    source_keys = sample_metadata.get("source_residue_keys")
    if isinstance(source_keys, list):
        for item in source_keys:
            if not isinstance(item, dict):
                continue
            group_id = item.get("group_id")
            if group_id is None:
                continue
            group = grouped.setdefault(
                str(group_id),
                {"group_id": str(group_id), "native_keys": [], "reference_residues": [], "folded_residues": []},
            )
            native_key = item.get("native_key")
            if native_key is None:
                native_chain = item.get("native_chain")
                native_res_idx = item.get("native_res_idx")
                if native_chain is not None and native_res_idx is not None:
                    native_key = _native_key(native_chain, native_res_idx)
            if native_key is not None:
                group["native_keys"].append(str(native_key))
            output_res_idx = item.get("output_res_idx")
            if output_res_idx is not None:
                group["reference_residues"].append(int(output_res_idx))
            if native_key is not None and str(native_key) in native_to_folded:
                group["folded_residues"].append(native_to_folded[str(native_key)])

    if not grouped:
        motif_groups = sample_metadata.get("motif_groups")
        if isinstance(motif_groups, list):
            for index, group_segments in enumerate(motif_groups, start=1):
                if not isinstance(group_segments, list):
                    continue
                native_keys = []
                for segment_spec in group_segments:
                    native_keys.extend(_expand_group_segment(segment_spec))
                grouped[str(index)] = {
                    "group_id": str(index),
                    "native_keys": native_keys,
                    "reference_residues": [native_to_ref[key] for key in native_keys if key in native_to_ref],
                    "folded_residues": [native_to_folded[key] for key in native_keys if key in native_to_folded],
                }

    out = []
    for group in grouped.values():
        ref_unique = _unique_ints(group["reference_residues"])
        folded_unique = _unique_ints(group["folded_residues"])
        if not ref_unique and not folded_unique:
            continue
        out.append(
            {
                "group_id": str(group["group_id"]),
                "native_keys": list(dict.fromkeys(group["native_keys"])),
                "reference_residues": ref_unique,
                "folded_residues": folded_unique,
                "bb_reference_residues": [res for res in ref_unique if res in bb_ref_set],
                "bb_folded_residues": [res for res in folded_unique if res in bb_folded_set],
            }
        )
    return out


def pose_indices_from_residue_pairs(pairs, pdb2pose) -> list[int]:
    """Map (resid, chain) pairs to deduplicated pose indices via ``pdb2pose(chain, resid)``."""
    indices: list[int] = []
    seen: set[int] = set()
    for resid, chain in pairs:
        pose_idx = int(pdb2pose(chain, int(resid)))
        if pose_idx > 0 and pose_idx not in seen:
            seen.add(pose_idx)
            indices.append(pose_idx)
    return indices
