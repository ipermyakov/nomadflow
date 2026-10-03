import copy
import os
import time
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch

from nomadflow.boltz.data import const
from nomadflow.boltz.data.types import SamplingResidue, Structure
from nomadflow.data.featurize.sampling import construct_atoms, update_structure
from nomadflow.data.featurize.tokenize import Tokenized
from nomadflow.data.write.pdb import to_pdb
from nomadflow.runtime.intra_rigid_conect import process_pdb_lines


_PROFILE_PDB_WRITE = os.getenv("NOMADFLOW_PROFILE_PDB_WRITE", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _maybe_print_timing(label: str, start_time: float) -> None:
    if _PROFILE_PDB_WRITE:
        elapsed = time.perf_counter() - start_time
        print(f"[pdb-profile] {label}: {elapsed:.4f}s")


def prediction_frame_to_tokenized(frame) -> Tokenized:
    struct = Structure(**frame["structure"])
    return Tokenized(
        tokens=frame["tokens"],
        rigids=frame["rigids"],
        bonds=frame["bonds"],
        structure=struct,
    )


def tokenized_with_updated_tensor7(data: Tokenized, rigid_tensor7) -> Tokenized:
    rigids = data.rigids.copy()
    tensor7 = torch.as_tensor(rigid_tensor7).detach().cpu().numpy()
    tensor7 = tensor7[: len(rigids)].astype(np.float32, copy=False)
    rigids["tensor7"] = tensor7
    return replace(data, rigids=rigids)


def _ensure_sampling_residues(struct: Structure) -> Structure:
    if "is_copy" in struct.residues.dtype.names:
        return struct

    residues = np.zeros(struct.residues.shape, dtype=SamplingResidue)
    for field_name in struct.residues.dtype.names:
        residues[field_name] = struct.residues[field_name]
    residues["is_copy"] = False
    return replace(struct, residues=residues)


def _prune_structure_to_tokenized_residues(struct: Structure, data: Tokenized) -> Structure:
    token_residue_ids = {int(token["res_idx"]) for token in data.tokens}
    if not token_residue_ids:
        return struct

    selected_residues = []
    selected_chains = []
    for chain in struct.chains[struct.mask]:
        res_start = int(chain["res_idx"])
        res_end = res_start + int(chain["res_num"])
        chain_residues = [
            residue
            for residue in struct.residues[res_start:res_end]
            if int(residue["res_idx"]) in token_residue_ids
        ]
        if not chain_residues:
            continue

        chain_copy = chain.copy()
        chain_copy["res_idx"] = len(selected_residues)
        chain_copy["res_num"] = len(chain_residues)
        chain_copy["atom_idx"] = int(chain_residues[0]["atom_idx"])
        chain_copy["atom_num"] = int(sum(int(residue["atom_num"]) for residue in chain_residues))
        selected_chains.append(tuple(chain_copy.tolist()))
        selected_residues.extend(tuple(residue.tolist()) for residue in chain_residues)

    if len(selected_residues) == len(struct.residues) and len(selected_chains) == int(struct.mask.sum()):
        return struct

    return replace(
        struct,
        residues=np.array(selected_residues, dtype=struct.residues.dtype),
        chains=np.array(selected_chains, dtype=struct.chains.dtype),
        mask=np.ones(len(selected_chains), dtype=bool),
    )


def _next_free_pdb_chain_name(chains: np.ndarray) -> str:
    used_names = {str(chain["name"]) for chain in chains}
    for chain_name in "VABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789":
        if chain_name not in used_names:
            return chain_name
    raise ValueError("output has too many chains to be represented in .pdb format")


def _adapt_motif_group_aggregators_for_pdb(data: Tokenized) -> Tokenized:
    """Render virtual group markers as glycine residues in their own chain.

    Aggregator markers are represented by one rigid in training data. The PDB
    path for standard residues expects three coarse-grain frames, so we duplicate
    the marker rigid and let the existing glycine reconstruction path render it.

    After sampling, ``lmod`` overwrites ``token_idx`` on motif copy tokens with the
    assigned scaffold residue index so ``construct_atoms`` can paste motif
    sequence/coords onto chain A. That assignment must survive this remapping:
    unindexed copy tokens keep their assigned scaffold index in ``token_idx``
    rather than receiving a dense PDB-local reindex.
    """
    has_marker_fields = (
        "is_motif_group_aggregator" in data.tokens.dtype.names
        and "is_motif_group_aggregator" in data.rigids.dtype.names
    )
    if not has_marker_fields:
        return data

    marker_mask = data.tokens["is_motif_group_aggregator"]
    if not bool(marker_mask.any()):
        return data

    struct = _ensure_sampling_residues(data.structure)
    marker_count = int(marker_mask.sum())
    non_marker_tokens = data.tokens[~marker_mask]
    marker_tokens = data.tokens[marker_mask]
    ordered_source_tokens = np.concatenate([non_marker_tokens, marker_tokens]).copy()

    old_to_new_token_idx: dict[int, int] = {}
    rigid_blocks = []
    next_rigid_idx = 0
    gly_res_type = const.token_ids["GLY"]
    marker_chain_asym_id = (
        max(
            [int(chain["asym_id"]) for chain in struct.chains]
            + [int(token["asym_id"]) for token in data.tokens]
            + [-1]
        )
        + 1
    )
    next_marker_res_idx = (
        max(
            [int(residue["res_idx"]) for residue in struct.residues]
            + [int(token["res_idx"]) for token in non_marker_tokens]
            + [-1]
        )
        + 1
    )
    marker_res_idx_by_old_token_idx = {
        int(token["token_idx"]): next_marker_res_idx + marker_offset
        for marker_offset, token in enumerate(marker_tokens)
    }

    for new_token_idx, token in enumerate(ordered_source_tokens):
        old_token_idx = int(token["token_idx"])
        old_rigid_idx = int(token["rigid_idx"])
        old_rigid_num = int(token["rigid_num"])
        is_marker = bool(token["is_motif_group_aggregator"])
        # After lmod, unindexed copies store assigned scaffold res_idx here — not a
        # tokenizer index. Do not use that value as an old→new remap key (it can
        # collide with real token indices) and do not overwrite it below.
        preserve_assigned_scaffold_idx = (
            (not is_marker)
            and bool(token["is_copy"])
            and bool(token["is_unindexed"])
        )
        if not preserve_assigned_scaffold_idx:
            old_to_new_token_idx[old_token_idx] = new_token_idx

        if is_marker:
            source_rigid = data.rigids[old_rigid_idx : old_rigid_idx + max(old_rigid_num, 1)]
            if len(source_rigid) == 0:
                raise ValueError("motif group aggregator token does not reference a rigid")
            rigid_block = np.repeat(source_rigid[:1].copy(), 3)

            token["res_idx"] = marker_res_idx_by_old_token_idx[old_token_idx]
            token["res_type"] = gly_res_type
            token["res_name"] = "GLY"
            token["rigid_num"] = 3
            token["sym_id"] = marker_chain_asym_id
            token["asym_id"] = marker_chain_asym_id
            token["entity_id"] = marker_chain_asym_id
            token["mol_type"] = const.chain_type_ids["PROTEIN"]
            token["resolved_mask"] = bool(source_rigid["is_present"].any())
            token["center_coords"] = source_rigid[0]["tensor7"][4:]
            token["is_copy"] = False
            token["is_unindexed"] = True
            token["is_atomized"] = False
            token["seq_noising_mask"] = False
            token["is_motif_group_aggregator"] = False

            rigid_block["sidechain_idx"] = np.arange(3, dtype=rigid_block["sidechain_idx"].dtype)
            rigid_block["is_atomized"] = False
            rigid_block["element"] = -1
            rigid_block["charge"] = 0
            rigid_block["is_present"] = bool(source_rigid["is_present"].any())
            rigid_block["rigids_noising_mask"] = True
            rigid_block["num_real_input_axes"] = 0
            rigid_block["is_motif_group_aggregator"] = False
        else:
            rigid_block = data.rigids[old_rigid_idx : old_rigid_idx + old_rigid_num].copy()

        if not preserve_assigned_scaffold_idx:
            token["token_idx"] = new_token_idx
        token["rigid_idx"] = next_rigid_idx
        if len(rigid_block):
            rigid_block["token_idx"] = int(token["token_idx"])
            rigid_block["rigid_idx"] = np.arange(
                next_rigid_idx,
                next_rigid_idx + len(rigid_block),
                dtype=rigid_block["rigid_idx"].dtype,
            )
        rigid_blocks.append(rigid_block)
        next_rigid_idx += len(rigid_block)

    if rigid_blocks:
        rigids = np.concatenate(rigid_blocks, axis=0)
    else:
        rigids = data.rigids[:0].copy()

    bonds = data.bonds.copy()
    if len(bonds):
        marker_old_token_idxs = set(marker_res_idx_by_old_token_idx)
        keep_mask = ~np.isin(bonds["token_1"], list(marker_old_token_idxs)) & ~np.isin(
            bonds["token_2"], list(marker_old_token_idxs)
        )
        # Drop endpoints that are not remappable (e.g. bonds to copy tokens whose
        # token_idx was overwritten to an assigned scaffold index by lmod).
        keep_mask = keep_mask & np.array(
            [
                int(t1) in old_to_new_token_idx and int(t2) in old_to_new_token_idx
                for t1, t2 in zip(bonds["token_1"], bonds["token_2"])
            ],
            dtype=bool,
        )
        bonds = bonds[keep_mask]
        if len(bonds):
            bonds["token_1"] = np.array(
                [old_to_new_token_idx[int(token_idx)] for token_idx in bonds["token_1"]],
                dtype=bonds["token_1"].dtype,
            )
            bonds["token_2"] = np.array(
                [old_to_new_token_idx[int(token_idx)] for token_idx in bonds["token_2"]],
                dtype=bonds["token_2"].dtype,
            )

    marker_residues = np.zeros(marker_count, dtype=struct.residues.dtype)
    marker_residues["name"] = "GLY"
    marker_residues["res_type"] = gly_res_type
    marker_residues["res_idx"] = list(marker_res_idx_by_old_token_idx.values())
    marker_residues["atom_idx"] = 0
    marker_residues["atom_num"] = len(const.ref_atoms["GLY"])
    marker_residues["atom_center"] = 1
    marker_residues["atom_disto"] = 1
    marker_residues["is_standard"] = True
    marker_residues["is_present"] = True
    marker_residues["is_copy"] = False

    marker_chain = np.zeros(1, dtype=struct.chains.dtype)
    marker_chain["name"] = _next_free_pdb_chain_name(struct.chains)
    marker_chain["mol_type"] = const.chain_type_ids["PROTEIN"]
    marker_chain["entity_id"] = marker_chain_asym_id
    marker_chain["sym_id"] = marker_chain_asym_id
    marker_chain["asym_id"] = marker_chain_asym_id
    marker_chain["atom_idx"] = 0
    marker_chain["atom_num"] = 0
    marker_chain["res_idx"] = len(struct.residues)
    marker_chain["res_num"] = marker_count
    marker_chain["cyclic_period"] = 0

    adapted_struct = replace(
        struct,
        residues=np.concatenate([struct.residues, marker_residues]),
        chains=np.concatenate([struct.chains, marker_chain]),
        mask=np.concatenate([struct.mask.astype(bool), np.array([True])]),
    )
    return replace(data, tokens=ordered_source_tokens, rigids=rigids, bonds=bonds, structure=adapted_struct)


def _copy_structure_subset(
    struct: Structure,
    residues_by_chain: list[tuple[np.void, list[tuple[np.void, int, bool]]]],
) -> Structure:
    selected_atoms = []
    selected_residues = []
    selected_chains = []
    atom_idx = 0

    for source_chain, residue_specs in residues_by_chain:
        if not residue_specs:
            continue

        chain_copy = source_chain.copy()
        chain_copy["res_idx"] = len(selected_residues)
        chain_copy["res_num"] = len(residue_specs)
        chain_copy["atom_idx"] = atom_idx

        for source_residue, new_res_idx, is_copy in residue_specs:
            residue_copy = source_residue.copy()
            atom_start = int(source_residue["atom_idx"])
            atom_end = atom_start + int(source_residue["atom_num"])
            residue_atoms = struct.atoms[atom_start:atom_end].copy()

            residue_copy["res_idx"] = new_res_idx
            residue_copy["atom_idx"] = atom_idx
            residue_copy["is_copy"] = is_copy

            selected_atoms.extend(tuple(atom.tolist()) for atom in residue_atoms)
            selected_residues.append(tuple(residue_copy.tolist()))
            atom_idx += len(residue_atoms)

        chain_copy["atom_num"] = atom_idx - int(chain_copy["atom_idx"])
        selected_chains.append(tuple(chain_copy.tolist()))

    return replace(
        struct,
        atoms=np.array(selected_atoms, dtype=struct.atoms.dtype),
        bonds=np.array([], dtype=struct.bonds.dtype),
        residues=np.array(selected_residues, dtype=struct.residues.dtype),
        chains=np.array(selected_chains, dtype=struct.chains.dtype),
        connections=np.array([], dtype=struct.connections.dtype),
        interfaces=np.array([], dtype=struct.interfaces.dtype),
        mask=np.ones(len(selected_chains), dtype=bool),
    )


def adapt_tokenized_for_train_viz_copy_semantics(data: Tokenized) -> Tokenized:
    """Build a sampling-like template for train-viz copy rendering.

    Training tokenization appends copy tokens, but keeps their original `res_idx`.
    For PDB export that causes the writer path to treat copy tokens as in-place
    replacements for main-chain residues. Here we rewrite copy tokens to point to
    appended residue indices and build a matching structure template whose residue
    order still mirrors the token/rigid order expected by `construct_atoms()` and
    `update_structure()`.
    """
    if not np.any(data.tokens["is_copy"]):
        return data

    struct = _ensure_sampling_residues(copy.deepcopy(data.structure))
    tokens = data.tokens.copy()
    main_tokens = tokens[~tokens["is_copy"]]

    residue_to_source = {}
    residue_to_chain = {}
    for chain in struct.chains[struct.mask]:
        res_start = int(chain["res_idx"])
        res_end = res_start + int(chain["res_num"])
        for residue in struct.residues[res_start:res_end]:
            residue_to_source[int(residue["res_idx"])] = residue
            residue_to_chain[int(residue["res_idx"])] = chain

    main_residue_ids = {int(token["res_idx"]) for token in main_tokens}
    residues_by_chain: list[tuple[np.void, list[tuple[np.void, int, bool]]]] = []
    for chain in struct.chains[struct.mask]:
        res_start = int(chain["res_idx"])
        res_end = res_start + int(chain["res_num"])
        chain_residues = []
        for residue in struct.residues[res_start:res_end]:
            residue_id = int(residue["res_idx"])
            if residue_id in main_residue_ids:
                chain_residues.append((residue, residue_id, False))
        if chain_residues:
            residues_by_chain.append((chain, chain_residues))

    next_copy_res_idx = (
        max((int(token["res_idx"]) for token in main_tokens), default=-1) + 1
    )
    copy_residue_mapping: dict[int, int] = {}
    copy_chain_buckets: dict[int, tuple[np.void, list[tuple[np.void, int, bool]]]] = {}

    for token_idx, token in enumerate(tokens):
        if not token["is_copy"]:
            continue

        source_res_idx = int(token["res_idx"])
        source_residue = residue_to_source[source_res_idx]
        source_chain = residue_to_chain[source_res_idx]

        if source_res_idx not in copy_residue_mapping:
            copy_residue_mapping[source_res_idx] = next_copy_res_idx
            next_copy_res_idx += 1

            chain_key = int(source_chain["asym_id"])
            if chain_key not in copy_chain_buckets:
                copy_chain_buckets[chain_key] = (source_chain, [])
            copy_chain_buckets[chain_key][1].append(
                (source_residue, copy_residue_mapping[source_res_idx], True)
            )

        tokens[token_idx]["res_idx"] = copy_residue_mapping[source_res_idx]

    residues_by_chain.extend(copy_chain_buckets.values())
    viz_struct = _copy_structure_subset(struct, residues_by_chain)
    return replace(data, tokens=tokens, structure=viz_struct)


def scaffold_token_mask(tokens: np.ndarray) -> np.ndarray:
    """Mask of scaffold tokens (exclude copy conditioning and aggregator markers)."""
    keep = ~tokens["is_copy"]
    if "is_motif_group_aggregator" in tokens.dtype.names:
        keep &= ~tokens["is_motif_group_aggregator"]
    return keep


def filter_tokenized_for_eval_pdb(data: Tokenized) -> Tokenized:
    """Keep scaffold tokens only; drop copy and aggregator markers."""
    keep_mask = scaffold_token_mask(data.tokens)
    if keep_mask.all():
        return data

    kept_tokens = data.tokens[keep_mask].copy()
    rigid_blocks = []
    next_rigid_idx = 0
    for new_token_idx, token in enumerate(kept_tokens):
        old_rigid_idx = int(token["rigid_idx"])
        old_rigid_num = int(token["rigid_num"])
        rigid_block = data.rigids[old_rigid_idx : old_rigid_idx + old_rigid_num].copy()
        token["token_idx"] = new_token_idx
        token["rigid_idx"] = next_rigid_idx
        if len(rigid_block):
            rigid_block["token_idx"] = new_token_idx
            rigid_block["rigid_idx"] = np.arange(
                next_rigid_idx,
                next_rigid_idx + len(rigid_block),
                dtype=rigid_block["rigid_idx"].dtype,
            )
        rigid_blocks.append(rigid_block)
        next_rigid_idx += len(rigid_block)

    if rigid_blocks:
        rigids = np.concatenate(rigid_blocks, axis=0)
    else:
        rigids = data.rigids[:0].copy()

    kept_old_token_idxs = set(int(idx) for idx in np.where(keep_mask)[0])
    bonds = data.bonds.copy()
    if len(bonds):
        bond_keep = np.isin(bonds["token_1"], list(kept_old_token_idxs)) & np.isin(
            bonds["token_2"], list(kept_old_token_idxs)
        )
        bonds = bonds[bond_keep]
        old_to_new = {
            int(old_idx): new_idx
            for new_idx, old_idx in enumerate(np.where(keep_mask)[0])
        }
        if len(bonds):
            bonds["token_1"] = np.array(
                [old_to_new[int(token_idx)] for token_idx in bonds["token_1"]],
                dtype=bonds["token_1"].dtype,
            )
            bonds["token_2"] = np.array(
                [old_to_new[int(token_idx)] for token_idx in bonds["token_2"]],
                dtype=bonds["token_2"].dtype,
            )

    filtered = replace(data, tokens=kept_tokens, rigids=rigids, bonds=bonds)
    struct = _prune_structure_to_tokenized_residues(filtered.structure, filtered)
    return replace(filtered, structure=struct)


def extract_chain_structure(struct: Structure, chain_name: str) -> Structure:
    """Extract one chain from a rendered structure, preserving residue numbering."""
    struct = _ensure_sampling_residues(struct)
    for chain in struct.chains[struct.mask]:
        if str(chain["name"]) != chain_name:
            continue
        res_start = int(chain["res_idx"])
        res_end = res_start + int(chain["res_num"])
        has_is_copy = "is_copy" in struct.residues.dtype.names
        residue_specs = []
        for residue in struct.residues[res_start:res_end]:
            is_copy = bool(residue["is_copy"]) if has_is_copy else False
            residue_specs.append((residue, int(residue["res_idx"]), is_copy))
        return _copy_structure_subset(struct, [(chain, residue_specs)])
    raise ValueError(f"Chain {chain_name!r} not found in structure")


def render_eval_structure_for_pdb(
    data: Tokenized,
    *,
    sample_chain_name: str = "A",
    take_copy_structure: bool = True,
) -> Structure:
    """Single-chain Protpardelle-style structure for MPNN/ESMFold downstream eval."""
    struct = render_tokenized_structure_for_pdb(
        data,
        output_motif_chains=True,
        take_copy_structure=take_copy_structure,
    )
    return extract_chain_structure(struct, sample_chain_name)


def write_eval_tokenized_structure_pdb(
    path: str | Path,
    data: Tokenized,
    *,
    sample_chain_name: str = "A",
    take_copy_structure: bool = True,
) -> Structure:
    struct = render_eval_structure_for_pdb(
        data,
        sample_chain_name=sample_chain_name,
        take_copy_structure=take_copy_structure,
    )
    Path(path).write_text(
        to_pdb(struct, rename_chains=_needs_chain_renaming(struct)),
        encoding="utf-8",
    )
    return struct


def _apply_motif_chain_output_policy(struct: Structure, output_motif_chains: bool) -> Structure:
    if output_motif_chains:
        num_chains = len(struct.chains)
        struct.chains["asym_id"] = np.arange(num_chains)

        def get_next_free_chain_name(seen_names):
            # Prefer letter-only chain IDs for widest downstream compatibility.
            chain_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
            for chain_name in chain_alphabet:
                if chain_name not in seen_names:
                    return chain_name
            raise ValueError("output has too many chains to be represented in .pdb format")

        seen_names = []
        new_chain_names = []
        for chain in struct.chains:
            if chain["name"] not in seen_names:
                seen_names.append(chain["name"])
                new_chain_names.append(chain["name"])
            else:
                chain_rename = get_next_free_chain_name(seen_names)
                seen_names.append(chain_rename)
                new_chain_names.append(chain_rename)
        struct.chains["name"] = np.array(new_chain_names)

        for chain in struct.chains:
            res_start = chain["res_idx"]
            res_end = chain["res_idx"] + chain["res_num"]
            residues = struct.residues[res_start:res_end]
            residues["res_idx"] = np.arange(chain["res_num"])
        return struct

    seen_asym_id = []
    chain_mask = []
    for chain in struct.chains:
        if chain["asym_id"] not in seen_asym_id:
            seen_asym_id.append(chain["asym_id"])
            chain_mask.append(True)
        else:
            chain_mask.append(False)
    return replace(struct, mask=np.array(chain_mask, dtype=bool))


def _needs_chain_renaming(struct: Structure) -> bool:
    return any(len(str(chain["name"])) != 1 for chain in struct.chains[struct.mask])


def render_tokenized_structure_for_pdb(
    data: Tokenized,
    *,
    output_motif_chains: bool = False,
    take_copy_structure: bool = True,
) -> Structure:
    render_start = time.perf_counter()
    data = _adapt_motif_group_aggregators_for_pdb(data)
    struct = data.structure
    struct = _ensure_sampling_residues(struct)
    struct = _prune_structure_to_tokenized_residues(struct, data)
    struct = replace(struct, mask=np.ones_like(struct.mask.astype(bool)))
    construct_start = time.perf_counter()
    struct = construct_atoms(data, struct, take_copy_structure=take_copy_structure)
    _maybe_print_timing("render_tokenized_structure_for_pdb construct_atoms", construct_start)
    update_start = time.perf_counter()
    struct = update_structure(struct, data.rigids["tensor7"], copy_struct=False)
    _maybe_print_timing("render_tokenized_structure_for_pdb update_structure", update_start)
    policy_start = time.perf_counter()
    struct = _apply_motif_chain_output_policy(
        struct, output_motif_chains=output_motif_chains
    )
    _maybe_print_timing("render_tokenized_structure_for_pdb chain_policy", policy_start)
    _maybe_print_timing("render_tokenized_structure_for_pdb total", render_start)
    return struct


def _build_structure_template(
    data: Tokenized,
    *,
    take_copy_structure: bool = True,
) -> Structure:
    data = _adapt_motif_group_aggregators_for_pdb(data)
    struct = data.structure
    struct = _ensure_sampling_residues(struct)
    struct = _prune_structure_to_tokenized_residues(struct, data)
    struct = replace(struct, mask=np.ones_like(struct.mask.astype(bool)))
    return construct_atoms(data, struct, take_copy_structure=take_copy_structure)


def _clone_structure_for_update(template: Structure) -> Structure:
    # Avoid deepcopy overhead on dataclass metadata while keeping mutable arrays isolated.
    return replace(
        template,
        atoms=template.atoms.copy(),
        residues=template.residues.copy(),
        chains=template.chains.copy(),
        mask=template.mask.copy(),
    )


def _frame_template_key(frame: Tokenized) -> tuple:
    # Sequence can vary per frame, so include residue identity fields in cache key.
    tokens = frame.tokens
    return (
        len(tokens),
        tokens["res_idx"].tobytes(),
        tokens["res_type"].tobytes(),
        tokens["is_copy"].tobytes(),
    )


@dataclass(frozen=True)
class BatchedTrajectoryPack:
    rigid_tensor7: torch.Tensor
    frame_indices: list[int]
    num_rigids: list[int]
    device: str


def pack_tokenized_frames_for_cuda(
    frames: list[Tokenized],
    *,
    device: str = "cuda",
) -> BatchedTrajectoryPack | None:
    if device != "cuda" or not torch.cuda.is_available() or not frames:
        return None
    max_rigids = max(len(frame.rigids) for frame in frames)
    if max_rigids == 0:
        return None
    packed = torch.zeros((len(frames), max_rigids, 7), dtype=torch.float32, device=device)
    num_rigids = []
    for frame_idx, frame in enumerate(frames):
        # Structured-array field views can be non-contiguous with unsupported strides.
        # Materialize a contiguous float32 buffer before moving to CUDA.
        tensor7_np = np.ascontiguousarray(frame.rigids["tensor7"], dtype=np.float32)
        tensor7 = torch.from_numpy(tensor7_np).to(device=device)
        packed[frame_idx, : tensor7.shape[0]] = tensor7
        num_rigids.append(int(tensor7.shape[0]))
    return BatchedTrajectoryPack(
        rigid_tensor7=packed,
        frame_indices=list(range(len(frames))),
        num_rigids=num_rigids,
        device=device,
    )


def render_tokenized_structures_batched(
    frames: list[Tokenized],
    *,
    output_motif_chains: bool = False,
    take_copy_structure: bool = True,
    cuda_pack: BatchedTrajectoryPack | None = None,
) -> list[Structure]:
    if not frames:
        return []

    output_structs: list[Structure] = [None] * len(frames)  # type: ignore
    for frame_idx, frame in enumerate(frames):
        frame = _adapt_motif_group_aggregators_for_pdb(frame)
        # Build per-frame template to avoid cross-frame state reuse.
        # Correctness is prioritized over cache-based speedups here.
        template = _build_structure_template(
            frame,
            take_copy_structure=take_copy_structure,
        )

        struct = _clone_structure_for_update(template)
        if cuda_pack is not None:
            tensor7 = (
                cuda_pack.rigid_tensor7[frame_idx, : cuda_pack.num_rigids[frame_idx]]
                .detach()
                .cpu()
                .numpy()
            )
        else:
            tensor7 = frame.rigids["tensor7"]
        struct = update_structure(struct, tensor7, copy_struct=False)
        struct = _apply_motif_chain_output_policy(
            struct, output_motif_chains=output_motif_chains
        )
        output_structs[frame_idx] = struct

    return output_structs


def write_tokenized_structure_pdb(
    path: str | Path,
    data: Tokenized,
    *,
    output_motif_chains: bool = False,
    take_copy_structure: bool = True,
) -> Structure:
    struct = render_tokenized_structure_for_pdb(
        data,
        output_motif_chains=output_motif_chains,
        take_copy_structure=take_copy_structure,
    )
    Path(path).write_text(
        to_pdb(struct, rename_chains=_needs_chain_renaming(struct)),
        encoding="utf-8",
    )
    return struct


def build_multimodel_pdb_string(
    frames: Iterable[Tokenized],
    *,
    output_motif_chains: bool = False,
    take_copy_structure: bool = True,
) -> str:
    model_strs = []
    build_start = time.perf_counter()
    for idx, frame in enumerate(frames):
        frame_start = time.perf_counter()
        struct = render_tokenized_structure_for_pdb(
            frame,
            output_motif_chains=output_motif_chains,
            take_copy_structure=take_copy_structure,
        )
        pdb_str = to_pdb(struct, rename_chains=_needs_chain_renaming(struct))
        model_strs.append(f"MODEL        {idx}\n")
        model_body, _, _ = pdb_str.partition("END")
        model_strs.append(model_body)
        model_strs.append("ENDMDL       \n")
        _maybe_print_timing(f"build_multimodel_pdb_string frame={idx}", frame_start)
    model_strs.append("END\n")
    result = "".join(model_strs)
    _maybe_print_timing("build_multimodel_pdb_string total", build_start)
    return result


def build_multimodel_pdb_string_batched(
    frames: Iterable[Tokenized],
    *,
    output_motif_chains: bool = False,
    take_copy_structure: bool = True,
    use_cuda_batch: bool = False,
) -> str:
    frames_list = [_adapt_motif_group_aggregators_for_pdb(frame) for frame in frames]
    build_start = time.perf_counter()
    cuda_pack_start = time.perf_counter()
    cuda_pack = (
        pack_tokenized_frames_for_cuda(frames_list, device="cuda")
        if use_cuda_batch
        else None
    )
    _maybe_print_timing("build_multimodel_pdb_string_batched pack_cuda", cuda_pack_start)
    render_start = time.perf_counter()
    structs = render_tokenized_structures_batched(
        frames_list,
        output_motif_chains=output_motif_chains,
        take_copy_structure=take_copy_structure,
        cuda_pack=cuda_pack,
    )
    _maybe_print_timing("build_multimodel_pdb_string_batched render_structs", render_start)

    model_strs = []
    to_pdb_total = 0.0
    assemble_total = 0.0
    for idx, struct in enumerate(structs):
        to_pdb_start = time.perf_counter()
        pdb_str = to_pdb(struct, rename_chains=_needs_chain_renaming(struct))
        to_pdb_total += time.perf_counter() - to_pdb_start
        assemble_start = time.perf_counter()
        model_strs.append(f"MODEL        {idx}\n")
        model_body, _, _ = pdb_str.partition("END")
        model_strs.append(model_body)
        model_strs.append("ENDMDL       \n")
        assemble_total += time.perf_counter() - assemble_start
    model_strs.append("END\n")
    result = "".join(model_strs)
    if _PROFILE_PDB_WRITE:
        print(f"[pdb-profile] build_multimodel_pdb_string_batched to_pdb_total: {to_pdb_total:.4f}s")
        print(f"[pdb-profile] build_multimodel_pdb_string_batched assemble_total: {assemble_total:.4f}s")
    _maybe_print_timing("build_multimodel_pdb_string_batched total", build_start)
    return result


def write_multimodel_pdb(
    path: str | Path,
    frames: Iterable[Tokenized],
    *,
    output_motif_chains: bool = False,
    take_copy_structure: bool = True,
    add_intra_rigid_conects: bool = True,
) -> Tuple[Path, int, int]:
    total_start = time.perf_counter()
    output_path = Path(path)
    build_start = time.perf_counter()
    pdb_text = build_multimodel_pdb_string(
        frames,
        output_motif_chains=output_motif_chains,
        take_copy_structure=take_copy_structure,
    )
    _maybe_print_timing("write_multimodel_pdb build_multimodel_pdb_string", build_start)

    if add_intra_rigid_conects:
        process_start = time.perf_counter()
        lines = pdb_text.splitlines()
        lines, total_conects, model_count = process_pdb_lines(lines)
        _maybe_print_timing("write_multimodel_pdb process_pdb_lines", process_start)
        write_start = time.perf_counter()
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        _maybe_print_timing("write_multimodel_pdb write_text(processed)", write_start)
    else:
        total_conects = 0
        model_count = pdb_text.count("MODEL        ")
        write_start = time.perf_counter()
        output_path.write_text(pdb_text, encoding="utf-8")
        _maybe_print_timing("write_multimodel_pdb write_text(raw)", write_start)
    _maybe_print_timing("write_multimodel_pdb total", total_start)
    return output_path, total_conects, model_count


def write_multimodel_pdb_batched(
    path: str | Path,
    frames: Iterable[Tokenized],
    *,
    output_motif_chains: bool = False,
    take_copy_structure: bool = True,
    add_intra_rigid_conects: bool = True,
    use_cuda_batch: bool = True,
) -> Tuple[Path, int, int]:
    total_start = time.perf_counter()
    output_path = Path(path)
    build_start = time.perf_counter()
    pdb_text = build_multimodel_pdb_string_batched(
        frames,
        output_motif_chains=output_motif_chains,
        take_copy_structure=take_copy_structure,
        use_cuda_batch=use_cuda_batch,
    )
    _maybe_print_timing("write_multimodel_pdb_batched build_multimodel_pdb_string_batched", build_start)
    split_start = time.perf_counter()
    lines = pdb_text.splitlines()
    _maybe_print_timing("write_multimodel_pdb_batched splitlines", split_start)
    if add_intra_rigid_conects:
        process_start = time.perf_counter()
        lines, total_conects, model_count = process_pdb_lines(lines)
        _maybe_print_timing("write_multimodel_pdb_batched process_pdb_lines", process_start)
        write_start = time.perf_counter()
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        _maybe_print_timing("write_multimodel_pdb_batched write_text(processed)", write_start)
    else:
        total_conects = 0
        model_count = pdb_text.count("MODEL        ")
        write_start = time.perf_counter()
        output_path.write_text(pdb_text, encoding="utf-8")
        _maybe_print_timing("write_multimodel_pdb_batched write_text(raw)", write_start)
    _maybe_print_timing("write_multimodel_pdb_batched total", total_start)
    return output_path, total_conects, model_count
