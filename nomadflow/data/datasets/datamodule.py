import os
from functools import partial
import hashlib
from typing import List, Optional
import sys
import time
from pathlib import Path
from dataclasses import dataclass, replace

import numpy as np
import torch
from torch.utils.data import DataLoader
import lightning as L
import pandas as pd


def _phase_ms(timings: Optional[dict], key: str, t0: float) -> None:
    """Accumulate wall-ms for an optional prepare phase timer dict."""
    if timings is None:
        return
    timings[key] = float(timings.get(key, 0.0)) + (time.perf_counter() - t0) * 1000.0

from nomadflow.boltz.data.types import (
    Structure,
    Connection,
    Record
)
from nomadflow.boltz.data import const
from nomadflow.boltz.data.sample.sampler import Sample

from nomadflow.data.featurize.cropper import (
    Cropper,
    filter_tokenized_to_resolved,
)
from nomadflow.data.featurize.tokenize import (
    append_motif_group_aggregators,
    tokenize_structure,
    Tokenized,
)
from nomadflow.data.featurize.assembler import featurize_training, collate

from nomadflow.runtime.sampling.dispatcher import BiomoleculeTaskDispatcher
from nomadflow.runtime.training.motif_scaffold import (
    MinimalMotifScaffolding,
    MotifScaffolding,
    VoronoiClusterMotifScaffolding,
)
from nomadflow.runtime.training.voronoi_motif import (
    order_clusters_near_query,
    partition_voronoi_clusters,
    sample_potential_motifs,
    select_cluster_atomic_crop,
)


def load_input(record: Record, data_dir):
    """Load the given input data.

    Parameters
    ----------
    record : Record
        The record to load.
    target_dir : Path
        The path to the data directory.
    msa_dir : Path
        The path to msa directory.

    Returns
    -------
    Input
        The loaded input.

    """
    # Load the structure
    try:
        # find the subdirectory
        if "AF-" in record.id:
            mid = record.id[6:8]
        elif "af-" in record.id:
            mid = record.id[6:8]
        else:
            mid = record.id[1:3]
        structure = np.load(data_dir / "structures" / mid / f"{record.id}.npz")
    except:
        # original boltz format
        structure = np.load(data_dir / "structures" / f"{record.id}.npz")

    # In order to add cyclic_period to chains if it does not exist
    # Extract the chains array
    chains = structure["chains"]
    # Check if the field exists
    if "cyclic_period" not in chains.dtype.names:
        # Create a new dtype with the additional field
        new_dtype = chains.dtype.descr + [("cyclic_period", "i4")]
        # Create a new array with the new dtype
        new_chains = np.empty(chains.shape, dtype=new_dtype)
        # Copy over existing fields
        for name in chains.dtype.names:
            new_chains[name] = chains[name]
        # Set the new field to 0
        new_chains["cyclic_period"] = 0
        # Replace old chains array with new one
        chains = new_chains

    structure = Structure(
        atoms=structure["atoms"],
        bonds=structure["bonds"],
        residues=structure["residues"],
        chains=chains, # chains var accounting for missing cyclic_period
        connections=structure["connections"].astype(Connection),
        interfaces=structure["interfaces"],
        mask=structure["mask"],
    )

    return structure


def mask_nonstandard_residues(struct: Structure):
    residues = struct.residues
    atoms = struct.atoms

    residues_copy = residues.copy()
    residues_copy['is_present'] = residues['is_present'] & residues['is_standard']
    atoms_copy = atoms.copy()
    for residue in residues:
        atom_idx = residue['atom_idx']
        atom_num = residue['atom_num']
        res_atoms = atoms_copy[atom_idx:atom_idx+atom_num]
        if not residue['is_standard']:
            res_atoms['is_present'] = False

    return replace(struct, atoms=atoms_copy, residues=residues_copy)


def _preview_task_data(struct: Structure) -> dict[str, np.ndarray]:
    """Build no-copy task data used only to select a pre-motif crop."""
    residue_mask = np.ones(len(struct.residues), dtype=bool)
    no_residues = np.zeros(len(struct.residues), dtype=bool)
    return {
        "atom_noising_mask": np.ones(len(struct.atoms), dtype=bool),
        "res_type_noising_mask": residue_mask,
        "copy_indexed_residue_mask": no_residues,
        "copy_unindexed_residue_mask": no_residues.copy(),
        "copy_atomized_residue_mask": no_residues.copy(),
        "motif_group_id_residue": np.zeros(len(struct.residues), dtype=np.int32),
    }


def _residue_array_index_map(struct: Structure) -> dict[tuple[int, int], int]:
    """Map (asym_id, per-chain local res_idx) → global residues array index."""
    mapping: dict[tuple[int, int], int] = {}
    for chain in struct.chains:
        asym_id = int(chain["asym_id"])
        start = int(chain["res_idx"])
        stop = start + int(chain["res_num"])
        for array_idx in range(start, stop):
            local_res_idx = int(struct.residues[array_idx]["res_idx"])
            mapping[(asym_id, local_res_idx)] = array_idx
    return mapping


def _global_residue_indices_from_tokens(
    struct: Structure,
    tokens: np.ndarray,
) -> np.ndarray:
    """Convert token (asym_id, local res_idx) pairs to global array indices."""
    mapping = _residue_array_index_map(struct)
    out: list[int] = []
    seen: set[int] = set()
    for token in tokens:
        key = (int(token["asym_id"]), int(token["res_idx"]))
        if key not in mapping:
            raise KeyError(
                "Token residue not found in structure residue map: "
                f"asym_id={key[0]}, res_idx={key[1]}"
            )
        array_idx = mapping[key]
        if array_idx not in seen:
            seen.add(array_idx)
            out.append(array_idx)
    return np.asarray(out, dtype=np.int32)


def _mask_structure_to_residue_indices(
    struct: Structure,
    residue_indices: np.ndarray,
) -> Structure:
    """Mark residues outside a crop unresolved without changing indices.

    ``residue_indices`` must be **global** positions into ``struct.residues``
    (not per-chain local ``residue["res_idx"]`` values).
    """
    keep = {int(idx) for idx in np.asarray(residue_indices).reshape(-1)}
    residues = struct.residues.copy()
    atoms = struct.atoms.copy()

    for array_idx in range(len(residues)):
        if array_idx in keep:
            continue
        residues["is_present"][array_idx] = False
        atom_start = int(residues[array_idx]["atom_idx"])
        atom_end = atom_start + int(residues[array_idx]["atom_num"])
        atoms["is_present"][atom_start:atom_end] = False

    return replace(struct, residues=residues, atoms=atoms)


def _pre_crop_motif_structure(
    struct: Structure,
    *,
    cropper,
    max_tokens: int,
    np_rng,
    chain_id,
    interface_id,
) -> Structure:
    if max_tokens <= 0:
        raise ValueError(f"Motif pre-crop budget must be positive, got {max_tokens}")

    token_data, rigid_data, token_bonds = tokenize_structure(
        struct,
        _preview_task_data(struct),
        shuffle_chains=False,
        shuffle_copied_fragments=False,
        np_rng=np_rng,
    )
    preview = Tokenized(
        tokens=token_data,
        rigids=rigid_data,
        bonds=token_bonds,
        structure=struct,
    )
    if len(preview.tokens) <= max_tokens:
        return struct

    cropped = cropper.crop(
        preview,
        max_tokens=max_tokens,
        random=np_rng,
        chain_id=chain_id,
        interface_id=interface_id,
    )
    global_indices = _global_residue_indices_from_tokens(struct, cropped.tokens)
    return _mask_structure_to_residue_indices(struct, global_indices)


def _prepare_voronoi_motif_example(
    struct: Structure,
    *,
    task: VoronoiClusterMotifScaffolding,
    max_crop_residues: int,
    np_rng,
    torch_generator,
    chain_id,
    interface_id,
    timings: Optional[dict] = None,
) -> tuple[Structure, dict, Tokenized]:
    t0 = time.perf_counter()
    partition = partition_voronoi_clusters(
        struct,
        cluster_target_res=task.cluster_target_res,
        cluster_max_res=task.cluster_max_res,
        np_rng=np_rng,
    )
    _phase_ms(timings, "partition_ms", t0)
    if not partition.clusters:
        raise ValueError("Voronoi motif prepare found no eligible protein residues")

    t0 = time.perf_counter()
    potentials = sample_potential_motifs(
        partition,
        p_catalytic=task.p_catalytic,
        catalytic_min_res=task.catalytic_min_res,
        catalytic_max_res=task.catalytic_max_res,
        catalytic_max_ca_dist=task.catalytic_max_ca_dist,
        catalytic_distance_temperature=task.catalytic_distance_temperature,
        catalytic_min_cluster_segments=task.catalytic_min_cluster_segments,
        structural_keep_fraction_min=task.structural_keep_fraction_min,
        structural_keep_fraction_max=task.structural_keep_fraction_max,
        np_rng=np_rng,
    )
    _phase_ms(timings, "potentials_ms", t0)

    # Pick crop query from eligible residues — no full-structure preview tokenize.
    t0 = time.perf_counter()
    eligible = np.asarray(partition.eligible_residue_indices, dtype=np.int32)
    query_pool = eligible
    if chain_id is not None and eligible.size:
        chain_rows = struct.chains[struct.chains["asym_id"] == int(chain_id)]
        if len(chain_rows):
            start = int(chain_rows[0]["res_idx"])
            stop = start + int(chain_rows[0]["res_num"])
            in_chain = eligible[(eligible >= start) & (eligible < stop)]
            if in_chain.size:
                query_pool = in_chain
    query_res_idx = int(query_pool[int(np_rng.randint(len(query_pool)))])
    _phase_ms(timings, "preview_tokenize_ms", t0)  # kept key; now query-pick only

    t0 = time.perf_counter()
    cluster_order = order_clusters_near_query(
        partition,
        query_res_idx,
        np_rng,
    )
    selection = select_cluster_atomic_crop(
        partition,
        potentials,
        cluster_order,
        max_crop_residues=max_crop_residues,
        min_motif_groups=task.min_groups,
        max_motif_groups=task.max_groups,
        p_motif_cluster=task.p_motif_cluster,
        min_cluster_sequence_contacts=task.min_cluster_sequence_contacts,
        np_rng=np_rng,
    )
    cropped_struct = _mask_structure_to_residue_indices(
        struct,
        selection.scaffold_residue_indices,
    )
    _phase_ms(timings, "order_crop_ms", t0)

    t0 = time.perf_counter()
    task_data = task.sample_activated_t_and_mask(
        cropped_struct,
        selection.motif_group_id_residue,
        group_shortfall=selection.group_shortfall,
        np_rng=np_rng,
        torch_generator=torch_generator,
    )
    task_data["voronoi_complete_cluster_ids"] = np.asarray(
        selection.complete_cluster_ids,
        dtype=np.int32,
    )
    task_data["voronoi_activated_cluster_ids"] = np.asarray(
        selection.activated_cluster_ids,
        dtype=np.int32,
    )
    task_data["voronoi_partial_cluster_ids"] = np.asarray(
        selection.partial_cluster_ids,
        dtype=np.int32,
    )
    _phase_ms(timings, "activate_mask_ms", t0)

    t0 = time.perf_counter()
    token_data, rigid_data, token_bonds = tokenize_structure(
        cropped_struct,
        task_data,
        np_rng=np_rng,
    )
    tokenized = filter_tokenized_to_resolved(
        Tokenized(
            tokens=token_data,
            rigids=rigid_data,
            bonds=token_bonds,
            structure=cropped_struct,
        )
    )
    _phase_ms(timings, "final_tokenize_ms", t0)

    activated_residues = set(
        np.flatnonzero(selection.motif_group_id_residue > 0).tolist()
    )
    copy_mask = (
        np.asarray(task_data["copy_indexed_residue_mask"], dtype=bool)
        | np.asarray(task_data["copy_unindexed_residue_mask"], dtype=bool)
    )
    copy_from_task = set(np.flatnonzero(copy_mask).tolist())
    if copy_from_task != activated_residues:
        raise RuntimeError(
            "Voronoi activated motifs and copy masks differ: "
            f"activated={sorted(activated_residues)}, "
            f"copy_mask={sorted(copy_from_task)}, "
            f"scaffold={len(selection.scaffold_residue_indices)}, "
            f"copy_count={selection.copy_count}, "
            f"projected={selection.projected_token_count}, "
            f"max_crop_residues={max_crop_residues}"
        )
    residue_map = _residue_array_index_map(cropped_struct)
    final_copy_residues: set[int] = set()
    for token in tokenized.tokens[tokenized.tokens["is_copy"]]:
        key = (int(token["asym_id"]), int(token["res_idx"]))
        if key not in residue_map:
            raise RuntimeError(
                "Voronoi copy token has no residue array index: "
                f"asym_id={key[0]}, res_idx={key[1]}"
            )
        final_copy_residues.add(residue_map[key])
    if final_copy_residues != activated_residues:
        raise RuntimeError(
            "Voronoi activated motifs and final copies differ: "
            f"activated={sorted(activated_residues)}, "
            f"copies={sorted(final_copy_residues)}, "
            f"scaffold={len(selection.scaffold_residue_indices)}, "
            f"copy_count={selection.copy_count}, "
            f"projected={selection.projected_token_count}, "
            f"n_tokens={len(tokenized.tokens)}, "
            f"max_crop_residues={max_crop_residues}"
        )
    sampled_groups = set(
        selection.motif_group_id_residue[
            selection.motif_group_id_residue > 0
        ].tolist()
    )
    final_groups = set(
        tokenized.rigids["motif_group_id"][
            tokenized.rigids["motif_group_id"] > 0
        ].tolist()
    )
    if final_groups != sampled_groups:
        raise RuntimeError(
            "Voronoi motif groups changed during tokenize: "
            f"sampled={sorted(sampled_groups)}, final={sorted(final_groups)}, "
            f"scaffold={len(selection.scaffold_residue_indices)}, "
            f"n_tokens={len(tokenized.tokens)}, "
            f"max_crop_residues={max_crop_residues}"
        )
    for cluster_id in selection.partial_cluster_ids:
        residues = partition.clusters[cluster_id].residue_indices
        if np.any(selection.motif_group_id_residue[residues] > 0):
            raise RuntimeError("Partial fallback cluster received a motif group")
    if len(tokenized.tokens) + len(sampled_groups) > max_crop_residues:
        raise RuntimeError(
            "Voronoi token projection exceeded max_crop_residues: "
            f"n_tokens={len(tokenized.tokens)}, n_groups={len(sampled_groups)}, "
            f"scaffold={len(selection.scaffold_residue_indices)}, "
            f"copy_count={selection.copy_count}, "
            f"projected={selection.projected_token_count}, "
            f"max_crop_residues={max_crop_residues}"
        )
    return cropped_struct, task_data, tokenized


def _stable_seed(*parts) -> int:
    joined = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**31 - 1)


def _make_torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def _allocate_counts(total: int, weights: List[float]) -> List[int]:
    if total <= 0:
        return [0 for _ in weights]
    if len(weights) == 0:
        return []
    weights_arr = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weights_arr).all() or weights_arr.sum() <= 0:
        weights_arr = np.ones_like(weights_arr)
    normalized = weights_arr / weights_arr.sum()
    raw = normalized * total
    counts = np.floor(raw).astype(int)
    remainder = total - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        counts[order[:remainder]] += 1
    return counts.tolist()


def _build_cropper(use_cropper: bool, crop_min_neighbors: int, crop_max_neighbors: int):
    if not use_cropper:
        return None
    return Cropper(
        min_neighborhood=crop_min_neighbors,
        max_neighborhood=crop_max_neighbors,
    )


def _resolve_remove_mol_types(remove_mol_types):
    if remove_mol_types is None:
        return []
    print("Removing chains of types:", remove_mol_types)
    return [const.chain_types.index(s) for s in remove_mol_types]


def _prepare_training_features(
    *,
    dataset,
    sample: Sample,
    task,
    cropper,
    max_crop_residues: int,
    max_crop_rigids: int,
    remove_mol_types,
    mask_nonstandard: bool,
    np_rng,
    torch_generator: Optional[torch.Generator] = None,
    metadata: Optional[dict] = None,
):
    prepared_example = prepare_training_tokenized_example(
        dataset=dataset,
        sample=sample,
        task=task,
        cropper=cropper,
        max_crop_residues=max_crop_residues,
        remove_mol_types=remove_mol_types,
        mask_nonstandard=mask_nonstandard,
        np_rng=np_rng,
        torch_generator=torch_generator,
        metadata=metadata,
    )
    return featurize_prepared_training_example(
        prepared_example,
        max_crop_residues=max_crop_residues,
        max_crop_rigids=max_crop_rigids,
    )


@dataclass(frozen=True)
class PreparedTrainingExample:
    tokenized: Tokenized
    task_data: dict
    task: object
    sample: Sample
    metadata: Optional[dict] = None


def prepare_training_tokenized_example(
    *,
    dataset,
    sample: Sample,
    task,
    cropper,
    max_crop_residues: int,
    remove_mol_types,
    mask_nonstandard: bool,
    np_rng,
    torch_generator: Optional[torch.Generator] = None,
    metadata: Optional[dict] = None,
    timings: Optional[dict] = None,
):
    t0 = time.perf_counter()
    struct = load_input(sample.record, Path(dataset.data_dir))
    chain_mask = struct.mask.copy()
    remove_chain_masks = [struct.chains['mol_type'] == i for i in remove_mol_types]
    for remove_mask in remove_chain_masks:
        chain_mask[remove_mask] = False
    struct = replace(struct, mask=chain_mask)

    if mask_nonstandard:
        struct = mask_nonstandard_residues(struct)
    _phase_ms(timings, "load_ms", t0)

    is_motif_task = isinstance(
        task,
        (MotifScaffolding, MinimalMotifScaffolding),
    )
    is_voronoi_task = isinstance(task, VoronoiClusterMotifScaffolding)
    if is_voronoi_task:
        struct, task_data, tokenized_data = _prepare_voronoi_motif_example(
            struct,
            task=task,
            max_crop_residues=max_crop_residues,
            np_rng=np_rng,
            torch_generator=torch_generator,
            chain_id=sample.chain_id,
            interface_id=sample.interface_id,
            timings=timings,
        )
    else:
        if cropper is not None and is_motif_task:
            t0 = time.perf_counter()
            group_reserve = int(task.max_motif_groups())
            struct = _pre_crop_motif_structure(
                struct,
                cropper=cropper,
                max_tokens=max_crop_residues - group_reserve,
                np_rng=np_rng,
                chain_id=sample.chain_id,
                interface_id=sample.interface_id,
            )
            _phase_ms(timings, "pre_crop_ms", t0)

        t0 = time.perf_counter()
        task_data = task.sample_t_and_mask(
            struct,
            np_rng=np_rng,
            torch_generator=torch_generator,
        )
        _phase_ms(timings, "sample_t_mask_ms", t0)

        t0 = time.perf_counter()
        token_data, rigid_data, token_bonds = tokenize_structure(
            struct,
            task_data,
            np_rng=np_rng,
        )
        tokenized_data = Tokenized(
            tokens=token_data,
            rigids=rigid_data,
            bonds=token_bonds,
            structure=struct
        )
        _phase_ms(timings, "tokenize_ms", t0)

        if cropper is not None:
            t0 = time.perf_counter()
            motif_group_ids = rigid_data["motif_group_id"]
            motif_group_count = len(np.unique(motif_group_ids[motif_group_ids > 0]))
            crop_size = max(max_crop_residues - motif_group_count, 0)
            source_copy_count = (
                int(token_data["is_copy"].sum()) if is_motif_task else 0
            )

            if len(tokenized_data.tokens) > crop_size:
                if is_motif_task:
                    required_token_indices = token_data["token_idx"][token_data["is_copy"]]
                    tokenized_data = cropper.crop(
                        tokenized_data,
                        max_tokens=crop_size,
                        random=np_rng,
                        chain_id=sample.chain_id,
                        interface_id=sample.interface_id,
                        required_token_indices=required_token_indices,
                    )
                else:
                    tokenized_data = cropper.crop(
                        tokenized_data,
                        max_tokens=crop_size,
                        random=np_rng,
                        chain_id=sample.chain_id,
                        interface_id=sample.interface_id,
                    )

            if is_motif_task:
                final_copy_count = int(tokenized_data.tokens["is_copy"].sum())
                if final_copy_count != source_copy_count:
                    raise RuntimeError(
                        "Motif-preserving crop removed copy tokens: "
                        f"expected {source_copy_count}, got {final_copy_count}"
                    )
                sampled_group_ids = np.asarray(
                    task_data.get("motif_group_id_residue", []),
                    dtype=np.int32,
                )
                sampled_groups = set(sampled_group_ids[sampled_group_ids > 0].tolist())
                final_group_ids = tokenized_data.rigids["motif_group_id"]
                final_groups = set(final_group_ids[final_group_ids > 0].tolist())
                if final_groups != sampled_groups:
                    raise RuntimeError(
                        "Motif groups changed during prepare: "
                        f"sampled={sorted(sampled_groups)}, final={sorted(final_groups)}"
                    )
            _phase_ms(timings, "post_crop_ms", t0)

    t0 = time.perf_counter()
    token_data, rigid_data = append_motif_group_aggregators(
        tokenized_data.tokens,
        tokenized_data.rigids,
        np_rng=np_rng,
    )
    tokenized_data = replace(tokenized_data, tokens=token_data, rigids=rigid_data)
    _phase_ms(timings, "aggregators_ms", t0)

    if is_motif_task and (cropper is not None or is_voronoi_task):
        if len(tokenized_data.tokens) > max_crop_residues:
            raise RuntimeError(
                f"Prepared motif example has {len(tokenized_data.tokens)} tokens, "
                f"exceeding max_crop_residues={max_crop_residues}"
            )
    if is_voronoi_task:
        aggregator_count = int(
            tokenized_data.tokens["is_motif_group_aggregator"].sum()
        )
        sampled_group_ids = np.asarray(task_data["motif_group_id_residue"])
        sampled_group_count = int(
            np.unique(sampled_group_ids[sampled_group_ids > 0]).size
        )
        if aggregator_count != sampled_group_count:
            raise RuntimeError(
                "Voronoi aggregator count does not match activated groups: "
                f"aggregators={aggregator_count}, groups={sampled_group_count}"
            )

    return PreparedTrainingExample(
        tokenized=tokenized_data,
        task_data=task_data,
        task=task,
        sample=sample,
        metadata=metadata.copy() if metadata is not None else None,
    )


def featurize_prepared_training_example(
    prepared_example: PreparedTrainingExample,
    *,
    max_crop_residues: int,
    max_crop_rigids: int,
):
    features = featurize_training(
        prepared_example.tokenized,
        prepared_example.task_data,
        max_tokens=max_crop_residues,
        max_rigids=max_crop_rigids
    )
    features['task'] = prepared_example.task
    if prepared_example.metadata:
        features.update(prepared_example.metadata)
    return features


@dataclass(frozen=True)
class ValidationItem:
    dataset_idx: int
    task_idx: int
    sample: Sample
    seed: int


class TrainingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datasets,
        max_crop_residues,
        max_crop_rigids,
        use_cropper=True,
        samples_per_epoch=1000,  # this is PER GPU
        crop_min_neighbors=0,
        crop_max_neighbors=40,
        dataset_probs=None,
        remove_mol_types=None,
        mask_nonstandard=False
    ):
        super().__init__()
        self.datasets = datasets
        self.max_crop_residues = max_crop_residues
        self.max_crop_rigids = max_crop_rigids
        self.samples_per_epoch = samples_per_epoch
        if dataset_probs is None:
            self.dataset_probs = [1/len(datasets) for _ in datasets]
        else:
            self.dataset_probs = dataset_probs
        self.samples = []
        self.cropper = _build_cropper(use_cropper, crop_min_neighbors, crop_max_neighbors)
        self.remove_mol_types = _resolve_remove_mol_types(remove_mol_types)
        self.mask_nonstandard = mask_nonstandard

        for dataset in datasets:
            records = dataset.manifest
            iterator = dataset.data_sampler.sample(records, np.random)
            self.samples.append(iterator)

    def __getitem__(self, idx):
        prepared_example = self.prepare_item(idx)
        return featurize_prepared_training_example(
            prepared_example,
            max_crop_residues=self.max_crop_residues,
            max_crop_rigids=self.max_crop_rigids,
        )

    def prepare_item(self, idx):
        del idx
        dataset_idx = np.random.choice(
            len(self.datasets),
            p=self.dataset_probs,
        )
        dataset = self.datasets[dataset_idx]
        task_sampler = dataset.task_sampler
        sample: Sample = next(self.samples[dataset_idx])
        task = task_sampler.sample_task()
        return prepare_training_tokenized_example(
            dataset=dataset,
            sample=sample,
            task=task,
            cropper=self.cropper,
            max_crop_residues=self.max_crop_residues,
            remove_mol_types=self.remove_mol_types,
            mask_nonstandard=self.mask_nonstandard,
            np_rng=np.random,
        )

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return self.samples_per_epoch


class ValidationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datasets,
        max_crop_residues,
        max_crop_rigids,
        samples_per_task=32,
        use_cropper=True,
        crop_min_neighbors=0,
        crop_max_neighbors=40,
        dataset_probs=None,
        remove_mol_types=None,
        mask_nonstandard=False,
        base_seed=0,
    ):
        super().__init__()
        self.datasets = datasets
        self.max_crop_residues = max_crop_residues
        self.max_crop_rigids = max_crop_rigids
        self.samples_per_task = samples_per_task
        self.base_seed = base_seed
        self.cropper = _build_cropper(use_cropper, crop_min_neighbors, crop_max_neighbors)
        self.remove_mol_types = _resolve_remove_mol_types(remove_mol_types)
        self.mask_nonstandard = mask_nonstandard
        if dataset_probs is None:
            self.dataset_probs = [1 / len(datasets) for _ in datasets]
        else:
            self.dataset_probs = dataset_probs
        self.items = self._build_items()

    def _build_items(self) -> List[ValidationItem]:
        grouped_tasks = {}
        task_order = []
        for dataset_idx, dataset in enumerate(self.datasets):
            task_list = getattr(dataset.task_sampler, "task_list", [])
            for task_idx, task in enumerate(task_list):
                task_key = (
                    f"{task.__class__.__module__}.{task.__class__.__name__}",
                    task.name,
                )
                if task_key not in grouped_tasks:
                    grouped_tasks[task_key] = []
                    task_order.append(task_key)
                grouped_tasks[task_key].append((dataset_idx, task_idx))

        items = []
        for task_key in task_order:
            eligible = grouped_tasks[task_key]
            eligible_weights = [self.dataset_probs[dataset_idx] for dataset_idx, _ in eligible]
            counts = _allocate_counts(self.samples_per_task, eligible_weights)
            for (dataset_idx, task_idx), count in zip(eligible, counts):
                if count <= 0:
                    continue
                dataset = self.datasets[dataset_idx]
                if len(dataset.manifest) == 0:
                    continue
                sampler_seed = _stable_seed(
                    "validation-sampler",
                    self.base_seed,
                    task_key[0],
                    task_key[1],
                    dataset_idx,
                )
                sampler_rng = np.random.RandomState(sampler_seed)
                iterator = dataset.data_sampler.sample(dataset.manifest, sampler_rng)
                for sample_idx in range(count):
                    sample = next(iterator)
                    item_seed = _stable_seed(
                        "validation-item",
                        self.base_seed,
                        task_key[0],
                        task_key[1],
                        dataset_idx,
                        sample.record.id,
                        sample.chain_id,
                        sample.interface_id,
                        sample_idx,
                    )
                    items.append(
                        ValidationItem(
                            dataset_idx=dataset_idx,
                            task_idx=task_idx,
                            sample=sample,
                            seed=item_seed,
                        )
                    )
        return items

    def __getitem__(self, idx):
        prepared_example = self.prepare_item(idx)
        return featurize_prepared_training_example(
            prepared_example,
            max_crop_residues=self.max_crop_residues,
            max_crop_rigids=self.max_crop_rigids,
        )

    def prepare_item(self, idx):
        item = self.items[idx]
        dataset = self.datasets[item.dataset_idx]
        task = dataset.task_sampler.task_list[item.task_idx]
        np_rng = np.random.RandomState(item.seed)
        torch_generator = _make_torch_generator(item.seed)
        return prepare_training_tokenized_example(
            dataset=dataset,
            sample=item.sample,
            task=task,
            cropper=self.cropper,
            max_crop_residues=self.max_crop_residues,
            remove_mol_types=self.remove_mol_types,
            mask_nonstandard=self.mask_nonstandard,
            np_rng=np_rng,
            torch_generator=torch_generator,
            metadata={
                "val_seed": torch.tensor(item.seed, dtype=torch.long),
            },
        )

    def __len__(self):
        return len(self.items)


def build_validation_dataset_from_training_dataset(
    train_dataset: TrainingDataset,
    *,
    samples_per_task: int = 32,
    base_seed: int = 0,
):
    if train_dataset.cropper is None:
        use_cropper = False
        crop_min_neighbors = 0
        crop_max_neighbors = 0
    else:
        use_cropper = True
        crop_min_neighbors = min(train_dataset.cropper.neighborhood_sizes)
        crop_max_neighbors = max(train_dataset.cropper.neighborhood_sizes)

    remove_mol_types = [const.chain_types[idx] for idx in train_dataset.remove_mol_types]

    return ValidationDataset(
        datasets=train_dataset.datasets,
        max_crop_residues=train_dataset.max_crop_residues,
        max_crop_rigids=train_dataset.max_crop_rigids,
        samples_per_task=samples_per_task,
        use_cropper=use_cropper,
        crop_min_neighbors=crop_min_neighbors,
        crop_max_neighbors=crop_max_neighbors,
        dataset_probs=train_dataset.dataset_probs,
        remove_mol_types=remove_mol_types,
        mask_nonstandard=train_dataset.mask_nonstandard,
        base_seed=base_seed,
    )


class BiomoleculeDataModule(L.LightningDataModule):
    def __init__(self,
                 train_dataset: TrainingDataset,
                 batch_size,  # this is PER GPU
                 num_workers,
                 val_dataset=None,
                 val_num_workers=0,
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_num_workers = val_num_workers
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def build_dataloader(self, x, collate_fn, num_workers=None):
        worker_count = self.num_workers if num_workers is None else num_workers
        dataloader = DataLoader(
            x,
            num_workers=worker_count,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            shuffle=False
        )
        return dataloader

    def train_dataloader(self):
        return self.build_dataloader(self.train_dataset, collate)

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return self.build_dataloader(self.val_dataset, collate, num_workers=self.val_num_workers)


class BiomoleculeSamplingDataModule(L.LightningDataModule):
    def __init__(self,
                 tasks_yaml,
                 batch_size,
                 batching_mode="optimal",
                 use_collate_for_pad=False
    ):
        super().__init__()
        self.batching_mode = batching_mode
        self.batch_size = batch_size

        self.task_dispatcher = BiomoleculeTaskDispatcher(
            tasks_yaml,
            1, #batch_size,
            batching_mode,
            use_collate_for_pad
        )

    def predict_dataloader(self):
        dataloader = DataLoader(
            self.task_dispatcher,
            batch_size=self.batch_size,
            collate_fn=collate,
            shuffle=False
        )
        return dataloader