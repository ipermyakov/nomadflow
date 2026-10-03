"""Eval PDB / metadata layout helpers shared by wrappers, prepare, and tests.

Contract: the single chain in ``samples_eval/*.pdb`` must match
``sample_chain_name``, ``fixed_seq_chain``, PMPNN dict keys, and
``rmsd_reference_chain``. ``rmsd_folded_chain`` stays ``A`` (ESMFold output).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nomadflow.benchmark.analysis_inputs import load_samples_metadata
from nomadflow.benchmark.pmpnn_mapping import (
    apply_eval_sample_metadata,
    build_pmpnn_fixed_pos_dict,
)
from nomadflow.benchmark.rmsd_mapping import build_rmsd_mapping


def pdb_chain_ids(pdb_path: Path) -> list[str]:
    """Unique chain IDs from ATOM records, first-seen order."""
    chains: list[str] = []
    seen: set[str] = set()
    for line in Path(pdb_path).read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM"):
            continue
        if len(line) < 22:
            continue
        chain = line[21:22]
        if chain not in seen:
            seen.add(chain)
            chains.append(chain)
    return chains


def pdb_ca_resnums(pdb_path: Path, chain_name: str) -> set[int]:
    resnums: set[int] = set()
    for line in Path(pdb_path).read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM"):
            continue
        if len(line) < 26:
            continue
        if line[21:22] != chain_name:
            continue
        if line[12:16].strip() != "CA":
            continue
        resnums.add(int(line[22:26]))
    return resnums


def eval_pdb_chain(pdb_path: Path) -> str:
    chains = pdb_chain_ids(pdb_path)
    if len(chains) != 1:
        raise ValueError(f"{pdb_path} has chains {chains!r}, expected exactly one")
    return chains[0]


def resolve_source_chain(present: list[str], requested: str | None) -> str:
    """Pick the scaffold chain from a viz/sample PDB."""
    if requested and requested in present:
        return requested
    if len(present) == 1:
        return present[0]
    if requested:
        raise KeyError(f"Chain {requested!r} not in sample PDB (have {present})")
    if "A" in present:
        return "A"
    raise KeyError(f"Cannot infer scaffold chain from {present}")


def requested_scaffold_chain(
    sample_metadata: dict[str, Any],
    *,
    override: str | None = None,
) -> str | None:
    """Named scaffold chain only; never fall back to viz ``fixed_seq_chain``.

    Mode-2 Stage-2 samples from nomadflow-main are two-chain ``A``+``B``.
    Unique ``fixed_seq_chain`` is the viz motif id (``B``), not the eval scaffold.
    Missing name → ``None`` so ``resolve_source_chain`` picks ``A`` (multi-chain)
    or the sole chain (Mode 3).
    """
    if override:
        return override
    for key in ("sample_chain_name", "stage2_sample_chain_name"):
        named = sample_metadata.get(key)
        if isinstance(named, str) and named:
            return named
    return None


def effective_requested_scaffold_chain(
    sample_metadata: dict[str, Any],
    viz_pdb: Path,
    *,
    override: str | None = None,
) -> str | None:
    """Drop a poisoned metadata chain name before extracting the eval PDB.

    CLI ``override`` is trusted. A metadata name is ignored when the viz PDB has
    several chains and none of ``fixed_seq_res_idx`` fall on that chain (CA
    resnums). That is the leftover ``sample_chain_name=B`` after a previous
    prepare extracted the viz motif instead of the scaffold.
    """
    if override:
        return override
    requested = requested_scaffold_chain(sample_metadata)
    if not requested:
        return None
    present = pdb_chain_ids(viz_pdb)
    if len(present) <= 1:
        return requested
    motif_res = [int(res_idx) for res_idx in (sample_metadata.get("fixed_seq_res_idx") or [])]
    if not motif_res:
        return requested
    chain_res = pdb_ca_resnums(viz_pdb, requested)
    if not any(res_idx in chain_res for res_idx in motif_res):
        return None
    return requested


def samples_eval_pdb_names(sampling_run: Path) -> set[str]:
    eval_dir = Path(sampling_run) / "samples_eval"
    if not eval_dir.is_dir():
        return set()
    return {path.stem for path in eval_dir.glob("*.pdb") if path.is_file()}


def samples_eval_is_complete(sampling_run: Path) -> bool:
    """True only when every metadata sample has an eval PDB and there are no extras."""
    sampling_run = Path(sampling_run)
    metadata = load_samples_metadata(sampling_run)
    if not metadata:
        return False
    names = samples_eval_pdb_names(sampling_run)
    return names == set(metadata.keys())


def pmpnn_seqs_is_complete(sampling_run: Path) -> bool:
    """True when ``seqs/<sample>.fa`` exists for every metadata sample."""
    sampling_run = Path(sampling_run)
    metadata = load_samples_metadata(sampling_run)
    if not metadata:
        return False
    seqs_dir = sampling_run / "seqs"
    if not seqs_dir.is_dir():
        return False
    for name in metadata:
        if not (seqs_dir / f"{name}.fa").is_file():
            return False
    return True


def _metadata_path(sampling_run: Path) -> Path:
    return Path(sampling_run) / "samples_metadata.json"


def sync_eval_chain_metadata(sampling_run: Path) -> bool:
    """Set ``sample_chain_name`` from the actual eval PDB chain; do not rewrite PDBs.

    Used when ``samples_eval/`` is already complete (including the Mode-3 always-A
    hotfix artifact) so PMPNN/RMSD keys match the files on disk.
    """
    sampling_run = Path(sampling_run)
    metadata = load_samples_metadata(sampling_run)
    eval_dir = sampling_run / "samples_eval"
    updated = False
    for sample_name, sample_metadata in metadata.items():
        pdb_path = eval_dir / f"{sample_name}.pdb"
        if not pdb_path.is_file():
            continue
        chain = eval_pdb_chain(pdb_path)
        before = json.dumps(sample_metadata, sort_keys=True)
        sample_metadata["sample_chain_name"] = chain
        apply_eval_sample_metadata(sample_metadata)
        if json.dumps(sample_metadata, sort_keys=True) != before:
            updated = True
    if updated:
        path = _metadata_path(sampling_run)
        path.write_text(json.dumps(metadata), encoding="utf-8")
    return updated


def check_eval_layout(sampling_run: Path) -> list[str]:
    """Return human-readable invariant violations; empty means OK."""
    sampling_run = Path(sampling_run)
    errors: list[str] = []
    metadata = load_samples_metadata(sampling_run)
    eval_dir = sampling_run / "samples_eval"
    samples_dir = sampling_run / "samples"
    if not eval_dir.is_dir() and not samples_dir.is_dir():
        return [f"Neither samples_eval/ nor samples/ under {sampling_run}"]

    pdb_root = eval_dir if eval_dir.is_dir() else samples_dir
    pmpnn_path = sampling_run / "pmpnn_fixed_pos_dict.jsonl"
    pmpnn_dict: dict[str, Any] | None = None
    if pmpnn_path.is_file():
        pmpnn_dict = json.loads(pmpnn_path.read_text(encoding="utf-8"))

    for sample_name, sample_metadata in metadata.items():
        pdb_path = pdb_root / f"{sample_name}.pdb"
        if not pdb_path.is_file():
            errors.append(f"{sample_name}: missing {pdb_path.name} under {pdb_root.name}/")
            continue
        chains = pdb_chain_ids(pdb_path)
        if len(chains) != 1:
            errors.append(f"{sample_name}: expected single eval chain, have {chains}")
            continue
        chain = chains[0]
        named = sample_metadata.get("sample_chain_name")
        if named is None:
            errors.append(
                f"{sample_name}: missing sample_chain_name (eval chain is {chain!r})"
            )
        elif str(named) != chain:
            errors.append(
                f"{sample_name}: sample_chain_name={named!r} != eval PDB chain {chain!r}"
            )
        fixed = sample_metadata.get("fixed_seq_chain") or []
        if isinstance(fixed, list) and any(str(item) != chain for item in fixed):
            errors.append(
                f"{sample_name}: fixed_seq_chain={fixed!r} != eval PDB chain {chain!r}"
            )
        mapping = build_rmsd_mapping(sample_metadata)
        if mapping["reference_chain"] != chain:
            errors.append(
                f"{sample_name}: rmsd_reference_chain={mapping['reference_chain']!r} "
                f"!= eval PDB chain {chain!r}"
            )
        folded = mapping["folded_chain"]
        if folded != "A":
            errors.append(f"{sample_name}: rmsd_folded_chain={folded!r}, expected 'A'")
        ca_resnums = pdb_ca_resnums(pdb_path, chain)
        motif_res = [int(res_idx) for res_idx in (sample_metadata.get("fixed_seq_res_idx") or [])]
        if motif_res and not any(res_idx in ca_resnums for res_idx in motif_res):
            errors.append(
                f"{sample_name}: none of fixed_seq residues {motif_res} are on eval chain {chain}"
            )
        expected_pmpnn = build_pmpnn_fixed_pos_dict({sample_name: dict(sample_metadata)})
        pmpnn_chains = set(expected_pmpnn.get(sample_name, {}))
        if pmpnn_chains - {chain, }:
            extra = pmpnn_chains - {chain}
            errors.append(
                f"{sample_name}: PMPNN dict has non-eval chains {sorted(extra)}"
            )
        if pmpnn_dict is not None and sample_name in pmpnn_dict:
            disk_chains = {str(c) for c, pos in pmpnn_dict[sample_name].items() if pos}
            if disk_chains - {chain}:
                errors.append(
                    f"{sample_name}: pmpnn_fixed_pos_dict.jsonl non-eval chains "
                    f"{sorted(disk_chains - {chain})}"
                )
    if eval_dir.is_dir() and not samples_eval_is_complete(sampling_run):
        n_eval = len(samples_eval_pdb_names(sampling_run))
        errors.append(
            f"samples_eval incomplete: {n_eval} PDBs vs {len(metadata)} metadata entries"
        )
    return errors


def rewrite_tasks_num_samples(tasks_yaml: Path, output_yaml: Path, num_samples: int) -> Path:
    """Copy a tasks YAML with every ``num_samples`` overridden."""
    import yaml

    payload = yaml.safe_load(Path(tasks_yaml).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {tasks_yaml}")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise TypeError(f"Expected tasks list in {tasks_yaml}")
    for task in tasks:
        if isinstance(task, dict):
            task["num_samples"] = int(num_samples)
    output_yaml = Path(output_yaml)
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    output_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_yaml
