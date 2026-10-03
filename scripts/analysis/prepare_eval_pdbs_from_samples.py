"""Post-hoc helper: extract single-chain eval PDBs from multi-chain samples/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nomadflow.benchmark.analysis_inputs import load_samples_metadata
from nomadflow.benchmark.eval_layout import (
    check_eval_layout,
    effective_requested_scaffold_chain,
    pdb_chain_ids,
    resolve_source_chain,
    samples_eval_is_complete,
    sync_eval_chain_metadata,
)
from nomadflow.benchmark.pmpnn_mapping import (
    apply_eval_sample_metadata,
    rebuild_pmpnn_from_eval_metadata,
)


def prepare_eval_pdbs_from_multichain_pdb(
    sample_pdb: Path,
    output_pdb: Path,
    *,
    sample_chain_name: str | None = None,
    output_chain_name: str | None = None,
) -> str:
    """Extract scaffold chain from a sample PDB and write it with ``output_chain_name``.

    ``output_chain_name`` defaults to the resolved source chain (native ``B``/``P``/``F``
    or ``sample_chain_name``). Downstream PMPNN/RMSD must use this same ID.
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("sample", str(sample_pdb))
    model = next(structure.get_models())
    present = pdb_chain_ids(sample_pdb)
    if not present:
        present = [chain.id for chain in model]
    source_chain_name = resolve_source_chain(present, sample_chain_name)
    chain = model[source_chain_name]
    written_chain = output_chain_name or source_chain_name

    lines = []
    for residue in chain:
        hetflag, resseq, icode = residue.id
        if hetflag.strip():
            continue
        for atom in residue:
            lines.append(
                f"ATOM  {atom.serial_number:5d} {atom.name:>4s} "
                f"{residue.resname:>3s} {written_chain:>1s}{resseq:4d}    "
                f"{atom.coord[0]:8.3f}{atom.coord[1]:8.3f}{atom.coord[2]:8.3f}"
                f"  1.00  0.00           {atom.element:>2s}  "
            )
    lines.append("END")
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    output_pdb.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return written_chain


def _sync_pmpnn_fixed_pos_dict(sampling_run: Path) -> None:
    """Write PMPNN dict from eval metadata, creating the jsonl if missing."""
    sampling_run = Path(sampling_run)
    pmpnn_path = sampling_run / "pmpnn_fixed_pos_dict.jsonl"
    metadata = load_samples_metadata(sampling_run)
    pmpnn_path.write_text(
        json.dumps(rebuild_pmpnn_from_eval_metadata(metadata)), encoding="utf-8"
    )


def prepare_eval_pdbs(
    sampling_run: Path,
    *,
    sample_chain_name: str | None = None,
    force: bool = False,
) -> str:
    """Write ``samples_eval/`` from ``samples/`` and persist the eval chain in metadata.

    Skip the rewrite only when eval PDB names already match metadata **and**
    ``check_eval_layout`` is empty (after syncing metadata to the on-disk eval
    chain). A complete but wrong extract (motif ``B`` instead of scaffold ``A``)
    is rewritten.
    """
    sampling_run = Path(sampling_run).resolve()
    if not force and samples_eval_is_complete(sampling_run):
        sync_eval_chain_metadata(sampling_run)
        _sync_pmpnn_fixed_pos_dict(sampling_run)
        if not check_eval_layout(sampling_run):
            return "skipped"

    samples_dir = sampling_run / "samples"
    if not samples_dir.is_dir():
        raise FileNotFoundError(f"Missing {samples_dir}")

    samples_eval_dir = sampling_run / "samples_eval"
    samples_eval_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_samples_metadata(sampling_run)
    metadata_updated = False
    for name, sample_metadata in metadata.items():
        viz_path = None
        for key in ("viz_path", "path"):
            raw = sample_metadata.get(key)
            if not raw:
                continue
            candidate = Path(raw)
            if candidate.is_file():
                viz_path = candidate
                break
        if viz_path is None:
            viz_path = samples_dir / f"{name}.pdb"
        requested = effective_requested_scaffold_chain(
            sample_metadata, viz_path, override=sample_chain_name
        )
        output_path = samples_eval_dir / f"{name}.pdb"
        written_chain = prepare_eval_pdbs_from_multichain_pdb(
            viz_path,
            output_path,
            sample_chain_name=requested,
        )
        before = json.dumps(sample_metadata, sort_keys=True)
        sample_metadata["sample_chain_name"] = written_chain
        apply_eval_sample_metadata(sample_metadata)
        if json.dumps(sample_metadata, sort_keys=True) != before:
            metadata_updated = True
        print(f"Wrote {output_path} chain={written_chain}")
    if metadata_updated:
        metadata_path = sampling_run / "samples_metadata.json"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        print(f"Updated RMSD mapping metadata in {metadata_path}")
    _sync_pmpnn_fixed_pos_dict(sampling_run)
    return "prepared"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write samples_eval/ by extracting scaffold chain from samples/."
    )
    parser.add_argument(
        "--sampling-run",
        type=Path,
        required=True,
        help="Sampling run directory containing samples/ and samples_metadata.json",
    )
    parser.add_argument(
        "--sample-chain-name",
        default=None,
        help="Scaffold chain id (default: sample_chain_name / stage2_sample_chain_name / resolve_source_chain)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite samples_eval/ even when names match and check_eval_layout is empty.",
    )
    args = parser.parse_args()
    status = prepare_eval_pdbs(
        args.sampling_run,
        sample_chain_name=args.sample_chain_name,
        force=args.force,
    )
    if status == "skipped":
        print(f"samples_eval already complete under {args.sampling_run.resolve()}")


if __name__ == "__main__":
    main()
