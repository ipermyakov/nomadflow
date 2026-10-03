"""Build reference PDBs for motif-scaffolding benchmark tasks.

For each task in a sampling config, writes a PDB with:
  - chain A: the full source protein chain from which motifs are taken
  - chains B, C, ...: duplicated motif groups (one output chain per motif group)

Motif residues on chains B+ are renumbered sequentially (1..N) within each chain.
"""

from __future__ import annotations

import argparse
import re
import string
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from Bio.PDB import PDBParser


_SEGMENT_RE = re.compile(r"^([A-Za-z])(\d+)(?:-(\d+))?$")


def expand_segment_spec(segment_spec: str) -> list[tuple[str, int]]:
    match = _SEGMENT_RE.fullmatch(segment_spec.strip())
    if match is None:
        raise ValueError(f"Invalid motif segment spec: {segment_spec}")
    chain_id = match.group(1)
    start = int(match.group(2))
    end = int(match.group(3)) if match.group(3) is not None else start
    if end < start:
        raise ValueError(f"Invalid segment range in motif spec: {segment_spec}")
    return [(chain_id, resid) for resid in range(start, end + 1)]


def get_motif_groups(task: Mapping[str, Any]) -> list[list[str]]:
    motif_groups = task.get("motif_groups")
    if motif_groups is not None:
        return [list(group) for group in motif_groups]
    pdb_contigs = str(task["pdb_contigs"])
    return [[segment.strip()] for segment in pdb_contigs.split(",") if segment.strip()]


def resolve_task_pdb_path(task: Mapping[str, Any], pdbs_dir: Path) -> Path:
    name = str(task.get("name", "")).strip()
    pdb_field = str(task.get("pdb", "")).strip()
    candidates: list[Path] = []
    if pdb_field:
        candidates.append(pdbs_dir / Path(pdb_field).name)
    if name:
        candidates.append(pdbs_dir / f"{name}.pdb")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates) or "(none)"
    raise FileNotFoundError(f"Could not resolve PDB for task {name or pdb_field}: tried {searched}")


def _format_atom_line(
    *,
    serial: int,
    atom_name: str,
    resname: str,
    chain_id: str,
    resseq: int,
    x: float,
    y: float,
    z: float,
    element: str,
) -> str:
    element = (element or "  ")[:2].rjust(2)
    return (
        f"ATOM  {serial:5d} {atom_name:>4s} {resname:>3s} {chain_id:>1s}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element}  "
    )


def _residue_lookup(model, chain_id: str) -> dict[int, Any]:
    chain = model[chain_id]
    lookup: dict[int, Any] = {}
    for residue in chain:
        if residue.id[0] != " ":
            continue
        lookup[int(residue.id[1])] = residue
    return lookup


def write_motif_group_reference_pdb(
    source_pdb: Path,
    output_pdb: Path,
    *,
    sample_chain_name: str,
    motif_groups: Sequence[Sequence[str]],
) -> None:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(source_pdb.stem, str(source_pdb))
    model = next(structure.get_models())

    if sample_chain_name not in model:
        raise KeyError(f"Chain {sample_chain_name!r} not found in {source_pdb}")

    lines: list[str] = []
    atom_serial = 1

    source_chain = model[sample_chain_name]
    for residue in source_chain:
        if residue.id[0] != " ":
            continue
        resseq = int(residue.id[1])
        for atom in residue:
            element = (atom.element or atom.name[0]).strip() or "  "
            lines.append(
                _format_atom_line(
                    serial=atom_serial,
                    atom_name=atom.name,
                    resname=residue.resname,
                    chain_id="A",
                    resseq=resseq,
                    x=float(atom.coord[0]),
                    y=float(atom.coord[1]),
                    z=float(atom.coord[2]),
                    element=element,
                )
            )
            atom_serial += 1
    lines.append("TER")

    referenced_chains = {
        chain_id
        for group in motif_groups
        for segment_spec in group
        for chain_id, _ in expand_segment_spec(segment_spec)
    }
    chain_lookup: dict[str, dict[int, Any]] = {}
    for chain_id in referenced_chains:
        if chain_id not in model:
            raise KeyError(f"Motif references chain {chain_id!r}, missing in {source_pdb}")
        chain_lookup[chain_id] = _residue_lookup(model, chain_id)

    motif_chain_ids = list(string.ascii_uppercase[1 : 1 + len(motif_groups)])
    if len(motif_groups) > len(string.ascii_uppercase) - 1:
        raise ValueError(f"Too many motif groups ({len(motif_groups)}); PDB chain ids exhausted")

    for out_chain_id, group_segments in zip(motif_chain_ids, motif_groups, strict=True):
        out_resseq = 1
        for segment_spec in group_segments:
            for source_chain_id, source_resid in expand_segment_spec(segment_spec):
                residue = chain_lookup[source_chain_id].get(source_resid)
                if residue is None:
                    raise KeyError(
                        f"Residue {source_chain_id}{source_resid} from {segment_spec!r} "
                        f"not found in {source_pdb}"
                    )
                for atom in residue:
                    element = (atom.element or atom.name[0]).strip() or "  "
                    lines.append(
                        _format_atom_line(
                            serial=atom_serial,
                            atom_name=atom.name,
                            resname=residue.resname,
                            chain_id=out_chain_id,
                            resseq=out_resseq,
                            x=float(atom.coord[0]),
                            y=float(atom.coord[1]),
                            z=float(atom.coord[2]),
                            element=element,
                        )
                    )
                    atom_serial += 1
                out_resseq += 1
        lines.append("TER")

    lines.append("END")
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    output_pdb.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_motif_group_reference_pdbs(
    config_path: Path,
    pdbs_dir: Path,
    output_dir: Path,
) -> list[Path]:
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    tasks = config.get("tasks") or []
    if not tasks:
        raise ValueError(f"No tasks found in {config_path}")

    written: list[Path] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        if task.get("task") != "motif_scaffolding":
            continue
        name = str(task.get("name", "")).strip()
        if not name:
            raise ValueError(f"Task in {config_path} is missing name")

        source_pdb = resolve_task_pdb_path(task, pdbs_dir)
        sample_chain_name = str(task.get("sample_chain_name", "A"))
        motif_groups = get_motif_groups(task)
        output_pdb = output_dir / f"{name}.pdb"
        write_motif_group_reference_pdb(
            source_pdb,
            output_pdb,
            sample_chain_name=sample_chain_name,
            motif_groups=motif_groups,
        )
        written.append(output_pdb)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write reference PDBs with full source chain on A and duplicated motif groups on B+."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Motif scaffolding tasks YAML (e.g. config_multi_motif_se3_selected.yaml)",
    )
    parser.add_argument(
        "--pdbs-dir",
        type=Path,
        default=Path("configs/sampling/protpardelle_motif_scaffolding/pdbs"),
        help="Directory with source benchmark PDBs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for output reference PDBs",
    )
    args = parser.parse_args()

    written = prepare_motif_group_reference_pdbs(
        args.config.resolve(),
        args.pdbs_dir.resolve(),
        args.output_dir.resolve(),
    )
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
