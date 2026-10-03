"""Tests for motif-group reference PDB export."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.analysis.prepare_motif_group_reference_pdbs import (
    expand_segment_spec,
    write_motif_group_reference_pdb,
)


PDBS_DIR = Path("configs/sampling/protpardelle_motif_scaffolding/pdbs")
SOURCE_1QJG = PDBS_DIR / "1QJG.pdb"


def _chain_residue_sets(pdb_path: Path) -> dict[str, set[int]]:
    chains: dict[str, set[int]] = {}
    for line in pdb_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM"):
            continue
        chain_id = line[21:22]
        resnum = int(line[22:26])
        chains.setdefault(chain_id, set()).add(resnum)
    return chains


def _chain_ca_coords(pdb_path: Path, chain_id: str, resnum: int) -> tuple[float, float, float]:
    for line in pdb_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[21:22] != chain_id or int(line[22:26]) != resnum:
            continue
        if line[12:16].strip() != "CA":
            continue
        return (
            float(line[30:38]),
            float(line[38:46]),
            float(line[46:54]),
        )
    raise KeyError((chain_id, resnum))


@pytest.mark.parametrize(
    ("segment", "expected"),
    [
        ("A38", [("A", 38)]),
        ("A22-36", [("A", resid) for resid in range(22, 37)]),
    ],
)
def test_expand_segment_spec(segment, expected):
    assert expand_segment_spec(segment) == expected


@pytest.mark.skipif(not SOURCE_1QJG.is_file(), reason="benchmark PDB missing")
def test_write_motif_group_reference_pdb_1qjg(tmp_path: Path):
    output_pdb = tmp_path / "1QJG.pdb"
    write_motif_group_reference_pdb(
        SOURCE_1QJG,
        output_pdb,
        sample_chain_name="A",
        motif_groups=[["A38", "A14", "A99"], ["A22-36"]],
    )

    chains = _chain_residue_sets(output_pdb)
    assert chains["A"] == set(range(1, 126))
    assert chains["B"] == {1, 2, 3}
    assert chains["C"] == set(range(1, 16))

    assert _chain_ca_coords(output_pdb, "A", 38) == _chain_ca_coords(SOURCE_1QJG, "A", 38)
    assert _chain_ca_coords(output_pdb, "B", 1) == _chain_ca_coords(SOURCE_1QJG, "A", 38)
    assert _chain_ca_coords(output_pdb, "B", 2) == _chain_ca_coords(SOURCE_1QJG, "A", 14)
    assert _chain_ca_coords(output_pdb, "B", 3) == _chain_ca_coords(SOURCE_1QJG, "A", 99)
    assert _chain_ca_coords(output_pdb, "C", 1) == _chain_ca_coords(SOURCE_1QJG, "A", 22)
    assert _chain_ca_coords(output_pdb, "C", 15) == _chain_ca_coords(SOURCE_1QJG, "A", 36)
