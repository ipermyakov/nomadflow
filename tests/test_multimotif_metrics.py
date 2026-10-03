from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from Bio.PDB import PDBIO, PDBParser, Structure, Model, Chain, Residue, Atom

from nomadflow.benchmark.multimotif_metrics import (
    aggregate_multimotif_slices,
    compute_run_multimotif_metrics,
    compute_sample_multimotif_metrics,
    write_multimotif_overlay_pdb,
    write_run_multimotif_overlay_pdbs,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_minimal_pdb(path: Path, residue_coords: dict[tuple[str, int], np.ndarray]) -> None:
    structure = Structure.Structure("test")
    model = Model.Model(0)
    structure.add(model)
    chains: dict[str, Chain.Chain] = {}
    for (chain_id, res_idx), ca_coord in residue_coords.items():
        if chain_id not in chains:
            chain = Chain.Chain(chain_id)
            model.add(chain)
            chains[chain_id] = chain
        residue = Residue.Residue((" ", res_idx, " "), "ALA", "")
        atom = Atom.Atom("CA", ca_coord, 1.0, 1.0, " ", " CA ", 1, "C")
        residue.add(atom)
        atom = Atom.Atom("N", ca_coord + np.array([1.0, 0.0, 0.0]), 1.0, 1.0, " ", " N  ", 2, "N")
        residue.add(atom)
        chains[chain_id].add(residue)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(path))


def test_identity_sample_has_zero_group_rmsd(tmp_path):
    native_coords = {
        ("A", 1): np.array([0.0, 0.0, 0.0]),
        ("A", 2): np.array([3.8, 0.0, 0.0]),
        ("A", 3): np.array([7.6, 0.0, 0.0]),
        ("A", 10): np.array([20.0, 0.0, 0.0]),
        ("A", 11): np.array([23.8, 0.0, 0.0]),
    }
    native_path = tmp_path / "native.pdb"
    sample_path = tmp_path / "sample.pdb"
    _write_minimal_pdb(native_path, native_coords)
    _write_minimal_pdb(sample_path, native_coords)

    metadata = {
        "path": str(sample_path),
        "native_pdb": str(native_path),
        "name": "task1",
        "source_residue_keys": [
            {
                "native_chain": "A",
                "native_res_idx": 1,
                "output_chain": "A",
                "output_res_idx": 1,
                "group_id": 1,
            },
            {
                "native_chain": "A",
                "native_res_idx": 2,
                "output_chain": "A",
                "output_res_idx": 2,
                "group_id": 1,
            },
            {
                "native_chain": "A",
                "native_res_idx": 10,
                "output_chain": "A",
                "output_res_idx": 10,
                "group_id": 2,
            },
            {
                "native_chain": "A",
                "native_res_idx": 11,
                "output_chain": "A",
                "output_res_idx": 11,
                "group_id": 2,
            },
        ],
    }

    rows = compute_sample_multimotif_metrics("sample1", metadata)
    group_rows = [row for row in rows if row["record_type"] == "group"]
    assert len(group_rows) == 2
    for row in group_rows:
        assert row["motif_ca_rmsd"] == pytest.approx(0.0, abs=1e-6)
        assert row["motif_ha_rmsd"] == pytest.approx(0.0, abs=1e-6)

    intergroup_rows = [row for row in rows if row["record_type"] == "intergroup"]
    assert len(intergroup_rows) == 1
    assert intergroup_rows[0]["com_distance_error"] == pytest.approx(0.0, abs=1e-6)


def test_translated_group_has_expected_com_distance_error(tmp_path):
    native_coords = {
        ("A", 1): np.array([0.0, 0.0, 0.0]),
        ("A", 2): np.array([3.8, 0.0, 0.0]),
        ("A", 10): np.array([20.0, 0.0, 0.0]),
        ("A", 11): np.array([23.8, 0.0, 0.0]),
    }
    sample_coords = {
        ("A", 1): np.array([0.0, 0.0, 0.0]),
        ("A", 2): np.array([3.8, 0.0, 0.0]),
        ("A", 10): np.array([25.0, 0.0, 0.0]),
        ("A", 11): np.array([28.8, 0.0, 0.0]),
    }
    native_path = tmp_path / "native.pdb"
    sample_path = tmp_path / "sample.pdb"
    _write_minimal_pdb(native_path, native_coords)
    _write_minimal_pdb(sample_path, sample_coords)

    metadata = {
        "path": str(sample_path),
        "native_pdb": str(native_path),
        "name": "task1",
        "source_residue_keys": [
            {
                "native_chain": "A",
                "native_res_idx": 1,
                "output_chain": "A",
                "output_res_idx": 1,
                "group_id": 1,
            },
            {
                "native_chain": "A",
                "native_res_idx": 2,
                "output_chain": "A",
                "output_res_idx": 2,
                "group_id": 1,
            },
            {
                "native_chain": "A",
                "native_res_idx": 10,
                "output_chain": "A",
                "output_res_idx": 10,
                "group_id": 2,
            },
            {
                "native_chain": "A",
                "native_res_idx": 11,
                "output_chain": "A",
                "output_res_idx": 11,
                "group_id": 2,
            },
        ],
    }

    rows = compute_sample_multimotif_metrics("sample1", metadata)
    intergroup = next(row for row in rows if row["record_type"] == "intergroup")
    native_dist = intergroup["native_com_distance"]
    sample_dist = intergroup["sample_com_distance"]
    assert intergroup["com_distance_error"] == pytest.approx(abs(sample_dist - native_dist))


def test_compute_run_skips_samples_without_group_metadata(tmp_path):
    native_path = tmp_path / "native.pdb"
    sample_path = tmp_path / "sample.pdb"
    _write_minimal_pdb(native_path, {("A", 1): np.array([0.0, 0.0, 0.0])})
    _write_minimal_pdb(sample_path, {("A", 1): np.array([0.0, 0.0, 0.0])})
    metadata = {
        "with_groups": {
            "path": str(sample_path),
            "native_pdb": str(native_path),
            "name": "task1",
            "source_residue_keys": [
                {
                    "native_chain": "A",
                    "native_res_idx": 1,
                    "output_chain": "A",
                    "output_res_idx": 1,
                    "group_id": 1,
                }
            ],
        },
        "without_groups": {
            "path": str(sample_path),
            "native_pdb": str(native_path),
            "name": "task1",
        },
    }
    df = compute_run_multimotif_metrics(metadata)
    assert set(df["name"].unique()) == {"with_groups"}


def test_aggregate_multimotif_slices():
    df = pd.DataFrame(
        [
            {
                "record_type": "group",
                "name": "a",
                "num_motif_groups": 2,
                "group_size": 2,
                "native_min_intergroup_dist": 15.0,
                "motif_ha_rmsd": 1.0,
                "motif_ca_rmsd": 0.5,
            },
            {
                "record_type": "group",
                "name": "b",
                "num_motif_groups": 2,
                "group_size": 2,
                "native_min_intergroup_dist": 25.0,
                "motif_ha_rmsd": 3.0,
                "motif_ca_rmsd": 1.5,
            },
        ]
    )
    summary = aggregate_multimotif_slices(df)
    assert not summary.empty
    assert "slice_type" in summary.columns
    assert "group_pass_rate" in summary.columns


def test_write_overlay_pdb_shows_design_with_native_motifs_on_motif_sites(tmp_path):
    native_coords = {
        ("A", 1): np.array([0.0, 0.0, 0.0]),
        ("A", 2): np.array([3.8, 0.0, 0.0]),
        ("A", 10): np.array([20.0, 0.0, 0.0]),
        ("A", 11): np.array([23.8, 0.0, 0.0]),
    }
    design_coords = {
        ("A", 1): np.array([5.0, 0.0, 0.0]),
        ("A", 2): np.array([8.8, 0.0, 0.0]),
        ("A", 10): np.array([25.0, 0.0, 0.0]),
        ("A", 11): np.array([28.8, 0.0, 0.0]),
    }
    folded_coords = {
        ("A", 1): np.array([5.2, 0.1, 0.0]),
        ("A", 2): np.array([9.0, 0.1, 0.0]),
        ("A", 10): np.array([25.2, 0.1, 0.0]),
        ("A", 11): np.array([29.0, 0.1, 0.0]),
    }
    native_path = tmp_path / "native.pdb"
    design_path = tmp_path / "design.pdb"
    folded_path = tmp_path / "folded.pdb"
    _write_minimal_pdb(native_path, native_coords)
    _write_minimal_pdb(design_path, design_coords)
    _write_minimal_pdb(folded_path, folded_coords)

    parser = PDBParser(QUIET=True)
    native_structure = parser.get_structure("native", str(native_path))
    design_structure = parser.get_structure("design", str(design_path))
    folded_structure = parser.get_structure("folded", str(folded_path))
    group_pairs = {
        1: [(("A", 1), ("A", 1)), (("A", 2), ("A", 2))],
        2: [(("A", 10), ("A", 10)), (("A", 11), ("A", 11))],
    }

    output_pdb = tmp_path / "overlay.pdb"
    write_multimotif_overlay_pdb(
        output_pdb,
        sample_name="sample1",
        variant_label="one_shot",
        scaffold_structure=design_structure,
        native_structure=native_structure,
        folded_structure=folded_structure,
        group_pairs=group_pairs,
    )

    pdb_text = output_pdb.read_text(encoding="utf-8")
    assert " chain A " in pdb_text
    assert " chain B " in pdb_text
    assert " chain C " in pdb_text
    assert " chain E " in pdb_text
    assert "native_overlay_chain=B" in pdb_text
    assert "chain A = ESMFold folded structure" in pdb_text
    # ESMFold CA stays on chain A; native overlay follows folded motif-1 CA.
    assert " A   1 " in pdb_text
    assert "5.200" in pdb_text or "5.199" in pdb_text
    assert " B   1 " in pdb_text
    assert " E   1 " in pdb_text


def test_write_run_overlay_pdbs_writes_one_file_per_esmfold_variant(tmp_path):
    native_coords = {("A", 1): np.array([0.0, 0.0, 0.0])}
    design_coords = {("A", 1): np.array([1.0, 0.0, 0.0])}
    native_path = tmp_path / "native.pdb"
    design_path = tmp_path / "design.pdb"
    _write_minimal_pdb(native_path, native_coords)
    _write_minimal_pdb(design_path, design_coords)

    esmfold_dir = tmp_path / "esmfold" / "with_groups"
    esmfold_dir.mkdir(parents=True)
    _write_minimal_pdb(esmfold_dir / "one, model_name=v_48_020.pdb", design_coords)
    _write_minimal_pdb(esmfold_dir / "T=0.1, sample=1, score=0.9.pdb", design_coords)

    metadata = {
        "with_groups": {
            "path": str(design_path),
            "native_pdb": str(native_path),
            "name": "task1",
            "source_residue_keys": [
                {
                    "native_chain": "A",
                    "native_res_idx": 1,
                    "output_chain": "A",
                    "output_res_idx": 1,
                    "group_id": 1,
                }
            ],
        },
    }
    alignment_dir = tmp_path / "alignments"
    written = write_run_multimotif_overlay_pdbs(
        metadata,
        esmfold_dir=tmp_path / "esmfold",
        alignment_pdb_dir=alignment_dir,
    )
    assert written == 2
    assert (alignment_dir / "with_groups" / "one_shot.pdb").is_file()
    assert (alignment_dir / "with_groups" / "pmpnn_sample_1.pdb").is_file()


def test_integration_with_benchmark_pdb(tmp_path):
    pdb_path = _repo_root() / "configs/sampling/protpardelle_motif_scaffolding/pdbs/1PRW.pdb"
    if not pdb_path.is_file():
        pytest.skip(f"Missing benchmark PDB fixture: {pdb_path}")

    sample_path = tmp_path / "sample.pdb"
    sample_path.write_bytes(pdb_path.read_bytes())
    metadata = {
        "path": str(sample_path),
        "native_pdb": str(pdb_path),
        "name": "1PRW",
        "source_residue_keys": [
            {
                "native_chain": "A",
                "native_res_idx": res_idx,
                "output_chain": "A",
                "output_res_idx": res_idx,
                "group_id": group_id,
            }
            for group_id, residues in ((1, (16, 17)), (2, (52, 53)))
            for res_idx in residues
        ],
    }
    rows = compute_sample_multimotif_metrics("sample1", metadata)
    assert rows
    group_rows = [row for row in rows if row["record_type"] == "group"]
    assert len(group_rows) == 2
