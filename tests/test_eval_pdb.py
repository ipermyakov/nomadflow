"""Tests for single-chain eval PDB export from multimotif sampling outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mashumaro")

from dataclasses import replace

from nomadflow.boltz.data import const
from nomadflow.boltz.data.types import Atom, Chain, Interface, SamplingResidue, Structure
from nomadflow.data.featurize.tokenize import Rigid, Token, TokenBond, Tokenized
from nomadflow.data.write import trajectory as traj


def _build_scaffold_plus_aggregator_tokenized() -> Tokenized:
    """Minimal scaffold + virtual aggregator marker (no copy chain)."""
    residues = np.zeros(1, dtype=SamplingResidue)
    residues["name"] = ["ALA"]
    residues["res_type"] = [const.token_ids["ALA"]]
    residues["res_idx"] = [0]
    residues["atom_num"] = [len(const.ref_atoms["ALA"])]
    residues["atom_center"] = 1
    residues["atom_disto"] = 1
    residues["is_standard"] = True
    residues["is_present"] = True
    residues["is_copy"] = False

    chains = np.zeros(1, dtype=Chain)
    chains["name"] = ["A"]
    chains["mol_type"] = const.chain_type_ids["PROTEIN"]
    chains["entity_id"] = 0
    chains["sym_id"] = 0
    chains["asym_id"] = 0
    chains["res_idx"] = 0
    chains["res_num"] = 1

    tokens = np.zeros(2, dtype=Token)
    tokens["token_idx"] = [0, 1]
    tokens["res_idx"] = [0, -1]
    tokens["res_type"] = [const.token_ids["ALA"], const.token_ids["GLY"]]
    tokens["res_name"] = ["ALA", "VGM"]
    tokens["rigid_idx"] = [0, 3]
    tokens["rigid_num"] = [3, 1]
    tokens["sym_id"] = [0, -1]
    tokens["asym_id"] = [0, -1]
    tokens["entity_id"] = [0, -1]
    tokens["mol_type"] = const.chain_type_ids["PROTEIN"]
    tokens["resolved_mask"] = True
    tokens["center_coords"] = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    tokens["is_copy"] = False
    tokens["is_unindexed"] = [False, True]
    tokens["is_atomized"] = False
    tokens["seq_noising_mask"] = False
    tokens["motif_group_id"] = [0, 1]
    tokens["is_motif_group_aggregator"] = [False, True]

    rigids = np.zeros(4, dtype=Rigid)
    rigids["rigid_idx"] = np.arange(4)
    rigids["token_idx"] = [0, 0, 0, 1]
    rigids["sidechain_idx"] = [0, 1, 2, 0]
    rigids["tensor7"][:, 0] = 1.0
    rigids["tensor7"][3, 4:] = [10.0, 0.0, 0.0]
    rigids["is_present"] = True
    rigids["rigids_noising_mask"] = True
    rigids["num_real_input_axes"] = 2
    rigids["motif_group_id"] = [0, 0, 0, 1]
    rigids["is_motif_group_aggregator"] = [False, False, False, True]

    return Tokenized(
        tokens=tokens,
        rigids=rigids,
        bonds=np.zeros(0, dtype=TokenBond),
        structure=Structure(
            atoms=np.zeros(0, dtype=Atom),
            bonds=np.zeros(0, dtype=[]),
            residues=residues,
            chains=chains,
            connections=np.zeros(0, dtype=[]),
            interfaces=np.zeros(0, dtype=Interface),
            mask=np.array([True]),
        ),
    )


def _build_multimotif_tokenized_with_copy() -> Tokenized:
    """Scaffold + copy + aggregator tokens for mask/filter tests only."""
    data = _build_scaffold_plus_aggregator_tokenized()
    tokens = np.zeros(3, dtype=Token)
    tokens["token_idx"] = [0, 1, 2]
    tokens["res_idx"] = [0, 10, -1]
    tokens["is_copy"] = [False, True, False]
    tokens["is_motif_group_aggregator"] = [False, False, True]
    tokens["motif_group_id"] = [0, 1, 1]
    return replace(data, tokens=tokens)


def _build_unindexed_copy_with_aggregator_tokenized() -> Tokenized:
    """Scaffold + unindexed motif copies + aggregator, as after lmod token_idx overwrite.

    Scaffold residues have predicted MET; copy residues hold native ASN/TYR.
    Copy ``token_idx`` stores assigned scaffold positions (0, 1) — the contract
    ``construct_atoms`` uses to paste motif sequence onto chain A.
    """
    residues = np.zeros(4, dtype=SamplingResidue)
    residues["name"] = ["MET", "MET", "ASN", "TYR"]
    residues["res_type"] = [
        const.token_ids["MET"],
        const.token_ids["MET"],
        const.token_ids["ASN"],
        const.token_ids["TYR"],
    ]
    residues["res_idx"] = [0, 1, 2, 3]
    residues["atom_num"] = [
        len(const.ref_atoms["MET"]),
        len(const.ref_atoms["MET"]),
        len(const.ref_atoms["ASN"]),
        len(const.ref_atoms["TYR"]),
    ]
    residues["atom_center"] = 1
    residues["atom_disto"] = 1
    residues["is_standard"] = True
    residues["is_present"] = True
    residues["is_copy"] = [False, False, True, True]

    chains = np.zeros(2, dtype=Chain)
    chains["name"] = ["A", "B"]
    chains["mol_type"] = const.chain_type_ids["PROTEIN"]
    chains["entity_id"] = [0, 1]
    chains["sym_id"] = [0, 1]
    chains["asym_id"] = [0, 1]
    chains["res_idx"] = [0, 2]
    chains["res_num"] = [2, 2]

    # token order: scaffold0, scaffold1, copy0, copy1, aggregator
    # After lmod: copy token_idx overwritten with assigned scaffold res_idx.
    tokens = np.zeros(5, dtype=Token)
    tokens["token_idx"] = [0, 1, 0, 1, 4]  # copies: assigned scaffold idx 0, 1
    tokens["res_idx"] = [0, 1, 2, 3, -1]
    tokens["res_type"] = [
        const.token_ids["MET"],
        const.token_ids["MET"],
        const.token_ids["ASN"],
        const.token_ids["TYR"],
        const.token_ids["GLY"],
    ]
    tokens["res_name"] = ["MET", "MET", "ASN", "TYR", "VGM"]
    tokens["rigid_idx"] = [0, 3, 6, 9, 12]
    tokens["rigid_num"] = [3, 3, 3, 3, 1]
    tokens["sym_id"] = [0, 0, 1, 1, -1]
    tokens["asym_id"] = [0, 0, 1, 1, -1]
    tokens["entity_id"] = [0, 0, 1, 1, -1]
    tokens["mol_type"] = const.chain_type_ids["PROTEIN"]
    tokens["resolved_mask"] = True
    tokens["center_coords"] = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    tokens["is_copy"] = [False, False, True, True, False]
    tokens["is_unindexed"] = [False, False, True, True, True]
    tokens["is_atomized"] = False
    tokens["seq_noising_mask"] = [True, True, False, False, False]
    tokens["motif_group_id"] = [0, 0, 1, 1, 1]
    tokens["is_motif_group_aggregator"] = [False, False, False, False, True]

    rigids = np.zeros(13, dtype=Rigid)
    rigids["rigid_idx"] = np.arange(13)
    rigids["token_idx"] = [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 4]
    rigids["sidechain_idx"] = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    rigids["tensor7"][:, 0] = 1.0
    # Place scaffold+copy CA-like translations so frames are valid.
    for token in tokens:
        start = int(token["rigid_idx"])
        n = int(token["rigid_num"])
        for j in range(n):
            rigids["tensor7"][start + j, 4:] = token["center_coords"]
    rigids["is_present"] = True
    rigids["rigids_noising_mask"] = True
    rigids["num_real_input_axes"] = 2
    rigids["motif_group_id"] = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    rigids["is_motif_group_aggregator"] = [False] * 12 + [True]

    return Tokenized(
        tokens=tokens,
        rigids=rigids,
        bonds=np.zeros(0, dtype=TokenBond),
        structure=Structure(
            atoms=np.zeros(0, dtype=Atom),
            bonds=np.zeros(0, dtype=[]),
            residues=residues,
            chains=chains,
            connections=np.zeros(0, dtype=[]),
            interfaces=np.zeros(0, dtype=Interface),
            mask=np.array([True, True]),
        ),
    )


def _chain_resnames_by_resnum(pdb_path: Path, chain_name: str) -> dict[int, str]:
    names: dict[int, str] = {}
    for line in pdb_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[21:22] != chain_name:
            continue
        if line[12:16].strip() != "CA":
            continue
        resnum = int(line[22:26])
        names[resnum] = line[17:20].strip()
    return names


def _chain_ca_coords_by_resnum(pdb_path: Path, chain_name: str) -> dict[int, np.ndarray]:
    coords: dict[int, np.ndarray] = {}
    for line in pdb_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[21:22] != chain_name:
            continue
        if line[12:16].strip() != "CA":
            continue
        resnum = int(line[22:26])
        coords[resnum] = np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=np.float64,
        )
    return coords


def test_scaffold_token_mask_drops_copy_and_aggregator():
    data = _build_multimotif_tokenized_with_copy()
    keep = traj.scaffold_token_mask(data.tokens)
    assert keep.tolist() == [True, False, False]
    filtered = traj.filter_tokenized_for_eval_pdb(data)
    assert len(filtered.tokens) == 1
    assert not filtered.tokens["is_copy"].any()
    assert not filtered.tokens["is_motif_group_aggregator"].any()


def test_unindexed_motif_seq_pasted_on_scaffold_with_aggregators(tmp_path):
    """Aggregator PDB adapt must not clobber lmod-assigned scaffold token_idx."""
    data = _build_unindexed_copy_with_aggregator_tokenized()
    full_path = tmp_path / "full.pdb"
    eval_path = tmp_path / "eval.pdb"

    traj.write_tokenized_structure_pdb(
        full_path,
        data,
        output_motif_chains=True,
        take_copy_structure=True,
    )
    traj.write_eval_tokenized_structure_pdb(
        eval_path,
        data,
        sample_chain_name="A",
        take_copy_structure=True,
    )

    full_a = _chain_resnames_by_resnum(full_path, "A")
    full_b = _chain_resnames_by_resnum(full_path, "B")
    eval_a = _chain_resnames_by_resnum(eval_path, "A")

    assert full_b == {1: "ASN", 2: "TYR"}
    assert full_a == {1: "ASN", 2: "TYR"}
    assert eval_a == {1: "ASN", 2: "TYR"}
    assert {"A", "B", "V"} == {
        line[21:22]
        for line in full_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("ATOM")
    }


def test_adapt_preserves_unindexed_copy_assigned_token_idx():
    data = _build_unindexed_copy_with_aggregator_tokenized()
    adapted = traj._adapt_motif_group_aggregators_for_pdb(data)
    copy_tokens = adapted.tokens[adapted.tokens["is_copy"]]
    assert copy_tokens["token_idx"].tolist() == [0, 1]
    assert copy_tokens["res_type"].tolist() == [
        const.token_ids["ASN"],
        const.token_ids["TYR"],
    ]


def test_eval_pdb_is_single_chain_with_sample_length(tmp_path):
    data = _build_scaffold_plus_aggregator_tokenized()
    full_path = tmp_path / "full.pdb"
    eval_path = tmp_path / "eval.pdb"

    traj.write_tokenized_structure_pdb(
        full_path,
        data,
        output_motif_chains=True,
        take_copy_structure=False,
    )
    traj.write_eval_tokenized_structure_pdb(eval_path, data, sample_chain_name="A")

    full_chains = {
        line[21:22]
        for line in full_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("ATOM")
    }
    eval_chains = {
        line[21:22]
        for line in eval_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("ATOM")
    }

    assert full_chains == {"A", "V"}
    assert eval_chains == {"A"}
    assert len(_chain_ca_coords_by_resnum(eval_path, "A")) == 1


def test_eval_pdb_scaffold_ca_coords_match_multichain_regression(tmp_path):
    data = _build_scaffold_plus_aggregator_tokenized()
    full_path = tmp_path / "full.pdb"
    eval_path = tmp_path / "eval.pdb"
    traj.write_tokenized_structure_pdb(
        full_path,
        data,
        output_motif_chains=True,
        take_copy_structure=False,
    )
    traj.write_eval_tokenized_structure_pdb(eval_path, data, sample_chain_name="A")

    full_chain_a = _chain_ca_coords_by_resnum(full_path, "A")
    eval_chain_a = _chain_ca_coords_by_resnum(eval_path, "A")
    assert full_chain_a.keys() == eval_chain_a.keys()
    for resnum in full_chain_a:
        assert np.allclose(full_chain_a[resnum], eval_chain_a[resnum], atol=1e-3)


def test_resolve_samples_dir_prefers_samples_eval(tmp_path):
    from nomadflow.benchmark.analysis_inputs import resolve_samples_dir

    (tmp_path / "samples").mkdir()
    (tmp_path / "samples_eval").mkdir()
    assert resolve_samples_dir(tmp_path).name == "samples_eval"


def test_prepare_eval_pdbs_keeps_native_chain_when_a_missing(tmp_path):
    from scripts.analysis.prepare_eval_pdbs_from_samples import (
        prepare_eval_pdbs_from_multichain_pdb,
    )

    src = tmp_path / "native_b.pdb"
    src.write_text(
        "ATOM      2  CA  ALA B   7       1.000   2.000   3.000  1.00  0.00           C  \n"
        "END\n",
        encoding="utf-8",
    )
    out = tmp_path / "eval.pdb"
    written = prepare_eval_pdbs_from_multichain_pdb(
        src, out, sample_chain_name="A"
    )
    assert written == "B"
    lines = [line for line in out.read_text(encoding="utf-8").splitlines() if line.startswith("ATOM")]
    assert lines[0][21] == "B"
    assert int(lines[0][22:26]) == 7


SMOKE_SAMPLES = Path(
    "article_results/sampling_runs/ours_multimotif__1qjg_smoke__best__20260701/samples"
)


def _first_smoke_sample_name() -> str:
    pdbs = sorted(SMOKE_SAMPLES.glob("*.pdb"))
    assert pdbs, f"no PDBs in {SMOKE_SAMPLES}"
    return pdbs[0].stem


@pytest.mark.skipif(not SMOKE_SAMPLES.is_dir(), reason="smoke run samples not available")
def test_posthoc_prepare_eval_pdbs_from_smoke_run(tmp_path):
    from scripts.analysis.prepare_eval_pdbs_from_samples import (
        prepare_eval_pdbs_from_multichain_pdb,
    )

    sample_name = _first_smoke_sample_name()
    full_path = SMOKE_SAMPLES / f"{sample_name}.pdb"
    eval_path = tmp_path / f"{sample_name}.pdb"
    prepare_eval_pdbs_from_multichain_pdb(full_path, eval_path, sample_chain_name="A")

    full_chains = {
        line[21:22]
        for line in full_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("ATOM")
    }
    eval_chains = {
        line[21:22]
        for line in eval_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("ATOM")
    }
    assert "A" in full_chains
    assert eval_chains == {"A"}

    full_chain_a = _chain_ca_coords_by_resnum(full_path, "A")
    eval_chain_a = _chain_ca_coords_by_resnum(eval_path, "A")
    assert eval_chain_a.keys() == full_chain_a.keys()
    for resnum in full_chain_a:
        assert np.allclose(full_chain_a[resnum], eval_chain_a[resnum], atol=1e-3)


@pytest.mark.skipif(not SMOKE_SAMPLES.is_dir(), reason="smoke run samples not available")
def test_smoke_run_motif_positions_on_scaffold_chain():
    metadata_path = SMOKE_SAMPLES.parent / "samples_metadata.json"
    sample_name = _first_smoke_sample_name()
    metadata = __import__("json").loads(metadata_path.read_text(encoding="utf-8"))[sample_name]

    full_path = SMOKE_SAMPLES / f"{sample_name}.pdb"
    scaffold_ca_resnums = set(_chain_ca_coords_by_resnum(full_path, "A"))
    for res_idx in metadata["fixed_seq_res_idx"]:
        assert res_idx in scaffold_ca_resnums
    sample_length = metadata.get("sample_length") or metadata.get("length")
    if sample_length is not None:
        assert int(sample_length) == len(scaffold_ca_resnums)
