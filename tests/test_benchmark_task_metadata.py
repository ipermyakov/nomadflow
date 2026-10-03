import json
from pathlib import Path

import numpy as np
import omegaconf
import pytest

from nomadflow.data.featurize.assembler import collate
from nomadflow.runtime.sampling.benchmark_metadata import (
    attach_output_chains_to_sample_metadata,
    build_motif_scaffolding_sample_metadata,
    build_motif_scaffolding_task_metadata,
    finalize_motif_sample_metadata,
)
from nomadflow.runtime.sampling.motif_scaffolding import MotifScaffoldingTask


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_build_motif_scaffolding_task_metadata_is_json_serializable():
    pdb_path = (
        _repo_root()
        / "configs/sampling/protpardelle_motif_scaffolding/pdbs/1PRW.pdb"
    )
    if not pdb_path.is_file():
        pytest.skip(f"Missing benchmark PDB fixture: {pdb_path}")

    task = MotifScaffoldingTask(
        pdb=str(pdb_path),
        pdb_contigs="A16-35,A52-71",
        num_samples=2,
        total_length="60-150",
        contigs_idx="",
        name="1PRW",
        motif_groups=[["A16-35"], ["A52-71"]],
        multi_motif_se3=True,
        multi_motif_partition_mode="fragments",
        multi_motif_max_groups=8,
        multi_motif_se3_trans_std=12.0,
        motif_t_lock=0.5,
    )

    metadata = build_motif_scaffolding_task_metadata(task)

    assert metadata["native_pdb"] == str(pdb_path)
    assert metadata["pdb_contigs"] == "A16-35,A52-71"
    assert metadata["motif_groups"] == [["A16-35"], ["A52-71"]]
    assert metadata["multi_motif_se3"] is True
    assert metadata["motif_t_lock"] == 0.5
    assert metadata["name"] == "1PRW"

    json.dumps(metadata)


def test_build_motif_scaffolding_task_metadata_converts_omegaconf():
    pdb_path = (
        _repo_root()
        / "configs/sampling/protpardelle_motif_scaffolding/pdbs/1PRW.pdb"
    )
    if not pdb_path.is_file():
        pytest.skip(f"Missing benchmark PDB fixture: {pdb_path}")

    task = MotifScaffoldingTask(
        pdb=str(pdb_path),
        pdb_contigs="A16-35,A52-71",
        num_samples=2,
        total_length="60-150",
        contigs_idx="",
        name="1PRW",
        multi_motif_se3=True,
        motif_t_lock=0.5,
    )
    task.kwargs["motif_groups"] = omegaconf.OmegaConf.create(
        [["A16-35"], ["A52-71"]]
    )

    metadata = build_motif_scaffolding_task_metadata(task)

    assert metadata["motif_groups"] == [["A16-35"], ["A52-71"]]
    assert not omegaconf.OmegaConf.is_config(metadata["motif_groups"])
    json.dumps(metadata)


def test_featurize_inference_and_collate_preserve_task_metadata():
    pdb_path = (
        _repo_root()
        / "configs/sampling/protpardelle_motif_scaffolding/pdbs/1PRW.pdb"
    )
    if not pdb_path.is_file():
        pytest.skip(f"Missing benchmark PDB fixture: {pdb_path}")

    task = MotifScaffoldingTask(
        pdb=str(pdb_path),
        pdb_contigs="A16-35,A52-71",
        num_samples=1,
        total_length="60-60",
        contigs_idx="",
        name="1PRW",
        motif_groups=[["A16-35"], ["A52-71"]],
        multi_motif_se3=True,
        motif_t_lock=0.5,
    )
    sample = next(task.sample_data())

    assert "task_metadata" in sample
    assert sample["task_metadata"]["native_pdb"] == str(pdb_path)

    batch = collate([sample])

    assert batch["task_metadata"] == [sample["task_metadata"]]


def test_build_motif_scaffolding_sample_metadata():
    pdb_path = (
        _repo_root()
        / "configs/sampling/protpardelle_motif_scaffolding/pdbs/1PRW.pdb"
    )
    if not pdb_path.is_file():
        pytest.skip(f"Missing benchmark PDB fixture: {pdb_path}")

    task = MotifScaffoldingTask(
        pdb=str(pdb_path),
        pdb_contigs="A16-35,A52-71",
        num_samples=1,
        total_length="60-60",
        contigs_idx="",
        name="1PRW",
        motif_groups=[["A16-35"], ["A52-71"]],
        multi_motif_se3=True,
        motif_t_lock=0.5,
    )
    sample = next(task.sample_data())

    assert "sample_metadata" in sample
    metadata = sample["sample_metadata"]
    assert metadata["sample_length"] == 60
    assert len(metadata["source_residue_keys"]) == 40

    group_one = [e for e in metadata["source_residue_keys"] if e["group_id"] == 1]
    group_two = [e for e in metadata["source_residue_keys"] if e["group_id"] == 2]
    assert len(group_one) == 20
    assert len(group_two) == 20
    assert group_one[0]["native_key"] == "A16"
    assert group_two[0]["native_key"] == "A52"

    json.dumps(metadata)


def test_finalize_motif_sample_metadata():
    raw = {
        "source_residue_keys": [
            {
                "native_key": "A16",
                "native_chain": "A",
                "native_res_idx": 16,
                "lookup_asym_id": 1,
                "lookup_res_idx": 75,
                "is_unindexed": True,
                "group_id": 1,
            }
        ],
        "sample_length": 60,
    }
    token_data = {
        "is_copy": np.array([False, True]),
        "asym_id": np.array([0, 1]),
        "res_idx": np.array([10, 75]),
    }
    motif_idx = np.array([10, 42])

    finalized = finalize_motif_sample_metadata(token_data, motif_idx, raw)

    assert finalized is not None
    entry = finalized["source_residue_keys"][0]
    assert entry["output_res_idx"] == 43
    assert entry["output_asym_id"] == 1
    assert entry["group_id"] == 1

    with_chains = attach_output_chains_to_sample_metadata(
        finalized,
        {0: "A", 1: "B"},
    )
    assert with_chains["source_residue_keys"][0]["output_chain"] == "B"
    assert with_chains["motif_group_assignments"] == {"B:43": 1}


def test_collate_preserves_sample_metadata():
    pdb_path = (
        _repo_root()
        / "configs/sampling/protpardelle_motif_scaffolding/pdbs/1PRW.pdb"
    )
    if not pdb_path.is_file():
        pytest.skip(f"Missing benchmark PDB fixture: {pdb_path}")

    task = MotifScaffoldingTask(
        pdb=str(pdb_path),
        pdb_contigs="A16-35,A52-71",
        num_samples=1,
        total_length="60-60",
        contigs_idx="",
        name="1PRW",
        motif_groups=[["A16-35"], ["A52-71"]],
        multi_motif_se3=True,
        motif_t_lock=0.5,
    )
    sample = next(task.sample_data())

    batch = collate([sample])

    assert batch["sample_metadata"] == [sample["sample_metadata"]]


def test_build_motif_scaffolding_task_metadata_passes_through_eval_fields():
    pdb_path = (
        _repo_root()
        / "configs/sampling/protpardelle_motif_scaffolding/pdbs/1PRW.pdb"
    )
    if not pdb_path.is_file():
        pytest.skip(f"Missing benchmark PDB fixture: {pdb_path}")

    task = MotifScaffoldingTask(
        pdb=str(pdb_path),
        pdb_contigs="A16-35,A52-71",
        num_samples=1,
        total_length="60-60",
        contigs_idx="",
        name="1PRW",
        multi_motif_se3=True,
        motif_t_lock=0.5,
        source_layout_id="layout_0",
        motif_reference_residues=[12, 45],
        rmsd_reference_chain="A",
        stage2_input_pdb=str(pdb_path),
        stage2_pdb_contigs="B1,B2",
    )

    metadata = build_motif_scaffolding_task_metadata(task)

    assert metadata["source_layout_id"] == "layout_0"
    assert metadata["motif_reference_residues"] == [12, 45]
    assert metadata["rmsd_reference_chain"] == "A"
    assert metadata["stage2_input_pdb"] == str(pdb_path)
    assert metadata["stage2_pdb_contigs"] == "B1,B2"
