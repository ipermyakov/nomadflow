from __future__ import annotations

from nomadflow.benchmark.group_rmsd_rows import (
    build_group_rmsd_row,
    pooled_rmsd_outputs_enabled,
)


def test_build_group_rmsd_row_maps_expected_columns():
    row = build_group_rmsd_row(
        name="sample_a",
        task="task_1",
        path="/tmp/sample_a.pdb",
        sample=2,
        group_id="1",
        num_group_res=3,
        all_atom_rmsd=1.23,
        ca_rmsd=0.45,
        bb_rmsd=0.67,
        rmsd_status="ok",
    )

    assert row == {
        "name": "sample_a",
        "task": "task_1",
        "path": "/tmp/sample_a.pdb",
        "sample": 2,
        "group_id": "1",
        "num_group_res": 3,
        "motif_group_all_atom_rmsd": 1.23,
        "motif_group_ca_rmsd": 0.45,
        "motif_group_bb_rmsd": 0.67,
        "rmsd_status": "ok",
    }


def test_pooled_rmsd_outputs_disabled_for_group_only():
    assert pooled_rmsd_outputs_enabled(True) is False
    assert pooled_rmsd_outputs_enabled(False) is True
