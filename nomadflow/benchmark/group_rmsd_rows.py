"""Helpers for per-motif-group RMSD CSV rows."""

from __future__ import annotations

from typing import Any


def build_group_rmsd_row(
    *,
    name: str,
    task: str,
    path: str,
    sample: int,
    group_id: str,
    num_group_res: int,
    all_atom_rmsd: float,
    ca_rmsd: float,
    bb_rmsd: float,
    rmsd_status: str,
) -> dict[str, Any]:
    """Build one group_rmsd.csv row from computed metrics."""
    return {
        "name": name,
        "task": task,
        "path": path,
        "sample": sample,
        "group_id": group_id,
        "num_group_res": num_group_res,
        "motif_group_all_atom_rmsd": all_atom_rmsd,
        "motif_group_ca_rmsd": ca_rmsd,
        "motif_group_bb_rmsd": bb_rmsd,
        "rmsd_status": rmsd_status,
    }


def pooled_rmsd_outputs_enabled(only_group_rmsd: bool) -> bool:
    """When --only-group-rmsd, do not overwrite folding_rmsd.csv / sc_rmsd.csv."""
    return not only_group_rmsd
