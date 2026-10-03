"""Unified success criteria from the ProteinZen paper (Li et al., §3.2.1 / Table H.2).

One-shot (design vs ESMFold): motif HA < 2 Å, motif Cα < 1 Å, global HA < 2 Å.
PMPNN+ESMFold: the same motif cuts; global BB < 2 Å instead of global HA.
Per-group checks use the same motif cuts (HA 2 / Cα 1). Group tables may
supply BB when a Cα column is absent; the numeric backbone cut is still 1.0 Å.
Strict `<` throughout. These are not the stricter upstream
`cluster_motif_successes.py` HA 1.5 thresholds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

CRITERIA_VERSION = "v3_groups_vs_native"

MOTIF_HA_RMSD_MAX = 2.0
MOTIF_CA_RMSD_MAX = 1.0
# Alias: group fold-vs-design tables may only have BB; paper backbone cut is Cα.
MOTIF_BB_RMSD_MAX = MOTIF_CA_RMSD_MAX
GLOBAL_HA_RMSD_MAX = 2.0
SSC_RMSD_MAX = 2.0
FOLDSEEK_UNIQUE_TM_THRESHOLD = 0.5


def _motif_ca_rmsd(row: pd.Series) -> float:
    """Paper motif backbone cut is Cα; fall back to BB if Cα is missing."""
    if "motif_ca_rmsd" in row.index and pd.notna(row["motif_ca_rmsd"]):
        return float(row["motif_ca_rmsd"])
    return float(row["motif_bb_rmsd"])


def one_shot_pass(row: pd.Series) -> bool:
    return bool(
        row["motif_all_atom_rmsd"] < MOTIF_HA_RMSD_MAX
        and _motif_ca_rmsd(row) < MOTIF_CA_RMSD_MAX
        and row["global_all_atom_rmsd"] < GLOBAL_HA_RMSD_MAX
    )


def pmpnn_esmfold_pass(row: pd.Series) -> bool:
    return bool(
        row["motif_all_atom_rmsd"] < MOTIF_HA_RMSD_MAX
        and _motif_ca_rmsd(row) < MOTIF_CA_RMSD_MAX
        and row["global_bb_rmsd"] < SSC_RMSD_MAX
    )


def group_ha_bb_pass(ha_rmsd: float, bb_rmsd: float) -> bool:
    """Per-group success: HA < 2.0 Å and Cα/BB < 1.0 Å (paper §3.2.1)."""
    return bool(ha_rmsd < MOTIF_HA_RMSD_MAX and bb_rmsd < MOTIF_CA_RMSD_MAX)


def all_groups_pass(
    group_ha: Sequence[float] | None,
    group_bb: Sequence[float] | None,
    *,
    expected_n_groups: int | None = None,
) -> bool:
    """True iff every expected group has HA < 2.0 and Cα/BB < 1.0.

    Missing group rows, empty lists, or a count mismatch are failures — never a
    silent pass.
    """
    if group_ha is None or group_bb is None:
        return False
    if len(group_ha) == 0 or len(group_bb) == 0:
        return False
    if len(group_ha) != len(group_bb):
        return False
    if expected_n_groups is None or expected_n_groups < 1:
        return False
    if len(group_ha) != int(expected_n_groups):
        return False
    return all(
        group_ha_bb_pass(float(ha), float(bb)) for ha, bb in zip(group_ha, group_bb)
    )


def one_shot_final_pass(row: pd.Series) -> bool:
    """Publication one-shot success: every motif group vs native (missing/false → fail)."""
    if "pass_all_groups" in row.index:
        return bool(row["pass_all_groups"])
    return False


def pmpnn_final_pass(row: pd.Series) -> bool:
    """Publication PMPNN success: every motif group vs native (missing/false → fail)."""
    if "pass_all_groups" in row.index:
        return bool(row["pass_all_groups"])
    return False


def filter_one_shot(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.apply(one_shot_pass, axis=1)]


def filter_pmpnn_esmfold(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.apply(pmpnn_esmfold_pass, axis=1)]


def filter_one_shot_final(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "pass_final" in df.columns:
        return df[df["pass_final"].astype(bool)]
    return df[df.apply(one_shot_final_pass, axis=1)]


def filter_pmpnn_final(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "pass_final" in df.columns:
        return df[df["pass_final"].astype(bool)]
    return df[df.apply(pmpnn_final_pass, axis=1)]


def expected_group_count(metadata: Mapping | None) -> int:
    """Number of annotated motif groups for a sample (0 if unknown)."""
    if not isinstance(metadata, Mapping):
        return 0
    source_residue_keys = metadata.get("source_residue_keys")
    if isinstance(source_residue_keys, list) and source_residue_keys:
        group_ids = {
            int(entry["group_id"])
            for entry in source_residue_keys
            if isinstance(entry, dict) and entry.get("group_id") is not None
        }
        if group_ids:
            return len(group_ids)
    motif_groups = metadata.get("motif_groups")
    if isinstance(motif_groups, list) and motif_groups:
        return len(motif_groups)
    return 0
