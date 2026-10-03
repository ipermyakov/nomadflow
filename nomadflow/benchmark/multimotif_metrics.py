"""Multi-motif benchmark metrics: per-group RMSD and inter-group geometry vs native."""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from scipy.spatial.transform import Rotation

from nomadflow.benchmark.analysis_inputs import list_esmfold_structure_paths
from nomadflow.benchmark.motif_group_annotation import apply_source_residue_key_repair

logger = logging.getLogger(__name__)

_HEAVY_ATOM_PREFIXES = ("C", "N", "O", "S")
DesignStructureSource = Literal["sample", "esmfold"]


def _is_heavy_atom(atom_name: str) -> bool:
    name = atom_name.strip()
    if name.startswith("H"):
        return False
    return name[0] in _HEAVY_ATOM_PREFIXES


def _get_residue(structure, chain_id: str, res_idx: int):
    chain = structure[0][chain_id]
    return chain[res_idx]


def _ca_coord(structure, chain_id: str, res_idx: int) -> np.ndarray:
    residue = _get_residue(structure, chain_id, res_idx)
    return np.asarray(residue["CA"].coord, dtype=np.float64)


def _heavy_atom_pairs(
    native_structure,
    sample_structure,
    residue_pairs: list[tuple[tuple[str, int], tuple[str, int]]],
) -> tuple[np.ndarray, np.ndarray]:
    native_coords: list[np.ndarray] = []
    sample_coords: list[np.ndarray] = []
    for (native_chain, native_res_idx), (sample_chain, sample_res_idx) in residue_pairs:
        native_res = _get_residue(native_structure, native_chain, native_res_idx)
        sample_res = _get_residue(sample_structure, sample_chain, sample_res_idx)
        native_atoms = {
            atom.name.strip(): np.asarray(atom.coord, dtype=np.float64)
            for atom in native_res
            if _is_heavy_atom(atom.name)
        }
        for atom_name, native_coord in native_atoms.items():
            if atom_name not in sample_res:
                continue
            sample_coords.append(np.asarray(sample_res[atom_name].coord, dtype=np.float64))
            native_coords.append(native_coord)
    if not native_coords:
        raise ValueError("No matching heavy atoms found for residue pairs")
    return np.stack(native_coords), np.stack(sample_coords)


def _kabsch_align(
    mobile: np.ndarray,
    target: np.ndarray,
) -> tuple[float, Rotation | None, np.ndarray, np.ndarray]:
    if mobile.shape[0] == 1:
        return (
            float(np.linalg.norm(mobile[0] - target[0])),
            None,
            mobile[0],
            target[0],
        )
    mobile_com = mobile.mean(axis=0)
    target_com = target.mean(axis=0)
    mobile_centered = mobile - mobile_com
    target_centered = target - target_com
    rot, _ = Rotation.align_vectors(target_centered, mobile_centered)
    aligned = rot.apply(mobile_centered) + target_com
    mse = np.mean(np.sum((aligned - target) ** 2, axis=-1))
    return float(np.sqrt(mse)), rot, mobile_com, target_com


def _kabsch_rmsd(mobile: np.ndarray, target: np.ndarray) -> float:
    rmsd, _, _, _ = _kabsch_align(mobile, target)
    return rmsd


def _apply_kabsch(
    coord: np.ndarray,
    rot: Rotation | None,
    mobile_com: np.ndarray,
    target_com: np.ndarray,
) -> np.ndarray:
    if rot is None:
        return coord - mobile_com + target_com
    return rot.apply(coord - mobile_com) + target_com


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


def _append_group_residue_atoms(
    lines: list[str],
    *,
    atom_serial: int,
    structure,
    residue_pairs: list[tuple[tuple[str, int], tuple[str, int]]],
    use_native: bool,
    chain_id: str,
    coord_transform: Any | None = None,
) -> int:
    for resseq, (_native_pair, _sample_pair) in enumerate(residue_pairs, start=1):
        chain, res_idx = _native_pair if use_native else _sample_pair
        residue = _get_residue(structure, chain, res_idx)
        for atom in residue:
            if not _is_heavy_atom(atom.name):
                continue
            coord = np.asarray(atom.coord, dtype=np.float64)
            if coord_transform is not None:
                coord = coord_transform(coord)
            element = (atom.element or atom.name[0]).strip() or "  "
            lines.append(
                _format_atom_line(
                    serial=atom_serial,
                    atom_name=atom.name,
                    resname=residue.resname,
                    chain_id=chain_id,
                    resseq=resseq,
                    x=float(coord[0]),
                    y=float(coord[1]),
                    z=float(coord[2]),
                    element=element,
                )
            )
            atom_serial += 1
    return atom_serial


def _native_overlay_chain_id(group_id: int) -> str:
    chain_id = chr(ord("A") + group_id)
    if chain_id > "Z":
        raise ValueError(f"group_id {group_id} is too large for single-character PDB chains")
    return chain_id


def _append_full_structure_atoms(
    lines: list[str],
    *,
    atom_serial: int,
    structure,
    chain_id: str,
    coord_transform: Any | None = None,
) -> int:
    model = structure[0]
    resseq = 0
    for chain in model:
        for residue in chain:
            if residue.id[0] != " ":
                continue
            resseq += 1
            for atom in residue:
                coord = np.asarray(atom.coord, dtype=np.float64)
                if coord_transform is not None:
                    coord = coord_transform(coord)
                element = (atom.element or atom.name[0]).strip() or "  "
                lines.append(
                    _format_atom_line(
                        serial=atom_serial,
                        atom_name=atom.name,
                        resname=residue.resname,
                        chain_id=chain_id,
                        resseq=resseq,
                        x=float(coord[0]),
                        y=float(coord[1]),
                        z=float(coord[2]),
                        element=element,
                    )
                )
                atom_serial += 1
    return atom_serial


def _structure_ca_coords_by_residx(structure, chain_id: str) -> dict[int, np.ndarray]:
    coords: dict[int, np.ndarray] = {}
    chain = structure[0][chain_id]
    for residue in chain:
        if residue.id[0] != " ":
            continue
        if "CA" not in residue:
            continue
        coords[int(residue.id[1])] = np.asarray(residue["CA"].coord, dtype=np.float64)
    return coords


def _global_align_transform(
    target_structure,
    mobile_structure,
    chain_id: str,
) -> tuple[Rotation | None, np.ndarray, np.ndarray]:
    """Return Kabsch transform that maps mobile_structure onto target_structure."""
    target_ca = _structure_ca_coords_by_residx(target_structure, chain_id)
    mobile_ca = _structure_ca_coords_by_residx(mobile_structure, chain_id)
    common = sorted(set(target_ca) & set(mobile_ca))
    if not common:
        raise ValueError(f"No common CA residues on chain {chain_id} for global alignment")
    target_coords = np.stack([target_ca[res_idx] for res_idx in common])
    mobile_coords = np.stack([mobile_ca[res_idx] for res_idx in common])
    _, rot, mobile_com, target_com = _kabsch_align(mobile_coords, target_coords)
    return rot, mobile_com, target_com


def _infer_output_chain_id(
    group_pairs: dict[int, list[tuple[tuple[str, int], tuple[str, int]]]],
) -> str:
    for pairs in group_pairs.values():
        if pairs:
            return pairs[0][1][0]
    return "A"


def write_multimotif_overlay_pdb(
    output_pdb: Path,
    *,
    sample_name: str,
    variant_label: str,
    scaffold_structure,
    native_structure,
    folded_structure,
    group_pairs: dict[int, list[tuple[tuple[str, int], tuple[str, int]]]],
) -> None:
    """Write ESMFold on A, design aligned on E, and native motifs on ESMFold motif sites."""
    output_pdb = Path(output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    output_chain_id = _infer_output_chain_id(group_pairs)

    lines = [
        f"REMARK sample={sample_name}",
        f"REMARK esmfold_variant={variant_label}",
        "REMARK chain A = ESMFold folded structure for this variant",
        f"REMARK chain E = design scaffold globally aligned onto chain A via CA on chain {output_chain_id}",
        "REMARK chains B+ = native motifs HA-aligned onto ESMFold motif sites (chain A frame)",
    ]
    atom_serial = 1

    atom_serial = _append_full_structure_atoms(
        lines,
        atom_serial=atom_serial,
        structure=folded_structure,
        chain_id="A",
    )
    lines.append("TER")

    scaffold_rot, scaffold_mobile_com, scaffold_target_com = _global_align_transform(
        folded_structure,
        scaffold_structure,
        output_chain_id,
    )

    def _scaffold_transform(
        coord: np.ndarray,
        _rot=scaffold_rot,
        _mobile_com=scaffold_mobile_com,
        _target_com=scaffold_target_com,
    ) -> np.ndarray:
        return _apply_kabsch(coord, _rot, _mobile_com, _target_com)

    atom_serial = _append_full_structure_atoms(
        lines,
        atom_serial=atom_serial,
        structure=scaffold_structure,
        chain_id="E",
        coord_transform=_scaffold_transform,
    )
    lines.append("TER")

    for group_id in sorted(group_pairs):
        pairs = group_pairs[group_id]
        overlay_chain = _native_overlay_chain_id(group_id)
        native_ha, folded_ha = _heavy_atom_pairs(native_structure, folded_structure, pairs)
        motif_ha_rmsd = _kabsch_rmsd(native_ha, folded_ha)
        _, rot, mobile_com, target_com = _kabsch_align(native_ha, folded_ha)
        lines.append(
            f"REMARK group_id={group_id} native_overlay_chain={overlay_chain} "
            f"motif_ha_rmsd={motif_ha_rmsd:.4f}"
        )

        def _transform(
            coord: np.ndarray,
            _rot=rot,
            _mobile_com=mobile_com,
            _target_com=target_com,
        ) -> np.ndarray:
            return _apply_kabsch(coord, _rot, _mobile_com, _target_com)

        atom_serial = _append_group_residue_atoms(
            lines,
            atom_serial=atom_serial,
            structure=native_structure,
            residue_pairs=pairs,
            use_native=True,
            chain_id=overlay_chain,
            coord_transform=_transform,
        )
        lines.append("TER")

    lines.append("END")
    output_pdb.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_multimotif_overlay_pdbs(
    samples_metadata: dict[str, Any],
    *,
    esmfold_dir: Path,
    alignment_pdb_dir: Path,
    parser: PDBParser | None = None,
) -> int:
    """Write overlay PDBs for every ESMFold structure under esmfold/<sample_name>/."""
    parser = parser or PDBParser(QUIET=True)
    esmfold_dir = Path(esmfold_dir)
    alignment_pdb_dir = Path(alignment_pdb_dir)
    written = 0

    for sample_name, metadata in samples_metadata.items():
        source_residue_keys = metadata.get("source_residue_keys")
        if not source_residue_keys:
            continue

        group_ids = sorted(
            {
                int(entry["group_id"])
                for entry in source_residue_keys
                if entry.get("group_id") is not None
            }
        )
        if not group_ids:
            continue

        scaffold_path = Path(metadata["path"])
        native_path = Path(metadata["native_pdb"])
        if not scaffold_path.is_file() or not native_path.is_file():
            logger.warning(
                "Skipping overlay PDBs for %s: missing scaffold=%s or native=%s",
                sample_name,
                scaffold_path,
                native_path,
            )
            continue

        folded_paths = list_esmfold_structure_paths(esmfold_dir, sample_name)
        if not folded_paths:
            logger.warning(
                "Skipping overlay PDBs for %s: no structures under %s",
                sample_name,
                esmfold_dir / sample_name,
            )
            continue

        try:
            scaffold_structure = parser.get_structure(f"{sample_name}_design", str(scaffold_path))
            native_structure = parser.get_structure("native", str(native_path))
        except Exception:
            logger.exception("Failed to load structures for overlay PDBs: %s", sample_name)
            continue

        group_pairs = {
            group_id: _group_residue_pairs(source_residue_keys, group_id)
            for group_id in group_ids
        }

        sample_out_dir = alignment_pdb_dir / sample_name
        for variant_label, folded_path in folded_paths:
            try:
                folded_structure = parser.get_structure(
                    f"{sample_name}_{variant_label}",
                    str(folded_path),
                )
                write_multimotif_overlay_pdb(
                    sample_out_dir / f"{variant_label}.pdb",
                    sample_name=sample_name,
                    variant_label=variant_label,
                    scaffold_structure=scaffold_structure,
                    native_structure=native_structure,
                    folded_structure=folded_structure,
                    group_pairs=group_pairs,
                )
                written += 1
            except Exception:
                logger.exception(
                    "Failed overlay PDB for %s variant=%s path=%s",
                    sample_name,
                    variant_label,
                    folded_path,
                )

    return written


def _group_residue_pairs(
    source_residue_keys: list[dict[str, Any]],
    group_id: int,
) -> list[tuple[tuple[str, int], tuple[str, int]]]:
    pairs: list[tuple[tuple[str, int], tuple[str, int]]] = []
    for entry in source_residue_keys:
        if entry.get("group_id") != group_id:
            continue
        native_pair = (entry["native_chain"], int(entry["native_res_idx"]))
        sample_pair = (entry["output_chain"], int(entry["output_res_idx"]))
        pairs.append((native_pair, sample_pair))
    return pairs


def _group_ca_coords(
    structure,
    residue_pairs: list[tuple[tuple[str, int], tuple[str, int]]],
    *,
    use_native: bool,
) -> np.ndarray:
    coords = []
    for native_pair, sample_pair in residue_pairs:
        chain_id, res_idx = native_pair if use_native else sample_pair
        coords.append(_ca_coord(structure, chain_id, res_idx))
    return np.stack(coords)


def _group_com(
    structure,
    residue_pairs: list[tuple[tuple[str, int], tuple[str, int]]],
    *,
    use_native: bool,
) -> np.ndarray:
    return _group_ca_coords(structure, residue_pairs, use_native=use_native).mean(axis=0)


def _pairwise_distances(com_by_group: dict[int, np.ndarray]) -> dict[tuple[int, int], float]:
    distances: dict[tuple[int, int], float] = {}
    for group_i, group_j in itertools.combinations(sorted(com_by_group), 2):
        dist = float(np.linalg.norm(com_by_group[group_i] - com_by_group[group_j]))
        distances[(group_i, group_j)] = dist
    return distances


def _angle_deg(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return float("nan")
    cos_angle = np.clip(np.dot(vec_a, vec_b) / (norm_a * norm_b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _intergroup_angles(com_by_group: dict[int, np.ndarray]) -> list[dict[str, Any]]:
    angles: list[dict[str, Any]] = []
    group_ids = sorted(com_by_group)
    if len(group_ids) < 3:
        return angles
    for apex in group_ids:
        others = [gid for gid in group_ids if gid != apex]
        for group_a, group_b in itertools.combinations(others, 2):
            vec_a = com_by_group[group_a] - com_by_group[apex]
            vec_b = com_by_group[group_b] - com_by_group[apex]
            angles.append(
                {
                    "apex_group_id": apex,
                    "group_a": group_a,
                    "group_b": group_b,
                    "angle_deg": _angle_deg(vec_a, vec_b),
                }
            )
    return angles


def _slice_metadata(native_distances: dict[tuple[int, int], float], num_groups: int) -> dict[str, Any]:
    dist_values = list(native_distances.values())
    if not dist_values:
        return {
            "num_motif_groups": num_groups,
            "native_min_intergroup_dist": float("nan"),
            "native_max_intergroup_dist": float("nan"),
            "native_mean_intergroup_dist": float("nan"),
        }
    return {
        "num_motif_groups": num_groups,
        "native_min_intergroup_dist": float(min(dist_values)),
        "native_max_intergroup_dist": float(max(dist_values)),
        "native_mean_intergroup_dist": float(np.mean(dist_values)),
    }


def compute_sample_multimotif_metrics(
    sample_name: str,
    metadata: dict[str, Any],
    *,
    parser: PDBParser | None = None,
    design_structure_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Compute per-group and inter-group metrics for one sample."""
    metadata = apply_source_residue_key_repair(dict(metadata))
    source_residue_keys = metadata.get("source_residue_keys")
    if not source_residue_keys:
        return []

    group_ids = sorted(
        {
            int(entry["group_id"])
            for entry in source_residue_keys
            if entry.get("group_id") is not None
        }
    )
    if not group_ids:
        return []

    design_path = Path(design_structure_path or metadata["path"])
    native_path = Path(metadata["native_pdb"])
    if not design_path.is_file() or not native_path.is_file():
        raise FileNotFoundError(
            f"Missing structure for {sample_name}: design={design_path}, native={native_path}"
        )

    parser = parser or PDBParser(QUIET=True)
    design_structure = parser.get_structure(sample_name, str(design_path))
    native_structure = parser.get_structure("native", str(native_path))

    task = metadata.get("name", "")
    rows: list[dict[str, Any]] = []

    group_pairs = {
        group_id: _group_residue_pairs(source_residue_keys, group_id) for group_id in group_ids
    }
    native_com_by_group = {
        group_id: _group_com(native_structure, pairs, use_native=True)
        for group_id, pairs in group_pairs.items()
    }
    design_com_by_group = {
        group_id: _group_com(design_structure, pairs, use_native=False)
        for group_id, pairs in group_pairs.items()
    }
    native_distances = _pairwise_distances(native_com_by_group)
    design_distances = _pairwise_distances(design_com_by_group)
    slice_meta = _slice_metadata(native_distances, len(group_ids))
    group_ha_rmsd: dict[int, float] = {}

    for group_id in group_ids:
        pairs = group_pairs[group_id]
        native_ca = _group_ca_coords(native_structure, pairs, use_native=True)
        design_ca = _group_ca_coords(design_structure, pairs, use_native=False)
        motif_ca_rmsd = _kabsch_rmsd(design_ca, native_ca)
        native_ha, design_ha = _heavy_atom_pairs(native_structure, design_structure, pairs)
        motif_ha_rmsd = _kabsch_rmsd(design_ha, native_ha)
        group_ha_rmsd[group_id] = motif_ha_rmsd
        rows.append(
            {
                "record_type": "group",
                "name": sample_name,
                "task": task,
                "group_id": group_id,
                "group_size": len(pairs),
                "motif_ca_rmsd": motif_ca_rmsd,
                "motif_ha_rmsd": motif_ha_rmsd,
                **slice_meta,
            }
        )

    for (group_i, group_j), native_dist in native_distances.items():
        design_dist = design_distances[(group_i, group_j)]
        rows.append(
            {
                "record_type": "intergroup",
                "name": sample_name,
                "task": task,
                "group_i": group_i,
                "group_j": group_j,
                "native_com_distance": native_dist,
                "sample_com_distance": design_dist,
                "com_distance_error": abs(design_dist - native_dist),
                **slice_meta,
            }
        )

    native_angles = {
        (entry["apex_group_id"], entry["group_a"], entry["group_b"]): entry["angle_deg"]
        for entry in _intergroup_angles(native_com_by_group)
    }
    design_angles = {
        (entry["apex_group_id"], entry["group_a"], entry["group_b"]): entry["angle_deg"]
        for entry in _intergroup_angles(design_com_by_group)
    }
    for key, native_angle in native_angles.items():
        design_angle = design_angles[key]
        angle_error = (
            abs(design_angle - native_angle)
            if np.isfinite(native_angle) and np.isfinite(design_angle)
            else float("nan")
        )
        apex_group_id, group_a, group_b = key
        rows.append(
            {
                "record_type": "angle",
                "name": sample_name,
                "task": task,
                "apex_group_id": apex_group_id,
                "group_a": group_a,
                "group_b": group_b,
                "native_angle_deg": native_angle,
                "sample_angle_deg": design_angle,
                "angle_error_deg": angle_error,
                **slice_meta,
            }
        )

    return rows


def compute_run_multimotif_metrics(
    samples_metadata: dict[str, Any],
    *,
    design_structure_paths: dict[str, Path] | None = None,
    design_structure_source: DesignStructureSource = "sample",
    esmfold_dir: Path | None = None,
    alignment_pdb_dir: Path | None = None,
) -> pd.DataFrame:
    """Compute multimotif metrics for all samples in a run."""
    parser = PDBParser(QUIET=True)
    rows: list[dict[str, Any]] = []
    skipped = 0
    for sample_name, metadata in samples_metadata.items():
        design_path = None
        if design_structure_paths is not None:
            design_path = design_structure_paths.get(sample_name)
            if design_path is None:
                logger.warning(
                    "Skipping %s: no design structure path for source=%s",
                    sample_name,
                    design_structure_source,
                )
                skipped += 1
                continue
        try:
            sample_rows = compute_sample_multimotif_metrics(
                sample_name,
                metadata,
                parser=parser,
                design_structure_path=design_path,
            )
        except Exception:
            logger.exception("Failed multimotif metrics for %s", sample_name)
            continue
        if not sample_rows:
            skipped += 1
            continue
        rows.extend(sample_rows)
    if skipped:
        logger.info("Skipped %d samples without multimotif group metadata", skipped)

    if alignment_pdb_dir is not None:
        if esmfold_dir is None:
            raise ValueError("alignment_pdb_dir requires esmfold_dir")
        num_written = write_run_multimotif_overlay_pdbs(
            samples_metadata,
            esmfold_dir=esmfold_dir,
            alignment_pdb_dir=alignment_pdb_dir,
            parser=parser,
        )
        logger.info("Wrote %d multimotif overlay PDBs under %s", num_written, alignment_pdb_dir)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def aggregate_multimotif_slices(
    df: pd.DataFrame,
    *,
    group_rmsd_threshold: float | None = None,
    ca_rmsd_threshold: float | None = None,
) -> pd.DataFrame:
    """Aggregate per-group recovery by motif count and native inter-group distance bins."""
    from nomadflow.benchmark.success_criteria import MOTIF_CA_RMSD_MAX, MOTIF_HA_RMSD_MAX

    if group_rmsd_threshold is None:
        group_rmsd_threshold = MOTIF_HA_RMSD_MAX
    if ca_rmsd_threshold is None:
        ca_rmsd_threshold = MOTIF_CA_RMSD_MAX

    if df.empty:
        return pd.DataFrame()

    group_df = df[df["record_type"] == "group"].copy()
    if group_df.empty:
        return pd.DataFrame()
    required = {"motif_ha_rmsd", "motif_ca_rmsd", "native_min_intergroup_dist"}
    if not required.issubset(group_df.columns):
        return pd.DataFrame()

    group_df["group_pass"] = (
        (group_df["motif_ha_rmsd"] < group_rmsd_threshold)
        & (group_df["motif_ca_rmsd"] < ca_rmsd_threshold)
    )

    dist_bins = [-np.inf, 10.0, 20.0, 30.0, np.inf]
    dist_labels = ["<10A", "10-20A", "20-30A", ">=30A"]
    group_df["native_dist_bin"] = pd.cut(
        group_df["native_min_intergroup_dist"],
        bins=dist_bins,
        labels=dist_labels,
        right=False,
    )

    summaries = []
    for slice_col in ("num_motif_groups", "group_size", "native_dist_bin"):
        grouped = group_df.groupby(slice_col, dropna=False, observed=False)
        summary = grouped.agg(
            num_group_records=("group_pass", "size"),
            group_pass_rate=("group_pass", "mean"),
            mean_motif_ha_rmsd=("motif_ha_rmsd", "mean"),
            mean_motif_ca_rmsd=("motif_ca_rmsd", "mean"),
        ).reset_index()
        summary.insert(0, "slice_type", slice_col)
        summary = summary.rename(columns={slice_col: "slice_value"})
        summaries.append(summary)

    return pd.concat(summaries, ignore_index=True)
