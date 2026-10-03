"""Relative-layout metrics from Kabsch frames on all group backbone N/CA/C atoms."""

from __future__ import annotations

import itertools
import logging
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from scipy.spatial.transform import Rotation

from nomadflow.benchmark.analysis_inputs import (
    list_esmfold_structure_paths,
    parent_motif_task_id,
    prefer_samples_eval_pdb,
    remap_repo_path,
)
from nomadflow.benchmark.motif_group_annotation import apply_source_residue_key_repair
from nomadflow.benchmark.success_criteria import expected_group_count

logger = logging.getLogger(__name__)

_BACKBONE_ATOMS = ("N", "CA", "C")
LAYOUT_FRAME = "kabsch_backbone_nca_c"
Frame = tuple[np.ndarray, np.ndarray]


class LayoutSkip(Exception):
    """Skip a sample during layout scoring with an explicit reason."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


def _resolve_chain(structure, chain_id: str) -> str:
    """Use the annotated chain when present; otherwise the sole chain (ESMFold is often A)."""
    model = structure[0]
    if chain_id in model:
        return chain_id
    chains = [chain.id for chain in model]
    if len(chains) == 1:
        return chains[0]
    if "A" in model:
        return "A"
    raise KeyError(f"Chain {chain_id!r} not in structure (have {chains})")


def _standard_residue(chain, res_idx: int):
    """Return a polymer residue by seq number, or None if absent."""
    try:
        residue = chain[int(res_idx)]
    except (KeyError, TypeError, ValueError):
        return None
    hetflag = residue.id[0] if residue.id else " "
    if hetflag.strip():
        return None
    return residue


def _chain_n_residues(chain) -> int:
    return sum(1 for residue in chain if not str(residue.id[0]).strip())


def _get_residue(structure, chain_id: str, res_idx: int, *, remap_missing: bool = False):
    """Look up a residue; for design frames, remap chain when the annotated id is wrong.

    Annotated ``output_chain:res`` can name the eval scaffold (A) while the viz PDB
    still has the motif on another chain and a native copy on A at higher res ids.
    When the annotated residue is missing, use the unique chain that has that
    number; if several do, prefer the longest chain (scaffold, not motif copy).
    """
    model = structure[0]
    resolved = _resolve_chain(structure, chain_id)
    residue = _standard_residue(model[resolved], res_idx)
    if residue is not None:
        return residue
    if not remap_missing:
        raise KeyError(f"Residue {res_idx} not in chain {resolved!r}")

    candidates: list[tuple[Any, Any]] = []
    for chain in model:
        found = _standard_residue(chain, res_idx)
        if found is not None:
            candidates.append((chain, found))
    if len(candidates) == 1:
        return candidates[0][1]
    if len(candidates) > 1:
        def _scaffold_key(item: tuple[Any, Any]) -> tuple[int, int, str]:
            chain, _res = item
            res_ids = [residue.id[1] for residue in chain if not str(residue.id[0]).strip()]
            min_id = min(res_ids) if res_ids else 0
            # Longest chain first (generated scaffold); then lower numbering
            # (not a trailing motif-copy with higher res ids).
            return (-_chain_n_residues(chain), min_id, str(chain.id))

        candidates.sort(key=_scaffold_key)
        return candidates[0][1]
    have = [chain.id for chain in model]
    raise KeyError(
        f"Residue {res_idx} not in chain {resolved!r} or any other chain (have {have})"
    )


def _atom_coord(residue, atom_name: str) -> np.ndarray:
    if atom_name not in residue:
        raise KeyError(f"Residue {residue.id} missing atom {atom_name}")
    return np.asarray(residue[atom_name].coord, dtype=np.float64)


def group_residue_pairs(
    source_residue_keys: Sequence[Mapping[str, Any]],
    group_id: int,
) -> list[tuple[tuple[str, int], tuple[str, int]]]:
    """Preserve annotation order; do not sort by residue number."""
    pairs: list[tuple[tuple[str, int], tuple[str, int]]] = []
    for entry in source_residue_keys:
        if not isinstance(entry, Mapping):
            continue
        if entry.get("group_id") != group_id:
            continue
        native_pair = (str(entry["native_chain"]), int(entry["native_res_idx"]))
        sample_pair = (str(entry["output_chain"]), int(entry["output_res_idx"]))
        pairs.append((native_pair, sample_pair))
    return pairs


def group_ids_in_annotation_order(source_residue_keys: Sequence[Mapping[str, Any]]) -> list[int]:
    ordered: list[int] = []
    seen: set[int] = set()
    for entry in source_residue_keys:
        if not isinstance(entry, Mapping) or entry.get("group_id") is None:
            continue
        group_id = int(entry["group_id"])
        if group_id in seen:
            continue
        seen.add(group_id)
        ordered.append(group_id)
    return ordered


def backbone_coords(
    structure,
    residue_pairs: Sequence[tuple[tuple[str, int], tuple[str, int]]],
    *,
    use_native: bool,
) -> np.ndarray:
    coords: list[np.ndarray] = []
    for native_pair, sample_pair in residue_pairs:
        chain_id, res_idx = native_pair if use_native else sample_pair
        residue = _get_residue(
            structure, chain_id, res_idx, remap_missing=not use_native
        )
        for atom_name in _BACKBONE_ATOMS:
            coords.append(_atom_coord(residue, atom_name))
    if not coords:
        raise ValueError("No backbone N/CA/C atoms for group")
    return np.stack(coords)


def group_backbone_com(
    structure,
    residue_pairs: Sequence[tuple[tuple[str, int], tuple[str, int]]],
    *,
    use_native: bool,
) -> np.ndarray:
    """Equal-weighted CoM of backbone N/CA/C over the whole group."""
    return backbone_coords(structure, residue_pairs, use_native=use_native).mean(axis=0)


def kabsch_backbone_frame(
    coords: np.ndarray,
    reference_coords: np.ndarray | None = None,
) -> Frame:
    """Rigid frame of a group vs a native/canonical template.

    Origin is the observed backbone CoM. Rotation is Kabsch of ``reference_coords``
    (native group N/CA/C, same atom correspondence) onto the observed coords.
    If ``reference_coords`` is omitted, this structure *is* the template (R = I).
    Relative pose is still ``T_ij = T_i⁻¹ ∘ T_j`` and is invariant to a global SE(3)
    applied to every group.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) backbone coords, got {coords.shape}")
    origin = coords.mean(axis=0)
    if reference_coords is None:
        return np.eye(3, dtype=np.float64), np.asarray(origin, dtype=np.float64)
    ref = np.asarray(reference_coords, dtype=np.float64)
    if ref.shape != coords.shape:
        raise ValueError(f"Reference shape {ref.shape} != observed {coords.shape}")
    rot, _rssd = Rotation.align_vectors(coords - origin, ref - ref.mean(axis=0))
    return np.asarray(rot.as_matrix(), dtype=np.float64), np.asarray(origin, dtype=np.float64)


def group_frame(
    structure,
    residue_pairs: Sequence[tuple[tuple[str, int], tuple[str, int]]],
    *,
    use_native: bool,
    reference_coords: np.ndarray | None = None,
) -> Frame:
    """Kabsch rigid transform of all backbone N/CA/C vs the native template."""
    if not residue_pairs:
        raise ValueError("Cannot build a frame for an empty group")
    return kabsch_backbone_frame(
        backbone_coords(structure, residue_pairs, use_native=use_native),
        reference_coords=reference_coords,
    )


def _rotation_to_quat_wxyz(rot: np.ndarray) -> tuple[float, float, float, float]:
    """Unit quaternion, scalar-first (w, x, y, z), Hamilton, representative with w >= 0."""
    quat = Rotation.from_matrix(np.asarray(rot, dtype=np.float64)).as_quat()
    if quat[3] < 0:
        quat = -quat
    return float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])


def pose_column_dict(prefix: str, frame: Frame | None) -> dict[str, float]:
    """Serialize a relative pose ``T_ij`` as translation + quaternion columns."""
    if frame is None:
        return {
            f"{prefix}_tij_tx": float("nan"),
            f"{prefix}_tij_ty": float("nan"),
            f"{prefix}_tij_tz": float("nan"),
            f"{prefix}_tij_qw": float("nan"),
            f"{prefix}_tij_qx": float("nan"),
            f"{prefix}_tij_qy": float("nan"),
            f"{prefix}_tij_qz": float("nan"),
        }
    rot, trans = frame
    trans = np.asarray(trans, dtype=np.float64).reshape(3)
    qw, qx, qy, qz = _rotation_to_quat_wxyz(rot)
    return {
        f"{prefix}_tij_tx": float(trans[0]),
        f"{prefix}_tij_ty": float(trans[1]),
        f"{prefix}_tij_tz": float(trans[2]),
        f"{prefix}_tij_qw": qw,
        f"{prefix}_tij_qx": qx,
        f"{prefix}_tij_qy": qy,
        f"{prefix}_tij_qz": qz,
    }


def invert_compose(frame_i: Frame, frame_j: Frame) -> Frame:
    """Relative pose T_ij = T_i^{-1} ∘ T_j (OpenFold invert().compose())."""
    rot_i, origin_i = frame_i
    rot_j, origin_j = frame_j
    rot_i_t = np.asarray(rot_i, dtype=np.float64).T
    t_ij = rot_i_t @ (np.asarray(origin_j, dtype=np.float64) - np.asarray(origin_i, dtype=np.float64))
    r_ij = rot_i_t @ np.asarray(rot_j, dtype=np.float64)
    return r_ij, t_ij


def so3_geodesic_deg(rot_delta: np.ndarray) -> float:
    rotation = Rotation.from_matrix(np.asarray(rot_delta, dtype=np.float64))
    return float(np.degrees(rotation.magnitude()))


def relative_pose_delta(rel_a: Frame, rel_b: Frame) -> tuple[float, float]:
    """Compare two relative poses: translation Å and SO(3) geodesic degrees, separately."""
    r_a, t_a = rel_a
    r_b, t_b = rel_b
    trans_A = float(np.linalg.norm(np.asarray(t_b, dtype=np.float64) - np.asarray(t_a, dtype=np.float64)))
    rot_delta = np.asarray(r_a, dtype=np.float64).T @ np.asarray(r_b, dtype=np.float64)
    return trans_A, so3_geodesic_deg(rot_delta)


def apply_rigid(coords: np.ndarray, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return np.asarray(coords, dtype=np.float64) @ np.asarray(rot, dtype=np.float64).T + np.asarray(
        trans, dtype=np.float64
    )


def _load_structure(parser: PDBParser, path: Path, structure_id: str):
    return parser.get_structure(structure_id, str(path))


def _frames_for_structure(
    structure,
    group_pairs: Mapping[int, list[tuple[tuple[str, int], tuple[str, int]]]],
    *,
    use_native: bool,
    reference_coords_by_group: Mapping[int, np.ndarray] | None = None,
) -> dict[int, Frame]:
    frames: dict[int, Frame] = {}
    for group_id, pairs in group_pairs.items():
        ref = None if reference_coords_by_group is None else reference_coords_by_group.get(group_id)
        frames[group_id] = group_frame(
            structure, pairs, use_native=use_native, reference_coords=ref
        )
    return frames


def _pair_relative_poses(frames: Mapping[int, Frame]) -> dict[tuple[int, int], Frame]:
    relatives: dict[tuple[int, int], Frame] = {}
    for group_i, group_j in itertools.combinations(sorted(frames), 2):
        relatives[(group_i, group_j)] = invert_compose(frames[group_i], frames[group_j])
    return relatives


def _one_shot_folded_path(
    esmfold_dir: Path | None,
    sample_name: str,
    folded_path: str | Path | None = None,
    *,
    repo_root: Path | None = None,
) -> Path | None:
    if folded_path:
        remapped = remap_repo_path(folded_path, repo_root=repo_root)
        if remapped is not None and remapped.is_file():
            return remapped
    if esmfold_dir is None:
        return None
    for label, path in list_esmfold_structure_paths(esmfold_dir, sample_name):
        if label == "one_shot":
            return path
    return None


def _pmpnn_folded_path(
    esmfold_dir: Path | None,
    sample_name: str,
    sample_idx: int | None,
    path_from_metrics: str | Path | None = None,
    *,
    repo_root: Path | None = None,
) -> Path | None:
    if path_from_metrics:
        remapped = remap_repo_path(path_from_metrics, repo_root=repo_root)
        if remapped is not None and remapped.is_file():
            return remapped
    if esmfold_dir is None or sample_idx is None:
        return None
    label = f"pmpnn_sample_{int(sample_idx)}"
    for fold_label, path in list_esmfold_structure_paths(esmfold_dir, sample_name):
        if fold_label == label:
            return path
    return None


def _pmpnn_layout_rows_for_sample(
    sample_name: str,
    seq_records: list[dict[str, Any]],
    metadata: dict[str, Any],
    esmfold_dir: str | None,
    repo_root: str | None,
) -> list[dict[str, Any]]:
    parser = PDBParser(QUIET=True)
    esmfold_path = Path(esmfold_dir) if esmfold_dir else None
    repo_path = Path(repo_root) if repo_root else None
    try:
        base_rows, design_rel = compute_sample_layout_pair_metrics(
            sample_name,
            metadata,
            parser=parser,
            repo_root=repo_path,
        )
    except Exception:
        logger.exception("Failed PMPNN layout design frames for %s", sample_name)
        return []
    if not base_rows or not design_rel:
        return []
    source_residue_keys = metadata.get("source_residue_keys")
    group_ids = group_ids_in_annotation_order(source_residue_keys)
    group_pairs = {
        group_id: group_residue_pairs(source_residue_keys, group_id) for group_id in group_ids
    }
    excursion_by_pair = {
        (int(row["group_i"]), int(row["group_j"])): (
            row["excursion_trans_A"],
            row["excursion_rot_deg"],
        )
        for row in base_rows
    }
    try:
        native_path = Path(
            remap_repo_path(metadata.get("native_pdb"), repo_root=repo_path)
            or metadata["native_pdb"]
        )
        native_structure = _load_structure(parser, native_path, f"{sample_name}_native")
        native_coords_by_group = {
            gid: backbone_coords(native_structure, pairs, use_native=True)
            for gid, pairs in group_pairs.items()
        }
    except Exception:
        logger.exception("Failed PMPNN native template for %s", sample_name)
        return []
    rows: list[dict[str, Any]] = []
    for seq_row in seq_records:
        sample_idx = seq_row.get("sample")
        if sample_idx is not None and pd.isna(sample_idx):
            sample_idx = None
        path_from_metrics = seq_row.get("path")
        if path_from_metrics is not None and pd.isna(path_from_metrics):
            path_from_metrics = None
        folded = _pmpnn_folded_path(
            esmfold_path,
            sample_name,
            int(sample_idx) if sample_idx is not None else None,
            path_from_metrics,
            repo_root=repo_path,
        )
        if folded is None:
            continue
        try:
            folded_structure = _load_structure(
                parser, folded, f"{sample_name}_pmpnn_{sample_idx}"
            )
            folded_rel = _pair_relative_poses(
                _frames_for_structure(
                    folded_structure,
                    group_pairs,
                    use_native=False,
                    reference_coords_by_group=native_coords_by_group,
                )
            )
        except Exception:
            logger.exception("Failed PMPNN layout fold for %s sample=%s", sample_name, sample_idx)
            continue
        task = str(base_rows[0].get("task") or _parent_task(metadata))
        native_by_pair = {
            (int(row["group_i"]), int(row["group_j"])): row for row in base_rows
        }
        for pair, design_pose in design_rel.items():
            if pair not in folded_rel:
                continue
            fold_pose = folded_rel[pair]
            retention_trans, retention_rot = relative_pose_delta(design_pose, fold_pose)
            excursion_trans, excursion_rot = excursion_by_pair.get(
                pair, (float("nan"), float("nan"))
            )
            group_i, group_j = pair
            base = native_by_pair.get(pair, {})
            row = {
                "name": sample_name,
                "task": task,
                "pmpnn_sample": int(sample_idx) if sample_idx is not None else pd.NA,
                "group_i": group_i,
                "group_j": group_j,
                "num_motif_groups": len(group_ids),
                "excursion_trans_A": excursion_trans,
                "excursion_rot_deg": excursion_rot,
                "retention_trans_A": retention_trans,
                "retention_rot_deg": retention_rot,
                "pass_final": True,
                "fold_source": "pmpnn_esmfold",
            }
            for prefix in ("native", "design"):
                for suffix in ("tx", "ty", "tz", "qw", "qx", "qy", "qz"):
                    key = f"{prefix}_tij_{suffix}"
                    row[key] = base.get(key, float("nan"))
            row.update(pose_column_dict("fold", fold_pose))
            rows.append(row)
    return rows


def compute_pmpnn_layout_pair_metrics(
    samples_metadata: dict[str, Any],
    pmpnn_df: pd.DataFrame,
    *,
    esmfold_dir: Path | None = None,
    repo_root: Path | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    """Retention of passing PMPNN folds vs the generated design (not one-shot ESMFold)."""
    if pmpnn_df is None or pmpnn_df.empty or "name" not in pmpnn_df.columns:
        return pd.DataFrame()
    passed = pmpnn_df
    if "pass_final" in pmpnn_df.columns:
        passed = pmpnn_df[pmpnn_df["pass_final"].fillna(False).astype(bool)].copy()
    if passed.empty:
        return pd.DataFrame()

    jobs: list[tuple[str, list[dict[str, Any]], dict[str, Any]]] = []
    for sample_name, seq_df in passed.groupby("name", sort=False):
        sample_name = str(sample_name)
        metadata = samples_metadata.get(sample_name)
        if not isinstance(metadata, dict) or expected_group_count(metadata) < 2:
            continue
        jobs.append((sample_name, seq_df.to_dict("records"), metadata))
    if not jobs:
        return pd.DataFrame()

    esmfold_s = str(esmfold_dir) if esmfold_dir is not None else None
    repo_s = str(repo_root) if repo_root is not None else None
    rows: list[dict[str, Any]] = []
    n_workers = max(1, int(workers))
    if n_workers == 1:
        for i, (sample_name, seq_records, metadata) in enumerate(jobs, start=1):
            rows.extend(
                _pmpnn_layout_rows_for_sample(
                    sample_name, seq_records, metadata, esmfold_s, repo_s
                )
            )
            if i % 200 == 0 or i == len(jobs):
                logger.info("PMPNN layout %s/%s designs", i, len(jobs))
    else:
        logger.info("PMPNN layout %s designs, %s workers", len(jobs), n_workers)
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [
                pool.submit(
                    _pmpnn_layout_rows_for_sample,
                    sample_name,
                    seq_records,
                    metadata,
                    esmfold_s,
                    repo_s,
                )
                for sample_name, seq_records, metadata in jobs
            ]
            for i, fut in enumerate(as_completed(futures), start=1):
                rows.extend(fut.result())
                if i % 200 == 0 or i == len(jobs):
                    logger.info("PMPNN layout %s/%s designs", i, len(jobs))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _parent_task(metadata: Mapping[str, Any], fallback: str = "") -> str:
    raw = metadata.get("name") or fallback
    return parent_motif_task_id(str(raw), dict(metadata) if metadata else None)


def compute_sample_layout_pair_metrics(
    sample_name: str,
    metadata: dict[str, Any],
    *,
    parser: PDBParser | None = None,
    design_path: Path | None = None,
    native_path: Path | None = None,
    folded_path: Path | None = None,
    repo_root: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], Frame]]:
    """Per-pair excursion (design vs native) and retention (design vs fold)."""
    metadata = apply_source_residue_key_repair(dict(metadata))
    source_residue_keys = metadata.get("source_residue_keys")
    if not isinstance(source_residue_keys, list) or not source_residue_keys:
        return [], {}
    group_ids = group_ids_in_annotation_order(source_residue_keys)
    if len(group_ids) < 2:
        return [], {}

    if design_path is None:
        remapped = remap_repo_path(metadata.get("path"), repo_root=repo_root)
        design_path = Path(remapped or metadata["path"])
        eval_path = prefer_samples_eval_pdb(design_path)
        if eval_path is not None:
            design_path = Path(eval_path)
    else:
        design_path = Path(design_path)
    native_path = Path(
        native_path
        or remap_repo_path(metadata.get("native_pdb"), repo_root=repo_root)
        or metadata["native_pdb"]
    )
    if not design_path.is_file():
        raise LayoutSkip("missing_structure", f"design={design_path}")
    if not native_path.is_file():
        raise LayoutSkip("eval_error", f"native pdb missing: {native_path}")

    parser = parser or PDBParser(QUIET=True)
    group_pairs = {
        group_id: group_residue_pairs(source_residue_keys, group_id) for group_id in group_ids
    }

    try:
        native_structure = _load_structure(parser, native_path, f"{sample_name}_native")
        native_coords_by_group = {
            group_id: backbone_coords(native_structure, pairs, use_native=True)
            for group_id, pairs in group_pairs.items()
        }
        native_rel = _pair_relative_poses(
            _frames_for_structure(native_structure, group_pairs, use_native=True)
        )
    except KeyError as exc:
        raise LayoutSkip("eval_error", f"native residue/atom/chain: {exc}") from exc

    try:
        design_structure = _load_structure(parser, design_path, f"{sample_name}_design")
        design_rel = _pair_relative_poses(
            _frames_for_structure(
                design_structure,
                group_pairs,
                use_native=False,
                reference_coords_by_group=native_coords_by_group,
            )
        )
    except KeyError as exc:
        raise LayoutSkip("missing_residue_in_design", str(exc)) from exc

    folded_rel: dict[tuple[int, int], Frame] | None = None
    if folded_path is not None and Path(folded_path).is_file():
        try:
            folded_structure = _load_structure(
                parser, Path(folded_path), f"{sample_name}_fold"
            )
            folded_rel = _pair_relative_poses(
                _frames_for_structure(
                    folded_structure,
                    group_pairs,
                    use_native=False,
                    reference_coords_by_group=native_coords_by_group,
                )
            )
        except KeyError:
            logger.warning(
                "Folded structure missing residue/atom for %s; keeping excursion, NaN retention",
                sample_name,
            )
            folded_rel = None

    task = _parent_task(metadata)
    source_layout_id = metadata.get("source_layout_id")
    rows: list[dict[str, Any]] = []
    for pair, native_pose in native_rel.items():
        group_i, group_j = pair
        design_pose = design_rel[pair]
        excursion_trans, excursion_rot = relative_pose_delta(native_pose, design_pose)
        fold_pose = folded_rel.get(pair) if folded_rel is not None else None
        retention_trans = float("nan")
        retention_rot = float("nan")
        if fold_pose is not None:
            retention_trans, retention_rot = relative_pose_delta(design_pose, fold_pose)
        row = {
            "name": sample_name,
            "task": task,
            "source_layout_id": source_layout_id if source_layout_id else pd.NA,
            "group_i": group_i,
            "group_j": group_j,
            "num_motif_groups": len(group_ids),
            "excursion_trans_A": excursion_trans,
            "excursion_rot_deg": excursion_rot,
            "retention_trans_A": retention_trans,
            "retention_rot_deg": retention_rot,
        }
        row.update(pose_column_dict("native", native_pose))
        row.update(pose_column_dict("design", design_pose))
        row.update(pose_column_dict("fold", fold_pose))
        rows.append(row)
    return rows, design_rel


def _pair_mean_median(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan")}
    return {"mean": float(np.mean(arr)), "median": float(np.median(arr))}


def rms_over_values(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(arr))))


def frame_from_wxyz_trans(
    qw: float, qx: float, qy: float, qz: float, tx: float, ty: float, tz: float
) -> Frame:
    """Rebuild a relative pose from scalar-first quaternion + translation."""
    rot = Rotation.from_quat(
        [float(qx), float(qy), float(qz), float(qw)]
    ).as_matrix()
    trans = np.array([float(tx), float(ty), float(tz)], dtype=np.float64)
    return np.asarray(rot, dtype=np.float64), trans


def frame_from_pose_row(row: Mapping[str, Any], prefix: str) -> Frame | None:
    keys = [f"{prefix}_tij_{part}" for part in ("tx", "ty", "tz", "qw", "qx", "qy", "qz")]
    vals = []
    for key in keys:
        if key not in row:
            return None
        value = row[key]
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return None
        if pd.isna(value):
            return None
        vals.append(float(value))
    return frame_from_wxyz_trans(vals[3], vals[4], vals[5], vals[6], vals[0], vals[1], vals[2])


def relatives_from_pair_df(
    df: pd.DataFrame,
    prefix: str = "design",
) -> dict[str, dict[tuple[int, int], Frame]]:
    """Rebuild per-design relative poses from stored T_ij columns."""
    out: dict[str, dict[tuple[int, int], Frame]] = {}
    if df is None or df.empty:
        return out
    for record in df.to_dict("records"):
        frame = frame_from_pose_row(record, prefix)
        if frame is None:
            continue
        name = str(record["name"])
        pair = (int(record["group_i"]), int(record["group_j"]))
        out.setdefault(name, {})[pair] = frame
    return out


def layout_pair_distance(
    rel_a: Mapping[tuple[int, int], Frame],
    rel_b: Mapping[tuple[int, int], Frame],
) -> tuple[float, float]:
    """Inter-sample distance: RMS over group pairs (equal pair weight)."""
    trans_sq: list[float] = []
    rot_sq: list[float] = []
    for pair, pose_a in rel_a.items():
        if pair not in rel_b:
            continue
        trans, rot = relative_pose_delta(pose_a, rel_b[pair])
        trans_sq.append(trans * trans)
        rot_sq.append(rot * rot)
    if not trans_sq:
        return float("nan"), float("nan")
    return float(np.sqrt(np.mean(trans_sq))), float(np.sqrt(np.mean(rot_sq)))


def pairwise_layout_distance_matrices(
    relatives_by_sample: Mapping[str, dict[tuple[int, int], Frame]],
    sample_names: Sequence[str],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """NxN RMS layout distances. Diagonal is 0; missing shared pairs are NaN.

    Still a design×design matrix (resample this, never precomputed pair rows).
    """
    names = [name for name in sample_names if name in relatives_by_sample]
    n = len(names)
    trans = np.full((n, n), np.nan, dtype=np.float64)
    rot = np.full((n, n), np.nan, dtype=np.float64)
    if n == 0:
        return names, trans, rot
    np.fill_diagonal(trans, 0.0)
    np.fill_diagonal(rot, 0.0)
    if n == 1:
        return names, trans, rot
    pair_keys = sorted(
        {pair for name in names for pair in relatives_by_sample[name]}
    )
    if not pair_keys:
        return names, trans, rot
    trans_sq = np.zeros((n, n), dtype=np.float64)
    rot_sq = np.zeros((n, n), dtype=np.float64)
    counts = np.zeros((n, n), dtype=np.float64)
    for pair in pair_keys:
        present_idx: list[int] = []
        rotations: list[np.ndarray] = []
        translations: list[np.ndarray] = []
        for i, name in enumerate(names):
            frame = relatives_by_sample[name].get(pair)
            if frame is None:
                continue
            rot_m, origin = frame
            present_idx.append(i)
            rotations.append(np.asarray(rot_m, dtype=np.float64))
            translations.append(np.asarray(origin, dtype=np.float64).reshape(3))
        if len(present_idx) < 2:
            continue
        idx = np.asarray(present_idx, dtype=np.intp)
        t_sub = np.stack(translations, axis=0)
        r_sub = np.stack(rotations, axis=0)
        d_t = np.linalg.norm(t_sub[:, None, :] - t_sub[None, :, :], axis=-1)
        rel = np.einsum("iba,jbc->ijac", r_sub, r_sub)
        d_r = np.degrees(
            Rotation.from_matrix(rel.reshape(-1, 3, 3)).magnitude()
        ).reshape(len(idx), len(idx))
        ii, jj = np.meshgrid(idx, idx, indexing="ij")
        trans_sq[ii, jj] += d_t * d_t
        rot_sq[ii, jj] += d_r * d_r
        counts[ii, jj] += 1.0
    good = counts > 0
    trans = np.where(good, np.sqrt(trans_sq / np.maximum(counts, 1.0)), np.nan)
    rot = np.where(good, np.sqrt(rot_sq / np.maximum(counts, 1.0)), np.nan)
    np.fill_diagonal(trans, 0.0)
    np.fill_diagonal(rot, 0.0)
    return names, trans, rot


def pairwise_layout_distances(
    relatives_by_sample: Mapping[str, dict[tuple[int, int], Frame]],
    sample_names: Sequence[str],
) -> tuple[list[float], list[float]]:
    names, trans_mat, rot_mat = pairwise_layout_distance_matrices(
        relatives_by_sample, sample_names
    )
    if len(names) < 2:
        return [], []
    ii, jj = np.triu_indices(len(names), k=1)
    trans_vals = trans_mat[ii, jj]
    rot_vals = rot_mat[ii, jj]
    finite = np.isfinite(trans_vals)
    return trans_vals[finite].tolist(), rot_vals[finite].tolist()


def _bootstrap_unit_map(
    names: Sequence[str],
    cluster_ids: Mapping[str, str] | None,
) -> tuple[dict[str, str], str]:
    """Stage-2: resample source layouts when several designs share one. Else designs."""
    if not cluster_ids:
        return {name: name for name in names}, "design"
    mapping = {name: str(cluster_ids.get(name, name)) for name in names}
    n_units = len(set(mapping.values()))
    if n_units < len(names):
        return mapping, "source_layout"
    return {name: name for name in names}, "design"


def _resample_design_names(
    names: Sequence[str],
    unit_of: Mapping[str, str],
    rng: np.random.Generator,
) -> list[str]:
    names_by_unit: dict[str, list[str]] = {}
    for name in names:
        names_by_unit.setdefault(unit_of[name], []).append(name)
    units = list(names_by_unit)
    drawn = rng.choice(units, size=len(units), replace=True)
    out: list[str] = []
    for unit in drawn:
        out.extend(names_by_unit[str(unit)])
    return out


def _bootstrap_median_pairwise_from_matrices(
    trans_mat: np.ndarray,
    rot_mat: np.ndarray,
    names: Sequence[str],
    *,
    cluster_ids: Mapping[str, str] | None = None,
    n_boot: int = 500,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    empty = {
        "bootstrap_trans_ci_lo": float("nan"),
        "bootstrap_trans_ci_hi": float("nan"),
        "bootstrap_rot_ci_lo": float("nan"),
        "bootstrap_rot_ci_hi": float("nan"),
        "n_boot": int(n_boot),
    }
    names = list(names)
    if n_boot < 1 or len(names) < 2:
        return empty
    unit_of, _unit_name = _bootstrap_unit_map(names, cluster_ids)
    rng = rng or np.random.default_rng(0)
    name_to_i = {name: i for i, name in enumerate(names)}
    med_t: list[float] = []
    med_r: list[float] = []
    for _ in range(int(n_boot)):
        drawn = _resample_design_names(names, unit_of, rng)
        idx = np.asarray([name_to_i[name] for name in drawn if name in name_to_i], dtype=np.intp)
        if idx.size < 2:
            continue
        ii, jj = np.triu_indices(idx.size, k=1)
        trans_vals = trans_mat[idx[ii], idx[jj]]
        rot_vals = rot_mat[idx[ii], idx[jj]]
        finite = np.isfinite(trans_vals)
        if not np.any(finite):
            continue
        med_t.append(float(np.median(trans_vals[finite])))
        med_r.append(float(np.median(rot_vals[finite])))
    if len(med_t) < 2:
        return empty
    return {
        "bootstrap_trans_ci_lo": float(np.percentile(med_t, 2.5)),
        "bootstrap_trans_ci_hi": float(np.percentile(med_t, 97.5)),
        "bootstrap_rot_ci_lo": float(np.percentile(med_r, 2.5)),
        "bootstrap_rot_ci_hi": float(np.percentile(med_r, 97.5)),
        "n_boot": int(n_boot),
    }


def bootstrap_median_pairwise(
    relatives_by_sample: Mapping[str, dict[tuple[int, int], Frame]],
    sample_names: Sequence[str],
    *,
    cluster_ids: Mapping[str, str] | None = None,
    n_boot: int = 500,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Resample designs (or source layouts), recompute pairwise medians each replicate.

    Never bootstrap the precomputed N(N-1)/2 pair rows. Distances are looked up
    in a design×design matrix so duplicate resampled designs still contribute
    a 0 self-distance, matching combinations over the drawn name list.
    """
    names, trans_mat, rot_mat = pairwise_layout_distance_matrices(
        relatives_by_sample, sample_names
    )
    return _bootstrap_median_pairwise_from_matrices(
        trans_mat,
        rot_mat,
        names,
        cluster_ids=cluster_ids,
        n_boot=n_boot,
        rng=rng,
    )


def bootstrap_median_of_designs(
    values_by_name: Mapping[str, float],
    sample_names: Sequence[str],
    *,
    cluster_ids: Mapping[str, str] | None = None,
    n_boot: int = 500,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Bootstrap CI for a per-design scalar (excursion/retention), resampling designs."""
    names = [name for name in sample_names if name in values_by_name]
    if n_boot < 1 or len(names) < 2:
        return float("nan"), float("nan")
    unit_of, _ = _bootstrap_unit_map(names, cluster_ids)
    rng = rng or np.random.default_rng(0)
    medians: list[float] = []
    for _ in range(int(n_boot)):
        drawn = _resample_design_names(names, unit_of, rng)
        vals = [values_by_name[name] for name in drawn if np.isfinite(values_by_name[name])]
        if not vals:
            continue
        medians.append(float(np.median(vals)))
    if len(medians) < 2:
        return float("nan"), float("nan")
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def pairwise_layout_stats(
    relatives_by_sample: Mapping[str, dict[tuple[int, int], Frame]],
    sample_names: Sequence[str],
    *,
    cluster_ids: Mapping[str, str] | None = None,
    n_boot: int = 0,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    names, trans_mat, rot_mat = pairwise_layout_distance_matrices(
        relatives_by_sample, sample_names
    )
    if len(names) >= 2:
        ii, jj = np.triu_indices(len(names), k=1)
        trans_vals = trans_mat[ii, jj]
        rot_vals = rot_mat[ii, jj]
        finite = np.isfinite(trans_vals)
        trans_vals = trans_vals[finite]
        rot_vals = rot_vals[finite]
    else:
        trans_vals = np.asarray([], dtype=np.float64)
        rot_vals = trans_vals
    unit_of, unit_name = _bootstrap_unit_map(names, cluster_ids)
    n = len(names)
    defined = n >= 2 and trans_vals.size > 0
    stats: dict[str, Any] = {
        "median_pairwise_layout_translation_distance": (
            float(np.median(trans_vals)) if trans_vals.size else float("nan")
        ),
        "median_pairwise_layout_rotation_distance": (
            float(np.median(rot_vals)) if rot_vals.size else float("nan")
        ),
        "num_design_pairs": int(trans_vals.size),
        "n_designs": int(n),
        "pairwise_defined": bool(defined),
        "weak_n": bool(n < 5),
        "bootstrap_unit": unit_name,
        "n_bootstrap_units": int(len(set(unit_of.values()))) if names else 0,
    }
    boot = _bootstrap_median_pairwise_from_matrices(
        trans_mat,
        rot_mat,
        names,
        cluster_ids=cluster_ids,
        n_boot=n_boot,
        rng=rng,
    )
    stats.update(boot)
    return stats


def layout_distance_matrix(
    relatives_by_sample: Mapping[str, dict[tuple[int, int], Frame]],
    names: Sequence[str],
    *,
    kind: str = "trans",
    tau_trans: float = 2.0,
    tau_rot: float = 15.0,
) -> tuple[list[str], np.ndarray]:
    """Pairwise distance matrix. ``kind``: trans | rot | combined.

    Combined uses ``D² = (d_trans/tau_trans)² + (d_rot/tau_rot)²`` (visualization
    only; main numeric results stay in Å and degrees). Default taus: 2 Å, 15 deg.
    """
    valid = [name for name in names if name in relatives_by_sample]
    n = len(valid)
    mat = np.zeros((n, n), dtype=np.float64)
    for i, name_a in enumerate(valid):
        for j in range(i + 1, n):
            d_t, d_r = layout_pair_distance(
                relatives_by_sample[name_a], relatives_by_sample[valid[j]]
            )
            if kind == "trans":
                value = d_t
            elif kind == "rot":
                value = d_r
            else:
                value = float(
                    np.sqrt((d_t / tau_trans) ** 2 + (d_r / tau_rot) ** 2)
                )
            mat[i, j] = mat[j, i] = value if np.isfinite(value) else 0.0
    return valid, mat


def classical_mds(dist_matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Deterministic Torgerson metric MDS (double-centering + eigendecomposition)."""
    dist = np.asarray(dist_matrix, dtype=np.float64)
    n = dist.shape[0]
    if n == 0:
        return np.zeros((0, n_components), dtype=np.float64)
    d2 = dist ** 2
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ d2 @ centering
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order][:n_components], 0.0, None)
    eigvecs = eigvecs[:, order][:, :n_components]
    return eigvecs * np.sqrt(eigvals)


def pairwise_layout_stats_by_task(
    relatives_by_sample: Mapping[str, dict[tuple[int, int], Frame]],
    names_by_task: Mapping[str, Sequence[str]],
    *,
    cluster_ids: Mapping[str, str] | None = None,
    n_boot: int = 0,
    rng: np.random.Generator | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        task: pairwise_layout_stats(
            relatives_by_sample,
            names,
            cluster_ids=cluster_ids,
            n_boot=n_boot,
            rng=rng,
        )
        for task, names in names_by_task.items()
    }


def _coverage_record(
    sample_name: str,
    metadata: Mapping[str, Any] | None,
    *,
    reason: str,
    detail: str = "",
    n_layout_pairs: int = 0,
) -> dict[str, Any]:
    meta = metadata if isinstance(metadata, Mapping) else {}
    raw_task = str(meta.get("name") or "")
    return {
        "name": sample_name,
        "task": _parent_task(meta, raw_task),
        "raw_task": raw_task,
        "num_groups_expected": int(expected_group_count(meta)) if meta else 0,
        "n_layout_pairs": int(n_layout_pairs),
        "reason": reason,
        "detail": detail,
        "eval_error": str(meta.get("eval_error") or ""),
    }


def _one_shot_layout_job(
    sample_name: str,
    metadata: dict[str, Any],
    folded_path: str | None,
    repo_root: str | None,
    pass_final: bool,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], Frame] | None, dict[str, Any]]:
    """Worker: score one design. Returns (rows, design_rel, coverage)."""
    if metadata.get("eval_error"):
        return (
            [],
            None,
            _coverage_record(
                sample_name,
                metadata,
                reason="eval_error",
                detail=str(metadata.get("eval_error")),
            ),
        )
    if expected_group_count(metadata) < 2:
        return (
            [],
            None,
            _coverage_record(sample_name, metadata, reason="not_multigroup"),
        )
    parser = PDBParser(QUIET=True)
    repo_path = Path(repo_root) if repo_root else None
    folded = Path(folded_path) if folded_path else None
    try:
        sample_rows, design_rel = compute_sample_layout_pair_metrics(
            sample_name,
            metadata,
            parser=parser,
            folded_path=folded,
            repo_root=repo_path,
        )
    except LayoutSkip as exc:
        logger.warning("Layout skip %s (%s): %s", sample_name, exc.reason, exc.detail)
        return [], None, _coverage_record(
            sample_name, metadata, reason=exc.reason, detail=exc.detail
        )
    except Exception as exc:
        logger.exception("Failed layout metrics for %s", sample_name)
        return [], None, _coverage_record(
            sample_name, metadata, reason="layout_exception", detail=str(exc)
        )
    if not sample_rows:
        return [], None, _coverage_record(
            sample_name, metadata, reason="no_pair_rows"
        )
    for row in sample_rows:
        row["pass_final"] = bool(pass_final)
    return (
        sample_rows,
        design_rel,
        _coverage_record(
            sample_name,
            metadata,
            reason="ok",
            n_layout_pairs=len(sample_rows),
        ),
    )


def compute_run_layout_metrics(
    samples_metadata: dict[str, Any],
    *,
    folding_df: pd.DataFrame | None = None,
    esmfold_dir: Path | None = None,
    repo_root: Path | None = None,
    workers: int = 1,
    n_boot: int = 0,
    rng_seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute pair metrics plus task-level excursion / pairwise / retention summaries.

    Stage-2 sample-level ``task`` ids are remapped to the parent motif *before*
    pairwise layout distances so designs of the same motif are pooled.
    """
    pass_final_by_name: dict[str, bool] = {}
    folded_path_by_name: dict[str, str] = {}
    if folding_df is not None and not folding_df.empty:
        for _, row in folding_df.iterrows():
            name = str(row["name"])
            if "pass_final" in folding_df.columns:
                pass_final_by_name[name] = bool(row["pass_final"])
            if "folded_path" in folding_df.columns and pd.notna(row.get("folded_path")):
                folded_path_by_name[name] = str(row["folded_path"])

    jobs: list[tuple[str, dict[str, Any], str | None, bool]] = []
    coverage_rows: list[dict[str, Any]] = []
    cluster_ids: dict[str, str] = {}
    for sample_name, metadata in samples_metadata.items():
        if not isinstance(metadata, dict):
            coverage_rows.append(
                _coverage_record(str(sample_name), None, reason="invalid_metadata")
            )
            continue
        if metadata.get("source_layout_id"):
            cluster_ids[str(sample_name)] = str(metadata["source_layout_id"])
        folded = _one_shot_folded_path(
            esmfold_dir,
            sample_name,
            folded_path_by_name.get(sample_name),
            repo_root=repo_root,
        )
        jobs.append(
            (
                str(sample_name),
                metadata,
                str(folded) if folded is not None else None,
                bool(pass_final_by_name.get(sample_name, False)),
            )
        )

    pair_rows: list[dict[str, Any]] = []
    relatives_by_sample: dict[str, dict[tuple[int, int], Frame]] = {}
    names_by_task: dict[str, list[str]] = {}
    repo_s = str(repo_root) if repo_root is not None else None
    n_workers = max(1, int(workers))

    def _accept(
        sample_name: str,
        sample_rows: list[dict[str, Any]],
        design_rel: dict[tuple[int, int], Frame] | None,
        coverage: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        coverage_rows.append(coverage)
        if not sample_rows or design_rel is None:
            return
        pair_rows.extend(sample_rows)
        relatives_by_sample[sample_name] = design_rel
        task = str(sample_rows[0].get("task") or _parent_task(metadata))
        names_by_task.setdefault(task, []).append(sample_name)

    if n_workers == 1:
        for i, (sample_name, metadata, folded, pass_final) in enumerate(jobs, start=1):
            sample_rows, design_rel, coverage = _one_shot_layout_job(
                sample_name, metadata, folded, repo_s, pass_final
            )
            _accept(sample_name, sample_rows, design_rel, coverage, metadata)
            if i % 200 == 0 or i == len(jobs):
                logger.info("One-shot layout %s/%s designs", i, len(jobs))
    else:
        logger.info("One-shot layout %s designs, %s workers", len(jobs), n_workers)
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            future_map = {
                pool.submit(
                    _one_shot_layout_job,
                    sample_name,
                    metadata,
                    folded,
                    repo_s,
                    pass_final,
                ): (sample_name, metadata)
                for sample_name, metadata, folded, pass_final in jobs
            }
            for i, fut in enumerate(as_completed(future_map), start=1):
                sample_name, metadata = future_map[fut]
                sample_rows, design_rel, coverage = fut.result()
                _accept(sample_name, sample_rows, design_rel, coverage, metadata)
                if i % 200 == 0 or i == len(jobs):
                    logger.info("One-shot layout %s/%s designs", i, len(jobs))

    pair_df = pd.DataFrame(pair_rows) if pair_rows else pd.DataFrame()
    skipped = sum(1 for row in coverage_rows if row.get("reason") != "ok")
    if skipped:
        logger.info("Skipped %d samples for layout metrics", skipped)

    reason_counts: dict[str, int] = {}
    for row in coverage_rows:
        reason = str(row.get("reason") or "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    summaries: dict[str, Any] = {
        "layout_frame": LAYOUT_FRAME,
        "layout_template": "native_group_backbone_nca_c",
        "pair_weighting": "equal_pairs_rms",
        "pose_rotation": "quaternion_wxyz_hamilton_qw_ge_0",
        "num_pair_rows": int(len(pair_df)),
        "num_samples_scored": int(pair_df["name"].nunique()) if not pair_df.empty else 0,
        "num_samples_considered": int(len(coverage_rows)),
        "skip_reason_counts": reason_counts,
        "coverage_rows": coverage_rows,
        "n_boot": int(n_boot),
    }
    if pair_df.empty:
        return pair_df, summaries

    def _layer_stats(frame: pd.DataFrame, trans_col: str, rot_col: str, prefix: str) -> dict[str, float]:
        sample_trans = frame.groupby("name")[trans_col].apply(rms_over_values)
        sample_rot = frame.groupby("name")[rot_col].apply(rms_over_values)
        trans_stats = _pair_mean_median(sample_trans.tolist())
        rot_stats = _pair_mean_median(sample_rot.tolist())
        return {
            f"{prefix}_trans_A_mean": trans_stats["mean"],
            f"{prefix}_trans_A_median": trans_stats["median"],
            f"{prefix}_rot_deg_mean": rot_stats["mean"],
            f"{prefix}_rot_deg_median": rot_stats["median"],
        }

    summaries.update(_layer_stats(pair_df, "excursion_trans_A", "excursion_rot_deg", "excursion"))
    summaries.update(_layer_stats(pair_df, "retention_trans_A", "retention_rot_deg", "retention"))

    rng = np.random.default_rng(int(rng_seed))
    pairwise_all_by_task = pairwise_layout_stats_by_task(
        relatives_by_sample, names_by_task, cluster_ids=cluster_ids, n_boot=n_boot, rng=rng
    )
    final_names_by_task = {
        task: [name for name in names if pass_final_by_name.get(name, False)]
        for task, names in names_by_task.items()
    }
    pairwise_final_by_task = pairwise_layout_stats_by_task(
        relatives_by_sample,
        final_names_by_task,
        cluster_ids=cluster_ids,
        n_boot=n_boot,
        rng=rng,
    )
    summaries["pairwise_all_by_task"] = pairwise_all_by_task
    summaries["pairwise_pass_final_by_task"] = pairwise_final_by_task
    return pair_df, summaries
