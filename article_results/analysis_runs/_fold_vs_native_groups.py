#!/usr/bin/env python3
"""Per-group ESMFold (and optional design) Kabsch vs native crystal.

This is not fold-vs-design (esm_compute_rmsds.py / group_rmsd.csv). Residue
pairs come from layout_manifest.jsonl + enrich_samples_metadata_from_layout_manifest,
then source_residue_keys (native_chain/native_res_idx ↔ output_chain/output_res_idx,
group_id). Same pairing as multimotif incrustation; residue lookup uses the
layout indexing helpers. Holey mapping → eval_error / skip, never shortened lists.

Writes sidecars under <analysis_id>__criteria_v3/. Does not touch __criteria_v2/,
sampling, ESMFold, PMPNN, Foldseek, All/Unique, or CRITERIA_VERSION.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

from nomadflow.benchmark.analysis_inputs import (
    list_esmfold_structure_paths,
    load_samples_metadata,
    parent_motif_task_id,
    prefer_samples_eval_pdb,
    remap_samples_metadata_paths,
)
from nomadflow.benchmark.layout_manifest_eval import (
    enrich_samples_metadata_from_layout_manifest,
)
from nomadflow.benchmark.layout_pose_metrics import _get_residue
from nomadflow.benchmark.multimotif_metrics import (
    _group_residue_pairs,
    _is_heavy_atom,
    _kabsch_rmsd,
)

REPO = Path(
    os.environ.get("REPO_ROOT")
    or os.environ.get("ZENPROTEIN_REPO")
    or Path(__file__).resolve().parents[2]
)
HA_MAX = 1.5
CA_MAX = 1.0
CORE_SIZES = (3, 4)
SEGMENT_SIZES = (10, 16, 20)
# Matched 2-group catalytic slice (same tasks in one-rigid and full stage1).
# Exclude *_ext2x10 from cores / default table; they can join "segment 10" only.
TABLE_TASKS = frozenset(
    {
        "1QJG_ext10",
        "1QJG_ext16",
        "5AOU_ext10",
        "5AOU_ext20",
        "5AOU_QUAD_ext10",
        "5AOU_QUAD_ext20",
        "7K4V_ext10",
        "7K4V_ext20",
    }
)
EXT2X10_TASKS = frozenset(
    {
        "1QJG_ext2x10",
        "5AOU_ext2x10",
        "5AOU_QUAD_ext2x10",
        "7K4V_ext2x10",
    }
)
RUNS: dict[str, dict[str, str]] = {
    "one-rigid": {
        "sampling_id": "one_motif_proteinzen",
        "analysis_id": "one_motif_proteinzen",
        "label": "one-rigid",
    },
    "multi-group": {
        "sampling_id": "0c62dc8a__tlock05_full_stage_1",
        "analysis_id": "0c62dc8a__tlock05_full_stage1",
        "label": "multi-group",
    },
    "se3-stage1": {
        "sampling_id": "0c62dc8a__tlock05_se3_stage_1",
        "analysis_id": "0c62dc8a__tlock05_se3_stage1",
        "label": "se3-stage1",
    },
    "se3-stage2": {
        "sampling_id": "0c62dc8a__tlock05_se3_stage_2",
        "analysis_id": "0c62dc8a__tlock05_se3_stage2",
        "label": "se3-stage2",
        # Same as v2 analysis: Stage-2 reads Stage-1 layout_manifest.jsonl.
        "layout_manifest_sampling_id": "0c62dc8a__tlock05_se3_stage_1",
    },
    "full-stage2": {
        "sampling_id": "0c62dc8a__tlock05_full_stage_2",
        "analysis_id": "0c62dc8a__tlock05_full_stage2",
        "label": "full-stage2",
        "layout_manifest_sampling_id": "0c62dc8a__tlock05_full_stage_1",
    },
}
REQUIRED_RUNS = ("one-rigid", "multi-group")
OPTIONAL_RUNS = ("full-stage2", "se3-stage1", "se3-stage2")
SLICE_ORDER = [
    ("core 3", 1, 3),
    ("core 4", 1, 4),
    ("segment 10", 2, 10),
    ("segment 16", 2, 16),
    ("segment 20", 2, 20),
]


def _sampling_path(sampling_id: str) -> Path:
    return REPO / "article_results" / "sampling_runs" / sampling_id


def _v3_dir(analysis_id: str) -> Path:
    return REPO / "article_results" / "analysis_runs" / f"{analysis_id}__criteria_v3"


def _v2_dir(analysis_id: str) -> Path:
    return REPO / "article_results" / "analysis_runs" / f"{analysis_id}__criteria_v2"


def _load_enriched_metadata(
    sampling: Path, *, layout_manifest_path: Path | None = None
) -> dict[str, Any]:
    meta = load_samples_metadata(sampling)
    meta = remap_samples_metadata_paths(meta, repo_root=REPO)
    enriched = enrich_samples_metadata_from_layout_manifest(
        meta,
        layout_manifest_path=layout_manifest_path,
        sampling_run=sampling,
    )
    # Layout entries may still carry the old server prefix for native_pdb.
    return remap_samples_metadata_paths(enriched, repo_root=REPO)


def _task_id(sample_name: str, md: dict[str, Any]) -> str:
    raw = md.get("name")
    if raw is None:
        return parent_motif_task_id(sample_name, md)
    return parent_motif_task_id(str(raw), md)


def _protocol_from_label(label: str) -> str:
    if label == "one_shot" or label.startswith("one_shot"):
        return "one_shot"
    if label.startswith("pmpnn_sample_"):
        return "pmpnn_esmfold"
    return "other"


def _slice_name(group_id: int, group_size: int) -> str | None:
    gid = int(group_id)
    nres = int(group_size)
    if gid == 1 and nres in CORE_SIZES:
        return f"core {nres}"
    # Both 10-res arms of ext2x10 (gid2 and gid3) count as "segment 10".
    if gid != 1 and nres in SEGMENT_SIZES:
        return f"segment {nres}"
    return None


def _tasks_for_slice(slice_name: str, *, include_ext2x10_seg10: bool) -> frozenset[str]:
    if include_ext2x10_seg10 and slice_name == "segment 10":
        return TABLE_TASKS | EXT2X10_TASKS
    return TABLE_TASKS


def _native_payload_for_pairs(native_structure, pairs: list) -> dict[str, Any]:
    """Extract native CA/HA for the full pair list, or raise KeyError listing misses."""
    missing: list[str] = []
    residues = []
    for native_pair, _sample_pair in pairs:
        chain_id, res_idx = native_pair
        try:
            residue = _get_residue(
                native_structure, chain_id, res_idx, remap_missing=False
            )
        except KeyError:
            missing.append(f"{chain_id}:{res_idx}")
            residues.append(None)
            continue
        residues.append(residue)
    if missing:
        raise KeyError("missing_residue_in_native:" + ",".join(missing))
    native_ca = []
    native_ha: list[list[tuple[str, np.ndarray]]] = []
    for residue in residues:
        if "CA" not in residue:
            raise KeyError(f"missing_ca_in_native:{residue.id}")
        native_ca.append(np.asarray(residue["CA"].coord, dtype=np.float64))
        atoms = [
            (atom.name.strip(), np.asarray(atom.coord, dtype=np.float64))
            for atom in residue
            if _is_heavy_atom(atom.name)
        ]
        native_ha.append(atoms)
    return {
        "native_ca": np.stack(native_ca),
        "native_ha": native_ha,
        "sample_pairs": [sample_pair for _native_pair, sample_pair in pairs],
        "native_pairs": [native_pair for native_pair, _sample_pair in pairs],
        "group_size": len(pairs),
    }


def _lookup_fold_residues(structure, sample_pairs: list[tuple[str, int]]) -> list:
    missing: list[str] = []
    residues = []
    for chain_id, res_idx in sample_pairs:
        try:
            residue = _get_residue(
                structure, chain_id, res_idx, remap_missing=True
            )
        except KeyError:
            missing.append(f"{chain_id}:{res_idx}")
            residues.append(None)
            continue
        residues.append(residue)
    if missing:
        raise KeyError("missing_residue_in_structure:" + ",".join(missing))
    return residues


def _kabsch_from_payload(residues: list, payload: dict[str, Any]) -> tuple[float, float]:
    fold_ca = []
    for residue in residues:
        if "CA" not in residue:
            raise KeyError(f"missing_ca:{residue.id}")
        fold_ca.append(np.asarray(residue["CA"].coord, dtype=np.float64))
    ca = _kabsch_rmsd(np.stack(fold_ca), payload["native_ca"])
    native_ha_coords: list[np.ndarray] = []
    fold_ha_coords: list[np.ndarray] = []
    for residue, native_atoms in zip(residues, payload["native_ha"], strict=True):
        for atom_name, native_coord in native_atoms:
            if atom_name not in residue:
                continue
            fold_ha_coords.append(np.asarray(residue[atom_name].coord, dtype=np.float64))
            native_ha_coords.append(native_coord)
    if not native_ha_coords:
        raise ValueError("No matching heavy atoms found for residue pairs")
    ha = _kabsch_rmsd(np.stack(fold_ha_coords), np.stack(native_ha_coords))
    return float(ca), float(ha)


def _prepare_run(
    run_key: str,
    *,
    include_one_shot: bool,
    task_filter: frozenset[str] | None,
) -> dict[str, Any]:
    spec = RUNS[run_key]
    sampling = _sampling_path(spec["sampling_id"])
    if not sampling.is_dir():
        raise FileNotFoundError(f"missing sampling run: {sampling}")
    layout_manifest_path = None
    layout_src = spec.get("layout_manifest_sampling_id")
    if layout_src:
        layout_manifest_path = _sampling_path(layout_src) / "layout_manifest.jsonl"
        if not layout_manifest_path.is_file():
            raise FileNotFoundError(f"missing layout manifest: {layout_manifest_path}")
    meta = _load_enriched_metadata(sampling, layout_manifest_path=layout_manifest_path)
    esmfold_dir = sampling / "esmfold"
    parser = PDBParser(QUIET=True)
    native_cache: dict[str, Any] = {}
    sample_infos: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    n_by_task: dict[str, int] = defaultdict(int)
    keys_by_task: dict[str, int] = defaultdict(int)
    err_by_task: dict[str, int] = defaultdict(int)
    size_by_task: dict[str, dict[int, int]] = {}

    for sample_name, md in meta.items():
        if not isinstance(md, dict):
            continue
        task = _task_id(sample_name, md)
        if task_filter is not None and task not in task_filter:
            continue
        n_by_task[task] += 1
        if md.get("eval_error"):
            err_by_task[task] += 1
            error_rows.extend(
                _error_group_placeholders(
                    run_key=run_key,
                    spec=spec,
                    sample_name=sample_name,
                    task=task,
                    status=f"eval_error:{md.get('eval_error')}",
                    md=md,
                )
            )
            continue
        keys = md.get("source_residue_keys")
        if not isinstance(keys, list) or not keys:
            err_by_task[task] += 1
            error_rows.extend(
                _error_group_placeholders(
                    run_key=run_key,
                    spec=spec,
                    sample_name=sample_name,
                    task=task,
                    status="eval_error:missing_source_residue_keys",
                    md=md,
                )
            )
            continue
        group_ids = sorted(
            {
                int(entry["group_id"])
                for entry in keys
                if isinstance(entry, dict) and entry.get("group_id") is not None
            }
        )
        if not group_ids:
            err_by_task[task] += 1
            error_rows.extend(
                _error_group_placeholders(
                    run_key=run_key,
                    spec=spec,
                    sample_name=sample_name,
                    task=task,
                    status="eval_error:missing_group_id",
                    md=md,
                    keys=keys,
                )
            )
            continue
        native_pdb = md.get("native_pdb")
        if not native_pdb or not Path(str(native_pdb)).is_file():
            err_by_task[task] += 1
            error_rows.extend(
                _error_group_placeholders(
                    run_key=run_key,
                    spec=spec,
                    sample_name=sample_name,
                    task=task,
                    status="eval_error:native_pdb_missing",
                    md=md,
                    group_ids=group_ids,
                    keys=keys,
                )
            )
            continue
        native_key = str(native_pdb)
        if native_key not in native_cache:
            native_cache[native_key] = parser.get_structure("native", native_key)
        native_structure = native_cache[native_key]
        payloads: dict[int, dict[str, Any]] = {}
        holey = False
        holey_status = ""
        for gid in group_ids:
            pairs = _group_residue_pairs(keys, gid)
            if not pairs:
                holey = True
                holey_status = f"eval_error:empty_group_{gid}"
                break
            try:
                payloads[gid] = _native_payload_for_pairs(native_structure, pairs)
            except KeyError as exc:
                holey = True
                holey_status = f"eval_error:{exc}"
                break
        if holey:
            err_by_task[task] += 1
            error_rows.extend(
                _error_group_placeholders(
                    run_key=run_key,
                    spec=spec,
                    sample_name=sample_name,
                    task=task,
                    status=holey_status,
                    md=md,
                    group_ids=group_ids,
                    keys=keys,
                )
            )
            continue
        sizes = {gid: int(payloads[gid]["group_size"]) for gid in group_ids}
        size_by_task.setdefault(task, sizes)
        keys_by_task[task] += 1
        eval_path = prefer_samples_eval_pdb(md.get("path"))
        sample_infos.append(
            {
                "run_key": run_key,
                "run_id": spec["analysis_id"],
                "run_label": spec["label"],
                "name": sample_name,
                "task": task,
                "eval_path": str(eval_path) if eval_path is not None else None,
                "payloads": payloads,
                "sizes": sizes,
            }
        )

    fold_jobs: list[dict[str, Any]] = []
    design_jobs: list[dict[str, Any]] = []
    n_fold_by_protocol: dict[str, int] = defaultdict(int)
    for info in sample_infos:
        if info["eval_path"] and Path(info["eval_path"]).is_file():
            design_jobs.append(
                {
                    **{k: info[k] for k in ("run_key", "run_id", "run_label", "name", "task", "sizes")},
                    "structure_path": info["eval_path"],
                    "label": "design",
                    "protocol": "design_native",
                    "payloads": info["payloads"],
                }
            )
        for label, fold_path in list_esmfold_structure_paths(esmfold_dir, info["name"]):
            protocol = _protocol_from_label(label)
            if protocol == "one_shot" and not include_one_shot:
                continue
            if protocol not in {"one_shot", "pmpnn_esmfold"}:
                continue
            n_fold_by_protocol[protocol] += 1
            fold_jobs.append(
                {
                    **{k: info[k] for k in ("run_key", "run_id", "run_label", "name", "task", "sizes")},
                    "structure_path": str(fold_path),
                    "label": label,
                    "protocol": protocol,
                    "payloads": info["payloads"],
                }
            )

    return {
        "spec": spec,
        "sampling": sampling,
        "n_by_task": dict(n_by_task),
        "keys_by_task": dict(keys_by_task),
        "err_by_task": dict(err_by_task),
        "size_by_task": size_by_task,
        "sample_infos": sample_infos,
        "fold_jobs": fold_jobs,
        "design_jobs": design_jobs,
        "error_rows": error_rows,
        "n_fold_by_protocol": dict(n_fold_by_protocol),
    }


def _error_group_placeholders(
    *,
    run_key: str,
    spec: dict[str, str],
    sample_name: str,
    task: str,
    status: str,
    md: dict[str, Any],
    group_ids: list[int] | None = None,
    keys: list | None = None,
) -> list[dict[str, Any]]:
    sizes: dict[int, int] = {}
    if keys is not None and group_ids:
        for gid in group_ids:
            sizes[gid] = len(_group_residue_pairs(keys, gid))
    gids = group_ids or [1, 2]
    rows = []
    for gid in gids:
        rows.append(
            {
                "run_key": run_key,
                "run_id": spec["analysis_id"],
                "run_label": spec["label"],
                "name": sample_name,
                "task": task,
                "label": "",
                "protocol": "unscored",
                "group_id": int(gid),
                "group_size": int(sizes.get(gid, 0)),
                "motif_ca_rmsd": float("nan"),
                "motif_ha_rmsd": float("nan"),
                "rmsd_status": status,
            }
        )
    return rows


def _score_job(job: dict[str, Any]) -> list[dict[str, Any]]:
    import warnings

    warnings.filterwarnings(
        "ignore",
        message="Optimal rotation is not uniquely or poorly defined",
    )
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("mobile", job["structure_path"])
    except Exception as exc:
        return [
            {
                "run_key": job["run_key"],
                "run_id": job["run_id"],
                "run_label": job["run_label"],
                "name": job["name"],
                "task": job["task"],
                "label": job["label"],
                "protocol": job["protocol"],
                "group_id": int(gid),
                "group_size": int(job["sizes"][gid]),
                "motif_ca_rmsd": float("nan"),
                "motif_ha_rmsd": float("nan"),
                "rmsd_status": f"failed:{type(exc).__name__}",
            }
            for gid in job["payloads"]
        ]
    try:
        residues_by_gid = {}
        for gid, payload in job["payloads"].items():
            residues_by_gid[gid] = _lookup_fold_residues(structure, payload["sample_pairs"])
    except KeyError as exc:
        status = f"eval_error:{exc}"
        return [
            {
                "run_key": job["run_key"],
                "run_id": job["run_id"],
                "run_label": job["run_label"],
                "name": job["name"],
                "task": job["task"],
                "label": job["label"],
                "protocol": job["protocol"],
                "group_id": int(gid),
                "group_size": int(job["sizes"][gid]),
                "motif_ca_rmsd": float("nan"),
                "motif_ha_rmsd": float("nan"),
                "rmsd_status": status,
            }
            for gid in job["payloads"]
        ]
    rows = []
    for gid, payload in job["payloads"].items():
        try:
            ca, ha = _kabsch_from_payload(residues_by_gid[gid], payload)
            status = "ok"
        except Exception as exc:
            ca = ha = float("nan")
            status = f"failed:{type(exc).__name__}"
        rows.append(
            {
                "run_key": job["run_key"],
                "run_id": job["run_id"],
                "run_label": job["run_label"],
                "name": job["name"],
                "task": job["task"],
                "label": job["label"],
                "protocol": job["protocol"],
                "group_id": int(gid),
                "group_size": int(payload["group_size"]),
                "motif_ca_rmsd": ca,
                "motif_ha_rmsd": ha,
                "rmsd_status": status,
            }
        )
    return rows


def _probe_structures(prepared: dict[str, Any], *, n_samples: int = 2) -> list[str]:
    """Check eval + one fold PDB residue presence; never shorten lists."""
    problems: list[str] = []
    parser = PDBParser(QUIET=True)
    esmfold_dir = prepared["sampling"] / "esmfold"
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for info in prepared["sample_infos"]:
        by_task[info["task"]].append(info)
    for task, infos in sorted(by_task.items()):
        for info in infos[:n_samples]:
            for gid, payload in info["payloads"].items():
                n_native = len(payload["native_pairs"])
                n_sample = len(payload["sample_pairs"])
                if n_native != n_sample or n_native != payload["group_size"]:
                    problems.append(
                        f"{prepared['spec']['label']} {task} {info['name']} gid={gid}: "
                        f"pair length mismatch native={n_native} sample={n_sample} "
                        f"size={payload['group_size']}"
                    )
            eval_path = info["eval_path"]
            if not eval_path or not Path(eval_path).is_file():
                problems.append(
                    f"{prepared['spec']['label']} {task} {info['name']}: missing samples_eval PDB"
                )
            else:
                eval_struct = parser.get_structure("eval", eval_path)
                for gid, payload in info["payloads"].items():
                    try:
                        _lookup_fold_residues(eval_struct, payload["sample_pairs"])
                    except KeyError as exc:
                        problems.append(
                            f"{prepared['spec']['label']} {task} {info['name']} "
                            f"eval gid={gid}: {exc}"
                        )
            folds = list_esmfold_structure_paths(esmfold_dir, info["name"])
            pmpnn = [(label, path) for label, path in folds if label.startswith("pmpnn_sample_")]
            if not pmpnn:
                problems.append(
                    f"{prepared['spec']['label']} {task} {info['name']}: no PMPNN ESMFold PDBs"
                )
                continue
            label, fold_path = pmpnn[0]
            fold_struct = parser.get_structure("fold", str(fold_path))
            for gid, payload in info["payloads"].items():
                try:
                    _lookup_fold_residues(fold_struct, payload["sample_pairs"])
                except KeyError as exc:
                    problems.append(
                        f"{prepared['spec']['label']} {task} {info['name']} "
                        f"{label} gid={gid}: {exc}"
                    )
    return problems


def _print_prepare_summary(prepared: dict[str, Any]) -> None:
    spec = prepared["spec"]
    print(f"\n=== {spec['label']}  sampling={spec['sampling_id']}  analysis={spec['analysis_id']}")
    print(
        f"  samples in table tasks: {sum(prepared['n_by_task'].values())}  "
        f"ok keys: {sum(prepared['keys_by_task'].values())}  "
        f"eval_error/skip: {sum(prepared['err_by_task'].values())}  "
        f"fold jobs: {len(prepared['fold_jobs'])}  "
        f"design jobs: {len(prepared['design_jobs'])}  "
        f"protocols={prepared['n_fold_by_protocol']}"
    )
    for task in sorted(TABLE_TASKS):
        n = prepared["n_by_task"].get(task, 0)
        if n == 0:
            continue
        sizes = prepared["size_by_task"].get(task, {})
        size_txt = ",".join(f"gid{gid}={nres}" for gid, nres in sorted(sizes.items()))
        print(
            f"  {task:18s} n={n:4d} keys={prepared['keys_by_task'].get(task, 0):4d} "
            f"err={prepared['err_by_task'].get(task, 0):4d}  {size_txt}"
        )


def _run_jobs(jobs: list[dict[str, Any]], *, workers: int) -> pd.DataFrame:
    if not jobs:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    workers = max(1, int(workers))
    print(f"scoring {len(jobs)} structures workers={workers}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, part in enumerate(pool.map(_score_job, jobs, chunksize=8), start=1):
            rows.extend(part)
            if i % 500 == 0 or i == len(jobs):
                print(f"  done {i}/{len(jobs)}", flush=True)
    return pd.DataFrame(rows)


def _pmpnn_design_pass_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    ok = df[(df["protocol"] == "pmpnn_esmfold") & (df["rmsd_status"] == "ok")].copy()
    if ok.empty:
        return pd.DataFrame()
    ok["pass"] = (ok["motif_ha_rmsd"] < HA_MAX) & (ok["motif_ca_rmsd"] < CA_MAX)
    anypass = ok.groupby(
        ["run_key", "run_label", "run_id", "name", "task", "group_id", "group_size"],
        as_index=False,
    ).agg(n_seq=("pass", "size"), n_seq_pass=("pass", "sum"), pass_any=("pass", "any"))
    anypass["slice"] = [
        _slice_name(int(gid), int(nres))
        for gid, nres in zip(anypass["group_id"], anypass["group_size"])
    ]
    return anypass


def _design_native_pass_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    ok = df[(df["protocol"] == "design_native") & (df["rmsd_status"] == "ok")].copy()
    if ok.empty:
        return pd.DataFrame()
    ok["pass"] = (ok["motif_ha_rmsd"] < HA_MAX) & (ok["motif_ca_rmsd"] < CA_MAX)
    ok["slice"] = [
        _slice_name(int(gid), int(nres))
        for gid, nres in zip(ok["group_id"], ok["group_size"])
    ]
    return ok


def _fold_vs_design_pass_table(analysis_id: str, run_key: str, run_label: str) -> pd.DataFrame:
    path = _v2_dir(analysis_id) / "group_fold_metrics.csv"
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()
    if "protocol" not in df.columns:
        from nomadflow.benchmark.analysis_tables import infer_group_rmsd_protocol

        df["protocol"] = df["path"].map(infer_group_rmsd_protocol)
    df = df[df["task"].astype(str).isin(TABLE_TASKS | EXT2X10_TASKS)].copy()
    df = df[df["protocol"] == "pmpnn_esmfold"].copy()
    if "rmsd_status" in df.columns:
        df = df[df["rmsd_status"].astype(str) == "ok"].copy()
    if df.empty:
        return pd.DataFrame()
    df["pass"] = (df["motif_group_all_atom_rmsd"] < HA_MAX) & (
        df["motif_group_bb_rmsd"] < 1.0
    )
    anypass = df.groupby(["name", "task", "group_id", "num_group_res"], as_index=False).agg(
        n_seq=("pass", "size"),
        n_seq_pass=("pass", "sum"),
        pass_any=("pass", "any"),
    )
    anypass["run_key"] = run_key
    anypass["run_label"] = run_label
    anypass["run_id"] = analysis_id
    anypass["group_size"] = anypass["num_group_res"].astype(int)
    anypass["slice"] = [
        _slice_name(int(gid), int(nres))
        for gid, nres in zip(anypass["group_id"], anypass["group_size"])
    ]
    return anypass


def _slice_summary(df: pd.DataFrame, pass_col: str) -> pd.DataFrame:
    rows = []
    if df.empty or "slice" not in df.columns:
        df = pd.DataFrame()
    for slice_name, gid, nres in SLICE_ORDER:
        part = df[df["slice"] == slice_name] if not df.empty else pd.DataFrame()
        n = int(len(part))
        n_pass = int(part[pass_col].sum()) if n else 0
        rows.append(
            {
                "slice": slice_name,
                "group_id": gid,
                "group_size": nres,
                "n_designs": n,
                "n_pass": n_pass,
                "pass_rate": (n_pass / n) if n else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _existing_structure_keys(path: Path) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    df = pd.read_csv(path, usecols=["name", "label"])
    if df.empty:
        return set()
    return set(zip(df["name"].astype(str), df["label"].astype(str)))


def _drop_scored_jobs(jobs: list[dict[str, Any]], scored: set[tuple[str, str]]) -> list[dict[str, Any]]:
    if not scored:
        return jobs
    kept = [job for job in jobs if (job["name"], job["label"]) not in scored]
    print(f"  skip-existing: {len(jobs) - len(kept)} already scored, {len(kept)} remaining")
    return kept


def _replace_tasks(old_path: Path, new_df: pd.DataFrame, tasks: frozenset[str] | None) -> pd.DataFrame:
    if new_df.empty and old_path.is_file():
        return pd.read_csv(old_path)
    if tasks is None or not old_path.is_file():
        return new_df
    old = pd.read_csv(old_path)
    if old.empty or "task" not in old.columns:
        return new_df
    kept = old[~old["task"].astype(str).isin(tasks)]
    if new_df.empty:
        return kept
    return pd.concat([kept, new_df], ignore_index=True)


def _load_v3_sidecars(kind: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for spec in RUNS.values():
        path = _v3_dir(spec["analysis_id"]) / f"group_{kind}_vs_native.csv"
        if path.is_file():
            parts.append(pd.read_csv(path))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _merge_written(old_path: Path, new_df: pd.DataFrame) -> pd.DataFrame:
    if new_df.empty and old_path.is_file():
        return pd.read_csv(old_path)
    if not old_path.is_file():
        return new_df
    old = pd.read_csv(old_path)
    if old.empty:
        return new_df
    merged = pd.concat([old, new_df], ignore_index=True)
    if {"name", "label", "group_id"}.issubset(merged.columns):
        merged = merged.drop_duplicates(subset=["name", "label", "group_id"], keep="last")
    return merged


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"wrote {path} n={len(df)}")


def _restrict_to_tasks(df: pd.DataFrame, tasks: frozenset[str]) -> pd.DataFrame:
    if df.empty or "task" not in df.columns:
        return df
    return df[df["task"].astype(str).isin(tasks)].copy()


def _build_five_row_table(
    fold_pass: pd.DataFrame,
    design_pass: pd.DataFrame,
    fold_vs_design: pd.DataFrame,
    run_keys: tuple[str, ...],
    *,
    include_ext2x10_seg10: bool = False,
) -> pd.DataFrame:
    rows = []
    for slice_name, gid, nres in SLICE_ORDER:
        tasks = _tasks_for_slice(slice_name, include_ext2x10_seg10=include_ext2x10_seg10)
        row: dict[str, Any] = {
            "slice": slice_name,
            "group_id": gid,
            "group_size": nres,
            "include_ext2x10_seg10": bool(include_ext2x10_seg10 and slice_name == "segment 10"),
        }
        for run_key in run_keys:
            label = RUNS[run_key]["label"]
            fp = _restrict_to_tasks(fold_pass, tasks)
            fp = fp[fp["run_key"] == run_key] if not fp.empty else fp
            fp = fp[fp["slice"] == slice_name] if not fp.empty else fp
            n = int(len(fp))
            n_pass = int(fp["pass_any"].sum()) if n else 0
            n_seq = int(fp["n_seq"].sum()) if n and "n_seq" in fp.columns else 0
            row[f"{label}__fold_vs_native_n"] = n
            row[f"{label}__fold_vs_native_n_seq"] = n_seq
            row[f"{label}__fold_vs_native_pass"] = n_pass
            row[f"{label}__fold_vs_native_rate"] = (n_pass / n) if n else float("nan")

            dp = _restrict_to_tasks(design_pass, tasks)
            dp = dp[dp["run_key"] == run_key] if not dp.empty else dp
            dp = dp[dp["slice"] == slice_name] if not dp.empty else dp
            nd = int(len(dp))
            nd_pass = int(dp["pass"].sum()) if nd else 0
            row[f"{label}__incrustation_n"] = nd
            row[f"{label}__incrustation_pass"] = nd_pass
            row[f"{label}__incrustation_rate"] = (nd_pass / nd) if nd else float("nan")

            vd = _restrict_to_tasks(fold_vs_design, tasks)
            vd = vd[vd["run_key"] == run_key] if not vd.empty else vd
            vd = vd[vd["slice"] == slice_name] if not vd.empty else vd
            nv = int(len(vd))
            nv_pass = int(vd["pass_any"].sum()) if nv else 0
            row[f"{label}__fold_vs_design_n"] = nv
            row[f"{label}__fold_vs_design_pass"] = nv_pass
            row[f"{label}__fold_vs_design_rate"] = (nv_pass / nv) if nv else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _print_main_table(table: pd.DataFrame, *, title: str | None = None) -> None:
    print("\nPMPNN design-level fold vs native (HA<1.5 & CA<1.0, ≥1 of 8 seq)")
    if title:
        print(title)
    print(f"{'slice':<14} {'one-rigid':>22} {'multi-group':>22}")
    for _, row in table.iterrows():
        def _cell(prefix: str) -> str:
            n = row.get(f"{prefix}__fold_vs_native_n", 0)
            n_pass = row.get(f"{prefix}__fold_vs_native_pass", 0)
            rate = row.get(f"{prefix}__fold_vs_native_rate", float("nan"))
            if pd.isna(rate) or n == 0:
                return "  n/a"
            return f"{100*rate:5.1f}%  {int(n_pass)}/{int(n)}"

        print(f"{row['slice']:<14} {_cell('one-rigid'):>22} {_cell('multi-group'):>22}")


def _write_and_print_tables(
    fold_pass: pd.DataFrame,
    design_pass: pd.DataFrame,
    fold_vs_design: pd.DataFrame,
    table_runs: tuple[str, ...],
    analysis_root: Path,
) -> None:
    table = _build_five_row_table(fold_pass, design_pass, fold_vs_design, table_runs)
    _write_csv(table, analysis_root / "_fold_vs_native_groups_table.csv")
    _print_main_table(table, title="8-task 2-group (excluding ext2x10)")

    table_seg10 = _build_five_row_table(
        fold_pass,
        design_pass,
        fold_vs_design,
        table_runs,
        include_ext2x10_seg10=True,
    )
    _write_csv(
        table_seg10,
        analysis_root / "_fold_vs_native_groups_table_seg10_ext2x10.csv",
    )
    _print_main_table(
        table_seg10,
        title="segment 10 = 2-group gid2 + both ext2x10 10-residue arms; cores unchanged",
    )


def _size_by_task_problems(run_key: str, size_by_task: dict[str, dict[int, int]]) -> list[str]:
    """Report unexpected group-id sets and gid1/gid2 sizes independently.

    The id-set check and the CORE_SIZES / SEGMENT_SIZES checks are siblings: a
    task whose group ids are exactly {1, 2} is still validated for core and
    segment lengths.
    """
    problems: list[str] = []
    for task, sizes in size_by_task.items():
        if set(sizes) != {1, 2}:
            problems.append(f"{run_key} {task}: group ids {sorted(sizes)}")
        if sizes.get(1) not in CORE_SIZES:
            problems.append(f"{run_key} {task}: gid1 size {sizes.get(1)} not in {CORE_SIZES}")
        if sizes.get(2) not in SEGMENT_SIZES:
            problems.append(f"{run_key} {task}: gid2 size {sizes.get(2)} not in {SEGMENT_SIZES}")
    return problems


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--run", action="store_true", help="score after a successful probe")
    parser.add_argument(
        "--runs",
        default="one-rigid,multi-group,full-stage2,se3-stage1,se3-stage2",
        help="comma-separated run keys",
    )
    parser.add_argument("--workers", type=int, default=int(os.environ.get("FOLD_VS_NATIVE_WORKERS", "16")))
    parser.add_argument("--no-one-shot", action="store_true")
    parser.add_argument("--no-design", action="store_true", help="skip design-vs-native comparison sidecar")
    parser.add_argument(
        "--table-tasks-only",
        action="store_true",
        help="restrict scoring to the 8 matched 2-group catalytic tasks (default: all tasks)",
    )
    parser.add_argument(
        "--tasks",
        default="",
        help="comma-separated task names to (re)score; replaces those tasks in existing v3 CSVs",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip (name,label) already present in v3 CSVs",
    )
    parser.add_argument("--force", action="store_true", help="score even if probe reports problems")
    parser.add_argument(
        "--rebuild-tables",
        action="store_true",
        help="rebuild both 5-row tables from existing v3 sidecars; do not score",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_keys = tuple(k.strip() for k in args.runs.split(",") if k.strip())
    unknown = [k for k in run_keys if k not in RUNS]
    if unknown:
        print(f"unknown runs: {unknown}", file=sys.stderr)
        return 2
    include_one_shot = not args.no_one_shot
    also_design = not args.no_design
    do_run = bool(args.run) or not args.probe_only
    if args.rebuild_tables:
        analysis_root = REPO / "article_results" / "analysis_runs"
        fold_all = _load_v3_sidecars("fold")
        design_all = _load_v3_sidecars("design")
        fold_pass = _pmpnn_design_pass_table(fold_all) if not fold_all.empty else pd.DataFrame()
        design_pass = (
            _design_native_pass_table(design_all) if not design_all.empty else pd.DataFrame()
        )
        fvd_parts = []
        for run_key, spec in RUNS.items():
            fvd = _fold_vs_design_pass_table(spec["analysis_id"], run_key, spec["label"])
            if not fvd.empty:
                fvd_parts.append(fvd)
        fold_vs_design = pd.concat(fvd_parts, ignore_index=True) if fvd_parts else pd.DataFrame()
        table_runs = tuple(
            k
            for k in ("one-rigid", "multi-group", "full-stage2")
            if _v3_dir(RUNS[k]["analysis_id"]).is_dir()
        )
        _write_and_print_tables(fold_pass, design_pass, fold_vs_design, table_runs, analysis_root)
        return 0
    task_filter: frozenset[str] | None = TABLE_TASKS if args.table_tasks_only else None
    if args.tasks.strip():
        extra = frozenset(k.strip() for k in args.tasks.split(",") if k.strip())
        task_filter = extra if task_filter is None else frozenset(task_filter & extra)

    prepared_by_run: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for run_key in run_keys:
        sampling = _sampling_path(RUNS[run_key]["sampling_id"])
        if not sampling.is_dir():
            msg = f"{run_key}: sampling dir missing ({sampling})"
            if run_key in REQUIRED_RUNS:
                print(f"ERROR {msg}", file=sys.stderr)
                return 1
            print(f"skip {msg}")
            continue
        v2 = _v2_dir(RUNS[run_key]["analysis_id"])
        if v2.is_dir():
            print(f"note: will not touch {v2}")
        prepared = _prepare_run(
            run_key, include_one_shot=include_one_shot, task_filter=task_filter
        )
        _print_prepare_summary(prepared)
        if prepared["n_by_task"] and not prepared["sample_infos"]:
            problems.append(f"{run_key}: tasks present but no usable source_residue_keys")
        if task_filter is not None and prepared["size_by_task"]:
            problems.extend(_size_by_task_problems(run_key, prepared["size_by_task"]))
        probe_problems = _probe_structures(prepared, n_samples=1)
        problems.extend(probe_problems)
        prepared_by_run[run_key] = prepared

    if problems:
        print("\nPROBE PROBLEMS (holey mapping → skip, lists not shortened):")
        for item in problems:
            print(f"  {item}")
        if args.table_tasks_only and not args.force:
            print("abort scoring (pass --force to score anyway, eval_error rows still recorded)")
            return 1
        print("continuing: ok samples will be scored; holey samples stay eval_error")
    else:
        print("\nPROBE OK: native / samples_eval / ESMFold residues match full pair lists")

    if args.probe_only or not do_run:
        return 0 if not problems else 1

    combined_fold: list[pd.DataFrame] = []
    combined_design: list[pd.DataFrame] = []
    fold_pass_parts: list[pd.DataFrame] = []
    design_pass_parts: list[pd.DataFrame] = []
    fvd_parts: list[pd.DataFrame] = []

    for run_key, prepared in prepared_by_run.items():
        spec = prepared["spec"]
        out_dir = _v3_dir(spec["analysis_id"])
        fold_path = out_dir / "group_fold_vs_native.csv"
        design_path = out_dir / "group_design_vs_native.csv"
        fold_jobs = prepared["fold_jobs"]
        design_jobs = prepared["design_jobs"]
        if args.skip_existing:
            fold_jobs = _drop_scored_jobs(fold_jobs, _existing_structure_keys(fold_path))
            design_jobs = _drop_scored_jobs(design_jobs, _existing_structure_keys(design_path))
        if fold_jobs:
            fold_df = _run_jobs(fold_jobs, workers=args.workers)
        else:
            fold_df = pd.DataFrame()
        if prepared["error_rows"]:
            err_df = pd.DataFrame(prepared["error_rows"])
            fold_df = pd.concat([fold_df, err_df], ignore_index=True) if not fold_df.empty else err_df
        if args.skip_existing:
            fold_df = _merge_written(fold_path, fold_df)
        elif task_filter is not None:
            fold_df = _replace_tasks(fold_path, fold_df, task_filter)
        if not fold_df.empty:
            _write_csv(fold_df, fold_path)
            combined_fold.append(fold_df)
            fold_pass_parts.append(_pmpnn_design_pass_table(fold_df))

        if also_design:
            if design_jobs:
                design_df = _run_jobs(design_jobs, workers=args.workers)
            else:
                design_df = pd.DataFrame()
            if args.skip_existing:
                design_df = _merge_written(design_path, design_df)
            elif task_filter is not None:
                design_df = _replace_tasks(design_path, design_df, task_filter)
            if not design_df.empty:
                _write_csv(design_df, design_path)
                combined_design.append(design_df)
                design_pass_parts.append(_design_native_pass_table(design_df))

        fvd = _fold_vs_design_pass_table(spec["analysis_id"], run_key, spec["label"])
        if not fvd.empty:
            fvd_parts.append(fvd)

    analysis_root = REPO / "article_results" / "analysis_runs"
    if task_filter is not None:
        fold_all = _load_v3_sidecars("fold")
        design_all = _load_v3_sidecars("design")
        fold_pass = (
            _pmpnn_design_pass_table(fold_all) if not fold_all.empty else pd.DataFrame()
        )
        design_pass = (
            _design_native_pass_table(design_all) if not design_all.empty else pd.DataFrame()
        )
        fvd_parts = []
        for run_key, spec in RUNS.items():
            fvd = _fold_vs_design_pass_table(spec["analysis_id"], run_key, spec["label"])
            if not fvd.empty:
                fvd_parts.append(fvd)
        fold_vs_design = pd.concat(fvd_parts, ignore_index=True) if fvd_parts else pd.DataFrame()
        table_runs = tuple(
            k for k in ("one-rigid", "multi-group", "full-stage2") if _v3_dir(RUNS[k]["analysis_id"]).is_dir()
        )
    else:
        fold_all = pd.concat(combined_fold, ignore_index=True) if combined_fold else pd.DataFrame()
        design_all = pd.concat(combined_design, ignore_index=True) if combined_design else pd.DataFrame()
        fold_pass = pd.concat(fold_pass_parts, ignore_index=True) if fold_pass_parts else pd.DataFrame()
        design_pass = pd.concat(design_pass_parts, ignore_index=True) if design_pass_parts else pd.DataFrame()
        fold_vs_design = pd.concat(fvd_parts, ignore_index=True) if fvd_parts else pd.DataFrame()
        table_runs = tuple(k for k in ("one-rigid", "multi-group", "full-stage2") if k in prepared_by_run)
    if not fold_all.empty:
        _write_csv(fold_all, analysis_root / "_fold_vs_native_groups.csv")
    _write_and_print_tables(fold_pass, design_pass, fold_vs_design, table_runs, analysis_root)

    if not fold_all.empty:
        pmpnn = fold_all[fold_all["protocol"] == "pmpnn_esmfold"]
        print(
            f"\nseq/designs: PMPNN rows={len(pmpnn)}  "
            f"designs={pmpnn['name'].nunique() if not pmpnn.empty else 0}  "
            f"ok={int((pmpnn['rmsd_status']=='ok').sum()) if not pmpnn.empty else 0}"
        )
    print(
        "fold-vs-native = ESMFold vs crystal (this CSV). "
        "fold-vs-design = existing group_fold_metrics HA/BB vs generated design. "
        "incrustation = design PDB vs crystal, same residue pairs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
