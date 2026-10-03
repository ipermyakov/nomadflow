"""Foldseek-based unique success clustering for motif benchmark analysis."""

from __future__ import annotations

import glob
import logging
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from nomadflow.benchmark.analysis_inputs import parent_motif_task_id, remap_repo_path
from nomadflow.benchmark.success_criteria import (
    CRITERIA_VERSION,
    FOLDSEEK_UNIQUE_TM_THRESHOLD,
    filter_one_shot_final,
    filter_pmpnn_final,
)

logger = logging.getLogger(__name__)

PdbResolver = Callable[[str, pd.Series], Path]


def _count_fasta_sequences(fasta_path: Path) -> int:
    count = 0
    with open(fasta_path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                count += 1
    return count


def run_foldseek_easy_cluster(
    pdb_dir: Path,
    work_dir: Path,
    *,
    tm_threshold: float = FOLDSEEK_UNIQUE_TM_THRESHOLD,
    threads: int = 1,
) -> int:
    """Cluster PDBs with foldseek easy-cluster; return number of representative clusters."""
    pdb_dir = Path(pdb_dir)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = list(pdb_dir.glob("*.pdb"))
    if not pdb_files:
        return 0
    if len(pdb_files) == 1:
        return 1

    easy_cluster_args = [
        "foldseek",
        "easy-cluster",
        str(pdb_dir),
        str(work_dir / "res"),
        str(work_dir),
        "--alignment-type",
        "1",
        "--cov-mode",
        "0",
        "--min-seq-id",
        "0",
        "--tmscore-threshold",
        str(tm_threshold),
        "--threads",
        str(threads),
        "--single-step-clustering",
    ]
    process = subprocess.Popen(
        easy_cluster_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate()
    if stdout:
        logger.debug(stdout.decode("utf-8"))
    if stderr:
        logger.debug(stderr.decode("utf-8"))
    if process.returncode != 0:
        raise RuntimeError(
            f"foldseek easy-cluster failed for {pdb_dir} (exit {process.returncode})"
        )

    rep_seq_fasta = work_dir / "res_rep_seq.fasta"
    if not rep_seq_fasta.is_file():
        raise FileNotFoundError(f"Missing foldseek output: {rep_seq_fasta}")
    return _count_fasta_sequences(rep_seq_fasta)


def _copy_passed_pdbs(
    passed_df: pd.DataFrame,
    samples_metadata: dict[str, Any],
    pdb_resolver: PdbResolver,
    output_root: Path,
) -> dict[str, Path]:
    """Wipe leftover PDBs, then copy the current pass set (sorted names) per task."""
    output_root = Path(output_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    task_dirs: dict[str, Path] = {}
    if passed_df.empty:
        return task_dirs

    ordered = passed_df.sort_values("name", kind="mergesort")
    for _, row in ordered.iterrows():
        sample_name = str(row["name"])
        metadata = samples_metadata[sample_name]
        task = parent_motif_task_id(str(metadata["name"]), metadata)
        task_dir = output_root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        task_dirs[task] = task_dir
        dest = task_dir / f"{sample_name}.pdb"
        src = pdb_resolver(sample_name, row)
        shutil.copy(src, dest)
    return task_dirs


def count_unique_successes_per_task(
    passed_df: pd.DataFrame,
    samples_metadata: dict[str, Any],
    pdb_resolver: PdbResolver,
    *,
    cluster_root: Path,
    all_tasks: list[str] | None = None,
    tm_threshold: float = FOLDSEEK_UNIQUE_TM_THRESHOLD,
) -> dict[str, int]:
    """Cluster passing designs per task and return unique cluster counts.

    Unique count is clamped to the number of copied PDBs so leftover files or
    Foldseek over-counting cannot yield unique > success.
    """
    cluster_root = Path(cluster_root)
    cluster_root.mkdir(parents=True, exist_ok=True)
    passed_tasks = set(passed_df["task"].unique()) if not passed_df.empty else set()
    if all_tasks is None:
        all_tasks = sorted(
            {
                parent_motif_task_id(str(metadata["name"]), metadata)
                for metadata in samples_metadata.values()
                if isinstance(metadata, dict) and metadata.get("name") is not None
            }
        )

    task_dirs = _copy_passed_pdbs(
        passed_df,
        samples_metadata,
        pdb_resolver,
        cluster_root / "passed_pdbs",
    )

    outputs: dict[str, int] = {}
    for task in all_tasks:
        if task not in passed_tasks:
            outputs[task] = 0
            continue
        task_dir = task_dirs.get(task, cluster_root / "passed_pdbs" / task)
        n_copied = len(sorted(task_dir.glob("*.pdb")))
        if n_copied <= 1:
            outputs[task] = n_copied
            continue
        out_path = cluster_root / "clusters" / task
        if out_path.exists():
            shutil.rmtree(out_path)
        n_clusters = run_foldseek_easy_cluster(
            task_dir,
            out_path,
            tm_threshold=tm_threshold,
        )
        unique_count = min(int(n_clusters), n_copied)
        if unique_count > n_copied:
            raise AssertionError(
                f"Foldseek unique_count {unique_count} > n_copied {n_copied} for task {task}"
            )
        outputs[task] = unique_count
    return outputs


def one_shot_pdb_resolver(
    samples_metadata: dict[str, Any],
    *,
    repo_root: Path | None = None,
) -> PdbResolver:
    def resolve(sample_name: str, _row: pd.Series) -> Path:
        raw = samples_metadata[sample_name]["path"]
        remapped = remap_repo_path(raw, repo_root=repo_root)
        path = remapped if remapped is not None else Path(raw)
        if not path.is_file():
            raise FileNotFoundError(f"Design PDB not found for {sample_name}: {path}")
        return path

    return resolve


def pmpnn_esmfold_pdb_resolver(
    samples_metadata: dict[str, Any],
    esmfold_dir: Path,
) -> PdbResolver:
    esmfold_dir = Path(esmfold_dir)

    def resolve(sample_name: str, row: pd.Series) -> Path:
        sample_id = int(row["sample"])
        possible_folded_paths = glob.glob(str(esmfold_dir / sample_name / "*.pdb"))
        for path in possible_folded_paths:
            if f"sample={sample_id}" in path:
                return Path(path)
        raise FileNotFoundError(
            f"No ESMFold PDB for {sample_name} sample={sample_id} under {esmfold_dir}"
        )

    return resolve


def _one_design_per_name(passed_df: pd.DataFrame) -> pd.DataFrame:
    """Keep one PMPNN sequence per design so unique matches pmpnn_final_count."""
    if passed_df.empty or "name" not in passed_df.columns:
        return passed_df
    return passed_df.sort_values("name", kind="mergesort").drop_duplicates("name")


def compute_and_unique_success(
    folding_df: pd.DataFrame,
    samples_metadata: dict[str, Any],
    *,
    cluster_root: Path,
    all_tasks: list[str] | None = None,
    tm_threshold: float = FOLDSEEK_UNIQUE_TM_THRESHOLD,
    repo_root: Path | None = None,
    pmpnn_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Cluster AND-success design backbones per task for one-shot and PMPNN."""
    if all_tasks is None:
        task_names: set[str] = set()
        for frame in (folding_df, pmpnn_df):
            if frame is not None and not frame.empty and "task" in frame.columns:
                task_names.update(frame["task"].astype(str).unique().tolist())
        if task_names:
            all_tasks = sorted(task_names)
        else:
            all_tasks = sorted(
                {
                    parent_motif_task_id(str(metadata["name"]), metadata)
                    for metadata in samples_metadata.values()
                    if isinstance(metadata, dict) and metadata.get("name") is not None
                }
            )

    resolver = one_shot_pdb_resolver(samples_metadata, repo_root=repo_root)
    one_shot_counts = count_unique_successes_per_task(
        filter_one_shot_final(folding_df),
        samples_metadata,
        resolver,
        cluster_root=cluster_root / "one_shot",
        all_tasks=all_tasks,
        tm_threshold=tm_threshold,
    )
    pmpnn_counts: dict[str, int] = {}
    if pmpnn_df is not None:
        pmpnn_counts = count_unique_successes_per_task(
            _one_design_per_name(filter_pmpnn_final(pmpnn_df)),
            samples_metadata,
            resolver,
            cluster_root=cluster_root / "pmpnn_esmfold",
            all_tasks=all_tasks,
            tm_threshold=tm_threshold,
        )
    return {
        "criteria_version": CRITERIA_VERSION,
        "foldseek_tm_threshold": tm_threshold,
        "foldseek_input": "and_success_design_backbones",
        "one_shot": one_shot_counts,
        "pmpnn_esmfold": pmpnn_counts,
        "skipped": False,
    }


def compute_unique_success(
    folding_df: pd.DataFrame,
    sc_df: pd.DataFrame,
    samples_metadata: dict[str, Any],
    *,
    cluster_root: Path,
    esmfold_dir: Path,
    tm_threshold: float = FOLDSEEK_UNIQUE_TM_THRESHOLD,
) -> dict[str, Any]:
    """Legacy unique-success helper; publication path uses compute_and_unique_success."""
    del sc_df, esmfold_dir
    return compute_and_unique_success(
        folding_df,
        samples_metadata,
        cluster_root=cluster_root,
        tm_threshold=tm_threshold,
    )
