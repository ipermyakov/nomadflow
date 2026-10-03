#!/usr/bin/env python3
"""Run motif benchmark analysis for an article_results sampling run."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nomadflow.benchmark.analysis_inputs import (
    build_esmfold_design_path_index,
    expected_group_counts,
    infer_num_samples_per_task,
    load_esmfold_tables,
    load_group_rmsd_table,
    load_native_group_table,
    load_samples_metadata,
    remap_dataframe_tasks_to_parent,
    remap_repo_path,
    remap_samples_metadata_paths,
    repo_root_from,
    task_sample_counts,
)
from nomadflow.benchmark.layout_manifest_eval import (
    enrich_samples_metadata_from_layout_manifest,
)
from nomadflow.benchmark.analysis_tables import (
    build_metrics_table,
    build_task_success_table,
)
from nomadflow.benchmark.foldseek_clustering import compute_and_unique_success
from nomadflow.benchmark.layout_pose_metrics import (
    LAYOUT_FRAME,
    compute_pmpnn_layout_pair_metrics,
    compute_run_layout_metrics,
)
from nomadflow.benchmark.multimotif_metrics import (
    aggregate_multimotif_slices,
    compute_run_multimotif_metrics,
)
from nomadflow.benchmark.success_criteria import (
    CRITERIA_VERSION,
    FOLDSEEK_UNIQUE_TM_THRESHOLD,
    GLOBAL_HA_RMSD_MAX,
    MOTIF_BB_RMSD_MAX,
    MOTIF_CA_RMSD_MAX,
    MOTIF_HA_RMSD_MAX,
)

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_repo_root(),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


_PRESERVE_SIDECAR_NAMES = frozenset(
    {"group_fold_vs_native.csv", "group_design_vs_native.csv"}
)


def _assert_analysis_out_writable(analysis_out: Path, overwrite: bool) -> None:
    if analysis_out.exists():
        nonempty = True
        if analysis_out.is_dir():
            nonempty = any(analysis_out.iterdir())
        if nonempty and not overwrite:
            raise FileExistsError(
                f"Analysis output directory is not empty: {analysis_out}. "
                "Pass --overwrite to replace analysis products, or choose a new "
                "--analysis-out (publication writes go to <run_id>__criteria_v3/)."
            )
        if nonempty and overwrite:
            if analysis_out.is_dir():
                for child in analysis_out.iterdir():
                    if child.name in _PRESERVE_SIDECAR_NAMES:
                        continue
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
            else:
                analysis_out.unlink()
    analysis_out.mkdir(parents=True, exist_ok=True)


def run_analysis(
    *,
    sampling_run: Path,
    analysis_out: Path,
    skip_foldseek: bool = False,
    skip_multimotif: bool = False,
    skip_layout: bool = False,
    write_multimotif_alignment_pdbs: bool = False,
    multimotif_design_structure: str = "sample",
    layout_manifest: Path | None = None,
    overwrite: bool = False,
) -> None:
    sampling_run = Path(sampling_run).resolve()
    analysis_out = Path(analysis_out).resolve()
    layout_manifest_path = Path(layout_manifest).resolve() if layout_manifest else None
    _assert_analysis_out_writable(analysis_out, overwrite)

    try:
        repo_root = repo_root_from(sampling_run)
    except FileNotFoundError:
        repo_root = _repo_root()

    samples_metadata = load_samples_metadata(sampling_run)
    # In-memory only — never persist enriched metadata into sampling_runs.
    samples_metadata = enrich_samples_metadata_from_layout_manifest(
        samples_metadata,
        sampling_run=sampling_run,
        layout_manifest_path=layout_manifest_path,
    )
    samples_metadata = remap_samples_metadata_paths(
        samples_metadata, repo_root=repo_root
    )
    folding_df, sc_df = load_esmfold_tables(sampling_run)
    folding_df = remap_dataframe_tasks_to_parent(folding_df, samples_metadata)
    sc_df = remap_dataframe_tasks_to_parent(sc_df, samples_metadata)
    if "folded_path" in folding_df.columns:
        folding_df = folding_df.copy()
        folding_df["folded_path"] = folding_df["folded_path"].map(
            lambda value: str(remap_repo_path(value, repo_root=repo_root) or value)
        )
    num_samples_per_task = infer_num_samples_per_task(samples_metadata)
    task_counts = task_sample_counts(samples_metadata)
    expected_groups = expected_group_counts(samples_metadata)

    fold_vs_design = load_group_rmsd_table(
        sampling_run, samples_metadata=samples_metadata
    )
    if not fold_vs_design.empty:
        fold_vs_design.to_csv(analysis_out / "group_fold_metrics.csv", index=False)
    elif (sampling_run / "esmfold" / "group_rmsd.csv").is_file():
        shutil.copy(
            sampling_run / "esmfold" / "group_rmsd.csv",
            analysis_out / "group_fold_metrics.csv",
        )

    group_rmsd = load_native_group_table(
        analysis_out,
        samples_metadata=samples_metadata,
        repo_root=repo_root,
    )
    if group_rmsd.empty:
        logger.error(
            "Missing native group table (group_fold_vs_native.csv) under %s "
            "or <run>__criteria_v3/; pass_all_groups / pass_final will be false. "
            "Not falling back to fold-vs-design group_rmsd.",
            analysis_out,
        )

    metrics_df = build_metrics_table(
        folding_df,
        sc_df,
        group_rmsd,
        expected_groups=expected_groups,
    )
    metrics_df.to_csv(analysis_out / "metrics.csv", index=False)

    one_shot_metrics = metrics_df[metrics_df["protocol"] == "one_shot"].copy()
    if not one_shot_metrics.empty and "folded_path" not in one_shot_metrics.columns:
        if "folded_path" in folding_df.columns:
            one_shot_metrics = one_shot_metrics.merge(
                folding_df[["name", "folded_path"]],
                on="name",
                how="left",
            )

    task_success_df = build_task_success_table(
        folding_df,
        sc_df,
        num_samples_per_task=num_samples_per_task,
        task_counts=task_counts,
        group_rmsd=group_rmsd,
        expected_groups=expected_groups,
    )
    task_success_df.to_csv(analysis_out / "task_success.csv", index=False)

    multimotif_df = None
    if not skip_multimotif:
        if write_multimotif_alignment_pdbs and multimotif_design_structure != "esmfold":
            multimotif_design_structure = "esmfold"
            logger.info(
                "Using multimotif_design_structure=esmfold for alignment PDB export"
            )
        design_structure_paths = None
        alignment_pdb_dir = None
        if multimotif_design_structure == "esmfold":
            design_structure_paths = build_esmfold_design_path_index(folding_df)
        if write_multimotif_alignment_pdbs:
            alignment_pdb_dir = analysis_out / "multimotif_alignments"
        multimotif_df = compute_run_multimotif_metrics(
            samples_metadata,
            design_structure_paths=design_structure_paths,
            design_structure_source=multimotif_design_structure,  # type: ignore[arg-type]
            esmfold_dir=sampling_run / "esmfold" if alignment_pdb_dir is not None else None,
            alignment_pdb_dir=alignment_pdb_dir,
        )
        multimotif_df.to_csv(analysis_out / "multimotif_metrics.csv", index=False)
        slices_df = aggregate_multimotif_slices(multimotif_df)
        slices_df.to_csv(analysis_out / "multimotif_slices.summary.csv", index=False)

    layout_df = None
    layout_summaries: dict = {}
    if not skip_layout:
        layout_df, layout_summaries = compute_run_layout_metrics(
            samples_metadata,
            folding_df=one_shot_metrics,
            esmfold_dir=sampling_run / "esmfold",
            repo_root=repo_root,
        )
        coverage_rows = layout_summaries.pop("coverage_rows", [])
        layout_df.to_csv(analysis_out / "layout_pair_metrics.csv", index=False)
        if coverage_rows:
            pd.DataFrame(coverage_rows).to_csv(
                analysis_out / "layout_coverage.csv", index=False
            )
        with open(analysis_out / "layout_summaries.json", "w", encoding="utf-8") as handle:
            json.dump(layout_summaries, handle, indent=2, sort_keys=True)
        pmpnn_layout_df = compute_pmpnn_layout_pair_metrics(
            samples_metadata,
            metrics_df[metrics_df["protocol"] == "pmpnn_esmfold"]
            if "protocol" in metrics_df.columns
            else metrics_df.iloc[0:0],
            esmfold_dir=sampling_run / "esmfold",
            repo_root=repo_root,
        )
        pmpnn_layout_df.to_csv(analysis_out / "layout_pmpnn_pair_metrics.csv", index=False)

    pmpnn_metrics = metrics_df[metrics_df["protocol"] == "pmpnn_esmfold"].copy()
    if skip_foldseek:
        unique_success = {
            "criteria_version": CRITERIA_VERSION,
            "foldseek_tm_threshold": None,
            "foldseek_input": "and_success_design_backbones",
            "one_shot": {},
            "pmpnn_esmfold": {},
            "skipped": True,
        }
    else:
        unique_success = compute_and_unique_success(
            one_shot_metrics,
            samples_metadata,
            cluster_root=analysis_out / "_foldseek_tmp",
            all_tasks=sorted(task_success_df["task"].astype(str).tolist())
            if not task_success_df.empty
            else None,
            repo_root=repo_root,
            pmpnn_df=pmpnn_metrics,
        )
        unique_success["skipped"] = False
        if not task_success_df.empty:
            one_shot_unique = unique_success.get("one_shot") or {}
            pmpnn_unique = unique_success.get("pmpnn_esmfold") or {}
            for _, row in task_success_df.iterrows():
                task = str(row["task"])
                unique_count = int(one_shot_unique.get(task, 0))
                success_count = int(row["one_shot_success_count"])
                if unique_count > success_count:
                    raise AssertionError(
                        f"Foldseek unique_count {unique_count} > "
                        f"one_shot_success_count {success_count} for task {task}"
                    )
                pmpnn_unique_count = int(pmpnn_unique.get(task, 0))
                if "pmpnn_final_count" in row.index:
                    pmpnn_success_count = int(row["pmpnn_final_count"])
                else:
                    pmpnn_success_count = int(row["pmpnn_success_count"])
                if pmpnn_unique_count > pmpnn_success_count:
                    raise AssertionError(
                        f"Foldseek unique_count {pmpnn_unique_count} > "
                        f"pmpnn_final_count {pmpnn_success_count} for task {task}"
                    )

    with open(analysis_out / "unique_success_all_groups.json", "w", encoding="utf-8") as handle:
        json.dump(unique_success, handle, indent=2, sort_keys=True)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sampling_run": str(sampling_run),
        "analysis_out": str(analysis_out),
        "layout_manifest": str(layout_manifest_path) if layout_manifest_path else None,
        "git_commit": _git_commit(),
        "criteria_version": CRITERIA_VERSION,
        "thresholds": {
            "motif_ha_rmsd_max": MOTIF_HA_RMSD_MAX,
            "motif_ca_rmsd_max": MOTIF_CA_RMSD_MAX,
            "motif_bb_rmsd_max": MOTIF_BB_RMSD_MAX,
            "global_ha_rmsd_max": GLOBAL_HA_RMSD_MAX,
            "foldseek_tm": FOLDSEEK_UNIQUE_TM_THRESHOLD,
            "comparison": "strict_lt",
        },
        "reference_frame": "fold_vs_native_groups",
        "pooled_rmsds": "diagnostic_only",
        "foldseek_input": "and_success_design_backbones",
        "layout_frame": LAYOUT_FRAME,
        "num_samples_metadata": len(samples_metadata),
        "num_metrics_rows": len(metrics_df),
        "num_task_rows": len(task_success_df),
        "num_multimotif_rows": 0 if multimotif_df is None else len(multimotif_df),
        "num_layout_pair_rows": 0 if layout_df is None else len(layout_df),
        "num_group_rmsd_rows": 0 if group_rmsd is None else len(group_rmsd),
        "skip_foldseek": skip_foldseek,
        "skip_multimotif": skip_multimotif,
        "skip_layout": skip_layout,
        "write_multimotif_alignment_pdbs": write_multimotif_alignment_pdbs,
        "multimotif_design_structure": multimotif_design_structure,
    }
    with open(analysis_out / "analysis_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    logger.info("Wrote analysis artifacts to %s", analysis_out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sampling-run",
        required=True,
        help="Path to article_results/sampling_runs/<run_id>/",
    )
    parser.add_argument(
        "--analysis-out",
        required=True,
        help="Path to article_results/analysis_runs/<run_id>__criteria_v3/",
    )
    parser.add_argument(
        "--layout-manifest",
        default=None,
        help="Stage-1 layout_manifest.jsonl (required for Stage-2 group mapping).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace analysis products in a non-empty --analysis-out. "
            "Never deletes group_fold_vs_native.csv / group_design_vs_native.csv."
        ),
    )
    parser.add_argument(
        "--skip-foldseek",
        action="store_true",
        help="Skip Foldseek unique-success clustering.",
    )
    parser.add_argument(
        "--skip-multimotif",
        action="store_true",
        help="Skip multimotif geometry metrics.",
    )
    parser.add_argument(
        "--skip-layout",
        action="store_true",
        help="Skip relative-layout pose metrics.",
    )
    parser.add_argument(
        "--write-multimotif-alignment-pdbs",
        action="store_true",
        help=(
            "Write overlay PDBs per ESMFold variant under "
            "analysis_out/multimotif_alignments/<sample>/: ESMFold (A), "
            "design aligned on A (E), native motifs on ESMFold sites (B+)."
        ),
    )
    parser.add_argument(
        "--multimotif-design-structure",
        choices=["sample", "esmfold"],
        default="sample",
        help="Structure used for multimotif metrics (default: raw design sample).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    run_analysis(
        sampling_run=Path(args.sampling_run),
        analysis_out=Path(args.analysis_out),
        skip_foldseek=args.skip_foldseek,
        skip_multimotif=args.skip_multimotif,
        skip_layout=args.skip_layout,
        write_multimotif_alignment_pdbs=args.write_multimotif_alignment_pdbs,
        multimotif_design_structure=args.multimotif_design_structure,
        layout_manifest=Path(args.layout_manifest) if args.layout_manifest else None,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
