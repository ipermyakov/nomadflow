#!/usr/bin/env python3
"""Backfill motif_groups / source_residue_keys into a baseline sampling run.

Use after vanilla proteinzen-main sampling (or to annotate older runs that
predate write-time annotation in main). New runs that already include
motif_groups in the tasks YAML are annotated automatically by PDBWriter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sampling-run",
        required=True,
        type=Path,
        help="Sampling run directory containing samples_metadata.json",
    )
    parser.add_argument(
        "--tasks-yaml",
        type=Path,
        default=None,
        help="Tasks YAML with motif_groups (default: <sampling-run>/config.yaml)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Optional repo root for resolving relative pdb paths",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without writing samples_metadata.json",
    )
    args = parser.parse_args(argv)

    # Import after path setup so the script works when run from repo root.
    from nomadflow.benchmark.motif_group_annotation import (
        annotate_samples_metadata_from_tasks_yaml,
        load_tasks_from_yaml,
        tasks_have_motif_groups,
    )

    sampling_run = args.sampling_run.resolve()
    metadata_path = sampling_run / "samples_metadata.json"
    if not metadata_path.is_file():
        print(f"Missing samples_metadata.json under {sampling_run}", file=sys.stderr)
        return 1

    tasks_yaml = args.tasks_yaml
    if tasks_yaml is None:
        tasks_yaml = sampling_run / "config.yaml"
    tasks_yaml = tasks_yaml.resolve()
    if not tasks_yaml.is_file():
        print(f"Missing tasks YAML: {tasks_yaml}", file=sys.stderr)
        return 1

    tasks = load_tasks_from_yaml(tasks_yaml)
    if not tasks_have_motif_groups(tasks):
        print(
            f"No motif_groups found in {tasks_yaml}; nothing to annotate.",
            file=sys.stderr,
        )
        return 1

    with metadata_path.open("r", encoding="utf-8") as handle:
        samples_metadata = json.load(handle)
    if not isinstance(samples_metadata, dict):
        print("samples_metadata.json must be a JSON object", file=sys.stderr)
        return 1

    repo_root = args.repo_root.resolve() if args.repo_root is not None else None
    annotated = annotate_samples_metadata_from_tasks_yaml(
        samples_metadata,
        tasks_yaml,
        repo_root=repo_root,
        skip_missing_groups=True,
    )

    n_with_groups = sum(
        1
        for sample in annotated.values()
        if isinstance(sample.get("source_residue_keys"), list)
        and any(entry.get("group_id") is not None for entry in sample["source_residue_keys"])
    )
    print(
        f"Annotated {n_with_groups}/{len(annotated)} samples "
        f"from {tasks_yaml.name} into {metadata_path}"
    )
    if args.dry_run:
        return 0

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(annotated, handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
