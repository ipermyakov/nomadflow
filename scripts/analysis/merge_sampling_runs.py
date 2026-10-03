#!/usr/bin/env python3
"""Merge one or more sampling run directories into a primary run.

Copies samples/*.pdb and metadata/samples_metadata_*.json from extras into
primary, remapping protein_id / batch indices to avoid collisions, then rebuilds
samples_metadata.json.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


_BATCH_RE = re.compile(r"samples_metadata_rank(?P<rank>\d+)_batch(?P<batch>\d+)\.json$")
_SAMPLE_RE = re.compile(
    r"^(?P<prefix>len_\d+_protein_id)(?P<pid>\d+)_(?P<batch>\d+)_(?P<local>\d+)(?P<suffix>.*)$"
)


def _next_batch_index(metadata_dir: Path) -> int:
    mx = -1
    for p in metadata_dir.glob("samples_metadata_rank*_batch*.json"):
        m = _BATCH_RE.match(p.name)
        if m:
            mx = max(mx, int(m.group("batch")))
    return mx + 1


def _remap_sample_name(name: str, new_batch: int, pid_offset: int) -> str:
    m = _SAMPLE_RE.match(name)
    if m is None:
        raise ValueError(f"Unrecognized sample name: {name}")
    new_pid = int(m.group("pid")) + pid_offset
    return f"{m.group('prefix')}{new_pid}_{new_batch}_{m.group('local')}{m.group('suffix')}"


def _copy_tree_files(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)


def merge_run(primary: Path, extra: Path, *, dry_run: bool = False) -> dict:
    primary_samples = primary / "samples"
    primary_meta = primary / "metadata"
    extra_samples = extra / "samples"
    extra_meta = extra / "metadata"
    if not extra_samples.is_dir() or not extra_meta.is_dir():
        raise FileNotFoundError(f"Extra run missing samples/ or metadata/: {extra}")

    primary_samples.mkdir(parents=True, exist_ok=True)
    primary_meta.mkdir(parents=True, exist_ok=True)

    next_batch = _next_batch_index(primary_meta)
    # Keep protein_id unique across merged files by offsetting with existing sample count.
    pid_offset = sum(1 for _ in primary_samples.glob("*.pdb"))
    stats = {"batches": 0, "samples": 0, "next_batch_start": next_batch, "pid_offset": pid_offset}

    batch_files = sorted(extra_meta.glob("samples_metadata_rank*_batch*.json"))
    for bf in batch_files:
        m = _BATCH_RE.match(bf.name)
        if m is None:
            continue
        rank = int(m.group("rank"))
        new_batch = next_batch
        next_batch += 1
        data = json.loads(bf.read_text())
        new_data = {}
        for old_id, sample in data.items():
            new_id = _remap_sample_name(old_id, new_batch, pid_offset)
            sample = dict(sample)
            # Rewrite paths if present.
            for key in ("path", "viz_path"):
                if key in sample and isinstance(sample[key], str):
                    old_name = Path(sample[key]).name
                    # old_name may equal old_id + .pdb
                    stem = Path(old_name).stem
                    new_stem = _remap_sample_name(stem, new_batch, pid_offset)
                    new_path = primary_samples / f"{new_stem}.pdb"
                    sample[key] = str(new_path)
            # Copy pdb from samples/
            src_pdb = extra_samples / f"{Path(old_id).name}.pdb"
            if not src_pdb.is_file():
                # sometimes key already includes no extension and file is key.pdb
                src_pdb = extra_samples / f"{old_id}.pdb"
            if not src_pdb.is_file():
                raise FileNotFoundError(f"Missing sample pdb for {old_id}: tried under {extra_samples}")
            dst_pdb = primary_samples / f"{new_id}.pdb"
            if not dry_run:
                shutil.copy2(src_pdb, dst_pdb)
            new_data[new_id] = sample
            stats["samples"] += 1

        out_bf = primary_meta / f"samples_metadata_rank{rank}_batch{new_batch}.json"
        if not dry_run:
            out_bf.write_text(json.dumps(new_data))
        stats["batches"] += 1

    # Rebuild combined samples_metadata.json
    combined = {}
    for bf in sorted(primary_meta.glob("samples_metadata_rank*_batch*.json")):
        combined.update(json.loads(bf.read_text()))
    if not dry_run:
        (primary / "samples_metadata.json").write_text(json.dumps(combined))
    stats["combined_entries"] = len(combined)
    return stats


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primary", type=Path, required=True)
    p.add_argument("--extra", type=Path, nargs="+", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    primary = args.primary.resolve()
    for extra in args.extra:
        stats = merge_run(primary, extra.resolve(), dry_run=args.dry_run)
        print(f"merged {extra} -> {primary}: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
