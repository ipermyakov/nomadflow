#!/usr/bin/env python3
"""
Stream `id` fields from a large Boltz-style manifest.json (AFDB512 metadata).

Usage:
  python export_manifest_ids.py --manifest ../../data/afdb512_clusters_metadata/afdb_metadata/manifest.json --out ../../data/afdb512_ids.txt

Compare downloaded mmCIF stems to the manifest (paths are stems lowercased, like rcsb.py):
  python export_manifest_ids.py --manifest .../manifest.json --compare-dir /path/to/cifs

Requires: pip install ijson (see requirements.txt)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ijson

from cif_manifest_stem import manifest_stem_from_path


def iter_manifest_ids(manifest_path: Path):
    with manifest_path.open("rb") as fp:
        for item in ijson.items(fp, "item"):
            yield item["id"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to manifest.json (afdb_metadata or framediff_metadata).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write one id per line (default: stdout).",
    )
    parser.add_argument(
        "--compare-dir",
        type=Path,
        default=None,
        help="Directory with downloaded *.cif / *.cif.gz; report missing vs manifest and extra files.",
    )
    parser.add_argument(
        "--write-missing",
        type=Path,
        default=None,
        help="With --compare-dir: write all missing ids (one per line) to this path.",
    )
    parser.add_argument(
        "--write-extra",
        type=Path,
        default=None,
        help="With --compare-dir: write all extra ids (one per line) to this path.",
    )
    args = parser.parse_args()

    if (args.write_missing is not None or args.write_extra is not None) and args.compare_dir is None:
        parser.error("--write-missing and --write-extra require --compare-dir")

    if not args.manifest.is_file():
        print(f"error: manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    if args.compare_dir is not None:
        if not args.compare_dir.is_dir():
            print(f"error: not a directory: {args.compare_dir}", file=sys.stderr)
            return 1
        expected = {i.lower() for i in iter_manifest_ids(args.manifest)}
        have = set()
        for path in args.compare_dir.rglob("*"):
            if not path.is_file():
                continue
            stem = manifest_stem_from_path(path)
            if stem is None:
                continue
            have.add(stem)
        missing = sorted(expected - have)
        extra = sorted(have - expected)
        print(f"manifest ids: {len(expected)}")
        print(f"local cif stems: {len(have)}")
        print(f"missing (in manifest, no matching file): {len(missing)}")
        print(f"extra (file present, not in manifest): {len(extra)}")
        if missing and len(missing) <= 50:
            print("missing:", *missing, sep="\n  ")
        elif missing:
            print("first 20 missing:", *missing[:20], sep="\n  ")
        if extra and len(extra) <= 50:
            print("extra:", *extra, sep="\n  ")
        elif extra:
            print("first 20 extra:", *extra[:20], sep="\n  ")
        if args.write_missing is not None:
            args.write_missing.parent.mkdir(parents=True, exist_ok=True)
            with args.write_missing.open("w", encoding="utf-8") as fp:
                for sid in missing:
                    fp.write(f"{sid}\n")
            print(f"wrote {len(missing)} missing ids to {args.write_missing}", file=sys.stderr)
        if args.write_extra is not None:
            args.write_extra.parent.mkdir(parents=True, exist_ok=True)
            with args.write_extra.open("w", encoding="utf-8") as fp:
                for sid in extra:
                    fp.write(f"{sid}\n")
            print(f"wrote {len(extra)} extra ids to {args.write_extra}", file=sys.stderr)
        return 0

    out_fp = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    try:
        n = 0
        for sid in iter_manifest_ids(args.manifest):
            out_fp.write(f"{sid}\n")
            n += 1
        if args.out:
            print(f"wrote {n} ids to {args.out}", file=sys.stderr)
    finally:
        if args.out:
            out_fp.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
