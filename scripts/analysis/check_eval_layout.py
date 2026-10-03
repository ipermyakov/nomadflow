#!/usr/bin/env python3
"""Check eval PDB / metadata chain-ID contract for a sampling run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nomadflow.benchmark.eval_layout import (
    check_eval_layout,
    pmpnn_seqs_is_complete,
    samples_eval_is_complete,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate samples_eval / metadata / PMPNN chain-ID layout."
    )
    parser.add_argument("--sampling-run", type=Path, required=True)
    parser.add_argument(
        "--seqs-complete",
        action="store_true",
        help="Exit 0 if seqs/*.fa matches metadata; 1 otherwise (no layout check).",
    )
    parser.add_argument(
        "--samples-eval-complete",
        action="store_true",
        help="Exit 0 if samples_eval PDB names match metadata; 1 otherwise.",
    )
    args = parser.parse_args(argv)
    sampling_run = args.sampling_run.resolve()

    if args.seqs_complete:
        ok = pmpnn_seqs_is_complete(sampling_run)
        print(f"seqs complete: {ok} ({sampling_run})")
        return 0 if ok else 1
    if args.samples_eval_complete:
        ok = samples_eval_is_complete(sampling_run)
        print(f"samples_eval complete: {ok} ({sampling_run})")
        return 0 if ok else 1

    errors = check_eval_layout(sampling_run)
    if not errors:
        print(f"eval layout OK: {sampling_run}")
        return 0
    print(f"eval layout FAILED: {sampling_run}", file=sys.stderr)
    for error in errors:
        print(f"  {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
