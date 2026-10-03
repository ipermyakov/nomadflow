#!/usr/bin/env python3
"""Add explicit intra-rigid CONECT records to a ProteinZen PDB trajectory."""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if "nomadflow._version" not in sys.modules:
    version_stub = types.ModuleType("nomadflow._version")
    version_stub.__version__ = "0.0"
    sys.modules["nomadflow._version"] = version_stub

from nomadflow.runtime.intra_rigid_conect import default_output_path, process_pdb_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a copy of a PDB file with explicit CONECT records that keep "
            "only bonds between atoms belonging to the same ProteinZen rigid."
        )
    )
    parser.add_argument("input_pdb", help="Input PDB or multi-model trajectory PDB")
    parser.add_argument(
        "-o",
        "--output",
        help="Output path. Defaults to <input>_intra_rigid_conect.pdb",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    input_path = Path(args.input_pdb)
    output_path = Path(args.output) if args.output else default_output_path(input_path)
    output_path, total_conects, model_count = process_pdb_file(input_path, output_path)

    print(
        f"Wrote {output_path} with {total_conects} CONECT records "
        f"across {model_count} model(s)."
    )
    print("For PyMOL, load with: set connect_mode, 1")


if __name__ == "__main__":
    main()
