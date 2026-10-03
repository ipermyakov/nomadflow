from __future__ import annotations

import argparse
import json
from pathlib import Path

from nomadflow.benchmark.pmpnn_mapping import rebuild_pmpnn_from_eval_metadata


def _load_parsed_chain_ids(parsed_jsonl: Path) -> dict[str, set[str]]:
    chain_ids_by_sample: dict[str, set[str]] = {}
    with open(parsed_jsonl, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            sample_name = entry["name"]
            chain_ids_by_sample[sample_name] = {
                key[len("seq_chain_") :]
                for key in entry
                if key.startswith("seq_chain_")
            }
    return chain_ids_by_sample


def _load_fixed_positions(fixed_positions_json: Path) -> dict[str, dict[str, list[int]]]:
    if not fixed_positions_json.exists():
        return {}
    with open(fixed_positions_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        str(sample_name): {
            str(chain_id): list(positions)
            for chain_id, positions in fixed_positions.items()
        }
        for sample_name, fixed_positions in payload.items()
    }


def _load_samples_metadata(samples_metadata_path: Path) -> dict[str, dict]:
    with open(samples_metadata_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict in {samples_metadata_path}")
    return payload


def ensure_pmpnn_fixed_positions(
    parsed_jsonl: Path,
    fixed_positions_json: Path,
    *,
    samples_metadata_path: Path | None = None,
) -> None:
    """Make ProteinMPNN fixed-position JSON cover every parsed PDB chain.

    ProteinMPNN indexes ``fixed_position_dict[sample_name][chain_id]`` for every
    chain emitted by ``parse_multiple_chains.py``. Chains with no fixed residues
    still need an explicit empty list.

    When ``samples_metadata.json`` is provided, always rebuild motif indices from
    eval metadata (empty / ``{"A": []}`` dicts are not a valid skip).
    """
    chain_ids_by_sample = _load_parsed_chain_ids(parsed_jsonl)
    fixed_positions = _load_fixed_positions(fixed_positions_json)

    if samples_metadata_path is not None and samples_metadata_path.is_file():
        samples_metadata = _load_samples_metadata(samples_metadata_path)
        fixed_positions = rebuild_pmpnn_from_eval_metadata(samples_metadata)

    for sample_name, chain_ids in chain_ids_by_sample.items():
        sample_fixed_positions = fixed_positions.setdefault(sample_name, {})
        for chain_id in chain_ids:
            sample_fixed_positions.setdefault(chain_id, [])

    with open(fixed_positions_json, "w", encoding="utf-8") as handle:
        json.dump(fixed_positions, handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed-jsonl", type=Path, required=True)
    parser.add_argument("--fixed-positions-json", type=Path, required=True)
    parser.add_argument(
        "--samples-metadata",
        type=Path,
        default=None,
        help="Optional samples_metadata.json used to rebuild eval fixed positions.",
    )
    args = parser.parse_args()

    ensure_pmpnn_fixed_positions(
        args.parsed_jsonl,
        args.fixed_positions_json,
        samples_metadata_path=args.samples_metadata,
    )


if __name__ == "__main__":
    main()
