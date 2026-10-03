"""ESMFold FASTA sort key: unprefixed and shard-prefixed names."""

from __future__ import annotations

import ast
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analysis"
    / "batched_esmfold_inference.py"
)


def _load_fasta_length_sort_key():
    """Load the helper without importing torch/esm."""
    import os

    tree = ast.parse(_SCRIPT.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "fasta_length_sort_key":
            code = compile(
                ast.Module(body=[node], type_ignores=[]),
                str(_SCRIPT),
                "exec",
            )
            namespace = {"os": os}
            exec(code, namespace)
            return namespace["fasta_length_sort_key"]
    raise AssertionError(f"fasta_length_sort_key not found in {_SCRIPT}")


def test_unprefixed_slice_names_sort_by_length():
    key = _load_fasta_length_sort_key()
    names = [
        "seqs/len_146_protein_id0_66_2.fa",
        "seqs/len_76_protein_id0_0_0.fa",
        "seqs/len_100_protein_id0_18_5.fa",
    ]
    assert [Path(p).name for p in sorted(names, key=key)] == [
        "len_76_protein_id0_0_0.fa",
        "len_100_protein_id0_18_5.fa",
        "len_146_protein_id0_66_2.fa",
    ]
    assert key("len_76_protein_id0_0_0.fa") == 76.0


def test_shard_prefixed_names_parse_len_token_not_shard_id():
    key = _load_fasta_length_sort_key()
    assert key("s3_7k4v__len_120_protein_id0_0_0.fa") == 120.0
    assert key("s1_short__len_84_protein_id0_0_0.fa") == 84.0
    assert key("s0_1qjg__len_100_protein_id0_25_0.fa") == 100.0
    names = [
        "seqs/s3_7k4v__len_120_protein_id0_0_0.fa",
        "seqs/s1_short__len_84_protein_id0_0_0.fa",
        "seqs/s0_1qjg__len_100_protein_id0_25_0.fa",
    ]
    assert [Path(p).name for p in sorted(names, key=key)] == [
        "s1_short__len_84_protein_id0_0_0.fa",
        "s0_1qjg__len_100_protein_id0_25_0.fa",
        "s3_7k4v__len_120_protein_id0_0_0.fa",
    ]


def test_missing_len_token_raises():
    key = _load_fasta_length_sort_key()
    try:
        key("not_a_pmpnn_name.fa")
    except ValueError as exc:
        assert "not_a_pmpnn_name.fa" in str(exc)
    else:
        raise AssertionError("expected ValueError")
