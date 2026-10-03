"""Tests for fold-vs-native group-id / size probe checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "article_results"
    / "analysis_runs"
    / "_fold_vs_native_groups.py"
)


@pytest.fixture(scope="module")
def fvn():
    spec = importlib.util.spec_from_file_location("fold_vs_native_groups", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ids_1_2_still_flag_gid1_outside_core_sizes(fvn):
    """Regression: size checks must not be nested under set(sizes) != {1, 2}."""
    problems = fvn._size_by_task_problems("one-rigid", {"1QJG_ext10": {1: 5, 2: 10}})
    assert problems == [
        f"one-rigid 1QJG_ext10: gid1 size 5 not in {fvn.CORE_SIZES}",
    ]


def test_ids_1_2_still_flag_gid2_outside_segment_sizes(fvn):
    problems = fvn._size_by_task_problems("one-rigid", {"1QJG_ext10": {1: 3, 2: 8}})
    assert problems == [
        f"one-rigid 1QJG_ext10: gid2 size 8 not in {fvn.SEGMENT_SIZES}",
    ]


def test_ids_1_2_ok_sizes_are_silent(fvn):
    assert fvn._size_by_task_problems("one-rigid", {"1QJG_ext10": {1: 3, 2: 10}}) == []
    assert fvn._size_by_task_problems("one-rigid", {"5AOU_ext20": {1: 4, 2: 20}}) == []


def test_unexpected_ids_reported_and_size_checks_still_run(fvn):
    problems = fvn._size_by_task_problems(
        "multi-group",
        {"7K4V_ext10": {1: 5, 2: 10, 3: 10}},
    )
    assert f"multi-group 7K4V_ext10: group ids {[1, 2, 3]}" in problems
    assert f"multi-group 7K4V_ext10: gid1 size 5 not in {fvn.CORE_SIZES}" in problems
    assert all("gid2 size" not in item for item in problems)
