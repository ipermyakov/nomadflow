from __future__ import annotations

from pathlib import Path

import pandas as pd

from nomadflow.benchmark.analysis_inputs import (
    build_esmfold_design_path_index,
    list_esmfold_structure_paths,
    remap_path_column,
    rewrite_repo_path,
)


def test_build_esmfold_design_path_index_uses_existing_folded_paths(tmp_path):
    folded_path = tmp_path / "esmfold" / "sample_a" / "folded.pdb"
    folded_path.parent.mkdir(parents=True)
    folded_path.write_text("END\n", encoding="utf-8")
    folding_df = pd.DataFrame(
        [
            {
                "name": "sample_a",
                "folded_path": str(folded_path),
            }
        ]
    )
    index = build_esmfold_design_path_index(folding_df)
    assert index == {"sample_a": folded_path}


def test_build_esmfold_design_path_index_skips_missing_files(tmp_path):
    folding_df = pd.DataFrame(
        [
            {
                "name": "sample_a",
                "folded_path": str(tmp_path / "missing.pdb"),
            }
        ]
    )
    assert build_esmfold_design_path_index(folding_df) == {}


def test_list_esmfold_structure_paths_labels_variants(tmp_path):
    folder = tmp_path / "esmfold" / "sample_a"
    folder.mkdir(parents=True)
    one_shot = folder / "sample, model_name=v_48_020.pdb"
    pmpnn = folder / "T=0.1, sample=12, score=0.9.pdb"
    one_shot.write_text("END\n", encoding="utf-8")
    pmpnn.write_text("END\n", encoding="utf-8")

    entries = list_esmfold_structure_paths(tmp_path / "esmfold", "sample_a")
    assert entries == [
        ("one_shot", one_shot),
        ("pmpnn_sample_12", pmpnn),
    ]


def test_rewrite_repo_path_does_not_require_the_file(tmp_path, monkeypatch):
    remote = Path(
        "/mnt/old-server/article_results/sampling_runs/demo/"
        "esmfold/s0/model_name=v_48_020.pdb"
    )
    expected = (
        tmp_path
        / "article_results"
        / "sampling_runs"
        / "demo"
        / "esmfold"
        / "s0"
        / "model_name=v_48_020.pdb"
    )

    def fail_stat(self):
        raise AssertionError(f"unexpected filesystem stat on {self}")

    monkeypatch.setattr(Path, "exists", fail_stat)
    monkeypatch.setattr(Path, "is_file", fail_stat)
    assert rewrite_repo_path(remote, repo_root=tmp_path) == expected


def test_remap_path_column_rewrites_unique_values(tmp_path):
    remote = (
        "/mnt/old-server/article_results/"
        "sampling_runs/demo/esmfold/s0/model_name=v_48_020.pdb"
    )
    series = pd.Series([remote, remote, None, remote])
    out = remap_path_column(series, repo_root=tmp_path)
    expected = str(
        tmp_path
        / "article_results"
        / "sampling_runs"
        / "demo"
        / "esmfold"
        / "s0"
        / "model_name=v_48_020.pdb"
    )
    assert list(out) == [expected, expected, None, expected]
