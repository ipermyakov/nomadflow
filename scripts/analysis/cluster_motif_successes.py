"""Legacy cwd-relative wrapper around nomadflow.benchmark.foldseek_clustering."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nomadflow.benchmark.foldseek_clustering import compute_unique_success
from nomadflow.benchmark.success_criteria import filter_one_shot, filter_pmpnn_esmfold


def main() -> None:
    with open("samples_metadata.json") as fp:
        samples_metadata = json.load(fp)

    sc_df = pd.read_csv("esmfold/sc_rmsd.csv")
    folding_df = pd.read_csv("esmfold/folding_rmsd.csv")

    unique_success = compute_unique_success(
        folding_df,
        sc_df,
        samples_metadata,
        cluster_root=Path("foldseek_clusters"),
        esmfold_dir=Path("esmfold"),
    )

    with open("clustered_pz_pass.json", "w") as fp:
        json.dump(unique_success["one_shot"], fp)

    with open("clustered_pmpnn_pass.json", "w") as fp:
        json.dump(unique_success["pmpnn_esmfold"], fp)


if __name__ == "__main__":
    main()
