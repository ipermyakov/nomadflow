""" Compute the bbRMSD between structures """
import argparse
import os
import glob
import json

import pandas as pd


if __name__ == '__main__':
    sc_df = pd.read_csv("esmfold/sc_rmsd.csv")
    folding_df = pd.read_csv("esmfold/folding_rmsd.csv")

    with open("exclude_samples.json") as fp:
        exclude_samples = json.load(fp)

    sc_df['keep'] = ~sc_df['name'].isin(exclude_samples)
    folding_df['keep'] = ~folding_df['name'].isin(exclude_samples)

    pmpnn_pass = sc_df[
        (sc_df['motif_all_atom_rmsd'] < 1.5)
        & (sc_df['motif_bb_rmsd'] < 1)
        & (sc_df['global_bb_rmsd'] < 2)
        & sc_df['keep']
    ]
    pmpnn_pass = pmpnn_pass.groupby("name").sample(1)
    pmpnn_task_pass = pmpnn_pass.groupby("task")
    pz_pass = folding_df[
        (folding_df['motif_all_atom_rmsd'] < 1.5)
        & (folding_df['motif_bb_rmsd'] < 1)
        & (folding_df['global_all_atom_rmsd'] < 2)
        & folding_df['keep']
    ]
    pz_task_pass = pz_pass.groupby("task")

    with open("num_designable_corrected.txt", 'w') as fp:
        fp.write(f"Num designable: {len(pmpnn_pass)}\n")
        fp.write(f"Num tasks passed: {len(pmpnn_task_pass)}\n")
        fp.write(f"Num consistent: {len(pz_pass)}\n")
        fp.write(f"Num tasks consistent: {len(pz_task_pass)}\n")

