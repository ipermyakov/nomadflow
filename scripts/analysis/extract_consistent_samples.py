import argparse
import os
import shutil
import glob

import pandas as pd
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Compute the ")
    parser.add_argument("--csv")
    parser.add_argument("--folded_folders", help="emfold folded structures")
    parser.add_argument("--samples", help="gen model samples")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    consistent_samples = df[df.sc_rmsd < 2]
    not_consistent_samples = df[df.sc_rmsd >= 2]

    out_dir1 = os.path.join(args.samples, "../consistent_samples")
    out_dir2 = os.path.join(args.samples, "../consistent_samples_folded")
    out_dir3 = os.path.join(args.samples, "../not_consistent_samples")
    out_dir4 = os.path.join(args.samples, "../not_consistent_samples_folded")
    if not os.path.isdir(out_dir1):
        os.mkdir(out_dir1)
    if not os.path.isdir(out_dir2):
        os.mkdir(out_dir2)
    if not os.path.isdir(out_dir3):
        os.mkdir(out_dir3)
    if not os.path.isdir(out_dir4):
        os.mkdir(out_dir4)

    for _, row in tqdm.tqdm(list(consistent_samples.iterrows())):
        sample_path = os.path.join(
            args.samples,
            row['name'] + ".pdb"
        )
        folded_path = os.path.join(
            args.folded_folders,
            row['name']
        )
        current_folded = None
        folded_paths = glob.glob(os.path.join(folded_path, "*"))
        for path in folded_paths:
            if row['name'] in path.split("/")[-1]:
                current_folded = path
                break

        shutil.copy(sample_path, out_dir1)
        shutil.copy(
            current_folded,
            os.path.join(out_dir2, row['name'] + "_sc.pdb")
        )

    for _, row in tqdm.tqdm(list(not_consistent_samples.iterrows())):
        sample_path = os.path.join(
            args.samples,
            row['name'] + ".pdb"
        )
        folded_path = os.path.join(
            args.folded_folders,
            row['name']
        )
        current_folded = None
        folded_paths = glob.glob(os.path.join(folded_path, "*"))
        for path in folded_paths:
            if row['name'] in path.split("/")[-1]:
                current_folded = path
                break

        shutil.copy(sample_path, out_dir3)
        shutil.copy(
            current_folded,
            os.path.join(out_dir4, row['name'] + "_sc.pdb")
        )
