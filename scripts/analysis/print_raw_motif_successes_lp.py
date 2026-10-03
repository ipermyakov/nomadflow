import json
import os
import pandas as pd


sc_df = pd.read_csv("esmfold/sc_rmsd.csv")
folding_df = pd.read_csv("esmfold/folding_rmsd.csv")

if os.path.exists("exclude_samples.json"):
    with open("exclude_samples.json") as fp:
        exclude_samples = json.load(fp)
else:
    exclude_samples = []

sc_df['keep'] = ~sc_df['name'].isin(exclude_samples)
folding_df['keep'] = ~folding_df['name'].isin(exclude_samples)

pmpnn_pass = sc_df[
    (sc_df['motif_all_atom_rmsd'] < 2)
    & (sc_df['motif_ca_rmsd'] < 1)
    # & (sc_df['motif_bb_rmsd'] < 1)
    & (sc_df['global_bb_rmsd'] < 2)
    & sc_df['keep']
]
pmpnn_pass = pmpnn_pass.groupby("name").sample(1)
print(sc_df['task'].value_counts().sort_index())
print(pmpnn_pass['task'].value_counts().sort_index())

pz_pass = folding_df[
    (folding_df['motif_all_atom_rmsd'] < 2)
    & (folding_df['motif_ca_rmsd'] < 1)
    # & (folding_df['motif_bb_rmsd'] < 1)
    & (folding_df['global_all_atom_rmsd'] < 2)
    & folding_df['keep']
]
print(folding_df['task'].value_counts().sort_index())
print(pz_pass['task'].value_counts().sort_index())
