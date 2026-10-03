import json
import glob
import os
import shutil
import tqdm
import pandas as pd

import subprocess
from biotite.sequence.io import fasta

# from jason yim multiflow
def run_easy_cluster(designable_dir, output_dir):
    # designable_dir should be a directory with individual PDB files in it that we want to cluster
    # output_dir is where we are going to save the easy cluster output files

    # Returns the number of clusters

    easy_cluster_args = [
        'foldseek',
        'easy-cluster',
        designable_dir,
        os.path.join(output_dir, 'res'),
        output_dir,
        '--alignment-type',
        '1',
        '--cov-mode',
        '0',
        '--min-seq-id',
        '0',
        '--tmscore-threshold',
        '0.5',
        '--threads',
        '1',
        # these three are mainly for clustering small sets which can error out with the default settings
        '--single-step-clustering',
        # '-s',
        # '6'
    ]
    process = subprocess.Popen(
        easy_cluster_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    print(stdout.decode("utf-8"), stderr.decode("utf-8"))
    del stdout # We don't actually need the stdout, we will read the number of clusters from the output files
    rep_seq_fasta = fasta.FastaFile.read(os.path.join(output_dir, 'res_rep_seq.fasta'))
    return len(rep_seq_fasta)


with open("samples_metadata.json") as fp:
    samples_metadata = json.load(fp)

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
pz_pass = folding_df[
    (folding_df['motif_all_atom_rmsd'] < 2)
    & (folding_df['motif_ca_rmsd'] < 1)
    # & (folding_df['motif_bb_rmsd'] < 1)
    & (folding_df['global_all_atom_rmsd'] < 2)
    & folding_df['keep']
]

# print(len(pmpnn_pass), len(pz_pass))
# exit()

os.makedirs("lp_pz_pass_by_task", exist_ok=True)
os.makedirs("lp_pz_pass_by_task/clusters", exist_ok=True)
os.makedirs("lp_pmpnn_pass_by_task", exist_ok=True)
os.makedirs("lp_pmpnn_pass_by_task/clusters", exist_ok=True)

for _, row in tqdm.tqdm(pz_pass.iterrows()):
    entry = row['name']
    metadata = samples_metadata[entry]
    copy_out_folder = os.path.join("lp_pz_pass_by_task", metadata['name'])
    if not os.path.exists(copy_out_folder):
        os.makedirs(copy_out_folder, exist_ok=True)
    copy_out_path = os.path.join("lp_pz_pass_by_task", metadata['name'], entry + ".pdb")
    if not os.path.exists(copy_out_path):
        shutil.copy(metadata["path"], copy_out_path)

for _, row in tqdm.tqdm(pmpnn_pass.iterrows()):
    entry = row['name']
    metadata = samples_metadata[entry]

    possible_folded_paths = glob.glob(
        os.path.join("esmfold", entry, "*.pdb")
    )
    folded_path = None
    sample = row['sample']
    for p in possible_folded_paths:
        if f"sample={sample}" in p:
            folded_path = p
            break
    assert folded_path is not None
    copy_out_folder = os.path.join("lp_pmpnn_pass_by_task", metadata['name'])
    if not os.path.exists(copy_out_folder):
        os.makedirs(copy_out_folder, exist_ok=True)
    copy_out_path = os.path.join("lp_pmpnn_pass_by_task", metadata['name'], entry + ".pdb")
    if not os.path.exists(copy_out_path):
        shutil.copy(folded_path, copy_out_path)

pmpnn_outputs = {}
pz_outputs = {}
pmpnn_passed_tasks = pmpnn_pass['task'].unique()
pz_passed_tasks = pz_pass['task'].unique()

for task in tqdm.tqdm(sc_df['task'].unique()):
    print(task)
    if task in pz_passed_tasks:
        path = f"lp_pz_pass_by_task/{task}"
        if len(glob.glob(f"{path}/*.pdb")) == 1:
            num_pz_clusters = 1
        else:
            out_path = f"lp_pz_pass_by_task/clusters/{task}"
            os.makedirs(out_path, exist_ok=True)
            num_pz_clusters = run_easy_cluster(path, out_path)
    else:
        num_pz_clusters = 0
    pz_outputs[task] = num_pz_clusters

    if task in pmpnn_passed_tasks:
        path = f"lp_pmpnn_pass_by_task/{task}"
        if len(glob.glob(f"{path}/*.pdb")) == 1:
            num_pmpnn_clusters = 1
        else:
            out_path = f"lp_pmpnn_pass_by_task/clusters/{task}"
            os.makedirs(out_path, exist_ok=True)
            num_pmpnn_clusters = run_easy_cluster(path, out_path)
    else:
        num_pmpnn_clusters = 0
    pmpnn_outputs[task] = num_pmpnn_clusters

with open("clustered_lp_pz_pass.json", 'w') as fp:
    json.dump(pz_outputs, fp)

with open("clustered_lp_pmpnn_pass.json", 'w') as fp:
    json.dump(pmpnn_outputs, fp)

