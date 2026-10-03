import numpy as np
import mdtraj as md

def compute_dssp(file_path):
    # MDtraj
    traj = md.load(file_path)
    # SS calculation
    pdb_ss = md.compute_dssp(traj, simplified=True)
    # DG calculation
    pdb_dg = md.compute_rg(traj)

    metadata = {}
    metadata['seq_len'] = pdb_ss.shape[1]
    metadata['coil_percent'] = np.mean(pdb_ss == 'C')
    metadata['helix_percent'] = np.mean(pdb_ss == 'H')
    metadata['strand_percent'] = np.mean(pdb_ss == 'E')

    # Radius of gyration
    metadata['radius_gyration'] = pdb_dg[0]
    return metadata

if __name__ == '__main__':
    import glob
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm import tqdm

    results = []
    for pdb in tqdm(glob.glob("samples/*.pdb")):
        results.append(compute_dssp(pdb))

    df = pd.DataFrame(results)
    df.to_csv("dssp_per_sample.csv")
    df['frac_helix'] = df['helix_percent']
    df['frac_strand'] = df['strand_percent']

    sns.jointplot(df, x="frac_helix", y="frac_strand", hue="seq_len", palette="colorblind")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.savefig("ss_comp_per_sample.png")

    results = []
    for pdb in tqdm(glob.glob("designable_samples_folded/*_sc.pdb")):
        results.append(compute_dssp(pdb))

    df = pd.DataFrame(results)
    df.to_csv("dssp_per_designable_sample.csv")
    df['frac_helix'] = df['helix_percent']
    df['frac_strand'] = df['strand_percent']

    sns.jointplot(df, x="frac_helix", y="frac_strand", hue="seq_len", palette="colorblind")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.title("Secondary structure content of designable samples")
    plt.savefig("ss_comp_per_designable_sample.png")
