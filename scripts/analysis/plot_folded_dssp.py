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
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm import tqdm
    import os
    mpl.rcParams.update({
        'font.size': 18,
        # 'figure.figsize': (9, 6)
    })

    if os.path.exists("dssp_per_precise_sample.csv"):
        df = pd.read_csv("dssp_per_precise_sample.csv")
    else:
        results = []
        for pdb in tqdm(glob.glob("precise_samples_folded/*_sc.pdb")):
            results.append(compute_dssp(pdb))

        df = pd.DataFrame(results)
        df.to_csv("dssp_per_precise_sample.csv")
    # df = pd.read_csv("dssp_per_precise_sample.csv")
    df['frac_helix'] = df['helix_percent']
    df['frac_strand'] = df['strand_percent']

    if len(set(df['seq_len'].tolist())) < 10:
        discrete = True
        colors = "colorblind"
    else:
        discrete = False
        colors = None
    # jointplot = sns.jointplot(df, x="frac_helix", y="frac_strand", hue="seq_len", palette=colors, height=9, s=100)
    jointplot = sns.jointplot(df, x="frac_helix", y="frac_strand", hue="seq_len", palette=colors, height=6.6, ratio=6, s=100)
    # plt.suptitle("Secondary structure content of\nsequence-structure consistent samples")

    ax = plt.gca()
    ax.set(xlabel="Fraction Helix", ylabel="Fraction Strand")
    plt.xlim(0,1)
    plt.ylim(0,1)
    # plt.suptitle("Secondary structure content of\nsequence-structure consistent samples")
    # Adjusting the layout to ensure the title is not overlapped
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.9)

    plt.savefig("ss_comp_per_precise_sample.png")
