import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from Bio import SeqIO

mpl.rcParams.update({
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
})

y_true = []
y_pred = []

designable_samples = [path.split("/")[-1].split(".")[0] for path in glob.glob("designable_samples/*.pdb")]

sc_rmsd_df = pd.read_csv("esmfold/sc_rmsd.csv")

for fasta_file in tqdm.tqdm(glob.glob("seqs/*")):
    name = fasta_file.split("/")[-1].split(".")[0]
    if name in designable_samples:
        subdf = sc_rmsd_df[sc_rmsd_df.name == name]
        seqs = [str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")]
        design_seq = seqs[0]
        pmpnn_seqs = seqs[1:]
        for idx, seq in enumerate(pmpnn_seqs):
            sc_rmsd = subdf[subdf['sample'] == idx+1].global_bb_rmsd.tolist()[0]
            motif_rmsd = subdf[subdf['sample'] == idx+1].motif_all_atom_rmsd.tolist()[0]
            if sc_rmsd < 2 and motif_rmsd < 1.5:
                y_true.append(np.array([c for c in seq]))
                y_pred.append(np.array([c for c in design_seq]))

y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

fig = plt.figure(figsize=(10, 10))
plt.title("ProteinZen vs ProteinMPNN sequences\n")
ax = fig.gca()
order = [c for c in "DEKRHQNSTPGAVILMCFWY"]
cmd = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize="true", ax=fig.gca(), labels=order, values_format=".2f", colorbar=False)

# Adding custom colorbar
cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
plt.colorbar(cmd.im_, cax=cax)

# cmd.plot(ax=fig.gca(), values_format=":.2f")
plt.savefig("confusion_matrix.png")