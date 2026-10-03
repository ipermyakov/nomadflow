import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
mpl.rcParams.update({
    'font.size': 18,
    'figure.figsize': (6, 6)
})

df = pd.read_csv("novelty/aln_PDB_precise.tsv", sep='\t', header=None)
# jointplot = sns.jointplot(df, x="frac_helix", y="frac_strand", hue="seq_len", palette=colors, height=9, s=100)
df = df.loc[df.groupby(0)[2].idxmax()]
jointplot = sns.histplot(df, x=2)
# plt.suptitle("Secondary structure content of\nsequence-structure consistent samples")

ax = plt.gca()
ax.set(xlabel="maxTM")
# plt.suptitle("Secondary structure content of\nsequence-structure consistent samples")
# Adjusting the layout to ensure the title is not overlapped
# plt.tight_layout()
# plt.subplots_adjust(top=0.9)

plt.savefig("novelty_histplot.png")
