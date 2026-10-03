import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    'font.size': 18,
    'figure.figsize': (9, 6)
})

df = pd.read_csv("esmfold/folding_rmsd.csv")
df["seq_len"] = pd.to_numeric(df["name"].str.split("_", expand=True)[1], errors="coerce").astype(np.int64)

if len(set(df['seq_len'].tolist())) < 10:
    discrete = True
else:
    discrete = False

designable = pd.to_numeric(df["sc_rmsd_all_atom"] < 2).astype(np.float32)
if discrete:
    df['perc_designable'] = designable
    percent_designable = df.groupby('seq_len')['perc_designable'].mean()
else:
    percent_designable = designable.rolling(window=50).mean()
df["fraction_designable"] = percent_designable

ax = plt.gca()



if discrete:
    sns.stripplot(df, x="seq_len", y="sc_rmsd_all_atom", hue="plddt", ax=ax, palette="viridis", legend=False)
else:
    sns.scatterplot(df, x="seq_len", y="sc_rmsd_all_atom", hue="plddt", ax=ax, palette="viridis", legend=False)
# ax.yaxis.label.set_fontweight('heavy')
# ax.xaxis.label.set_fontweight('heavy')
ax.set(xlabel="Sequence Length", ylabel="All-atom RMSD (aaRMSD)")
ax2 = ax.twinx()
if discrete:
    ref_loc = sorted(set(df['seq_len'].tolist()))
    loc = [ref_loc.index(i) for i in df['seq_len']]
    df['plot_loc'] = loc
    sns.lineplot(df, x="plot_loc", y="fraction_designable", color="orange", ax=ax2, errorbar=None, linewidth=3)
else:
    sns.lineplot(df, x="seq_len", y="fraction_designable", color="orange", ax=ax2, errorbar=None)
ax2.set(ylabel="Fraction SSC")
ax2.set_ylim(0, 1)
ax2.yaxis.label.set_color('orange')

# add a colorbar instead of the sampled legend
cmap = sns.color_palette('viridis', as_cmap=True)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize())
cbar = plt.colorbar(sm, ax=ax2)
cbar.set_label('pLDDT')
pos = cbar.ax.get_position()
pos.x0 += 0.10
pos.x1 += 0.10
cbar.ax.set_position(pos)

ax.axhline(y=2.0, color='red', linestyle='--')
# plt.suptitle("Sequence-structure consistency over length")
plt.savefig("folding_over_len.png")
