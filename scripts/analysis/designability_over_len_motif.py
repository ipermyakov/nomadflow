import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import tqdm

mpl.rcParams.update({
    'font.size': 18,
    'figure.figsize': (9, 6)
})

df = pd.read_csv("esmfold/sc_rmsd.csv")
df["seq_len"] = pd.to_numeric(df["name"].str.split("_", expand=True)[1], errors="coerce").astype(np.int64)

if len(set(df['seq_len'].tolist())) < 10:
    discrete = True
else:
    discrete = False

designable = pd.to_numeric((df["motif_all_atom_rmsd"] < 1) & (df["global_bb_rmsd"] < 2)).astype(np.float32)
if discrete:
    df['perc_designable'] = designable
    percent_designable = df.groupby('seq_len')['perc_designable'].mean()
else:
    percent_designable = designable.rolling(window=50).mean()
df["fraction_designable"] = percent_designable

ax = plt.gca()

if discrete:
    sns.stripplot(df, x="seq_len", y="motif_all_atom_rmsd", hue="plddt", ax=ax, palette="viridis", legend=False)
else:
    sns.scatterplot(df, x="seq_len", y="motif_all_atom_rmsd", hue="plddt", ax=ax, palette="viridis", legend=False)
ax.set(xlabel="Sequence Length", ylabel="Motif all-atom RMSD")
ax2 = ax.twinx()
if discrete:
    ref_loc = sorted(set(df['seq_len'].tolist()))
    loc = [ref_loc.index(i) for i in df['seq_len']]
    df['plot_loc'] = loc
    sns.lineplot(df, x="plot_loc", y="fraction_designable", color="orange", ax=ax2, errorbar=None, linewidth=3)
else:
    sns.lineplot(df, x="seq_len", y="fraction_designable", color="orange", ax=ax2, errorbar=None)
ax2.set(ylabel="Fraction pass")
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

ax.axhline(y=1.0, color='red', linestyle='--')
# plt.suptitle("Sequence-structure consistency over length")
plt.savefig("designable_motif_rmsd_over_len.png")

plt.clf()

ax = plt.gca()

if discrete:
    sns.stripplot(df, x="seq_len", y="global_bb_rmsd", hue="plddt", ax=ax, palette="viridis", legend=False)
else:
    sns.scatterplot(df, x="seq_len", y="global_bb_rmsd", hue="plddt", ax=ax, palette="viridis", legend=False)
ax.set(xlabel="Sequence Length", ylabel="Global CA RMSD")
ax2 = ax.twinx()
if discrete:
    ref_loc = sorted(set(df['seq_len'].tolist()))
    loc = [ref_loc.index(i) for i in df['seq_len']]
    df['plot_loc'] = loc
    sns.lineplot(df, x="plot_loc", y="fraction_designable", color="orange", ax=ax2, errorbar=None, linewidth=3)
else:
    sns.lineplot(df, x="seq_len", y="fraction_designable", color="orange", ax=ax2, errorbar=None)
ax2.set(ylabel="Fraction pass")
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
plt.savefig("designable_global_rmsd_over_len.png")

plt.clf()
ax = plt.gca()
g = sns.jointplot(df, x="motif_all_atom_rmsd", y="global_bb_rmsd", hue="plddt", ax=ax, palette="viridis")
g.ax_joint.axhline(y=2.0, color='red', linestyle='--')
g.ax_joint.axvline(x=1.0, color='red', linestyle='--')
g.ax_joint.set_ylim(0, 20)
g.ax_joint.set_xlim(0, 20)
plt.savefig("designable_rmsd_dist.png")

plt.clf()
fig, axs = plt.subplots(6, 5)
fig.set_size_inches(25, 25)
plt.subplots_adjust(wspace=0.35, hspace=0.35)
tasks_sorted = sorted(df['task'].unique().tolist())
for i, task in tqdm.tqdm(enumerate(tasks_sorted)):
    ax = axs[i // 5, (i - i//5 * 5)]
    ax.set_title(task)
    g = sns.scatterplot(df[df['task'] == task], x="motif_all_atom_rmsd", y="global_bb_rmsd", hue="plddt", ax=ax, palette="viridis", legend=False)
    ax.axhline(y=2.0, color='red', linestyle='--')
    ax.axvline(x=1.0, color='red', linestyle='--')
    ax.axvline(x=1.5, color='orange', linestyle='--')
plt.savefig("designable_rmsd_dist_by_task.png")