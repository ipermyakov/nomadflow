import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--min_epoch", default=0, type=int)
parser.add_argument("--max_epoch", default=-1, type=int)
args = parser.parse_args()


df = pd.read_csv("metrics.csv")
df = df.iloc[1:]
df = df[df['train/loss_epoch'].astype(str).str.len() > 0]
df = df[~df['train/loss_epoch'].isna()]
df = df[df.epoch.astype(int) >= args.min_epoch]
if args.max_epoch != -1:
    df = df[df.epoch.astype(int) <= args.max_epoch]

n_epochs = len(df)
# Few points: show markers so the curve is readable (otherwise it can look like a dull straight segment).
_lineplot_kw = (
    {"marker": "o", "markersize": 5} if n_epochs <= 40 else {}
)

fig, axs = plt.subplots(5, 3)
fig.set_size_inches(12, 20)

get_ax_idx = lambda x: (x // 3, x % 3)

loss_terms = [
    'train/loss_epoch',
    'train/frame_vf_loss_epoch',
    'train/rot_vf_loss_epoch',
    'train/trans_vf_loss_epoch',
    'train/seq_loss_epoch',
    'train/smooth_lddt_epoch',
    'task/motif_scaffold_unindexed_frame_vf_loss_epoch',
    'task/motif_scaffold_indexed_frame_vf_loss_epoch',
    'task/unconditional_frame_vf_loss_epoch',
    'train/scaled_fafe_epoch',
    'grad_norm_epoch',
    # ('task/motif_scaffold_unindexed_frame_vf_loss_epoch', 'task/motif_scaffold_indexed_frame_vf_loss_epoch', 'task/unconditional_frame_vf_loss_epoch')
    ('task/motif_scaffold_unindexed_frame_vf_loss_epoch', 'task/unconditional_frame_vf_loss_epoch')

]

def moving_average(x, w):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0 or w <= 1:
        return x
    w = int(w)
    # If n < w, clipping w to n makes every point the global mean → a flat line.
    if n < w:
        return x.copy()
    ret = np.convolve(x, np.ones(w), 'valid') / w
    ret = np.concatenate([np.full(w - 1, ret[0]), ret])
    return ret

for idx, l in enumerate(loss_terms):
    if isinstance(l, str):
        if l in df.columns:
            df[l] = moving_average(df[l].to_numpy(), 10)
            sns.lineplot(
                df,
                x="epoch",
                y=l,
                ax=axs[get_ax_idx(idx)],
                **_lineplot_kw,
            )
            # axs[get_ax_idx(idx)].set_ylim(bottom=1.2, top=1.45)
    elif isinstance(l, tuple):
        plot = True
        for term in l:
            if term not in df.columns:
                plot = False
        if plot:
            long = df[list(l) + ["epoch"]].melt(
                id_vars="epoch", var_name="metric", value_name="value"
            )
            sns.lineplot(
                long,
                x="epoch",
                y="value",
                hue="metric",
                ax=axs[get_ax_idx(idx)],
                **_lineplot_kw,
            )

# plt.legend()
plt.savefig("loss_curve.png")
