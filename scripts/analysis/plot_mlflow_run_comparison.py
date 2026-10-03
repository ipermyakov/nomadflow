#!/usr/bin/env python3
"""Plot training curves from MLflow runs for quarterly report figures."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    label: str


@dataclass(frozen=True)
class PlotLimits:
    max_epoch: int = 300
    ymax: float = 6.0


def filter_epoch_frame(df: pd.DataFrame, limits: PlotLimits) -> tuple[pd.DataFrame, bool]:
    """Filter by epoch; clip values only if any exceed ymax. Returns (frame, cap_ymax)."""
    if df.empty:
        return df, False
    out = df[df["epoch_idx"] <= limits.max_epoch].copy()
    if limits.ymax is None:
        return out, False
    cap_ymax = bool((out["value"] > limits.ymax).any())
    if cap_ymax:
        out["value"] = out["value"].clip(upper=limits.ymax)
    return out, cap_ymax


def limits_title_suffix(limits: PlotLimits, cap_ymax: bool) -> str:
    base = f"epochs 1–{limits.max_epoch}"
    if limits.ymax is not None and cap_ymax:
        return f"{base}, y ≤ {limits.ymax}"
    return base


def style_train_axis(ax: plt.Axes, limits: PlotLimits, *, cap_ymax: bool = False) -> None:
    ax.set_xlim(0, limits.max_epoch)
    if cap_ymax and limits.ymax is not None:
        ymin, _ = ax.get_ylim()
        ax.set_ylim(bottom=ymin, top=limits.ymax)


def style_val_axis(ax: plt.Axes, limits: PlotLimits, *, cap_ymax: bool = False) -> None:
    if cap_ymax and limits.ymax is not None:
        ymin, _ = ax.get_ylim()
        ax.set_ylim(bottom=ymin, top=limits.ymax)


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0 or w <= 1:
        return x
    w = int(w)
    if n < w:
        return x.copy()
    ret = np.convolve(x, np.ones(w), "valid") / w
    return np.concatenate([np.full(w - 1, ret[0]), ret])


def parse_run_arg(raw: str) -> RunSpec:
    if ":" in raw:
        run_id, label = raw.split(":", 1)
        return RunSpec(run_id=run_id.strip(), label=label.strip())
    return RunSpec(run_id=raw.strip(), label=raw.strip()[:12])


def fetch_epoch_series(client: MlflowClient, run_id: str, metric: str) -> pd.DataFrame:
    history = client.get_metric_history(run_id, metric)
    if not history:
        return pd.DataFrame(columns=["epoch_idx", "value", "global_step"])
    rows = []
    for i, point in enumerate(history, start=1):
        rows.append(
            {
                "epoch_idx": i,
                "value": float(point.value),
                "global_step": int(point.step),
            }
        )
    return pd.DataFrame(rows)


def run_legend_label(client: MlflowClient, spec: RunSpec) -> str:
    run = client.get_run(spec.run_id)
    dataset = run.data.tags.get("dataset_config", "?")
    dataset_short = re.sub(r"\.yaml$", "", dataset)
    if len(dataset_short) > 28:
        dataset_short = dataset_short[:25] + "..."
    status = run.info.status
    epoch = run.data.metrics.get("epoch")
    ep_suffix = f", {int(epoch)} ep" if epoch is not None else ""
    if status == "FAILED":
        ep_suffix += ", FAILED"
    elif status == "RUNNING":
        ep_suffix += ", running"
    return f"{spec.label} ({dataset_short}{ep_suffix})"


def load_runs_frame(
    client: MlflowClient,
    specs: list[RunSpec],
    metric: str,
    smooth: int,
    limits: PlotLimits,
) -> tuple[pd.DataFrame, bool]:
    frames = []
    cap_ymax = False
    for spec in specs:
        df = fetch_epoch_series(client, spec.run_id, metric)
        if df.empty:
            continue
        if smooth > 1:
            df = df.copy()
            df["value"] = moving_average(df["value"].to_numpy(), smooth)
        df, cap = filter_epoch_frame(df, limits)
        cap_ymax = cap_ymax or cap
        if df.empty:
            continue
        df["run"] = run_legend_label(client, spec)
        df["metric"] = metric
        frames.append(df)
    if not frames:
        return pd.DataFrame(), False
    return pd.concat(frames, ignore_index=True), cap_ymax


def _lineplot_kw(n_points: int) -> dict:
    if n_points <= 40:
        return {"marker": "o", "markersize": 4}
    return {}


def plot_metric_comparison(
    client: MlflowClient,
    specs: list[RunSpec],
    metric: str,
    out_path: Path,
    *,
    title: str,
    ylabel: str,
    smooth: int,
    limits: PlotLimits,
    figsize: tuple[float, float] = (11, 5),
) -> bool:
    df, cap_ymax = load_runs_frame(client, specs, metric, smooth, limits)
    if df.empty:
        print(f"skip {out_path.name}: no data for {metric}")
        return False

    fig, ax = plt.subplots(figsize=figsize)
    for run_name, sub in df.groupby("run"):
        sns.lineplot(
            sub,
            x="epoch_idx",
            y="value",
            label=run_name,
            ax=ax,
            **_lineplot_kw(len(sub)),
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} ({limits_title_suffix(limits, cap_ymax)})")
    style_train_axis(ax, limits, cap_ymax=cap_ymax)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return True


def plot_loss_components(
    client: MlflowClient,
    specs: list[RunSpec],
    out_path: Path,
    smooth: int,
    limits: PlotLimits,
) -> None:
    metrics = [
        ("train/frame_vf_loss_epoch", "Frame VF loss"),
        ("train/seq_loss_epoch", "Seq loss"),
        ("train/rot_vf_loss_epoch", "Rot VF loss"),
        ("train/trans_vf_loss_epoch", "Trans VF loss"),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    axs_flat = axs.ravel()

    any_cap = False
    for ax, (metric, ylab) in zip(axs_flat, metrics):
        df, cap_ymax = load_runs_frame(client, specs, metric, smooth, limits)
        any_cap = any_cap or cap_ymax
        if df.empty:
            ax.set_title(f"{ylab} (no data)")
            continue
        for run_name, sub in df.groupby("run"):
            sns.lineplot(
                sub,
                x="epoch_idx",
                y="value",
                label=run_name,
                ax=ax,
                **_lineplot_kw(len(sub)),
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylab)
        style_train_axis(ax, limits, cap_ymax=cap_ymax)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)

    fig.suptitle(
        f"Train loss components ({limits_title_suffix(limits, any_cap)})",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _val_points_up_to_epoch(
    client: MlflowClient,
    spec: RunSpec,
    history: list,
    limits: PlotLimits,
) -> list:
    run = client.get_run(spec.run_id)
    total_ep = int(float(run.data.metrics.get("epoch", limits.max_epoch)))
    if total_ep <= 0:
        total_ep = limits.max_epoch
    n_keep = max(1, round(len(history) * limits.max_epoch / total_ep))
    return history[:n_keep]


def plot_val_loss(
    client: MlflowClient,
    specs: list[RunSpec],
    out_path: Path,
    limits: PlotLimits,
) -> None:
    val_metrics = [
        ("val/loss", "Val loss"),
        ("val_task/motif_scaffold_unindexed_loss", "Val motif scaffold loss"),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))

    any_cap = False
    for ax, (metric, ylab) in zip(axs, val_metrics):
        frames = []
        cap_ymax = False
        for spec in specs:
            history = client.get_metric_history(spec.run_id, metric)
            if not history:
                continue
            history = _val_points_up_to_epoch(client, spec, history, limits)
            rows = []
            for i, p in enumerate(history, start=1):
                raw = float(p.value)
                if limits.ymax is not None and raw > limits.ymax:
                    cap_ymax = True
                    raw = limits.ymax
                rows.append(
                    {
                        "val_idx": i,
                        "value": raw,
                        "run": run_legend_label(client, spec),
                    }
                )
            frames.append(pd.DataFrame(rows))
        any_cap = any_cap or cap_ymax
        if not frames:
            ax.set_title(f"{ylab} (no data)")
            continue
        df = pd.concat(frames, ignore_index=True)
        for run_name, sub in df.groupby("run"):
            sns.lineplot(
                sub,
                x="val_idx",
                y="value",
                label=run_name,
                ax=ax,
                marker="o",
                markersize=4,
            )
        ax.set_xlabel("Validation checkpoint")
        ax.set_ylabel(ylab)
        style_val_axis(ax, limits, cap_ymax=cap_ymax)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)

    fig.suptitle(
        f"Validation metrics (≈ {limits_title_suffix(limits, any_cap)})",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_spatial_aggregator(
    client: MlflowClient,
    spatial_spec: RunSpec,
    out_path: Path,
    smooth: int,
    limits: PlotLimits,
) -> None:
    metrics = [
        ("train/aggregator_rot_loss_epoch", "Aggregator rot loss"),
        ("train/aggregator_trans_loss_epoch", "Aggregator trans loss"),
        ("train/motif_real_rot_loss_epoch", "Motif real rot loss"),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    any_cap = False
    for ax, (metric, ylab) in zip(axs, metrics):
        df, cap_ymax = load_runs_frame(client, [spatial_spec], metric, smooth, limits)
        any_cap = any_cap or cap_ymax
        if df.empty:
            ax.set_title(f"{ylab} (no data)")
            continue
        sns.lineplot(
            df,
            x="epoch_idx",
            y="value",
            ax=ax,
            color="C0",
            **_lineplot_kw(len(df)),
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylab)
        style_train_axis(ax, limits, cap_ymax=cap_ymax)
        ax.grid(True, alpha=0.3)

    label = run_legend_label(client, spatial_spec)
    fig.suptitle(
        f"VGM / aggregator losses — {label} ({limits_title_suffix(limits, any_cap)})",
        fontsize=10,
        y=1.05,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_comparison_panel(
    client: MlflowClient,
    specs: list[RunSpec],
    out_path: Path,
    smooth: int,
    limits: PlotLimits,
) -> None:
    panels = [
        ("train/loss_epoch", "Train loss"),
        ("task/motif_scaffold_unindexed_loss_epoch", "Motif scaffold task loss"),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))

    any_cap = False
    for ax, (metric, title) in zip(axs, panels):
        df, cap_ymax = load_runs_frame(client, specs, metric, smooth, limits)
        any_cap = any_cap or cap_ymax
        if df.empty:
            ax.set_title(f"{title} (no data)")
            continue
        for run_name, sub in df.groupby("run"):
            sns.lineplot(
                sub,
                x="epoch_idx",
                y="value",
                label=run_name,
                ax=ax,
                **_lineplot_kw(len(sub)),
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(title)
        style_train_axis(ax, limits, cap_ymax=cap_ymax)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)

    fig.suptitle(
        "pretrain_spatial vs baseline_afdb "
        f"({limits_title_suffix(limits, any_cap)}; different datasets)",
        y=1.05,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def print_final_summary(client: MlflowClient, specs: list[RunSpec]) -> None:
    keys = [
        "train/loss_epoch",
        "task/motif_scaffold_unindexed_loss_epoch",
        "train/frame_vf_loss_epoch",
        "grad_norm_epoch",
        "val/loss",
        "val_task/motif_scaffold_unindexed_loss",
    ]
    print("\n=== Final metric values ===")
    for spec in specs:
        run = client.get_run(spec.run_id)
        print(f"\n{spec.label} ({spec.run_id}) status={run.info.status}")
        for key in keys:
            hist = client.get_metric_history(spec.run_id, key)
            if hist:
                print(f"  {key}: {hist[-1].value:.4f} (n={len(hist)})")
            else:
                print(f"  {key}: —")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI (default: REPO_ROOT/mlruns as file:// URI)",
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="RUN_ID:LABEL",
        help="Run id and display label, e.g. abc123:pretrain_spatial",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "docs" / "reports" / "figures",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=0,
        help="Moving-average window for train epoch curves (0=off)",
    )
    parser.add_argument(
        "--spatial-run-id",
        default=None,
        help="Run id for fig_c5 aggregator panel (default: first --run)",
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=300,
        help="Show train metrics only up to this epoch (default: 300)",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=6.0,
        help="Clip values above this and fix y-axis top; auto top if all values ≤ this (default: 6)",
    )
    parser.add_argument(
        "--no-ymax",
        action="store_true",
        help="Do not clip y-axis or metric values",
    )
    args = parser.parse_args()

    tracking_uri = args.tracking_uri
    if tracking_uri is None:
        mlruns = REPO_ROOT / "mlruns"
        tracking_uri = mlruns.as_uri()

    specs = [parse_run_arg(r) for r in args.run]
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="notebook")
    client = MlflowClient(tracking_uri=tracking_uri)

    spatial_id = args.spatial_run_id or specs[0].run_id
    spatial_spec = next((s for s in specs if s.run_id == spatial_id), specs[0])

    limits = PlotLimits(
        max_epoch=args.max_epoch,
        ymax=None if args.no_ymax else args.ymax,
    )

    plot_metric_comparison(
        client,
        specs,
        "train/loss_epoch",
        out_dir / "fig_c1_train_loss.png",
        title="Train loss (epoch)",
        ylabel="train/loss",
        smooth=args.smooth,
        limits=limits,
    )
    plot_metric_comparison(
        client,
        specs,
        "task/motif_scaffold_unindexed_loss_epoch",
        out_dir / "fig_c2_motif_task_loss.png",
        title="Motif scaffold unindexed task loss (epoch)",
        ylabel="task/motif_scaffold_unindexed_loss",
        smooth=args.smooth,
        limits=limits,
    )
    plot_loss_components(
        client, specs, out_dir / "fig_c3_loss_components.png", args.smooth, limits
    )
    plot_val_loss(client, specs, out_dir / "fig_c4_val_loss.png", limits)
    plot_spatial_aggregator(
        client,
        spatial_spec,
        out_dir / "fig_c5_spatial_aggregator.png",
        args.smooth,
        limits,
    )
    plot_comparison_panel(
        client, specs, out_dir / "fig_c6_comparison_panel.png", args.smooth, limits
    )
    print_final_summary(client, specs)


if __name__ == "__main__":
    main()
