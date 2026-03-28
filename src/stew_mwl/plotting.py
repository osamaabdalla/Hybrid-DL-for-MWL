"""Figures for papers and sanity checks (confusion matrix, baselines, ablations, VAE losses)."""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import CLASS_NAMES, Config


def plot_confusion_matrix(cm: np.ndarray, cfg: Config, name: str = "confusion_matrix_proposed.png") -> Path:
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (proposed)")
    path = cfg.figures_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_vae_loss_curves(csv_path: Path, cfg: Config) -> Path | None:
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    if "train_total_loss" not in df.columns:
        return None
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    if "fold_id" in df.columns:
        for fid in sorted(df["fold_id"].unique())[:12]:
            sub = df[df["fold_id"] == fid]
            ax.plot(sub["epoch"], sub["train_total_loss"], alpha=0.5, label=f"fold {int(fid)}")
    else:
        ax.plot(df["epoch"], df["train_total_loss"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("VAE loss")
    ax.set_title("VAE training (subset of folds)")
    if "fold_id" in df.columns and df["fold_id"].nunique() > 1:
        ax.legend(fontsize=6, ncol=2)
    path = cfg.figures_dir / "vae_loss_curves.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_baseline_bar(summary_path: Path, cfg: Config) -> Path | None:
    if not summary_path.is_file():
        return None
    df = pd.read_csv(summary_path)
    if len(df) == 0:
        return None
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(df))
    ax.bar(x, df["mean_accuracy"], yerr=df.get("std_accuracy", 0), capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model_name"], rotation=25, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Baseline comparison (LOSO mean ± std)")
    path = cfg.figures_dir / "baseline_comparison.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_ablation_bar(summary_path: Path, cfg: Config) -> Path | None:
    if not summary_path.is_file():
        return None
    df = pd.read_csv(summary_path)
    if len(df) == 0:
        return None
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(df))
    ax.bar(x, df["mean_accuracy"], yerr=df.get("std_accuracy", 0), capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(df["variant"], rotation=25, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Ablation study (LOSO)")
    path = cfg.figures_dir / "ablation_comparison.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_gradcam_heatmap(hm: np.ndarray, cfg: Config, name: str = "gradcam_example.png", title: str = "Grad-CAM (topomap)") -> Path:
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    h2 = np.asarray(hm, dtype=np.float32)
    while h2.ndim > 2:
        h2 = np.mean(h2, axis=0)
    fig, ax = plt.subplots(figsize=(4, 5))
    im = ax.imshow(h2, cmap="magma", aspect="auto")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path = cfg.figures_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_gradcam_region_bars(summary_path: Path, cfg: Config, name: str = "gradcam_region_importance.png") -> Path | None:
    """Bar chart of frontal vs parietal importance by class (from `gradcam_region_summary.csv`)."""
    if not summary_path.is_file():
        return None
    df = pd.read_csv(summary_path)
    if len(df) == 0 or "class_name" not in df.columns:
        return None
    frontal = df.get("frontal_mean_importance", df.get("frontal_score"))
    parietal = df.get("parietal_mean_importance", df.get("parietal_score"))
    if frontal is None or parietal is None:
        return None
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(df))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, frontal, w, label="Frontal (anterior grid)")
    ax.bar(x + w / 2, parietal, w, label="Parietal (posterior grid)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["class_name"])
    ax.set_ylabel("Mean heatmap activation")
    ax.set_title("Grad-CAM regional importance by class")
    ax.legend()
    path = cfg.figures_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_gradcam_panel_from_samples(
    heatmaps: list[np.ndarray],
    cfg: Config,
    name: str = "gradcam_class_panel.png",
    titles: list[str] | None = None,
) -> Path | None:
    """Small multiples of 2D heatmaps (e.g. one per class)."""
    if not heatmaps:
        return None
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    n = len(heatmaps)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i, hm in enumerate(heatmaps):
        h2 = np.asarray(hm, dtype=np.float32)
        while h2.ndim > 2:
            h2 = np.mean(h2, axis=0)
        ax = axes[i]
        ax.imshow(h2, cmap="magma", aspect="auto")
        ax.set_title(titles[i] if titles and i < len(titles) else f"sample {i}")
        ax.axis("off")
    for j in range(len(heatmaps), len(axes)):
        axes[j].axis("off")
    path = cfg.figures_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_gradcam_outputs_on_disk(cfg: Config) -> list[Path]:
    """If `gradcam_region_summary.csv` exists, save region-importance figure. Call after Grad-CAM CSV export."""
    paths: list[Path] = []
    p = plot_gradcam_region_bars(cfg.csv_dir / "gradcam_region_summary.csv", cfg)
    if p is not None:
        paths.append(p)
    return paths
