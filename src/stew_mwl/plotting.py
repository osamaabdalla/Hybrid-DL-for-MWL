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
