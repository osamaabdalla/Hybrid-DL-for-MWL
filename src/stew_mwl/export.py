"""Write result tables under outputs/csv/ for the notebook and OSF-style sharing."""

from __future__ import annotations
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import CLASS_NAMES, Config
from .data import load_preprocessed_signal
from .eval import aggregate_fold_metrics, paired_ttest_detail, wilcoxon_paired_detail
from .features import build_sequence_images


def export_dataset_manifest(manifest: pd.DataFrame, cfg: Config) -> Path:
    cfg.csv_dir.mkdir(parents=True, exist_ok=True)
    out = manifest.copy()
    out["final_label_task"] = out["task_class"]
    out.to_csv(cfg.csv_dir / "dataset_manifest.csv", index=False)
    return cfg.csv_dir / "dataset_manifest.csv"


def export_segmentation_summary(manifest: pd.DataFrame, cfg: Config) -> Path:
    rows = []
    window_seconds = cfg.parent_window_seconds
    seq_steps = cfg.seq_len
    for _, row in manifest.iterrows():
        sid = int(row["subject"])
        for side, path_key, label in (
            ("lo", "lo_path", "BL"),
            ("hi", "hi_path", row["task_class"]),
        ):
            sig = load_preprocessed_signal(Path(row[path_key]), sid, side, cfg)
            seqs = build_sequence_images(sig, cfg, window_seconds=window_seconds, sequence_length=seq_steps)
            epoch_samp = max(1, int(cfg.epoch_seconds * cfg.sfreq))
            num_epochs = int(sig.shape[0] // epoch_samp)
            rows.append(
                {
                    "subject_id": sid,
                    "class_name": label,
                    "original_samples": int(sig.shape[0]),
                    "num_epochs": num_epochs,
                    "num_sequences": int(len(seqs)),
                    "window_length_sec": window_seconds,
                    "sequence_steps": seq_steps,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(cfg.csv_dir / "segmentation_summary.csv", index=False)
    return cfg.csv_dir / "segmentation_summary.csv"


def export_fold_metrics_and_predictions(
    full_df: pd.DataFrame,
    predictions: list[dict],
    cfg: Config,
) -> None:
    cfg.csv_dir.mkdir(parents=True, exist_ok=True)
    fd = full_df.copy()
    fd.to_csv(cfg.csv_dir / "fold_metrics_all.csv", index=False)
    pd.DataFrame(predictions).to_csv(cfg.csv_dir / "predictions_all.csv", index=False)


def export_classification_report_and_cm(
    predictions: list[dict],
    cfg: Config,
) -> None:
    if not predictions:
        return
    df = pd.DataFrame(predictions)
    if not len(df):
        return
    yt = df["y_true"].values
    yp = df["y_pred"].values
    from sklearn.metrics import classification_report, confusion_matrix

    rep = classification_report(yt, yp, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    rep_rows = []
    for c in CLASS_NAMES:
        if c in rep:
            rep_rows.append(
                {
                    "class_name": c,
                    "precision": rep[c]["precision"],
                    "recall": rep[c]["recall"],
                    "f1_score": rep[c]["f1-score"],
                    "support": rep[c]["support"],
                }
            )
    pd.DataFrame(rep_rows).to_csv(cfg.csv_dir / "classification_report_proposed.csv", index=False)
    cm = confusion_matrix(yt, yp, labels=list(range(len(CLASS_NAMES))))
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in CLASS_NAMES], columns=CLASS_NAMES)
    cm_df.to_csv(cfg.csv_dir / "confusion_matrix_proposed.csv")


def export_baseline_comparison_summary(baseline_results: dict[str, pd.DataFrame], cfg: Config) -> None:
    rows = []
    for name, df in baseline_results.items():
        if len(df) == 0:
            continue
        _, s = aggregate_fold_metrics(df)
        rows.append(
            {
                "model_name": name,
                "mean_accuracy": s["accuracy"],
                "std_accuracy": s["std_accuracy"],
                "mean_macro_f1": s["macro_f1"],
                "std_macro_f1": s.get("std_macro_f1", 0.0),
                "mean_balanced_accuracy": s["balanced_accuracy"],
                "mean_kappa": s["cohen_kappa"],
            }
        )
    pd.DataFrame(rows).to_csv(cfg.csv_dir / "baseline_comparison_summary.csv", index=False)


def export_ablation_summary(
    ablation_results: dict[str, pd.DataFrame],
    full_df: pd.DataFrame,
    cfg: Config,
    *,
    include_full_proposed_summary_row: bool = True,
) -> None:
    rows = []
    if include_full_proposed_summary_row and len(full_df):
        _, s = aggregate_fold_metrics(full_df)
        rows.append(
            {
                "variant": "full_proposed",
                "mean_accuracy": s["accuracy"],
                "std_accuracy": s["std_accuracy"],
                "mean_macro_f1": s["macro_f1"],
                "std_macro_f1": s.get("std_macro_f1", 0.0),
                "p_value_vs_full": float("nan"),
            }
        )
    for name, df in ablation_results.items():
        if len(df) == 0:
            continue
        _, s = aggregate_fold_metrics(df)
        p = (
            paired_ttest_detail(full_df, df, "accuracy")["p_value"]
            if len(full_df) > 0 and len(df) > 0
            else float("nan")
        )
        rows.append(
            {
                "variant": name,
                "mean_accuracy": s["accuracy"],
                "std_accuracy": s["std_accuracy"],
                "mean_macro_f1": s["macro_f1"],
                "std_macro_f1": s.get("std_macro_f1", 0.0),
                "p_value_vs_full": p,
            }
        )
    pd.DataFrame(rows).to_csv(cfg.csv_dir / "ablation_summary.csv", index=False)


def export_cbam_config_results(
    cfg: Config,
    sensitivity_cbam_df: pd.DataFrame | None = None,
) -> Path:
    """
    PRD Stage G3: CBAM hyperparameters for the active YAML plus optional sensitivity sweep rows
    (mean metrics from `sensitivity_cbam.csv` aggregation).
    """
    cfg.csv_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = [
        {
            "config_label": "active_yaml",
            "cbam_enabled": cfg.cbam_enabled,
            "reduction_ratio": cfg.cbam_reduction_ratio,
            "spatial_kernel": cfg.cbam_spatial_kernel,
            "attention_order": cfg.cbam_attention_order,
            "mean_accuracy": float("nan"),
            "mean_macro_f1": float("nan"),
            "mean_balanced_accuracy": float("nan"),
            "mean_cohen_kappa": float("nan"),
            "std_accuracy": float("nan"),
            "std_macro_f1": float("nan"),
        }
    ]
    def _row_float(series: pd.Series, key: str) -> float:
        if key not in series.index:
            return float("nan")
        v = series[key]
        return float(v) if pd.notna(v) else float("nan")

    if sensitivity_cbam_df is not None and len(sensitivity_cbam_df):
        for _, r in sensitivity_cbam_df.iterrows():
            rows.append(
                {
                    "config_label": f"sweep_r{int(r['reduction_ratio'])}_k{int(r['spatial_kernel'])}",
                    "cbam_enabled": True,
                    "reduction_ratio": int(r["reduction_ratio"]),
                    "spatial_kernel": int(r["spatial_kernel"]),
                    "attention_order": cfg.cbam_attention_order,
                    "mean_accuracy": _row_float(r, "accuracy"),
                    "mean_macro_f1": _row_float(r, "macro_f1"),
                    "mean_balanced_accuracy": _row_float(r, "balanced_accuracy"),
                    "mean_cohen_kappa": _row_float(r, "cohen_kappa"),
                    "std_accuracy": _row_float(r, "std_accuracy"),
                    "std_macro_f1": _row_float(r, "std_macro_f1"),
                }
            )
    path = cfg.csv_dir / "cbam_config_results.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def export_statistical_tests(
    full_df: pd.DataFrame,
    baseline_results: dict[str, pd.DataFrame],
    ablation_results: dict[str, pd.DataFrame] | None,
    cfg: Config,
) -> None:
    """PRD Stage M: paired t-test and Wilcoxon signed-rank on matched LOSO subjects (`accuracy`)."""
    cfg.csv_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    def _add_rows(name: str, model_b: str, odf: pd.DataFrame) -> None:
        dt = paired_ttest_detail(full_df, odf, "accuracy")
        dw = wilcoxon_paired_detail(full_df, odf, "accuracy")
        rows.append(
            {
                "comparison_name": name,
                "model_a": "proposed",
                "model_b": model_b,
                "metric": "accuracy",
                "t_statistic": dt["t_statistic"],
                "p_value": dt["p_value"],
                "significant": dt["significant"],
                "wilcoxon_statistic": dw["wilcoxon_statistic"],
                "wilcoxon_p_value": dw["p_value"],
                "wilcoxon_significant": dw["significant"],
            }
        )

    for bname, df in baseline_results.items():
        if len(df) == 0:
            continue
        _add_rows(f"proposed_vs_{bname}", bname, df)
    if ablation_results:
        for aname, df in ablation_results.items():
            if len(df) == 0:
                continue
            _add_rows(f"proposed_vs_{aname}", aname, df)
    pd.DataFrame(rows).to_csv(cfg.csv_dir / "statistical_tests.csv", index=False)


def export_gradcam_summaries(
    region_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    cfg: Config,
) -> None:
    cfg.csv_dir.mkdir(parents=True, exist_ok=True)
    if len(region_df):
        region_df.to_csv(cfg.csv_dir / "gradcam_region_summary.csv", index=False)
    if len(sample_df):
        sample_df.to_csv(cfg.csv_dir / "gradcam_sample_scores.csv", index=False)


def export_experiment_registry(cfg: Config, notes: str = "") -> None:
    commit = ""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent.parent, text=True
        ).strip()
    except Exception:
        pass
    cpath = cfg.config_path
    config_name = cpath.stem if cpath else ""
    pd.DataFrame(
        [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "git_commit": commit,
                "seed": cfg.seed,
                "config_name": config_name,
                "config_path": str(cpath) if cpath else "",
                "output_root": str(cfg.output_root.resolve()),
                "data_root": str(cfg.data_root.resolve()),
                "feature_method": cfg.feature_method,
                "latent_dim": cfg.latent_dim,
                "reference_mode": cfg.reference_mode,
                "cbam_enabled": cfg.cbam_enabled,
                "cbam_attention_order": cfg.cbam_attention_order,
                "cache_preprocessed": cfg.cache_preprocessed,
                "cache_sequences": cfg.cache_sequences,
                "notes": notes,
            }
        ]
    ).to_csv(cfg.csv_dir / "experiment_registry.csv", index=False)
