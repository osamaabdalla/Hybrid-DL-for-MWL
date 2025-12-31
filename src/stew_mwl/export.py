"""Write result tables under outputs/csv/ for the notebook and OSF-style sharing."""

from __future__ import annotations
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import CLASS_NAMES, Config
from .data import preprocess_signal, read_signal_txt
from .eval import aggregate_fold_metrics, paired_ttest_detail
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
        for _, path_key, label in (
            ("lo", "lo_path", "BL"),
            ("hi", "hi_path", row["task_class"]),
        ):
            sig = preprocess_signal(read_signal_txt(Path(row[path_key])), cfg)
            seqs = build_sequence_images(sig, cfg, window_seconds=window_seconds, sequence_length=seq_steps)
            rows.append(
                {
                    "subject_id": sid,
                    "class_name": label,
                    "original_samples": int(sig.shape[0]),
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


def export_ablation_summary(ablation_results: dict[str, pd.DataFrame], full_df: pd.DataFrame, cfg: Config) -> None:
    rows = []
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


def export_statistical_tests(
    full_df: pd.DataFrame,
    baseline_results: dict[str, pd.DataFrame],
    ablation_results: dict[str, pd.DataFrame] | None,
    cfg: Config,
) -> None:
    rows = []
    for bname, df in baseline_results.items():
        if len(df) == 0:
            continue
        d = paired_ttest_detail(full_df, df, "accuracy")
        rows.append(
            {
                "comparison_name": f"proposed_vs_{bname}",
                "model_a": "proposed",
                "model_b": bname,
                "metric": "accuracy",
                "t_statistic": d["t_statistic"],
                "p_value": d["p_value"],
                "significant": d["significant"],
            }
        )
    if ablation_results:
        for aname, df in ablation_results.items():
            if len(df) == 0:
                continue
            d = paired_ttest_detail(full_df, df, "accuracy")
            rows.append(
                {
                    "comparison_name": f"proposed_vs_{aname}",
                    "model_a": "proposed",
                    "model_b": aname,
                    "metric": "accuracy",
                    "t_statistic": d["t_statistic"],
                    "p_value": d["p_value"],
                    "significant": d["significant"],
                }
            )
    pd.DataFrame(rows).to_csv(cfg.csv_dir / "statistical_tests.csv", index=False)


def export_experiment_registry(cfg: Config, notes: str = "") -> None:
    commit = ""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent.parent, text=True
        ).strip()
    except Exception:
        pass
    pd.DataFrame(
        [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "git_commit": commit,
                "seed": cfg.seed,
                "config_path": str(cfg.config_path) if cfg.config_path else "",
                "notes": notes,
            }
        ]
    ).to_csv(cfg.csv_dir / "experiment_registry.csv", index=False)
