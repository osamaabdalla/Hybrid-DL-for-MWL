
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import ttest_rel, wilcoxon

def summarize_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }

def psd_svm_baseline(train_x, train_y, test_x, seed=42):
    """Legacy: mean RGB over space-time (weak proxy). Prefer `psd_svm_baseline_from_features`."""
    xtr = train_x.mean(axis=(1, 2, 3))
    xte = test_x.mean(axis=(1, 2, 3))
    return psd_svm_baseline_from_features(xtr, train_y, xte, seed=seed)


def psd_svm_baseline_from_features(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """PSD-SVM on flat feature rows (e.g. Welch θ/α/β × 14 channels = 42-D per trial)."""
    xtr = np.asarray(train_x, dtype=np.float32)
    xte = np.asarray(test_x, dtype=np.float32)
    n_comp = min(20, xtr.shape[1], max(1, xtr.shape[0] - 1))
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_comp, random_state=seed)),
            ("svm", SVC(kernel="rbf", C=2.0, gamma="scale", random_state=seed)),
        ]
    )
    pipe.fit(xtr, train_y)
    return pipe.predict(xte)

def aggregate_fold_metrics(rows):
    df = pd.DataFrame(rows)
    if len(df) == 0:
        summary = {k: float("nan") for k in ["accuracy", "macro_f1", "balanced_accuracy", "cohen_kappa"]}
        summary["std_accuracy"] = float("nan")
        summary["std_macro_f1"] = float("nan")
        return df, summary
    summary = df[["accuracy", "macro_f1", "balanced_accuracy", "cohen_kappa"]].mean().to_dict()
    summary["std_accuracy"] = float(df["accuracy"].std(ddof=1)) if len(df) > 1 else 0.0
    summary["std_macro_f1"] = float(df["macro_f1"].std(ddof=1)) if len(df) > 1 else 0.0
    return df, summary


def paired_ttest_detail(
    full_df: pd.DataFrame,
    other_df: pd.DataFrame,
    metric: str,
    alpha: float = 0.05,
) -> dict:
    """Paired t-test on LOSO folds matched by subject."""
    if len(full_df) == 0 or len(other_df) == 0:
        return {
            "t_statistic": float("nan"),
            "p_value": float("nan"),
            "significant": False,
        }
    merged = full_df[["subject", metric]].merge(
        other_df[["subject", metric]], on="subject", suffixes=("_full", "_oth")
    )
    if len(merged) < 2:
        return {
            "t_statistic": float("nan"),
            "p_value": float("nan"),
            "significant": False,
        }
    a = merged[f"{metric}_full"].to_numpy(dtype=float)
    b = merged[f"{metric}_oth"].to_numpy(dtype=float)
    t_stat, p_value = ttest_rel(a, b, nan_policy="omit")
    p = float(p_value) if not np.isnan(p_value) else float("nan")
    return {
        "t_statistic": float(t_stat) if not np.isnan(t_stat) else float("nan"),
        "p_value": p,
        "significant": bool(p < alpha) if not np.isnan(p) else False,
    }


def paired_ttest_vs_full(full_df: pd.DataFrame, other_df: pd.DataFrame, metric: str) -> float:
    return paired_ttest_detail(full_df, other_df, metric)["p_value"]


def wilcoxon_paired_detail(
    full_df: pd.DataFrame,
    other_df: pd.DataFrame,
    metric: str,
    alpha: float = 0.05,
) -> dict:
    """Wilcoxon signed-rank test on per-subject paired differences (full − other), same pairing as `paired_ttest_detail`."""
    if len(full_df) == 0 or len(other_df) == 0:
        return {
            "wilcoxon_statistic": float("nan"),
            "p_value": float("nan"),
            "significant": False,
        }
    merged = full_df[["subject", metric]].merge(
        other_df[["subject", metric]], on="subject", suffixes=("_full", "_oth")
    )
    if len(merged) < 2:
        return {
            "wilcoxon_statistic": float("nan"),
            "p_value": float("nan"),
            "significant": False,
        }
    d = (
        merged[f"{metric}_full"].to_numpy(dtype=float)
        - merged[f"{metric}_oth"].to_numpy(dtype=float)
    )
    if np.allclose(d, 0.0):
        return {"wilcoxon_statistic": 0.0, "p_value": 1.0, "significant": False}
    try:
        res = wilcoxon(d, alternative="two-sided", zero_method="wilcox")
    except ValueError:
        return {"wilcoxon_statistic": float("nan"), "p_value": float("nan"), "significant": False}
    if hasattr(res, "statistic"):
        stat, p = float(res.statistic), float(res.pvalue)
    else:
        stat, p = float(res[0]), float(res[1])
    sig = bool(p < alpha) if not np.isnan(p) else False
    return {"wilcoxon_statistic": stat, "p_value": p, "significant": sig}


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def wilcoxon_vs_full(full_scores, baseline_scores):
    stat, p = wilcoxon(full_scores, baseline_scores, alternative="greater")
    return {"wilcoxon_statistic": float(stat), "p_value": float(p)}

def classification_tables(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(report).T, cm
