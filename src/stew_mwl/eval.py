
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
from scipy.stats import wilcoxon

def summarize_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }

def psd_svm_baseline(train_x, train_y, test_x, seed=42):
    # train_x/test_x: [N, T, H, W, C] -> flattened for baseline
    xtr = train_x.mean(axis=(1,2,3))  # [N, C]
    xte = test_x.mean(axis=(1,2,3))
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(3, xtr.shape[1]), random_state=seed)),
        ("svm", SVC(kernel="rbf", C=2.0, gamma="scale", random_state=seed)),
    ])
    pipe.fit(xtr, train_y)
    return pipe.predict(xte)

def aggregate_fold_metrics(rows):
    df = pd.DataFrame(rows)
    summary = df[["accuracy","macro_f1","balanced_accuracy","cohen_kappa"]].mean().to_dict()
    summary["std_accuracy"] = float(df["accuracy"].std(ddof=1))
    return df, summary

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
