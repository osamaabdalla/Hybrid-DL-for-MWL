
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf


def make_gradcam_heatmap(model, sequence_batch, class_index=None, conv_layer_name=None):
    if conv_layer_name is None:
        conv_candidates = [l.name for l in model.layers if "conv" in l.name.lower()]
        if not conv_candidates:
            raise ValueError("Could not infer a convolutional layer for Grad-CAM.")
        conv_layer_name = conv_candidates[-1]

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(sequence_batch)
        if class_index is None:
            class_index = tf.argmax(preds[0])
        loss = preds[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def activation_frequency_summary(heatmaps, threshold=0.6):
    heatmaps = np.asarray(heatmaps)
    return (heatmaps >= threshold).mean(axis=0)


def default_frontal_parietal_masks(image_h: int, image_w: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Anterior / posterior halves of the interpolated topomap grid as proxies for
    frontal vs parietal regions (STEW 10–20 layout does not include Fp1/Fz/Pz).
    """
    frontal = np.zeros((image_h, image_w), dtype=bool)
    parietal = np.zeros((image_h, image_w), dtype=bool)
    frontal[: max(1, image_h // 2), :] = True
    parietal[image_h // 2 :, :] = True
    return frontal, parietal


def _heatmap_to_2d(hm: np.ndarray) -> np.ndarray:
    a = np.asarray(hm, dtype=np.float64)
    while a.ndim > 2:
        a = np.mean(a, axis=0)
    return a


def region_scores_from_heatmap(hm: np.ndarray, frontal_mask: np.ndarray, parietal_mask: np.ndarray) -> dict[str, float]:
    h2 = _heatmap_to_2d(hm)
    if h2.shape != frontal_mask.shape:
        raise ValueError(f"Heatmap shape {h2.shape} does not match mask {frontal_mask.shape}.")
    return {
        "frontal_mean_importance": float(np.mean(h2[frontal_mask])),
        "parietal_mean_importance": float(np.mean(h2[parietal_mask])),
    }


def collect_gradcam_export_rows(
    model: tf.keras.Model,
    x: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    fold_id: int = 0,
    subject_id: int = -1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Build rows for `gradcam_sample_scores.csv` and aggregated `gradcam_region_summary.csv`.
    x: [N, T, H, W, C], y: integer class indices.
    """
    f_mask, p_mask = default_frontal_parietal_masks(int(x.shape[2]), int(x.shape[3]))
    sample_rows: list[dict[str, Any]] = []
    agg: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for i in range(len(x)):
        batch = x[i : i + 1]
        ci = int(y[i])
        cname = class_names[ci] if 0 <= ci < len(class_names) else str(ci)
        hm = make_gradcam_heatmap(model, batch, class_index=ci)
        scores = region_scores_from_heatmap(hm, f_mask, p_mask)
        sample_rows.append(
            {
                "fold_id": fold_id,
                "subject_id": subject_id,
                "class_name": cname,
                "frontal_score": scores["frontal_mean_importance"],
                "parietal_score": scores["parietal_mean_importance"],
            }
        )
        agg[cname].append((scores["frontal_mean_importance"], scores["parietal_mean_importance"]))

    region_rows = []
    for cname, pairs in agg.items():
        region_rows.append(
            {
                "class_name": cname,
                "frontal_mean_importance": float(np.mean([p[0] for p in pairs])),
                "parietal_mean_importance": float(np.mean([p[1] for p in pairs])),
            }
        )
    return region_rows, sample_rows


def run_gradcam_export_for_checkpoint(
    cfg: Any,
    manifest: pd.DataFrame,
    checkpoint: Any,
    max_samples: int = 16,
    save_example_figure: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load one fold checkpoint and score a small batch from the first manifest row (smoke / demo).
    Returns (region_summary_df, sample_scores_df).
    """
    from tensorflow import keras

    from .config import CLASS_NAMES
    from .train import build_dataset_for_subjects

    path = Path(str(checkpoint))
    if not path.is_file():
        return pd.DataFrame(), pd.DataFrame()

    model = keras.models.load_model(path)
    row = manifest.iloc[0:1]
    x, y, _ = build_dataset_for_subjects(row, cfg)
    if len(x) == 0:
        return pd.DataFrame(), pd.DataFrame()
    n = min(max_samples, len(x))
    x, y = x[:n], y[:n]
    sid = int(row["subject"].iloc[0])
    region, samples = collect_gradcam_export_rows(
        model, x, y, CLASS_NAMES, fold_id=0, subject_id=sid
    )
    if save_example_figure and len(x):
        from .plotting import plot_gradcam_heatmap

        ci = int(y[0])
        hm = make_gradcam_heatmap(model, x[:1], class_index=ci)
        plot_gradcam_heatmap(
            hm,
            cfg,
            name="gradcam_example.png",
            title=f"Grad-CAM (true={CLASS_NAMES[ci]})",
        )
    return pd.DataFrame(region), pd.DataFrame(samples)
