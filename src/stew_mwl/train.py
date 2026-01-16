"""LOSO training, baselines, ablations, and sensitivity grids for the STEW MWL pipeline."""

from __future__ import annotations
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score

from .config import CLASS_NAMES, Config, RGB_BANDS
from .data import loso_fold_subject_ids, make_loso_splits, preprocess_signal, read_signal_txt
from .eval import aggregate_fold_metrics, psd_svm_baseline, summarize_metrics
from .features import build_sequence_images
from .models import (
    build_blstm_lstm_classifier,
    build_classifier_from_encoder,
    build_vae,
    compile_classifier,
    copy_vae_encoder_weights_to_classifier,
)


def build_dataset_for_subjects(
    subject_rows: pd.DataFrame,
    cfg: Config,
    window_seconds: int | None = None,
    sequence_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y, subjects = [], [], []
    n_ch = len(RGB_BANDS)
    for _, row in subject_rows.iterrows():
        subject = int(row["subject"])
        baseline = preprocess_signal(read_signal_txt(Path(row["lo_path"])), cfg)
        task = preprocess_signal(read_signal_txt(Path(row["hi_path"])), cfg)
        x_bl = build_sequence_images(baseline, cfg, window_seconds=window_seconds, sequence_length=sequence_length)
        x_task = build_sequence_images(task, cfg, window_seconds=window_seconds, sequence_length=sequence_length)
        tc = row["task_class"]
        if len(x_bl):
            X.append(x_bl)
            y.append(np.full(len(x_bl), cfg.class_to_id["BL"], dtype=np.int64))
            subjects.append(np.full(len(x_bl), subject, dtype=np.int64))
        if len(x_task):
            X.append(x_task)
            y.append(np.full(len(x_task), cfg.class_to_id[tc], dtype=np.int64))
            subjects.append(np.full(len(x_task), subject, dtype=np.int64))
    if not X:
        return (
            np.empty((0, 1, cfg.image_h, cfg.image_w, n_ch), np.float32),
            np.empty((0,), np.int64),
            np.empty((0,), np.int64),
        )
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    subjects = np.concatenate(subjects, axis=0)
    return X, y, subjects


class ValMacroF1Callback(tf.keras.callbacks.Callback):
    """Logs val_macro_f1 each epoch (early stopping still uses val_loss — stable with custom metrics in Keras)."""

    def __init__(self, x_val: np.ndarray, y_val: np.ndarray):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        preds = self.model.predict(self.x_val, verbose=0)
        y_p = np.argmax(preds, axis=1)
        logs["val_macro_f1"] = float(f1_score(self.y_val, y_p, average="macro", zero_division=0))


def _classification_callbacks(x_val: np.ndarray, y_val: np.ndarray) -> list[tf.keras.callbacks.Callback]:
    return [
        ValMacroF1Callback(x_val, y_val),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, mode="min", restore_best_weights=True, verbose=0
        ),
    ]


def train_vae_on_frames(
    x_train: np.ndarray,
    cfg: Config,
    log_rows: list[dict] | None = None,
    fold_id: int = 0,
) -> tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model, tf.keras.callbacks.History]:
    n_ch = x_train.shape[-1]
    frames = x_train.reshape(-1, cfg.image_h, cfg.image_w, n_ch).astype(np.float32)
    rng = np.random.default_rng(cfg.seed + fold_id + 17)
    order = rng.permutation(len(frames))
    frames = frames[order]
    vae, encoder, decoder = build_vae(
        image_shape=(cfg.image_h, cfg.image_w, n_ch),
        latent_dim=cfg.latent_dim,
    )
    vae.compile(optimizer=tf.keras.optimizers.Adam(cfg.learning_rate))
    hist = vae.fit(
        frames,
        epochs=cfg.vae_epochs,
        batch_size=cfg.batch_size,
        shuffle=False,
        verbose=0,
    )
    if log_rows is not None:
        losses = hist.history.get("loss", [])
        n = len(losses)
        recon = hist.history.get("reconstruction_loss", [float("nan")] * n)
        kl = hist.history.get("kl_loss", [float("nan")] * n)
        for i, loss in enumerate(losses):
            log_rows.append(
                {
                    "fold_id": fold_id,
                    "epoch": i + 1,
                    "train_total_loss": float(loss),
                    "train_recon_loss": float(recon[i]) if i < len(recon) else float("nan"),
                    "train_kl_loss": float(kl[i]) if i < len(kl) else float("nan"),
                    "val_total_loss": float("nan"),
                }
            )
    return vae, encoder, decoder, hist


def train_classifier_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Config,
    vae: tf.keras.Model | None = None,
    use_cbam: bool = True,
    use_encoder: bool = True,
    bidirectional: bool = True,
    sequence_model: str = "lstm",
    cbam_reduction_ratio: int | None = None,
    cbam_spatial_kernel: int | None = None,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    cr = cbam_reduction_ratio if cbam_reduction_ratio is not None else cfg.cbam_reduction_ratio
    ck = cbam_spatial_kernel if cbam_spatial_kernel is not None else cfg.cbam_spatial_kernel
    model = build_classifier_from_encoder(
        frame_shape=(cfg.image_h, cfg.image_w, x_train.shape[-1]),
        sequence_length=x_train.shape[1],
        latent_dim=cfg.latent_dim,
        n_classes=len(CLASS_NAMES),
        dropout=cfg.dropout,
        use_cbam=use_cbam,
        use_encoder=use_encoder,
        bidirectional=bidirectional,
        sequence_model=sequence_model,
        blstm_units=cfg.blstm_units,
        cbam_reduction_ratio=cr,
        cbam_spatial_kernel=ck,
    )
    if vae is not None and use_encoder:
        copy_vae_encoder_weights_to_classifier(vae, model)
    n = max(len(x_train), 1)
    decay_steps = max(n // cfg.batch_size, 1)
    model = compile_classifier(model, cfg.learning_rate, use_decay=True, decay_steps=decay_steps)
    hist = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=cfg.clf_epochs,
        batch_size=cfg.batch_size,
        verbose=0,
        callbacks=_classification_callbacks(x_val, y_val),
    )
    return model, hist


def train_blstm_lstm_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Config,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    model = build_blstm_lstm_classifier(
        frame_shape=(cfg.image_h, cfg.image_w, x_train.shape[-1]),
        sequence_length=x_train.shape[1],
        n_classes=len(CLASS_NAMES),
        dropout=cfg.dropout,
        blstm_units=cfg.blstm_units,
        lstm_units=cfg.blstm_units,
        use_encoder=False,
    )
    n = max(len(x_train), 1)
    decay_steps = max(n // cfg.batch_size, 1)
    model = compile_classifier(model, cfg.learning_rate, use_decay=True, decay_steps=decay_steps)
    hist = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=cfg.clf_epochs,
        batch_size=cfg.batch_size,
        verbose=0,
        callbacks=_classification_callbacks(x_val, y_val),
    )
    return model, hist


def train_cnn_baseline_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Config,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    return train_classifier_model(
        x_train,
        y_train,
        x_val,
        y_val,
        cfg,
        vae=None,
        use_cbam=False,
        use_encoder=False,
        bidirectional=False,
        sequence_model="cnn",
    )


def train_val_split(x_train: np.ndarray, y_train: np.ndarray, cfg: Config, split_seed: int):
    """90/10 shuffle split; deterministic given cfg.seed and split_seed."""
    rng = np.random.default_rng(cfg.seed + int(split_seed))
    idx = np.arange(len(x_train))
    rng.shuffle(idx)
    n = len(idx)
    if n < 2:
        return x_train, y_train, x_train[:1], y_train[:1]
    split = max(1, int(n * 0.9))
    if split >= n:
        split = n - 1
    tr_idx, va_idx = idx[:split], idx[split:]
    return x_train[tr_idx], y_train[tr_idx], x_train[va_idx], y_train[va_idx]


def run_loso_training(
    cfg: Config,
    manifest: pd.DataFrame,
    model_name: str = "proposed",
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    cfg.ensure_dirs()
    rows: list[dict] = []
    all_preds: list[dict] = []
    vae_log_rows: list[dict] = []
    splits = make_loso_splits(manifest, cfg)
    fold_id = 0
    for sp in splits:
        subject = sp["test_subject"]
        train_rows = manifest[manifest["subject"].isin(sp["train_subjects"])]
        test_rows = manifest[manifest["subject"] == subject]
        x_train, y_train, _ = build_dataset_for_subjects(train_rows, cfg)
        x_test, y_test, _ = build_dataset_for_subjects(test_rows, cfg)
        if len(x_train) == 0 or len(x_test) == 0:
            fold_id += 1
            continue
        x_tr, y_tr, x_va, y_va = train_val_split(x_train, y_train, cfg, int(subject))

        vae, _, _, _ = train_vae_on_frames(x_tr, cfg, log_rows=vae_log_rows, fold_id=fold_id)
        model, _ = train_classifier_model(
            x_tr,
            y_tr,
            x_va,
            y_va,
            cfg,
            vae=vae,
            use_cbam=True,
            use_encoder=True,
            bidirectional=True,
            sequence_model="lstm",
        )
        probs = model.predict(x_test, verbose=0)
        preds = probs.argmax(axis=1)
        m = summarize_metrics(y_test, preds)
        m["subject"] = int(subject)
        m["fold_id"] = fold_id
        m["model_name"] = model_name
        rows.append(m)
        for i in range(len(y_test)):
            all_preds.append(
                {
                    "fold_id": fold_id,
                    "subject_id": int(subject),
                    "model_name": model_name,
                    "y_true": int(y_test[i]),
                    "y_pred": int(preds[i]),
                    "prob_BL": float(probs[i, 0]),
                    "prob_LW": float(probs[i, 1]),
                    "prob_MW": float(probs[i, 2]),
                    "prob_HW": float(probs[i, 3]),
                }
            )
        path = cfg.models_dir / f"fold_{fold_id:02d}_{model_name}.keras"
        try:
            model.save(path)
        except OSError as err:
            warnings.warn(f"Could not save model weights to {path}: {err}", UserWarning, stacklevel=2)
        fold_id += 1
        print(f"LOSO test subject {subject:02d} | acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f}")

    df = pd.DataFrame(rows)
    if vae_log_rows:
        pd.DataFrame(vae_log_rows).to_csv(cfg.csv_dir / "vae_fold_losses.csv", index=False)
    return df, splits, all_preds


def run_baseline_models(
    cfg: Config,
    manifest: pd.DataFrame,
    loso_splits: list[dict[str, Any]] | None = None,
) -> dict[str, pd.DataFrame]:
    cfg.ensure_dirs()
    if loso_splits is None:
        loso_splits = make_loso_splits(manifest, cfg)
    out: dict[str, pd.DataFrame] = {}
    for name, trainer in (
        ("psd_svm", _run_loso_psd_svm),
        ("cnn", _run_loso_cnn),
        ("blstm_lstm", _run_loso_blstm_lstm),
    ):
        rows = []
        fold_id = 0
        for sp in loso_splits:
            subject = sp["test_subject"]
            train_rows = manifest[manifest["subject"].isin(sp["train_subjects"])]
            test_rows = manifest[manifest["subject"] == subject]
            x_train, y_train, _ = build_dataset_for_subjects(train_rows, cfg)
            x_test, y_test, _ = build_dataset_for_subjects(test_rows, cfg)
            if len(x_train) == 0 or len(x_test) == 0:
                fold_id += 1
                continue
            preds = trainer(x_train, y_train, x_test, cfg)
            m = summarize_metrics(y_test, preds)
            m["subject"] = int(subject)
            m["fold_id"] = fold_id
            m["model_name"] = name
            rows.append(m)
            fold_id += 1
        out[name] = pd.DataFrame(rows)
        out[name].to_csv(cfg.csv_dir / f"baseline_{name}_fold_metrics.csv", index=False)
    return out


def _run_loso_psd_svm(x_train, y_train, x_test, cfg: Config) -> np.ndarray:
    return psd_svm_baseline(x_train, y_train, x_test, seed=cfg.seed)


def _run_loso_cnn(x_train, y_train, x_test, cfg: Config) -> np.ndarray:
    x_tr, y_tr, x_va, y_va = train_val_split(x_train, y_train, cfg, 0)
    model, _ = train_cnn_baseline_model(x_tr, y_tr, x_va, y_va, cfg)
    return model.predict(x_test, verbose=0).argmax(axis=1)


def _run_loso_blstm_lstm(x_train, y_train, x_test, cfg: Config) -> np.ndarray:
    x_tr, y_tr, x_va, y_va = train_val_split(x_train, y_train, cfg, 1)
    model, _ = train_blstm_lstm_model(x_tr, y_tr, x_va, y_va, cfg)
    return model.predict(x_test, verbose=0).argmax(axis=1)


def run_ablation_variants(
    cfg: Config,
    manifest: pd.DataFrame,
    loso_splits: list[dict[str, Any]] | None = None,
) -> dict[str, pd.DataFrame]:
    cfg.ensure_dirs()
    if loso_splits is None:
        loso_splits = make_loso_splits(manifest, cfg)
    variants: dict[str, dict[str, Any]] = {
        "no_vae": dict(use_cbam=True, use_encoder=False, bidirectional=True, sequence_model="lstm"),
        "no_cbam": dict(use_cbam=False, use_encoder=True, bidirectional=True, sequence_model="lstm"),
        "uni_lstm": dict(use_cbam=True, use_encoder=True, bidirectional=False, sequence_model="lstm"),
        "cnn_only": dict(use_cbam=False, use_encoder=False, bidirectional=False, sequence_model="cnn"),
    }
    results: dict[str, pd.DataFrame] = {}
    for vname, kwargs in variants.items():
        rows = []
        fold_id = 0
        for sp in loso_splits:
            subject = sp["test_subject"]
            train_rows = manifest[manifest["subject"].isin(sp["train_subjects"])]
            test_rows = manifest[manifest["subject"] == subject]
            x_train, y_train, _ = build_dataset_for_subjects(train_rows, cfg)
            x_test, y_test, _ = build_dataset_for_subjects(test_rows, cfg)
            if len(x_train) == 0 or len(x_test) == 0:
                fold_id += 1
                continue
            x_tr, y_tr, x_va, y_va = train_val_split(x_train, y_train, cfg, int(subject))
            vae = None
            if kwargs["use_encoder"]:
                vae, _, _, _ = train_vae_on_frames(x_tr, cfg, fold_id=fold_id)
            model, _ = train_classifier_model(x_tr, y_tr, x_va, y_va, cfg, vae=vae, **kwargs)
            preds = model.predict(x_test, verbose=0).argmax(axis=1)
            m = summarize_metrics(y_test, preds)
            m["subject"] = int(subject)
            m["fold_id"] = fold_id
            m["variant"] = vname
            rows.append(m)
            fold_id += 1
        results[vname] = pd.DataFrame(rows)
    ab_fold = []
    for vname, df in results.items():
        df = df.copy()
        df["variant"] = vname
        ab_fold.append(df)
    if ab_fold:
        pd.concat(ab_fold, ignore_index=True).to_csv(cfg.csv_dir / "ablation_fold_metrics.csv", index=False)
    return results


def run_sensitivity_grids(cfg: Config, manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cfg.ensure_dirs()
    subs = loso_fold_subject_ids(manifest, cfg)
    if len(subs) > 5 and cfg.quick_mode:
        subs = subs[:3]
    manifest_s = manifest[manifest["subject"].isin(subs)].copy()
    rows_latent = []
    for ld in (64, 128, 256):
        c = replace(cfg, latent_dim=ld)
        df, _, _ = run_loso_training(c, manifest_s, model_name=f"latent_{ld}")
        if len(df) == 0:
            continue
        _, summ = aggregate_fold_metrics(df)
        rows_latent.append({"latent_dim": ld, **summ})
    df_latent = pd.DataFrame(rows_latent)
    if len(df_latent):
        df_latent.to_csv(cfg.csv_dir / "sensitivity_latent_size.csv", index=False)

    rows_cbam = []
    for red, sk in ((4, 3), (8, 7), (16, 7)):
        fold_rows = []
        splits = make_loso_splits(manifest_s, cfg)
        for sp in splits:
            subject = sp["test_subject"]
            train_rows = manifest_s[manifest_s["subject"].isin(sp["train_subjects"])]
            test_rows = manifest_s[manifest_s["subject"] == subject]
            x_train, y_train, _ = build_dataset_for_subjects(train_rows, cfg)
            x_test, y_test, _ = build_dataset_for_subjects(test_rows, cfg)
            if len(x_train) == 0 or len(x_test) == 0:
                continue
            x_tr, y_tr, x_va, y_va = train_val_split(x_train, y_train, cfg, int(subject))
            vae, _, _, _ = train_vae_on_frames(x_tr, cfg)
            model, _ = train_classifier_model(
                x_tr,
                y_tr,
                x_va,
                y_va,
                cfg,
                vae=vae,
                cbam_reduction_ratio=red,
                cbam_spatial_kernel=sk,
            )
            preds = model.predict(x_test, verbose=0).argmax(axis=1)
            m = summarize_metrics(y_test, preds)
            m["subject"] = int(subject)
            fold_rows.append(m)
        if fold_rows:
            _, summ = aggregate_fold_metrics(fold_rows)
            rows_cbam.append({"reduction_ratio": red, "spatial_kernel": sk, **summ})
    df_cbam = pd.DataFrame(rows_cbam)
    if len(df_cbam):
        df_cbam.to_csv(cfg.csv_dir / "sensitivity_cbam.csv", index=False)

    rows_win = []
    for w in (5, 10, 15):
        fold_rows = []
        splits = make_loso_splits(manifest_s, cfg)
        for sp in splits:
            subject = sp["test_subject"]
            train_rows = manifest_s[manifest_s["subject"].isin(sp["train_subjects"])]
            test_rows = manifest_s[manifest_s["subject"] == subject]
            x_train, y_train, _ = build_dataset_for_subjects(train_rows, cfg, window_seconds=w, sequence_length=w)
            x_test, y_test, _ = build_dataset_for_subjects(test_rows, cfg, window_seconds=w, sequence_length=w)
            if len(x_train) == 0 or len(x_test) == 0:
                continue
            x_tr, y_tr, x_va, y_va = train_val_split(x_train, y_train, cfg, int(subject))
            vae, _, _, _ = train_vae_on_frames(x_tr, cfg)
            model, _ = train_classifier_model(x_tr, y_tr, x_va, y_va, cfg, vae=vae)
            preds = model.predict(x_test, verbose=0).argmax(axis=1)
            m = summarize_metrics(y_test, preds)
            m["subject"] = int(subject)
            fold_rows.append(m)
        if fold_rows:
            _, summ = aggregate_fold_metrics(fold_rows)
            rows_win.append({"window_length_sec": w, **summ})
    df_win = pd.DataFrame(rows_win)
    if len(df_win):
        df_win.to_csv(cfg.csv_dir / "sensitivity_window.csv", index=False)

    rows_steps = []
    for steps in (5, 10, 15):
        hop = cfg.parent_window_seconds / max(steps, 1)
        fold_rows = []
        c = replace(cfg, frame_hop_seconds=hop)
        splits = make_loso_splits(manifest_s, c)
        for sp in splits:
            subject = sp["test_subject"]
            train_rows = manifest_s[manifest_s["subject"].isin(sp["train_subjects"])]
            test_rows = manifest_s[manifest_s["subject"] == subject]
            x_train, y_train, _ = build_dataset_for_subjects(train_rows, c, sequence_length=steps)
            x_test, y_test, _ = build_dataset_for_subjects(test_rows, c, sequence_length=steps)
            if len(x_train) == 0 or len(x_test) == 0:
                continue
            x_tr, y_tr, x_va, y_va = train_val_split(x_train, y_train, c, int(subject))
            vae, _, _, _ = train_vae_on_frames(x_tr, c)
            model, _ = train_classifier_model(x_tr, y_tr, x_va, y_va, c, vae=vae)
            preds = model.predict(x_test, verbose=0).argmax(axis=1)
            m = summarize_metrics(y_test, preds)
            m["subject"] = int(subject)
            fold_rows.append(m)
        if fold_rows:
            _, summ = aggregate_fold_metrics(fold_rows)
            rows_steps.append({"sequence_steps": steps, **summ})
    df_steps = pd.DataFrame(rows_steps)
    if len(df_steps):
        df_steps.to_csv(cfg.csv_dir / "sensitivity_sequence_steps.csv", index=False)

    parts = []
    for df_, stype in (
        (df_latent, "latent"),
        (df_cbam, "cbam"),
        (df_win, "window"),
        (df_steps, "sequence_steps"),
    ):
        if len(df_):
            parts.append(df_.assign(sensitivity_type=stype))
    summary = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(summary):
        summary.to_csv(cfg.csv_dir / "sensitivity_all_summary.csv", index=False)
    return {
        "latent": df_latent,
        "cbam": df_cbam,
        "window": df_win,
        "sequence_steps": df_steps,
        "all": summary,
    }
