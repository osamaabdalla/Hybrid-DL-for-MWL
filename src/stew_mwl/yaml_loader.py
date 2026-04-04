"""Load `Config` from YAML files under `configs/`."""

from __future__ import annotations
from pathlib import Path
from typing import Any

import yaml

from .config import Config


def load_config_from_yaml(path: Path | str, project_root: Path | None = None) -> Config:
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as f:
        y: dict[str, Any] = yaml.safe_load(f) or {}

    project = y.get("project", {}) or {}
    paths = y.get("paths", {}) or {}
    ds = y.get("dataset", {}) or {}
    prep = y.get("preprocessing", {}) or {}
    feat = y.get("features", {}) or {}
    vae = y.get("vae", {}) or {}
    cbam = y.get("cbam", {}) or {}
    model = y.get("model", {}) or {}
    train_y = y.get("training", {}) or {}
    rep = y.get("reproducibility", {}) or {}

    raw = paths.get("raw_data_dir", "data/STEW")
    out = paths.get("output_dir", "outputs")
    interim = paths.get("interim_data_dir", "data/STEW/interim")
    proc = paths.get("processed_data_dir", "data/STEW/processed")
    cache = y.get("cache", {}) or {}
    parent_window = float(feat.get("window_length_sec", 10))
    seq_steps = int(feat.get("sequence_steps", 10))
    hop = parent_window / max(seq_steps, 1)

    loso_lim = rep.get("loso_subjects_limit")
    if loso_lim is not None:
        loso_lim = int(loso_lim)

    notch = prep.get("notch_freq", 50)
    if notch is not None:
        notch = float(notch)

    ref_mode = str(prep.get("reference_mode", "none")).lower().strip()
    if ref_mode not in ("none", "average", "cz_proxy"):
        ref_mode = "none"

    cbam_order = str(cbam.get("attention_order", "channel_spatial")).lower().strip()
    if cbam_order not in ("channel_spatial", "spatial_channel", "parallel"):
        cbam_order = "channel_spatial"

    cache_seq = bool(cache.get("sequences", cache.get("cache_sequences", False)))

    cfg = Config(
        data_root=Path(raw),
        output_root=Path(out),
        interim_dir=Path(interim),
        processed_dir=Path(proc),
        cache_preprocessed=bool(cache.get("preprocessed", False)),
        cache_sequences=cache_seq,
        sfreq=int(ds.get("sfreq", 128)),
        image_h=int(feat.get("image_height", 80)),
        image_w=int(feat.get("image_width", 60)),
        epoch_seconds=float(feat.get("epoch_seconds", 2.0)),
        parent_window_seconds=int(round(parent_window)),
        frame_hop_seconds=float(hop),
        feature_method=str(feat.get("method", "welch")).lower(),
        apply_ica=bool(prep.get("use_ica", False)),
        low_cut=float(prep.get("bandpass_low", 1.0)),
        high_cut=float(prep.get("bandpass_high", 40.0)),
        notch_freq=notch,
        reference_mode=ref_mode,
        cz_proxy_reference=bool(prep.get("cz_proxy_reference", False)),
        latent_dim=int(vae.get("latent_dim", 128)),
        vae_epochs=int(vae.get("epochs", 20)),
        vae_val_fraction=float(vae.get("val_fraction", 0.1)),
        clf_epochs=int(train_y.get("epochs", 35)),
        batch_size=int(train_y.get("batch_size", 32)),
        learning_rate=float(train_y.get("learning_rate", 1e-3)),
        lr_schedule=str(train_y.get("lr_schedule", "exponential")).lower(),
        early_stopping_monitor=str(train_y.get("early_stopping_monitor", "val_macro_f1")),
        early_stopping_patience=int(train_y.get("early_stopping_patience", 10)),
        dropout=float(model.get("dropout", 0.3)),
        blstm_units=int(model.get("blstm_units", 20)),
        cbam_reduction_ratio=int(cbam.get("reduction_ratio", 8)),
        cbam_spatial_kernel=int(cbam.get("spatial_kernel_size", 7)),
        cbam_enabled=bool(cbam.get("enabled", True)),
        cbam_attention_order=cbam_order,
        loso_subjects_limit=loso_lim,
        quick_mode=bool(rep.get("quick_mode", False)),
        run_sensitivity=bool(rep.get("run_sensitivity", False)),
        strict_subject_count=bool(rep.get("strict_subject_count", False)),
        strict_signal_audit=bool(ds.get("strict_signal_audit", False)),
        verify_stew_conventions=bool(ds.get("verify_stew_conventions", False)),
        min_recording_samples=int(ds.get("min_recording_samples", 0)),
        expected_n_subjects=int(rep.get("expected_n_subjects", 48)),
        seed=int(project.get("seed", 42)),
        config_path=path,
    )
    if project_root is not None:
        root = project_root.resolve()
        if not cfg.data_root.is_absolute():
            cfg.data_root = (root / cfg.data_root).resolve()
        if not cfg.output_root.is_absolute():
            cfg.output_root = (root / cfg.output_root).resolve()
        if not cfg.interim_dir.is_absolute():
            cfg.interim_dir = (root / cfg.interim_dir).resolve()
        if not cfg.processed_dir.is_absolute():
            cfg.processed_dir = (root / cfg.processed_dir).resolve()
    return cfg
