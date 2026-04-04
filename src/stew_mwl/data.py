
from __future__ import annotations
import hashlib
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal
import mne

from .config import CHANNELS, Config

_SUBJECT_RE = re.compile(r"sub(\d+)_([a-z]+)\.txt$", re.IGNORECASE)

def set_global_determinism(seed: int = 42) -> None:
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except Exception:
        pass

def discover_signal_files(data_root: Path) -> pd.DataFrame:
    rows = []
    for path in data_root.rglob("*.txt"):
        m = _SUBJECT_RE.search(path.name)
        if not m:
            continue
        subject = int(m.group(1))
        task = m.group(2).lower()
        rows.append({"subject": subject, "task": task, "path": path})
    if not rows:
        raise FileNotFoundError(
            f"No STEW signal files found under {data_root}. Expected files like sub01_lo.txt and sub01_hi.txt."
        )
    df = pd.DataFrame(rows).sort_values(["subject", "task"]).reset_index(drop=True)
    return df

def read_signal_txt(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array in {path}, got shape {arr.shape}")
    if arr.shape[1] != len(CHANNELS):
        raise ValueError(
            f"{path} has {arr.shape[1]} columns; expected {len(CHANNELS)} EEG channels."
        )
    return arr

def detect_ratings_file(data_root: Path) -> Path | None:
    candidates = []
    for p in data_root.rglob("*"):
        if p.is_file() and re.search(r"rating", p.name, flags=re.IGNORECASE):
            candidates.append(p)
    return sorted(candidates)[0] if candidates else None

def parse_ratings_file(path: Path) -> pd.DataFrame:
    # Flexible parser for CSV/TXT/TSV-like files with subject and rating columns
    ext = path.suffix.lower()
    seps = [",", "\t", None]
    frames = []
    for sep in seps:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if df.shape[1] >= 2:
                frames.append(df)
        except Exception:
            pass
    if not frames:
        raise ValueError(f"Could not parse ratings file: {path}")
    df = max(frames, key=lambda x: x.shape[1]).copy()

    lower_map = {c.lower().strip(): c for c in df.columns}
    subject_col = None
    rating_col = None
    for c in df.columns:
        lc = c.lower().strip()
        if any(k in lc for k in ["subject", "sub", "participant"]):
            subject_col = c
        if any(k in lc for k in ["rating", "score", "workload"]):
            rating_col = c

    if subject_col is None:
        subject_col = df.columns[0]
    if rating_col is None:
        rating_col = df.columns[1]

    out = df[[subject_col, rating_col]].rename(columns={subject_col: "subject", rating_col: "rating"}).copy()
    out["subject"] = (
        out["subject"].astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(float)
        .astype("Int64")
    )
    out["rating"] = pd.to_numeric(out["rating"], errors="coerce")
    out = out.dropna().astype({"subject": int})
    out = out.groupby("subject", as_index=False)["rating"].mean()
    return out

def rating_to_level(rating: float) -> str:
    """Map subjective rating (1–9) to manuscript class codes LW / MW / HW."""
    if 1 <= rating <= 3:
        return "LW"
    if 4 <= rating <= 6:
        return "MW"
    if 7 <= rating <= 9:
        return "HW"
    raise ValueError(f"Unexpected rating {rating}; expected range 1..9")


def _validate_signal_file_uniqueness(cfg: Config) -> list[str]:
    """Ensure at most one STEW text file per (subject, task) under data_root."""
    issues: list[str] = []
    try:
        files = discover_signal_files(cfg.data_root)
    except FileNotFoundError:
        return issues  # no STEW-like files yet; other checks / manifest will surface missing data
    for (sub, task), g in files.groupby(["subject", "task"]):
        if len(g) <= 1:
            continue
        paths = "; ".join(sorted(str(p) for p in g["path"].tolist()))
        issues.append(f"Duplicate signal files for subject {sub} task {task} ({len(g)} paths): {paths}")
    return issues


def validate_stew_dataset(cfg: Config, manifest: pd.DataFrame) -> list[str]:
    """Return human-readable issues; empty list means checks passed."""
    issues: list[str] = []
    issues.extend(_validate_signal_file_uniqueness(cfg))
    if getattr(cfg, "verify_stew_conventions", False) and int(cfg.sfreq) != 128:
        issues.append(
            f"dataset.sfreq={cfg.sfreq} but STEW is conventionally 128 Hz "
            "(set verify_stew_conventions: false if intentional)."
        )
    subjects = sorted(manifest["subject"].unique().tolist())
    n = len(subjects)
    if cfg.strict_subject_count and n != cfg.expected_n_subjects:
        issues.append(f"Subject count {n} != expected {cfg.expected_n_subjects} (strict_subject_count).")
    for col in ("lo_path", "hi_path"):
        if col not in manifest.columns:
            issues.append(f"Manifest missing column {col}.")
            return issues
    missing = manifest[["lo_path", "hi_path"]].isna().any(axis=1)
    if missing.any():
        issues.append(f"{int(missing.sum())} manifest row(s) missing lo/hi paths.")
    if manifest["subject"].duplicated().any():
        issues.append("Manifest has duplicate subject rows (expected one row per subject).")
    if getattr(cfg, "strict_signal_audit", False):
        issues.extend(_audit_signal_files(cfg, manifest))
    return issues


def _audit_signal_files(cfg: Config, manifest: pd.DataFrame) -> list[str]:
    """Load each manifest path: channel count, optional minimum duration in samples."""
    issues: list[str] = []
    min_s = int(getattr(cfg, "min_recording_samples", 0) or 0)
    for _, row in manifest.iterrows():
        for col in ("lo_path", "hi_path"):
            p = Path(row[col])
            try:
                x = read_signal_txt(p)
            except Exception as err:
                issues.append(f"{p}: {err}")
                continue
            if x.shape[1] != len(CHANNELS):
                issues.append(f"{p}: expected {len(CHANNELS)} channels, got {x.shape[1]}.")
            if min_s > 0 and x.shape[0] < min_s:
                issues.append(f"{p}: length {x.shape[0]} < min_recording_samples={min_s}.")
    return issues

def build_subject_manifest(cfg: Config) -> pd.DataFrame:
    files = discover_signal_files(cfg.data_root)
    ratings_path = detect_ratings_file(cfg.data_root)
    if ratings_path is None:
        raise FileNotFoundError(
            "Could not find a ratings file in the STEW directory. "
            "Please place the official ratings file under data/STEW/."
        )
    ratings = parse_ratings_file(ratings_path)
    hi = files[files["task"].eq("hi")].rename(columns={"path": "hi_path"})
    lo = files[files["task"].eq("lo")].rename(columns={"path": "lo_path"})
    manifest = lo.merge(hi[["subject","hi_path"]], on="subject", how="inner")
    manifest = manifest.merge(ratings, on="subject", how="inner")
    manifest["task_label"] = manifest["rating"].apply(rating_to_level)
    manifest["task_class"] = manifest["task_label"]
    return manifest.sort_values("subject").reset_index(drop=True)


def loso_fold_subject_ids(manifest: pd.DataFrame, cfg: Config) -> list[int]:
    subs = sorted(manifest["subject"].unique().tolist())
    if cfg.loso_subjects_limit is not None:
        subs = subs[: int(cfg.loso_subjects_limit)]
    return subs


def make_loso_splits(manifest: pd.DataFrame, cfg: Config) -> list[dict[str, Any]]:
    subjects = loso_fold_subject_ids(manifest, cfg)
    splits: list[dict[str, Any]] = []
    for test_subject in subjects:
        train_subjects = [s for s in subjects if s != test_subject]
        splits.append({"test_subject": test_subject, "train_subjects": train_subjects})
    return splits

def butter_bandpass_filter(x: np.ndarray, sfreq: int, low: float, high: float, order: int = 5) -> np.ndarray:
    sos = signal.butter(order, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)

def notch_filter_if_needed(x: np.ndarray, sfreq: int, notch_freq: float | None) -> np.ndarray:
    if notch_freq is None:
        return x
    b, a = signal.iirnotch(notch_freq, Q=30, fs=sfreq)
    return signal.filtfilt(b, a, x, axis=0)

def apply_ica_if_enabled(x: np.ndarray, sfreq: int, enabled: bool) -> np.ndarray:
    if not enabled:
        return x
    info = mne.create_info(ch_names=CHANNELS, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(x.T, info, verbose="ERROR")
    raw.set_montage("standard_1020", verbose="ERROR")
    raw.filter(1.0, 40.0, verbose="ERROR")
    ica = mne.preprocessing.ICA(n_components=min(12, len(CHANNELS)), random_state=42, max_iter="auto", verbose="ERROR")
    ica.fit(raw, verbose="ERROR")
    cleaned = raw.copy()
    ica.apply(cleaned, verbose="ERROR")
    return cleaned.get_data().T.astype(np.float32)

def _preprocess_cache_path(cfg: Config, path: Path, subject: int, side: str) -> Path:
    h = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
    return cfg.interim_dir / f"sub{subject:02d}_{side}_{h}.npz"


def load_preprocessed_signal(path: Path, subject: int, side: str, cfg: Config) -> np.ndarray:
    """
    Read raw STEW text, apply full preprocessing, optionally cache under `cfg.interim_dir`
    (invalidated when source file mtime changes).
    """
    path = Path(path)
    if not cfg.cache_preprocessed:
        return preprocess_signal(read_signal_txt(path), cfg)

    cfg.interim_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _preprocess_cache_path(cfg, path, subject, side)
    st = path.stat()
    mtime_ns = int(st.st_mtime_ns)
    if cache_file.is_file():
        z = np.load(cache_file, allow_pickle=False)
        if int(z["mtime_ns"]) == mtime_ns:
            return np.asarray(z["data"], dtype=np.float32)
    x = preprocess_signal(read_signal_txt(path), cfg)
    np.savez_compressed(cache_file, data=x, mtime_ns=mtime_ns)
    return x


def preprocess_signal(x: np.ndarray, cfg: Config) -> np.ndarray:
    x = notch_filter_if_needed(x, cfg.sfreq, cfg.notch_freq)
    x = butter_bandpass_filter(x, cfg.sfreq, cfg.low_cut, cfg.high_cut)
    ref_mode = (getattr(cfg, "reference_mode", None) or "none").lower()
    if ref_mode == "none" and getattr(cfg, "cz_proxy_reference", False):
        ref_mode = "cz_proxy"
    if ref_mode == "average":
        x = x - np.mean(x, axis=1, keepdims=True)
    elif ref_mode == "cz_proxy":
        i0 = CHANNELS.index("AF3")
        i1 = CHANNELS.index("AF4")
        ref = (x[:, i0 : i0 + 1] + x[:, i1 : i1 + 1]) / 2.0
        x = x - ref
    x = apply_ica_if_enabled(x, cfg.sfreq, cfg.apply_ica)
    x = x - x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True) + 1e-8
    return (x / x_std).astype(np.float32)

def sliding_windows(x: np.ndarray, sfreq: int, window_seconds: float, hop_seconds: float) -> np.ndarray:
    win = int(window_seconds * sfreq)
    hop = int(hop_seconds * sfreq)
    if x.shape[0] < win:
        return np.empty((0, win, x.shape[1]), dtype=np.float32)
    starts = np.arange(0, x.shape[0] - win + 1, hop, dtype=int)
    return np.stack([x[s:s+win] for s in starts], axis=0).astype(np.float32)
