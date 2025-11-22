
from __future__ import annotations
import os
import re
from pathlib import Path
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
    df = pd.DataFrame(rows).sort_values(["subject", "task"]).reset_index(drop=True)
    if df.empty:
        raise FileNotFoundError(
            f"No STEW signal files found under {data_root}. Expected files like sub01_lo.txt and sub01_hi.txt."
        )
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
    if 1 <= rating <= 3:
        return "low"
    if 4 <= rating <= 6:
        return "medium"
    if 7 <= rating <= 9:
        return "high"
    raise ValueError(f"Unexpected rating {rating}; expected range 1..9")

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
    return manifest.sort_values("subject").reset_index(drop=True)

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

def preprocess_signal(x: np.ndarray, cfg: Config) -> np.ndarray:
    x = notch_filter_if_needed(x, cfg.sfreq, cfg.notch_freq)
    x = butter_bandpass_filter(x, cfg.sfreq, cfg.low_cut, cfg.high_cut)
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
