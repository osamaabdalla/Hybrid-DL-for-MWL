
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata
from scipy.signal import welch
import mne
from mne.time_frequency import tfr_array_morlet

from .config import CHANNELS, BANDS, RGB_BANDS, Config


def _integrate_band(psd_slice: np.ndarray, freqs_slice: np.ndarray) -> np.ndarray:
    fn = getattr(np, "trapezoid", None)
    if fn is not None:
        return fn(psd_slice, freqs_slice, axis=0).astype(np.float32)
    return np.trapz(psd_slice, freqs_slice, axis=0).astype(np.float32)


def _channel_positions():
    info = mne.create_info(CHANNELS, sfreq=128, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage)
    pos = np.array([info.get_montage().get_positions()["ch_pos"][ch] for ch in CHANNELS], dtype=np.float32)
    return pos[:, :2]

CH_POS_2D = _channel_positions()

def bandpower_morlet_epoch(epoch: np.ndarray, sfreq: int, bands: dict[str, tuple[float, float]]) -> dict[str, np.ndarray]:
    """Band power per channel via Morlet TFR (MNE), averaged over time and frequencies in each band."""
    x = np.transpose(epoch.astype(np.float64), (1, 0))[np.newaxis, ...]
    out: dict[str, np.ndarray] = {}
    for name, (fmin, fmax) in bands.items():
        n_freq = max(2, min(8, int((fmax - fmin) * 2)))
        freqs = np.linspace(fmin + 0.25, fmax - 0.25, num=n_freq)
        n_cycles = np.clip(freqs / 3.0, 2.0, 12.0)
        power = tfr_array_morlet(
            x,
            sfreq=float(sfreq),
            freqs=freqs,
            n_cycles=n_cycles,
            output="power",
            verbose="ERROR",
        )
        p = np.mean(power, axis=(2, 3))[0].astype(np.float32)
        out[name] = p
    return out


def bandpower_epoch(epoch: np.ndarray, sfreq: int, bands: dict[str, tuple[float, float]]) -> dict[str, np.ndarray]:
    # epoch: [samples, channels]
    freqs, psd = welch(epoch, fs=sfreq, nperseg=min(epoch.shape[0], sfreq), axis=0)
    out = {}
    for name, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs < fmax)
        bp = _integrate_band(psd[mask], freqs[mask])
        out[name] = bp
    return out

def topomap_from_band_values(values: np.ndarray, image_h: int, image_w: int) -> np.ndarray:
    x = CH_POS_2D[:, 0]
    y = CH_POS_2D[:, 1]
    grid_x, grid_y = np.mgrid[x.min():x.max():complex(image_h), y.min():y.max():complex(image_w)]
    grid = griddata((x, y), values, (grid_x, grid_y), method="cubic", fill_value=float(np.mean(values)))
    grid = np.nan_to_num(grid)
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    return grid.astype(np.float32)

def epoch_to_rgb_topomap(epoch: np.ndarray, cfg: Config) -> np.ndarray:
    method = (cfg.feature_method or "welch").lower()
    if method == "morlet":
        bp = bandpower_morlet_epoch(epoch, cfg.sfreq, BANDS)
    else:
        bp = bandpower_epoch(epoch, cfg.sfreq, BANDS)
    channels = []
    for band in RGB_BANDS:
        channels.append(topomap_from_band_values(bp[band], cfg.image_h, cfg.image_w))
    img = np.stack(channels, axis=-1)
    return img.astype(np.float32)

def build_sequence_images(signal_array: np.ndarray, cfg: Config, window_seconds: int | None = None, sequence_length: int | None = None):
    # Build parent windows and, inside each parent window, generate sequence_length frames.
    if window_seconds is None:
        window_seconds = cfg.parent_window_seconds
    if sequence_length is None:
        sequence_length = int(window_seconds / cfg.frame_hop_seconds)

    parent_samples = int(window_seconds * cfg.sfreq)
    hop_samples = parent_samples  # non-overlapping parent windows
    frame_win = int(cfg.epoch_seconds * cfg.sfreq)
    frame_hop = int(cfg.frame_hop_seconds * cfg.sfreq)

    seqs = []
    starts = np.arange(0, signal_array.shape[0] - parent_samples + 1, hop_samples, dtype=int)
    for s in starts:
        parent = signal_array[s:s+parent_samples]
        frame_starts = np.arange(0, parent_samples - frame_win + 1, frame_hop, dtype=int)
        frames = []
        for fs in frame_starts[:sequence_length]:
            epoch = parent[fs:fs+frame_win]
            frames.append(epoch_to_rgb_topomap(epoch, cfg))
        if len(frames) == sequence_length:
            seqs.append(np.stack(frames, axis=0))
    if not seqs:
        return np.empty((0, sequence_length, cfg.image_h, cfg.image_w, len(RGB_BANDS)), dtype=np.float32)
    return np.stack(seqs, axis=0).astype(np.float32)


def _sequence_cache_meta(
    path: Path,
    cfg: Config,
    subject: int,
    side: str,
    window_seconds: int | None,
    sequence_length: int | None,
) -> tuple[dict, str]:
    path = Path(path)
    w = int(window_seconds if window_seconds is not None else cfg.parent_window_seconds)
    sl = int(sequence_length if sequence_length is not None else cfg.seq_len)
    meta = {
        "mtime_ns": int(path.stat().st_mtime_ns),
        "subject": subject,
        "side": side,
        "window_sec": w,
        "seq_len": sl,
        "feature_method": cfg.feature_method,
        "parent_window_seconds": cfg.parent_window_seconds,
        "frame_hop_seconds": cfg.frame_hop_seconds,
        "epoch_seconds": cfg.epoch_seconds,
        "sfreq": cfg.sfreq,
        "image_h": cfg.image_h,
        "image_w": cfg.image_w,
        "reference_mode": getattr(cfg, "reference_mode", "none"),
        "cz_proxy": bool(getattr(cfg, "cz_proxy_reference", False)),
        "apply_ica": cfg.apply_ica,
        "low_cut": cfg.low_cut,
        "high_cut": cfg.high_cut,
        "notch": cfg.notch_freq,
    }
    h = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:28]
    return meta, h


def build_sequence_images_cached(
    signal_array: np.ndarray,
    path: Path,
    subject: int,
    side: str,
    cfg: Config,
    window_seconds: int | None = None,
    sequence_length: int | None = None,
) -> np.ndarray:
    """Same as `build_sequence_images` but optionally persist under `cfg.processed_dir`."""
    if not getattr(cfg, "cache_sequences", False):
        return build_sequence_images(signal_array, cfg, window_seconds=window_seconds, sequence_length=sequence_length)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    meta, h = _sequence_cache_meta(path, cfg, subject, side, window_seconds, sequence_length)
    fp = cfg.processed_dir / f"seq_{subject:02d}_{side}_{h}.npz"
    if fp.is_file():
        z = np.load(fp, allow_pickle=False)
        if int(z["mtime_ns"]) == meta["mtime_ns"]:
            return np.asarray(z["data"], dtype=np.float32)
    out = build_sequence_images(signal_array, cfg, window_seconds=window_seconds, sequence_length=sequence_length)
    np.savez_compressed(fp, data=out, mtime_ns=meta["mtime_ns"])
    return out


def _psd_feature_vector_for_parent(parent: np.ndarray, cfg: Config) -> np.ndarray:
    """Welch PSD integrated over theta, alpha, beta per channel → 14×3 vector (PSD–SVM baseline)."""
    freqs, psd = welch(parent, fs=cfg.sfreq, nperseg=min(parent.shape[0], cfg.sfreq), axis=0)
    feats: list[np.ndarray] = []
    for band in RGB_BANDS:
        fmin, fmax = BANDS[band]
        mask = (freqs >= fmin) & (freqs < fmax)
        bp = _integrate_band(psd[mask], freqs[mask])
        feats.append(bp)
    return np.concatenate(feats, axis=0)


def build_psd_sequence_features(
    signal_array: np.ndarray,
    cfg: Config,
    window_seconds: int | None = None,
    sequence_length: int | None = None,
) -> np.ndarray:
    """
    One PSD feature vector per classification sequence window (aligned with topomap sequence count).
    Uses the full parent window raw EEG (same segmentation as `build_sequence_images`).
    """
    if window_seconds is None:
        window_seconds = cfg.parent_window_seconds
    if sequence_length is None:
        sequence_length = int(window_seconds / cfg.frame_hop_seconds)

    parent_samples = int(window_seconds * cfg.sfreq)
    hop_samples = parent_samples
    frame_win = int(cfg.epoch_seconds * cfg.sfreq)
    frame_hop = int(cfg.frame_hop_seconds * cfg.sfreq)

    vecs = []
    starts = np.arange(0, signal_array.shape[0] - parent_samples + 1, hop_samples, dtype=int)
    for s in starts:
        parent = signal_array[s : s + parent_samples]
        frame_starts = np.arange(0, parent_samples - frame_win + 1, frame_hop, dtype=int)
        if len(frame_starts) < sequence_length:
            continue
        vecs.append(_psd_feature_vector_for_parent(parent, cfg))
    if not vecs:
        feat_dim = len(CHANNELS) * len(RGB_BANDS)
        return np.empty((0, feat_dim), dtype=np.float32)
    return np.stack(vecs, axis=0).astype(np.float32)
