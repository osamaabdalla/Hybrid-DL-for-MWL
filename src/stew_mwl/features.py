
from __future__ import annotations
import numpy as np
from scipy.interpolate import griddata
from scipy.signal import welch
import mne

from .config import CHANNELS, BANDS, RGB_BANDS, Config

def _channel_positions():
    info = mne.create_info(CHANNELS, sfreq=128, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage)
    pos = np.array([info.get_montage().get_positions()["ch_pos"][ch] for ch in CHANNELS], dtype=np.float32)
    return pos[:, :2]

CH_POS_2D = _channel_positions()

def bandpower_epoch(epoch: np.ndarray, sfreq: int, bands: dict[str, tuple[float, float]]) -> dict[str, np.ndarray]:
    # epoch: [samples, channels]
    freqs, psd = welch(epoch, fs=sfreq, nperseg=min(epoch.shape[0], sfreq), axis=0)
    out = {}
    for name, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs < fmax)
        bp = np.trapz(psd[mask], freqs[mask], axis=0).astype(np.float32)
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
