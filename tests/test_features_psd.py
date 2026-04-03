import numpy as np

from stew_mwl.config import CHANNELS, Config, RGB_BANDS
from stew_mwl.features import build_psd_sequence_features


def test_build_psd_sequence_features_non_empty_and_dim():
    cfg = Config(
        sfreq=128,
        parent_window_seconds=10,
        epoch_seconds=2.0,
        frame_hop_seconds=2.0,
    )
    n = 8000
    x = np.random.randn(n, len(CHANNELS)).astype(np.float32)
    out = build_psd_sequence_features(x, cfg)
    fdim = len(CHANNELS) * len(RGB_BANDS)
    assert out.shape[1] == fdim
    assert out.shape[0] >= 1
