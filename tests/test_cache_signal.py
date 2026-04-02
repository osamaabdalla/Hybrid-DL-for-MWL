import numpy as np

from stew_mwl.config import Config
from stew_mwl.data import CHANNELS, load_preprocessed_signal


def test_preprocess_cache_roundtrip(tmp_path):
    n_samples = 500
    raw = np.random.randn(n_samples, len(CHANNELS)).astype(np.float32)
    sig_path = tmp_path / "sub01_lo.txt"
    np.savetxt(sig_path, raw)

    interim = tmp_path / "interim"
    cfg = Config(
        data_root=tmp_path,
        output_root=tmp_path / "out",
        interim_dir=interim,
        cache_preprocessed=True,
        apply_ica=False,
    )
    a = load_preprocessed_signal(sig_path, 1, "lo", cfg)
    b = load_preprocessed_signal(sig_path, 1, "lo", cfg)
    assert a.shape == b.shape == (n_samples, len(CHANNELS))
    assert np.allclose(a, b)
    assert any(interim.glob("*.npz"))


def test_preprocess_no_cache(tmp_path):
    raw = np.random.randn(200, len(CHANNELS)).astype(np.float32)
    sig_path = tmp_path / "sub02_hi.txt"
    np.savetxt(sig_path, raw)
    cfg = Config(
        data_root=tmp_path,
        output_root=tmp_path / "out",
        interim_dir=tmp_path / "interim",
        cache_preprocessed=False,
        apply_ica=False,
    )
    x = load_preprocessed_signal(sig_path, 2, "hi", cfg)
    assert x.shape == (200, len(CHANNELS))
