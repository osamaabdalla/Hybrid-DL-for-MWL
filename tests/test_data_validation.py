import numpy as np
import pandas as pd

from stew_mwl.config import CHANNELS, Config
from stew_mwl.data import validate_stew_dataset


def test_validate_detects_duplicate_signal_files(tmp_path):
    raw = np.random.randn(100, len(CHANNELS)).astype(np.float32)
    np.savetxt(tmp_path / "sub01_lo.txt", raw)
    sub = tmp_path / "copy"
    sub.mkdir()
    np.savetxt(sub / "sub01_lo.txt", raw)

    cfg = Config(data_root=tmp_path)
    manifest = pd.DataFrame(
        {
            "subject": [1],
            "lo_path": [tmp_path / "sub01_lo.txt"],
            "hi_path": [tmp_path / "sub01_hi.txt"],
        }
    )
    issues = validate_stew_dataset(cfg, manifest)
    assert any("Duplicate signal files" in x for x in issues)


def test_validate_manifest_duplicate_subjects(tmp_path):
    cfg = Config(data_root=tmp_path)
    manifest = pd.DataFrame(
        {
            "subject": [1, 1],
            "lo_path": ["a", "b"],
            "hi_path": ["c", "d"],
        }
    )
    issues = validate_stew_dataset(cfg, manifest)
    assert any("duplicate subject rows" in x.lower() for x in issues)


def test_verify_stew_conventions_sfreq(tmp_path):
    cfg = Config(data_root=tmp_path, sfreq=256, verify_stew_conventions=True)
    manifest = pd.DataFrame({"subject": [1], "lo_path": ["a"], "hi_path": ["b"]})
    issues = validate_stew_dataset(cfg, manifest)
    assert any("128" in x for x in issues)
