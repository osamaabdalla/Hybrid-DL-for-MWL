from pathlib import Path

import pytest

from stew_mwl.yaml_loader import load_config_from_yaml


def test_load_default_yaml():
    root = Path(__file__).resolve().parent.parent
    path = root / "configs" / "default.yaml"
    if not path.is_file():
        pytest.skip("configs/default.yaml not present")
    cfg = load_config_from_yaml(path, project_root=root)
    assert cfg.config_path == path.resolve()
    assert cfg.sfreq == 128
    assert cfg.seq_len >= 1
    assert cfg.data_root == (root / "data" / "STEW").resolve()
    assert cfg.interim_dir == (root / "data" / "STEW" / "interim").resolve()
    assert cfg.processed_dir == (root / "data" / "STEW" / "processed").resolve()
    assert cfg.cbam_attention_order == "channel_spatial"
    assert cfg.cbam_enabled is True
    assert cfg.cz_proxy_reference is True


def test_yaml_loader_optional_keys(tmp_path):
    p = tmp_path / "custom.yaml"
    p.write_text(
        """
project:
  seed: 1
paths:
  raw_data_dir: data/in
  output_dir: out
  interim_data_dir: int
  processed_data_dir: proc
cache:
  sequences: true
dataset:
  sfreq: 128
  strict_signal_audit: true
  min_recording_samples: 100
preprocessing:
  reference_mode: average
features:
  method: welch
  sequence_steps: 10
  window_length_sec: 10
vae: {}
cbam:
  enabled: false
  attention_order: spatial_channel
model: {}
training: {}
reproducibility: {}
""".strip(),
        encoding="utf-8",
    )
    cfg = load_config_from_yaml(p, project_root=tmp_path)
    assert cfg.reference_mode == "average"
    assert cfg.cache_sequences is True
    assert cfg.strict_signal_audit is True
    assert cfg.min_recording_samples == 100
    assert cfg.cbam_attention_order == "spatial_channel"
    assert cfg.cbam_enabled is False
    assert cfg.processed_dir == (tmp_path / "proc").resolve()
