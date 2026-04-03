import pandas as pd

from stew_mwl.config import Config
from stew_mwl.export import export_experiment_registry


def test_export_experiment_registry_columns(tmp_path):
    cfg = Config(
        data_root=tmp_path,
        output_root=tmp_path / "out",
        reference_mode="average",
        cache_sequences=True,
        feature_method="welch",
    )
    cfg.ensure_dirs()
    export_experiment_registry(cfg, notes="test")
    df = pd.read_csv(cfg.csv_dir / "experiment_registry.csv")
    for col in (
        "output_root",
        "data_root",
        "feature_method",
        "latent_dim",
        "reference_mode",
        "cbam_enabled",
        "cbam_attention_order",
        "cache_preprocessed",
        "cache_sequences",
        "notes",
    ):
        assert col in df.columns
    assert df["reference_mode"].iloc[0] == "average"
    assert bool(df["cache_sequences"].iloc[0]) is True
