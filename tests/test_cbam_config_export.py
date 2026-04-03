import pandas as pd

from stew_mwl.config import Config
from stew_mwl.export import export_cbam_config_results


def test_export_cbam_config_results_active_and_sweep(tmp_path):
    cfg = Config(
        data_root=tmp_path,
        output_root=tmp_path / "out",
        cbam_enabled=True,
        cbam_reduction_ratio=8,
        cbam_spatial_kernel=7,
        cbam_attention_order="parallel",
    )
    cfg.ensure_dirs()
    p = export_cbam_config_results(cfg)
    assert p.is_file()
    df = pd.read_csv(p)
    assert len(df) == 1
    assert df["config_label"].iloc[0] == "active_yaml"
    assert df["attention_order"].iloc[0] == "parallel"

    sens = pd.DataFrame(
        [
            {
                "reduction_ratio": 4,
                "spatial_kernel": 3,
                "accuracy": 0.5,
                "macro_f1": 0.4,
                "balanced_accuracy": 0.45,
                "cohen_kappa": 0.3,
                "std_accuracy": 0.1,
                "std_macro_f1": 0.05,
            }
        ]
    )
    export_cbam_config_results(cfg, sensitivity_cbam_df=sens)
    df2 = pd.read_csv(cfg.csv_dir / "cbam_config_results.csv")
    assert len(df2) == 2
    assert df2["config_label"].iloc[1].startswith("sweep_r4")
