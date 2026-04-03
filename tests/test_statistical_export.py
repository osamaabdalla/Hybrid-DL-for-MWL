import pandas as pd

from stew_mwl.config import Config
from stew_mwl.export import export_statistical_tests


def test_statistical_tests_csv_has_wilcoxon_columns(tmp_path):
    cfg = Config(data_root=tmp_path, output_root=tmp_path / "out")
    cfg.ensure_dirs()
    full_df = pd.DataFrame(
        {"subject": [1, 2], "accuracy": [0.9, 0.85], "macro_f1": [0.8, 0.75]}
    )
    base = {
        "m": pd.DataFrame({"subject": [1, 2], "accuracy": [0.8, 0.8], "macro_f1": [0.7, 0.7]})
    }
    export_statistical_tests(full_df, base, None, cfg)
    out = pd.read_csv(cfg.csv_dir / "statistical_tests.csv")
    assert "wilcoxon_p_value" in out.columns
    assert "wilcoxon_statistic" in out.columns
    assert "p_value" in out.columns
