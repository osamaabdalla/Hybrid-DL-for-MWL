import pandas as pd

from stew_mwl.config import Config
from stew_mwl.export import export_ablation_summary


def test_ablation_summary_prepends_full_proposed(tmp_path):
    cfg = Config(data_root=tmp_path, output_root=tmp_path / "out")
    cfg.ensure_dirs()
    full_df = pd.DataFrame(
        [
            {"subject": 1, "accuracy": 0.8, "macro_f1": 0.7, "balanced_accuracy": 0.75, "cohen_kappa": 0.6},
            {"subject": 2, "accuracy": 0.82, "macro_f1": 0.72, "balanced_accuracy": 0.76, "cohen_kappa": 0.62},
        ]
    )
    ab = {
        "no_vae": pd.DataFrame(
            [
                {
                    "subject": 1,
                    "accuracy": 0.7,
                    "macro_f1": 0.6,
                    "balanced_accuracy": 0.65,
                    "cohen_kappa": 0.5,
                },
            ]
        )
    }
    export_ablation_summary(ab, full_df, cfg)
    out = pd.read_csv(cfg.csv_dir / "ablation_summary.csv")
    assert out["variant"].iloc[0] == "full_proposed"
    assert pd.isna(out["p_value_vs_full"].iloc[0])
    assert "no_vae" in set(out["variant"])
