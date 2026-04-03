from pathlib import Path

import pandas as pd

from stew_mwl.config import Config
from stew_mwl.reports import build_manuscript_tables


def test_build_manuscript_tables(tmp_path):
    csv = tmp_path / "csv"
    csv.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "subject": 1,
                "accuracy": 0.8,
                "macro_f1": 0.75,
                "balanced_accuracy": 0.78,
                "cohen_kappa": 0.7,
                "fold_id": 0,
                "model_name": "proposed",
            }
        ]
    ).to_csv(csv / "fold_metrics_all.csv", index=False)
    pd.DataFrame([{"model_name": "cnn", "mean_accuracy": 0.7}]).to_csv(
        csv / "baseline_comparison_summary.csv", index=False
    )

    cfg = Config(output_root=tmp_path)
    # cfg.csv_dir is tmp_path/csv — property uses output_root
    out = build_manuscript_tables(cfg)
    assert "main_proposed" in out
    assert (tmp_path / "reports" / "MANUSCRIPT_TABLES.md").is_file()
