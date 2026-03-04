from pathlib import Path

import pandas as pd

from stew_mwl.config import Config
from stew_mwl.export import export_dataset_manifest


def test_export_manifest_csv(tmp_path):
    cfg = Config(data_root=tmp_path, output_root=tmp_path / "out")
    cfg.ensure_dirs()
    manifest = pd.DataFrame(
        {
            "subject": [1],
            "lo_path": ["x"],
            "hi_path": ["y"],
            "rating": [5.0],
            "task_label": ["medium"],
            "task_class": ["MW"],
        }
    )
    p = export_dataset_manifest(manifest, cfg)
    assert p.is_file()
    df = pd.read_csv(p)
    assert "task_class" in df.columns or "final_label_task" in df.columns
