import numpy as np
import pandas as pd

from stew_mwl.config import Config
from stew_mwl.data import make_loso_splits


def test_loso_covers_all_subjects():
    manifest = pd.DataFrame({"subject": [1, 2, 4, 7]})
    cfg = Config(loso_subjects_limit=None)
    splits = make_loso_splits(manifest, cfg)
    assert len(splits) == 4
    held = {s["test_subject"] for s in splits}
    assert held == {1, 2, 4, 7}
    for sp in splits:
        train = set(sp["train_subjects"])
        assert sp["test_subject"] not in train
        assert len(train) == 3
