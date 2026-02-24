import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from stew_mwl.config import CLASS_NAMES
from stew_mwl.eval import aggregate_fold_metrics, paired_ttest_vs_full


def test_aggregate_empty():
    df, s = aggregate_fold_metrics([])
    assert np.isnan(s["accuracy"])


def test_paired_ttest():
    a = pd.DataFrame({"subject": [1, 2], "accuracy": [0.8, 0.9]})
    b = pd.DataFrame({"subject": [1, 2], "accuracy": [0.7, 0.85]})
    p = paired_ttest_vs_full(a, b, "accuracy")
    assert 0 <= p <= 1


def test_confusion_fixed_labels():
    yt = [0, 1, 2, 3]
    yp = [0, 1, 2, 2]
    cm = confusion_matrix(yt, yp, labels=list(range(len(CLASS_NAMES))))
    assert cm.shape == (4, 4)
