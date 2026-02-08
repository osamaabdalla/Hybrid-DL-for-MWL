import pytest

from stew_mwl.config import CLASS_NAMES, TASK_RATING_TO_CLASS, Config
from stew_mwl.data import rating_to_level


def test_class_order():
    assert CLASS_NAMES == ["BL", "LW", "MW", "HW"]


def test_rating_bins():
    assert rating_to_level(2) == "low"
    assert TASK_RATING_TO_CLASS["low"] == "LW"
    cfg = Config()
    assert cfg.class_to_id["BL"] == 0
    assert cfg.class_to_id["HW"] == 3
