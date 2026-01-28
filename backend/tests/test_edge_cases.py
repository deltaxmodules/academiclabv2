import numpy as np
import pandas as pd

from backend.main import _compute_stats


def test_empty_dataset():
    df = pd.DataFrame()
    stats = _compute_stats(df)
    assert stats["rows"] == 0
    assert stats["columns"] == 0


def test_single_row_dataset():
    df = pd.DataFrame({"col": [None]})
    stats = _compute_stats(df)
    assert stats["rows"] == 1
    assert stats["missing_percentage"]["col"] == 100.0
    assert stats["missing_types"]["col"] == "MNAR"


def test_single_column_dataset():
    df = pd.DataFrame({"col": [None, 1, 2]})
    stats = _compute_stats(df)
    assert "col" in stats["missing_types"]


def test_all_missing_column():
    df = pd.DataFrame({"col": [None] * 100})
    stats = _compute_stats(df)
    assert stats["missing_types"]["col"] == "MNAR"


def test_completely_present_column():
    df = pd.DataFrame({"col": list(range(100))})
    stats = _compute_stats(df)
    assert "col" not in stats["missing_types"]


def test_mixed_types():
    df = pd.DataFrame(
        {
            "age": [25, None, 35, None, 45],
            "role": ["A", "B", "A", "B", None],
            "date": ["2020-01-01", None, "2020-01-03", None, "2020-01-05"],
        }
    )
    stats = _compute_stats(df)
    assert "age" in stats["missing_types"]
    assert "role" in stats["missing_types"]


def test_time_series_pattern():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=100),
            "value": [np.nan if i % 7 < 2 else i for i in range(100)],
        }
    )
    stats = _compute_stats(df)
    assert "value" in stats["missing_types"]
