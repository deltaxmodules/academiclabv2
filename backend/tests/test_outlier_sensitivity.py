import numpy as np
import pandas as pd

from backend.main import _compute_stats
from backend.state import create_initial_state


def _make_df():
    # Simple numeric column with two extreme values
    return pd.DataFrame({"value": [10, 11, 12, 13, 14, 15, 100, 120]})


def test_zscore_sensitivity_increases_outliers():
    df = _make_df()
    stats_s1 = _compute_stats(df, thresholds={"outlier_sensitivity": 1.0})
    stats_s3 = _compute_stats(df, thresholds={"outlier_sensitivity": 3.0})
    stats_s5 = _compute_stats(df, thresholds={"outlier_sensitivity": 5.0})

    count_s1 = stats_s1["outliers"]["value"]["count"]
    count_s3 = stats_s3["outliers"]["value"]["count"]
    count_s5 = stats_s5["outliers"]["value"]["count"]

    assert count_s1 >= count_s3 >= count_s5


def test_context_overrides_sensitivity():
    df = _make_df()
    context = {"value": {"min_expected": 9, "max_expected": 16}}
    stats_context = _compute_stats(
        df,
        context_by_column=context,
        thresholds={"outlier_sensitivity": 1.0},
    )
    # Outliers should be only outside 9-16 (i.e., 100 and 120)
    assert stats_context["outliers"]["value"]["count"] == 2


def test_context_warning_no_overlap():
    df = _make_df()
    context = {"value": {"min_expected": 150, "max_expected": 200}}
    stats = _compute_stats(df, context_by_column=context)
    warnings = stats.get("outlier_warnings", {}).get("value", [])
    assert any("does not overlap" in w for w in warnings)


def test_context_warning_covers_all_data():
    df = _make_df()
    context = {"value": {"min_expected": 0, "max_expected": 200}}
    stats = _compute_stats(df, context_by_column=context)
    warnings = stats.get("outlier_warnings", {}).get("value", [])
    assert any("covers all data points" in w for w in warnings)


def test_dismiss_removes_from_active():
    df = _make_df()
    stats = _compute_stats(df)
    state = create_initial_state(
        student_id="test",
        session_id="session_test",
        csv_filename="test.csv",
        csv_stats=stats,
    )
    # Simulate detection with P03
    state["problems_detected"] = [{"problem_id": "P03", "severity": "HIGH", "message": "outliers"}]
    state["problems_dismissed"]["P03"] = {"reason": "valid", "timestamp": "now"}

    active = [
        p["problem_id"]
        for p in state["problems_detected"]
        if p["problem_id"] not in state["problems_dismissed"]
    ]
    assert "P03" not in active
