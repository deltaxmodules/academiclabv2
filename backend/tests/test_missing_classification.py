import pandas as pd

from backend.main import _compute_stats


def _classify(df, column):
    stats = _compute_stats(df)
    return stats["missing_types"].get(column)


def test_mcar_random_missing():
    df = pd.DataFrame(
        {
            "age": [25, None, 35, None, 45, None],
            "salary": [50000, 60000, None, 70000, 80000, 90000],
            "department": ["A", "B", "C", "A", "B", "C"],
        }
    )
    assert _classify(df, "age") == "MCAR"


def test_mar_missing_by_group():
    df = pd.DataFrame(
        {
            "income": [None, None, 100000, 120000, 140000, 150000],
            "age_group": ["young", "young", "mid", "mid", "senior", "senior"],
        }
    )
    assert _classify(df, "income") == "MAR"


def test_mnar_high_missing_rate():
    df = pd.DataFrame(
        {
            "bonus": [None, None, None, None, None, 10000],
            "salary": [50000, 60000, 70000, 80000, 90000, 100000],
        }
    )
    assert _classify(df, "bonus") == "MNAR"


def test_mar_missing_by_role():
    df = pd.DataFrame(
        {
            "income": [None, None, 100000, 120000, 150000, 180000],
            "role": ["intern", "junior", "mid", "senior", "lead", "director"],
        }
    )
    assert _classify(df, "income") == "MAR"


def test_mcar_healthcare_like():
    df = pd.DataFrame(
        {
            "age": [None, 45, None, 60, 70, None],
            "health_score": [8, 7, 9, 6, 5, 8],
        }
    )
    assert _classify(df, "age") == "MCAR"
