from pathlib import Path

import pandas as pd

from backend.main import _compute_stats


ROOT = Path("/Users/jataide/Desktop")
DATASETS = [
    ROOT / "finance_credit_defaults.csv",
    ROOT / "healthcare_medical_records.csv",
    ROOT / "ecommerce_customer_data.csv",
    ROOT / "iot_sensor_data.csv",
]


def _load_if_exists(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)


def test_real_world_datasets_if_present():
    # This test is optional and only runs if datasets exist locally.
    for path in DATASETS:
        df = _load_if_exists(path)
        if df is None:
            continue
        stats = _compute_stats(df)
        assert stats["rows"] == len(df)
