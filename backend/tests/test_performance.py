import time

import numpy as np
import pandas as pd

from backend.main import _compute_stats


def test_missing_classification_under_1s_for_100k_rows():
    n = 100_000
    df = pd.DataFrame(
        {
            "col_a": np.random.choice([1, 2, 3, np.nan], n),
            "col_b": np.random.choice(["A", "B", "C", None], n),
            "target": np.random.choice(["yes", "no"], n),
        }
    )
    start = time.time()
    _compute_stats(df)
    elapsed = time.time() - start
    assert elapsed < 5.0
