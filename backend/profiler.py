from __future__ import annotations

from typing import Dict

import pandas as pd


class DataProfiler:
    """Compute a lightweight profiling report for a dataset."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def run_full_profile(self) -> Dict:
        return {
            "structure": self._analyze_structure(),
            "data_types": self._analyze_data_types(),
            "missing_analysis": self._analyze_missing_values(),
            "numeric_distributions": self._analyze_numeric(),
            "categorical_distributions": self._analyze_categorical(),
            "temporal_info": self._analyze_temporal(),
            "granularity": self._analyze_granularity(),
            "questions_for_user": self._generate_user_questions(),
        }

    def _analyze_structure(self) -> Dict:
        return {
            "n_rows": int(len(self.df)),
            "n_columns": int(len(self.df.columns)),
            "size_mb": round(self.df.memory_usage(deep=True).sum() / 1024**2, 2),
        }

    def _analyze_data_types(self) -> Dict:
        return {
            "numeric": list(self.df.select_dtypes(include=["int", "float"]).columns),
            "categorical": list(self.df.select_dtypes(include=["object", "category"]).columns),
            "datetime": list(self.df.select_dtypes(include=["datetime64"]).columns),
        }

    def _analyze_missing_values(self) -> Dict:
        total_missing = self.df.isnull().sum().sum()
        total_cells = max(len(self.df) * max(len(self.df.columns), 1), 1)
        total_missing_pct = (total_missing / total_cells) * 100
        by_column = (self.df.isnull().sum() / max(len(self.df), 1) * 100).to_dict()
        return {
            "total_missing_pct": round(total_missing_pct, 2),
            "by_column": {k: round(v, 2) for k, v in by_column.items()},
            "missing_pattern": "UNKNOWN",
        }

    def _analyze_numeric(self) -> Dict:
        result: Dict[str, Dict] = {}
        for col in self.df.select_dtypes(include=["int", "float"]).columns:
            data = self.df[col].dropna()
            if data.empty:
                continue
            q1, q3 = data.quantile(0.25), data.quantile(0.75)
            iqr = q3 - q1
            outlier_count = 0
            if iqr != 0:
                outlier_count = ((data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)).sum()
            result[col] = {
                "mean": round(float(data.mean()), 4),
                "std": round(float(data.std()), 4),
                "skewness": round(float(data.skew()), 4),
                "outliers_pct": round(float(outlier_count / len(data) * 100), 2),
            }
        return result

    def _analyze_categorical(self) -> Dict:
        result: Dict[str, Dict] = {}
        for col in self.df.select_dtypes(include=["object", "category"]).columns:
            value_counts = self.df[col].value_counts(dropna=False).head(3).to_dict()
            result[col] = {
                "unique_count": int(self.df[col].nunique(dropna=False)),
                "most_frequent": value_counts,
            }
        return result

    def _analyze_temporal(self) -> Dict:
        dt_cols = list(self.df.select_dtypes(include=["datetime64"]).columns)
        return {
            "has_datetime": len(dt_cols) > 0,
            "datetime_columns": dt_cols,
            "is_timeseries": len(dt_cols) > 0,
        }

    def _analyze_granularity(self) -> Dict:
        return {
            "duplicate_rows": int(self.df.duplicated().sum()),
            "unit_of_analysis": "UNKNOWN",
        }

    def _generate_user_questions(self) -> Dict:
        return {
            "unit_of_analysis": "Is this per customer, per transaction, or per day?",
            "missing_pattern": "Is missing MCAR, MAR, or MNAR?",
            "target_variable": "What variable are we predicting?",
            "temporal_importance": "Is temporal order important?",
            "sensitive_attributes": "Are there sensitive attributes (gender, age, race)?",
            "business_rules": "What are valid value ranges?",
        }
