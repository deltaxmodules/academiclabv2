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
            "analysis_type": self._detect_analysis_type(),
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

    def _detect_analysis_type(self) -> Dict:
        numeric_cols = self.df.select_dtypes(include=["int", "float"]).columns
        categorical_cols = self.df.select_dtypes(include=["object", "category"]).columns
        datetime_cols = self.df.select_dtypes(include=["datetime64"]).columns

        has_text = any(
            self.df[col].astype(str).str.len().mean() > 50
            for col in categorical_cols
        )
        has_images = any(
            self.df[col].astype(str)
            .str.contains(r"\\.(jpg|png|jpeg|gif|webp)$", case=False, na=False)
            .any()
            for col in categorical_cols
        )
        has_audio = any(
            self.df[col].astype(str)
            .str.contains(r"\\.(wav|mp3|flac|ogg)$", case=False, na=False)
            .any()
            for col in categorical_cols
        )

        sensitive_keywords = ["gender", "race", "ethnicity", "age", "sex", "religion"]
        has_sensitive = any(
            keyword in col.lower()
            for col in self.df.columns
            for keyword in sensitive_keywords
        )

        is_imbalanced = False
        if len(self.df.columns) > 0:
            target_col = self.df.columns[-1]
            if self.df[target_col].nunique(dropna=False) == 2:
                distribution = self.df[target_col].value_counts(dropna=False)
                if len(distribution) >= 2:
                    ratio = distribution.max() / max(distribution.min(), 1)
                    is_imbalanced = ratio > 3

        is_high_dimensional = len(numeric_cols) > 50 or len(self.df.columns) > 100

        total_cells = max(len(self.df) * max(len(self.df.columns), 1), 1)
        is_sparse = (self.df.isnull().sum().sum() / total_cells) > 0.3

        if has_text:
            primary_type = "text"
        elif has_images:
            primary_type = "image"
        elif has_audio:
            primary_type = "audio"
        elif len(datetime_cols) > 0:
            primary_type = "temporal"
        elif len(categorical_cols) > len(numeric_cols):
            primary_type = "categorical"
        elif len(numeric_cols) > len(categorical_cols):
            primary_type = "numeric"
        else:
            primary_type = "mixed"

        return {
            "primary_type": primary_type,
            "has_temporal": len(datetime_cols) > 0,
            "has_categorical": len(categorical_cols) > 0,
            "has_text": has_text,
            "has_images": has_images,
            "has_audio": has_audio,
            "has_sensitive_attributes": has_sensitive,
            "is_imbalanced": is_imbalanced,
            "is_high_dimensional": is_high_dimensional,
            "is_sparse": is_sparse,
        }
