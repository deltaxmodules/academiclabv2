"""Dynamic node router based on dataset analysis_type."""

from __future__ import annotations

from typing import Dict, List


def get_nodes_for_analysis(analysis_type: Dict) -> List[str]:
    primary_type = analysis_type.get("primary_type", "mixed")
    has_temporal = analysis_type.get("has_temporal", False)
    has_sensitive = analysis_type.get("has_sensitive_attributes", False)
    is_imbalanced = analysis_type.get("is_imbalanced", False)
    is_high_dim = analysis_type.get("is_high_dimensional", False)
    has_categorical = analysis_type.get("has_categorical", False)
    is_sparse = analysis_type.get("is_sparse", False)

    nodes: List[str] = []

    nodes.append("profile_data")

    if has_sensitive:
        nodes.append("p10_bias")

    if primary_type == "numeric":
        nodes.extend(["p01_missing", "p03_outliers", "p07_scaling"])
        if is_imbalanced:
            nodes.append("p09_imbalance")
        if is_high_dim:
            nodes.append("p31_dimensionality")
        if has_temporal:
            nodes.extend(["p16_timeseries", "p17_drift"])

    elif primary_type == "temporal":
        nodes.extend(["p16_timeseries", "p01_missing", "p17_drift"])
        if is_sparse:
            nodes.append("p18_noise")

    elif primary_type == "categorical":
        nodes.extend(["p05_categories", "p25_rare", "p08_encoding"])

    elif primary_type == "text":
        nodes.append("p19_text")
        if has_categorical:
            nodes.append("p08_encoding")
        if is_imbalanced:
            nodes.append("p09_imbalance")

    elif primary_type == "image":
        nodes.append("p20_images")
        if has_categorical:
            nodes.append("p08_encoding")
        if is_imbalanced:
            nodes.append("p09_imbalance")

    elif primary_type == "audio":
        nodes.append("p21_audio")
        if has_categorical:
            nodes.append("p08_encoding")

    elif primary_type == "mixed":
        nodes.extend(["p01_missing", "p03_outliers", "p07_scaling"])
        if has_categorical:
            nodes.extend(["p05_categories", "p25_rare", "p08_encoding"])
        if is_imbalanced:
            nodes.append("p09_imbalance")
        if has_temporal:
            nodes.append("p16_timeseries")

    return nodes

