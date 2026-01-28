from analyze_csv import QuickAnalyzer


def test_quick_analyzer_detects_missing_and_duplicates():
    stats = {
        "rows": 100,
        "missing_percentage": {"price": 12.5, "qty": 0.0},
        "duplicates": 5,
    }

    problems = QuickAnalyzer.analyze_stats(stats)
    ids = {p["problem_id"] for p in problems}

    assert "P01" in ids
    assert "P02" in ids
