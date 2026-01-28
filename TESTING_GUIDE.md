# Testing Guide

## Summary
Total: 18 tests across 4 files

### Files
- `backend/tests/test_missing_classification.py`
- `backend/tests/test_edge_cases.py`
- `backend/tests/test_real_datasets.py`
- `backend/tests/test_performance.py`

## Run All Tests
```
backend/.venv312/bin/pytest -q
```

## Run a Specific File
```
backend/.venv312/bin/pytest backend/tests/test_edge_cases.py -v
```

## Run a Specific Test
```
backend/.venv312/bin/pytest backend/tests/test_edge_cases.py::test_empty_dataset -v
```

## Notes
- `test_real_datasets.py` is optional and only runs if those datasets exist locally.
