from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def load_framework() -> Dict:
    """Load and cache the framework JSON."""
    data_path = _project_root() / "backend" / "data" / "data_preparation_framework_complete.json"
    with data_path.open("r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_checklist() -> Dict:
    """Load and cache the checklist JSON."""
    data_path = _project_root() / "backend" / "data" / "data_preparation_checklist.json"
    with data_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def lookup_problem(problem_id: str) -> Optional[Dict]:
    """Find a problem by id in the framework."""
    framework = load_framework()
    for problem in framework.get("data_preparation_complete", []):
        if problem.get("id") == problem_id:
            return problem
    return None


def lookup_checklist_item(checklist_id: str) -> Optional[Dict]:
    """Find a checklist item by id in the checklist."""
    checklist = load_checklist()
    for category in checklist.get("categories", []):
        for item in category.get("items", []):
            if item.get("chk_id") == checklist_id:
                return item
    return None
