from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional, TypedDict


class Message(TypedDict, total=False):
    """Conversation message structure."""
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime


class Problem(TypedDict, total=False):
    """Detected problem (P01-P35)."""
    problem_id: str
    problem_name: str
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    description: str
    column: Optional[str]
    percentage: Optional[float]
    checklist_ref: str
    message: str
    details: Dict


class StudentState(TypedDict):
    """Full student session state for LangGraph."""
    # Identity
    student_id: str
    session_id: str

    # Dataset
    csv_filename: str
    csv_stats: Dict
    csv_version: int

    # Problems
    problems_detected: List[Problem]
    current_problem: Optional[str]

    # Progress
    problems_solved: List[str]
    checklist_status: Dict[str, bool]

    # Understanding
    understanding_level: Literal["beginner", "intermediate", "advanced"]
    attempts_current_problem: int
    last_validation_result: Optional[Dict]

    # Conversation
    conversation: List[Message]
    last_response: str
    last_action: str
    messages_count: int

    # Guardrails
    guardrail_failed: bool
    guardrail_reason: Optional[str]
    guardrail_history: List[Dict]

    # Reupload guidance
    reupload_required: bool

    # Metadata
    timestamp_start: datetime
    timestamp_last_update: datetime


def create_initial_state(
    student_id: str,
    session_id: str,
    csv_filename: str,
    csv_stats: Dict,
) -> StudentState:
    """Create a fresh StudentState for a new session."""
    now = datetime.now()
    return StudentState(
        student_id=student_id,
        session_id=session_id,
        csv_filename=csv_filename,
        csv_stats=csv_stats,
        csv_path=None,
        csv_version=1,
        problems_detected=[],
        current_problem=None,
        problems_solved=[],
        checklist_status={},
        understanding_level="beginner",
        attempts_current_problem=0,
        last_validation_result=None,
        conversation=[],
        last_response="",
        last_action="init",
        messages_count=0,
        guardrail_failed=False,
        guardrail_reason=None,
        guardrail_history=[],
        reupload_required=False,
        timestamp_start=now,
        timestamp_last_update=now,
    )
