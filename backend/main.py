from __future__ import annotations

import io
import json
import logging
import os
import re
import uuid
from datetime import datetime
from typing import Dict

import pandas as pd
import numpy as np
from langdetect import detect
from dotenv import load_dotenv
from pathlib import Path
from fastapi import Body, FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from graph import AGENT_GRAPH
from state import StudentState, create_initial_state
from nodes import translate_message


load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("academiclab")

app = FastAPI(title="Academic Lab - Data Prep Tutor (LangGraph)")

cors_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in cors_origins.split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store (use DB in production)
STUDENT_SESSIONS: Dict[str, StudentState] = {}


def _compute_stats(
    df: pd.DataFrame,
    context_by_column: Dict | None = None,
    thresholds: Dict | None = None,
) -> Dict:
    """Compute lightweight dataset stats for the analyzer."""
    rows = len(df)
    dtypes = df.dtypes.astype(str).to_dict()
    missing_pct = (df.isnull().sum() / rows * 100).to_dict() if rows else {}
    duplicates = int(df.duplicated().sum()) if rows else 0

    # Numeric outliers (context-aware + sensitivity)
    outliers = {}
    outlier_warnings: Dict[str, list[str]] = {}
    context_by_column = context_by_column or {}
    thresholds = thresholds or {}
    sensitivity = float(thresholds.get("outlier_sensitivity", 3.0))
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        try:
            context = context_by_column.get(col)
            if context and context.get("min_expected") is not None and context.get("max_expected") is not None:
                min_expected = float(context["min_expected"])
                max_expected = float(context["max_expected"])
                data_min = float(df[col].min())
                data_max = float(df[col].max())
                warnings = []
                if max_expected < data_min or min_expected > data_max:
                    warnings.append(
                        f"Context range [{min_expected}, {max_expected}] does not overlap data range [{data_min}, {data_max}]"
                    )
                if min_expected <= data_min and max_expected >= data_max:
                    warnings.append(
                        "Context range covers all data points; no outliers will be marked within the range"
                    )
                if warnings:
                    outlier_warnings[col] = warnings
                outlier_mask = (df[col] < min_expected) | (df[col] > max_expected)
            else:
                mean = df[col].mean()
                std = df[col].std()
                if std == 0 or pd.isna(std):
                    outlier_mask = pd.Series([False] * len(df))
                else:
                    outlier_mask = (df[col] - mean).abs() > (sensitivity * std)
            count = int(outlier_mask.sum())
        except Exception:
            count = 0
        outliers[col] = {"count": count}

    # Correlations
    high_corr = []
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().abs()
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                if corr.iloc[i, j] > 0.95:
                    high_corr.append(
                        {
                            "col1": corr.columns[i],
                            "col2": corr.columns[j],
                            "correlation": float(corr.iloc[i, j]),
                        }
                    )

    # Target distribution
    target_candidates = [
        col for col in df.columns if col.lower() in {"target", "label", "y"}
    ]
    if len(df.columns) == 0:
        target_col = ""
    else:
        target_col = target_candidates[0] if target_candidates else df.columns[-1]
    target_dist = {}
    if target_col in df.columns:
        value_counts = df[target_col].value_counts(dropna=False)
        if len(value_counts) <= 10:
            target_dist = value_counts.to_dict()

    # Missing type heuristic (MCAR/MAR/MNAR)
    missing_types = {}
    if rows:
        for col, rate in missing_pct.items():
            if rate <= 0:
                continue
            if rate > 50:
                missing_types[col] = "MNAR"
                continue
            # Heuristic: if missing rate varies by categorical target -> MAR
            if (
                target_col in df.columns
                and col != target_col
                and not pd.api.types.is_numeric_dtype(df[target_col])
            ):
                try:
                    groups = df.groupby(target_col)[col].apply(lambda s: s.isna().mean())
                    if not groups.empty and (groups.max() - groups.min()) > 0.1:
                        missing_types[col] = "MAR"
                        continue
                except Exception:
                    pass
            missing_types[col] = "MCAR"

    stats = {
        "rows": rows,
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "dtypes": dtypes,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        "missing_percentage": missing_pct,
        "missing_types": missing_types,
        "duplicates": duplicates,
        "outliers": outliers,
        "outlier_warnings": outlier_warnings,
        "high_correlations": high_corr,
        "target_column": target_col,
        "target_distribution": target_dist,
    }
    return _sanitize_for_json(stats)


def _sanitize_for_json(payload: Dict) -> Dict:
    """Replace NaN/inf values to keep JSON encoding safe."""
    def clean(value):
        if isinstance(value, (pd.Timestamp, datetime)):
            return value.isoformat()
        if isinstance(value, (np.floating, float)):
            if value != value or value in (float("inf"), float("-inf")):
                return None
            return float(value)
        if isinstance(value, np.integer):
            return int(value)
        if value is pd.NA:
            return None
        if isinstance(value, dict):
            return {clean(k): clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    return clean(payload)


def _select_problem_id(message: str, state: StudentState) -> str | None:
    """Extract a problem id from the user message if present."""
    match = re.search(r"\bP\d{2}\b", message.upper())
    if not match:
        return None
    problem_id = match.group(0)
    available = {p["problem_id"] for p in state["problems_detected"]}
    return problem_id if problem_id in available else None


def _next_unsolved_problem(state: StudentState) -> str | None:
    """Get the next unsolved problem id."""
    solved = set(state["problems_solved"])
    dismissed = set(state.get("problems_dismissed", {}).keys())
    for problem in state["problems_detected"]:
        if problem["problem_id"] not in solved and problem["problem_id"] not in dismissed:
            return problem["problem_id"]
    return None


def _active_problem_ids(state: StudentState) -> list[str]:
    solved = set(state.get("problems_solved", []))
    dismissed = set(state.get("problems_dismissed", {}).keys())
    return [
        p["problem_id"]
        for p in state.get("problems_detected", [])
        if p["problem_id"] not in solved and p["problem_id"] not in dismissed
    ]


def _recompute_state(state: StudentState) -> StudentState:
    """Recompute stats and problems using stored CSV bytes and session settings."""
    csv_bytes = state.get("csv_bytes")
    if not csv_bytes:
        return state

    try:
        df = pd.read_csv(io.BytesIO(csv_bytes))
    except Exception:
        return state

    stats = _compute_stats(
        df,
        context_by_column=state.get("context_by_column", {}),
        thresholds=state.get("thresholds", {}),
    )
    state["csv_stats"] = stats
    state["problems_detected"] = []
    state["current_problem"] = None
    state["attempts_current_problem"] = 0
    state["last_validation_result"] = None
    state["last_action"] = "analyze"

    state = AGENT_GRAPH.invoke(state)
    next_problem = _next_unsolved_problem(state)
    state["current_problem"] = next_problem
    return state


def _compare_problem_sets(old: StudentState, new: StudentState) -> Dict[str, set]:
    before = {p["problem_id"] for p in old.get("problems_detected", [])}
    after = {p["problem_id"] for p in new.get("problems_detected", [])}
    return {
        "resolved": before - after,
        "remaining": after,
        "new": after - before,
    }


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Upload CSV, analyze stats, and initialize a session."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV only")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {exc}") from exc

    stats = _compute_stats(df)
    session_id = f"session_{uuid.uuid4().hex}"

    state = create_initial_state(
        student_id="default_user",
        session_id=session_id,
        csv_filename=file.filename,
        csv_stats=stats,
    )
    state["csv_bytes"] = contents
    state["csv_version"] = 1
    state["last_action"] = "upload"

    state = AGENT_GRAPH.invoke(state)
    STUDENT_SESSIONS[session_id] = state

    response = {
        "success": True,
        "session_id": session_id,
        "filename": file.filename,
        "csv_version": 1,
        "dataset_info": stats,
        "problems_detected": [
            {
                "id": p["problem_id"],
                "severity": p["severity"],
                "description": p.get("message", ""),
            }
            for p in state["problems_detected"]
        ],
        "active_problems": _active_problem_ids(state),
        "thresholds": state.get("thresholds", {}),
        "message": state["last_response"],
    }
    return _sanitize_for_json(response)


@app.post("/reupload/{session_id}")
async def reupload_csv(session_id: str, file: UploadFile = File(...)):
    """Reupload CSV for an existing session and re-run analysis."""
    if session_id not in STUDENT_SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV only")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {exc}") from exc

    old_state = STUDENT_SESSIONS[session_id]
    before_state = old_state.copy()
    stats = _compute_stats(df)
    new_version = old_state.get("csv_version", 1) + 1

    old_state["csv_filename"] = file.filename
    old_state["csv_stats"] = stats
    old_state["csv_version"] = new_version
    old_state["csv_bytes"] = contents
    old_state["problems_detected"] = []
    old_state["current_problem"] = None
    old_state["problems_solved"] = []
    old_state["checklist_status"] = {}
    old_state["attempts_current_problem"] = 0
    old_state["last_validation_result"] = None
    old_state["conversation"] = []
    old_state["last_response"] = ""
    old_state["last_action"] = "upload"
    old_state["messages_count"] = 0
    old_state["guardrail_failed"] = False
    old_state["guardrail_reason"] = None
    old_state["guardrail_history"] = []
    old_state["reupload_required"] = False
    old_state["timestamp_last_update"] = datetime.now()

    state = AGENT_GRAPH.invoke(old_state)
    STUDENT_SESSIONS[session_id] = state

    diff = _compare_problem_sets(before_state, state)
    resolved_list = sorted(diff["resolved"])
    remaining_list = sorted(diff["remaining"])
    cols = stats.get("column_names", []) or []
    cabin_present = "Cabin" in cols
    cabin_missing = stats.get("missing_percentage", {}).get("Cabin")
    summary = f"📥 Reupload received: {file.filename}\n"
    summary += f"🔎 Columns: {len(cols)} | Cabin present: {'yes' if cabin_present else 'no'}"
    if cabin_present and cabin_missing is not None:
        summary += f" (missing {cabin_missing:.1f}%)"
    summary += "\n"
    if resolved_list:
        summary += f"✅ Resolved issue(s): {', '.join(resolved_list)}\n"
    if remaining_list:
        summary += f"⚠️ Still present: {', '.join(remaining_list)}\n"
    if summary.strip() == f"📥 Reupload received: {file.filename}":
        summary = f"📥 Reupload received: {file.filename}\nℹ️ No changes detected in the issues."
    if state.get("last_response"):
        state["last_response"] = summary.strip() + "\n\n" + state["last_response"]
    else:
        state["last_response"] = summary.strip()

    response = {
        "success": True,
        "session_id": session_id,
        "filename": file.filename,
        "csv_version": new_version,
        "dataset_info": stats,
        "problems_detected": [
            {
                "id": p["problem_id"],
                "severity": p["severity"],
                "description": p.get("message", ""),
            }
            for p in state["problems_detected"]
        ],
        "active_problems": _active_problem_ids(state),
        "thresholds": state.get("thresholds", {}),
        "message": state["last_response"],
    }
    return _sanitize_for_json(response)


@app.post("/session/{session_id}/dismiss-problem/{problem_id}")
async def dismiss_problem(session_id: str, problem_id: str, payload: Dict = Body(...)):
    """Dismiss a problem as a false alarm with required explanation."""
    if session_id not in STUDENT_SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    reason = (payload.get("reason") or "").strip()
    if not reason:
        raise HTTPException(status_code=400, detail="reason required")

    state = STUDENT_SESSIONS[session_id]
    detected = {p["problem_id"] for p in state.get("problems_detected", [])}
    problem_id = problem_id.upper()
    if problem_id not in detected:
        raise HTTPException(status_code=400, detail="problem_id not in detected issues")

    state.setdefault("problems_dismissed", {})
    state["problems_dismissed"][problem_id] = {
        "reason": reason,
        "timestamp": datetime.now().isoformat(),
    }

    state["last_response"] = f"✅ {problem_id} dismissed as false alarm."
    state["last_action"] = "dismiss_problem"
    state["timestamp_last_update"] = datetime.now()
    STUDENT_SESSIONS[session_id] = state

    return _sanitize_for_json(
        {
            "success": True,
            "problem_id": problem_id,
            "message": state["last_response"],
            "active_problems": _active_problem_ids(state),
        }
    )


@app.post("/session/{session_id}/problem/{problem_id}/context")
async def provide_context(session_id: str, problem_id: str, payload: Dict = Body(...)):
    """Provide domain context for a column to refine detection."""
    if session_id not in STUDENT_SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    column = (payload.get("column") or "").strip()
    explanation = (payload.get("explanation") or "").strip()
    if not column or not explanation:
        raise HTTPException(status_code=400, detail="column and explanation required")

    state = STUDENT_SESSIONS[session_id]
    state.setdefault("context_by_column", {})
    state["context_by_column"][column] = {
        "dataset_type": payload.get("dataset_type"),
        "min_expected": payload.get("min_expected"),
        "max_expected": payload.get("max_expected"),
        "explanation": explanation,
        "timestamp": datetime.now().isoformat(),
    }

    state = _recompute_state(state)
    STUDENT_SESSIONS[session_id] = state

    return _sanitize_for_json(
        {
            "success": True,
            "message": "✅ Context saved. Analysis updated.",
            "active_problems": _active_problem_ids(state),
            "outlier_warnings": state.get("csv_stats", {}).get("outlier_warnings", {}),
        }
    )


@app.post("/session/{session_id}/threshold/{threshold_name}")
async def set_threshold(session_id: str, threshold_name: str, payload: Dict = Body(...)):
    """Adjust analysis thresholds (e.g., outlier sensitivity)."""
    if session_id not in STUDENT_SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    value = payload.get("value")
    if value is None:
        raise HTTPException(status_code=400, detail="value required")

    threshold_name = threshold_name.strip()
    state = STUDENT_SESSIONS[session_id]
    state.setdefault("thresholds", {})

    if threshold_name == "outlier_sensitivity":
        try:
            value = float(value)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="value must be numeric") from exc
        if not 1.0 <= value <= 5.0:
            raise HTTPException(status_code=400, detail="outlier_sensitivity must be 1.0-5.0")

    state["thresholds"][threshold_name] = value
    state = _recompute_state(state)
    STUDENT_SESSIONS[session_id] = state

    return _sanitize_for_json(
        {
            "success": True,
            "message": f"✅ Threshold updated: {threshold_name} = {value}",
            "active_problems": _active_problem_ids(state),
            "thresholds": state.get("thresholds", {}),
            "outlier_warnings": state.get("csv_stats", {}).get("outlier_warnings", {}),
        }
    )


@app.websocket("/chat/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket chat endpoint."""
    await websocket.accept()

    if session_id not in STUDENT_SESSIONS:
        await websocket.close(code=4000, reason="Session not found")
        return

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            message = payload.get("message", "")

            state = STUDENT_SESSIONS[session_id]
            state["conversation"].append(
                {"role": "user", "content": message, "timestamp": datetime.now()}
            )
            state["messages_count"] += 1

            msg_lower = message.lower()
            if "list" in msg_lower or "issues" in msg_lower:
                state["last_action"] = "analyze"
            elif "next" in msg_lower:
                next_problem = _next_unsolved_problem(state)
                if next_problem:
                    state["current_problem"] = next_problem
                    state["last_action"] = "show_problems"
            else:
                selected = _select_problem_id(message, state)
                if selected:
                    state["current_problem"] = selected
                    state["last_action"] = "show_problems"

            state = AGENT_GRAPH.invoke(state)
            STUDENT_SESSIONS[session_id] = state

            # Auto-detect user language and translate assistant response
            target_lang = None
            last_user = next(
                (m for m in reversed(state["conversation"]) if m.get("role") == "user"),
                None,
            )
            if last_user and last_user.get("content"):
                try:
                    lang = detect(last_user["content"])
                    if lang in {"pt", "fr", "es", "it", "de"}:
                        target_lang = lang
                except Exception:
                    target_lang = None

            content = state["last_response"]
            if target_lang:
                content = translate_message(content, target_lang)

            await websocket.send_json(
                {
                    "type": "response",
                    "content": content,
                    "action": state["last_action"],
                    "problems_solved": state["problems_solved"],
                    "understanding_level": state["understanding_level"],
                    "reupload_required": state.get("reupload_required", False),
                    "active_problems": _active_problem_ids(state),
                    "thresholds": state.get("thresholds", {}),
                }
            )

    except Exception as exc:
        logger.error("WebSocket error: %s", exc)
        await websocket.close(code=4001, reason=str(exc))


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Return session summary for debugging."""
    if session_id not in STUDENT_SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    state = STUDENT_SESSIONS[session_id]
    duration = (datetime.now() - state["timestamp_start"]).total_seconds() / 60
    return {
        "session_id": session_id,
        "filename": state["csv_filename"],
        "problems_detected": len(state["problems_detected"]),
        "problems_solved": state["problems_solved"],
        "problems_dismissed": state.get("problems_dismissed", {}),
        "context_by_column": state.get("context_by_column", {}),
        "thresholds": state.get("thresholds", {}),
        "understanding_level": state["understanding_level"],
        "messages_count": state["messages_count"],
        "checklist_report": state.get("checklist_report", {}),
        "duration_minutes": round(duration, 2),
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "graph": "langgraph_initialized"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
def _session_upload_dir(session_id: str) -> str:
    return str((Path(__file__).resolve().parent / "uploads" / session_id))


def _save_csv(contents: bytes, session_id: str, version: int) -> str:
    upload_dir = Path(_session_upload_dir(session_id))
    upload_dir.mkdir(parents=True, exist_ok=True)
    filename = f"dataset_v{version}.csv"
    path = upload_dir / filename
    path.write_bytes(contents)
    return str(path)


def _compare_problem_sets(old: StudentState, new: StudentState) -> Dict[str, set]:
    before = {p["problem_id"] for p in old.get("problems_detected", [])}
    after = {p["problem_id"] for p in new.get("problems_detected", [])}
    return {
        "resolved": before - after,
        "remaining": after,
        "new": after - before,
    }
