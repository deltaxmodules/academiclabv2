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
from dotenv import load_dotenv
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from graph import AGENT_GRAPH
from state import StudentState, create_initial_state


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


def _compute_stats(df: pd.DataFrame) -> Dict:
    """Compute lightweight dataset stats for the analyzer."""
    rows = len(df)
    dtypes = df.dtypes.astype(str).to_dict()
    missing_pct = (df.isnull().sum() / rows * 100).to_dict() if rows else {}
    duplicates = int(df.duplicated().sum()) if rows else 0

    # Numeric outliers (IQR)
    outliers = {}
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        count = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
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
    target_col = target_candidates[0] if target_candidates else df.columns[-1]
    target_dist = {}
    if target_col in df.columns:
        value_counts = df[target_col].value_counts(dropna=False)
        if len(value_counts) <= 10:
            target_dist = value_counts.to_dict()

    stats = {
        "rows": rows,
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "dtypes": dtypes,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        "missing_percentage": missing_pct,
        "duplicates": duplicates,
        "outliers": outliers,
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
    for problem in state["problems_detected"]:
        if problem["problem_id"] not in solved:
            return problem["problem_id"]
    return None


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Upload CSV, analyze stats, and initialize a session."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Apenas CSV")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"CSV inválido: {exc}") from exc

    stats = _compute_stats(df)
    session_id = f"session_{uuid.uuid4().hex}"

    state = create_initial_state(
        student_id="default_user",
        session_id=session_id,
        csv_filename=file.filename,
        csv_stats=stats,
    )
    state["last_action"] = "upload"

    state = AGENT_GRAPH.invoke(state)
    STUDENT_SESSIONS[session_id] = state

    response = {
        "success": True,
        "session_id": session_id,
        "filename": file.filename,
        "dataset_info": stats,
        "problems_detected": [
            {
                "id": p["problem_id"],
                "severity": p["severity"],
                "description": p.get("message", ""),
            }
            for p in state["problems_detected"]
        ],
        "message": state["last_response"],
    }
    return _sanitize_for_json(response)


@app.post("/reupload/{session_id}")
async def reupload_csv(session_id: str, file: UploadFile = File(...)):
    """Reupload CSV for an existing session and re-run analysis."""
    if session_id not in STUDENT_SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Apenas CSV")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"CSV inválido: {exc}") from exc

    state = STUDENT_SESSIONS[session_id]
    stats = _compute_stats(df)

    state["csv_filename"] = file.filename
    state["csv_stats"] = stats
    state["problems_detected"] = []
    state["current_problem"] = None
    state["last_action"] = "upload"
    state["timestamp_last_update"] = datetime.now()

    state = AGENT_GRAPH.invoke(state)
    STUDENT_SESSIONS[session_id] = state

    response = {
        "success": True,
        "session_id": session_id,
        "filename": file.filename,
        "dataset_info": stats,
        "problems_detected": [
            {
                "id": p["problem_id"],
                "severity": p["severity"],
                "description": p.get("message", ""),
            }
            for p in state["problems_detected"]
        ],
        "message": state["last_response"],
    }
    return _sanitize_for_json(response)


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
            if "lista" in msg_lower or "problemas" in msg_lower:
                state["last_action"] = "analyze"
            elif "próximo" in msg_lower or "proximo" in msg_lower or "next" in msg_lower:
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

            await websocket.send_json(
                {
                    "type": "response",
                    "content": state["last_response"],
                    "action": state["last_action"],
                    "problems_solved": state["problems_solved"],
                    "understanding_level": state["understanding_level"],
                    "reupload_required": state.get("reupload_required", False),
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
        "understanding_level": state["understanding_level"],
        "messages_count": state["messages_count"],
        "duration_minutes": round(duration, 2),
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "graph": "langgraph_initialized"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
