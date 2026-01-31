from __future__ import annotations

import json
from langdetect import detect
import re
from datetime import datetime
from typing import Dict, List

import os

from langchain_openai import ChatOpenAI

from analyze_csv import QuickAnalyzer, _missing_action_hint
from data_access import lookup_checklist_item, lookup_problem
from state import StudentState


_FORBIDDEN_EXECUTION_PATTERNS = [
    "os.system",
    "subprocess",
    "exec(",
    "eval(",
    "__import__",
]

_EXECUTION_CLAIM_PATTERNS = [
    r"\bexecutei\b",
    r"\beu executei\b",
    r"\bexecutando\b",
    r"\bI executed\b",
    r"\bI ran\b",
    r"\brunning on the server\b",
]

_LANG_CODE_MAP = {
    "pt": "Portuguese",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "de": "German",
}


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def should_route_to_expert(text: str) -> bool:
    """Route to expert only when the user explicitly asks via prefix."""
    text = _clean_text(text).lower()
    return text.startswith("expert:")


def _now() -> datetime:
    return datetime.now()


def _get_llm() -> ChatOpenAI:
    """Create a singleton LLM client."""
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"),
        temperature=0.6,
        max_retries=2,
        request_timeout=30,
    )


def translate_message(text: str, target_lang: str) -> str:
    """Translate assistant text while preserving code blocks."""
    if target_lang not in _LANG_CODE_MAP:
        return text
    llm = _get_llm()
    system_prompt = (
        "You are a translation engine. Translate the message to the target language. "
        "Preserve code blocks, inline code, and identifiers exactly. Do not translate code."
    )
    user_prompt = f"Target language: {_LANG_CODE_MAP[target_lang]}\n\nMessage:\n{text}"
    try:
        response = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return response.content
    except Exception:
        return text


def analyze_csv_node(state: StudentState) -> StudentState:
    """Analyze CSV stats and detect problems P01-P35."""
    problems = QuickAnalyzer.analyze_stats(state["csv_stats"])

    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    problems_sorted = sorted(
        problems,
        key=lambda p: severity_order.get(p.get("severity", "BAIXO"), 999),
    )

    state["problems_detected"] = problems_sorted
    state["current_problem"] = problems_sorted[0]["problem_id"] if problems_sorted else None
    state["checklist_report"] = _build_chk001_report(state["csv_stats"])
    state["last_action"] = "analyze"
    state["timestamp_last_update"] = _now()

    _mark_resolved_by_feature_engineering(state)

    return state


def _build_chk001_report(stats: Dict) -> Dict:
    """In-memory CHK-001 report based on missing values."""
    report = {"CHK-001": {"criteria": [], "passed": True}}
    rows = stats.get("rows", 0) or 0
    missing = stats.get("missing_percentage", {}) or {}
    missing_types = stats.get("missing_types", {}) or {}

    for col, rate in missing.items():
        if rate <= 0:
            continue
        missing_type = missing_types.get(col, "MCAR")
        decision = _missing_action_hint(rate, rows, missing_type)
        report["CHK-001"]["criteria"].append(
            {
                "column": col,
                "missing_pct": rate,
                "missing_type": missing_type,
                "recommended_action": decision,
            }
        )

    return report


def _mark_resolved_by_feature_engineering(state: StudentState) -> None:
    """Auto-resolve missing-value issues when a derived presence flag exists."""
    problems = state.get("problems_detected", [])
    if not problems:
        return

    columns = set(state.get("csv_stats", {}).get("column_names", []))
    if not columns:
        return

    resolved = state.setdefault("problems_resolved", {})

    for problem in problems:
        if problem.get("problem_id") != "P01":
            continue
        column = problem.get("column")
        if not column:
            continue

        derived_candidates = {
            f"{column}_present",
            f"{column}_missing",
            f"{column}_flag",
            f"has_{column}",
        }
        if derived_candidates & columns:
            if problem["problem_id"] not in state.get("problems_solved", []):
                state["problems_solved"].append(problem["problem_id"])
            resolved[problem["problem_id"]] = {
                "status": "resolved_by_feature_engineering",
                "note": f"Derived column exists for {column}.",
                "column": column,
            }


def show_problems_node(state: StudentState) -> StudentState:
    """Show detected problems grouped by severity."""
    output = "Hi! I analyzed your dataset and found issues:\n\n"
    by_severity: Dict[str, List[Dict]] = {}
    for problem in state["problems_detected"]:
        sev = problem.get("severity", "MEDIUM")
        by_severity.setdefault(sev, []).append(problem)

    for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        if severity not in by_severity:
            continue
        output += f"{severity}:\n"
        for p in by_severity[severity]:
            output += f"  • {p['problem_id']}: {p.get('message', p.get('problem_name', ''))}\n"
        output += "\n"

    output += "Which issue would you like to explore first?\n"
    output += "Tip: start with CRITICAL issues."

    state["conversation"].append({"role": "assistant", "content": output, "timestamp": _now()})
    state["last_response"] = output
    state["last_action"] = "show_problems"
    state["timestamp_last_update"] = _now()

    return state


def explain_problem_node(state: StudentState) -> StudentState:
    """Explain a specific problem using the LLM."""
    problem_id = state.get("current_problem")
    if not problem_id:
        output = "No problem selected. Please choose one."
        state["conversation"].append({"role": "assistant", "content": output, "timestamp": _now()})
        state["last_response"] = output
        state["last_action"] = "explain"
        state["timestamp_last_update"] = _now()
        return state

    problem_detail = lookup_problem(problem_id)
    if not problem_detail:
        output = f"Problem {problem_id} was not found in the framework."
        state["conversation"].append({"role": "assistant", "content": output, "timestamp": _now()})
        state["last_response"] = output
        state["last_action"] = "explain"
        state["timestamp_last_update"] = _now()
        return state

    checklist_ref = problem_detail.get("checklist_ref", "CHK-001")
    checklist_item = lookup_checklist_item(checklist_ref)
    checklist_text = ""
    if checklist_item:
        checklist_text = (
            f"{checklist_item.get('chk_id')}: {checklist_item.get('task')} - "
            f"{checklist_item.get('description')}"
        )

    style = state.get("response_style", "fast")
    tutor_word_limit = "90" if style == "fast" else "160"

    system_prompt = f"""
You are a DATA SCIENCE INSTRUCTOR focused on education.

RULES:
- Never execute code or access the filesystem
- Never claim you executed anything
- Keep it concise and objective (max ~{tutor_word_limit} words)
- Focus on the student's immediate need
- Cite the related checklist item

OUTPUT FORMAT:
1) Summary (1–2 sentences)
2) Actions (2–3 bullets)
3) Checklist reference (1 line)
"""

    # Include P01-specific context
    p01_context = ""
    if problem_id == "P01":
        missing_types = state.get("csv_stats", {}).get("missing_types", {})
        missing_pct = state.get("csv_stats", {}).get("missing_percentage", {})
        col = None
        for item in state.get("problems_detected", []):
            if item.get("problem_id") == "P01":
                col = item.get("column")
                break
        if col:
            mtype = missing_types.get(col, "MCAR")
            mpct = missing_pct.get(col, 0)
            hint = _missing_action_hint(mpct, state.get("csv_stats", {}).get("rows", 0), mtype)
            p01_context = f"Missing type heuristic: {mtype}, missing %: {mpct:.1f}, recommended: {hint}"

    user_prompt = f"""
The student wants to learn about {problem_id}: {problem_detail.get('name')}

Context:
- Understanding level: {state['understanding_level']}
- Already solved: {state['problems_solved']}
- Attempts on this problem: {state['attempts_current_problem']}

Framework:
{json.dumps(problem_detail, ensure_ascii=False, indent=2)}

Checklist: {checklist_ref}
Checklist detail: {checklist_text if checklist_text else 'N/A'}
{p01_context}

Answer briefly and directly. Avoid repeating prior explanations.
"""

    llm = _get_llm()
    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    explanation = response.content

    # Build examples inline for efficiency (single response)
    examples_block = ""
    solutions = problem_detail.get("branches", [{}])[0].get("solutions", [])
    solution = _select_solution(problem_id, solutions, state)
    if solution:
        examples_block += "\n\n---\n\n"
        examples_block += f"CODE EXAMPLE for {problem_id}:\n\n"
        examples_block += "IMPORTANT: This is EDUCATIONAL code.\n"
        examples_block += "   Copy it to your environment and run it there.\n"
        examples_block += "   I will not execute anything here.\n\n"
        examples_block += f"**Method: {solution.get('method', 'Method 1')}**\n"
        examples_block += f"Use case: {solution.get('use_case', 'When to use')}\n\n"
        examples_block += f"```python\n{solution.get('code', '# Code not available')}\n```\n\n"
        examples_block += "**Pros:**\n"
        for pro in solution.get("pros", []):
            examples_block += f"  {pro}\n"
        examples_block += "\n**Cons:**\n"
        for con in solution.get("cons", []):
            examples_block += f"  {con}\n"
        examples_block += "\n---\n\n"
        examples_block += "Next step: run the code locally and come back with the result.\n"

    combined = explanation + examples_block

    state["conversation"].append({"role": "assistant", "content": combined, "timestamp": _now()})
    state["last_response"] = combined
    state["last_action"] = "explain"
    state["attempts_current_problem"] += 1
    state["timestamp_last_update"] = _now()

    return state


def expert_help_node(state: StudentState) -> StudentState:
    """Provide a technical, expert-level answer with Python code and opinion."""
    problem_id = state.get("current_problem")
    problem_detail = lookup_problem(problem_id) if problem_id else None

    checklist_ref = problem_detail.get("checklist_ref", "CHK-001") if problem_detail else "CHK-001"
    checklist_item = lookup_checklist_item(checklist_ref) if problem_detail else None
    checklist_text = ""
    if checklist_item:
        checklist_text = (
            f"{checklist_item.get('chk_id')}: {checklist_item.get('task')} - "
            f"{checklist_item.get('description')}"
        )

    last_user = next(
        (m for m in reversed(state.get("conversation", [])) if m.get("role") == "user"),
        None,
    )
    target_lang = "English"
    if last_user and last_user.get("content"):
        try:
            lang = detect(last_user["content"])
            if lang == "pt":
                target_lang = "Portuguese"
            elif lang == "fr":
                target_lang = "French"
            elif lang == "es":
                target_lang = "Spanish"
            elif lang == "it":
                target_lang = "Italian"
            elif lang == "de":
                target_lang = "German"
        except Exception:
            target_lang = "English"

    style = state.get("response_style", "fast")
    expert_word_limit = "110" if style == "fast" else "180"

    student_message = state["conversation"][-1]["content"]
    msg_lower = student_message.lower()

    # Gate: only respond as expert to data science questions.
    domain_prompt = (
        "Answer ONLY 'YES' or 'NO'. "
        "Is the user's message a data science / data analysis / ML / statistics question "
        "that expects technical guidance? "
        "If the user asks about time/date/news/weather or general chit-chat, answer NO."
    )
    try:
        domain_resp = _get_llm().invoke(
            [
                {"role": "system", "content": domain_prompt},
                {"role": "user", "content": student_message},
            ]
        )
        is_ds = domain_resp.content.strip().lower().startswith("y")
    except Exception:
        is_ds = True
    if not is_ds:
        output = {
            "Portuguese": "Posso ajudar apenas com dúvidas técnicas de data science relacionadas com o exercício.",
            "Spanish": "Solo puedo ayudar con dudas técnicas de data science relacionadas con el ejercicio.",
            "Italian": "Posso aiutare solo con dubbi tecnici di data science legati all'esercizio.",
            "French": "Je peux aider uniquement avec des questions techniques de data science liées à l’exercice.",
            "German": "Ich kann nur bei technischen Data-Science-Fragen zum Thema helfen.",
            "English": "I can only help with technical data science questions related to the task.",
        }.get(target_lang, "I can only help with technical data science questions related to the task.")
        state["conversation"].append({"role": "assistant", "content": output, "timestamp": _now()})
        state["last_response"] = output
        state["last_action"] = "expert_help"
        state["expert_requested"] = False
        state["timestamp_last_update"] = _now()
        return state

    system_prompt = f"""
You are a senior DATA SCIENCE SPECIALIST and teacher.

RULES:
- Never execute code or claim execution
- Be technical but clear
- Focus on the student's specific question and their results
- Do NOT restate generic method steps already applied
- If the student shows results, interpret them and recommend next actions
- If the student asks how to compute a method (e.g., IQR), explain it briefly
- Ask a brief clarifying question if key context is missing
- Provide a short Python code example ONLY if it helps the next step
- In code blocks: only Python code and comments, no prose
- Avoid the word 'python' as a standalone line
- Keep it concise and actionable (max ~{expert_word_limit} words)
- Respond in {target_lang}
- CRITICAL: Do NOT repeat explanations already given in the recent conversation.
  If the student is building on a previous answer, reference briefly and extend.

OUTPUT FORMAT:
1) Direct answer to the student's question (1–2 sentences)
2) Next steps (2 bullets)
3) Optional Python example (very short, only if needed)
4) Checklist reference (1 line)
"""

    p01_context = ""
    if problem_id == "P01":
        missing_types = state.get("csv_stats", {}).get("missing_types", {})
        missing_pct = state.get("csv_stats", {}).get("missing_percentage", {})
        col = None
        for item in state.get("problems_detected", []):
            if item.get("problem_id") == "P01":
                col = item.get("column")
                break
        if col:
            mtype = missing_types.get(col, "MCAR")
            mpct = missing_pct.get(col, 0)
            hint = _missing_action_hint(mpct, state.get("csv_stats", {}).get("rows", 0), mtype)
            p01_context = f"Missing type heuristic: {mtype}, missing %: {mpct:.1f}, recommended: {hint}"

    extra_focus = ""

    recent_conversation_full = []
    for msg in state.get("conversation", [])[-10:]:
        role = (msg.get("role", "") or "").upper()
        content = (msg.get("content", "") or "").strip()
        if content:
            recent_conversation_full.append(f"[{role}]: {content[:300]}")
    recent_context = "\n".join(recent_conversation_full) if recent_conversation_full else "N/A"

    framework_block = ""
    if re.search(r"\bP\d{2}\b", student_message, re.IGNORECASE) and problem_detail:
        framework_block = f"\nFramework:\n{json.dumps(problem_detail, ensure_ascii=False, indent=2)}\n"

    user_prompt = f"""
Problem: {problem_id or 'N/A'} - {problem_detail.get('name') if problem_detail else 'N/A'}

=== RECENT CONVERSATION CONTEXT ===
{recent_context}

=== LATEST STUDENT QUESTION ===
{student_message}

Conversation context:
- Understanding level: {state['understanding_level']}
- Attempts on this problem: {state['attempts_current_problem']}

Checklist reference: {checklist_ref}
{p01_context}
{extra_focus}
{framework_block}

Answer the LATEST student question directly.
Do NOT repeat explanations already provided above.
Focus on what to do next and why.
"""

    llm = _get_llm()
    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    state["conversation"].append({"role": "assistant", "content": response.content, "timestamp": _now()})
    state["last_response"] = response.content
    state["last_action"] = "expert_help"
    state["expert_requested"] = False
    state["timestamp_last_update"] = _now()
    return state

def _select_solution(problem_id: str, solutions: List[Dict], state: StudentState) -> Dict | None:
    """Pick the most appropriate solution (rules first, LLM fallback)."""
    if not solutions:
        return None

    # Rule-based keywords for all problems (fallback to LLM when no match)
    keyword_map = {
        "P01": ["missing", "impute", "median", "mean", "drop", "remove", "flag"],
        "P02": ["duplicate", "drop_duplicates", "deduplicate", "unique"],
        "P03": ["outlier", "iqr", "z-score", "cap", "clip", "winsor"],
        "P04": ["dtype", "cast", "convert", "parse", "to_datetime"],
        "P05": ["category", "normalize", "standardize", "strip", "lower"],
        "P06": ["invalid", "rule", "range", "constraint", "business"],
        "P07": ["scale", "standard", "minmax", "normalize", "robust"],
        "P08": ["encode", "one-hot", "ordinal", "target encoding"],
        "P09": ["imbalance", "resample", "smote", "class_weight"],
        "P10": ["bias", "fairness", "reweigh", "balance"],
        "P11": ["irrelevant", "drop", "feature selection"],
        "P12": ["id", "identifier", "remove id"],
        "P13": ["multicollinearity", "correlation", "vif", "drop"],
        "P14": ["leakage", "remove", "future", "target"],
        "P15": ["join", "merge", "key", "integrity"],
        "P16": ["time series", "order", "sort", "lag"],
        "P17": ["drift", "stability", "monitor"],
        "P18": ["noise", "smoothing", "filter"],
        "P19": ["text", "clean", "tokenize", "tf-idf", "embedding"],
        "P20": ["image", "resize", "normalize", "augment"],
        "P21": ["audio", "mfcc", "spectrogram"],
        "P22": ["referential", "foreign key", "integrity"],
        "P23": ["frequency", "resample", "aggregate"],
        "P24": ["gap", "interpolate", "missing period"],
        "P25": ["rare", "other", "group"],
        "P26": ["constant", "variance", "drop"],
        "P27": ["noisy target", "clean", "filter"],
        "P28": ["granularity", "aggregate", "unit"],
        "P29": ["indirect leakage", "proxy", "remove"],
        "P30": ["split", "leakage", "time", "stratify"],
        "P31": ["high dimensional", "feature selection", "pca"],
        "P32": ["interaction", "feature crossing", "polynomial"],
        "P33": ["inf", "nan", "special values", "replace"],
        "P34": ["precision", "decimal", "round"],
        "P35": ["cross-validation", "cv", "timeseries split", "group kfold"],
    }

    keywords = keyword_map.get(problem_id)
    if keywords:
        matched = _match_solution(solutions, keywords)
        if matched:
            return matched

    # LLM fallback: ask to choose best solution by use_case/method
    llm = _get_llm()
    prompt = f"""
Select the best solution for {problem_id} based on the dataset context.
CSV stats: {json.dumps(state.get('csv_stats', {}), indent=2)}
Return ONLY the index (0-based) of the best solution.
Solutions: {json.dumps(solutions, indent=2)}
"""
    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        index = int(re.findall(r"\\d+", response.content)[0])
        if 0 <= index < len(solutions):
            return solutions[index]
    except Exception:
        pass

    return solutions[0]


def _match_solution(solutions: List[Dict], keywords: List[str]) -> Dict | None:
    """Find a solution whose method/use_case mentions any keyword."""
    for solution in solutions:
        hay = f"{solution.get('method','')} {solution.get('use_case','')} {solution.get('description','')}".lower()
        if any(k in hay for k in keywords):
            return solution
    return solutions[0] if solutions else None


def show_examples_node(state: StudentState) -> StudentState:
    """Show educational code examples (no execution)."""
    problem_id = state.get("current_problem")
    problem_detail = lookup_problem(problem_id) if problem_id else None

    output = f"CODE EXAMPLE for {problem_id}:\n\n"
    output += "IMPORTANT: This is EDUCATIONAL code.\n"
    output += "   Copy it to your environment and run it there.\n"
    output += "   I will not execute anything here.\n\n"
    output += "---\n\n"

    if problem_detail:
        solutions = (
            problem_detail.get("branches", [{}])[0].get("solutions", [])
        )
        if solutions:
            solution = solutions[0]
            output += f"**Method: {solution.get('method', 'Method 1')}**\n"
            output += f"Use case: {solution.get('use_case', 'When to use')}\n\n"
            output += f"```python\n{solution.get('code', '# Code not available')}\n```\n\n"
            output += "**Pros:**\n"
            for pro in solution.get("pros", []):
                output += f"  {pro}\n"
            output += "\n**Cons:**\n"
            for con in solution.get("cons", []):
                output += f"  {con}\n"

    output += "\n---\n\n"
    output += "Next step: run the code locally and come back with the result.\n"

    state["conversation"].append({"role": "assistant", "content": output, "timestamp": _now()})
    state["last_response"] = output
    state["last_action"] = "show_examples"
    state["timestamp_last_update"] = _now()

    return state


def ask_reflection_node(state: StudentState) -> StudentState:
    """Ask a reflection question to check understanding."""
    problem_id = state.get("current_problem")
    questions = {
        "P01": "Why is it important to handle missing values before training a model?",
        "P02": "How would you distinguish exact duplicates from repeated real events?",
        "P03": "Are these outliers valid business cases or data errors? Give a concrete example.",
        "P09": "Why is accuracy misleading with imbalanced classes?",
        "P14": "Could you use 'lucro_realizado' to predict sales in production?",
    }
    question = questions.get(problem_id, f"What did you learn about {problem_id}?")

    output = f"\nReflection question:\n\n**{question}**\n\nTell me your answer."
    state["conversation"].append({"role": "assistant", "content": output, "timestamp": _now()})
    state["last_action"] = "ask_reflection"
    state["timestamp_last_update"] = _now()

    return state


def validate_understanding_node(state: StudentState) -> StudentState:
    """Use LLM to assess whether the student understood the concept."""
    problem_id = state.get("current_problem")

    prompt = f"""
Analyze the student's conversation about {problem_id}.

Conversation (last 4 messages):
{json.dumps(state['conversation'][-4:], ensure_ascii=False, indent=2)}

Judge whether the student UNDERSTOOD the concept.

Respond ONLY in JSON:
{{
  "understood": true/false,
  "confidence": 0.0-1.0,
  "level": "beginner|intermediate|advanced",
  "feedback": "brief explanation why"
}}
"""

    llm = _get_llm()
    response = llm.invoke([{"role": "user", "content": prompt}])

    try:
        result = json.loads(response.content)
    except Exception:
        result = {
            "understood": True,
            "confidence": 0.5,
            "level": "intermediate",
            "feedback": "Ambiguous response",
        }

    state["last_validation_result"] = result
    state["understanding_level"] = result.get("level", "beginner")

    if result.get("understood"):
        if problem_id and problem_id not in state["problems_solved"]:
            state["problems_solved"].append(problem_id)
        feedback = (
            f"Great! You understood {problem_id}!\n\n"
            "Apply the fixes locally and upload the updated CSV "
            "so we can revalidate the next issue.\n\n"
        )
        state["reupload_required"] = True
    else:
        feedback = "It seems there are still doubts. Want me to explain again?\n\n"

    state["conversation"].append(
        {"role": "assistant", "content": feedback + result.get("feedback", ""), "timestamp": _now()}
    )

    state["last_response"] = feedback + result.get("feedback", "")
    state["last_action"] = "validate"
    state["timestamp_last_update"] = _now()

    return state


def validate_no_code_exec_node(state: StudentState) -> StudentState:
    """Guardrail to avoid claims of execution or system commands."""
    last_message = state.get("last_response", "")

    for pattern in _FORBIDDEN_EXECUTION_PATTERNS:
        if pattern.lower() in last_message.lower():
            state["guardrail_failed"] = True
            state["guardrail_reason"] = f"Forbidden pattern detected: {pattern}"
            state["guardrail_history"].append(
                {"when": _now().isoformat(), "reason": state["guardrail_reason"], "action": "flagged"}
            )
            return state

    for regex in _EXECUTION_CLAIM_PATTERNS:
        if re.search(regex, last_message, flags=re.IGNORECASE):
            state["guardrail_failed"] = True
            state["guardrail_reason"] = "Execution claim detected"
            state["guardrail_history"].append(
                {"when": _now().isoformat(), "reason": state["guardrail_reason"], "action": "flagged"}
            )
            return state

    state["guardrail_failed"] = False
    state["guardrail_reason"] = None
    return state


def mark_problem_solved_node(state: StudentState) -> StudentState:
    """Mark the current problem as solved and update checklist."""
    problem_id = state.get("current_problem")
    if not problem_id:
        return state

    if problem_id not in state["problems_solved"]:
        state["problems_solved"].append(problem_id)

    problem_detail = lookup_problem(problem_id)
    chk_id = problem_detail.get("checklist_ref") if problem_detail else None
    if chk_id:
        state["checklist_status"][chk_id] = True

    followup = (
        "\n\nAfter applying fixes locally, upload the updated dataset "
        "so we can revalidate and pick the next issue."
    )
    state["last_response"] = (state.get("last_response", "") + followup).strip()
    state["reupload_required"] = True
    state["last_action"] = "mark_solved"
    state["timestamp_last_update"] = _now()
    return state


def route_next_step(state: StudentState) -> str:
    """Route to the next node based on validation and remaining problems."""
    if state.get("guardrail_failed"):
        return "retry_explain"

    validation = state.get("last_validation_result")
    if validation and not validation.get("understood"):
        if state["attempts_current_problem"] < 3:
            return "explain_again"
        return "mark_solved"

    problems_remaining = [
        p
        for p in state["problems_detected"]
        if p["problem_id"] not in state["problems_solved"]
    ]

    if not problems_remaining:
        return "congratulations"

    state["current_problem"] = problems_remaining[0]["problem_id"]
    state["attempts_current_problem"] = 0
    return "explain_next"
