from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Dict, List

import os

from langchain_openai import ChatOpenAI

from analyze_csv import QuickAnalyzer
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
    state["last_action"] = "analyze"
    state["timestamp_last_update"] = _now()

    return state


def show_problems_node(state: StudentState) -> StudentState:
    """Show detected problems grouped by severity."""
    output = "🎓 Hi! I analyzed your dataset and found issues:\n\n"

    by_severity: Dict[str, List[Dict]] = {}
    for problem in state["problems_detected"]:
        sev = problem.get("severity", "MEDIUM")
        by_severity.setdefault(sev, []).append(problem)

    for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        if severity not in by_severity:
            continue
        emoji = {"CRITICAL": "🔴", "HIGH": "🟡", "MEDIUM": "🟢", "LOW": "⚪"}
        output += f"{emoji[severity]} {severity}:\n"
        for p in by_severity[severity]:
            output += f"  • {p['problem_id']}: {p.get('message', p.get('problem_name', ''))}\n"
        output += "\n"

    output += "❓ Which issue would you like to explore first?\n"
    output += "💡 Tip: start with CRITICAL issues."

    state["conversation"].append({"role": "assistant", "content": output, "timestamp": _now()})
    state["last_response"] = output
    state["last_action"] = "show_problems"
    state["timestamp_last_update"] = _now()

    return state


def explain_problem_node(state: StudentState) -> StudentState:
    """Explain a specific problem using the LLM."""
    problem_id = state.get("current_problem")
    if not problem_id:
        output = "❌ No problem selected. Please choose one."
        state["conversation"].append({"role": "assistant", "content": output, "timestamp": _now()})
        state["last_response"] = output
        state["last_action"] = "explain"
        state["timestamp_last_update"] = _now()
        return state

    problem_detail = lookup_problem(problem_id)
    if not problem_detail:
        output = f"❌ Problem {problem_id} was not found in the framework."
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

    system_prompt = """
You are a DATA SCIENCE INSTRUCTOR focused on education.

RULES:
❌ Never execute code or access the filesystem
❌ Never claim you executed anything
✅ Always explain in 3 parts: WHAT, WHY, HOW
✅ Use simple analogies
✅ Ask reflective questions
✅ Cite the related checklist item

OUTPUT FORMAT:
1) WHAT is the issue
2) WHY it matters
3) HOW to address it (suggested actions, 2-3 steps)
4) Checklist item reference
"""

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

Explain in an educational way. The student will solve it on their own.
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
    if solutions:
        solution = solutions[0]
        examples_block += "\n\n---\n\n"
        examples_block += f"🔧 CODE EXAMPLE for {problem_id}:\n\n"
        examples_block += "⚠️ IMPORTANT: This is EDUCATIONAL code.\n"
        examples_block += "   Copy it to your environment and run it there.\n"
        examples_block += "   I will not execute anything here.\n\n"
        examples_block += f"**Method: {solution.get('method', 'Method 1')}**\n"
        examples_block += f"Use case: {solution.get('use_case', 'When to use')}\n\n"
        examples_block += f"```python\n{solution.get('code', '# Code not available')}\n```\n\n"
        examples_block += "**Pros:**\n"
        for pro in solution.get("pros", []):
            examples_block += f"  ✅ {pro}\n"
        examples_block += "\n**Cons:**\n"
        for con in solution.get("cons", []):
            examples_block += f"  ❌ {con}\n"
        examples_block += "\n---\n\n"
        examples_block += "📝 Next step: run the code locally and come back with the result.\n"

    combined = explanation + examples_block

    state["conversation"].append({"role": "assistant", "content": combined, "timestamp": _now()})
    state["last_response"] = combined
    state["last_action"] = "explain"
    state["attempts_current_problem"] += 1
    state["timestamp_last_update"] = _now()

    return state


def show_examples_node(state: StudentState) -> StudentState:
    """Show educational code examples (no execution)."""
    problem_id = state.get("current_problem")
    problem_detail = lookup_problem(problem_id) if problem_id else None

    output = f"🔧 CODE EXAMPLE for {problem_id}:\n\n"
    output += "⚠️ IMPORTANT: This is EDUCATIONAL code.\n"
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
                output += f"  ✅ {pro}\n"
            output += "\n**Cons:**\n"
            for con in solution.get("cons", []):
                output += f"  ❌ {con}\n"

    output += "\n---\n\n"
    output += "📝 Next step: run the code locally and come back with the result.\n"

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
        "P09": "Why is accuracy misleading with imbalanced classes?",
        "P14": "Could you use 'lucro_realizado' to predict sales in production?",
    }
    question = questions.get(problem_id, f"What did you learn about {problem_id}?")

    output = f"\n🤔 Reflection question:\n\n**{question}**\n\nTell me your answer."
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
            f"✅ Great! You understood {problem_id}!\n\n"
            "📤 Apply the fixes locally and upload the updated CSV "
            "so we can revalidate the next issue.\n\n"
        )
        state["reupload_required"] = True
    else:
        feedback = "🤔 It seems there are still doubts. Want me to explain again?\n\n"

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
        "\n\n📤 After applying fixes locally, upload the updated dataset "
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
