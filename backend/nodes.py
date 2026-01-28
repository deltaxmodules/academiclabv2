from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Dict, List

import os

from langchain_openai import ChatOpenAI

from analyze_csv import QuickAnalyzer
from data_access import lookup_problem
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

    severity_order = {"CRÍTICO": 0, "ALTO": 1, "MÉDIO": 2, "BAIXO": 3}
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
    output = "🎓 Olá! Analisei seu dataset e encontrei problemas:\n\n"

    by_severity: Dict[str, List[Dict]] = {}
    for problem in state["problems_detected"]:
        sev = problem.get("severity", "MÉDIO")
        by_severity.setdefault(sev, []).append(problem)

    for severity in ["CRÍTICO", "ALTO", "MÉDIO", "BAIXO"]:
        if severity not in by_severity:
            continue
        emoji = {"CRÍTICO": "🔴", "ALTO": "🟡", "MÉDIO": "🟢", "BAIXO": "⚪"}
        output += f"{emoji[severity]} {severity}:\n"
        for p in by_severity[severity]:
            output += f"  • {p['problem_id']}: {p.get('message', p.get('problem_name', ''))}\n"
        output += "\n"

    output += "❓ Qual problema gostaria de aprofundar primeiro?\n"
    output += "💡 Dica: Comece pelos CRÍTICOS!"

    state["conversation"].append({"role": "assistant", "content": output, "timestamp": _now()})
    state["last_response"] = output
    state["last_action"] = "show_problems"
    state["timestamp_last_update"] = _now()

    return state


def explain_problem_node(state: StudentState) -> StudentState:
    """Explain a specific problem using the LLM."""
    problem_id = state.get("current_problem")
    if not problem_id:
        output = "❌ Nenhum problema selecionado. Por favor, escolha um problema."
        state["conversation"].append({"role": "assistant", "content": output, "timestamp": _now()})
        state["last_response"] = output
        state["last_action"] = "explain"
        state["timestamp_last_update"] = _now()
        return state

    problem_detail = lookup_problem(problem_id)
    if not problem_detail:
        output = f"❌ Problema {problem_id} não encontrado no framework."
        state["conversation"].append({"role": "assistant", "content": output, "timestamp": _now()})
        state["last_response"] = output
        state["last_action"] = "explain"
        state["timestamp_last_update"] = _now()
        return state

    checklist_ref = problem_detail.get("checklist_ref", "CHK-001")

    system_prompt = """
Você é um PROFESSOR DE DATA SCIENCE especializado em educação.

REGRAS:
❌ NUNCA execute código ou acesse filesystem
❌ NUNCA diga que executou algo
✅ SEMPRE explique em 3 partes: O QUÊ, POR QUÊ, COMO
✅ Use analogias simples
✅ Faça perguntas reflexivas
✅ Cite o checklist item associado
"""

    user_prompt = f"""
Aluno quer aprender sobre {problem_id}: {problem_detail.get('name')}

Contexto:
- Nível de compreensão: {state['understanding_level']}
- Já resolveu: {state['problems_solved']}
- Tentativas neste problema: {state['attempts_current_problem']}

Framework:
{json.dumps(problem_detail, ensure_ascii=False, indent=2)}

Checklist: {checklist_ref}

Explique de forma educativa. O aluno vai tentar resolver por conta própria.
"""

    llm = _get_llm()
    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    explanation = response.content

    state["conversation"].append({"role": "assistant", "content": explanation, "timestamp": _now()})
    state["last_response"] = explanation
    state["last_action"] = "explain"
    state["attempts_current_problem"] += 1
    state["timestamp_last_update"] = _now()

    return state


def show_examples_node(state: StudentState) -> StudentState:
    """Show educational code examples (no execution)."""
    problem_id = state.get("current_problem")
    problem_detail = lookup_problem(problem_id) if problem_id else None

    output = f"🔧 EXEMPLO DE CÓDIGO para {problem_id}:\n\n"
    output += "⚠️ IMPORTANTE: Este é código EDUCACIONAL.\n"
    output += "   Copie para seu ambiente e EXECUTE lá.\n"
    output += "   Eu não vou executar nada aqui!\n\n"
    output += "---\n\n"

    if problem_detail:
        solutions = (
            problem_detail.get("branches", [{}])[0].get("solutions", [])
        )
        if solutions:
            solution = solutions[0]
            output += f"**Método: {solution.get('method', 'Método 1')}**\n"
            output += f"Uso: {solution.get('use_case', 'Quando usar')}\n\n"
            output += f"```python\n{solution.get('code', '# Código não disponível')}\n```\n\n"
            output += "**Pros:**\n"
            for pro in solution.get("pros", []):
                output += f"  ✅ {pro}\n"
            output += "\n**Cons:**\n"
            for con in solution.get("cons", []):
                output += f"  ❌ {con}\n"

    output += "\n---\n\n"
    output += "📝 Próximo passo: Copie o código, execute no seu PC, e volte aqui com o resultado!\n"

    state["conversation"].append({"role": "assistant", "content": output, "timestamp": _now()})
    state["last_response"] = output
    state["last_action"] = "show_examples"
    state["timestamp_last_update"] = _now()

    return state


def ask_reflection_node(state: StudentState) -> StudentState:
    """Ask a reflection question to check understanding."""
    problem_id = state.get("current_problem")
    questions = {
        "P01": "Por quê é importante tratar valores em falta antes de treinar um modelo?",
        "P02": "Como você diferenciaria entre duplicados exatos e eventos reais repetidos?",
        "P09": "Por quê usar acurácia é ruim quando temos classes desbalanceadas?",
        "P14": "Você conseguiria usar 'lucro_realizado' para prever vendas em produção?",
    }
    question = questions.get(problem_id, f"O que você aprendeu sobre {problem_id}?")

    output = f"\n🤔 Pergunta reflexiva:\n\n**{question}**\n\nMe diga sua resposta!"
    state["conversation"].append({"role": "assistant", "content": output, "timestamp": _now()})
    state["last_action"] = "ask_reflection"
    state["timestamp_last_update"] = _now()

    return state


def validate_understanding_node(state: StudentState) -> StudentState:
    """Use LLM to assess whether the student understood the concept."""
    problem_id = state.get("current_problem")

    prompt = f"""
Analise a conversa do aluno sobre {problem_id}.

Conversa (últimas 4 mensagens):
{json.dumps(state['conversation'][-4:], ensure_ascii=False, indent=2)}

Julgue se o aluno COMPREENDEU o conceito.

Responda APENAS em JSON:
{{
  "understood": true/false,
  "confidence": 0.0-1.0,
  "level": "beginner|intermediate|advanced",
  "feedback": "explicação breve do porquê"
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
            "feedback": "Resposta ambígua",
        }

    state["last_validation_result"] = result
    state["understanding_level"] = result.get("level", "beginner")

    if result.get("understood"):
        if problem_id and problem_id not in state["problems_solved"]:
            state["problems_solved"].append(problem_id)
        feedback = (
            f"✅ Parabéns! Você compreendeu {problem_id}!\n\n"
            "📤 Aplique as correções no seu ambiente e envie o CSV atualizado "
            "para revalidarmos o próximo problema.\n\n"
        )
        state["reupload_required"] = True
    else:
        feedback = "🤔 Parece que ainda há dúvidas. Quer que eu explique novamente?\n\n"

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
            state["guardrail_reason"] = f"Padrão proibido detectado: {pattern}"
            state["guardrail_history"].append(
                {"when": _now().isoformat(), "reason": state["guardrail_reason"], "action": "flagged"}
            )
            return state

    for regex in _EXECUTION_CLAIM_PATTERNS:
        if re.search(regex, last_message, flags=re.IGNORECASE):
            state["guardrail_failed"] = True
            state["guardrail_reason"] = "Reivindicação de execução detectada"
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
        "\n\n📤 Quando aplicar as correções no seu ambiente, faça upload do dataset atualizado "
        "para revalidarmos e escolher o próximo problema."
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
