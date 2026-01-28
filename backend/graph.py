from __future__ import annotations

from langgraph.graph import END, StateGraph

from nodes import (
    analyze_csv_node,
    ask_reflection_node,
    explain_problem_node,
    mark_problem_solved_node,
    route_next_step,
    show_examples_node,
    show_problems_node,
    validate_no_code_exec_node,
    validate_understanding_node,
)
from state import StudentState


def analyze_and_show_node(state: StudentState) -> StudentState:
    """Run analysis and immediately show problems (first response)."""
    state = analyze_csv_node(state)
    return show_problems_node(state)


def congratulations_node(state: StudentState) -> StudentState:
    """Final message when all problems are solved."""
    message = (
        "🎉 Parabéns! Você concluiu todos os problemas detectados.\n\n"
        "✅ Próximos passos:\n"
        "1. Aplique as soluções no seu ambiente\n"
        "2. Reanalise seu dataset\n"
        "3. Compartilhe resultados com seu instrutor\n"
    )
    state["conversation"].append({"role": "assistant", "content": message})
    state["last_response"] = message
    state["last_action"] = "congratulations"
    return state


def _step_router(state: StudentState) -> str:
    """Decide which step to run based on the current state."""
    last_action = state.get("last_action", "init")
    last_message = state.get("conversation", [])[-1] if state.get("conversation") else None
    has_user_reply = bool(last_message and last_message.get("role") == "user")
    last_user_text = (last_message.get("content", "").lower() if has_user_reply else "")

    wants_examples = any(
        term in last_user_text
        for term in ["exemplo", "exemplos", "codigo", "código", "code", "example", "ejemplo", "exemple"]
    )

    if last_action in {"init", "upload"}:
        return "analyze_and_show"

    if last_action == "analyze":
        return "show_problems"

    if last_action == "show_problems":
        if not state.get("current_problem"):
            return "await_user"
        if wants_examples:
            state["last_action"] = "show_examples"
            return "show_examples"
        return "explain_problem"

    if last_action == "explain":
        return "show_examples" if wants_examples else "await_user"

    if last_action == "show_examples":
        return "ask_reflection"

    if last_action == "ask_reflection":
        return "validate_understanding" if has_user_reply else "await_user"

    if last_action == "validate":
        decision = route_next_step(state)
        if decision == "mark_solved":
            return "mark_solved"
        if decision == "congratulations":
            return "congratulations"
        if decision in {"retry_explain", "explain_again", "explain_next"}:
            return "explain_problem"
        return "await_user"

    return "await_user"


def create_agent_graph():
    """Build and compile the LangGraph flow with step routing."""
    graph = StateGraph(StudentState)

    graph.add_node("step_router", lambda state: state)
    graph.add_node("analyze_csv", analyze_csv_node)
    graph.add_node("analyze_and_show", analyze_and_show_node)
    graph.add_node("show_problems", show_problems_node)
    graph.add_node("explain_problem", explain_problem_node)
    graph.add_node("show_examples", show_examples_node)
    graph.add_node("ask_reflection", ask_reflection_node)
    graph.add_node("validate_understanding", validate_understanding_node)
    graph.add_node("validate_no_code", validate_no_code_exec_node)
    graph.add_node("mark_solved", mark_problem_solved_node)
    graph.add_node("congratulations", congratulations_node)

    graph.set_entry_point("step_router")

    graph.add_conditional_edges(
        "step_router",
        _step_router,
        {
            "analyze_csv": "analyze_csv",
            "analyze_and_show": "analyze_and_show",
            "show_problems": "show_problems",
            "explain_problem": "explain_problem",
            "ask_reflection": "ask_reflection",
            "validate_understanding": "validate_understanding",
            "mark_solved": "mark_solved",
            "congratulations": "congratulations",
            "await_user": END,
        },
    )

    graph.add_edge("analyze_csv", END)
    graph.add_edge("analyze_and_show", END)
    graph.add_edge("show_problems", END)
    graph.add_edge("explain_problem", "validate_no_code")
    graph.add_edge("validate_no_code", END)
    graph.add_edge("show_examples", END)
    graph.add_edge("ask_reflection", END)
    graph.add_edge("validate_understanding", END)
    graph.add_edge("mark_solved", END)
    graph.add_edge("congratulations", END)

    graph.add_conditional_edges(
        "show_problems",
        lambda state: "show_examples" if "show_examples" in state.get("last_action", "") else "await_user",
        {
            "show_examples": "show_examples",
            "await_user": END,
        },
    )

    return graph.compile()


AGENT_GRAPH = create_agent_graph()
def analyze_and_show_node(state: StudentState) -> StudentState:
    """Run analysis and immediately show problems (first response)."""
    state = analyze_csv_node(state)
    return show_problems_node(state)
