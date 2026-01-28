from nodes import validate_no_code_exec_node
from state import create_initial_state


def test_guardrail_blocks_forbidden_patterns():
    state = create_initial_state(
        student_id="test",
        session_id="test",
        csv_filename="test.csv",
        csv_stats={},
    )
    state["last_response"] = "Use os.system('rm -rf /') to clean."

    result = validate_no_code_exec_node(state)

    assert result["guardrail_failed"] is True
    assert result["guardrail_reason"]


def test_guardrail_allows_safe_response():
    state = create_initial_state(
        student_id="test",
        session_id="test",
        csv_filename="test.csv",
        csv_stats={},
    )
    state["last_response"] = "Explique o conceito sem executar nada."

    result = validate_no_code_exec_node(state)

    assert result["guardrail_failed"] is False
