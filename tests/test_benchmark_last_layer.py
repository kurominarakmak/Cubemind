from cubemind.benchmark_last_layer import _neural_full_row
from cubemind.environment.cube import SOLVED_STATE, apply_sequence
from cubemind.neural_solver import cube_solved


def test_full_rows_keep_cross_f2l_and_last_layer_trace():
    start = apply_sequence(SOLVED_STATE, "U")
    prefix = {
        "moves": [], "cross_moves": [], "f2l_moves": [], "last_state": start,
        "verified": True, "cross_complete": True, "f2l_complete": True,
        "search_nodes": 0, "runtime_seconds": 0.0, "failure_reason": None,
    }
    result = {
        "moves": ["U'"], "moves_by_phase": {"OLL": [], "PLL": ["U'"], "ZBLL": []},
        "phase_states": [("OLL", SOLVED_STATE), ("PLL", SOLVED_STATE)],
        "last_state": SOLVED_STATE, "verified": True, "states_explored": 1,
        "runtime_seconds": 0.1, "failure_reason": None, "recovery_used": False,
        "route": "OLL → PLL", "provenance": "test policy",
    }

    row = _neural_full_row("U", start, prefix, result, "test")

    assert row["verified"] and cube_solved(apply_sequence(start, row["moves"]))
    assert row["moves_by_phase"] == {
        "CROSS": [], "F2L": [], "OLL": [], "PLL": ["U'"], "ZBLL": [],
    }
    assert row["phase_order"] == ["CROSS", "F2L", "OLL", "PLL"]
    assert dict(row["phase_states"]) == {
        "CROSS": start, "F2L": start, "OLL": SOLVED_STATE, "PLL": SOLVED_STATE,
    }


def test_full_rows_keep_chronological_zbll_recovery_order():
    start = apply_sequence(SOLVED_STATE, "U")
    prefix = {
        "moves": [], "cross_moves": [], "f2l_moves": [], "last_state": start,
        "verified": True, "cross_complete": True, "f2l_complete": True,
        "search_nodes": 0, "runtime_seconds": 0.0, "failure_reason": None,
    }
    result = {
        "moves": ["U", "U'", "U'"],
        "moves_by_phase": {"OLL": ["U'"], "PLL": ["U'"], "ZBLL": ["U"]},
        "phase_states": [("ZBLL", apply_sequence(start, "U")),
                         ("OLL", start), ("PLL", SOLVED_STATE)],
        "last_state": SOLVED_STATE, "verified": True, "states_explored": 3,
        "runtime_seconds": 0.1, "failure_reason": None, "recovery_used": True,
        "route": "ZBLL → OLL → PLL neural recovery", "provenance": "test policy",
    }

    row = _neural_full_row("U", start, prefix, result, "eo-aware-search")

    assert row["verified"] and cube_solved(apply_sequence(start, row["moves"]))
    assert row["phase_order"] == ["CROSS", "F2L", "ZBLL", "OLL", "PLL"]
    assert [move for phase in row["phase_order"] for move in row["moves_by_phase"][phase]] == row["moves"]
