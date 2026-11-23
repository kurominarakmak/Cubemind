import json

from cubemind.environment.cube import (
    SOLVED_STATE,
    apply_sequence,
    cross_complete,
    f2l_complete,
    f2l_slot_count,
    last_layer_edges_oriented,
    normalize_cube_orientation,
)
from cubemind.last_layer import (
    ZBLL_PATH,
    recognize_zbll_case,
    solution_for_zbll_state,
    verify_zbll_database,
    zbll_case_state,
)


def test_exact_recognition_and_primary_algorithms_cover_all_493_cases():
    assert verify_zbll_database()["cases"] == 493
    for case in json.loads(ZBLL_PATH.read_text())["cases"]:
        state = zbll_case_state(case["case_id"])
        assert recognize_zbll_case(state) == case["case_id"]
        solution = solution_for_zbll_state(state)
        assert normalize_cube_orientation(apply_sequence(state, solution)) == SOLVED_STATE


def test_zbll_recognition_normalizes_auf_and_cube_orientation():
    case_id = "PLL-Ua"
    state = zbll_case_state(case_id)
    presented = apply_sequence(state, "U x y2")
    assert recognize_zbll_case(presented) == case_id
    solution = solution_for_zbll_state(presented)
    assert solution is not None
    assert normalize_cube_orientation(apply_sequence(presented, solution)) == SOLVED_STATE


def test_all_302_zbls_primary_algorithms_start_from_three_slots_and_orient_edges():
    database = json.loads((ZBLL_PATH.parent / "zbls_cases.json").read_text())
    assert len(database["cases"]) == 302
    for case in database["cases"]:
        state = apply_sequence(SOLVED_STATE, case["setup"])
        assert cross_complete(state)
        assert f2l_slot_count(state) == 3
        algorithm = case["algorithms"][0]
        result = normalize_cube_orientation(apply_sequence(state, algorithm))
        assert f2l_complete(result)
        assert last_layer_edges_oriented(result)
