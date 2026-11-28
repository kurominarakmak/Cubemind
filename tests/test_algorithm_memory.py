import json

from cubemind.algorithm_memory import (
    OLL_PATH, algorithm_records, oll_candidates, recognize_oll_case,
    rank_algorithm_records, solve_symbolic_last_layer, solve_symbolic_zbls,
    verify_pll_database, pll_candidates,
    verify_algorithm_memory, verify_oll_database,
)
from cubemind.environment.cube import (
    SOLVED_STATE, apply_sequence, f2l_complete, inverse_sequence,
    last_layer_edges_oriented, normalize_cube_orientation,
)
from cubemind.last_layer import zbll_case_state
from cubemind.models.algorithm_preference import AlgorithmPreferenceRanker


def _oll_state(case):
    return normalize_cube_orientation(apply_sequence(
        SOLVED_STATE, inverse_sequence(case["algorithms"][0])))


def test_verified_algorithm_memory_covers_all_checked_in_families():
    report = verify_algorithm_memory()
    assert report["OLL"]["cases"] == 57
    assert report["OLL"]["verified_algorithms"] == 240
    assert report["ZBLL_PLL"]["cases"] == 493
    assert report["PLL"]["covered_cases"] == 21
    assert report["PLL"]["source_variants"] == 104
    assert report["PLL"]["verified_variants"] == 104
    assert report["ZBLS"]["cases"] == 302
    assert report["ZBLS"]["verified_algorithms"] == 302
    assert report["other_families"] == {"OLL": 240, "PLL": 125, "ZBLL": 472, "ZBLS": 302}
    assert all(row["observed_human_usage"] is None for row in algorithm_records())
    try:
        rank_algorithm_records(algorithm_records(), mode="human_preference")
    except ValueError as error:
        assert "supports only mode='shortest'" in str(error).lower()
    else:
        raise AssertionError("Human algorithm preferences must not be fabricated")


def test_oll_recognition_is_auf_and_whole_cube_orientation_invariant():
    case = json.loads(OLL_PATH.read_text())["cases"][0]
    state = _oll_state(case)
    presented = apply_sequence(state, "U x y2")
    assert recognize_oll_case(presented) == case["case_id"]
    candidates = oll_candidates(presented, case["case_id"])
    assert candidates
    assert all(candidates[i]["move_metrics"]["stm"] <= candidates[i + 1]["move_metrics"]["stm"]
               for i in range(len(candidates) - 1))


def test_every_oll_variant_is_replay_verified_and_oll_then_pll_solves():
    assert verify_oll_database()["algorithms"] == 240
    data = json.loads(OLL_PATH.read_text())
    state = _oll_state(data["cases"][0])
    pll = next(case for case in json.loads(
        (OLL_PATH.parent / "zbll_cases.json").read_text())["cases"] if case["case_id"] == "PLL-Ua")
    state = apply_sequence(state, inverse_sequence(pll["algorithms"][0]))
    solution = solve_symbolic_last_layer(state, strategy="oll-pll")
    assert solution["verified"]
    assert solution["route"] == "OLL → PLL"
    assert solution["oll_case_id"] == data["cases"][0]["case_id"]
    assert solution["pll_case_id"].startswith("PLL-")
    assert normalize_cube_orientation(apply_sequence(state, solution["moves"])) == SOLVED_STATE


def test_symbolic_oll_route_handles_pll_skip_with_final_auf():
    case = json.loads(OLL_PATH.read_text())["cases"][0]
    state = _oll_state(case)
    result = solve_symbolic_last_layer(state, strategy="oll-pll")
    assert result["verified"]
    assert result["pll_case_id"] == "solved"
    assert normalize_cube_orientation(apply_sequence(state, result["moves"])) == SOLVED_STATE


def test_symbolic_last_layer_uses_direct_zbll_or_pll_table_when_eligible():
    for case_id, phase in (("T1-01", "ZBLL"), ("PLL-Ua", "PLL")):
        state = zbll_case_state(case_id)
        result = solve_symbolic_last_layer(state)
        assert result["verified"]
        assert result["route"] == phase
        assert normalize_cube_orientation(apply_sequence(state, result["moves"])) == SOLVED_STATE
        if phase == "PLL":
            assert result["algorithm_candidates"]
            assert result["algorithm_candidates"][0]["source"].startswith("SpeedCubeDB")
            assert all(normalize_cube_orientation(apply_sequence(state, row["moves"])) == SOLVED_STATE
                       for row in pll_candidates(state, case_id))


def test_symbolic_last_layer_preference_ranking_preserves_replay_verification():
    state = zbll_case_state("PLL-Ua")
    case_ids = sorted({row["case_id"] for row in algorithm_records() if row["family"] in ("OLL", "PLL")})
    model = AlgorithmPreferenceRanker(len(case_ids))
    model.case_ids = tuple(case_ids)
    result = solve_symbolic_last_layer(state, preference_model=model)
    assert result["verified"]
    assert result["algorithm_candidates"]
    assert "preference_score" in result["algorithm_candidates"][0]
    assert normalize_cube_orientation(apply_sequence(state, result["moves"])) == SOLVED_STATE


def test_zbls_algorithm_candidates_require_verified_three_pair_to_eo_transition():
    case = json.loads((OLL_PATH.parent / "zbls_cases.json").read_text())["cases"][0]
    state = apply_sequence(SOLVED_STATE, case["setup"])
    result = solve_symbolic_zbls(state)
    assert result["verified"]
    assert result["case_id"] is not None
    final = apply_sequence(state, result["moves"])
    assert f2l_complete(final) and last_layer_edges_oriented(final)
