from cubemind.benchmark_neurosymbolic import _mcnemar_p, _paired_comparison, _paired_modes


def test_exact_mcnemar_probability_for_four_one_sided_discordances():
    assert _mcnemar_p(0, 4) == 0.125


def test_neural_and_symbolic_routes_are_compared_on_paired_outcomes():
    rows = [
        {"neural": {"verified": True, "stm": 12,
                     "routes": {"cfop_oll_pll": {"verified": True, "stm": 12}}},
         "symbolic": {"verified": True, "stm": 10,
                       "routes": {"symbolic_cfop": {"verified": True, "stm": 10}}}},
        {"neural": {"verified": True, "stm": 14,
                     "routes": {"cfop_oll_pll": {"verified": True, "stm": 14}}},
         "symbolic": {"verified": False, "stm": None,
                       "routes": {"symbolic_cfop": {"verified": False, "stm": 0}}}},
        {"neural": {"verified": False, "stm": None,
                     "routes": {"cfop_oll_pll": {"verified": False, "stm": 0}}},
         "symbolic": {"verified": True, "stm": 11,
                       "routes": {"symbolic_cfop": {"verified": True, "stm": 11}}}},
    ]
    result = _paired_comparison(rows)["cfop_oll_pll"]
    assert result["neural_only_success"] == 1
    assert result["symbolic_only_success"] == 1
    assert result["exact_mcnemar_two_sided_p"] == 1
    assert result["both_success"] == 1
    assert result["common_success_count"] == 1
    assert result["mean_neural_minus_symbolic_stm_on_common_success"] == 2


def test_shortest_and_preference_full_solve_comparison_is_paired():
    rows = [
        {"shortest": {"verified": True, "stm": 50, "routes": {"symbolic_cfop": {"verified": True, "stm": 50}}},
         "preference": {"verified": True, "stm": 52, "routes": {"symbolic_cfop": {"verified": True, "stm": 52}}}},
        {"shortest": {"verified": True, "stm": 53, "routes": {"symbolic_cfop": {"verified": True, "stm": 53}}},
         "preference": {"verified": False, "stm": None, "routes": {"symbolic_cfop": {"verified": False, "stm": 0}}}},
    ]
    paired = _paired_modes(rows, "shortest", "preference")
    assert paired["selected_full_solve"]["left_only_success"] == 1
    assert paired["selected_full_solve"]["both_success"] == 1
    assert paired["selected_full_solve"]["exact_mcnemar_two_sided_p"] == 1
    assert paired["selected_full_solve"]["mean_left_minus_right_stm_on_common_success"] == -2
