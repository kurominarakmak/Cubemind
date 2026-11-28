import pytest
import torch

from cubemind.environment.cube import COLORS, SOLVED_STATE, cross_frame_rotation
from cubemind.models.solver import SolverPolicyValue
from cubemind.models.zbls_policy import ZBLSPolicyValue
from cubemind.models.hierarchical_f2l import HierarchicalF2LPolicy
from cubemind.v2_planner import solve_v2, solve_v2_profiles
from cubemind.benchmark_v2_planner import _paired_routes


@pytest.mark.parametrize("cross_color", [COLORS["D"], COLORS["U"]])
def test_global_planner_only_selects_replay_verified_candidates(cross_color):
    result = solve_v2(
        SOLVED_STATE, None, 0, HierarchicalF2LPolicy(), SolverPolicyValue(),
        ZBLSPolicyValue(), torch.device("cpu"), cross_color=cross_color,
    )
    assert result["status"] == "SOLVED"
    assert result["selected"]["verified"]
    assert result["selected"]["moves"] == list(cross_frame_rotation(SOLVED_STATE, cross_color))
    assert all(candidate["verified"] for candidate in result["candidates"])
    assert result["cross_plus_one_pairs"] == 4


def test_paired_benchmark_reports_exact_discordance_and_common_success_moves():
    rows = [
        {"candidates": [
            {"policy": "cfop_oll_pll", "verified": True, "stm": 10},
            {"policy": "cfop_eo_aware", "verified": False, "stm": 0},
        ]},
        {"candidates": [
            {"policy": "cfop_oll_pll", "verified": False, "stm": 0},
            {"policy": "cfop_eo_aware", "verified": True, "stm": 12},
        ]},
        {"candidates": [
            {"policy": "cfop_oll_pll", "verified": True, "stm": 15},
            {"policy": "cfop_eo_aware", "verified": True, "stm": 12},
        ]},
        {"candidates": []},
    ]
    paired = _paired_routes(rows)["cfop_oll_pll_vs_cfop_eo_aware"]
    assert paired["left_only_success"] == 1
    assert paired["right_only_success"] == 1
    assert paired["both_success"] == 1
    assert paired["neither_success"] == 1
    assert paired["exact_mcnemar_two_sided_p"] == 1.0
    assert paired["mean_stm_difference_left_minus_right"] == 3
    assert paired["paired_stm_sign_test"]["right_shorter"] == 1


def test_symbolic_profile_planner_replays_all_six_cross_frames():
    result = solve_v2_profiles(
        SOLVED_STATE, None, 0, None, None, None, torch.device("cpu"),
        cross_colors=list(range(6)), last_layer_engine="algorithm_memory",
        zbls_engine="disabled", include_zbls=False,
    )
    assert result["status"] == "SOLVED"
    assert result["cross_colors_evaluated"] == list(range(6))
    assert result["selected"]["cross_color"] in range(6)
    assert all(candidate["verified"] for candidate in result["candidates"])
