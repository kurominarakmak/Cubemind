import torch

from cubemind.environment.cube import (
    MOVE_TOKENS, SIMULATOR_MOVE_TOKENS, SOLVED_STATE, apply_sequence, cross_complete, f2l_complete,
)
from cubemind.last_layer import solution_for_zbll_state, zbll_case_state
from cubemind.models.solver import SOLVER_FEATURE_SIZE
from cubemind.models.solver import SolverPolicyValue
from cubemind.neural_solver import (
    ExpertF2LPolicyAdapter, encode_features, load_last_layer_checkpoint, load_solver_checkpoint,
    oll_complete, solve_expert_aware_full, solve_last_layer, solve_neural,
)


class FixedMovePolicy(torch.nn.Module):
    def __init__(self, move):
        super().__init__()
        self.move_id = SIMULATOR_MOVE_TOKENS.index(move)

    def forward(self, features):
        logits = torch.full((len(features), len(SIMULATOR_MOVE_TOKENS)), -10.0, device=features.device)
        logits[:, self.move_id] = 10.0
        return logits, torch.zeros(len(features), device=features.device)


class FixedCrossPolicy(torch.nn.Module):
    def __init__(self, move):
        super().__init__()
        self.move_id = MOVE_TOKENS.index(move)

    def forward(self, states, histories):
        logits = torch.full((len(states), len(MOVE_TOKENS)), -10.0, device=states.device)
        logits[:, self.move_id] = 10.0
        return logits


class CapturingExpertPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = None

    def forward(self, features):
        self.features = features
        return torch.zeros((len(features), len(SIMULATOR_MOVE_TOKENS)), device=features.device)


class ZeroValuePolicy(torch.nn.Module):
    def forward(self, features):
        return torch.zeros((len(features), len(SIMULATOR_MOVE_TOKENS)), device=features.device), torch.ones(
            len(features), device=features.device,
        )


def test_phase_features_and_last_layer_goal_shape():
    features = encode_features(SOLVED_STATE, [], "ZBLL", 0, torch.device("cpu"))
    assert features.shape == (SOLVER_FEATURE_SIZE,)
    assert oll_complete(SOLVED_STATE)
    color_features = encode_features(SOLVED_STATE, [], "ZBLL", 0, torch.device("cpu"), cross_color=0)
    assert color_features[910] == 1.0 and features[910 + 3] == 1.0


def test_last_layer_inference_preserves_explicit_move_count_and_history():
    class CapturingPolicy(FixedMovePolicy):
        def forward(self, features):
            self.features = features.detach().clone()
            return super().forward(features)

    policy = CapturingPolicy("U'")
    result = solve_last_layer(apply_sequence(SOLVED_STATE, "U"), policy, torch.device("cpu"),
                              mode="greedy", strategy="oll-pll",
                              history=[SIMULATOR_MOVE_TOKENS.index("R")], move_count=30)

    assert result["verified"]
    history_index = 54 * 6 + 7 * len(SIMULATOR_MOVE_TOKENS) + SIMULATOR_MOVE_TOKENS.index("R")
    assert policy.features[0, history_index] == 1.0
    assert policy.features[0, 919] == 0.5


def test_neural_solver_checkpoint_loads_with_safe_torch_deserialization(tmp_path):
    checkpoint_path = tmp_path / "solver.pt"
    torch.save({
        "sample_kind": "synthetic", "data_source": "synthetic:cfop-teacher+dagger+zbll-algorithms",
        "feature_size": SOLVER_FEATURE_SIZE, "actions": len(SIMULATOR_MOVE_TOKENS),
        "model_state": SolverPolicyValue().state_dict(),
    }, checkpoint_path)
    model, _checkpoint = load_solver_checkpoint(checkpoint_path, torch.device("cpu"))
    assert isinstance(model, SolverPolicyValue)


def test_last_layer_checkpoint_loads_with_safe_torch_deserialization(tmp_path):
    checkpoint_path = tmp_path / "last-layer.pt"
    torch.save({
        "sample_kind": "mixed:expert+synthetic", "feature_size": SOLVER_FEATURE_SIZE,
        "actions": len(SIMULATOR_MOVE_TOKENS), "phase_names": ("OLL", "PLL", "ZBLL"),
        "model_state": SolverPolicyValue().state_dict(),
    }, checkpoint_path)
    model, _checkpoint = load_last_layer_checkpoint(checkpoint_path, torch.device("cpu"))
    assert isinstance(model, SolverPolicyValue)


def test_reco_expert_adapter_maps_shared_features_to_human_f2l_context():
    expert = CapturingExpertPolicy()
    adapter = ExpertF2LPolicyAdapter(expert, ZeroValuePolicy())
    features = encode_features(apply_sequence(SOLVED_STATE, "R U R'"), [], "F2L", 7,
                               torch.device("cpu")).unsqueeze(0)
    logits, values = adapter(features)
    assert logits.shape == (1, len(SIMULATOR_MOVE_TOKENS))
    assert values.tolist() == [1.0]
    assert expert.features.shape == (1, 919)
    assert expert.features[0, 902] == 1.0  # F2L in the reco checkpoint's phase order.
    assert expert.features[0, 904] == 1.0  # Standard F2L strategy context.
    assert expert.features[0, 912] == 1.0  # White cross color.


def test_neural_greedy_and_search_replay_their_own_moves_to_solved(monkeypatch):
    monkeypatch.setattr("cubemind.neural_solver.teacher_request",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("no reference fallback")))
    start = apply_sequence(SOLVED_STATE, "R")
    policy = FixedMovePolicy("R'")
    for mode in ("greedy", "search"):
        result = solve_neural(start, policy, torch.device("cpu"), mode=mode,
                              beam_width=2, branching=1, max_nodes=20)
        assert result["status"] == "SOLVED"
        assert result["verified"]
        assert result["moves"] == ["R'"]
        assert cross_complete(result["last_state"])
        assert result["states_explored"] > 0


def test_neural_policy_returns_explicit_failure_at_phase_budget():
    start = apply_sequence(SOLVED_STATE, "R")
    result = solve_neural(start, FixedMovePolicy("U"), torch.device("cpu"),
                          mode="greedy", budgets={"CROSS": 1})
    assert result["status"] == "FAILED / INCOMPLETE"
    assert not result["verified"]
    assert result["failure_reason"] == "phase_move_budget:CROSS"
    assert result["moves"] == ["U"]


def test_neural_greedy_stops_on_a_repeated_state():
    start = apply_sequence(SOLVED_STATE, "R")
    result = solve_neural(start, FixedMovePolicy("U2"), torch.device("cpu"),
                          mode="greedy", budgets={"CROSS": 5})
    assert result["status"] == "FAILED / INCOMPLETE"
    assert result["failure_reason"] == "repeated_state:CROSS"
    assert len(result["moves"]) == 2


def test_extended_pipeline_uses_the_learned_cross_model_for_greedy_and_search():
    start = apply_sequence(SOLVED_STATE, "R")
    phase_policy = FixedMovePolicy("U")
    cross_policy = FixedCrossPolicy("R'")
    for mode in ("greedy", "search"):
        result = solve_neural(start, phase_policy, torch.device("cpu"), mode=mode,
                              cross_model=cross_policy, cross_history_length=0,
                              cross_search_width=4, cross_search_depth=3,
                              beam_width=2, branching=2, max_nodes=10)
        assert result["verified"]
        assert result["moves"] == ["R'"]


def test_extended_pipeline_uses_reco_policy_for_f2l_when_provided():
    class RecordingPolicy(FixedMovePolicy):
        def __init__(self):
            super().__init__("U")
            self.called = False

        def forward(self, features):
            self.called = True
            return super().forward(features)

    expert_policy = RecordingPolicy()
    result = solve_neural(
        apply_sequence(SOLVED_STATE, "R U R'"), FixedMovePolicy("U"), torch.device("cpu"),
        mode="greedy", budgets={"F2L": 1}, f2l_model=expert_policy,
    )
    assert not result["verified"]
    assert expert_policy.called
    assert result["failure_reason"] == "phase_move_budget:F2L"


def test_neural_last_layer_routes_eo_eligible_states_to_zbll(monkeypatch):
    monkeypatch.setattr("cubemind.neural_solver.teacher_request",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("no deterministic fallback")))
    start = apply_sequence(SOLVED_STATE, "U")
    result = solve_last_layer(start, FixedMovePolicy("U'"), torch.device("cpu"), mode="greedy")
    assert result["verified"]
    assert result["route"] == "ZBLL"
    assert result["moves_by_phase"]["ZBLL"] == ["U'"]


def test_neural_last_layer_can_run_conventional_oll_then_pll_route():
    start = apply_sequence(SOLVED_STATE, "U")
    result = solve_last_layer(start, FixedMovePolicy("U'"), torch.device("cpu"),
                              mode="greedy", strategy="oll-pll")
    assert result["verified"]
    assert result["route"] == "OLL → PLL"
    assert result["moves_by_phase"]["OLL"] == []
    assert result["moves_by_phase"]["PLL"] == ["U'"]


def test_neural_last_layer_failure_does_not_call_reference_teacher(monkeypatch):
    monkeypatch.setattr("cubemind.neural_solver.teacher_request",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("fallback used")))
    result = solve_last_layer(apply_sequence(SOLVED_STATE, "U"), FixedMovePolicy("R"),
                              torch.device("cpu"), mode="greedy", budgets={"ZBLL": 8},
                              allow_zbll_recovery=False)
    assert not result["verified"]
    assert result["failure_reason"] is not None


def test_failed_zbll_rolls_back_state_history_and_candidate_moves(monkeypatch):
    case_id = "T1-01"
    start = zbll_case_state(case_id)
    solution = solution_for_zbll_state(start, case_id)
    initial_history = [SIMULATOR_MOVE_TOKENS.index("F")]
    calls = []

    def fail_zbll_then_solve_oll(state, history, phase, _model, _device, *_args):
        calls.append((phase, tuple(state), list(history)))
        if phase == "ZBLL":
            candidate = apply_sequence(state, "R")
            assert not f2l_complete(candidate)
            return (candidate, list(history) + [SIMULATOR_MOVE_TOKENS.index("R")], ["R"], 3,
                    "search_budget:ZBLL")
        assert phase == "OLL"
        assert tuple(state) == start
        assert history == initial_history
        final = apply_sequence(state, solution)
        return (final, list(history) + [SIMULATOR_MOVE_TOKENS.index(move) for move in solution],
                solution, 7, None)

    monkeypatch.setattr("cubemind.neural_solver._solve_phase", fail_zbll_then_solve_oll)
    result = solve_last_layer(start, FixedMovePolicy("U"), torch.device("cpu"), mode="search",
                              history=initial_history, move_count=23)

    assert result["verified"]
    assert [phase for phase, _state, _history in calls] == ["ZBLL", "OLL"]
    assert result["moves_by_phase"]["ZBLL"] == []
    assert result["moves_by_phase"]["OLL"] == solution
    assert result["moves"] == solution
    assert result["states_explored"] == 10
    assert result["failed_attempts"][0]["candidate_moves"] == ["R"]
    assert result["failed_attempts"][0]["f2l_preserved"] is False
    assert apply_sequence(start, result["moves"]) == result["last_state"]


def test_value_only_search_uses_random_branches_instead_of_policy_logits():
    torch.manual_seed(0)
    result = solve_last_layer(
        apply_sequence(SOLVED_STATE, "U"), FixedMovePolicy("R"), torch.device("cpu"),
        mode="search", strategy="oll-pll", branching=len(SIMULATOR_MOVE_TOKENS),
        max_nodes=200, budgets={"PLL": 1}, action_guidance=False,
    )
    assert result["verified"]
    assert result["provenance"] == "learned-value-only search ablation; no action-policy guidance"


def test_last_layer_search_rejects_negative_action_score_weight():
    try:
        solve_last_layer(SOLVED_STATE, FixedMovePolicy("U"), torch.device("cpu"),
                         action_score_weight=-0.1)
    except ValueError as error:
        assert "search parameters must be valid" in str(error)
    else:
        raise AssertionError("negative action-score weights must be rejected")


def test_expert_aware_full_pipeline_replays_neural_last_layer():
    result = solve_expert_aware_full(
        apply_sequence(SOLVED_STATE, "U"), FixedMovePolicy("U'"), torch.device("cpu"),
        cross_model=None, cross_history_length=0, f2l_policy=None, mode="greedy",
    )
    assert result["verified"]
    assert result["moves"] == ["U'"]
    assert result["moves_by_phase"]["ZBLL"] == ["U'"]
