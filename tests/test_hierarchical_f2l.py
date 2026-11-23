import pytest
import torch

from cubemind.benchmark_hierarchical_f2l import _summary
from cubemind.environment.cube import (
    SIMULATOR_MOVE_TOKENS, SOLVED_STATE, apply_move, apply_sequence,
    cross_complete, f2l_complete, inverse_sequence, solved_pair_mask,
)
from cubemind.hierarchical_f2l import solve_hierarchical_f2l
from cubemind.models.hierarchical_f2l import (
    HIERARCHICAL_F2L_FEATURE_SIZE, HierarchicalF2LPolicy, encode_f2l_features,
)
from cubemind.train_hierarchical_f2l import label_trajectory, SOURCE_TEACHER, MOVE_TO_ID


class ScriptedPairPolicy(torch.nn.Module):
    def __init__(self, start, route):
        super().__init__()
        self.next_move = {}
        current = tuple(start)
        for move in route:
            move_id = SIMULATOR_MOVE_TOKENS.index(move)
            self.next_move[current] = move_id
            current = apply_move(current, move)

    def forward(self, features):
        states = features[:, :324].reshape(-1, 54, 6).argmax(dim=2).tolist()
        actions = torch.full((len(features), len(SIMULATOR_MOVE_TOKENS)), -20.0, device=features.device)
        for index, state in enumerate(states):
            actions[index, self.next_move.get(tuple(state), 0)] = 20.0
        values = torch.zeros(len(features), device=features.device)
        pairs = torch.zeros((len(features), 4), device=features.device)
        pairs[:, 0] = 10.0
        return actions, values, pairs


def test_hierarchical_feature_schema_contains_masks_and_target_pair():
    state = apply_sequence(SOLVED_STATE, "U R U' R'")
    features = encode_f2l_features(state, [3, 4], 0b1010, 0, 12)
    assert features.shape == (HIERARCHICAL_F2L_FEATURE_SIZE,)
    assert features[900:904].tolist() == [0.0, 1.0, 1.0, 1.0]
    assert features[904:908].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert features[908:912].tolist() == [1.0, 0.0, 0.0, 0.0]
    actions, values, pairs = HierarchicalF2LPolicy()(features.unsqueeze(0))
    assert actions.shape == (1, len(SIMULATOR_MOVE_TOKENS))
    assert values.shape == (1,)
    assert pairs.shape == (1, 4)


def test_neural_guided_pair_search_replays_to_all_four_f2l_pairs():
    route = ["R", "U", "R'", "U'"]
    start = apply_sequence(SOLVED_STATE, inverse_sequence(route))
    assert cross_complete(start)
    assert solved_pair_mask(start) == (False, True, True, True)
    assert not cross_complete(apply_move(start, route[0]))
    policy = ScriptedPairPolicy(start, route)
    result = solve_hierarchical_f2l(
        start, policy, torch.device("cpu"), mode="search", width=1, branching=1,
        max_nodes=20, max_depth=8, max_moves=8, max_actions_per_pair=8,
        max_order_attempts=4,
    )
    assert result["verified"]
    assert result["moves"] == route[:3]
    assert cross_complete(result["last_state"])
    assert f2l_complete(result["last_state"])
    assert result["route"] == "hierarchical neural F2L; no deterministic fallback"


def test_end_to_end_cross_then_hierarchical_f2l_replays_from_scramble():
    route = ["R", "U", "R'", "U'"]
    cross_state = apply_sequence(SOLVED_STATE, inverse_sequence(route))
    start = apply_move(cross_state, "R'")
    assert not cross_complete(start)
    cross_moves = ["R"]
    cross_state = apply_sequence(start, cross_moves)
    assert cross_complete(cross_state)
    result = solve_hierarchical_f2l(
        cross_state, ScriptedPairPolicy(cross_state, route), torch.device("cpu"),
        mode="search", width=1, branching=1, max_nodes=20, max_depth=8,
        max_moves=8, max_actions_per_pair=8, max_order_attempts=4,
    )
    combined = cross_moves + result["moves"]
    final_state = apply_sequence(start, combined)
    assert result["verified"]
    assert cross_complete(final_state) and f2l_complete(final_state)


def test_hierarchical_solver_reports_partial_state_when_cross_is_missing():
    state = apply_sequence(SOLVED_STATE, "R")
    result = solve_hierarchical_f2l(state, HierarchicalF2LPolicy(), torch.device("cpu"))
    assert not result["verified"]
    assert result["moves"] == []
    assert result["failure_reason"] == "cross_not_solved"


def test_search_rejects_negative_value_weight():
    with pytest.raises(ValueError, match="value_weight"):
        solve_hierarchical_f2l(SOLVED_STATE, HierarchicalF2LPolicy(), torch.device("cpu"), value_weight=-0.1)


def test_benchmark_separates_cross_start_from_cross_preservation():
    rows = [
        {"cross_start_solved": True, "cross_complete": False, "f2l_complete": False,
         "initial_solved_pairs": 0, "final_pairs": 3, "pair_attempts": 5,
         "pair_destruction_events": 1, "nodes": 10, "runtime_seconds": 1.0},
        {"cross_start_solved": True, "cross_complete": True, "f2l_complete": True,
         "initial_solved_pairs": 0, "final_pairs": 4, "pair_attempts": 5,
         "pair_destruction_events": 2, "nodes": 20, "runtime_seconds": 2.0,
         "stm": 32, "pair_results": [{"verified": True, "moves": ["R", "U", "R'"]}]},
    ]
    summary = _summary(rows)
    assert summary["cross_success"] == 2
    assert summary["cross_preserved"] == 1
    assert summary["complete_f2l_rate"] == 0.5
    assert summary["complete_f2l_given_cross_rate"] == 0.5
    assert summary["recovery_attempt_cases"] == 2
    assert summary["recovery_successes"] == 1
    assert summary["pair_moves_median"] == 3


def test_training_pair_labels_require_simulator_verified_cross_and_slot_transition():
    route = ["R", "U", "R'", "U'"]
    start = apply_sequence(SOLVED_STATE, inverse_sequence(route))
    transitions = []
    current, history = start, []
    for index, move in enumerate(route):
        transitions.append({"state": current, "history": list(history), "move": move,
                           "move_count": index, "sample_kind": "synthetic"})
        history.append(MOVE_TO_ID[move])
        current = apply_move(current, move)
    actions, selectors = label_trajectory(transitions, SOURCE_TEACHER, "train", "unit-route")
    assert len(actions) == len(route) - 1
    assert len(selectors) == 1
    assert {row["target_pair"] for row in actions} == {0}
    assert not cross_complete(actions[1]["state"])
    assert selectors[0]["data_source"] == SOURCE_TEACHER
    assert selectors[0]["split"] == "train"
