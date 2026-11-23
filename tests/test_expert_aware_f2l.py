import torch
import torch.nn.functional as F

from cubemind.environment.cube import (
    COLORS, SIMULATOR_MOVE_TOKENS, SOLVED_STATE, apply_move, apply_sequence,
    cross_complete, f2l_complete, inverse_sequence,
)
import pytest

from cubemind.expert_aware_f2l import (
    EXPERT_FEATURE_SIZE, ExpertAwareF2LPolicy, encode_expert_features,
    load_canonical_expert_action_model, load_expert_action_model,
)
from cubemind.hierarchical_f2l import solve_hierarchical_f2l
from cubemind.models.f2l import F2LPolicy
from cubemind.models.hierarchical_f2l import HIERARCHICAL_F2L_FEATURE_SIZE
from cubemind.train_expert_aware_f2l import _action_metrics, _adaptive_action_metrics
from cubemind.models.hierarchical_f2l import encode_f2l_features


class ScriptedPlanner(torch.nn.Module):
    def __init__(self, start, route):
        super().__init__()
        self.next_move = {}
        current = tuple(start)
        for move in route:
            self.next_move[current] = SIMULATOR_MOVE_TOKENS.index(move)
            current = apply_move(current, move)

    def forward(self, features):
        states = features[:, :324].reshape(-1, 54, 6).argmax(dim=2).tolist()
        actions = torch.full((len(features), len(SIMULATOR_MOVE_TOKENS)), -20.0,
                             device=features.device)
        for index, state in enumerate(states):
            actions[index, self.next_move.get(tuple(state), 0)] = 20.0
        values = torch.zeros(len(features), device=features.device)
        pairs = torch.zeros((len(features), 4), device=features.device)
        pairs[:, 0] = 10.0
        return actions, values, pairs


class ScriptedExpert(torch.nn.Module):
    def __init__(self, start, route):
        super().__init__()
        self.next_move = {}
        current = tuple(start)
        for move in route:
            self.next_move[current] = SIMULATOR_MOVE_TOKENS.index(move)
            current = apply_move(current, move)

    def forward(self, features):
        states = features[:, :324].reshape(-1, 54, 6).argmax(dim=2).tolist()
        logits = torch.full((len(features), len(SIMULATOR_MOVE_TOKENS)), -20.0,
                            device=features.device)
        for index, state in enumerate(states):
            logits[index, self.next_move.get(tuple(state), 0)] = 20.0
        return logits


def test_hierarchical_features_map_to_reco_context_consistently():
    state = apply_sequence(SOLVED_STATE, "R U F'")
    history = [SIMULATOR_MOVE_TOKENS.index(move) for move in ("R", "U", "F'")]
    features = encode_f2l_features(state, history, 0b0010, 0, 12).unsqueeze(0)
    mapped = encode_expert_features(features)
    assert mapped.shape == (1, 919)
    assert torch.equal(mapped[:, :900], features[:, :900])
    assert mapped[0, 902] == 1  # F2L
    assert mapped[0, 904] == 1  # standard strategy context
    assert mapped[0, 909 + COLORS["D"]] == 1  # white Cross
    assert torch.equal(mapped[:, 915:918], features[:, 913:915].new_tensor([
        [features[0, 913], features[0, 912], features[0, 914]],
    ]))
    assert torch.isclose(mapped[0, 918], torch.tensor(0.2))


def test_probability_fusion_is_normalized_and_has_policy_endpoints():
    class FixedPlanner(torch.nn.Module):
        def forward(self, features):
            logits = torch.tensor([[0.0, 1.0] + [0.0] * 70], device=features.device).expand(len(features), -1)
            return logits, torch.zeros(len(features)), torch.zeros((len(features), 4))

    class FixedExpert(torch.nn.Module):
        def forward(self, features):
            return torch.tensor([[2.0, 0.0] + [0.0] * 70], device=features.device).expand(len(features), -1)

    features = encode_f2l_features(SOLVED_STATE, [], 0, None, 0).unsqueeze(0)
    planner = FixedPlanner()
    expert = FixedExpert()
    expert_only = ExpertAwareF2LPolicy(planner, expert, 1, 0)
    planner_only = ExpertAwareF2LPolicy(planner, expert, 0, 1)
    fused = ExpertAwareF2LPolicy(planner, expert, 0.5, 0.5)
    e_scores = expert_only(features)[0]
    p_scores = planner_only(features)[0]
    f_scores = fused(features)[0]
    assert torch.allclose(e_scores.exp().sum(-1), torch.ones(1))
    assert torch.allclose(p_scores.exp().sum(-1), torch.ones(1))
    assert torch.allclose(f_scores.exp().sum(-1), torch.ones(1))
    assert e_scores.argmax(-1).item() == 0
    assert p_scores.argmax(-1).item() == 1
    assert torch.allclose(F.log_softmax(f_scores, dim=-1), f_scores)


def test_expert_checkpoint_loader_validates_source_and_action_schema(tmp_path):
    checkpoint_path = tmp_path / "expert.pt"
    model = F2LPolicy(EXPERT_FEATURE_SIZE, len(SIMULATOR_MOVE_TOKENS))
    torch.save({
        "sample_kind": "expert", "data_source": "reco.nz 3x3 CFOP reconstructions",
        "model_config": {"input_size": EXPERT_FEATURE_SIZE,
                         "actions": len(SIMULATOR_MOVE_TOKENS)},
        "model_state": model.state_dict(),
    }, checkpoint_path)
    loaded, metadata = load_expert_action_model(checkpoint_path, torch.device("cpu"))
    assert metadata["sample_kind"] == "expert"
    assert loaded(torch.zeros(1, EXPERT_FEATURE_SIZE)).shape == (1, len(SIMULATOR_MOVE_TOKENS))
    torch.save({**torch.load(checkpoint_path, weights_only=True), "data_source": "synthetic"}, checkpoint_path)
    with pytest.raises(ValueError, match="reco.nz"):
        load_expert_action_model(checkpoint_path, torch.device("cpu"))


def test_canonical_expert_checkpoint_loader_preserves_hierarchical_schema(tmp_path):
    checkpoint_path = tmp_path / "canonical.pt"
    model = F2LPolicy(HIERARCHICAL_F2L_FEATURE_SIZE, len(SIMULATOR_MOVE_TOKENS))
    torch.save({
        "sample_kind": "expert", "data_source": "reco.nz 3x3 CFOP reconstructions",
        "model_config": {"input_size": HIERARCHICAL_F2L_FEATURE_SIZE,
                         "actions": len(SIMULATOR_MOVE_TOKENS)},
        "model_state": model.state_dict(),
    }, checkpoint_path)
    loaded, _metadata = load_canonical_expert_action_model(checkpoint_path, torch.device("cpu"))
    assert loaded(torch.zeros(1, HIERARCHICAL_F2L_FEATURE_SIZE)).shape == (1, len(SIMULATOR_MOVE_TOKENS))


def test_calibrated_masked_fusion_is_finite_at_weight_endpoints():
    expert = torch.zeros(3, len(SIMULATOR_MOVE_TOKENS))
    planner = torch.ones_like(expert)
    legal = torch.zeros_like(expert, dtype=torch.bool)
    legal[:, :4] = True
    labels = torch.tensor([0, 1, 2])
    metrics = _action_metrics(expert, planner, labels, legal, 1.0, 1.0, 0.0)
    adaptive = _adaptive_action_metrics(expert, planner, labels, legal, 1.0, 1.0, 0.0)
    assert metrics["fused"]["cross_entropy"] == pytest.approx(1.386294, abs=1e-5)
    assert adaptive["cross_entropy"] == pytest.approx(1.386294, abs=1e-5)


def test_expert_action_vocab_mismatch_is_rejected_before_search():
    class BadExpert(torch.nn.Module):
        def forward(self, features):
            return torch.zeros(len(features), len(SIMULATOR_MOVE_TOKENS) - 1)

    class Planner(torch.nn.Module):
        def forward(self, features):
            return (torch.zeros(len(features), len(SIMULATOR_MOVE_TOKENS)),
                    torch.zeros(len(features)), torch.zeros(len(features), 4))

    state = encode_f2l_features(SOLVED_STATE, [], 0, None, 0).unsqueeze(0)
    with pytest.raises(ValueError, match="vocabularies"):
        ExpertAwareF2LPolicy(Planner(), BadExpert(), 1, 0)(state)


def test_expert_action_policy_can_complete_and_replay_f2l_without_fallback():
    route = ["R", "U", "R'", "U'"]
    start = apply_sequence(SOLVED_STATE, inverse_sequence(route))
    assert cross_complete(start)
    planner = ScriptedPlanner(start, route)
    expert = ScriptedExpert(start, route)
    policy = ExpertAwareF2LPolicy(planner, expert, expert_weight=1, planner_weight=0)
    result = solve_hierarchical_f2l(
        start, policy, torch.device("cpu"), mode="search", width=1, branching=1,
        max_nodes=20, max_depth=8, max_moves=8, max_actions_per_pair=8,
        max_order_attempts=4, conditional_legal=True, execution_cost_weight=1.0,
    )
    replay = apply_sequence(start, result["moves"])
    assert result["verified"] and f2l_complete(replay)
    assert result["route"] == "hierarchical neural F2L; no deterministic fallback"


def test_negative_execution_cost_weight_is_rejected():
    class Planner(torch.nn.Module):
        def forward(self, features):
            return (torch.zeros(len(features), len(SIMULATOR_MOVE_TOKENS)),
                    torch.zeros(len(features)), torch.zeros(len(features), 4))

    with pytest.raises(ValueError, match="execution_cost_weight"):
        solve_hierarchical_f2l(SOLVED_STATE, Planner(), torch.device("cpu"),
                               execution_cost_weight=-0.1)
