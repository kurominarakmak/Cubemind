"""Source-aware policy fusion for hierarchical F2L search."""

import torch
import torch.nn.functional as F

from cubemind.environment.cube import COLORS, SIMULATOR_MOVE_TOKENS
from cubemind.models.f2l import F2LPolicy
from cubemind.models.expert_pair import ExpertPairSelector
from cubemind.models.hierarchical_f2l import HIERARCHICAL_F2L_FEATURE_SIZE

EXPERT_FEATURE_SIZE = 54 * 6 + 8 * len(SIMULATOR_MOVE_TOKENS) + 4 + 5 + 6 + 4
ACTION_COUNT = len(SIMULATOR_MOVE_TOKENS)


def encode_expert_features(hierarchical_features):
    """Map the shared white-Down F2L state/history into the reco.nz context schema."""
    if hierarchical_features.ndim != 2 or hierarchical_features.shape[1] != HIERARCHICAL_F2L_FEATURE_SIZE:
        raise ValueError("Expected batched hierarchical F2L features")
    result = hierarchical_features.new_zeros((len(hierarchical_features), EXPERT_FEATURE_SIZE))
    result[:, :900] = hierarchical_features[:, :900]
    result[:, 902] = 1.0  # F2L phase.
    result[:, 904] = 1.0  # Standard F2L context; no unobserved strategy is inferred.
    result[:, 909 + COLORS["D"]] = 1.0  # The planner uses a white-Down Cross.
    result[:, 915] = hierarchical_features[:, 913]  # Edge orientation.
    result[:, 916] = hierarchical_features[:, 912]  # Cross complete.
    result[:, 917] = hierarchical_features[:, 914]  # Solved-slot fraction.
    result[:, 918] = (hierarchical_features[:, 915] * 2.0).clamp(max=1.0)  # /120 to /60.
    return result


def load_expert_action_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = checkpoint.get("model_config", {})
    if (checkpoint.get("sample_kind") != "expert"
            or checkpoint.get("data_source") != "reco.nz 3x3 CFOP reconstructions"
            or config.get("input_size") != EXPERT_FEATURE_SIZE
            or config.get("actions") != ACTION_COUNT):
        raise ValueError("Checkpoint is not the expected reco.nz F2L action model")
    model = F2LPolicy(EXPERT_FEATURE_SIZE, ACTION_COUNT).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def load_canonical_expert_action_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = checkpoint.get("model_config", {})
    if (checkpoint.get("sample_kind") != "expert"
            or checkpoint.get("data_source") != "reco.nz 3x3 CFOP reconstructions"
            or config.get("input_size") != HIERARCHICAL_F2L_FEATURE_SIZE
            or config.get("actions") != ACTION_COUNT):
        raise ValueError("Checkpoint is not the expected canonical reco.nz F2L policy")
    model = F2LPolicy(HIERARCHICAL_F2L_FEATURE_SIZE, ACTION_COUNT).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def load_expert_pair_selector(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if (checkpoint.get("sample_kind") != "expert"
            or checkpoint.get("data_source") != "reco.nz 3x3 CFOP reconstructions"
            or checkpoint.get("feature_size") != HIERARCHICAL_F2L_FEATURE_SIZE
            or checkpoint.get("actions") != 4):
        raise ValueError("Checkpoint is not the expected expert pair selector")
    model = ExpertPairSelector().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


class ExpertAwareF2LPolicy(torch.nn.Module):
    """Fuse action probabilities while retaining the planner's cost/value head."""

    def __init__(self, planner, expert, expert_weight=0.5, planner_weight=0.5,
                 expert_temperature=1.0, planner_temperature=1.0,
                 expert_pair_selector=None, canonical_expert=False):
        super().__init__()
        if expert_weight < 0 or planner_weight < 0 or expert_weight + planner_weight <= 0:
            raise ValueError("Policy weights must be nonnegative and not both zero")
        if expert_temperature <= 0 or planner_temperature <= 0:
            raise ValueError("Policy temperatures must be positive")
        self.planner = planner
        self.expert = expert
        self.expert_pair_selector = expert_pair_selector
        self.canonical_expert = bool(canonical_expert)
        total = expert_weight + planner_weight
        self.expert_weight = expert_weight / total
        self.planner_weight = planner_weight / total
        self.expert_temperature = float(expert_temperature)
        self.planner_temperature = float(planner_temperature)

    def forward(self, features):
        planner_logits, values, pair_logits = self.planner(features)
        expert_features = features if self.canonical_expert else encode_expert_features(features)
        expert_logits = self.expert(expert_features)
        expected = (len(features), ACTION_COUNT)
        if planner_logits.shape != expected or expert_logits.shape != expected:
            raise ValueError("Expert and planner action vocabularies must match the simulator")
        if not torch.isfinite(planner_logits).all() or not torch.isfinite(expert_logits).all():
            raise ValueError("Expert-aware policy returned non-finite action scores")
        fused_log_probs = (
            self.expert_weight * F.log_softmax(expert_logits / self.expert_temperature, dim=-1)
            + self.planner_weight * F.log_softmax(planner_logits / self.planner_temperature, dim=-1)
        )
        fused_logits = F.log_softmax(fused_log_probs, dim=-1)
        if self.expert_pair_selector is not None:
            pair_logits = self.expert_pair_selector(features)
        return fused_logits, values, pair_logits
