"""Small state-only policy for human-demonstrated Cross actions."""

import torch
import torch.nn.functional as F

from cubemind.environment.cube import MOVE_TOKENS
from cubemind.models.f2l import F2LPolicy

FEATURE_SIZE = 54 * 6


class ExpertCrossPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = F2LPolicy(FEATURE_SIZE, len(MOVE_TOKENS))

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != FEATURE_SIZE:
            raise ValueError("Expected batched 54-facelet one-hot Cross features")
        return self.policy(features)


class ExpertCrossPolicyAdapter(torch.nn.Module):
    """Adapt the existing cross-search call signature to the human Cross classifier."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, states, _histories):
        if states.ndim != 2 or states.shape[1] != 54:
            raise ValueError("Expected batched 54-facelet cube states")
        features = F.one_hot(states.long(), num_classes=6).float().flatten(1)
        return self.model(features)


def load_expert_cross_policy(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if (checkpoint.get("task") != "expert_cross_action_policy"
            or checkpoint.get("sample_kind") != "expert"
            or checkpoint.get("data_source") != "reco.nz 3x3 CFOP reconstructions"
            or checkpoint.get("feature_size") != FEATURE_SIZE
            or checkpoint.get("actions") != len(MOVE_TOKENS)):
        raise ValueError("Checkpoint is not a supported human-only Cross policy")
    model = ExpertCrossPolicy().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return ExpertCrossPolicyAdapter(model).eval(), checkpoint
