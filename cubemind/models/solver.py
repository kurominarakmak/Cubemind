"""Small shared policy/value MLP for synthetic CFOP solver experiments."""

import torch

from cubemind.environment.cube import SIMULATOR_MOVE_TOKENS

SOLVER_FEATURE_SIZE = 54 * 6 + 8 * len(SIMULATOR_MOVE_TOKENS) + 5 + 5 + 6 + 4


class SolverPolicyValue(torch.nn.Module):
    def __init__(self, actions=len(SIMULATOR_MOVE_TOKENS)):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(SOLVER_FEATURE_SIZE, 256), torch.nn.GELU(),
            torch.nn.Linear(256, 128), torch.nn.GELU(),
        )
        self.policy = torch.nn.Linear(128, actions)
        self.value = torch.nn.Linear(128, 1)

    def forward(self, features):
        encoded = self.encoder(features.float())
        return self.policy(encoded), self.value(encoded).squeeze(-1)
