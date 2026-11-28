"""Goal-conditioned policy/value model for the last F2L slot and edge orientation."""

import torch

from cubemind.environment.cube import SIMULATOR_MOVE_TOKENS

ZBLS_FEATURE_SIZE = 54 * 6 + 8 * len(SIMULATOR_MOVE_TOKENS) + 4 + 4 + 4 + 6 + 4 + 1


class ZBLSPolicyValue(torch.nn.Module):
    def __init__(self, actions=len(SIMULATOR_MOVE_TOKENS)):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(ZBLS_FEATURE_SIZE, 256), torch.nn.GELU(),
            torch.nn.Linear(256, 128), torch.nn.GELU(),
        )
        self.action = torch.nn.Linear(128, actions)
        self.value = torch.nn.Linear(128, 1)
        self.edge_orientation = torch.nn.Linear(128, 1)

    def forward(self, features):
        encoded = self.encoder(features.float())
        return (self.action(encoded), self.value(encoded).squeeze(-1),
                self.edge_orientation(encoded).squeeze(-1))
