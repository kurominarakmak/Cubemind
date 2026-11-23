"""Pair-order classifier trained only on simulator-verified human transitions."""

import torch

from cubemind.models.hierarchical_f2l import HIERARCHICAL_F2L_FEATURE_SIZE


class ExpertPairSelector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(HIERARCHICAL_F2L_FEATURE_SIZE, 256), torch.nn.GELU(),
            torch.nn.Linear(256, 128), torch.nn.GELU(),
        )
        self.pair = torch.nn.Linear(128, 4)

    def forward(self, features):
        return self.pair(self.encoder(features.float()))
