"""Compact from-scratch action policy for human F2L demonstrations."""

import torch


class F2LPolicy(torch.nn.Module):
    def __init__(self, input_size, actions):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, 256), torch.nn.GELU(),
            torch.nn.Linear(256, 128), torch.nn.GELU(),
            torch.nn.Linear(128, actions),
        )

    def forward(self, features):
        return self.network(features.float())
