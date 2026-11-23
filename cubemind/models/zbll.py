"""Small from-scratch ZBLL case classifier."""

import torch


class ZBLLClassifier(torch.nn.Module):
    def __init__(self, classes=493):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(54 * 6, 512), torch.nn.GELU(),
            torch.nn.Linear(512, 256), torch.nn.GELU(),
            torch.nn.Linear(256, classes),
        )

    def forward(self, states):
        features = torch.nn.functional.one_hot(states.long(), num_classes=6).float().flatten(1)
        return self.network(features)
