import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import PAD_MOVE_ID, state_one_hot


class MLPPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_history = bool(config.get("use_history", False))
        self.max_history = int(config.get("max_history", 8)) if self.use_history else 0
        width = 54 * 6 + (self.max_history * (PAD_MOVE_ID + 1) if self.use_history else 0)
        layers = []
        for hidden in config.get("hidden_dims", [256, 128]):
            layers.extend((nn.Linear(width, int(hidden)), nn.ReLU()))
            width = int(hidden)
        layers.append(nn.Linear(width, 18))
        self.network = nn.Sequential(*layers)

    def forward(self, states, histories):
        features = [state_one_hot(states)]
        if self.use_history:
            features.append(F.one_hot(histories.long(), num_classes=PAD_MOVE_ID + 1).float().flatten(start_dim=1))
        return self.network(torch.cat(features, dim=1))
