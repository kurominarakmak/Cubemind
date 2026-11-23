"""Pair-conditioned policy, value, and pair-selection heads for F2L."""

import torch
import torch.nn.functional as F

from cubemind.environment.cube import (
    SIMULATOR_MOVE_TOKENS, cross_complete, last_layer_edges_oriented, solved_pair_mask,
)

HISTORY_LENGTH = 8
HIERARCHICAL_F2L_FEATURE_SIZE = 54 * 6 + HISTORY_LENGTH * len(SIMULATOR_MOVE_TOKENS) + 4 + 4 + 4 + 4


def encode_f2l_features(state, history, protected_mask, target_pair, move_count, device="cpu"):
    current_mask = sum((1 << index) for index, solved in enumerate(solved_pair_mask(state)) if solved)
    features = torch.zeros(HIERARCHICAL_F2L_FEATURE_SIZE, dtype=torch.float32, device=device)
    features[:324] = F.one_hot(torch.tensor(state, dtype=torch.long, device=device), 6).float().flatten()
    offset = 324
    history = list(history)[-HISTORY_LENGTH:]
    for index, move_id in enumerate(history, HISTORY_LENGTH - len(history)):
        if move_id not in range(len(SIMULATOR_MOVE_TOKENS)):
            raise ValueError("Invalid move-history ID")
        features[offset + index * len(SIMULATOR_MOVE_TOKENS) + move_id] = 1.0
    offset += HISTORY_LENGTH * len(SIMULATOR_MOVE_TOKENS)
    for index in range(4):
        features[offset + index] = float(bool(current_mask & (1 << index)))
        features[offset + 4 + index] = float(bool(protected_mask & (1 << index)))
    offset += 8
    if target_pair is not None:
        if target_pair not in range(4):
            raise ValueError("Target pair must be 0..3")
        features[offset + target_pair] = 1.0
    offset += 4
    features[offset:offset + 4] = torch.tensor([
        float(cross_complete(state)), float(last_layer_edges_oriented(state)),
        float(bin(current_mask).count("1")) / 4.0, min(float(move_count) / 120.0, 1.0),
    ], dtype=torch.float32, device=device)
    return features


class HierarchicalF2LPolicy(torch.nn.Module):
    def __init__(self, actions=len(SIMULATOR_MOVE_TOKENS)):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(HIERARCHICAL_F2L_FEATURE_SIZE, 256), torch.nn.GELU(),
            torch.nn.Linear(256, 128), torch.nn.GELU(),
        )
        self.action = torch.nn.Linear(128, actions)
        self.value = torch.nn.Linear(128, 1)
        self.pair = torch.nn.Linear(128, 4)

    def forward(self, features):
        encoded = self.encoder(features.float())
        return self.action(encoded), self.value(encoded).squeeze(-1), self.pair(encoded)
