"""Small from-scratch ranker for replay-matched human OLL/PLL choices."""

import torch
from torch import nn
from pathlib import Path

from cubemind.environment.cube import SIMULATOR_MOVE_TOKENS, parse_moves


MOVE_TO_ID = {move: index for index, move in enumerate(SIMULATOR_MOVE_TOKENS)}
FEATURE_SIZE = 6 + 3 * len(SIMULATOR_MOVE_TOKENS)


def normalize_algorithm(sequence):
    moves = list(parse_moves(sequence))
    removable = {"U", "U2", "U'", "x", "x2", "x'", "y", "y2", "y'", "z", "z2", "z'"}
    while moves and moves[0] in removable:
        moves.pop(0)
    while moves and moves[-1] in removable:
        moves.pop()
    return tuple(moves)


def algorithm_features(sequence):
    moves = normalize_algorithm(sequence)
    metrics = [
        sum(2 if move[0].islower() or move[1:2] == "w" else 1 for move in moves if move[0] not in "xyz") / 40,
        sum(move[0] in "URFDLB" and move[1:2] != "w" for move in moves) / 40,
        sum(move[0] in "xyz" for move in moves) / 10,
        sum(move[0] in "MES" for move in moves) / 10,
        sum(move[0].islower() or move[1:2] == "w" for move in moves) / 10,
        len(moves) / 40,
    ]
    histogram = [0.0] * len(SIMULATOR_MOVE_TOKENS)
    for move in moves:
        if move not in MOVE_TO_ID:
            raise ValueError("Algorithm contains unsupported move: {}".format(move))
        histogram[MOVE_TO_ID[move]] += 1 / max(1, len(moves))
    first = [0.0] * len(MOVE_TO_ID)
    last = [0.0] * len(MOVE_TO_ID)
    if moves:
        first[MOVE_TO_ID[moves[0]]] = 1.0
        last[MOVE_TO_ID[moves[-1]]] = 1.0
    return metrics + histogram + first + last


class AlgorithmPreferenceRanker(nn.Module):
    def __init__(self, case_count, feature_size=FEATURE_SIZE):
        super().__init__()
        self.case_embedding = nn.Embedding(case_count, 8)
        self.score = nn.Sequential(nn.Linear(8 + feature_size, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, case_indices, candidate_features):
        if candidate_features.ndim != 2 or candidate_features.shape[1] != FEATURE_SIZE:
            raise ValueError("Unexpected algorithm feature width")
        return self.score(torch.cat((self.case_embedding(case_indices), candidate_features), dim=1)).squeeze(1)


def load_algorithm_preference_model(path, device="cpu"):
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if checkpoint.get("task") != "algorithm_preference_ranker" or checkpoint.get("feature_size") != FEATURE_SIZE:
        raise ValueError("Unsupported algorithm-preference checkpoint")
    case_ids = checkpoint.get("case_ids")
    if not isinstance(case_ids, list) or not case_ids:
        raise ValueError("Preference checkpoint has no case vocabulary")
    model = AlgorithmPreferenceRanker(len(case_ids), FEATURE_SIZE).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.case_ids = tuple(case_ids)
    model.eval()
    return model


def rank_algorithm_candidates(candidates, model):
    """Use human-choice scores when the case has held-out labels; otherwise retain input order."""
    if not candidates or model is None:
        return candidates
    case_ids = getattr(model, "case_ids", ())
    case_id = candidates[0].get("case_id")
    if case_id not in case_ids or any(row.get("case_id") != case_id for row in candidates):
        return candidates
    from cubemind.algorithm_memory import algorithm_records
    by_id = {(row["case_id"], row["algorithm_id"]): row for row in algorithm_records()}
    sequences = [by_id.get((case_id, row["algorithm_id"]), {}).get("moves", row["moves"])
                 for row in candidates]
    features = torch.tensor([algorithm_features(sequence) for sequence in sequences], dtype=torch.float32)
    cases = torch.full((len(candidates),), case_ids.index(case_id), dtype=torch.long)
    with torch.inference_mode():
        scores = model(cases, features).cpu().tolist()
    ranked = [dict(row, preference_score=scores[index]) for index, row in enumerate(candidates)]
    return sorted(ranked, key=lambda row: (-row["preference_score"],
                                            row.get("move_metrics", {}).get("stm", 0),
                                            row.get("algorithm_id", "")))
