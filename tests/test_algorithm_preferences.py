import torch

from cubemind.algorithm_memory import algorithm_records
from cubemind.environment.cube import (
    SIMULATOR_MOVE_TOKENS, SOLVED_STATE, apply_sequence, inverse_sequence,
)
from cubemind.train_algorithm_preferences import _candidate_replays
from cubemind.models.algorithm_preference import (
    FEATURE_SIZE, AlgorithmPreferenceRanker, algorithm_features, normalize_algorithm,
    rank_algorithm_candidates,
)


def test_algorithm_features_ignore_edge_auf_and_rotation_and_rank_candidates():
    assert normalize_algorithm("U' x R U R' y U2") == ("R", "U", "R'")
    features = torch.tensor([
        algorithm_features("R U R'"),
        algorithm_features("F R U R' U' F'"),
    ], dtype=torch.float32)
    assert features.shape == (2, FEATURE_SIZE)
    model = AlgorithmPreferenceRanker(3)
    scores = model(torch.tensor([1, 1]), features)
    assert scores.shape == (2,)
    assert len(SIMULATOR_MOVE_TOKENS) > 0


def test_preference_ranking_keeps_case_compatible_retrieval_rows():
    model = AlgorithmPreferenceRanker(1)
    model.case_ids = ("oll-1",)
    rows = [
        {"case_id": "oll-1", "algorithm_id": "oll-1:1", "moves": ["R", "U", "R'"],
         "move_metrics": {"stm": 3}},
        {"case_id": "oll-1", "algorithm_id": "oll-1:2", "moves": ["F", "R", "U", "R'", "U'", "F'"],
         "move_metrics": {"stm": 6}},
    ]
    ranked = rank_algorithm_candidates(rows, model)
    assert len(ranked) == 2
    assert all("preference_score" in row for row in ranked)
    assert all(row["case_id"] == "oll-1" for row in ranked)


def test_candidate_label_requires_its_own_algorithm_to_replay():
    records = algorithm_records()
    for family in ("OLL", "PLL"):
        record = next(row for row in records if row["family"] == family)
        before = apply_sequence(SOLVED_STATE, inverse_sequence(record["moves"]))
        assert _candidate_replays(before, record, family)
        assert not _candidate_replays(SOLVED_STATE, record, family)
