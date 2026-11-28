import torch

from cubemind.environment.cube import (
    COLOR_NAMES, MOVE_TOKENS, SIMULATOR_MOVE_TOKENS, SOLVED_STATE,
    apply_sequence, cross_complete,
)
from cubemind.models.expert_cross import (
    ExpertCrossPolicy, ExpertCrossPolicyAdapter, FEATURE_SIZE,
)
from cubemind.train_expert_cross import _cross_rows, _features


def test_human_cross_rows_are_normalized_and_outer_turn_labels_are_kept():
    state = apply_sequence(SOLVED_STATE, "R")
    move = "R'"
    row = {"phase": "CROSS", "sample_kind": "expert", "data_source": "reco.nz",
           "cross_color": "white", "cube_state": list(state),
           "next_expert_move": move, "move_id": SIMULATOR_MOVE_TOKENS.index(move)}
    rows, excluded = _cross_rows([row, {**row, "phase": "CROSS+1"}])
    assert len(rows) == 1 and excluded == {}
    assert not cross_complete(rows[0]["state"])
    assert _features(rows).shape == (1, FEATURE_SIZE)


def test_human_cross_policy_adapter_uses_existing_cross_search_signature():
    model = ExpertCrossPolicy()
    adapter = ExpertCrossPolicyAdapter(model)
    states = torch.tensor([SOLVED_STATE, apply_sequence(SOLVED_STATE, "R")])
    logits = adapter(states, torch.empty((2, 0), dtype=torch.long))
    assert logits.shape == (2, len(MOVE_TOKENS))
    assert len(COLOR_NAMES) == 6
