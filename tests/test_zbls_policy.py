import json

import torch

from cubemind.environment.cube import (
    SIMULATOR_MOVE_TOKENS, SOLVED_STATE, apply_move, apply_sequence, f2l_slot_count,
    inverse_move, solved_pair_mask,
)
from cubemind.train_zbls_policy import strip_regrips
from cubemind.zbls import (
    ZBLS_FEATURE_SIZE, encode_zbls_features, goal_view, solve_zbls, zbls_complete,
)


def test_zbls_features_are_goal_conditioned_and_model_sized():
    database = json.loads(open("data/algorithms/zbls_cases.json", encoding="utf-8").read())
    case = database["cases"][0]
    state = apply_sequence(SOLVED_STATE, case["setup"])
    view = goal_view(state, 3)
    mask = sum(1 << i for i, solved in enumerate(solved_pair_mask(view)) if solved)
    target = next(i for i in range(4) if not mask & (1 << i))
    features = encode_zbls_features(state, [], mask, target, 3, 0)
    assert features.shape == (ZBLS_FEATURE_SIZE,)
    assert f2l_slot_count(view) == 3


def test_neural_guided_search_replays_a_zbls_teacher_case():
    database = json.loads(open("data/algorithms/zbls_cases.json", encoding="utf-8").read())
    selected = None
    for case in database["cases"]:
        start = apply_sequence(SOLVED_STATE, case["setup"])
        target = strip_regrips(case["algorithms"][0])
        previous, path_states = None, set()
        state, policy = start, {}
        for move in target:
            if move[0] in "xyz" or state in path_states:
                break
            move_id = SIMULATOR_MOVE_TOKENS.index(move)
            if previous is not None and move == inverse_move(SIMULATOR_MOVE_TOKENS[previous]):
                break
            policy[state] = move_id
            path_states.add(state)
            state = apply_move(state, move)
            previous = move_id
        if len(policy) == len(target) and zbls_complete(state, 3):
            selected = (start, target, policy)
            break
    assert selected is not None
    start, _target, policy = selected

    class TeacherPolicy(torch.nn.Module):
        def forward(self, features):
            logits = features.new_full((len(features), len(SIMULATOR_MOVE_TOKENS)), -20.0)
            states = features[:, :324].reshape(-1, 54, 6).argmax(dim=2).cpu().tolist()
            for index, facelets in enumerate(states):
                logits[index, policy[tuple(facelets)]] = 20.0
            values = features.new_zeros(len(features))
            return logits, values, values

    result = solve_zbls(start, TeacherPolicy(), torch.device("cpu"), width=2,
                        branching=1, max_depth=30, max_moves=30, max_nodes=500)
    assert result["verified"]
    assert zbls_complete(apply_sequence(start, result["moves"]), 3)
