"""Goal-conditioned neural search for the four CFOP F2L pairs."""

import random
import time

import torch

from cubemind.environment.cube import (
    F2L_PAIR_NAMES, MOVE_TOKENS, SIMULATOR_MOVE_TOKENS, apply_move, cross_complete, cross_distance,
    f2l_complete, f2l_pair_distance, inverse_move, pair_solved, solved_pair_mask,
)
from cubemind.models.hierarchical_f2l import (
    HIERARCHICAL_F2L_FEATURE_SIZE, HierarchicalF2LPolicy, encode_f2l_features,
)

MOVE_TO_ID = {move: index for index, move in enumerate(SIMULATOR_MOVE_TOKENS)}
OUTER_MOVE_IDS = {MOVE_TO_ID[move] for move in MOVE_TOKENS}
DEFAULT_CHECKPOINT = "artifacts/hierarchical-f2l.pt"


def _popcount(mask):
    return bin(int(mask)).count("1")


def load_hierarchical_f2l(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if (checkpoint.get("feature_size") != HIERARCHICAL_F2L_FEATURE_SIZE
            or checkpoint.get("actions") != len(SIMULATOR_MOVE_TOKENS)
            or not checkpoint.get("data_sources")):
        raise ValueError("Unsupported hierarchical F2L checkpoint")
    model = HierarchicalF2LPolicy().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _mask_bits(state):
    return sum((1 << index) for index, solved in enumerate(solved_pair_mask(state)) if solved)


def _model_output(model, state, history, protected_mask, target, move_count, device):
    features = encode_f2l_features(state, history, protected_mask, target, move_count, device).unsqueeze(0)
    with torch.inference_mode():
        actions, value, pairs = model(features)
    if (actions.ndim != 2 or actions.shape[1] != len(SIMULATOR_MOVE_TOKENS)
            or value.shape != (1,) or pairs.shape != (1, 4)
            or not torch.isfinite(actions).all() or not torch.isfinite(value).all()
            or not torch.isfinite(pairs).all()):
        raise ValueError("Hierarchical F2L model returned invalid output")
    return actions[0], float(value[0].clamp_min(0)), pairs[0]


def _batch_values(model, candidates, protected_mask, target, base_move_count, device):
    features = torch.stack([
        encode_f2l_features(state, history, protected_mask, target, base_move_count + len(path), "cpu")
        for state, history, path, _visited, _cost in candidates
    ]).to(device)
    with torch.inference_mode():
        _actions, values, _pairs = model(features)
    if not torch.isfinite(values).all():
        raise ValueError("Hierarchical F2L value head returned invalid estimates")
    return values.clamp_min(0).detach().cpu().tolist()


def _base(move):
    return move.rstrip("'2")


def _legal_moves(history):
    previous = SIMULATOR_MOVE_TOKENS[history[-1]] if history else None
    previous_base = _base(previous) if previous else None
    result = []
    for move_id, move in enumerate(SIMULATOR_MOVE_TOKENS):
        if move_id not in OUTER_MOVE_IDS:
            continue
        if previous and move == inverse_move(previous):
            continue
        if previous_base and _base(move) == previous_base:
            continue
        result.append(move_id)
    return result


def _search_pair(start, history, target, protected_mask, model, device,
                 width, branching, max_depth, node_limit, guidance, seed, base_move_count,
                 value_weight, conditional_legal=False):
    start = tuple(start)
    start_history = list(history)
    beam = [(start, start_history, (), frozenset((start,)), 0.0)]
    start_mask = _mask_bits(start)
    explored, best_partial = 0, (start, start_history, (), _popcount(start_mask), float("inf"))
    found = None
    for depth in range(1, max_depth + 1):
        parents = beam[:max(0, min(len(beam), node_limit - explored))]
        if not parents:
            break
        parent_features = torch.stack([
            encode_f2l_features(state, hist, protected_mask, target, base_move_count + len(path), "cpu")
            for state, hist, path, _visited, _cost in parents
        ]).to(device)
        with torch.inference_mode():
            parent_logits, _parent_values, _parent_pairs = model(parent_features)
        if not torch.isfinite(parent_logits).all():
            raise ValueError("Hierarchical F2L policy returned invalid action logits")
        explored += len(parents)
        candidates = {}
        goals = []
        for parent_index, (state, hist, path, visited, cost) in enumerate(parents):
            logits = parent_logits[parent_index]
            legal = _legal_moves(hist)
            if conditional_legal:
                legal_ids = torch.tensor(legal, dtype=torch.long, device=logits.device)
                legal_log_probs = torch.log_softmax(logits[legal_ids], dim=0)
                action_log_probs = {move_id: float(legal_log_probs[index])
                                    for index, move_id in enumerate(legal)}
            else:
                log_probs = torch.log_softmax(logits, dim=0)
                action_log_probs = {move_id: float(log_probs[move_id]) for move_id in legal}
            if guidance == "neural":
                ordered = [move_id for move_id in logits.argsort(descending=True).tolist() if move_id in legal]
                move_ids = ordered[:branching]
            else:
                rng = random.Random(seed + depth * 1000003 + sum(state) * 9176 + parent_index)
                rng.shuffle(legal)
                move_ids = legal[:branching]
            for move_id in move_ids:
                move = SIMULATOR_MOVE_TOKENS[move_id]
                child = apply_move(state, move)
                child_history = (hist + [move_id])[-8:]
                child_path = path + (move,)
                child_visited = visited | {child}
                child_mask = _mask_bits(child)
                if (cross_complete(child) and (child_mask & protected_mask) == protected_mask
                        and child_mask & (1 << target)):
                    policy_cost = cost + (-action_log_probs[move_id] if guidance == "neural" else 0.0)
                    goals.append((policy_cost, child, child_history, child_path))
                    continue
                if child in visited:
                    continue
                policy_cost = cost + (-action_log_probs[move_id] if guidance == "neural" else 0.0)
                key = (child, tuple(child_history[-8:]))
                candidate = (child, child_history, child_path, child_visited, policy_cost)
                previous = candidates.get(key)
                if previous is None or policy_cost < previous[4]:
                    candidates[key] = candidate
            if explored >= node_limit:
                break
        if goals:
            policy_cost, final, final_history, final_path = min(goals, key=lambda item: (len(item[3]), item[0]))
            found = (final, final_history, list(final_path), explored, None)
            break
        if not candidates or explored >= node_limit:
            break
        candidate_rows = list(candidates.values())
        values = _batch_values(model, candidate_rows, protected_mask, target, base_move_count, device)
        ranked = []
        for node, value in zip(candidate_rows, values):
            state, node_history, path, _visited, policy_cost = node
            protected_distance = max(
                (f2l_pair_distance(state, pair) for pair in range(4) if protected_mask & (1 << pair)),
                default=0,
            )
            priority = (depth + value_weight * float(value) + 1.25 * f2l_pair_distance(state, target)
                        + cross_distance(state) + protected_distance + 0.02 * policy_cost)
            ranked.append((priority, node))
            mask_count = _popcount(_mask_bits(state))
            if cross_complete(state) and (mask_count > best_partial[3]
                    or (mask_count == best_partial[3] and priority < best_partial[4])):
                best_partial = (state, node_history, path, mask_count, priority)
        ranked.sort(key=lambda item: item[0])
        beam = [node for _priority, node in ranked[:width]]
    if found:
        return found
    final, final_history, final_path, _mask_count, _priority = best_partial
    return final, final_history, list(final_path), explored, "pair_search_budget:{}".format(F2L_PAIR_NAMES[target])


def _greedy_pair(start, history, target, protected_mask, model, device, max_depth, base_move_count):
    current, current_history = tuple(start), list(history)
    path, visited = [], {tuple(start)}
    for _ in range(max_depth):
        logits, _value, _pairs = _model_output(
            model, current, current_history, protected_mask, target, base_move_count + len(path), device,
        )
        legal = _legal_moves(current_history)
        if not legal:
            return current, current_history, path, "repeated_state:{}".format(F2L_PAIR_NAMES[target])
        move_id = next((index for index in logits.argsort(descending=True).tolist() if index in legal), None)
        if move_id is None:
            return current, current_history, path, "no_legal_action:{}".format(F2L_PAIR_NAMES[target])
        move = SIMULATOR_MOVE_TOKENS[move_id]
        current = apply_move(current, move)
        current_history = (current_history + [move_id])[-8:]
        path.append(move)
        mask = _mask_bits(current)
        if cross_complete(current) and (mask & protected_mask) == protected_mask and mask & (1 << target):
            return current, current_history, path, None
        if current in visited:
            return current, current_history, path, "repeated_state:{}".format(F2L_PAIR_NAMES[target])
        visited.add(current)
    return current, current_history, path, "pair_move_budget:{}".format(F2L_PAIR_NAMES[target])


def solve_hierarchical_f2l(state, model, device, history=(), mode="search", width=24,
                           branching=8, max_nodes=6000, max_depth=18, max_moves=100,
                           max_actions_per_pair=30, max_order_attempts=16,
                           guidance="neural", seed=551, move_count=None, value_weight=0.5,
                           conditional_legal=False, execution_cost_weight=0.0,
                           stop_after_pairs=None):
    if mode not in ("greedy", "search"):
        raise ValueError("mode must be greedy or search")
    if guidance not in ("neural", "uniform"):
        raise ValueError("guidance must be neural or uniform")
    if value_weight < 0:
        raise ValueError("value_weight must be nonnegative")
    if execution_cost_weight < 0:
        raise ValueError("execution_cost_weight must be nonnegative")
    if stop_after_pairs is not None and stop_after_pairs not in range(1, 5):
        raise ValueError("stop_after_pairs must be 1..4")
    if min(width, branching, max_nodes, max_depth, max_moves, max_actions_per_pair, max_order_attempts) < 1:
        raise ValueError("F2L search budgets must be positive")
    start = tuple(state)
    if len(start) != 54:
        raise ValueError("Cube state must contain 54 facelets")
    initial_move_count = len(history) if move_count is None else int(move_count)
    if initial_move_count < 0:
        raise ValueError("Move count must be nonnegative")
    initial_history = list(history)[-8:]
    started = time.perf_counter()
    if not cross_complete(start):
        return {
            "status": "FAILED / INCOMPLETE", "moves": [], "last_state": start,
            "verified": False, "cross_complete": False, "f2l_complete": False,
            "solved_pair_mask": solved_pair_mask(start), "pair_results": [],
            "pair_attempts": [], "pair_destruction_events": 0, "max_solved_pairs": _popcount(_mask_bits(start)),
            "final_solved_pairs": _popcount(_mask_bits(start)), "states_explored": 0,
            "runtime_seconds": time.perf_counter() - started, "failure_reason": "cross_not_solved",
            "route": "hierarchical neural F2L; no deterministic fallback",
        }

    initial_mask = _mask_bits(start)
    best = {"moves": [], "mask_count": _popcount(initial_mask), "order": []}
    attempts = []
    nodes_used, orders_used = 0, 0
    max_seen = best["mask_count"]

    def update_best(current, moves, order):
        nonlocal best, max_seen
        mask = _mask_bits(current)
        required = initial_mask | sum(1 << index for index in order)
        if not cross_complete(current) or mask & required != required:
            return
        count = _popcount(mask)
        max_seen = max(max_seen, count)
        if count > best["mask_count"] or (count == best["mask_count"] and len(moves) < len(best["moves"])):
            best = {"moves": list(moves), "mask_count": count, "order": list(order)}

    def visit(current, current_history, moves, order):
        nonlocal nodes_used, orders_used
        update_best(current, moves, order)
        current_mask = _mask_bits(current)
        if current_mask == 15 or (stop_after_pairs is not None
                                  and _popcount(current_mask) >= stop_after_pairs):
            return current, current_history, moves, order
        if len(moves) >= max_moves or orders_used >= max_order_attempts or nodes_used >= max_nodes:
            return None
        _logits, _value, pair_logits = _model_output(
            model, current, current_history, current_mask, None, initial_move_count + len(moves), device,
        )
        ranked_pairs = [index for index in pair_logits.argsort(descending=True).tolist()
                        if not current_mask & (1 << index)]
        complete_candidates = []
        for target in ranked_pairs:
            if orders_used >= max_order_attempts or nodes_used >= max_nodes:
                break
            orders_used += 1
            max_pair_nodes = max(1, min(width * max_depth, max_nodes - nodes_used))
            if mode == "greedy":
                next_state, next_history, segment, failure = _greedy_pair(
                    current, current_history, target, current_mask, model, device,
                    min(max_actions_per_pair, max_moves - len(moves)),
                    initial_move_count + len(moves),
                )
                expanded = len(segment)
            else:
                next_state, next_history, segment, expanded, failure = _search_pair(
                    current, current_history, target, current_mask, model, device,
                    width, branching, min(max_depth, max_actions_per_pair, max_moves - len(moves)),
                    max_pair_nodes, guidance, seed + orders_used * 17,
                    initial_move_count + len(moves),
                    value_weight, conditional_legal,
                )
            nodes_used += expanded
            result = {
                "target_pair": target, "target_slot": F2L_PAIR_NAMES[target],
                "moves": segment, "verified": failure is None and pair_solved(next_state, target),
                "nodes": expanded, "failure_reason": failure,
                "protected_mask": tuple(bool(current_mask & (1 << index)) for index in range(4)),
            }
            attempts.append(result)
            if result["verified"]:
                combined = moves + segment
                combined_order = order + [target]
                update_best(next_state, combined, combined_order)
                complete = visit(next_state, next_history, combined, combined_order)
                if complete is not None:
                    if execution_cost_weight == 0:
                        return complete
                    complete_candidates.append(complete)
            else:
                next_mask = _mask_bits(next_state)
                new_pairs = [index for index in range(4)
                             if not current_mask & (1 << index) and next_mask & (1 << index)]
                if (cross_complete(next_state) and next_mask & current_mask == current_mask
                        and new_pairs and len(moves) + len(segment) <= max_moves):
                    repaired_order = order + new_pairs
                    update_best(next_state, moves + segment, repaired_order)
                    complete = visit(next_state, next_history, moves + segment, repaired_order)
                    if complete is not None:
                        if execution_cost_weight == 0:
                            return complete
                        complete_candidates.append(complete)
                else:
                    update_best(next_state, moves + segment, order)
        if complete_candidates:
            return min(complete_candidates, key=lambda candidate: len(candidate[2]) * execution_cost_weight)
        return None

    complete = visit(start, initial_history, [], [])
    final_moves = complete[2] if complete else best["moves"]
    final_state = start
    current_mask = _mask_bits(start)
    destruction_events = 0
    max_seen = _popcount(current_mask)
    for move in final_moves:
        final_state = apply_move(final_state, move)
        next_mask = _mask_bits(final_state)
        destruction_events += _popcount(current_mask & ~next_mask)
        max_seen = max(max_seen, _popcount(next_mask))
        current_mask = next_mask
    verified = cross_complete(final_state) and f2l_complete(final_state)
    prefix_verified = (stop_after_pairs is not None and cross_complete(final_state)
                       and _popcount(_mask_bits(final_state)) >= stop_after_pairs)
    pair_results = []
    protected = initial_mask
    pending_moves = []
    replay = start
    for move in final_moves:
        pending_moves.append(move)
        replay = apply_move(replay, move)
        replay_mask = _mask_bits(replay)
        added = replay_mask & ~protected
        if cross_complete(replay) and replay_mask & protected == protected and added and _popcount(added) == 1:
            pair_id = (added & -added).bit_length() - 1
            pair_results.append({"target_pair": pair_id, "target_slot": F2L_PAIR_NAMES[pair_id],
                                 "moves": list(pending_moves), "verified": True})
            protected = replay_mask
            pending_moves = []
    failure_reason = None if verified or prefix_verified else (
        attempts[-1]["failure_reason"] if attempts else "no_unsolved_pair")
    return {
        "status": ("F2L COMPLETE" if verified else "F2L PREFIX COMPLETE" if prefix_verified
                   else "FAILED / INCOMPLETE"),
        "moves": final_moves, "last_state": final_state, "verified": verified,
        "prefix_verified": prefix_verified,
        "cross_complete": cross_complete(final_state), "f2l_complete": f2l_complete(final_state),
        "solved_pair_mask": solved_pair_mask(final_state),
        "pair_results": pair_results, "pair_attempts": attempts,
        "pair_order": [F2L_PAIR_NAMES[index] for index in (complete[3] if complete else best["order"])],
        "pair_destruction_events": destruction_events,
        "max_solved_pairs": max_seen, "final_solved_pairs": _popcount(_mask_bits(final_state)),
        "states_explored": nodes_used, "pair_order_attempts": orders_used,
        "runtime_seconds": time.perf_counter() - started,
        "failure_reason": failure_reason,
        "guidance": guidance, "mode": mode,
        "execution_cost_weight": execution_cost_weight,
        "route": "hierarchical neural F2L; no deterministic fallback",
    }
