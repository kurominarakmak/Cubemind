"""Neural-guided, simulator-verified ZBLS search with exact goal checks."""

import math
import time

import torch
import torch.nn.functional as F

from cubemind.environment.cube import (
    COLORS, COLOR_NAMES, SIMULATOR_MOVE_TOKENS, apply_move, apply_sequence, cross_complete, f2l_complete,
    cross_frame_rotation, inverse_move, last_layer_edges_oriented, normalize_cross_frame,
    parse_moves, solved_pair_mask,
)
from cubemind.models.zbls_policy import ZBLS_FEATURE_SIZE, ZBLSPolicyValue

HISTORY_LENGTH = 8
MOVE_IDS = {move: index for index, move in enumerate(SIMULATOR_MOVE_TOKENS)}
ROTATIONS = {index for index, move in enumerate(SIMULATOR_MOVE_TOKENS) if move[0] in "xyz"}


def _center_color_map(state):
    centers = tuple(state[face * 9 + 4] for face in range(6))
    return {color: face for face, color in enumerate(centers)}


def recolor_to_center_frame(state):
    """Rename stickers so the current centers use the fixed URFDLB color IDs."""
    mapping = _center_color_map(state)
    return tuple(mapping[color] for color in state)


def goal_view(state, cross_color):
    """Return the selected Cross frame with colors canonicalized for exact subgoal checks."""
    return recolor_to_center_frame(normalize_cross_frame(state, cross_color))


def zbls_complete(state, cross_color):
    view = goal_view(state, cross_color)
    return cross_complete(view) and f2l_complete(view) and last_layer_edges_oriented(view)


def encode_zbls_features(state, history, protected_mask, target_pair, cross_color,
                         move_count, device="cpu"):
    if type(cross_color) is not int or cross_color not in range(6):
        raise ValueError("Cross color must be an ID in 0..5")
    if protected_mask not in range(16) or target_pair not in range(4):
        raise ValueError("Protected mask and target pair are out of range")
    if type(move_count) is not int or move_count < 0:
        raise ValueError("Move count must be a non-negative integer")
    state = tuple(state)
    view = goal_view(state, cross_color)
    mask = sum((1 << i) for i, solved in enumerate(solved_pair_mask(view)) if solved)
    features = torch.zeros(ZBLS_FEATURE_SIZE, dtype=torch.float32, device=device)
    features[:324] = F.one_hot(torch.tensor(state, dtype=torch.long, device=device), 6).float().flatten()
    offset = 324
    history = list(history)[-HISTORY_LENGTH:]
    for index, move_id in enumerate(history, HISTORY_LENGTH - len(history)):
        if move_id not in range(len(SIMULATOR_MOVE_TOKENS)):
            raise ValueError("Invalid move-history ID")
        features[offset + index * len(SIMULATOR_MOVE_TOKENS) + move_id] = 1.0
    offset += HISTORY_LENGTH * len(SIMULATOR_MOVE_TOKENS)
    for index in range(4):
        features[offset + index] = float(bool(mask & (1 << index)))
        features[offset + 4 + index] = float(bool(protected_mask & (1 << index)))
    offset += 8
    features[offset + target_pair] = 1.0
    offset += 4
    features[offset + cross_color] = 1.0
    offset += 6
    features[offset:offset + 4] = torch.tensor([
        float(cross_complete(view)), float(last_layer_edges_oriented(view)),
        sum(solved_pair_mask(view)) / 4.0, min(move_count / 120.0, 1.0),
    ], dtype=torch.float32, device=device)
    features[-1] = 1.0  # Goal-conditioned ZBLS, rather than ordinary F2L completion.
    return features


def load_zbls_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if (checkpoint.get("task") != "neural_zbls_policy"
            or checkpoint.get("feature_size") != ZBLS_FEATURE_SIZE
            or checkpoint.get("actions") != len(SIMULATOR_MOVE_TOKENS)):
        raise ValueError("Unsupported neural ZBLS checkpoint")
    model = ZBLSPolicyValue().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _priority(state, cross_color, depth, value, eo_probability, log_score, value_weight):
    view = goal_view(state, cross_color)
    pairs = sum(solved_pair_mask(view))
    return (depth + value_weight * max(0.0, value) + 4.0 * (4 - pairs)
            + (0.0 if cross_complete(view) else 8.0)
            + (0.0 if last_layer_edges_oriented(view) else 2.0 * (1.0 - eo_probability))
            - 0.02 * log_score)


def solve_zbls(state, model, device, cross_color=COLORS["D"], history=(),
               width=48, branching=8, max_depth=30, max_nodes=5000,
               max_moves=35, value_weight=0.5):
    """Search for Cross + all F2L pairs + oriented LL edges; no algorithm lookup is used."""
    if isinstance(cross_color, str) and cross_color in COLOR_NAMES:
        cross_color = COLOR_NAMES.index(cross_color)
    if type(cross_color) is not int or cross_color not in range(6):
        raise ValueError("Cross color must be an ID in 0..5 or a known color name")
    if (min(width, branching, max_depth, max_nodes, max_moves) < 1
            or not math.isfinite(value_weight) or value_weight < 0):
        raise ValueError("ZBLS search limits must be positive and value weight non-negative")
    original = tuple(state)
    started = time.perf_counter()
    try:
        frame_moves = cross_frame_rotation(original, cross_color)
    except (ValueError, KeyError) as error:
        return {"verified": False, "failure_reason": "invalid_cross_frame: {}".format(error),
                "moves": [], "last_state": original, "states_explored": 0,
                "runtime_seconds": time.perf_counter() - started}
    frame_state = tuple(state)
    for move in frame_moves:
        frame_state = apply_move(frame_state, move)
    view = goal_view(frame_state, cross_color)
    initial_mask = sum((1 << i) for i, solved in enumerate(solved_pair_mask(view)) if solved)
    if not cross_complete(view) or bin(initial_mask).count("1") != 3:
        return {"verified": False, "failure_reason": "requires_cross_and_three_f2l_pairs",
                "moves": [], "last_state": original, "states_explored": 0,
                "runtime_seconds": time.perf_counter() - started}
    target_pair = next(index for index in range(4) if not initial_mask & (1 << index))
    if zbls_complete(frame_state, cross_color):
        end_frame = cross_frame_rotation(frame_state, cross_color)
        solution = list(frame_moves) + list(end_frame)
        final = apply_sequence(original, solution)
        verified = (cross_complete(final) and f2l_complete(final)
                    and last_layer_edges_oriented(final))
        return {"verified": verified, "failure_reason": None if verified else "final_replay_failed:ZBLS",
                "moves": solution, "last_state": final, "states_explored": 0,
                "runtime_seconds": time.perf_counter() - started,
                "target_pair": target_pair, "edge_oriented": verified}

    initial_history = list(history)[-HISTORY_LENGTH:]
    if any(move_id not in range(len(SIMULATOR_MOVE_TOKENS)) for move_id in initial_history):
        raise ValueError("Invalid initial move-history ID")
    initial_history.extend(MOVE_IDS[move] for move in frame_moves)
    initial_history = initial_history[-HISTORY_LENGTH:]
    # Nodes retain committed state/path/history only; failed branches never enter the result.
    beam = [(frame_state, (), tuple(initial_history), 0.0, frozenset((frame_state,)))]
    best_scores = {(frame_state, tuple(initial_history)): 0.0}
    best = beam[0]
    best_rank = (sum(solved_pair_mask(view)), last_layer_edges_oriented(view), cross_complete(view))
    explored = 0
    failure = "search_frontier_empty:ZBLS"
    allowed = [i for i in range(len(SIMULATOR_MOVE_TOKENS)) if i not in ROTATIONS]
    model.eval()
    for depth in range(1, min(max_depth, max_moves) + 1):
        parents = beam[:max(0, min(len(beam), max_nodes - explored))]
        if not parents:
            break
        features = torch.stack([
            encode_zbls_features(node[0], node[2], initial_mask, target_pair,
                                 cross_color, len(node[1]) + len(frame_moves), "cpu")
            for node in parents
        ]).to(device)
        with torch.inference_mode():
            logits, _values, _eo = model(features)
        if (logits.shape != (len(parents), len(SIMULATOR_MOVE_TOKENS))
                or not torch.isfinite(logits).all()):
            raise ValueError("Neural ZBLS model returned invalid policy/value output")
        log_probs = torch.log_softmax(logits, dim=1)
        explored += len(parents)
        candidates = {}
        for parent_index, (parent_state, path, parent_history, log_score, visited) in enumerate(parents):
            legal = []
            previous = SIMULATOR_MOVE_TOKENS[parent_history[-1]] if parent_history else None
            for move_id in allowed:
                move = SIMULATOR_MOVE_TOKENS[move_id]
                if previous and move == inverse_move(previous):
                    continue
                legal.append(move_id)
            if not legal:
                continue
            legal_tensor = torch.tensor(legal, dtype=torch.long, device=device)
            ordered = legal_tensor[log_probs[parent_index, legal_tensor].argsort(descending=True)].tolist()
            selected, selected_states = [], set()
            for move_id in ordered:
                candidate_move = SIMULATOR_MOVE_TOKENS[move_id]
                candidate_state = apply_move(parent_state, candidate_move)
                if candidate_state in selected_states:
                    continue
                selected_states.add(candidate_state)
                selected.append((move_id, candidate_state))
                if len(selected) == branching:
                    break
            for move_id, child in selected:
                move = SIMULATOR_MOVE_TOKENS[move_id]
                child_path = path + (move,)
                child_history = (parent_history + (move_id,))[-HISTORY_LENGTH:]
                child_log_score = log_score + float(log_probs[parent_index, move_id])
                if zbls_complete(child, cross_color):
                    end_frame = cross_frame_rotation(child, cross_color)
                    solution = list(frame_moves) + list(child_path) + list(end_frame)
                    final = apply_sequence(original, solution)
                    verified = (cross_complete(final) and f2l_complete(final)
                                and last_layer_edges_oriented(final)
                                and zbls_complete(final, cross_color))
                    if not verified:
                        raise AssertionError("ZBLS goal changed between search and replay")
                    return {"verified": True, "failure_reason": None, "moves": solution,
                            "last_state": final, "states_explored": explored,
                            "runtime_seconds": time.perf_counter() - started,
                            "target_pair": target_pair, "edge_oriented": True,
                            "cross_color": cross_color}
                if len(child_path) >= max_moves or child in visited:
                    continue
                key = (child, child_history)
                if child_log_score <= best_scores.get(key, float("-inf")):
                    continue
                node = (child, child_path, child_history, child_log_score, visited | {child})
                candidates[key] = node
                best_scores[key] = child_log_score
                child_view = goal_view(child, cross_color)
                rank = (sum(solved_pair_mask(child_view)),
                        last_layer_edges_oriented(child_view), cross_complete(child_view))
                if rank > best_rank or (rank == best_rank and len(child_path) < len(best[1])):
                    best, best_rank = node, rank
        if explored >= max_nodes or not candidates:
            break
        remaining = max_nodes - explored
        candidate_rows = sorted(candidates.values(), key=lambda node: node[3], reverse=True)[:remaining]
        if not candidate_rows:
            failure = "search_budget:ZBLS"
            break
        child_features = torch.stack([
            encode_zbls_features(node[0], node[2], initial_mask, target_pair,
                                 cross_color, len(node[1]) + len(frame_moves), "cpu")
            for node in candidate_rows
        ]).to(device)
        with torch.inference_mode():
            _child_logits, child_values, child_eo = model(child_features)
        explored += len(candidate_rows)
        if not torch.isfinite(child_values).all() or not torch.isfinite(child_eo).all():
            raise ValueError("Neural ZBLS model returned invalid value/EO estimates")
        ranked = []
        for node, value, eo_logit in zip(candidate_rows, child_values, child_eo):
            priority = _priority(node[0], cross_color, depth,
                                 float(value), float(torch.sigmoid(eo_logit)),
                                 node[3], value_weight)
            ranked.append((priority, node))
        beam = [node for _score, node in sorted(ranked, key=lambda item: item[0])[:width]]
    if explored >= max_nodes:
        failure = "search_budget:ZBLS"
    elif depth >= min(max_depth, max_moves):
        failure = "depth_limit:ZBLS"
    _partial_state, partial_path, _partial_history, _log_score, _visited = best
    partial_moves = list(frame_moves) + list(partial_path)
    partial_state = apply_sequence(original, partial_moves)
    return {"verified": False, "failure_reason": failure,
            "moves": list(frame_moves) + list(partial_path), "last_state": partial_state,
            "states_explored": explored, "runtime_seconds": time.perf_counter() - started,
            "target_pair": target_pair, "edge_oriented": last_layer_edges_oriented(goal_view(partial_state, cross_color)),
            "cross_color": cross_color}
