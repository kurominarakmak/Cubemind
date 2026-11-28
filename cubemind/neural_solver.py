"""Phase-aware neural rollout and neural-guided beam search."""

import json
import math
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from cubemind.environment.cube import (
    COLORS, MOVE_TOKENS, SIMULATOR_MOVE_TOKENS, SOLVED_STATE, apply_move, apply_sequence,
    cross_complete, f2l_complete, f2l_slot_count, inverse_move,
    last_layer_edges_oriented, normalize_cube_orientation, parse_moves, zbll_eligible,
)
from cubemind.models.solver import SOLVER_FEATURE_SIZE, SolverPolicyValue
from cubemind.models.f2l import F2LPolicy
from cubemind.models.common import encode_history

ROOT = Path(__file__).resolve().parents[1]
TEACHER_SCRIPT = ROOT / "scripts" / "cfop_teacher.mjs"
PHASES = ("CROSS", "F2L", "OLL", "PLL", "ZBLL")
PHASE_IDS = {phase: index for index, phase in enumerate(PHASES)}
MOVE_IDS = {move: index for index, move in enumerate(SIMULATOR_MOVE_TOKENS)}
HISTORY_LENGTH = 8
EXPERT_F2L_FEATURE_SIZE = 54 * 6 + HISTORY_LENGTH * len(SIMULATOR_MOVE_TOKENS) + 4 + 5 + 6 + 4
MOVE_BUDGETS = {"CROSS": 20, "F2L": 60, "OLL": 25, "PLL": 25, "ZBLL": 35}


def oll_complete(state):
    """Detect a correctly oriented upper face after F2L."""
    return f2l_complete(state) and all(color == state[COLORS["U"] * 9 + 4] for color in state[:9])


def cube_solved(state):
    try:
        return normalize_cube_orientation(state) == SOLVED_STATE
    except KeyError:
        return False


def phase_complete(state, phase):
    if phase == "CROSS":
        return cross_complete(state)
    if phase == "F2L":
        return f2l_complete(state)
    if phase == "OLL":
        return oll_complete(state)
    if phase in ("PLL", "ZBLL"):
        return cube_solved(state)
    raise ValueError("Unknown solving phase: {}".format(phase))


def encode_features(state, history, phase, move_count, device, cross_color=None):
    if phase not in PHASE_IDS:
        raise ValueError("Unknown solving phase: {}".format(phase))
    features = torch.zeros(SOLVER_FEATURE_SIZE, dtype=torch.float32, device=device)
    state_size = 54 * 6
    features[:state_size] = F.one_hot(torch.tensor(state, device=device), 6).float().flatten()
    offset = state_size
    for index, move_id in enumerate(history[-HISTORY_LENGTH:], HISTORY_LENGTH - len(history[-HISTORY_LENGTH:])):
        if move_id not in range(len(SIMULATOR_MOVE_TOKENS)):
            raise ValueError("Invalid move-history ID")
        features[offset + index * len(SIMULATOR_MOVE_TOKENS) + move_id] = 1.0
    offset += HISTORY_LENGTH * len(SIMULATOR_MOVE_TOKENS)
    features[offset + PHASE_IDS[phase]] = 1.0
    offset += len(PHASES)
    features[offset] = 1.0  # Standard F2L context.
    offset += 5
    if cross_color is None:
        cross_color = COLORS["D"]  # Existing solver checkpoints use a white Down cross.
    if type(cross_color) is not int or cross_color not in range(6):
        raise ValueError("Cross color must be a color ID in 0..5")
    features[offset + cross_color] = 1.0
    offset += 6
    features[offset:offset + 4] = torch.tensor([
        float(last_layer_edges_oriented(state)), float(cross_complete(state)),
        f2l_slot_count(state) / 4.0, min(float(move_count) / 60.0, 1.0),
    ], dtype=torch.float32, device=device)
    return features


def load_solver_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("sample_kind") != "synthetic" or checkpoint.get("data_source") not in (
            "synthetic:cfop-teacher+zbll-algorithms", "synthetic:cfop-teacher+dagger+zbll-algorithms"):
        raise ValueError("This checkpoint is not the labeled synthetic CFOP solver model")
    if checkpoint.get("feature_size") != SOLVER_FEATURE_SIZE or checkpoint.get("actions") != len(SIMULATOR_MOVE_TOKENS):
        raise ValueError("Unsupported neural solver checkpoint schema")
    model = SolverPolicyValue().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def load_last_layer_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if (checkpoint.get("sample_kind") != "mixed:expert+synthetic"
            or checkpoint.get("feature_size") != SOLVER_FEATURE_SIZE
            or checkpoint.get("actions") != len(SIMULATOR_MOVE_TOKENS)
            or tuple(checkpoint.get("phase_names", ())) != ("OLL", "PLL", "ZBLL")):
        raise ValueError("Checkpoint is not a supported expert-aware last-layer policy")
    model = SolverPolicyValue().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


class ExpertF2LPolicyAdapter(torch.nn.Module):
    """Maps the shared solver state to the existing reco.nz F2L model's context schema."""

    def __init__(self, expert_model, value_model):
        super().__init__()
        self.expert_model = expert_model
        self.value_model = value_model

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != SOLVER_FEATURE_SIZE:
            raise ValueError("Expected the shared solver feature schema")
        expert_features = features.new_zeros((len(features), EXPERT_F2L_FEATURE_SIZE))
        expert_features[:, :900] = features[:, :900]
        expert_features[:, 902] = 1.0  # Human checkpoint phase order: CROSS, CROSS+1, F2L, ZBLS.
        expert_features[:, 904] = 1.0  # Use the standard-F2L context at inference.
        expert_features[:, 909 + COLORS["D"]] = 1.0
        expert_features[:, 915:919] = features[:, 916:920]
        logits = self.expert_model(expert_features)
        _policy_logits, values = self.value_model(features)
        return logits, values


def load_expert_f2l_policy(path, value_model, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = checkpoint.get("model_config", {})
    if (checkpoint.get("sample_kind") != "expert" or checkpoint.get("data_source") != "reco.nz 3x3 CFOP reconstructions"
            or config.get("input_size") != EXPERT_F2L_FEATURE_SIZE
            or config.get("actions") != len(SIMULATOR_MOVE_TOKENS)):
        raise ValueError("Checkpoint is not the expected reco.nz F2L action model")
    expert_model = F2LPolicy(EXPERT_F2L_FEATURE_SIZE, len(SIMULATOR_MOVE_TOKENS)).to(device)
    expert_model.load_state_dict(checkpoint["model_state"])
    expert_model.eval()
    return ExpertF2LPolicyAdapter(expert_model, value_model).to(device).eval(), checkpoint


def teacher_request(payload):
    try:
        completed = subprocess.run(
            ["node", str(TEACHER_SCRIPT)], input=json.dumps(payload), text=True,
            capture_output=True, check=True, cwd=ROOT,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        raise RuntimeError("CFOP teacher command failed: {}".format(detail.strip())) from error
    return json.loads(completed.stdout)


def reference_solve(scramble):
    row = teacher_request({"mode": "solve", "scramble": scramble})[0]
    moves = [move for phase in PHASES[:4] for move in row["phases"][phase]]
    end = apply_sequence(apply_sequence(SOLVED_STATE, parse_moves(scramble)), moves)
    return {
        "status": "SOLVED" if row["reference_verified"] and cube_solved(end) else "FAILED / INCOMPLETE",
        "moves_by_phase": row["phases"], "moves": moves, "last_state": end,
        "verified": cube_solved(end), "states_explored": 0,
        "runtime_seconds": row["reference_ms"] / 1000.0, "failure_reason": None,
        "zbll_eligible_after_f2l": zbll_eligible(_state_after(scramble, row["phases"]["CROSS"] + row["phases"]["F2L"])),
        "route": "deterministic CFOP reference: Cross → F2L → OLL → PLL",
    }


def _state_after(scramble, moves):
    return apply_sequence(apply_sequence(SOLVED_STATE, parse_moves(scramble)), moves)


def _policy_output(model, states, histories, phases, move_counts, device):
    features = torch.stack([
        encode_features(state, history, phase, move_count, device)
        for state, history, phase, move_count in zip(states, histories, phases, move_counts)
    ])
    with torch.inference_mode():
        logits, values = model(features)
    if logits.ndim != 2 or logits.shape[1] != len(SIMULATOR_MOVE_TOKENS) or not torch.isfinite(logits).all():
        raise ValueError("Neural policy returned invalid action logits")
    if values.ndim != 1 or not torch.isfinite(values).all():
        raise ValueError("Neural value head returned invalid estimates")
    return logits, values.clamp_min(0)


class _ValueOnlyPolicy(torch.nn.Module):
    """Matched search ablation that retains the learned value but removes action guidance."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features):
        logits, values = self.model(features)
        return torch.zeros_like(logits), values


def _solve_cross_model(state, history, model, history_length, device, mode,
                       beam_width, max_depth, max_nodes, seen_total):
    current, global_history, path = tuple(state), list(history), []
    if mode == "greedy":
        seen = {current}
        for _step in range(max_depth):
            states = torch.tensor([current], dtype=torch.long, device=device)
            encoded_history = encode_history([[MOVE_TOKENS.index(move) for move in path]], history_length, device)
            with torch.inference_mode():
                logits = model(states, encoded_history)
            if logits.ndim != 2 or logits.shape[1] != len(MOVE_TOKENS) or not torch.isfinite(logits).all():
                return current, global_history, path, len(path), "invalid_cross_policy_output"
            move = MOVE_TOKENS[int(logits[0].argmax())]
            current = apply_move(current, move)
            path.append(move)
            global_history.append(MOVE_IDS[move])
            if cross_complete(current):
                return current, global_history, path, len(path), None
            if current in seen or current in seen_total:
                return current, global_history, path, len(path), "repeated_state:CROSS"
            seen.add(current)
            seen_total.add(current)
        return current, global_history, path, len(path), "phase_move_budget:CROSS"

    beam = [(current, [], 0.0)]
    seen = {current}
    best = beam[0]
    explored = 0
    for _depth in range(1, max_depth + 1):
        parents = beam[:max(0, min(len(beam), max_nodes - explored))]
        if not parents:
            break
        states = torch.tensor([node[0] for node in parents], dtype=torch.long, device=device)
        histories = encode_history(
            [[MOVE_TOKENS.index(move) for move in node[1]] for node in parents], history_length, device,
        )
        with torch.inference_mode():
            logits = model(states, histories)
        if logits.ndim != 2 or logits.shape[1] != len(MOVE_TOKENS) or not torch.isfinite(logits).all():
            return current, global_history, path, explored, "invalid_cross_policy_output"
        explored += len(parents)
        log_probs = torch.log_softmax(logits, dim=1)
        next_by_state, found = {}, None
        for parent_index, (parent_state, parent_path, parent_score) in enumerate(parents):
            previous_face = parent_path[-1][0] if parent_path else None
            for move_id in log_probs[parent_index].argsort(descending=True).tolist():
                move = MOVE_TOKENS[move_id]
                if previous_face == move[0]:
                    continue
                child = apply_move(parent_state, move)
                child_path = parent_path + [move]
                score = parent_score + float(log_probs[parent_index, move_id])
                if cross_complete(child):
                    found = (child, child_path)
                    break
                if child in seen or child in seen_total:
                    continue
                previous = next_by_state.get(child)
                if previous is None or score > previous[2]:
                    next_by_state[child] = (child, child_path, score)
                    seen.add(child)
            if found:
                break
        if found:
            current, path = found
            global_history.extend(MOVE_IDS[move] for move in path)
            return current, global_history, path, explored, None
        beam = sorted(next_by_state.values(), key=lambda node: node[2], reverse=True)[:beam_width]
        if beam:
            best = beam[0]
        if explored >= max_nodes or not beam:
            break
    current, path, _score = best
    global_history.extend(MOVE_IDS[move] for move in path)
    return current, global_history, path, explored, "search_budget:CROSS"


def _solve_phase(state, history, phase, model, device, mode, budget, beam_width, branching,
                 max_nodes, seen_total, move_count=None, action_score_weight=0.05):
    current, history = tuple(state), list(history)
    base_move_count = len(history) if move_count is None else int(move_count)
    if base_move_count < 0:
        raise ValueError("Move count cannot be negative")
    path, explored = [], 0
    if mode == "greedy":
        phase_seen = {current}
        for _step in range(int(budget)):
            logits, _values = _policy_output(
                model, [current], [history], [phase], [base_move_count + len(path)], device,
            )
            explored += 1
            move_id = int(logits[0].argmax())
            move = SIMULATOR_MOVE_TOKENS[move_id]
            current = apply_move(current, move)
            history.append(move_id)
            path.append(move)
            if phase_complete(current, phase):
                return current, history, path, explored, None
            if current in phase_seen or current in seen_total:
                return current, history, path, explored, "repeated_state:{}".format(phase)
            phase_seen.add(current)
            seen_total.add(current)
        return current, history, path, explored, "phase_move_budget:{}".format(phase)

    beam = [(current, history, [], 0.0)]
    phase_seen = {current}
    best = beam[0]
    for depth in range(1, int(budget) + 1):
        parents = beam[:max(0, min(len(beam), max_nodes - explored))]
        if not parents:
            break
        logits, _values = _policy_output(
            model, [item[0] for item in parents], [item[1] for item in parents],
            [phase] * len(parents), [base_move_count + len(item[2]) for item in parents], device,
        )
        explored += len(parents)
        log_probs = torch.log_softmax(logits, dim=1)
        next_by_state, found = {}, None
        for parent_index, (parent_state, parent_history, parent_path, log_score) in enumerate(parents):
            if isinstance(model, _ValueOnlyPolicy):
                top = torch.randperm(len(SIMULATOR_MOVE_TOKENS))[:branching].tolist()
            else:
                top = log_probs[parent_index].topk(min(branching, len(SIMULATOR_MOVE_TOKENS))).indices.tolist()
            previous = SIMULATOR_MOVE_TOKENS[parent_history[-1]] if parent_history else None
            previous_base = previous.rstrip("'2") if previous else None
            for move_id in top:
                move = SIMULATOR_MOVE_TOKENS[move_id]
                if previous and move == inverse_move(previous):
                    continue
                if previous_base and move.rstrip("'2") == previous_base:
                    continue
                child = apply_move(parent_state, move)
                child_path = parent_path + [move]
                child_history = parent_history + [move_id]
                child_log_score = log_score + float(log_probs[parent_index, move_id])
                if phase_complete(child, phase):
                    found = (child, child_history, child_path)
                    break
                if child in phase_seen or child in seen_total:
                    continue
                prior = next_by_state.get(child)
                if prior is None or child_log_score > prior[3]:
                    next_by_state[child] = (child, child_history, child_path, child_log_score)
                    phase_seen.add(child)
            if found:
                break
        if found:
            final, final_history, final_path = found
            return final, final_history, final_path, explored, None
        candidates = list(next_by_state.values())
        if not candidates:
            break
        _candidate_logits, values = _policy_output(
            model, [item[0] for item in candidates], [item[1] for item in candidates],
            [phase] * len(candidates), [base_move_count + len(item[2]) for item in candidates], device,
        )
        ranked = sorted(
            ((depth + float(value) - action_score_weight * node[3], node)
             for node, value in zip(candidates, values)),
            key=lambda item: item[0],
        )
        beam = [node for _score, node in ranked[:beam_width]]
        best = beam[0]
        if explored >= max_nodes:
            break
    best_state, best_history, best_path, _score = best
    return best_state, best_history, best_path, explored, "search_budget:{}".format(phase)


def solve_last_layer(state, model, device, mode="search", history=(), strategy="eo-aware",
                     budgets=None, beam_width=128, branching=8, max_nodes=20000, max_moves=None,
                     allow_zbll_recovery=True, action_guidance=True, move_count=None,
                     action_score_weight=1.0):
    """Solve an already completed F2L using only the learned policy and simulator."""
    if mode not in ("greedy", "search"):
        raise ValueError("mode must be greedy or search")
    if strategy not in ("eo-aware", "oll-pll"):
        raise ValueError("strategy must be eo-aware or oll-pll")
    if (beam_width < 1 or branching < 1 or max_nodes < 1
            or (max_moves is not None and max_moves < 0)
            or not math.isfinite(action_score_weight) or action_score_weight < 0
            or any(int(value) < 1 for value in (budgets or {}).values())):
        raise ValueError("Last-layer search parameters must be valid")
    if not action_guidance and mode != "search":
        raise ValueError("The value-only action-guidance ablation requires search mode")
    current = tuple(state)
    if len(current) != 54:
        raise ValueError("Cube state must contain 54 facelets")
    if not f2l_complete(current):
        return {"status": "FAILED / INCOMPLETE", "moves": [], "moves_by_phase": {"OLL": [], "PLL": [], "ZBLL": []},
                "last_state": current, "verified": False, "failure_reason": "f2l_not_complete",
                "states_explored": 0, "runtime_seconds": 0.0, "route": strategy}

    started = time.perf_counter()
    original = current
    history = list(history)
    if any(move_id not in range(len(SIMULATOR_MOVE_TOKENS)) for move_id in history):
        raise ValueError("Invalid initial move-history ID")
    if move_count is not None and (type(move_count) is not int or move_count < 0):
        raise ValueError("Move count must be a non-negative integer")
    initial_move_count = len(history) if move_count is None else move_count
    moves_by_phase = {"OLL": [], "PLL": [], "ZBLL": []}
    chronological_moves = []
    phase_states = []
    seen_total = {current}
    states_explored = 0
    phase_budgets = {"OLL": 40, "PLL": 40, "ZBLL": 100, **(budgets or {})}
    policy = model if action_guidance else _ValueOnlyPolicy(model)
    eligible = zbll_eligible(current)
    phases = ["ZBLL"] if eligible and strategy == "eo-aware" else ["OLL", "PLL"]
    recovery_used = False
    failure = None
    attempts = []
    phase_index = 0
    while phase_index < len(phases):
        phase = phases[phase_index]
        if phase_complete(current, phase):
            phase_states.append((phase, current))
            phase_index += 1
            continue
        used = sum(len(path) for path in moves_by_phase.values())
        overall_budget = (sum(int(value) for value in phase_budgets.values())
                          if max_moves is None else int(max_moves))
        remaining = overall_budget - used
        if remaining <= 0:
            failure = "total_move_budget"
            phase_states.append((phase, current))
            break
        checkpoint_state, checkpoint_history = current, list(history)
        phase_seen_total = set(seen_total)
        attempt_started = time.perf_counter()
        candidate_state, candidate_history, path, explored, failure = _solve_phase(
            checkpoint_state, checkpoint_history, phase, policy, device, mode,
            min(int(phase_budgets[phase]), remaining), beam_width, branching,
            max_nodes, phase_seen_total, initial_move_count + len(chronological_moves),
            action_score_weight,
        )
        states_explored += explored
        attempt_seconds = time.perf_counter() - attempt_started
        if failure:
            attempts.append({
                "phase": phase, "candidate_moves": list(path),
                "candidate_state": candidate_state,
                "failure_reason": failure, "states_explored": explored,
                "runtime_seconds": attempt_seconds,
                "f2l_preserved": f2l_complete(candidate_state),
            })
            if (phase == "ZBLL" and allow_zbll_recovery and f2l_complete(checkpoint_state)
                    and not recovery_used):
                # The ZBLL path is a speculative branch; fallback starts at the saved F2L boundary.
                current, history = checkpoint_state, checkpoint_history
                phase_states.append((phase, checkpoint_state))
                phases = phases[:phase_index + 1] + ["OLL", "PLL"]
                recovery_used = True
                failure = None
                phase_index += 1
                continue
            current, history = candidate_state, candidate_history
            moves_by_phase[phase].extend(path)
            chronological_moves.extend(path)
            phase_states.append((phase, current))
            break
        current, history = candidate_state, candidate_history
        moves_by_phase[phase].extend(path)
        chronological_moves.extend(path)
        seen_total.update(phase_seen_total)
        phase_states.append((phase, current))
        phase_index += 1

    verified = (failure is None and cube_solved(current) and f2l_complete(current)
                and all(phase_complete(current, phase) for phase in ("OLL", "PLL", "ZBLL")
                        if moves_by_phase[phase] or phase in phases))
    if failure is None and not verified:
        failure = "final_state_not_solved_or_f2l_not_preserved"
    route = "ZBLL" if eligible and strategy == "eo-aware" else "OLL → PLL"
    if recovery_used:
        route = "ZBLL → OLL → PLL neural recovery"
    return {
        "status": "SOLVED" if verified else "FAILED / INCOMPLETE",
        "moves_by_phase": moves_by_phase,
        "moves": chronological_moves,
        "last_state": current, "phase_states": phase_states, "verified": verified,
        "states_explored": states_explored, "runtime_seconds": time.perf_counter() - started,
        "failure_reason": failure, "zbll_eligible": eligible, "route": route,
        "strategy": strategy, "recovery_used": recovery_used,
        "failed_attempts": attempts,
        "f2l_preserved": f2l_complete(current) if verified else None,
        "initial_state": original,
        "provenance": ("CubeMind last-layer neural policy; simulator verified; no deterministic fallback"
                       if action_guidance else "learned-value-only search ablation; no action-policy guidance"),
    }


def solve_expert_aware_full(state, last_layer_model, device, cross_model, cross_history_length,
                            f2l_policy, mode="search", strategy="eo-aware", max_moves=120,
                            cross_search_width=256, cross_search_depth=10,
                            cross_search_max_nodes=10000, f2l_search=None,
                            last_layer_search=None, last_layer_mode=None):
    """Run the existing expert-aware Cross/F2L stack, then neural last-layer solving."""
    from cubemind.hierarchical_f2l import solve_hierarchical_f2l

    if mode not in ("greedy", "search"):
        raise ValueError("mode must be greedy or search")
    initial = tuple(state)
    if len(initial) != 54:
        raise ValueError("Cube state must contain 54 facelets")
    started = time.perf_counter()
    last_layer_mode = mode if last_layer_mode is None else last_layer_mode
    current, history = initial, []
    moves_by_phase = {phase: [] for phase in PHASES}
    phase_states = []
    nodes = 0
    failure = None
    f2l_result = None

    if not cross_complete(current):
        current, history, cross_moves, explored, failure = _solve_cross_model(
            current, history, cross_model, cross_history_length, device, mode,
            cross_search_width, cross_search_depth, cross_search_max_nodes, {current},
        )
        moves_by_phase["CROSS"].extend(cross_moves)
        nodes += explored
    phase_states.append(("CROSS", current))
    if failure is None and cross_complete(current) and not f2l_complete(current):
        config = {"width": 128, "branching": 6, "max_nodes": 6000, "max_depth": 22,
                  "max_moves": max_moves, "max_actions_per_pair": 22, "max_order_attempts": 24,
                  **(f2l_search or {})}
        f2l = solve_hierarchical_f2l(
            current, f2l_policy, device, history=history, mode=mode,
            width=config["width"], branching=config["branching"], max_nodes=config["max_nodes"],
            max_depth=config["max_depth"], max_moves=config["max_moves"],
            max_actions_per_pair=config["max_actions_per_pair"],
            max_order_attempts=config["max_order_attempts"], move_count=len(history),
            conditional_legal=True,
        )
        moves_by_phase["F2L"].extend(f2l["moves"])
        f2l_result = f2l
        current = apply_sequence(current, f2l["moves"])
        history.extend(MOVE_IDS[move] for move in f2l["moves"])
        nodes += f2l["states_explored"]
        if not f2l["verified"] or not f2l_complete(current):
            failure = f2l["failure_reason"] or "f2l_result_failed_replay"
    phase_states.append(("F2L", current))

    last_layer = None
    if failure is None:
        remaining = max(0, max_moves - len(moves_by_phase["CROSS"]) - len(moves_by_phase["F2L"]))
        last_layer_options = {"max_moves": remaining, **(last_layer_search or {})}
        last_layer = solve_last_layer(
            current, last_layer_model, device, mode=last_layer_mode, history=history, strategy=strategy,
            **last_layer_options,
        )
        for phase, path in last_layer["moves_by_phase"].items():
            moves_by_phase[phase].extend(path)
        current = apply_sequence(current, last_layer["moves"])
        phase_states.extend(last_layer.get("phase_states", []))
        nodes += last_layer["states_explored"]
        failure = last_layer["failure_reason"]

    moves = ([move for phase in PHASES for move in moves_by_phase[phase]]
             if last_layer is None else
             moves_by_phase["CROSS"] + moves_by_phase["F2L"] + last_layer["moves"])
    replay = apply_sequence(initial, moves)
    verified = failure is None and cube_solved(replay) and cube_solved(current) and replay == current
    if failure is None and not verified:
        failure = "full_solution_replay_failed"
    return {
        "status": "SOLVED" if verified else "FAILED / INCOMPLETE",
        "moves_by_phase": moves_by_phase, "moves": moves, "last_state": replay,
        "phase_states": phase_states, "verified": verified,
        "cross_complete": cross_complete(dict(phase_states).get("CROSS", current)),
        "f2l_complete": f2l_complete(dict(phase_states).get("F2L", current)),
        "zbll_eligible_after_f2l": zbll_eligible(dict(phase_states).get("F2L", current)),
        "solved_pair_mask": f2l_result["solved_pair_mask"] if f2l_result else None,
        "pair_results": f2l_result["pair_results"] if f2l_result else [],
        "pair_attempts": f2l_result["pair_attempts"] if f2l_result else [],
        "states_explored": nodes, "runtime_seconds": time.perf_counter() - started,
        "failure_reason": failure, "last_layer_route": last_layer["route"] if last_layer else None,
        "recovery_used": last_layer["recovery_used"] if last_layer else False,
        "provenance": "existing expert-aware neural Cross/F2L + CubeMind neural last-layer policy; simulator replay",
        "human_execution_or_recognition_time": "not measured",
    }


def solve_neural(state, model, device, mode="greedy", budgets=None,
                 beam_width=16, branching=6, max_nodes=3000, max_moves=120,
                 cross_model=None, cross_history_length=8, cross_search_width=256,
                 cross_search_depth=10, cross_search_max_nodes=10000, f2l_model=None):
    if mode not in ("greedy", "search"):
        raise ValueError("mode must be greedy or search")
    current = tuple(state)
    if len(current) != 54:
        raise ValueError("Cube state must contain 54 facelets")
    phase_budgets = {**MOVE_BUDGETS, **(budgets or {})}
    if (max_moves < 1 or beam_width < 1 or branching < 1 or max_nodes < 1
            or cross_search_width < 1 or cross_search_depth < 1 or cross_search_max_nodes < 1
            or any(int(value) < 1 for value in phase_budgets.values())):
        raise ValueError("Move and search budgets must be positive")
    history, moves_by_phase, phase_states = [], {phase: [] for phase in PHASES}, []
    seen_total = {current}
    states_explored = 0
    started = time.perf_counter()
    zbll_eligible_after_f2l = False
    failure = None
    route = ["CROSS", "F2L"]
    for phase in route:
        if not phase_complete(current, phase):
            remaining_moves = max_moves - len(moves_by_phase[phase]) - sum(
                len(moves_by_phase[prior]) for prior in route if prior != phase
            )
            if remaining_moves <= 0:
                failure = "total_move_budget"
                phase_states.append((phase, current))
                break
            if phase == "CROSS" and cross_model is not None:
                cross_limit = min(phase_budgets[phase], remaining_moves)
                if mode == "search":
                    cross_limit = min(cross_limit, cross_search_depth)
                current, history, path, explored, failure = _solve_cross_model(
                    current, history, cross_model, cross_history_length, device, mode,
                    cross_search_width, cross_limit,
                    cross_search_max_nodes, seen_total,
                )
            else:
                phase_model = f2l_model if phase == "F2L" and f2l_model is not None else model
                current, history, path, explored, failure = _solve_phase(
                    current, history, phase, phase_model, device, mode, min(phase_budgets[phase], remaining_moves),
                    beam_width, branching, max_nodes, seen_total,
                )
            moves_by_phase[phase].extend(path)
            states_explored += explored
        phase_states.append((phase, current))
        if failure:
            break

    if failure is None:
        zbll_eligible_after_f2l = zbll_eligible(current)
        remaining_phases = ["ZBLL"] if zbll_eligible_after_f2l else ["OLL", "PLL"]
        route.extend(remaining_phases)
        for phase in remaining_phases:
            if not phase_complete(current, phase):
                remaining_moves = max_moves - sum(len(moves) for moves in moves_by_phase.values())
                if remaining_moves <= 0:
                    failure = "total_move_budget"
                    phase_states.append((phase, current))
                    break
                current, history, path, explored, failure = _solve_phase(
                    current, history, phase, model, device, mode, min(phase_budgets[phase], remaining_moves),
                    beam_width, branching, max_nodes, seen_total,
                )
                moves_by_phase[phase].extend(path)
                states_explored += explored
            phase_states.append((phase, current))
            if failure:
                break

    verified = failure is None and cube_solved(current) and all(phase_complete(dict(phase_states).get(p, current), p) for p in route)
    if failure is None and not verified:
        failure = "final_state_not_solved"
    return {
        "status": "SOLVED" if verified else "FAILED / INCOMPLETE",
        "moves_by_phase": moves_by_phase,
        "moves": [move for phase in PHASES for move in moves_by_phase[phase]],
        "phase_states": phase_states,
        "last_state": current,
        "verified": verified,
        "states_explored": states_explored,
        "runtime_seconds": time.perf_counter() - started,
        "failure_reason": failure,
        "zbll_eligible_after_f2l": zbll_eligible_after_f2l,
        "route": ("project Cross MLP + reco.nz F2L policy + synthetic phase/value MLP with neural-guided search"
                  if mode == "search" and cross_model is not None and f2l_model is not None else
                  "project Cross MLP + reco.nz F2L policy + synthetic phase/value MLP greedy rollout"
                  if cross_model is not None and f2l_model is not None else
                  "phase policy/value MLP with neural-guided search"
                  if mode == "search" and cross_model is not None else
                  "project Cross MLP + phase policy/value MLP greedy rollout"
                  if cross_model is not None else
                  "phase policy/value MLP with neural-guided search" if mode == "search" else
                  "phase policy/value MLP greedy rollout"),
    }
