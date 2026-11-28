"""Bounded full-solve candidate comparison for CFOP and ZBLS-aware routes."""

import time

import torch

from cubemind.environment.cube import (
    COLORS, SIMULATOR_MOVE_TOKENS, SOLVED_STATE, apply_move, apply_sequence,
    cross_complete, cross_frame_rotation, f2l_complete, f2l_slot_count, zbll_eligible,
)
from cubemind.hierarchical_f2l import solve_hierarchical_f2l
from cubemind.neural_solver import _solve_cross_model, cube_solved, solve_last_layer
from cubemind.algorithm_memory import (
    solve_symbolic_last_layer, solve_symbolic_zbls,
)
from cubemind.zbls import solve_zbls


def _move_counts(moves):
    turns = [move for move in moves if move[0] not in "xyz"]
    return {
        "stm": sum(2 if move[0].islower() or move[1:2] == "w" else 1 for move in turns),
        "htm": sum(move[0] in "URFDLB" and move[1:2] != "w" for move in moves),
        "rotations": sum(move[0] in "xyz" for move in moves),
    }


def _canonicalize_color_labels(state):
    centers = tuple(state[face * 9 + 4] for face in range(6))
    canonical = tuple(SOLVED_STATE[face * 9 + 4] for face in range(6))
    color_map = dict(zip(centers, canonical))
    return tuple(color_map[color] for color in state)


def _candidate(name, start, moves, f2l_state, search_nodes, runtime, last_layer, phase_moves):
    moves = list(moves)
    final = apply_sequence(start, moves)
    verified = bool(last_layer and last_layer["verified"] and cube_solved(final)
                    and f2l_complete(f2l_state))
    counts = _move_counts(moves)
    return {
        "policy": name, "moves": moves, "last_state": final, "verified": verified,
        "moves_by_phase": phase_moves,
        "f2l_complete": f2l_complete(f2l_state),
        "zbll_eligible": zbll_eligible(f2l_state),
        "last_layer_route": last_layer.get("route") if last_layer else None,
        "last_layer_provenance": (last_layer.get("algorithm_source", "neural generation")
                                  if last_layer else None),
        "last_layer_case_ids": {key: last_layer[key] for key in
                                ("case_id", "oll_case_id", "pll_case_id")
                                if last_layer and last_layer.get(key)},
        "last_layer_algorithms": last_layer.get("algorithm_candidates", []) if last_layer else [],
        "failed_last_layer_attempts": last_layer.get("failed_attempts", []) if last_layer else [],
        "stm": counts["stm"], "htm": counts["htm"], "rotations": counts["rotations"],
        "search_nodes": search_nodes, "runtime_seconds": runtime,
        "failure_reason": None if verified else (
            last_layer.get("failure_reason") if last_layer else "phase_search_failed"),
    }


def solve_v2(state, cross_model, cross_history_length, f2l_model,
             last_layer_model, zbls_model, device, cross_color=COLORS["D"],
             f2l_options=None, last_layer_options=None, zbls_options=None,
             last_layer_engine="neural", zbls_engine="neural", include_zbls=True,
             compare_last_layer_routes=True, algorithm_preference_model=None):
    """Compare verified conventional CFOP, EO-aware CFOP, and neural ZBLS routes."""
    if last_layer_engine not in ("neural", "algorithm_memory"):
        raise ValueError("last_layer_engine must be neural or algorithm_memory")
    if zbls_engine not in ("neural", "algorithm_memory", "disabled"):
        raise ValueError("zbls_engine must be neural, algorithm_memory, or disabled")
    original = tuple(state)
    started = time.perf_counter()
    f2l_options, last_layer_options, zbls_options = (
        dict(f2l_options or {}), dict(last_layer_options or {}), dict(zbls_options or {}))
    frame_moves = cross_frame_rotation(original, cross_color)
    # Checkpoints use canonical white-down labels; color relabeling preserves every legal move.
    cross_start = _canonicalize_color_labels(apply_sequence(original, frame_moves))
    cross_started = time.perf_counter()
    if cross_complete(cross_start):
        cross_state, cross_moves, cross_nodes, cross_failure = cross_start, [], 0, None
    else:
        cross_state, _cross_history, cross_moves, cross_nodes, cross_failure = _solve_cross_model(
            cross_start, [], cross_model, cross_history_length, device, "search",
            256, 10, 10000, {cross_start},
        )
    cross_runtime = time.perf_counter() - cross_started
    total_nodes, total_seconds = cross_nodes, cross_runtime
    cross_prefix = list(frame_moves) + list(cross_moves)
    cross_state = apply_sequence(cross_start, cross_moves)
    if cross_failure or not cross_complete(cross_state):
        return {"status": "FAILED / INCOMPLETE", "selected": None, "candidates": [],
                "cross_moves": cross_prefix, "cross_complete": cross_complete(cross_state),
                "failure_reason": cross_failure or "cross_replay_failed",
                "selected_cross_color": cross_color, "cross_plus_one_pairs": 0,
                "zbls_attempt": None, "stages": {},
                "search_nodes": cross_nodes, "runtime_seconds": time.perf_counter() - started}
    history = [SIMULATOR_MOVE_TOKENS.index(move) for move in cross_prefix]
    candidates = []
    conventional = solve_hierarchical_f2l(
        cross_state, f2l_model, device, history=history, move_count=len(cross_prefix),
        conditional_legal=True, **f2l_options,
    )
    total_nodes += conventional["states_explored"]
    total_seconds += conventional["runtime_seconds"]
    stages = {
        "conventional_f2l": {
            "verified": conventional["verified"],
            "failure_reason": conventional["failure_reason"],
            "final_pairs": conventional["final_solved_pairs"],
            "states_explored": conventional["states_explored"],
            "runtime_seconds": conventional["runtime_seconds"],
        },
    }
    conventional_state = apply_sequence(cross_state, conventional["moves"])
    if conventional["verified"] and f2l_complete(conventional_state):
        prefix = cross_prefix + conventional["moves"]
        history_after_f2l = [SIMULATOR_MOVE_TOKENS.index(move) for move in prefix]
        if last_layer_engine == "algorithm_memory":
            strategies = (("oll-pll", "symbolic_cfop"), ("eo-aware", "symbolic_eo_aware")) \
                if compare_last_layer_routes else (("oll-pll", "symbolic_cfop"),)
        else:
            strategies = (("oll-pll", "cfop_oll_pll"), ("eo-aware", "cfop_eo_aware"))
        for strategy, name in strategies:
            if last_layer_engine == "algorithm_memory":
                last_layer_started = time.perf_counter()
                last_layer = solve_symbolic_last_layer(
                    conventional_state, strategy=strategy,
                    preference_model=algorithm_preference_model)
                last_layer.update({"states_explored": 0,
                                   "runtime_seconds": time.perf_counter() - last_layer_started,
                                   "failed_attempts": []})
                last_layer["algorithm_candidates"] = last_layer.get("algorithm_candidates", [])[:16]
            else:
                last_layer = solve_last_layer(
                    conventional_state, last_layer_model, device, mode="search",
                    history=history_after_f2l, strategy=strategy,
                    move_count=len(prefix), **last_layer_options,
                )
            total_nodes += last_layer["states_explored"]
            total_seconds += last_layer["runtime_seconds"]
            candidates.append(_candidate(
                name, original, prefix + last_layer["moves"], conventional_state,
                cross_nodes + conventional["states_explored"] + last_layer["states_explored"],
                cross_runtime + conventional["runtime_seconds"] + last_layer["runtime_seconds"],
                last_layer, {"CROSS": cross_prefix, "F2L": conventional["moves"],
                             **last_layer["moves_by_phase"]},
            ))

    prefix_f2l = solve_hierarchical_f2l(
        cross_state, f2l_model, device, history=history, move_count=len(cross_prefix),
        conditional_legal=True, stop_after_pairs=3, **f2l_options,
    )
    total_nodes += prefix_f2l["states_explored"]
    total_seconds += prefix_f2l["runtime_seconds"]
    stages["three_pair_prefix"] = {
        "verified": prefix_f2l["prefix_verified"],
        "failure_reason": prefix_f2l["failure_reason"],
        "final_pairs": prefix_f2l["final_solved_pairs"],
        "states_explored": prefix_f2l["states_explored"],
        "runtime_seconds": prefix_f2l["runtime_seconds"],
    }
    zbls_attempt = None
    if (include_zbls and zbls_engine != "disabled" and prefix_f2l.get("prefix_verified")
            and not f2l_complete(prefix_f2l["last_state"])):
        prefix = cross_prefix + prefix_f2l["moves"]
        prefix_state = apply_sequence(cross_state, prefix_f2l["moves"])
        zbls_started = time.perf_counter()
        if zbls_engine == "algorithm_memory":
            zbls_attempt = solve_symbolic_zbls(prefix_state, **zbls_options)
            zbls_attempt["states_explored"] = zbls_attempt.pop("states_checked")
        else:
            zbls_attempt = solve_zbls(
                prefix_state, zbls_model, device, cross_color=COLORS["D"],
                history=[SIMULATOR_MOVE_TOKENS.index(move) for move in prefix], **zbls_options,
            )
        zbls_attempt["runtime_seconds"] = time.perf_counter() - zbls_started
        total_nodes += zbls_attempt["states_explored"]
        total_seconds += zbls_attempt["runtime_seconds"]
        if zbls_attempt["verified"]:
            zbls_prefix = prefix + zbls_attempt["moves"]
            zbls_state = apply_sequence(prefix_state, zbls_attempt["moves"])
            if not (cross_complete(zbls_state) and f2l_complete(zbls_state)):
                raise AssertionError("ZBLS candidate failed committed Cross/F2L replay")
            if last_layer_engine == "algorithm_memory":
                last_layer_started = time.perf_counter()
                last_layer = solve_symbolic_last_layer(zbls_state, strategy="eo-aware")
                last_layer.update({"states_explored": 0,
                                   "runtime_seconds": time.perf_counter() - last_layer_started,
                                   "failed_attempts": [], "algorithm_candidates": []})
            else:
                last_layer = solve_last_layer(
                    zbls_state, last_layer_model, device, mode="search",
                    history=[SIMULATOR_MOVE_TOKENS.index(move) for move in zbls_prefix],
                    strategy="eo-aware", move_count=len(zbls_prefix), **last_layer_options,
                )
            total_nodes += last_layer["states_explored"]
            total_seconds += last_layer["runtime_seconds"]
            zbls_policy_name = ("algorithm_zbls_global" if zbls_engine == "algorithm_memory"
                                else "neural_zbls_global")
            candidates.append(_candidate(
                zbls_policy_name, original, zbls_prefix + last_layer["moves"],
                zbls_state,
                cross_nodes + prefix_f2l["states_explored"] + zbls_attempt["states_explored"]
                + last_layer["states_explored"],
                cross_runtime + prefix_f2l["runtime_seconds"] + zbls_attempt["runtime_seconds"]
                + last_layer["runtime_seconds"], last_layer,
                {"CROSS": cross_prefix, "F2L": prefix_f2l["moves"],
                 **({"ZBLS": zbls_attempt["moves"]} if zbls_engine == "algorithm_memory"
                    else {"F2L": zbls_attempt["moves"]}),
                 **last_layer["moves_by_phase"]},
            ))

    for candidate in candidates:
        candidate["cross_color"] = cross_color
    successful = [candidate for candidate in candidates if candidate["verified"]]
    selected = min(successful, key=lambda row: (row["stm"], row["rotations"], row["search_nodes"])) if successful else None
    return {
        "status": "SOLVED" if selected else "FAILED / INCOMPLETE",
        "failure_reason": None if selected else (
            stages["three_pair_prefix"]["failure_reason"]
            or stages["conventional_f2l"]["failure_reason"] or "no_verified_route"),
        "selected": selected, "candidates": candidates,
        "selected_cross_color": cross_color, "cross_moves": cross_prefix,
        "cross_complete": True, "cross_failure": None,
        "cross_plus_one_pairs": f2l_slot_count(cross_state),
        "zbls_attempt": (None if zbls_attempt is None else {
            "verified": zbls_attempt["verified"], "failure_reason": zbls_attempt["failure_reason"],
            "states_explored": zbls_attempt["states_explored"],
            "runtime_seconds": zbls_attempt["runtime_seconds"],
            "candidate_moves": zbls_attempt["moves"],
            "case_id": zbls_attempt.get("case_id"),
            "recognition": zbls_attempt.get("recognition"),
            "algorithm_candidates": zbls_attempt.get("candidates", []),
        }),
        "stages": stages,
        "selection_rule": "verified candidates only; minimum STM, then rotations, then search nodes",
        "search_nodes": total_nodes,
        "runtime_seconds": time.perf_counter() - started,
        "search_runtime_seconds": total_seconds,
        "last_layer_engine": last_layer_engine, "zbls_engine": zbls_engine,
    }


def solve_v2_profiles(state, cross_model, cross_history_length, f2l_model,
                       last_layer_model, zbls_model, device, cross_colors,
                       **options):
    """Run the bounded planner for an explicit White, dual-color, or six-color profile."""
    colors = list(dict.fromkeys(cross_colors))
    if not colors or any(type(color) is not int or color not in range(6) for color in colors):
        raise ValueError("cross_colors must contain one or more color IDs in 0..5")
    plans = [(color, solve_v2(state, cross_model, cross_history_length, f2l_model,
                              last_layer_model, zbls_model, device,
                              cross_color=color, **options)) for color in colors]
    candidates = []
    for color, plan in plans:
        for candidate in plan["candidates"]:
            candidates.append({**candidate, "cross_color": color})
    successful = [row for row in candidates if row["verified"]]
    selected = min(successful, key=lambda row: (row["stm"], row["rotations"], row["search_nodes"])) if successful else None
    selected_plan = next((plan for color, plan in plans if selected and color == selected["cross_color"]),
                         plans[0][1])
    return {**selected_plan, "status": "SOLVED" if selected else "FAILED / INCOMPLETE",
            "selected": selected, "candidates": candidates,
            "selected_cross_color": selected["cross_color"] if selected else None,
            "cross_colors_evaluated": colors,
            "cross_color_plans": [{"cross_color": color, "status": plan["status"],
                                   "candidate_count": len(plan["candidates"]),
                                   "selected_stm": plan["selected"]["stm"] if plan["selected"] else None,
                                   "runtime_seconds": plan["runtime_seconds"]}
                                  for color, plan in plans],
            "search_nodes": sum(plan["search_nodes"] for _color, plan in plans),
            "runtime_seconds": sum(plan["runtime_seconds"] for _color, plan in plans),
            "selection_rule": "verified candidates only; minimum STM, then rotations, then search nodes"}
