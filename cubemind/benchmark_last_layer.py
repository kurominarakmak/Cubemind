"""Held-out stage and full-solve evaluation for the neural last-layer policy."""

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import torch

from cubemind.benchmark_expert_aware_f2l import _fresh_scrambles
from cubemind.environment.cube import (
    SIMULATOR_MOVE_TOKENS, SOLVED_STATE, apply_sequence, cross_complete, f2l_complete, parse_moves,
    zbll_eligible,
)
from cubemind.expert_aware_f2l import (
    ExpertAwareF2LPolicy, load_canonical_expert_action_model, load_expert_pair_selector,
)
from cubemind.hierarchical_f2l import load_hierarchical_f2l, solve_hierarchical_f2l
from cubemind.last_layer import recognize_zbll_case
from cubemind.models.zbll import ZBLLClassifier
from cubemind.neural_solver import (
    cube_solved, load_last_layer_checkpoint, oll_complete, solve_last_layer,
    teacher_request, _solve_cross_model,
)
from cubemind.train_last_layer import DATA_DIR, DEFAULT_CHECKPOINT
from cubemind.training.engine import choose_device, model_from_checkpoint

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DEFAULT_OUTPUT = ARTIFACTS / "last-layer-benchmark.json"
LL_SEARCH = {"beam_width": 128, "branching": 8, "max_nodes": 20000,
             "budgets": {"OLL": 40, "PLL": 40, "ZBLL": 100},
             "action_score_weight": 1.0}
F2L_SEARCH = {"width": 128, "branching": 6, "max_nodes": 6000, "max_depth": 22,
              "max_moves": 100, "max_actions_per_pair": 22, "max_order_attempts": 24}


def _wilson(successes, total):
    if not total:
        return [None, None]
    z, p = 1.96, successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _move_counts(moves):
    turns = [move for move in moves if move[0] not in "xyz"]
    return {
        "stm": sum(2 if move[0].islower() or move[1:2] == "w" else 1 for move in turns),
        "htm": sum(move[0] in "URFDLB" and move[1:2] != "w" for move in moves),
        "rotations": sum(move[0] in "xyz" for move in moves),
    }


def _aggregate(rows):
    count = len(rows)
    successes = [row for row in rows if row["verified"]]
    successful_moves = [row["stm"] for row in successes]
    phase_fields = ("cross_complete", "f2l_complete", "oll_complete", "pll_complete", "zbll_eligible")
    phase_samples = {field: sum(row.get(field) is not None for row in rows) for field in phase_fields}
    phase_rates = {
        field: (sum(bool(row[field]) for row in rows if row.get(field) is not None) / phase_samples[field]
                if phase_samples[field] else None)
        for field in phase_fields
    }
    return {
        "cases": count, "successes": len(successes), "success_rate": len(successes) / count if count else None,
        "success_rate_wilson_95": _wilson(len(successes), count),
        "cross_success_rate": phase_rates["cross_complete"],
        "f2l_success_rate": phase_rates["f2l_complete"],
        "oll_success_rate": phase_rates["oll_complete"],
        "pll_success_rate": phase_rates["pll_complete"],
        "zbll_eligibility_rate": phase_rates["zbll_eligible"],
        "phase_metric_samples": phase_samples,
        "direct_zbll_successes": sum(row.get("direct_zbll_success", False) for row in rows),
        "zbll_recoveries": sum(row.get("recovery_used", False) for row in rows),
        "mean_stm_successful": statistics.mean(successful_moves) if successful_moves else None,
        "median_stm_successful": statistics.median(successful_moves) if successful_moves else None,
        "mean_rotations": statistics.mean(row["rotations"] for row in rows) if rows else None,
        "mean_search_nodes": statistics.mean(row.get("search_nodes", row.get("states_explored", 0))
                                              for row in rows) if rows else None,
        "mean_runtime_ms": statistics.mean(row["runtime_seconds"] for row in rows) * 1000 if rows else None,
        "failure_reasons": {reason: sum(row.get("failure_reason") == reason for row in rows)
                            for reason in sorted({row.get("failure_reason") for row in rows if row.get("failure_reason")})},
    }


def _load_f2l_models(device):
    cross, cross_checkpoint = model_from_checkpoint(ARTIFACTS / "mlp-state-history.pt", device)
    history_length = int(cross_checkpoint["model_config"].get("max_history", 8))
    planner, planner_checkpoint = load_hierarchical_f2l(ARTIFACTS / "hierarchical-f2l.pt", device)
    expert, expert_checkpoint = load_canonical_expert_action_model(ARTIFACTS / "canonical-expert-f2l.pt", device)
    selector, selector_checkpoint = load_expert_pair_selector(ARTIFACTS / "expert-aware-pair-selector.pt", device)
    evaluation = json.loads((ARTIFACTS / "expert-aware-f2l-evaluation.json").read_text(encoding="utf-8"))
    calibration = evaluation["action_fusion"]["calibration"]
    expert_policy = ExpertAwareF2LPolicy(
        planner, expert, calibration["expert_weight"], 1.0 - calibration["expert_weight"],
        calibration["expert_temperature"], calibration["planner_temperature"], selector,
        canonical_expert=True,
    ).eval()
    return cross, history_length, planner, expert_policy, {
        "cross": cross_checkpoint, "planner": planner_checkpoint,
        "expert": expert_checkpoint, "selector": selector_checkpoint,
        "calibration": calibration,
    }


def _f2l_prefix(scramble, start, cross, history_length, policy, device, config):
    started = time.perf_counter()
    if cross_complete(start):
        cross_state, history, cross_moves, cross_nodes, cross_failure = tuple(start), [], [], 0, None
    else:
        cross_state, history, cross_moves, cross_nodes, cross_failure = _solve_cross_model(
            start, [], cross, history_length, device, "search", 256, 10, 10000, {start},
        )
    if cross_failure or not cross_complete(cross_state):
        return {"moves": cross_moves, "last_state": cross_state, "verified": False,
                "cross_complete": cross_complete(cross_state), "f2l_complete": False,
                "cross_moves": cross_moves, "f2l_moves": [], "failure_reason": cross_failure or "cross_failed",
                "search_nodes": cross_nodes, "runtime_seconds": time.perf_counter() - started,
                "pair_results": [], "pair_attempts": []}
    if f2l_complete(cross_state):
        return {"moves": cross_moves, "last_state": cross_state, "verified": True,
                "cross_complete": True, "f2l_complete": True, "cross_moves": cross_moves,
                "f2l_moves": [], "failure_reason": None, "search_nodes": cross_nodes,
                "runtime_seconds": time.perf_counter() - started, "pair_results": [], "pair_attempts": []}
    f2l = solve_hierarchical_f2l(
        cross_state, policy, device, history=history, mode="search",
        width=config["width"], branching=config["branching"], max_nodes=config["max_nodes"],
        max_depth=config["max_depth"], max_moves=config["max_moves"],
        max_actions_per_pair=config["max_actions_per_pair"],
        max_order_attempts=config["max_order_attempts"], move_count=len(cross_moves),
        conditional_legal=True,
    )
    moves = cross_moves + f2l["moves"]
    final = apply_sequence(start, moves)
    verified = f2l["verified"] and f2l_complete(final)
    return {
        "moves": moves, "last_state": final, "verified": verified,
        "cross_complete": f2l["cross_complete"], "f2l_complete": verified,
        "cross_moves": cross_moves, "f2l_moves": f2l["moves"],
        "failure_reason": f2l["failure_reason"] if not verified else None,
        "search_nodes": cross_nodes + f2l["states_explored"],
        "runtime_seconds": time.perf_counter() - started,
        "pair_results": f2l["pair_results"], "pair_attempts": f2l["pair_attempts"],
    }


def _neural_full_row(scramble, start, prefix, result, route):
    moves = prefix["moves"] + result["moves"]
    final = apply_sequence(start, moves)
    verified = bool(prefix["verified"] and result["verified"] and cube_solved(final)
                    and final == result["last_state"])
    cross_state = apply_sequence(start, prefix["cross_moves"])
    f2l_state = apply_sequence(cross_state, prefix["f2l_moves"])
    phase_states = {"CROSS": cross_state, "F2L": f2l_state}
    phase_states.update(dict(result.get("phase_states", [])))
    phase_moves = {"CROSS": prefix["cross_moves"], "F2L": prefix["f2l_moves"]}
    phase_moves.update(result["moves_by_phase"])
    if result["route"].startswith("ZBLL → OLL"):
        ll_order = ("ZBLL", "OLL", "PLL")
    elif result["route"] == "ZBLL":
        ll_order = ("ZBLL",)
    else:
        ll_order = ("OLL", "PLL")
    phase_order = ["CROSS", "F2L", *ll_order]
    counts = _move_counts(moves)
    eligible = zbll_eligible(prefix["last_state"])
    direct_zbll = (eligible and result["verified"] and not result.get("recovery_used")
                   and result["route"] == "ZBLL")
    oll_state = phase_states.get("OLL", prefix["last_state"])
    return {
        "scramble": scramble, "status": "SOLVED" if verified else "FAILED / INCOMPLETE",
        "verified": verified, "moves": moves, "moves_by_phase": phase_moves,
        "phase_order": phase_order,
        "phase_states": [(phase, phase_states[phase]) for phase in phase_order if phase in phase_states],
        "last_state": final, "cross_complete": prefix["cross_complete"],
        "f2l_complete": prefix["f2l_complete"],
        "oll_complete": f2l_complete(oll_state) and oll_complete(oll_state),
        "pll_complete": verified, "zbll_eligible": eligible,
        "direct_zbll_success": direct_zbll,
        "recovery_used": result.get("recovery_used", False), "route": result["route"],
        "stm": counts["stm"], "htm": counts["htm"], "rotations": counts["rotations"],
        "search_nodes": prefix["search_nodes"] + result["states_explored"],
        "runtime_seconds": prefix["runtime_seconds"] + result["runtime_seconds"],
        "failure_reason": result["failure_reason"] or prefix["failure_reason"],
        "provenance": result["provenance"],
    }


def _deterministic_from_prefix(scramble, start, prefix, row):
    reference_moves = [move for phase in ("CROSS", "F2L", "OLL", "PLL")
                       for move in parse_moves(row["phases"][phase])]
    moves = prefix["moves"] + reference_moves
    final = apply_sequence(start, moves)
    verified = bool(prefix["verified"] and row["reference_verified"] and cube_solved(final))
    counts = _move_counts(moves)
    phase_moves = {
        "CROSS": prefix["cross_moves"] + parse_moves(row["phases"]["CROSS"]),
        "F2L": prefix["f2l_moves"] + parse_moves(row["phases"]["F2L"]),
        "OLL": parse_moves(row["phases"]["OLL"]),
        "PLL": parse_moves(row["phases"]["PLL"]),
    }
    phase_states = {}
    current = start
    for phase in ("CROSS", "F2L", "OLL", "PLL"):
        current = apply_sequence(current, phase_moves[phase])
        phase_states[phase] = current
    return {
        "scramble": scramble, "status": "SOLVED" if verified else "FAILED / INCOMPLETE",
        "verified": verified, "moves": moves, "moves_by_phase": phase_moves,
        "phase_order": ["CROSS", "F2L", "OLL", "PLL"],
        "phase_states": list(phase_states.items()),
        "last_state": final, "cross_complete": prefix["cross_complete"],
        "f2l_complete": prefix["f2l_complete"], "oll_complete": verified,
        "pll_complete": verified, "zbll_eligible": zbll_eligible(prefix["last_state"]),
        "direct_zbll_success": False, "recovery_used": False,
        "route": "hierarchical neural Cross/F2L + deterministic CFOP reference from its F2L state",
        "stm": counts["stm"], "htm": counts["htm"], "rotations": counts["rotations"],
        "search_nodes": prefix["search_nodes"],
        "runtime_seconds": prefix["runtime_seconds"] + row["reference_ms"] / 1000,
        "failure_reason": None if verified else "deterministic_suffix_replay_failed",
        "provenance": "neural Cross/F2L + deterministic @moishy/cfop reference; baseline only",
    }


def _scramble_hashes(manifest):
    stored = set(manifest.get("synthetic_cfop_scramble_hashes", []))
    if stored:
        return stored
    counts = manifest.get("synthetic_cfop_scramble_counts", {"train": 1200, "validation": 160, "test": 160})
    seed = int(manifest.get("seed", 73))
    all_scrambles, hashes = set(), set()
    for name, shift in (("train", 0), ("validation", 100000), ("test", 200000)):
        rng = random.Random(seed + shift)
        generated = 0
        while generated < int(counts[name]):
            moves = []
            while len(moves) < 25:
                face = rng.choice("URFDLB")
                if moves and moves[-1][0] == face:
                    continue
                moves.append(face + rng.choice(("", "'", "2")))
            scramble = " ".join(moves)
            if scramble not in all_scrambles:
                all_scrambles.add(scramble)
                hashes.add(hashlib.sha256(scramble.encode("utf-8")).hexdigest())
                generated += 1
    return hashes


def _fresh_evaluation_scrambles(count, seed, manifest):
    excluded_hashes = _scramble_hashes(manifest)
    candidate_seed = seed
    while True:
        rows = _fresh_scrambles(count, candidate_seed)
        collisions = {scramble for scramble in rows
                      if hashlib.sha256(scramble.encode("utf-8")).hexdigest() in excluded_hashes}
        if not collisions:
            return rows, candidate_seed
        candidate_seed += 1


def _case_rows_from_zbll(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    starts, seen = [], set()
    for row in rows:
        key = row["trajectory_id"]
        if key not in seen:
            starts.append(row)
            seen.add(key)
    return starts


def _compare_case_recognition(rows, device):
    path = ARTIFACTS / "zbll-classifier.pt"
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    cases = json.loads((ROOT / "data/algorithms/zbll_cases.json").read_text(encoding="utf-8"))["cases"]
    if checkpoint.get("model_config", {}).get("classes") != len(cases):
        raise ValueError("ZBLL classifier class count does not match the verified case database")
    classifier = ZBLLClassifier(len(cases)).to(device)
    classifier.load_state_dict(checkpoint["model_state"])
    classifier.eval()
    exact_correct = neural_correct = 0
    with torch.inference_mode():
        for row in rows:
            state = tuple(row["state"])
            exact = recognize_zbll_case(state)
            input_state = torch.tensor([state], dtype=torch.long, device=device)
            predicted = int(classifier(input_state)[0].argmax())
            exact_correct += exact == row["case_id"]
            neural_correct += cases[predicted]["case_id"] == row["case_id"]
    return {
        "samples": len(rows), "exact_deterministic_accuracy": exact_correct / len(rows) if rows else None,
        "neural_classifier_accuracy": neural_correct / len(rows) if rows else None,
        "classifier_training_seed": checkpoint.get("seed"),
        "classifier_checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "classifier_identity_overlap": "known case labels; this measures pose recognition, not unseen-case discovery",
        "interpretation": "case recognition does not generate a solving sequence; this classifier is not used by neural solving",
    }


def _phase_starts_from_data(data_dir, split):
    rows = [json.loads(line) for line in
            (Path(data_dir) / ("synthetic_cfop_" + split + ".jsonl")).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    stage_starts = {"OLL": [], "PLL": []}
    seen = set()
    for row in rows:
        phase = row["phase"]
        key = (row["trajectory_id"], phase)
        if phase not in stage_starts or key in seen:
            continue
        state = tuple(row["state"])
        if not f2l_complete(state) or (phase == "PLL" and not oll_complete(state)):
            raise ValueError("Held-out {} stage input violates its CFOP precondition".format(phase))
        stage_starts[phase].append({"trajectory_id": row["trajectory_id"], "state": state,
                                    "history": row["history"], "move_count": row["move_count"]})
        seen.add(key)
    return stage_starts


def evaluate_controlled_stages(model, device, data_dir=DATA_DIR, split="test", search=None):
    config = {**LL_SEARCH, **(search or {})}
    zbll_starts = _case_rows_from_zbll(Path(data_dir) / ("synthetic_zbll_" + split + ".jsonl"))
    stages = _phase_starts_from_data(data_dir, split)
    rows = {"pll": [], "oll_then_pll": [], "zbll": []}
    for phase, starts in (("oll_then_pll", stages["OLL"]), ("pll", stages["PLL"])):
        for index, row in enumerate(starts, 1):
            state = row["state"]
            result = solve_last_layer(state, model, device, mode="search", strategy="oll-pll",
                                      beam_width=config["beam_width"], branching=config["branching"],
                                      max_nodes=config["max_nodes"], budgets=config["budgets"],
                                      history=row["history"], move_count=row["move_count"],
                                      action_score_weight=config["action_score_weight"])
            rows[phase].append({"verified": result["verified"], "failure_reason": result["failure_reason"],
                                "states_explored": result["states_explored"], "runtime_seconds": result["runtime_seconds"],
                                "stm": _move_counts(result["moves"])["stm"],
                                "rotations": _move_counts(result["moves"])["rotations"],
                                "cross_complete": cross_complete(state),
                                "f2l_complete": f2l_complete(result["last_state"]),
                                "zbll_eligible": zbll_eligible(state),
                                "oll_complete": any(name == "OLL" and f2l_complete(value) and oll_complete(value)
                                                    for name, value in result["phase_states"]),
                                "pll_complete": cube_solved(result["last_state"])})
            if index % 20 == 0 or index == len(starts):
                print("evaluated {} {}/{} states".format(phase, index, len(starts)), flush=True)
    for index, row in enumerate(zbll_starts, 1):
        state = tuple(row["state"])
        if not zbll_eligible(state):
            raise ValueError("Held-out ZBLL stage input is not eligible")
        case_id = recognize_zbll_case(state)
        if case_id != row["case_id"]:
            raise ValueError("Exact ZBLL reference identity failed for a held-out state")
        result = solve_last_layer(state, model, device, mode="search", strategy="eo-aware",
                                  beam_width=config["beam_width"], branching=config["branching"],
                                  max_nodes=config["max_nodes"], budgets=config["budgets"],
                                  allow_zbll_recovery=False, history=row["history"],
                                  move_count=row["move_count"],
                                  action_score_weight=config["action_score_weight"])
        rows["zbll"].append({"verified": result["verified"], "case_id": case_id,
                             "failure_reason": result["failure_reason"],
                             "states_explored": result["states_explored"], "runtime_seconds": result["runtime_seconds"],
                             "stm": _move_counts(result["moves"])["stm"],
                             "rotations": _move_counts(result["moves"])["rotations"],
                             "cross_complete": cross_complete(state),
                             "f2l_complete": f2l_complete(result["last_state"]),
                             "oll_complete": None, "pll_complete": None,
                             "zbll_eligible": True, "direct_zbll_success": result["verified"],
                             "exact_reference_recognition": case_id == recognize_zbll_case(state)})
        if index % 20 == 0 or index == len(zbll_starts):
            print("evaluated ZBLL {}/{} states".format(index, len(zbll_starts)), flush=True)
    output = {name: _aggregate(group) for name, group in rows.items()}
    output["exact_zbll_recognition_rate"] = (
        sum(row["exact_reference_recognition"] for row in rows["zbll"]) / len(rows["zbll"])
        if rows["zbll"] else None
    )
    output["sample_kind"] = "held-out simulator-generated stage states"
    output["split"] = split
    output["case_recognition_comparison"] = _compare_case_recognition(zbll_starts, device)
    output["rows"] = rows
    return output


def evaluate_stages_only(checkpoint_path, output_path, split="validation", device_name="auto",
                         beam_width=128, branching=8, max_nodes=20000, action_score_weight=1.0,
                         budgets=None):
    torch.manual_seed(0)
    device = choose_device(device_name)
    torch.set_num_threads(2)
    model, checkpoint = load_last_layer_checkpoint(checkpoint_path, device)
    manifest = json.loads((Path(DATA_DIR) / "manifest.json").read_text(encoding="utf-8"))
    result = evaluate_controlled_stages(
        model, device, DATA_DIR, split,
        {"beam_width": beam_width, "branching": branching, "max_nodes": max_nodes,
         "action_score_weight": action_score_weight,
         "budgets": {**LL_SEARCH["budgets"], **(budgets or {})}},
    )
    result.update({"checkpoint": str(checkpoint_path), "checkpoint_seed": checkpoint.get("seed"),
                   "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
                   "search_config": {"beam_width": beam_width, "branching": branching,
                                     "max_nodes": max_nodes,
                                     "budgets": {**LL_SEARCH["budgets"], **(budgets or {})},
                                     "action_score_weight": action_score_weight},
                   "human_execution_or_recognition_time": "not measured"})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    return result


def benchmark(checkpoint_path=DEFAULT_CHECKPOINT, output_path=DEFAULT_OUTPUT, count=100,
              seed=20261123, device_name="auto", stage_split="test", include_stages=True,
              ll_search=None, f2l_search=None):
    torch.manual_seed(0)
    device = choose_device(device_name)
    torch.set_num_threads(2)
    model, checkpoint = load_last_layer_checkpoint(checkpoint_path, device)
    manifest_path = Path(DATA_DIR) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cross, cross_history, planner, expert_policy, sources = _load_f2l_models(device)
    ll_config = {**LL_SEARCH, **(ll_search or {})}
    f2l_config = {**F2L_SEARCH, **(f2l_search or {})}
    scrambles, used_seed = _fresh_evaluation_scrambles(count, seed, manifest)
    full_references = []
    for offset in range(0, count, 8):
        full_references.extend(teacher_request({"mode": "solve_many", "scrambles": scrambles[offset:offset + 8]}))

    original_prefixes = []
    deterministic_full, neural_oll_pll = [], []
    original_hybrid_by_index = [None] * count
    neural_eo_search, neural_eo_greedy, neural_value_only = [], [], []
    pending_suffixes = []
    for index, (scramble, reference) in enumerate(zip(scrambles, full_references)):
        start = apply_sequence(SOLVED_STATE, parse_moves(scramble))
        reference_phase_moves = {phase: parse_moves(reference["phases"][phase])
                                 for phase in ("CROSS", "F2L", "OLL", "PLL")}
        reference_moves = [move for phase in ("CROSS", "F2L", "OLL", "PLL")
                           for move in reference_phase_moves[phase]]
        reference_final = start
        reference_phase_states = {}
        for phase in ("CROSS", "F2L", "OLL", "PLL"):
            reference_final = apply_sequence(reference_final, reference_phase_moves[phase])
            reference_phase_states[phase] = reference_final
        reference_verified = bool(reference["reference_verified"] and cube_solved(reference_final))
        reference_cross_state = apply_sequence(start, parse_moves(reference["phases"]["CROSS"]))
        reference_f2l_state = apply_sequence(reference_cross_state, parse_moves(reference["phases"]["F2L"]))
        deterministic_full.append({
            "scramble": scramble, "verified": reference_verified,
            "moves": reference_moves, "last_state": reference_final,
            "moves_by_phase": reference_phase_moves,
            "phase_order": ["CROSS", "F2L", "OLL", "PLL"],
            "phase_states": list(reference_phase_states.items()),
            "cross_complete": cross_complete(reference_cross_state), "f2l_complete": f2l_complete(reference_f2l_state),
            "oll_complete": reference_verified, "pll_complete": reference_verified,
            "zbll_eligible": zbll_eligible(reference_f2l_state),
            "direct_zbll_success": False, "recovery_used": False,
            "route": "deterministic CFOP full-cube reference baseline",
            **_move_counts(reference_moves),
            "search_nodes": 0, "runtime_seconds": reference["reference_ms"] / 1000,
            "failure_reason": None if reference_verified else "deterministic_reference_replay_failed",
            "provenance": "deterministic @moishy/cfop reference; baseline only",
        })

        original = _f2l_prefix(scramble, start, cross, cross_history, planner, device, f2l_config)
        original_prefixes.append(original)
        expert = _f2l_prefix(scramble, start, cross, cross_history, expert_policy, device, f2l_config)
        if original["verified"]:
            state_sequence = parse_moves(scramble) + original["moves"]
            pending_suffixes.append(("original", index, " ".join(state_sequence)))

        if expert["verified"]:
            history = [SIMULATOR_MOVE_TOKENS.index(move) for move in expert["moves"]]
            common = {"beam_width": ll_config["beam_width"], "branching": ll_config["branching"],
                      "max_nodes": ll_config["max_nodes"], "budgets": ll_config["budgets"],
                      "action_score_weight": ll_config["action_score_weight"]}
            conventional = solve_last_layer(expert["last_state"], model, device, mode="search",
                                            history=history, strategy="oll-pll", **common)
            neural_oll_pll.append(_neural_full_row(scramble, start, expert, conventional, "oll-pll-search"))
            if zbll_eligible(expert["last_state"]):
                eo = solve_last_layer(expert["last_state"], model, device, mode="search",
                                      history=history, strategy="eo-aware", **common)
            else:
                eo = conventional
            neural_eo_search.append(_neural_full_row(scramble, start, expert, eo, "eo-aware-search"))
            greedy = solve_last_layer(expert["last_state"], model, device, mode="greedy",
                                      history=history, strategy="eo-aware",
                                      budgets=ll_config["budgets"], beam_width=ll_config["beam_width"],
                                      branching=ll_config["branching"], max_nodes=ll_config["max_nodes"])
            neural_eo_greedy.append(_neural_full_row(scramble, start, expert, greedy, "eo-aware-greedy"))
            ablation = solve_last_layer(expert["last_state"], model, device, mode="search",
                                        history=history, strategy="eo-aware", action_guidance=False,
                                        **common)
            neural_value_only.append(_neural_full_row(scramble, start, expert, ablation, "value-only-search"))
        else:
            for output in (neural_oll_pll, neural_eo_search, neural_eo_greedy, neural_value_only):
                output.append({"scramble": scramble, "verified": False, "moves": expert["moves"],
                               "last_state": expert["last_state"], "cross_complete": expert["cross_complete"],
                               "f2l_complete": False, "oll_complete": False, "pll_complete": False,
                               "zbll_eligible": False, "direct_zbll_success": False, "recovery_used": False,
                               "moves_by_phase": {"CROSS": expert["cross_moves"],
                                                  "F2L": expert["f2l_moves"],
                                                  "OLL": [], "PLL": [], "ZBLL": []},
                               "phase_order": ["CROSS", "F2L"],
                               "phase_states": [("CROSS", apply_sequence(start, expert["cross_moves"])),
                                                ("F2L", expert["last_state"])],
                               **_move_counts(expert["moves"]), "search_nodes": expert["search_nodes"],
                               "runtime_seconds": expert["runtime_seconds"],
                               "failure_reason": expert["failure_reason"], "route": "upstream expert-aware F2L failed",
                               "provenance": "neural Cross/F2L partial; no deterministic fallback"})
        if (index + 1) % 10 == 0:
            print("prepared {}/{} neural phase cases".format(index + 1, count), flush=True)

    reference_suffixes = []
    for offset in range(0, len(pending_suffixes), 8):
        batch = pending_suffixes[offset:offset + 8]
        reference_suffixes.extend(teacher_request({"mode": "solve_many", "scrambles": [row[2] for row in batch]}))
    for (kind, index, _prefix), suffix in zip(pending_suffixes, reference_suffixes):
        start = apply_sequence(SOLVED_STATE, parse_moves(scrambles[index]))
        prefix = original_prefixes[index]
        hybrid = _deterministic_from_prefix(scrambles[index], start, prefix, suffix)
        original_hybrid_by_index[index] = hybrid

    for index, prefix in enumerate(original_prefixes):
        if original_hybrid_by_index[index] is not None:
            continue
        moves = prefix["moves"]
        counts = _move_counts(moves)
        original_hybrid_by_index[index] = {
            "scramble": scrambles[index], "status": "FAILED / INCOMPLETE", "verified": False,
            "moves": moves, "moves_by_phase": {"CROSS": prefix["cross_moves"], "F2L": prefix["f2l_moves"]},
            "last_state": prefix["last_state"], "cross_complete": prefix["cross_complete"],
            "f2l_complete": prefix["f2l_complete"], "oll_complete": False, "pll_complete": False,
            "zbll_eligible": False, "direct_zbll_success": False, "recovery_used": False,
            "route": "hierarchical neural Cross/F2L failed before deterministic reference suffix",
            **counts, "search_nodes": prefix["search_nodes"], "runtime_seconds": prefix["runtime_seconds"],
            "failure_reason": prefix["failure_reason"],
            "provenance": "neural Cross/F2L partial; deterministic baseline not run",
        }
    original_hybrid = original_hybrid_by_index

    stage_results = evaluate_controlled_stages(
        model, device, DATA_DIR, stage_split, ll_config,
    ) if include_stages else None
    training_report = json.loads((ARTIFACTS / "last-layer-training.json").read_text(encoding="utf-8"))
    report = {
        "benchmark": "Milestone C/D neural last-layer and expert-aware full CFOP evaluation",
        "sample_kind": "held-out fresh 3x3 scrambles plus isolated simulator-generated phase states",
        "scramble_count": count, "scramble_seed": used_seed,
        "scramble_generator": "25 random outer turns, filtered against reco.nz, prior benchmark, and synthetic training hashes",
        "excluded_synthetic_training_scramble_hashes": len(_scramble_hashes(manifest)),
        "last_layer_checkpoint": str(checkpoint_path),
        "last_layer_checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
        "selected_training_seed": checkpoint.get("seed"), "selected_training_epoch": checkpoint.get("best_epoch"),
        "training_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "training_data": training_report["training_data"],
        "human_solver_held_out_action_agreement": training_report["selected_human_solver_held_out_action_agreement"],
        "search_config": {"last_layer": ll_config, "expert_aware_f2l": f2l_config,
                          "cross": {"width": 256, "depth": 10, "max_nodes": 10000},
                          "value_only_ablation_rng_seed": 0},
        "upstream_checkpoints": {name: str((ARTIFACTS / filename)) for name, filename in {
            "cross": "mlp-state-history.pt", "hierarchical_f2l": "hierarchical-f2l.pt",
            "expert_f2l": "canonical-expert-f2l.pt", "pair_selector": "expert-aware-pair-selector.pt",
        }.items()},
        "device": str(device), "hardware": {"torch": str(torch.__version__),
                                                   "python": platform.python_version(),
                                                   "platform": platform.platform(),
                                                   "processor": platform.processor()},
        "original_hierarchical_f2l_plus_deterministic_ll": _aggregate(original_hybrid),
        "expert_aware_neural_oll_pll_search": _aggregate(neural_oll_pll),
        "expert_aware_neural_eo_aware_search": _aggregate(neural_eo_search),
        "expert_aware_neural_eo_aware_greedy": _aggregate(neural_eo_greedy),
        "expert_aware_neural_value_only_search_ablation": _aggregate(neural_value_only),
        "deterministic_full_cfop_reference": _aggregate(deterministic_full),
        "controlled_last_layer_stage_tests": stage_results,
        "expert_aware_f2l_calibration": sources["calibration"],
        "human_execution_or_recognition_time": "not measured",
        "rows": {"original_hierarchical_f2l_plus_deterministic_ll": original_hybrid,
                 "expert_aware_neural_oll_pll_search": neural_oll_pll,
                 "expert_aware_neural_eo_aware_search": neural_eo_search,
                 "expert_aware_neural_eo_aware_greedy": neural_eo_greedy,
                 "expert_aware_neural_value_only_search_ablation": neural_value_only,
                 "deterministic_full_cfop_reference": deterministic_full},
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stage-split", choices=("validation", "test"), default="test")
    parser.add_argument("--skip-stages", action="store_true")
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--branching", type=int, default=8)
    parser.add_argument("--max-nodes", type=int, default=20000)
    parser.add_argument("--action-score-weight", type=float, default=1.0)
    parser.add_argument("--oll-budget", type=int, default=40)
    parser.add_argument("--pll-budget", type=int, default=40)
    parser.add_argument("--zbll-budget", type=int, default=100)
    parser.add_argument("--stages-only", action="store_true")
    args = parser.parse_args(argv)
    if args.stages_only:
        evaluate_stages_only(args.checkpoint, args.output, args.stage_split, args.device,
                             args.beam_width, args.branching, args.max_nodes, args.action_score_weight,
                             {"OLL": args.oll_budget, "PLL": args.pll_budget, "ZBLL": args.zbll_budget})
        return
    benchmark(args.checkpoint, args.output, args.count, args.seed, args.device, args.stage_split,
              not args.skip_stages,
              {"beam_width": args.beam_width, "branching": args.branching, "max_nodes": args.max_nodes,
               "action_score_weight": args.action_score_weight,
               "budgets": {"OLL": args.oll_budget, "PLL": args.pll_budget, "ZBLL": args.zbll_budget}})


if __name__ == "__main__":
    main()
