"""Compare expert-aware and hierarchical F2L policies on fresh 3x3 scrambles."""

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from cubemind.benchmark_hierarchical_f2l import _wilson
from cubemind.benchmark_solver import _scrambles
from cubemind.environment.cube import (
    SIMULATOR_MOVE_TOKENS, SOLVED_STATE, apply_sequence, cross_complete,
    f2l_complete, f2l_slot_count, parse_moves,
)
from cubemind.expert_aware_f2l import (
    ExpertAwareF2LPolicy, load_canonical_expert_action_model, load_expert_action_model,
    load_expert_pair_selector,
)
from cubemind.hierarchical_f2l import load_hierarchical_f2l, solve_hierarchical_f2l
from cubemind.neural_solver import _solve_cross_model, teacher_request
from cubemind.training.engine import choose_device, model_from_checkpoint

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "expert-aware-f2l-benchmark.json"
SEARCH = {"width": 128, "branching": 6, "max_nodes": 6000, "max_depth": 22,
          "max_moves": 100, "max_actions_per_pair": 22, "max_order_attempts": 24}


def _scramble_exclusions():
    excluded = set()
    for path in (ROOT / "data/processed/neural_solver/cfop_teacher.jsonl",
                 ROOT / "data/processed/neural_solver/dagger_teacher.jsonl",
                 ROOT / "data/raw/reco_nz/cfop_3x3.jsonl",
                 ROOT / "data/processed/hierarchical_f2l/on_policy_recovery.jsonl"):
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    value = row.get("scramble") or row.get("on_policy_scramble")
                    if value:
                        excluded.add(" ".join(parse_moves(value)))
    for path in (ROOT / "artifacts/hierarchical-f2l-benchmark.json",
                 ROOT / "artifacts/expert-aware-baseline-reproduction.json",
                 ROOT / "artifacts/neural-cfop-benchmark-hybrid-search.json"):
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            rows = value.get("rows", [])
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict) and row.get("scramble"):
                        excluded.add(" ".join(parse_moves(row["scramble"])))
            elif isinstance(rows, dict):
                for group in rows.values():
                    for row in group if isinstance(group, list) else []:
                        if isinstance(row, dict) and row.get("scramble"):
                            excluded.add(" ".join(parse_moves(row["scramble"])))
            for scramble in value.get("scrambles", []):
                if isinstance(scramble, str):
                    excluded.add(" ".join(parse_moves(scramble)))
    return excluded


def _fresh_scrambles(count, seed):
    return _scrambles(count, seed, _scramble_exclusions())


def _metrics(rows):
    successes = [row for row in rows if row["f2l_complete"]]
    moves = [row["stm"] for row in successes]
    pair_lengths = [len(pair["moves"]) for row in successes for pair in row.get("pair_results", [])]
    pair_orders = {}
    for row in successes:
        order = " → ".join(row.get("pair_order", []))
        pair_orders[order] = pair_orders.get(order, 0) + 1
    totals = len(rows)
    return {
        "cases": totals, "cross_success": sum(row["cross_start_solved"] for row in rows),
        "cross_preserved": sum(row["cross_complete"] for row in rows),
        "complete_f2l": len(successes), "complete_f2l_rate": len(successes) / totals,
        "complete_f2l_wilson_95": _wilson(len(successes), totals),
        "success_stm_mean": statistics.mean(moves) if moves else None,
        "success_stm_median": statistics.median(moves) if moves else None,
        "success_htm_mean": statistics.mean(row["htm"] for row in successes) if successes else None,
        "verified_pair_insertions_mean": statistics.mean(len(row.get("pair_results", [])) for row in successes) if successes else None,
        "pair_insertion_moves_mean": statistics.mean(pair_lengths) if pair_lengths else None,
        "pair_order_distribution": pair_orders,
        "rotation_count_mean": statistics.mean(row["rotations"] for row in rows),
        "adjacent_same_face_pairs_mean": statistics.mean(row["adjacent_same_face_pairs"] for row in rows),
        "search_nodes_mean": statistics.mean(row["nodes"] for row in rows),
        "runtime_mean_ms": statistics.mean(row["runtime_seconds"] for row in rows) * 1000,
        "failure_causes": {reason: sum(row.get("failure_reason") == reason for row in rows)
                           for reason in sorted({row.get("failure_reason") for row in rows if row.get("failure_reason")})},
    }


def _counts(moves):
    rotations = sum(move[0] in "xyz" for move in moves)
    same_face = sum(left.rstrip("'2") == right.rstrip("'2") for left, right in zip(moves, moves[1:]))
    return {
        "stm": sum(move[0] not in "xyz" for move in moves),
        "htm": sum(move[0] not in "xyz" for move in moves),
        "rotations": rotations,
        "adjacent_same_face_pairs": same_face,
    }


def benchmark(count=100, seed=20261129, output_path=DEFAULT_OUTPUT, device_name="auto",
              modes=None, width=128, branching=6, max_nodes=6000, max_depth=22):
    if count < 1:
        raise ValueError("Benchmark count must be positive")
    if modes is None:
        modes = ("planner_original", "planner_legal", "legacy_expert_action_only", "expert_action_only", "fused",
                 "expert_pair_order", "combined", "uniform_action", "execution_cost")
    valid_modes = {"planner_original", "planner_legal", "legacy_expert_action_only", "expert_action_only", "fused",
                   "expert_pair_order", "combined", "uniform_action", "execution_cost"}
    if not modes or any(mode not in valid_modes for mode in modes):
        raise ValueError("Unknown expert-aware benchmark mode")
    device = choose_device(device_name)
    torch.set_num_threads(2)
    cross_model, cross_checkpoint = model_from_checkpoint(ROOT / "artifacts/mlp-state-history.pt", device)
    cross_history_length = int(cross_checkpoint["model_config"].get("max_history", 8))
    planner, planner_checkpoint = load_hierarchical_f2l(ROOT / "artifacts/hierarchical-f2l.pt", device)
    legacy_expert, legacy_checkpoint = load_expert_action_model(ROOT / "artifacts/reco-cfop-context.pt", device)
    expert, expert_checkpoint = load_canonical_expert_action_model(ROOT / "artifacts/canonical-expert-f2l.pt", device)
    selector, selector_checkpoint = load_expert_pair_selector(ROOT / "artifacts/expert-aware-pair-selector.pt", device)
    calibration = json.loads((ROOT / "artifacts/expert-aware-f2l-evaluation.json").read_text(encoding="utf-8"))
    tuned = calibration["action_fusion"]["calibration"]
    expert_temperature, planner_temperature = tuned["expert_temperature"], tuned["planner_temperature"]
    expert_weight = tuned["expert_weight"]
    policy = {
        "planner_legal": planner,
        "legacy_expert_action_only": ExpertAwareF2LPolicy(
            planner, legacy_expert, 1.0, 0.0, 2.0, planner_temperature),
        "expert_action_only": ExpertAwareF2LPolicy(
            planner, expert, 1.0, 0.0, expert_temperature, planner_temperature,
            canonical_expert=True),
        "fused": ExpertAwareF2LPolicy(
            planner, expert, expert_weight, 1.0 - expert_weight,
            expert_temperature, planner_temperature, canonical_expert=True),
        "expert_pair_order": ExpertAwareF2LPolicy(
            planner, expert, 0.0, 1.0, expert_temperature, planner_temperature, selector,
            canonical_expert=True),
        "combined": ExpertAwareF2LPolicy(
            planner, expert, expert_weight, 1.0 - expert_weight,
            expert_temperature, planner_temperature, selector, canonical_expert=True),
        "uniform_action": planner,
        "execution_cost": planner,
        "planner_original": planner,
    }
    scrambles = _fresh_scrambles(count, seed)
    references = []
    for offset in range(0, count, 8):
        references.extend(teacher_request({"mode": "solve_many", "scrambles": scrambles[offset:offset + 8]}))
    modes_rows = {name: [] for name in modes}
    cases = []
    for case_index, (scramble, reference) in enumerate(zip(scrambles, references), 1):
        start = apply_sequence(SOLVED_STATE, parse_moves(scramble))
        cross_started = time.perf_counter()
        cross_state, cross_history, cross_moves, cross_nodes, cross_failure = _solve_cross_model(
            start, [], cross_model, cross_history_length, device, "search", 256, 10, 10000, {start},
        )
        cross_runtime = time.perf_counter() - cross_started
        cross_ok = cross_failure is None and cross_complete(cross_state)
        case = {"scramble": scramble, "cross_moves": cross_moves, "cross_success": cross_ok,
                "cross_nodes": cross_nodes, "modes": {}}
        for name in modes:
            mode_started = time.perf_counter()
            if cross_ok:
                result = solve_hierarchical_f2l(
                    cross_state, policy[name], device, history=cross_history, mode="search",
                    width=width, branching=branching, max_nodes=max_nodes, max_depth=max_depth,
                    max_moves=100, max_actions_per_pair=max_depth, max_order_attempts=24,
                    guidance="uniform" if name == "uniform_action" else "neural",
                    seed=seed + case_index, move_count=len(cross_moves),
                    conditional_legal=name not in ("planner_original",),
                    execution_cost_weight=1.0 if name == "execution_cost" else 0.0,
                )
                moves = list(cross_moves) + list(result["moves"])
                replay = apply_sequence(start, moves)
                verified = (result["verified"] and cross_complete(replay)
                            and f2l_complete(replay) and f2l_complete(result["last_state"]))
                pair_order = result.get("pair_order", [])
                result_row = {
                    "cross_start_solved": True, "cross_complete": cross_complete(replay),
                    "f2l_complete": verified, "verified": verified,
                    "initial_solved_pairs": f2l_slot_count(cross_state),
                    "final_pairs": f2l_slot_count(replay), "max_pairs": result.get("max_solved_pairs", 0),
                    "moves": moves, "pair_order": pair_order,
                    "pair_results": result.get("pair_results", []),
                    "pair_attempts": len(result.get("pair_attempts", [])),
                    "nodes": cross_nodes + result["states_explored"],
                    "runtime_seconds": cross_runtime + (time.perf_counter() - mode_started),
                    "failure_reason": result.get("failure_reason") if not verified else None,
                    "route": result["route"], **_counts(moves),
                }
            else:
                result_row = {
                    "cross_start_solved": False, "cross_complete": cross_complete(cross_state),
                    "f2l_complete": False, "verified": False, "initial_solved_pairs": None,
                    "final_pairs": f2l_slot_count(cross_state), "max_pairs": f2l_slot_count(cross_state),
                    "moves": list(cross_moves), "pair_order": [], "pair_results": [],
                    "pair_attempts": 0, "nodes": cross_nodes,
                    "runtime_seconds": cross_runtime, "failure_reason": cross_failure,
                    "route": "neural Cross failed; no deterministic fallback", **_counts(cross_moves),
                }
            result_row["human_execution_time"] = "not measured"
            case["modes"][name] = result_row
            modes_rows[name].append(result_row)
        reference_moves = parse_moves(reference["phases"]["CROSS"]) + parse_moves(reference["phases"]["F2L"])
        reference_state = apply_sequence(start, reference_moves)
        case["reference"] = {
            "cross_complete": cross_complete(reference_state), "f2l_complete": f2l_complete(reference_state),
            "verified": bool(reference["reference_verified"] and f2l_complete(reference_state)),
            "stm": _counts(reference_moves)["stm"],
            "runtime_ms": reference.get("reference_ms"),
        }
        cases.append(case)
        if case_index % 5 == 0 or case_index == count:
            print("evaluated {}/{} scrambles".format(case_index, count), flush=True)
    reference_rows = [case["reference"] for case in cases]
    reference_moves = [row["stm"] for row in reference_rows if row["verified"]]
    output = {
        "benchmark": "fresh held-out expert-aware Cross + all-pair F2L",
        "seed": seed, "scramble_count": count,
        "scramble_generator": "25 random outer turns; excludes training/reconstruction and previous reported evaluation scrambles",
        "scramble_set_sha256": hashlib.sha256("\n".join(scrambles).encode()).hexdigest(),
        "device": str(device), "python": platform.python_version(), "torch": torch.__version__,
        "models": {"cross": "artifacts/mlp-state-history.pt", "planner": "artifacts/hierarchical-f2l.pt",
                   "legacy_expert_action": "artifacts/reco-cfop-context.pt",
                   "expert_action": "artifacts/canonical-expert-f2l.pt",
                   "expert_pair_selector": "artifacts/expert-aware-pair-selector.pt"},
        "checkpoint_sha256": {
            "planner": hashlib.sha256((ROOT / "artifacts/hierarchical-f2l.pt").read_bytes()).hexdigest(),
            "legacy_expert_action": hashlib.sha256((ROOT / "artifacts/reco-cfop-context.pt").read_bytes()).hexdigest(),
            "expert_action": hashlib.sha256((ROOT / "artifacts/canonical-expert-f2l.pt").read_bytes()).hexdigest(),
            "expert_pair_selector": hashlib.sha256((ROOT / "artifacts/expert-aware-pair-selector.pt").read_bytes()).hexdigest(),
        },
        "action_fusion": {"expert_weight": expert_weight, "planner_weight": 1.0 - expert_weight,
                          "expert_temperature": expert_temperature, "planner_temperature": planner_temperature,
                          "tuned_on": "solver-held-out validation actions only"},
        "matched_search_budget": {"width": width, "branching": branching, "max_nodes": max_nodes,
                                  "max_depth": max_depth, "max_order_attempts": 24},
        "mode_definitions": {
            "planner_original": "Historical hierarchical policy and original 72-action normalization.",
            "planner_legal": "Planner action distribution renormalized over legal outer turns.",
            "legacy_expert_action_only": "Original reco.nz action model through a canonical feature adapter; planner value and pair selector.",
            "expert_action_only": "New human-only canonical action policy; planner value and pair selector.",
            "fused": "Validation-calibrated log-probability fusion; planner value and pair selector.",
            "expert_pair_order": "Planner action policy and expert-only human pair-order selector.",
            "combined": "Validation-calibrated action fusion plus expert-only pair-order selector.",
            "uniform_action": "Uniform legal action branch ordering, other planner heads retained.",
            "execution_cost": "Planner plus exhaustive pair-order selection by STM; move-count proxy only, not human speed.",
        },
        "metrics": {name: _metrics(rows) for name, rows in modes_rows.items()},
        "reference_metrics": {
            "solved": sum(row["verified"] for row in reference_rows),
            "mean_stm": statistics.mean(reference_moves) if reference_moves else None,
            "median_stm": statistics.median(reference_moves) if reference_moves else None,
            "mean_runtime_ms": statistics.mean(row["runtime_ms"] for row in reference_rows),
        },
        "execution_metrics": "STM/HTM and rotation counts are simulator move counts; TPS, recognition, pauses, fingertricks and human ratings not measured",
        "deterministic_reference": "comparison only; never called from neural inference",
        "human_recons_test_split_sha256": hashlib.sha256((ROOT / "data/processed/reco_cfop/test.jsonl").read_bytes()).hexdigest(),
        "rows": cases,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["metrics"], indent=2))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261129)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--modes", default="planner_original,planner_legal,legacy_expert_action_only,expert_action_only,fused,expert_pair_order,combined,uniform_action,execution_cost")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--branching", type=int, default=6)
    parser.add_argument("--max-nodes", type=int, default=6000)
    parser.add_argument("--max-depth", type=int, default=22)
    args = parser.parse_args(argv)
    benchmark(args.count, args.seed, args.output, args.device,
              tuple(value.strip() for value in args.modes.split(",") if value.strip()),
              args.width, args.branching, args.max_nodes, args.max_depth)


if __name__ == "__main__":
    main()
