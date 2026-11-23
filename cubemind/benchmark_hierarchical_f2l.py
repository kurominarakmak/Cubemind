"""Benchmark flat baselines against hierarchical neural F2L on unseen scrambles."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from cubemind.benchmark_solver import _scrambles
from cubemind.environment.cube import (
    SIMULATOR_MOVE_TOKENS, SOLVED_STATE, apply_sequence, cross_complete,
    f2l_complete, f2l_slot_count, parse_moves,
)
from cubemind.hierarchical_f2l import load_hierarchical_f2l, solve_hierarchical_f2l
from cubemind.neural_solver import (
    MOVE_BUDGETS, _solve_cross_model, _solve_phase, load_expert_f2l_policy,
    load_solver_checkpoint, teacher_request,
)
from cubemind.training.engine import choose_device, model_from_checkpoint

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "hierarchical-f2l-benchmark.json"


def _wilson(successes, total, z=1.96):
    if total == 0:
        return [None, None]
    rate = successes / total
    den = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / den
    margin = z * ((rate * (1 - rate) / total + z * z / (4 * total * total)) ** 0.5) / den
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _summary(rows):
    total = len(rows)
    cross_starts = [row for row in rows if row.get("cross_start_solved")]
    crossed = [row for row in rows if row.get("cross_complete")]
    final_counts = [int(row.get("final_pairs", 0)) for row in rows]
    successes = [row for row in rows if row.get("f2l_complete")]
    pair_moves = [len(pair["moves"]) for row in rows for pair in row.get("pair_results", [])
                  if pair.get("verified")]
    stages = {}
    for initial_pairs in sorted({row.get("initial_solved_pairs") for row in rows
                                 if row.get("initial_solved_pairs") is not None}):
        stage = [row for row in rows if row.get("initial_solved_pairs") == initial_pairs]
        stage_successes = sum(bool(row.get("f2l_complete")) for row in stage)
        stages[str(initial_pairs)] = {
            "remaining_pairs": 4 - initial_pairs, "cases": len(stage),
            "complete_f2l": stage_successes,
            "complete_f2l_rate": stage_successes / len(stage),
        }
    retry_cases = [row for row in rows if row.get("initial_solved_pairs") is not None
                   and row.get("pair_attempts", 0) > 4 - row["initial_solved_pairs"]]
    retry_successes = sum(bool(row.get("f2l_complete")) for row in retry_cases)
    out = {
        "cases": total,
        "cross_success": len(cross_starts),
        "cross_rate": len(cross_starts) / total if total else 0.0,
        "cross_wilson_95": _wilson(len(cross_starts), total),
        "cross_preserved": len(crossed),
        "cross_preservation_rate": len(crossed) / len(cross_starts) if cross_starts else None,
        "final_cross_rate": len(crossed) / total if total else 0.0,
        "final_cross_wilson_95": _wilson(len(crossed), total),
        "final_pair_distribution": {str(count): final_counts.count(count) for count in range(5)},
        "at_least_1_pair": sum(value >= 1 for value in final_counts),
        "at_least_2_pairs": sum(value >= 2 for value in final_counts),
        "at_least_3_pairs": sum(value >= 3 for value in final_counts),
        "all_4_pairs": len(successes),
        "complete_f2l_rate": len(successes) / total if total else 0.0,
        "complete_f2l_wilson_95": _wilson(len(successes), total),
        "complete_f2l_given_cross_rate": len(successes) / len(cross_starts) if cross_starts else None,
        "complete_f2l_given_cross_wilson_95": _wilson(len(successes), len(cross_starts)),
        "mean_final_pairs": statistics.mean(final_counts) if final_counts else 0,
        "max_solved_pairs_mean": statistics.mean(row.get("max_pairs", 0) for row in rows) if rows else 0,
        "pair_attempts_mean": statistics.mean(row.get("pair_attempts", 0) for row in rows) if rows else 0,
        "pair_destruction_events": sum(row.get("pair_destruction_events", 0) for row in rows),
        "verified_pair_insertions": len(pair_moves),
        "pair_moves_mean": statistics.mean(pair_moves) if pair_moves else None,
        "pair_moves_median": statistics.median(pair_moves) if pair_moves else None,
        "recovery_attempt_cases": len(retry_cases),
        "recovery_successes": retry_successes,
        "recovery_success_rate": retry_successes / len(retry_cases) if retry_cases else None,
        "initial_solved_pair_stages": stages,
        "search_nodes_mean": statistics.mean(row.get("nodes", 0) for row in rows) if rows else 0,
        "runtime_mean_ms": statistics.mean(row.get("runtime_seconds", 0) for row in rows) * 1000 if rows else 0,
        "runtime_median_ms": statistics.median(row.get("runtime_seconds", 0) for row in rows) * 1000 if rows else 0,
        "success_moves_mean": statistics.mean(row["stm"] for row in successes) if successes else None,
        "success_moves_median": statistics.median(row["stm"] for row in successes) if successes else None,
        "failure_causes": {},
    }
    for row in rows:
        if not row.get("f2l_complete"):
            cause = row.get("failure_reason") or "incomplete"
            out["failure_causes"][cause] = out["failure_causes"].get(cause, 0) + 1
    return out


def _unseen_scrambles(count, seed):
    excluded = []
    for path in (ROOT / "data/processed/neural_solver/cfop_teacher.jsonl",
                 ROOT / "data/processed/neural_solver/dagger_teacher.jsonl",
                 ROOT / "data/raw/reco_nz/cfop_3x3.jsonl",
                 ROOT / "data/processed/hierarchical_f2l/on_policy_recovery.jsonl"):
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    scramble = row.get("scramble") or row.get("on_policy_scramble")
                    if scramble:
                        excluded.append(scramble)
    return _scrambles(count, seed, excluded)


def _flat_result(state, history, model, device, f2l_model=None):
    started = time.perf_counter()
    if f2l_model is None:
        f2l_model = model
    final, _history, moves, nodes, failure = _solve_phase(
        state, history, "F2L", f2l_model, device, "search", MOVE_BUDGETS["F2L"], 4, 3, 500, {tuple(state)},
    )
    return {
        "moves": moves, "last_state": final, "cross_complete": cross_complete(final),
        "f2l_complete": f2l_complete(final), "final_pairs": f2l_slot_count(final),
        "max_pairs": max(f2l_slot_count(apply_sequence(state, moves[:index])) for index in range(len(moves) + 1)),
        "nodes": nodes, "runtime_seconds": time.perf_counter() - started,
        "failure_reason": failure,
    }


def benchmark(count=100, seed=20261127, output_path=DEFAULT_OUTPUT, device_name="auto",
              width=16, branching=6, max_nodes=3000, max_depth=14):
    if count < 1:
        raise ValueError("Benchmark count must be positive")
    device = choose_device(device_name)
    torch.set_num_threads(2)
    cross_model, cross_ckpt = model_from_checkpoint(ROOT / "artifacts" / "mlp-state-history.pt", device)
    cross_history_length = int(cross_ckpt["model_config"].get("max_history", 8))
    synthetic, synthetic_checkpoint = load_solver_checkpoint(ROOT / "artifacts" / "neural-cfop-policy.pt", device)
    expert_f2l, expert_checkpoint = load_expert_f2l_policy(
        ROOT / "artifacts" / "reco-cfop-context.pt", synthetic, device,
    )
    hierarchical, hierarchical_checkpoint = load_hierarchical_f2l(
        ROOT / "artifacts" / "hierarchical-f2l.pt", device,
    )
    scrambles = _unseen_scrambles(count, seed)
    references = []
    for offset in range(0, len(scrambles), 8):
        references.extend(teacher_request({"mode": "solve_many", "scrambles": scrambles[offset:offset + 8]}))
        print("reference solves: {}/{}".format(len(references), count), flush=True)
    if len(references) != count:
        raise ValueError("Reference teacher returned the wrong case count")

    modes = {name: [] for name in (
        "existing_synthetic_flat", "existing_reco_f2l_hybrid_flat",
        "hierarchical_greedy", "hierarchical_neural_search", "hierarchical_uniform_policy_ablation",
        "deterministic_reference",
    )}
    rows = []
    for index, (scramble, reference) in enumerate(zip(scrambles, references), 1):
        start = apply_sequence(SOLVED_STATE, parse_moves(scramble))
        cross_started = time.perf_counter()
        if cross_complete(start):
            cross_moves = []
            cross_state, cross_history, cross_nodes, cross_failure = start, [], 0, None
        else:
            cross_state, cross_history, cross_moves, cross_nodes, cross_failure = _solve_cross_model(
                start, [], cross_model, cross_history_length, device, "search", 256, 10, 10000, {start},
            )
        cross_runtime = time.perf_counter() - cross_started
        cross_ok = cross_failure is None and cross_complete(cross_state)
        case = {"scramble": scramble, "cross_moves": cross_moves, "cross_success": cross_ok,
                "cross_nodes": cross_nodes, "cross_runtime_seconds": cross_runtime,
                "initial_solved_pairs": f2l_slot_count(cross_state) if cross_ok else None,
                "modes": {}}
        if cross_ok:
            base_history = list(cross_history)
            for mode, policy in (("existing_synthetic_flat", synthetic),
                                 ("existing_reco_f2l_hybrid_flat", expert_f2l)):
                result = _flat_result(cross_state, base_history, synthetic, device,
                                      f2l_model=policy if mode.endswith("hybrid_flat") else None)
                result["moves"] = cross_moves + result["moves"]
                result["stm"] = sum(move not in ("x", "y", "z") for move in result["moves"])
                result["runtime_seconds"] += cross_runtime
                result["nodes"] += cross_nodes
                result["cross_start_solved"] = cross_ok
                result["cross_complete"] = cross_complete(result["last_state"])
                result["initial_solved_pairs"] = f2l_slot_count(cross_state) if cross_ok else None
                case["modes"][mode] = result
                modes[mode].append(result)
            for mode, kind, guidance in (
                ("hierarchical_greedy", "greedy", "neural"),
                ("hierarchical_neural_search", "search", "neural"),
                ("hierarchical_uniform_policy_ablation", "search", "uniform"),
            ):
                result = solve_hierarchical_f2l(
                    cross_state, hierarchical, device, history=base_history, mode=kind,
                    width=width, branching=branching, max_nodes=max_nodes,
                    max_depth=max_depth, max_moves=100, max_actions_per_pair=max_depth,
                    max_order_attempts=24, guidance=guidance, seed=seed + index,
                    move_count=len(cross_moves),
                )
                result["moves"] = cross_moves + result["moves"]
                result["stm"] = sum(move not in ("x", "y", "z") for move in result["moves"])
                result["runtime_seconds"] += cross_runtime
                result["cross_start_solved"] = cross_ok
                result["cross_complete"] = cross_complete(result["last_state"])
                result["initial_solved_pairs"] = f2l_slot_count(cross_state)
                result["final_pairs"] = f2l_slot_count(result["last_state"])
                result["max_pairs"] = result.get("max_solved_pairs", result["final_pairs"])
                result["pair_attempts"] = len(result.get("pair_attempts", []))
                result["nodes"] = result["states_explored"] + cross_nodes
                case["modes"][mode] = result
                modes[mode].append(result)
        else:
            for mode in ("existing_synthetic_flat", "existing_reco_f2l_hybrid_flat",
                         "hierarchical_greedy", "hierarchical_neural_search",
                         "hierarchical_uniform_policy_ablation"):
                result = {"cross_complete": False, "f2l_complete": False, "final_pairs": 0,
                          "cross_start_solved": False, "initial_solved_pairs": None,
                          "max_pairs": 0, "nodes": cross_nodes, "runtime_seconds": cross_runtime,
                          "failure_reason": cross_failure, "moves": cross_moves,
                          "stm": sum(move not in ("x", "y", "z") for move in cross_moves)}
                case["modes"][mode] = result
                modes[mode].append(result)

        reference_moves = (parse_moves(reference["phases"]["CROSS"])
                           + parse_moves(reference["phases"]["F2L"]))
        reference_cross_state = apply_sequence(start, parse_moves(reference["phases"]["CROSS"]))
        reference_f2l_state = apply_sequence(reference_cross_state, parse_moves(reference["phases"]["F2L"]))
        reference_result = {
            "moves": reference_moves,
            "verified": bool(reference["reference_verified"] and f2l_complete(reference_f2l_state)),
            "cross_start_solved": cross_complete(reference_cross_state),
            "initial_solved_pairs": f2l_slot_count(reference_cross_state),
            "cross_complete": cross_complete(reference_f2l_state),
            "f2l_complete": f2l_complete(reference_f2l_state),
            "final_pairs": f2l_slot_count(reference_f2l_state),
            "max_pairs": f2l_slot_count(reference_f2l_state), "nodes": 0,
            "stm": sum(move not in ("x", "y", "z") for move in reference_moves),
            "last_state": reference_f2l_state,
            "runtime_seconds": reference["reference_ms"] / 1000.0,
            "failure_reason": None,
        }
        case["modes"]["deterministic_reference"] = reference_result
        modes["deterministic_reference"].append(reference_result)
        rows.append(case)
        if index % 5 == 0 or index == count:
            print("evaluated {}/{} scrambles".format(index, count), flush=True)

    summaries = {name: _summary(results) for name, results in modes.items()}
    training_path = ROOT / "data/processed/hierarchical_f2l/train.jsonl"
    output = {
        "benchmark": "unseen 3x3 Cross + all-pair F2L evaluation",
        "seed": seed, "scramble_count": count,
        "scramble_generator": "25 random outer turns; seed-separated from synthetic training split",
        "training_data_sha256": hashlib.sha256(training_path.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256((ROOT / "artifacts" / "hierarchical-f2l.pt").read_bytes()).hexdigest(),
        "device": str(device),
        "cross_model": "artifacts/mlp-state-history.pt",
        "existing_solver_checkpoint": "artifacts/neural-cfop-policy.pt",
        "expert_f2l_checkpoint": "artifacts/reco-cfop-context.pt",
        "hierarchical_checkpoint": "artifacts/hierarchical-f2l.pt",
        "training_metrics": "artifacts/hierarchical-f2l-training.json",
        "hierarchical_data_sources": hierarchical_checkpoint.get("data_sources"),
        "reference": "@moishy/cfop 0.4.0; comparison only, never inference fallback",
        "matched_hierarchical_search_budget": {"width": width, "branching": branching,
                                                "max_nodes": max_nodes, "max_depth": max_depth,
                                                "max_order_attempts": 24, "value_weight": 0.5},
        "flat_baseline_budget": {"width": 4, "branching": 3, "max_nodes": 500,
                                 "F2L_max_moves": MOVE_BUDGETS["F2L"]},
        "move_count": "STM; x/y/z whole cube turns excluded",
        "human_execution_or_recognition_time": "not measured",
        "summaries": summaries,
        "rows": rows,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261127)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--branching", type=int, default=6)
    parser.add_argument("--max-nodes", type=int, default=3000)
    parser.add_argument("--max-depth", type=int, default=14)
    args = parser.parse_args(argv)
    benchmark(args.count, args.seed, args.output, args.device, args.width,
              args.branching, args.max_nodes, args.max_depth)


if __name__ == "__main__":
    main()
