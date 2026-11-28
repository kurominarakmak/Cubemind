"""Paired fresh-scramble benchmark for learned versus retrieved last-layer moves."""

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

from cubemind.benchmark_expert_aware_f2l import _fresh_scrambles
from cubemind.environment.cube import COLORS, SOLVED_STATE, apply_sequence, parse_moves
from cubemind.models.expert_cross import load_expert_cross_policy
from cubemind.models.algorithm_preference import load_algorithm_preference_model
from cubemind.hierarchical_f2l import load_hierarchical_f2l
from cubemind.neural_solver import cube_solved, load_last_layer_checkpoint, reference_solve
from cubemind.training.engine import choose_device
from cubemind.v2_planner import solve_v2
from cubemind.zbls import load_zbls_checkpoint


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/v2/neurosymbolic-100-benchmark.json"
PREFERENCE_OUTPUT = ROOT / "artifacts/v2/neurosymbolic-preference-100-benchmark.json"


def _wilson(successes, total):
    if not total:
        return [None, None]
    z, p = 1.96, successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _mcnemar_p(left_only, right_only):
    discordant = left_only + right_only
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(left_only, right_only) + 1))
    return min(1.0, 2 * tail / (2 ** discordant))


def _paired_comparison(rows):
    result = {}
    left = [row["neural"] for row in rows]
    right = [row["symbolic"] for row in rows]
    left_ok = [row["verified"] for row in left]
    right_ok = [row["verified"] for row in right]
    common = [(left[i]["stm"], right[i]["stm"]) for i in range(len(rows))
              if left_ok[i] and right_ok[i]]
    neural_only = sum(a and not b for a, b in zip(left_ok, right_ok))
    symbolic_only = sum(b and not a for a, b in zip(left_ok, right_ok))
    result["selected_full_solve"] = {
        "neural_only_success": neural_only,
        "symbolic_only_success": symbolic_only,
        "exact_mcnemar_two_sided_p": _mcnemar_p(neural_only, symbolic_only),
        "both_success": sum(a and b for a, b in zip(left_ok, right_ok)),
        "neither_success": sum(not a and not b for a, b in zip(left_ok, right_ok)),
        "common_success_count": len(common),
        "mean_neural_minus_symbolic_stm_on_common_success": (
            statistics.mean(a - b for a, b in common) if common else None),
    }
    route_pairs = (("cfop_oll_pll", "cfop_oll_pll", "symbolic_cfop"),
                   ("cfop_eo_aware", "cfop_eo_aware", "symbolic_eo_aware"),
                   ("zbls_global", "neural_zbls_global", "algorithm_zbls_global"))
    for route, neural_name, symbolic_name in route_pairs:
        left = [row["neural"]["routes"].get(neural_name) for row in rows]
        right = [row["symbolic"]["routes"].get(symbolic_name) for row in rows]
        left_ok = [bool(x and x["verified"]) for x in left]
        right_ok = [bool(x and x["verified"]) for x in right]
        common = [(left[i]["stm"], right[i]["stm"]) for i in range(len(rows))
                  if left_ok[i] and right_ok[i]]
        neural_only = sum(a and not b for a, b in zip(left_ok, right_ok))
        symbolic_only = sum(b and not a for a, b in zip(left_ok, right_ok))
        result[route] = {
            "neural_only_success": neural_only,
            "symbolic_only_success": symbolic_only,
            "exact_mcnemar_two_sided_p": _mcnemar_p(neural_only, symbolic_only),
            "both_success": sum(a and b for a, b in zip(left_ok, right_ok)),
            "neither_success": sum(not a and not b for a, b in zip(left_ok, right_ok)),
            "common_success_count": len(common),
            "mean_neural_minus_symbolic_stm_on_common_success": (
                statistics.mean(a - b for a, b in common) if common else None),
        }
    return result


def _summary(rows, key):
    outcomes = [row[key]["verified"] for row in rows]
    successful = [row[key]["stm"] for row in rows if row[key]["verified"]]
    route_names = sorted({name for row in rows for name in row[key]["routes"]})
    return {
        "cases": len(rows), "successes": sum(outcomes),
        "success_rate": sum(outcomes) / len(rows) if rows else None,
        "success_rate_wilson_95": _wilson(sum(outcomes), len(rows)),
        "successful_stm_mean": statistics.mean(successful) if successful else None,
        "successful_stm_median": statistics.median(successful) if successful else None,
        "successful_stm_p95": sorted(successful)[math.ceil(.95 * len(successful)) - 1]
        if successful else None,
        "mean_runtime_ms": statistics.mean(row[key]["runtime_seconds"] for row in rows) * 1000
        if rows else None,
        "mean_search_nodes": statistics.mean(row[key]["search_nodes"] for row in rows) if rows else None,
        "cross_plus_one_distribution": {
            str(n): sum(row[key]["cross_plus_one_pairs"] == n for row in rows) for n in range(5)},
        "routes": {
            name: {"successes": sum(bool(row[key]["routes"].get(name, {}).get("verified")) for row in rows),
                   "eligible": sum(bool(row[key]["routes"].get(name, {}).get("zbll_eligible")) for row in rows)}
            for name in route_names
        },
        "zbls_replay_verified": sum(bool((row[key].get("zbls_attempt") or {}).get("verified"))
                                    for row in rows),
    }


def benchmark(count=100, seed=20261129, device_name="cpu", output_path=OUTPUT):
    if count < 1:
        raise ValueError("count must be positive")
    device = choose_device(device_name)
    cross, cross_checkpoint = load_expert_cross_policy(ROOT / "artifacts/v2/expert-cross-policy.pt", device)
    f2l, f2l_checkpoint = load_hierarchical_f2l(ROOT / "artifacts/hierarchical-f2l.pt", device)
    last_layer, last_checkpoint = load_last_layer_checkpoint(ROOT / "artifacts/last-layer-policy.pt", device)
    zbls, zbls_checkpoint = load_zbls_checkpoint(ROOT / "artifacts/v2/zbls-synthetic-policy.pt", device)
    checkpoint_paths = {
        "expert_cross": ROOT / "artifacts/v2/expert-cross-policy.pt",
        "hierarchical_f2l": ROOT / "artifacts/hierarchical-f2l.pt",
        "last_layer": ROOT / "artifacts/last-layer-policy.pt",
        "neural_zbls": ROOT / "artifacts/v2/zbls-synthetic-policy.pt",
    }
    report = {
        "task": "paired fresh-scramble comparison using the same human Cross and F2L policies",
        "seed": seed, "requested_scrambles": count, "cross_color": "white", "device": str(device),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
        "checkpoint_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest()
                               for key, path in checkpoint_paths.items()},
        "checkpoint_epochs": {"cross": cross_checkpoint["best_epoch"],
                              "f2l": f2l_checkpoint.get("best_epoch"),
                              "last_layer": last_checkpoint.get("best_epoch"),
                              "zbls": zbls_checkpoint.get("best_epoch")},
        "search_budgets": {
            "f2l": {"width": 128, "branching": 6, "max_nodes": 6000, "max_depth": 22},
            "last_layer": {"beam_width": 128, "branching": 8, "max_nodes": 20000,
                           "budgets": {"OLL": 40, "PLL": 40, "ZBLL": 100}},
            "neural_zbls": {"width": 48, "branching": 8, "max_depth": 30,
                            "max_nodes": 5000, "max_moves": 35},
            "symbolic_zbls": {"source_cases": 302, "verified_candidates_only": True},
        },
        "rows": [], "complete": False,
    }
    f2l_options = {"width": 128, "branching": 6, "max_nodes": 6000,
                   "max_depth": 22, "max_moves": 100,
                   "max_actions_per_pair": 22, "max_order_attempts": 24}
    last_options = {"beam_width": 128, "branching": 8, "max_nodes": 20000,
                    "budgets": {"OLL": 40, "PLL": 40, "ZBLL": 100}}
    for index, scramble in enumerate(_fresh_scrambles(count, seed), 1):
        state = apply_sequence(SOLVED_STATE, parse_moves(scramble))
        pair = {}
        for name, last_engine, zbls_engine in (
            ("neural", "neural", "neural"),
            ("symbolic", "algorithm_memory", "algorithm_memory"),
        ):
            started = time.perf_counter()
            plan = solve_v2(
                state, cross, 0, f2l, last_layer, zbls, device,
                cross_color=COLORS["D"], f2l_options=f2l_options,
                last_layer_options=last_options,
                zbls_options=({"width": 48, "branching": 8, "max_depth": 30,
                               "max_nodes": 5000, "max_moves": 35}
                              if zbls_engine == "neural" else {"max_candidates": 302}),
                last_layer_engine=last_engine, zbls_engine=zbls_engine,
                include_zbls=True, compare_last_layer_routes=True,
            )
            selected = plan["selected"]
            pair[name] = {
                "verified": bool(selected and cube_solved(apply_sequence(state, selected["moves"]))),
                "moves": selected["moves"] if selected else [],
                "stm": selected["stm"] if selected else None,
                "runtime_seconds": time.perf_counter() - started,
                "selected_route": selected["policy"] if selected else None,
                "routes": {row["policy"]: {"verified": row["verified"], "stm": row["stm"],
                                             "zbll_eligible": row["zbll_eligible"],
                                             "runtime_seconds": row["runtime_seconds"],
                                             "search_nodes": row["search_nodes"],
                                             "failure_reason": row["failure_reason"],
                                             "last_layer_route": row["last_layer_route"]}
                           for row in plan["candidates"]},
                "cross_plus_one_pairs": plan["cross_plus_one_pairs"],
                "zbls_attempt": plan["zbls_attempt"],
                "failure_reason": plan["failure_reason"],
                "search_nodes": plan["search_nodes"],
            }
        reference = reference_solve(scramble)
        reference_ok = bool(reference["verified"] and cube_solved(
            apply_sequence(state, reference["moves"])))
        report["rows"].append({"index": index, "scramble": scramble, **pair,
                               "deterministic_reference": {
                                   "verified": reference_ok, "moves": reference["moves"],
                                   "stm": len([move for move in reference["moves"] if move[0] not in "xyz"]),
                                   "runtime_seconds": reference["runtime_seconds"]}})
        report["complete"] = False
        Path(output_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("{}/{} neural={} symbolic={} routes={}/{}".format(
            index, count, pair["neural"]["verified"], pair["symbolic"]["verified"],
            pair["neural"]["selected_route"], pair["symbolic"]["selected_route"]), flush=True)
    report["complete"] = True
    report["summary"] = {
        "neural": _summary(report["rows"], "neural"),
        "symbolic": _summary(report["rows"], "symbolic"),
        "deterministic_reference": {
            "cases": count,
            "successes": sum(row["deterministic_reference"]["verified"] for row in report["rows"]),
            "successful_stm_mean": statistics.mean(
                row["deterministic_reference"]["stm"] for row in report["rows"]
                if row["deterministic_reference"]["verified"])
            if any(row["deterministic_reference"]["verified"] for row in report["rows"]) else None,
        },
        "paired_routes": _paired_comparison(report["rows"]),
    }
    Path(output_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _paired_modes(rows, left_name, right_name):
    left = [row[left_name] for row in rows]
    right = [row[right_name] for row in rows]
    result = {}
    for label, left_rows, right_rows in (("selected_full_solve", left, right),):
        left_ok = [row["verified"] for row in left_rows]
        right_ok = [row["verified"] for row in right_rows]
        common = [(left_rows[i]["stm"], right_rows[i]["stm"]) for i in range(len(rows))
                  if left_ok[i] and right_ok[i]]
        result[label] = {
            "left_only_success": sum(a and not b for a, b in zip(left_ok, right_ok)),
            "right_only_success": sum(b and not a for a, b in zip(left_ok, right_ok)),
            "exact_mcnemar_two_sided_p": _mcnemar_p(
                sum(a and not b for a, b in zip(left_ok, right_ok)),
                sum(b and not a for a, b in zip(left_ok, right_ok))),
            "both_success": sum(a and b for a, b in zip(left_ok, right_ok)),
            "neither_success": sum(not a and not b for a, b in zip(left_ok, right_ok)),
            "common_success_count": len(common),
            "mean_left_minus_right_stm_on_common_success": (
                statistics.mean(a - b for a, b in common) if common else None),
        }
    route_names = sorted({name for row in left + right for name in row["routes"]})
    result["routes"] = {}
    for route in route_names:
        left_ok = [bool(row["routes"].get(route, {}).get("verified")) for row in left]
        right_ok = [bool(row["routes"].get(route, {}).get("verified")) for row in right]
        common = [(left[i]["routes"][route]["stm"], right[i]["routes"][route]["stm"])
                  for i in range(len(rows)) if left_ok[i] and right_ok[i]
                  and route in left[i]["routes"] and route in right[i]["routes"]]
        result["routes"][route] = {
            "left_only_success": sum(a and not b for a, b in zip(left_ok, right_ok)),
            "right_only_success": sum(b and not a for a, b in zip(left_ok, right_ok)),
            "exact_mcnemar_two_sided_p": _mcnemar_p(
                sum(a and not b for a, b in zip(left_ok, right_ok)),
                sum(b and not a for a, b in zip(left_ok, right_ok))),
            "both_success": sum(a and b for a, b in zip(left_ok, right_ok)),
            "neither_success": sum(not a and not b for a, b in zip(left_ok, right_ok)),
            "common_success_count": len(common),
            "mean_left_minus_right_stm_on_common_success": (
                statistics.mean(a - b for a, b in common) if common else None),
        }
    return result


def benchmark_preference(base_path=OUTPUT, output_path=PREFERENCE_OUTPUT, device_name="cpu"):
    """Add learned algorithm ranking on the exact completed neural/shortest scramble set."""
    base = json.loads(Path(base_path).read_text(encoding="utf-8"))
    if not base.get("complete"):
        raise ValueError("A completed paired neural/shortest benchmark is required")
    device = choose_device(device_name)
    cross, _cross_checkpoint = load_expert_cross_policy(ROOT / "artifacts/v2/expert-cross-policy.pt", device)
    f2l, _f2l_checkpoint = load_hierarchical_f2l(ROOT / "artifacts/hierarchical-f2l.pt", device)
    last_layer, _last_checkpoint = load_last_layer_checkpoint(ROOT / "artifacts/last-layer-policy.pt", device)
    zbls, _zbls_checkpoint = load_zbls_checkpoint(ROOT / "artifacts/v2/zbls-synthetic-policy.pt", device)
    preference = load_algorithm_preference_model(ROOT / "artifacts/v2/algorithm-preference-policy.pt", device)
    rows = []
    f2l_options = {"width": 128, "branching": 6, "max_nodes": 6000, "max_depth": 22,
                   "max_moves": 100, "max_actions_per_pair": 22, "max_order_attempts": 24}
    last_options = {"beam_width": 128, "branching": 8, "max_nodes": 20000,
                    "budgets": {"OLL": 40, "PLL": 40, "ZBLL": 100}}
    for index, old in enumerate(base["rows"], 1):
        state = apply_sequence(SOLVED_STATE, parse_moves(old["scramble"]))
        started = time.perf_counter()
        plan = solve_v2(
            state, cross, 0, f2l, last_layer, zbls, device,
            cross_color=COLORS["D"], f2l_options=f2l_options, last_layer_options=last_options,
            zbls_options={"max_candidates": 302}, last_layer_engine="algorithm_memory",
            zbls_engine="algorithm_memory", include_zbls=True, compare_last_layer_routes=True,
            algorithm_preference_model=preference,
        )
        selected = plan["selected"]
        preference_row = {
            "verified": bool(selected and cube_solved(apply_sequence(state, selected["moves"]))),
            "moves": selected["moves"] if selected else [],
            "stm": selected["stm"] if selected else None,
            "runtime_seconds": time.perf_counter() - started,
            "selected_route": selected["policy"] if selected else None,
            "routes": {candidate["policy"]: {
                "verified": candidate["verified"], "stm": candidate["stm"],
                "zbll_eligible": candidate["zbll_eligible"],
                "runtime_seconds": candidate["runtime_seconds"],
                "search_nodes": candidate["search_nodes"],
                "failure_reason": candidate["failure_reason"],
                "last_layer_route": candidate["last_layer_route"],
            } for candidate in plan["candidates"]},
            "cross_plus_one_pairs": plan["cross_plus_one_pairs"],
            "zbls_attempt": plan["zbls_attempt"],
            "failure_reason": plan["failure_reason"], "search_nodes": plan["search_nodes"],
        }
        rows.append({"index": index, "scramble": old["scramble"],
                     "neural": old["neural"], "shortest": old["symbolic"],
                     "preference": preference_row,
                     "deterministic_reference": old["deterministic_reference"]})
        report = {
            "task": "paired neural, shortest-memory, and learned-preference routes on identical scrambles",
            "seed": base["seed"], "cross_color": base["cross_color"], "device": str(device),
            "environment": base.get("environment"),
            "checkpoint_epochs": base.get("checkpoint_epochs"),
            "search_budgets": base.get("search_budgets"),
            "base_benchmark": str(Path(base_path)), "complete": False, "rows": rows,
        }
        Path(output_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("preference {}/{} {}".format(index, len(base["rows"]),
              "verified" if preference_row["verified"] else preference_row["failure_reason"]), flush=True)
    report["complete"] = True
    report["summary"] = {
        key: _summary(rows, key) for key in ("neural", "shortest", "preference")
    }
    report["paired_shortest_vs_preference"] = _paired_modes(rows, "shortest", "preference")
    report["paired_neural_vs_shortest"] = base["summary"]["paired_routes"]
    report["checkpoint_sha256"] = {
        "preference": hashlib.sha256((ROOT / "artifacts/v2/algorithm-preference-policy.pt").read_bytes()).hexdigest(),
        **base["checkpoint_sha256"],
    }
    Path(output_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261129)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--preference-base", type=Path)
    parser.add_argument("--preference-output", type=Path, default=PREFERENCE_OUTPUT)
    args = parser.parse_args()
    if args.preference_base:
        result = benchmark_preference(args.preference_base, args.preference_output, args.device)
    else:
        result = benchmark(args.count, args.seed, args.device, args.output)
    print(json.dumps(result.get("summary", result.get("paired_shortest_vs_preference")), indent=2))
