"""Paired fresh-scramble evaluation of the bounded V2 route planner."""

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
from pathlib import Path

from cubemind.benchmark_expert_aware_f2l import _fresh_scrambles
from cubemind.benchmark_last_layer import _load_f2l_models
from cubemind.environment.cube import COLORS, SOLVED_STATE, apply_sequence, parse_moves
from cubemind.neural_solver import cube_solved, load_last_layer_checkpoint, reference_solve
from cubemind.training.engine import choose_device
from cubemind.v2_planner import solve_v2
from cubemind.zbls import load_zbls_checkpoint

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/v2/full-solve-benchmark.json"
POLICIES = ("cfop_oll_pll", "cfop_eo_aware", "neural_zbls_global")


def _wilson(successes, total):
    if not total:
        return [None, None]
    z, p = 1.96, successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _paired_sign_test(pairs):
    left_shorter = sum(left < right for left, right in pairs)
    right_shorter = sum(left > right for left, right in pairs)
    discordant = left_shorter + right_shorter
    tail = (sum(math.comb(discordant, k) for k in range(min(left_shorter, right_shorter) + 1))
            / (2 ** discordant)) if discordant else 0.5
    return {
        "left_shorter": left_shorter, "right_shorter": right_shorter,
        "ties": len(pairs) - discordant,
        "exact_two_sided_sign_test_p": min(1.0, 2.0 * tail),
    }


def _counts(moves):
    turns = [move for move in moves if move[0] not in "xyz"]
    return {
        "stm": sum(2 if move[0].islower() or move[1:2] == "w" else 1 for move in turns),
        "htm": sum(move[0] in "URFDLB" and move[1:2] != "w" for move in moves),
        "rotations": sum(move[0] in "xyz" for move in moves),
    }


def _aggregate(rows):
    cases = len(rows)
    output = {}
    for policy in POLICIES:
        candidates = [candidate for row in rows for candidate in row["candidates"]
                      if candidate["policy"] == policy]
        successful = [candidate for candidate in candidates if candidate["verified"]]
        output[policy] = {
            "candidate_count": len(candidates), "successes": len(successful),
            "success_rate_over_all_cases": len(successful) / cases if cases else None,
            "success_rate_wilson_95": _wilson(len(successful), cases),
            "eligible_after_f2l": sum(candidate["zbll_eligible"] for candidate in candidates),
            "eligible_rate_over_candidates": (sum(candidate["zbll_eligible"] for candidate in candidates)
                                               / len(candidates) if candidates else None),
            "last_layer_recoveries": sum(bool(candidate["failed_last_layer_attempts"])
                                         for candidate in candidates),
            "successful_stm_mean": statistics.mean(candidate["stm"] for candidate in successful)
            if successful else None,
            "successful_stm_median": statistics.median(candidate["stm"] for candidate in successful)
            if successful else None,
            "mean_candidate_nodes": statistics.mean(candidate["search_nodes"] for candidate in candidates)
            if candidates else None,
            "mean_candidate_runtime_ms": statistics.mean(candidate["runtime_seconds"] for candidate in candidates) * 1000
            if candidates else None,
            "failure_reasons": {reason: sum(candidate["failure_reason"] == reason for candidate in candidates)
                                for reason in sorted({candidate["failure_reason"] for candidate in candidates
                                                      if candidate["failure_reason"]})},
        }
    attempted = [row["zbls_attempt"] for row in rows if row["zbls_attempt"] is not None]
    prefix_cases = [row for row in rows if row["stages"].get("three_pair_prefix", {}).get("verified")]
    cross_plus_one = [row["cross_plus_one_pairs"] for row in rows]
    output["planner"] = {
        "selected_route_counts": {policy: sum(row["selected_route"] == policy for row in rows)
                                  for policy in POLICIES},
        "selected_verified_solves": sum(row["verified"] for row in rows),
        "selected_verified_rate": sum(row["verified"] for row in rows) / cases if cases else None,
        "selected_verified_wilson_95": _wilson(sum(row["verified"] for row in rows), cases),
        "mean_total_nodes": statistics.mean(row["search_nodes"] for row in rows) if rows else None,
        "mean_total_runtime_ms": statistics.mean(row["runtime_seconds"] for row in rows) * 1000 if rows else None,
        "zbls_attempted": len(attempted), "zbls_successes": sum(row["verified"] for row in attempted),
        "three_pair_prefixes": len(prefix_cases),
        "mean_cross_plus_one_pairs": statistics.mean(cross_plus_one) if cross_plus_one else None,
    }
    return output


def _paired_routes(rows):
    comparisons = {}
    for left_index, left in enumerate(POLICIES):
        for right in POLICIES[left_index + 1:]:
            left_only = right_only = both = neither = 0
            paired_stm = []
            for row in rows:
                candidates = {candidate["policy"]: candidate for candidate in row["candidates"]}
                a, b = candidates.get(left), candidates.get(right)
                a_ok = bool(a and a["verified"])
                b_ok = bool(b and b["verified"])
                both += a_ok and b_ok
                left_only += a_ok and not b_ok
                right_only += b_ok and not a_ok
                neither += not a_ok and not b_ok
                if a_ok and b_ok:
                    paired_stm.append((a["stm"], b["stm"]))
            discordant = left_only + right_only
            tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1)) / (2 ** discordant) if discordant else 0.5
            comparisons["{}_vs_{}".format(left, right)] = {
                "left_only_success": left_only, "right_only_success": right_only,
                "both_success": both, "neither_success": neither,
                "exact_mcnemar_two_sided_p": min(1.0, 2.0 * tail),
                "common_success_cases": len(paired_stm),
                "left_mean_stm_common_success": statistics.mean(a for a, _b in paired_stm)
                if paired_stm else None,
                "right_mean_stm_common_success": statistics.mean(b for _a, b in paired_stm)
                if paired_stm else None,
                "mean_stm_difference_left_minus_right": statistics.mean(a - b for a, b in paired_stm)
                if paired_stm else None,
                "paired_stm_sign_test": _paired_sign_test(paired_stm),
            }
    return comparisons


def _paired_selected(rows):
    comparisons = {}
    for policy in POLICIES:
        selected_only = fixed_only = both = neither = 0
        common = []
        for row in rows:
            fixed = next((candidate for candidate in row["candidates"]
                          if candidate["policy"] == policy), None)
            selected_ok = bool(row["verified"])
            fixed_ok = bool(fixed and fixed["verified"])
            both += selected_ok and fixed_ok
            selected_only += selected_ok and not fixed_ok
            fixed_only += fixed_ok and not selected_ok
            neither += not selected_ok and not fixed_ok
            if selected_ok and fixed_ok:
                common.append((row["selected_stm"], fixed["stm"]))
        discordant = selected_only + fixed_only
        tail = (sum(math.comb(discordant, k) for k in range(min(selected_only, fixed_only) + 1))
                / (2 ** discordant)) if discordant else 0.5
        comparisons["planner_vs_" + policy] = {
            "planner_only_success": selected_only, "fixed_route_only_success": fixed_only,
            "both_success": both, "neither_success": neither,
            "exact_mcnemar_two_sided_p": min(1.0, 2.0 * tail),
            "common_success_cases": len(common),
            "planner_mean_stm_common_success": statistics.mean(a for a, _b in common) if common else None,
            "fixed_route_mean_stm_common_success": statistics.mean(b for _a, b in common) if common else None,
            "mean_stm_difference_planner_minus_fixed": statistics.mean(a - b for a, b in common)
            if common else None,
            "paired_stm_sign_test": _paired_sign_test(common),
        }
    return comparisons


def _paired_reference(rows):
    comparisons = {}
    for policy in (*POLICIES, "planner"):
        left_only = right_only = both = neither = 0
        common_stm = []
        for row in rows:
            reference = row["deterministic_reference"]
            if policy == "planner":
                left_ok, left_stm = bool(row["verified"]), row["selected_stm"]
            else:
                candidate = next((item for item in row["candidates"] if item["policy"] == policy), None)
                left_ok = bool(candidate and candidate["verified"])
                left_stm = candidate["stm"] if candidate else None
            right_ok = bool(reference["verified"])
            both += left_ok and right_ok
            left_only += left_ok and not right_ok
            right_only += right_ok and not left_ok
            neither += not left_ok and not right_ok
            if left_ok and right_ok:
                common_stm.append((left_stm, reference["stm"]))
        discordant = left_only + right_only
        tail = (sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1))
                / (2 ** discordant)) if discordant else 0.5
        comparisons[policy + "_vs_deterministic"] = {
            "neural_only_success": left_only, "reference_only_success": right_only,
            "both_success": both, "neither_success": neither,
            "exact_mcnemar_two_sided_p": min(1.0, 2.0 * tail),
            "common_success_cases": len(common_stm),
            "neural_mean_stm_common_success": statistics.mean(a for a, _b in common_stm)
            if common_stm else None,
            "reference_mean_stm_common_success": statistics.mean(b for _a, b in common_stm)
            if common_stm else None,
            "mean_stm_difference_neural_minus_reference": statistics.mean(a - b for a, b in common_stm)
            if common_stm else None,
            "paired_stm_sign_test": _paired_sign_test(common_stm),
        }
    return comparisons


def _save(path, report, rows, complete):
    report["rows"] = rows
    report["summary"] = _aggregate(rows)
    report["paired_route_comparisons"] = _paired_routes(rows)
    report["planner_selection_comparisons"] = _paired_selected(rows)
    report["complete"] = complete
    report["completed_cases"] = len(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def finalize_report(path=OUTPUT):
    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("complete"):
        raise ValueError("Cannot finalize a partial V2 benchmark")
    report["paired_route_comparisons"] = _paired_routes(report["rows"])
    report["planner_selection_comparisons"] = _paired_selected(report["rows"])
    if report["rows"] and all("deterministic_reference" in row for row in report["rows"]):
        report["paired_reference_comparisons"] = _paired_reference(report["rows"])
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report["paired_route_comparisons"]


def add_deterministic_reference(path=OUTPUT):
    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("complete"):
        raise ValueError("Cannot add a reference to a partial V2 benchmark")
    rows = report["rows"]
    for index, row in enumerate(rows, 1):
        reference = reference_solve(row["scramble"])
        start = apply_sequence(SOLVED_STATE, parse_moves(row["scramble"]))
        verified = bool(reference["verified"] and cube_solved(apply_sequence(start, reference["moves"])))
        counts = _counts(reference["moves"])
        row["deterministic_reference"] = {
            "verified": verified, "moves": reference["moves"], **counts,
            "runtime_seconds": reference["runtime_seconds"],
            "failure_reason": None if verified else reference["failure_reason"],
        }
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("reference {}/{} {}".format(index, len(rows), "verified" if verified else "failed"), flush=True)
    successes = [row["deterministic_reference"] for row in rows if row["deterministic_reference"]["verified"]]
    report["summary"]["deterministic_reference"] = {
        "successes": len(successes), "cases": len(rows),
        "success_rate": len(successes) / len(rows) if rows else None,
        "success_rate_wilson_95": _wilson(len(successes), len(rows)),
        "successful_stm_mean": statistics.mean(row["stm"] for row in successes) if successes else None,
        "successful_htm_mean": statistics.mean(row["htm"] for row in successes) if successes else None,
        "mean_compute_runtime_ms": statistics.mean(row["runtime_seconds"] for row in successes) * 1000
        if successes else None,
    }
    report["paired_route_comparisons"] = _paired_routes(rows)
    report["planner_selection_comparisons"] = _paired_selected(rows)
    report["paired_reference_comparisons"] = _paired_reference(rows)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report["summary"]["deterministic_reference"]


def benchmark(count=100, seed=20261128, device_name="cpu", cross_color="white", output_path=OUTPUT):
    if count < 1:
        raise ValueError("count must be positive")
    colors = {"white": COLORS["D"], "yellow": COLORS["U"]}
    requested_colors = tuple(colors) if cross_color == "both" else (cross_color,)
    if any(color not in colors for color in requested_colors):
        raise ValueError("cross_color must be white, yellow, or both")
    device = choose_device(device_name)
    cross, history_length, _planner, f2l, checkpoints = _load_f2l_models(device)
    last_layer, last_layer_checkpoint = load_last_layer_checkpoint(ROOT / "artifacts/last-layer-policy.pt", device)
    zbls, zbls_checkpoint = load_zbls_checkpoint(ROOT / "artifacts/v2/zbls-synthetic-policy.pt", device)
    checkpoint_paths = {
        "cross": ROOT / "artifacts/mlp-state-history.pt",
        "hierarchical_f2l": ROOT / "artifacts/hierarchical-f2l.pt",
        "expert_f2l": ROOT / "artifacts/canonical-expert-f2l.pt",
        "pair_selector": ROOT / "artifacts/expert-aware-pair-selector.pt",
        "last_layer": ROOT / "artifacts/last-layer-policy.pt",
        "zbls": ROOT / "artifacts/v2/zbls-synthetic-policy.pt",
    }
    report = {
        "task": "paired fresh-scramble comparison of conventional, EO-aware, and neural-ZBLS V2 routes",
        "seed": seed, "requested_scrambles_per_color": count,
        "cross_colors": list(requested_colors), "device": str(device),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
        "checkpoint_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for name, path in checkpoint_paths.items()},
        "checkpoint_epochs": {"last_layer": last_layer_checkpoint.get("best_epoch"),
                              "zbls": zbls_checkpoint.get("best_epoch")},
        "search_budgets": {
            "cross": {"width": 256, "branching": 10, "max_nodes": 10000},
            "f2l": {"width": 128, "branching": 6, "max_nodes": 6000, "max_depth": 22},
            "last_layer": {"beam_width": 128, "branching": 8, "max_nodes": 20000,
                           "budgets": {"OLL": 40, "PLL": 40, "ZBLL": 100}},
            "zbls": {"width": 48, "branching": 8, "max_depth": 30,
                     "max_nodes": 5000, "max_moves": 35},
        },
        "timing_scope": "software compute only; human execution and recognition were not measured",
        "selection_rule": "minimum STM among full-cube replay-verified candidates, then rotations, then nodes",
    }
    scrambles = _fresh_scrambles(count, seed)
    rows = []
    output_path = Path(output_path)
    for scramble_index, scramble in enumerate(scrambles, 1):
        start = apply_sequence(SOLVED_STATE, parse_moves(scramble))
        for color in requested_colors:
            plan = solve_v2(
                start, cross, history_length, f2l, last_layer, zbls, device,
                cross_color=colors[color],
                f2l_options={"width": 128, "branching": 6, "max_nodes": 6000,
                             "max_depth": 22, "max_moves": 100,
                             "max_actions_per_pair": 22, "max_order_attempts": 24},
                last_layer_options={"beam_width": 128, "branching": 8, "max_nodes": 20000,
                                    "budgets": {"OLL": 40, "PLL": 40, "ZBLL": 100}},
                zbls_options={"width": 48, "branching": 8, "max_depth": 30,
                              "max_nodes": 5000, "max_moves": 35},
            )
            candidates = [{key: candidate[key] for key in (
                "policy", "verified", "moves", "stm", "htm", "rotations", "zbll_eligible",
                "search_nodes", "runtime_seconds", "failure_reason", "last_layer_route",
                "failed_last_layer_attempts",
            )} for candidate in plan["candidates"]]
            rows.append({
                "scramble_index": scramble_index, "scramble": scramble, "cross_color": color,
                "status": plan["status"], "verified": plan["selected"] is not None,
                "selected_route": plan["selected"]["policy"] if plan["selected"] else None,
                "selected_moves": plan["selected"]["moves"] if plan["selected"] else [],
                "selected_stm": plan["selected"]["stm"] if plan["selected"] else None,
                "cross_complete": plan["cross_complete"],
                "cross_plus_one_pairs": plan.get("cross_plus_one_pairs", 0),
                "stages": plan["stages"], "zbls_attempt": plan["zbls_attempt"],
                "candidates": candidates, "failure_reason": plan["failure_reason"],
                "search_nodes": plan["search_nodes"], "runtime_seconds": plan["runtime_seconds"],
            })
            _save(output_path, report, rows, False)
            print("{}/{} {} {} {}".format(
                scramble_index, count, color, plan["status"], plan["selected"]["policy"]
                if plan["selected"] else plan["failure_reason"]), flush=True)
    _save(output_path, report, rows, True)
    add_deterministic_reference(output_path)
    report = json.loads(output_path.read_text(encoding="utf-8"))
    print(json.dumps(report["summary"], indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cross-color", choices=("white", "yellow", "both"), default="white")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    benchmark(args.count, args.seed, args.device, args.cross_color, args.output)
