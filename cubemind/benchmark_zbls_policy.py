"""Held-out case-identity evaluation for the neural ZBLS policies."""

import hashlib
import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from cubemind.environment.cube import COLORS, SOLVED_STATE, apply_sequence
from cubemind.train_zbls_policy import _recolor, _rotated_color_map
from cubemind.training.engine import choose_device
from cubemind.zbls import load_zbls_checkpoint, solve_zbls

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/v2/zbls-training-evaluation.json"
OUTPUT = ROOT / "artifacts/v2/zbls-search-confirmatory.json"
PILOT_CASES = {
    "CubeRoot-6088", "CubeRoot-5301", "CubeRoot-5359", "CubeRoot-6068",
    "CubeRoot-5297", "CubeRoot-6070", "CubeRoot-5299", "CubeRoot-5330",
    "CubeRoot-6052", "CubeRoot-5280",
}


def _wilson(successes, total):
    if not total:
        return [None, None]
    z, p = 1.96, successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


class _UniformPolicy(torch.nn.Module):
    def forward(self, features):
        zeros = features.new_zeros(len(features))
        return features.new_zeros((len(features), 72)), zeros, zeros


def benchmark(device_name="cpu", max_nodes=3000, output_path=OUTPUT, include_pilot=False):
    training = json.loads(REPORT.read_text(encoding="utf-8"))
    split_by_case = training["synthetic"]["split_by_case"]
    cases = json.loads((ROOT / "data/algorithms/zbls_cases.json").read_text(encoding="utf-8"))["cases"]
    by_id = {case["case_id"]: case for case in cases}
    heldout = sorted(case_id for case_id, split in split_by_case.items()
                     if split == "test" and (include_pilot or case_id not in PILOT_CASES))
    expected = 46 if include_pilot else 36
    if len(heldout) != expected:
        raise ValueError("Expected {} held-out ZBLS case identities".format(expected))
    starts = []
    for case_id in heldout:
        case = by_id[case_id]
        state = apply_sequence(SOLVED_STATE, case["setup"])
        state = _recolor(state, _rotated_color_map(COLORS["D"]))
        yaw_index = int(hashlib.sha256(case_id.encode()).hexdigest()[:8], 16) % 4
        yaw = ([], ["y"], ["y2"], ["y'"])[yaw_index]
        starts.append((case_id, apply_sequence(state, yaw)))
    device = choose_device(device_name)
    torch.set_num_threads(2)
    result = {
        "task": "neural-guided ZBLS search on unseen canonical case identities",
        "cases": len(starts),
        "case_split": "case-disjoint 15% test; {} prior pilot cases excluded".format(
            0 if include_pilot else len(PILOT_CASES)),
        "cross_color": "white", "search_budget": {
            "width": 32, "branching": 6, "max_depth": 28, "max_nodes": max_nodes,
            "max_moves": 32,
        },
        "device": str(device), "models": {},
    }
    variants = {}
    for name in ("synthetic", "human", "mixed"):
        checkpoint = ROOT / "artifacts/v2" / ("zbls-{}-policy.pt".format(name))
        model, _metadata = load_zbls_checkpoint(checkpoint, device)
        variants[name + "_search"] = (model, 32, 6, hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        if name == "synthetic":
            variants["synthetic_greedy"] = (model, 1, 1, hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    variants["uniform_search"] = (_UniformPolicy().to(device), 32, 6, None)
    for name, (model, width, branching, checkpoint_hash) in variants.items():
        rows, start_time = [], time.perf_counter()
        for index, (case_id, state) in enumerate(starts, 1):
            solved = solve_zbls(
                state, model, device, cross_color=COLORS["D"], width=width,
                branching=branching, max_depth=28, max_nodes=max_nodes, max_moves=32,
            )
            rows.append({"case_id": case_id, "verified": solved["verified"],
                         "moves": solved["moves"], "states_explored": solved["states_explored"],
                         "runtime_seconds": solved["runtime_seconds"],
                         "failure_reason": solved["failure_reason"]})
            print("{} {}/{} {}".format(name, index, len(starts), case_id), flush=True)
        successes = [row for row in rows if row["verified"]]
        result["models"][name] = {
            "checkpoint_sha256": checkpoint_hash,
            "successes": len(successes), "success_rate": len(successes) / len(rows),
            "success_rate_wilson_95": _wilson(len(successes), len(rows)),
            "successful_stm_mean": statistics.mean(
                sum(2 if move[0].islower() or move[1:2] == "w" else 1
                    for move in row["moves"] if move[0] not in "xyz") for row in successes
            ) if successes else None,
            "search_nodes_mean": statistics.mean(row["states_explored"] for row in rows),
            "runtime_mean_ms": statistics.mean(row["runtime_seconds"] for row in rows) * 1000,
            "wall_runtime_seconds": time.perf_counter() - start_time,
            "failure_reasons": {reason: sum(row["failure_reason"] == reason for row in rows)
                                for reason in sorted({row["failure_reason"] for row in rows if row["failure_reason"]})},
            "per_case": rows,
        }
    result["offline_algorithm_reference"] = {
        "verified_cases": len(cases), "successes": len(starts),
        "role": "algorithm database replay reference only; not used during neural inference",
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {key: value for key, value in model.items()
                             if key not in ("per_case", "failure_reasons")}
                      for name, model in result["models"].items()}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-nodes", type=int, default=3000)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--include-pilot", action="store_true")
    args = parser.parse_args()
    benchmark(args.device, args.max_nodes, args.output, args.include_pilot)
