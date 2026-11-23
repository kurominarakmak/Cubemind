"""Reproducible held-out benchmark for greedy, search, and reference CFOP solving."""

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import time
from pathlib import Path

import torch

from cubemind.environment.cube import (
    MOVE_TOKENS, SIMULATOR_MOVE_TOKENS, SOLVED_STATE, apply_move, apply_sequence,
    cross_complete, f2l_complete, parse_moves, zbll_eligible,
)
from cubemind.models.common import encode_history
from cubemind.neural_solver import (
    PHASES, cube_solved, encode_features, load_expert_f2l_policy, load_solver_checkpoint, solve_neural, teacher_request,
)
from cubemind.training.engine import choose_device, model_from_checkpoint

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "neural-cfop-benchmark.json"
DEFAULT_CHECKPOINT = ROOT / "artifacts" / "neural-cfop-policy.pt"
BENCHMARK_PHASE_BUDGETS = {"CROSS": 20, "F2L": 30, "OLL": 20, "PLL": 20, "ZBLL": 30}


def _scrambles(count, seed, excluded=()):
    rng = random.Random(seed)
    faces, suffixes = "URFDLB", ("", "'", "2")
    excluded = set(excluded)
    rows = []
    while len(rows) < count:
        moves = []
        while len(moves) < 25:
            face = rng.choice(faces)
            if moves and moves[-1][0] == face:
                continue
            moves.append(face + rng.choice(suffixes))
        scramble = " ".join(moves)
        if scramble not in excluded:
            rows.append(scramble)
            excluded.add(scramble)
    return rows


def _wilson(successes, total):
    if total == 0:
        return [None, None]
    z = 1.96
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _p95(values):
    values = sorted(values)
    return values[max(0, math.ceil(0.95 * len(values)) - 1)] if values else None


def _aggregate(rows):
    successes = [row for row in rows if row["verified"]]
    cross = sum(row.get("cross_complete", False) for row in rows)
    f2l = sum(row.get("f2l_complete", False) for row in rows)
    eligible = sum(row.get("zbll_eligible", False) for row in rows)
    runtimes = [row["runtime_seconds"] for row in rows]
    moves = [row["stm"] for row in successes]
    return {
        "scrambles": len(rows),
        "full_solve_successes": len(successes),
        "full_solve_success_rate": len(successes) / len(rows),
        "full_solve_success_rate_wilson_95": _wilson(len(successes), len(rows)),
        "cross_completion_rate": cross / len(rows),
        "f2l_completion_rate": f2l / len(rows),
        "last_layer_completion_rate": len(successes) / len(rows),
        "zbll_eligibility_rate_after_f2l": eligible / len(rows),
        "mean_stm_successful": statistics.mean(moves) if moves else None,
        "median_stm_successful": statistics.median(moves) if moves else None,
        "median_runtime_ms": statistics.median(runtimes) * 1000 if runtimes else None,
        "p95_runtime_ms": _p95(runtimes) * 1000 if runtimes else None,
        "mean_runtime_ms": statistics.mean(runtimes) * 1000 if runtimes else None,
        "mean_search_nodes": statistics.mean([row.get("states_explored", 0) for row in rows]) if rows else None,
        "repeated_state_failures": sum(str(row.get("failure_reason", "")).startswith("repeated_state") for row in rows),
    }


def _original_cross(model, start, device, history_length, max_moves=20):
    current, history, moves, states = tuple(start), [], [], [tuple(start)]
    started = time.perf_counter()
    reason = None
    for _step in range(max_moves):
        if cross_complete(current):
            break
        states_tensor = torch.tensor([current], dtype=torch.long, device=device)
        histories = encode_history([history], history_length, device)
        with torch.inference_mode():
            logits = model(states_tensor, histories)
        if logits.ndim != 2 or logits.shape[1] != 18 or not torch.isfinite(logits).all():
            reason = "invalid_action_output"
            break
        move = MOVE_TOKENS[int(logits[0].argmax())]
        current = apply_move(current, move)
        moves.append(move)
        states.append(current)
        if current in states[:-1]:
            reason = "repeated_state:CROSS"
            break
        history.append(MOVE_TOKENS.index(move))
    if reason is None and not cross_complete(current):
        reason = "cross_move_budget"
    return {
        "status": "CROSS COMPLETE" if cross_complete(current) else "FAILED / INCOMPLETE",
        "moves": moves, "last_state": current, "verified": False,
        "cross_complete": cross_complete(current), "f2l_complete": False,
        "zbll_eligible": False, "states_explored": len(moves),
        "runtime_seconds": time.perf_counter() - started,
        "failure_reason": reason or "cross_only_model_has_no_f2l_or_last_layer_policy",
    }


def _reference_row(scramble, row):
    current = apply_sequence(SOLVED_STATE, parse_moves(scramble))
    phase_states = {}
    moves = []
    for phase in PHASES[:4]:
        segment = parse_moves(row["phases"][phase])
        current = apply_sequence(current, segment)
        moves.extend(segment)
        phase_states[phase] = current
    verified = bool(row["reference_verified"]) and cube_solved(current)
    if not verified:
        raise ValueError("Reference trajectory failed CubeMind replay for {}".format(scramble))
    return {
        "status": "SOLVED", "moves_by_phase": row["phases"], "moves": moves,
        "last_state": current, "phase_states": phase_states, "verified": True,
        "cross_complete": cross_complete(phase_states["CROSS"]),
        "f2l_complete": f2l_complete(phase_states["F2L"]),
        "zbll_eligible": zbll_eligible(phase_states["F2L"]),
        "states_explored": 0, "runtime_seconds": row["reference_ms"] / 1000.0,
        "failure_reason": None,
    }


def _accuracy_on_reference(model, rows, device):
    samples = []
    for scramble, row in rows:
        state = apply_sequence(SOLVED_STATE, parse_moves(scramble))
        history = []
        for phase in PHASES[:4]:
            for move in parse_moves(row["phases"][phase]):
                samples.append((encode_features(state, history, phase, len(history), device),
                                SIMULATOR_MOVE_TOKENS.index(move), phase))
                state = apply_move(state, move)
                history.append(SIMULATOR_MOVE_TOKENS.index(move))
    result = {}
    model.eval()
    with torch.inference_mode():
        for phase in PHASES[:4]:
            selected = [item for item in samples if item[2] == phase]
            if not selected:
                continue
            top1 = top3 = count = 0
            for start in range(0, len(selected), 512):
                batch = selected[start:start + 512]
                features = torch.stack([item[0] for item in batch])
                labels = torch.tensor([item[1] for item in batch], dtype=torch.long, device=device)
                logits, _values = model(features)
                ordering = logits.topk(3, dim=1).indices
                top1 += int((ordering[:, 0] == labels).sum())
                top3 += int((ordering == labels[:, None]).any(dim=1).sum())
                count += len(batch)
            result[phase.lower()] = {"samples": count, "top1_accuracy": top1 / count, "top3_accuracy": top3 / count}
    return result


def _annotate_neural(result, scramble):
    result["scramble"] = scramble
    result["cross_complete"] = any(phase == "CROSS" and cross_complete(state) for phase, state in result["phase_states"])
    result["f2l_complete"] = any(phase == "F2L" and f2l_complete(state) for phase, state in result["phase_states"])
    result["zbll_eligible"] = result["zbll_eligible_after_f2l"]
    result["stm"] = sum(move not in ("x", "y", "z") for move in result["moves"])
    return result


def benchmark(checkpoint_path=DEFAULT_CHECKPOINT, output_path=DEFAULT_OUTPUT, count=100, seed=20261119,
              device_name="auto", beam_width=16, branching=6, max_nodes=3000, max_moves=120):
    device = choose_device(device_name)
    model, checkpoint = load_solver_checkpoint(checkpoint_path, device)
    original, original_checkpoint = model_from_checkpoint(ROOT / "artifacts" / "mlp-state-history.pt", device)
    original_history = int(original_checkpoint["model_config"].get("max_history", 8)) if original_checkpoint["model_config"].get("use_history", False) else 0
    expert_f2l, expert_f2l_checkpoint = load_expert_f2l_policy(
        ROOT / "artifacts" / "reco-cfop-context.pt", model, device,
    )
    training_path = ROOT / "data" / "processed" / "neural_solver" / "cfop_teacher.jsonl"
    excluded = []
    if training_path.exists():
        excluded = [json.loads(line)["scramble"] for line in training_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    scrambles = _scrambles(count, seed, excluded)
    teacher_rows = teacher_request({"mode": "solve_many", "scrambles": scrambles})
    if len(teacher_rows) != count:
        raise ValueError("Reference teacher returned the wrong number of evaluation cases")
    reference_rows, original_rows, phase_only_rows, greedy_rows, search_rows = [], [], [], [], []
    expert_f2l_rows, expert_f2l_search_rows = [], []
    neural_teacher = list(zip(scrambles, teacher_rows))
    for index, (scramble, teacher) in enumerate(neural_teacher):
        start = apply_sequence(SOLVED_STATE, parse_moves(scramble))
        reference = _reference_row(scramble, teacher)
        reference["scramble"] = scramble
        reference_rows.append(reference)

        original_result = _original_cross(original, start, device, original_history)
        original_result.update({"scramble": scramble, "stm": sum(move not in ("x", "y", "z") for move in original_result["moves"])})
        original_rows.append(original_result)

        phase_only = solve_neural(start, model, device, mode="greedy", max_moves=max_moves,
                                  budgets=BENCHMARK_PHASE_BUDGETS)
        phase_only_rows.append(_annotate_neural(phase_only, scramble))

        greedy = solve_neural(start, model, device, mode="greedy", max_moves=max_moves,
                              budgets=BENCHMARK_PHASE_BUDGETS,
                              cross_model=original, cross_history_length=original_history)
        greedy_rows.append(_annotate_neural(greedy, scramble))

        expert_greedy = solve_neural(
            start, model, device, mode="greedy", max_moves=max_moves,
            budgets=BENCHMARK_PHASE_BUDGETS,
            cross_model=original, cross_history_length=original_history,
            f2l_model=expert_f2l,
        )
        expert_f2l_rows.append(_annotate_neural(expert_greedy, scramble))

        searched = solve_neural(start, model, device, mode="search", beam_width=beam_width,
                                branching=branching, max_nodes=max_nodes, max_moves=max_moves,
                                budgets=BENCHMARK_PHASE_BUDGETS,
                                cross_model=original, cross_history_length=original_history,
                                cross_search_width=256, cross_search_depth=10, cross_search_max_nodes=10000)
        search_rows.append(_annotate_neural(searched, scramble))

        expert_searched = solve_neural(
            start, model, device, mode="search", beam_width=beam_width,
            branching=branching, max_nodes=max_nodes, max_moves=max_moves,
            budgets=BENCHMARK_PHASE_BUDGETS,
            cross_model=original, cross_history_length=original_history,
            cross_search_width=256, cross_search_depth=10, cross_search_max_nodes=10000,
            f2l_model=expert_f2l,
        )
        expert_f2l_search_rows.append(_annotate_neural(expert_searched, scramble))
        if (index + 1) % 10 == 0:
            print("evaluated {}/{} scrambles".format(index + 1, count), flush=True)

    model_accuracy = _accuracy_on_reference(model, neural_teacher, device)
    reference_rows = [dict(row, stm=sum(move not in ("x", "y", "z") for move in row["moves"])) for row in reference_rows]
    checkpoint_file = Path(checkpoint_path)
    dagger_manifest_path = training_path.with_name("dagger_manifest.json")
    dagger_manifest = json.loads(dagger_manifest_path.read_text(encoding="utf-8")) if dagger_manifest_path.exists() else None
    report = {
        "benchmark": "held-out 3x3 CFOP full-solve evaluation",
        "sample_kind": "evaluation_only",
        "seed": seed,
        "scramble_count": count,
        "scramble_generator": "25 random outer turns with no adjacent same face; seed-separated from training scrambles",
        "training_dataset_sha256": checkpoint["cfop_data_sha256"],
        "training_seed": checkpoint.get("seed"),
        "training_config": checkpoint.get("training_config"),
        "training_data": {
            "human_expert_rows": 0,
            "cfop_teacher_source": "@moishy/cfop 0.4.0; synthetic reference targets",
            "cfop_dataset_sha256": checkpoint.get("cfop_data_sha256"),
            "dagger_dataset_sha256": checkpoint.get("dagger_data_sha256"),
            "zbll_algorithm_source": checkpoint.get("zbll_data_source"),
            "zbll_algorithms_sha256": checkpoint.get("zbll_algorithms_sha256"),
            "dagger_manifest": dagger_manifest,
        },
        "model_checkpoint": str(checkpoint_path),
        "model_checkpoint_sha256": hashlib.sha256(checkpoint_file.read_bytes()).hexdigest(),
        "model_source": checkpoint["data_source"],
        "device": str(device),
        "hardware": {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__},
        "search_config": {"beam_width": beam_width, "branching": branching, "max_nodes_per_phase": max_nodes,
                          "max_moves_per_solve": max_moves, "cross_search_width": 256,
                          "cross_search_depth": 10, "cross_search_max_nodes": 10000,
                          "phase_move_budgets": BENCHMARK_PHASE_BUDGETS},
        "cross_model_checkpoint": "artifacts/mlp-state-history.pt",
        "reco_nz_f2l_checkpoint": "artifacts/reco-cfop-context.pt",
        "reco_nz_f2l_checkpoint_sha256": hashlib.sha256(
            (ROOT / "artifacts" / "reco-cfop-context.pt").read_bytes()).hexdigest(),
        "reco_nz_f2l_training_samples": expert_f2l_checkpoint.get("train_samples"),
        "reference": "@moishy/cfop 0.4.0; deterministic comparison only",
        "move_count": "STM; x/y/z whole-cube rotations excluded; wide and slice turns count as one",
        "original_mlp_cross_only": _aggregate(original_rows),
        "phase_policy_only_greedy": _aggregate(phase_only_rows),
        "neural_greedy": _aggregate(greedy_rows),
        "reco_nz_f2l_hybrid_greedy": _aggregate(expert_f2l_rows),
        "neural_guided_search": _aggregate(search_rows),
        "reco_nz_f2l_hybrid_search": _aggregate(expert_f2l_search_rows),
        "deterministic_reference": _aggregate(reference_rows),
        "synthetic_teacher_forced_action_accuracy_on_test_scrambles": model_accuracy,
        "human_execution_or_recognition_time": "not measured",
        "rows": {
            "original_mlp_cross_only": original_rows,
            "phase_policy_only_greedy": phase_only_rows,
            "neural_greedy": greedy_rows,
            "reco_nz_f2l_hybrid_greedy": expert_f2l_rows,
            "neural_guided_search": search_rows,
            "reco_nz_f2l_hybrid_search": expert_f2l_search_rows,
            "deterministic_reference": reference_rows,
        },
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
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--branching", type=int, default=3)
    parser.add_argument("--max-nodes", type=int, default=500)
    parser.add_argument("--max-moves", type=int, default=120)
    args = parser.parse_args(argv)
    benchmark(args.checkpoint, args.output, args.count, args.seed, args.device,
              args.beam_width, args.branching, args.max_nodes, args.max_moves)


if __name__ == "__main__":
    main()
