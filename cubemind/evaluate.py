"""Evaluate teacher-forced moves, a frequency baseline, and Cross rollouts."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

from cubemind.environment.cube import MOVE_TOKENS, cross_complete, shortest_cross_solution
from cubemind.training.engine import CrossDataset, beam_search_cross, choose_device, model_from_checkpoint, rollout, score_model


def frequency_baseline(train_rows, test_rows):
    counts = Counter(row["move_id"] for row in train_rows)
    order = sorted(range(18), key=lambda move: (-counts[move], move))
    total = sum(counts.values()) + 18
    probabilities = {move: (counts[move] + 1) / total for move in range(18)}
    top = [0, 0, 0]
    for row in test_rows:
        for index, k in enumerate((1, 3, 5)):
            top[index] += int(row["move_id"] in order[:k])
    return {
        "method": "global training-set move frequency with add-one smoothing",
        "samples": len(test_rows),
        "cross_entropy": sum(-math.log(probabilities[row["move_id"]]) for row in test_rows) / len(test_rows),
        "top1_accuracy": top[0] / len(test_rows),
        "top3_accuracy": top[1] / len(test_rows),
        "top5_accuracy": top[2] / len(test_rows),
        "ranked_moves": [MOVE_TOKENS[move] for move in order],
    }


def rollout_metrics(model, test_rows, max_history, device, max_moves=20):
    starts = {}
    for row in test_rows:
        starts.setdefault(row["solve_id"], row["cube_state"])
    completed_lengths = []
    for initial_state in starts.values():
        moves, state = rollout(model, initial_state, max_history, device, max_moves=max_moves)
        if cross_complete(state):
            completed_lengths.append(len(moves))
    return {
        "solves": len(starts),
        "cross_completion_rate": len(completed_lengths) / len(starts),
        "successful_cross_move_count_mean": sum(completed_lengths) / len(completed_lengths) if completed_lengths else None,
        "successful_cross_move_count_median": sorted(completed_lengths)[len(completed_lengths) // 2] if completed_lengths else None,
        "rollout_move_limit": max_moves,
    }


def beam_search_metrics(model, test_rows, max_history, device, beam_width, max_moves=10):
    starts = {}
    for row in test_rows:
        starts.setdefault(row["solve_id"], row["cube_state"])
    completed_lengths = []
    excess_moves = []
    model_nodes = 0
    started = time.perf_counter()
    for initial_state in starts.values():
        moves, final_state, nodes, solved = beam_search_cross(
            model, initial_state, max_history, device, beam_width, max_moves
        )
        model_nodes += nodes
        if solved:
            if not cross_complete(final_state):
                raise AssertionError("Beam search reported success without a completed Cross")
            completed_lengths.append(len(moves))
            excess_moves.append(len(moves) - len(shortest_cross_solution(initial_state)))
    elapsed = time.perf_counter() - started
    return {
        "method": "cumulative neural log-probability beam search; same-face turns pruned",
        "solves": len(starts),
        "beam_width": beam_width,
        "move_limit": max_moves,
        "cross_completion_rate": len(completed_lengths) / len(starts),
        "successful_cross_move_count_mean": sum(completed_lengths) / len(completed_lengths) if completed_lengths else None,
        "successful_cross_move_count_median": sorted(completed_lengths)[len(completed_lengths) // 2] if completed_lengths else None,
        "optimal_cross_length_rate_among_successes": sum(value == 0 for value in excess_moves) / len(excess_moves) if excess_moves else None,
        "extra_moves_over_exact_shortest_mean": sum(excess_moves) / len(excess_moves) if excess_moves else None,
        "model_parent_states_scored": model_nodes,
        "elapsed_seconds": elapsed,
        "milliseconds_per_start": elapsed * 1000 / len(starts),
    }


def evaluate(checkpoint_path, test_path, train_path, output_path=None, beam_width=0, beam_max_moves=10):
    if beam_width < 0 or beam_max_moves < 0:
        raise ValueError("beam_width and beam_max_moves must be non-negative")
    device = choose_device("auto")
    model, checkpoint = model_from_checkpoint(checkpoint_path, device)
    test = CrossDataset(test_path).rows
    train = CrossDataset(train_path).rows
    for key in ("solve_id", "scramble_id"):
        overlap = {row[key] for row in train}.intersection(row[key] for row in test)
        if overlap:
            raise ValueError("Train/test {} overlap: {}".format(key, len(overlap)))
    max_history = int(checkpoint["model_config"].get("max_history", 8)) if checkpoint["model_config"].get("use_history", False) else 0
    result = {
        "checkpoint": str(checkpoint_path),
        "model": checkpoint["name"],
        "device": str(device),
        "test_samples": len(test),
        "test_scrambles": len({row["scramble_id"] for row in test}),
        "sample_kinds": sorted({row.get("sample_kind", "unknown") for row in test}),
        "teacher_forced": score_model(model, test, max_history, device),
        "frequency_baseline": frequency_baseline(train, test),
        "autoregressive_cross_rollout": rollout_metrics(model, test, max_history, device),
        "warning": "Synthetic-oracle results are not evidence of human imitation.",
    }
    if beam_width:
        result["policy_guided_beam_search"] = beam_search_metrics(
            model, test, max_history, device, beam_width, beam_max_moves
        )
    rendered = json.dumps(result, indent=2) + "\n"
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--test", type=Path, default=Path("data/processed/test.jsonl"))
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--beam-width", type=int, default=0, help="Also evaluate policy-guided beam search (0 disables it)")
    parser.add_argument("--beam-max-moves", type=int, default=10)
    args = parser.parse_args(argv)
    evaluate(args.checkpoint, args.test, args.train, args.output, args.beam_width, args.beam_max_moves)


if __name__ == "__main__":
    main()
