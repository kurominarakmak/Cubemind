"""Prepare a solve-grouped Cross behavior-cloning dataset."""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from cubemind.environment.cube import (
    MOVE_TOKENS,
    SOLVED_STATE,
    apply_move,
    apply_sequence,
    cross_complete,
    parse_moves,
    shortest_cross_solution,
)

MOVE_TO_ID = {move: index for index, move in enumerate(MOVE_TOKENS)}


def _random_scramble(rng: random.Random, min_length: int = 18, max_length: int = 25) -> List[str]:
    target_length = rng.randint(min_length, max_length)
    result = []
    while len(result) < target_length:
        move = rng.choice(MOVE_TOKENS)
        if result and move[0] == result[-1][0]:
            continue
        result.append(move)
    return result


def make_synthetic_records(count: int, seed: int) -> List[dict]:
    """Create non-human shortest-Cross oracle trajectories for pipeline checks."""
    records = []
    seen_scrambles = set()
    for index in range(count):
        rng = random.Random(seed + index * 104729)
        while True:
            scramble = _random_scramble(rng)
            scramble_text = " ".join(scramble)
            if scramble_text not in seen_scrambles:
                seen_scrambles.add(scramble_text)
                break
        state = apply_sequence(SOLVED_STATE, scramble)
        cross_moves = shortest_cross_solution(state)
        if not cross_complete(apply_sequence(state, cross_moves)):
            raise AssertionError("Synthetic Cross oracle produced an invalid trajectory")
        scramble_id = hashlib.sha256(scramble_text.encode("utf-8")).hexdigest()[:16]
        records.append({
            "solver_id": "synthetic-bfs-oracle",
            "solve_id": "synthetic-{:06d}".format(index),
            "scramble_id": scramble_id,
            "scramble": scramble_text,
            "solution_moves": " ".join(cross_moves),
            "solution_scope": "CROSS_ONLY",
            "cross_face": "D",
            "cross_color": "white",
            "data_source": "synthetic:shortest-cross-bfs-v1",
            "sample_kind": "synthetic",
        })
    return records


def load_expert_records(path: Path) -> List[dict]:
    """Load permission-cleared JSONL records without copying or rewriting originals."""
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        required = ("event", "solver_id", "solve_id", "scramble_id", "scramble", "solution_moves", "cross_face",
                    "cross_color", "data_source", "reuse_permission", "reuse_permission_verified")
        missing = [name for name in required if name not in record or (name != "reuse_permission_verified" and not record[name])]
        if missing:
            raise ValueError("{}:{} missing required fields: {}".format(path, line_number, ", ".join(missing)))
        if record["reuse_permission_verified"] is not True:
            raise ValueError("{}:{} is not marked as permission-verified; refusing to ingest it".format(path, line_number))
        if record["event"] != "3x3":
            raise ValueError("{}:{} current dataset accepts only 3x3 solves".format(path, line_number))
        if record["cross_face"] != "D" or not isinstance(record["cross_color"], str) or record["cross_color"].lower() != "white":
            raise ValueError("{}:{} current prototype requires a white Down Cross".format(path, line_number))
        allowed_ranges = {"white_only", "yellow_only", "white_yellow", "all_six", "unknown"}
        color_range = record.get("solver_cross_color_range", "unknown")
        if color_range not in allowed_ranges:
            raise ValueError("{}:{} invalid solver_cross_color_range".format(path, line_number))
        if color_range == "yellow_only":
            raise ValueError("{}:{} current white-Cross dataset cannot include yellow-only solvers".format(path, line_number))
        if record.get("solution_scope", "FULL_SOLVE") != "FULL_SOLVE":
            raise ValueError("{}:{} expert input must contain a complete solve".format(path, line_number))
        records.append(record)
    if not records:
        raise ValueError("No records found in {}".format(path))
    return records


def _examples_from_record(record: dict) -> List[dict]:
    scramble = parse_moves(record["scramble"])
    moves = parse_moves(record["solution_moves"])
    start_state = apply_sequence(SOLVED_STATE, scramble)
    state = start_state
    history: List[int] = []
    examples = []
    for move in moves:
        if cross_complete(state):
            break
        if move not in MOVE_TO_ID:
            raise ValueError("Cross expert moves must be outer turns in the fixed frame; normalize rotations before import")
        examples.append({
            "cube_state": list(state),
            "move_history": history[-32:],
            "phase": "CROSS",
            "cross_face": record["cross_face"],
            "cross_color": record["cross_color"].lower(),
            "next_expert_move": move,
            "move_id": MOVE_TO_ID[move],
            "solver_id": record["solver_id"],
            "solve_id": record["solve_id"],
            "scramble_id": record["scramble_id"],
            "data_source": record["data_source"],
            "sample_kind": record.get("sample_kind", "expert"),
        })
        if record.get("sample_kind") != "synthetic":
            examples[-1]["solver_cross_color_range"] = record.get("solver_cross_color_range", "unknown")
            examples[-1]["event"] = record["event"]
        state = apply_move(state, move)
        history.append(MOVE_TO_ID[move])
    if not cross_complete(state):
        raise ValueError("Solve {} does not complete the Down Cross".format(record["solve_id"]))
    if record.get("solution_scope", "FULL_SOLVE") == "FULL_SOLVE" and state != SOLVED_STATE:
        final_state = apply_sequence(start_state, moves)
        if final_state != SOLVED_STATE:
            raise ValueError("Expert solve {} fails replay verification".format(record["solve_id"]))
    return examples


def _split_records(records: Sequence[dict], seed: int, fractions: Tuple[float, float, float]) -> Dict[str, List[dict]]:
    if abs(sum(fractions) - 1.0) > 1e-9 or any(value <= 0 for value in fractions):
        raise ValueError("Split fractions must be positive and sum to 1")
    by_scramble: Dict[str, List[dict]] = defaultdict(list)
    solve_ids = set()
    for record in records:
        if record["solve_id"] in solve_ids:
            raise ValueError("Duplicate solve_id: {}".format(record["solve_id"]))
        solve_ids.add(record["solve_id"])
        by_scramble[record["scramble_id"]].append(record)
    groups = sorted(by_scramble)
    random.Random(seed).shuffle(groups)
    n = len(groups)
    train_end = max(1, min(n - 2, int(n * fractions[0])))
    val_end = max(train_end + 1, min(n - 1, int(n * (fractions[0] + fractions[1]))))
    assignments = {"train": groups[:train_end], "validation": groups[train_end:val_end], "test": groups[val_end:]}
    if not assignments["validation"] or not assignments["test"]:
        raise ValueError("At least three distinct scrambles are needed for train/validation/test")
    return {name: [record for group in group_ids for record in by_scramble[group]] for name, group_ids in assignments.items()}


def prepare(output: Path, records: Sequence[dict], seed: int = 17) -> dict:
    split_records = _split_records(records, seed, (0.8, 0.1, 0.1))
    splits = {}
    for name, group in split_records.items():
        examples = [sample for record in group for sample in _examples_from_record(record)]
        if not examples:
            raise ValueError("Split {} contains no Cross examples".format(name))
        output.mkdir(parents=True, exist_ok=True)
        path = output / (name + ".jsonl")
        with path.open("w", encoding="utf-8") as handle:
            for sample in examples:
                handle.write(json.dumps(sample, separators=(",", ":")) + "\n")
        splits[name] = {
            "solves": len(group),
            "scrambles": len({record["scramble_id"] for record in group}),
            "samples": len(examples),
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "sample_kinds": sorted({record.get("sample_kind", "expert") for record in records}),
        "source_kinds": sorted({record["data_source"] for record in records}),
        "split_policy": "grouped_by_scramble_id; each solve is kept intact",
        "splits": splits,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: Iterable[str] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--input", type=Path, help="permission-verified expert JSONL; omit for synthetic data")
    parser.add_argument("--solves", type=int, default=256, help="number of synthetic solve groups")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    records = load_expert_records(args.input) if args.input else make_synthetic_records(args.solves, args.seed)
    print(json.dumps(prepare(args.output, records, args.seed), indent=2))


if __name__ == "__main__":
    main()
