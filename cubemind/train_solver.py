"""Train a from-scratch phase policy/value model on labeled synthetic CFOP paths."""

import argparse
import hashlib
import json
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from cubemind.environment.cube import (
    COLORS, SIMULATOR_MOVE_TOKENS, SOLVED_STATE, apply_move, apply_sequence, cross_complete,
    f2l_complete, inverse_sequence, parse_moves, zbll_eligible,
)
from cubemind.last_layer import ZBLL_PATH, zbll_case_state
from cubemind.models.solver import SOLVER_FEATURE_SIZE, SolverPolicyValue
from cubemind.neural_solver import (
    PHASES, _solve_cross_model, cube_solved, encode_features, oll_complete, teacher_request,
)
from cubemind.training.engine import choose_device, model_from_checkpoint, set_seed

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "processed" / "neural_solver" / "cfop_teacher.jsonl"
DEFAULT_CHECKPOINT = ROOT / "artifacts" / "neural-cfop-policy.pt"
DEFAULT_METRICS = ROOT / "artifacts" / "neural-cfop-training.json"
SOURCE = "synthetic:cfop-teacher+dagger+zbll-algorithms"
ZBLL_SOURCE = "synthetic:zbll-algorithms:Alg-Trainer-MIT"
CFOP_SOURCE = "synthetic:cfop-teacher:@moishy/cfop@0.4.0"
DAGGER_SOURCE = "synthetic:dagger-reference-after-project-cross-mlp"


def _load_teacher_rows(path):
    path = Path(path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("No synthetic CFOP teacher trajectories in {}".format(path))
    seen_scrambles, seen_ids, splits = set(), set(), {}
    for row in records:
        if row.get("sample_kind") != "synthetic" or row.get("data_source") not in (CFOP_SOURCE, DAGGER_SOURCE):
            raise ValueError("CFOP teacher rows must be explicitly labeled synthetic")
        split = row.get("split")
        if split not in ("train", "validation"):
            raise ValueError("CFOP rows may only use train or validation splits")
        if row.get("scramble_id") in seen_ids or row.get("scramble") in seen_scrambles:
            raise ValueError("Duplicate synthetic scramble or ID in teacher dataset")
        seen_ids.add(row["scramble_id"])
        seen_scrambles.add(row["scramble"])
        previous = splits.setdefault(row["scramble_id"], split)
        if previous != split:
            raise ValueError("A scramble crosses train and validation splits")
        if not row.get("reference_verified"):
            raise ValueError("Synthetic teacher target was not replay-verified")
    if not any(row["split"] == "train" for row in records) or not any(row["split"] == "validation" for row in records):
        raise ValueError("Both train and validation trajectories are required")
    return records


def _trajectory_examples(start, phases, source, split, trajectory_id, initial_history=()):
    state, history, examples = tuple(start), list(initial_history), []
    if any(move_id not in range(len(SIMULATOR_MOVE_TOKENS)) for move_id in history):
        raise ValueError("Invalid initial move-history ID")
    for phase in PHASES:
        moves = phases.get(phase, [])
        if not moves:
            continue
        if phase == "ZBLL" and not zbll_eligible(state):
            raise ValueError("ZBLL synthetic paths require a ZBLL-eligible starting state")
        for offset, move in enumerate(parse_moves(moves)):
            if move not in SIMULATOR_MOVE_TOKENS:
                raise ValueError("Teacher emitted an unsupported move: {}".format(move))
            examples.append({
                "features": encode_features(state, history, phase, len(history), torch.device("cpu")),
                "label": SIMULATOR_MOVE_TOKENS.index(move),
                "value": float(len(moves) - offset - 1),
                "phase": phase,
                "source": source,
                "split": split,
                "trajectory_id": trajectory_id,
            })
            state = apply_move(state, move)
            history.append(SIMULATOR_MOVE_TOKENS.index(move))
        reached = {"CROSS": cross_complete, "F2L": f2l_complete, "OLL": oll_complete,
                   "PLL": cube_solved, "ZBLL": cube_solved}[phase](state)
        if not reached:
            raise ValueError("{} teacher target failed its {} phase replay".format(trajectory_id, phase))
        if phase == "CROSS" and state[COLORS["D"] * 9 + 4] != COLORS["D"]:
            raise ValueError("CFOP teacher changed the fixed white Down cross orientation")
    return examples, state


def _zbll_examples():
    database = json.loads(ZBLL_PATH.read_text(encoding="utf-8"))
    examples = []
    all_states = set()
    for case in database["cases"]:
        algorithm = parse_moves(case["algorithms"][0])
        start = tuple(zbll_case_state(case["case_id"]))
        if apply_sequence(start, algorithm) != SOLVED_STATE or not zbll_eligible(start):
            raise ValueError("ZBLL algorithm failed the simulator for {}".format(case["case_id"]))
        # Hold out AUF presentations, while keeping each known case represented in training.
        for auf_index, auf in enumerate(([], ["U"], ["U2"], ["U'"])):
            split = ("train", "train", "validation", "test")[auf_index]
            state = apply_sequence(start, auf)
            target = inverse_sequence(auf) + algorithm
            if state in all_states:
                raise ValueError("Duplicate ZBLL state presentation crosses pose splits")
            all_states.add(state)
            if apply_sequence(state, target) != SOLVED_STATE:
                raise ValueError("AUF-normalized ZBLL target failed for {}".format(case["case_id"]))
            rows, _final = _trajectory_examples(
                state, {"ZBLL": target}, ZBLL_SOURCE, split,
                "{}-auf-{}".format(case["case_id"], auf_index),
            )
            examples.extend(rows)
    return examples


def _generate_dagger(dataset_path, dagger_path, seed):
    source_rows = [row for row in _load_teacher_rows(dataset_path) if row["data_source"] == CFOP_SOURCE]
    # Bound recovery collection to keep this first DAgger pass practical. Each split
    # remains isolated and the held-out benchmark is never queried.
    selected = []
    for split in ("train", "validation"):
        selected.extend([row for row in source_rows if row["split"] == split][:48 if split == "train" else 8])
    device = choose_device("auto")
    torch.set_num_threads(2)
    cross_model, cross_checkpoint = model_from_checkpoint(ROOT / "artifacts" / "mlp-state-history.pt", device)
    cross_history_length = int(cross_checkpoint["model_config"].get("max_history", 8)) if cross_checkpoint["model_config"].get("use_history", False) else 0
    requests, prefixes, cross_failures = [], [], []
    for index, row in enumerate(selected, 1):
        start = apply_sequence(SOLVED_STATE, parse_moves(row["scramble"]))
        if cross_complete(start):
            cross_moves = []
        else:
            _final, _history, cross_moves, _nodes, failure = _solve_cross_model(
                start, [], cross_model, cross_history_length, device, "search",
                32, 10, 5000, {start},
            )
            if failure or not cross_complete(apply_sequence(start, cross_moves)):
                cross_failures.append(row["scramble_id"])
                continue
        query = parse_moves(row["scramble"]) + cross_moves
        requests.append(" ".join(query))
        prefixes.append((row, [SIMULATOR_MOVE_TOKENS.index(move) for move in cross_moves]))
        if index % 8 == 0 or index == len(selected):
            print("DAgger Cross paths: {}/{}".format(index, len(selected)), flush=True)
    if not any(row["split"] == "train" for row, _history in prefixes) or not any(
            row["split"] == "validation" for row, _history in prefixes):
        raise ValueError("DAgger Cross search must succeed in both train and validation splits")
    targets = []
    for offset in range(0, len(requests), 8):
        targets.extend(teacher_request({"mode": "solve_many", "scrambles": requests[offset:offset + 8]}))
        print("DAgger teacher paths: {}/{}".format(len(targets), len(requests)), flush=True)
    records = []
    for index, (teacher, (origin, prefix_history)) in enumerate(zip(targets, prefixes)):
        record = {
            "scramble_id": "dagger-{}-{}".format(origin["split"], origin["scramble_id"]),
            "scramble": requests[index], "sample_kind": "synthetic", "data_source": DAGGER_SOURCE,
            "split": origin["split"], "source_scramble_id": origin["scramble_id"],
            "prefix_history": prefix_history, "phases": teacher["phases"],
            "reference_verified": teacher["reference_verified"],
        }
        start = apply_sequence(SOLVED_STATE, parse_moves(record["scramble"]))
        _examples, final = _trajectory_examples(
            start, record["phases"], DAGGER_SOURCE, record["split"], record["scramble_id"], prefix_history,
        )
        if not cube_solved(final):
            raise ValueError("DAgger reference path failed replay for {}".format(record["scramble_id"]))
        records.append(record)
    dagger_path = Path(dagger_path)
    dagger_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    manifest = {
        "sample_kind": "synthetic", "data_source": DAGGER_SOURCE,
        "training_source": "project Cross MLP search (width 32, depth 10, 5000-node cap) on up to 48 base train and 8 base validation scrambles",
        "label_source": "@moishy/cfop 0.4.0; post-Cross paths replay-verified by CubeMind",
        "seed": seed, "trajectories": len(records), "cross_failures": cross_failures,
        "train_trajectories": sum(row["split"] == "train" for row in records),
        "validation_trajectories": sum(row["split"] == "validation" for row in records),
        "base_data_sha256": hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest(),
        "cross_checkpoint_sha256": hashlib.sha256((ROOT / "artifacts" / "mlp-state-history.pt").read_bytes()).hexdigest(),
        "split_policy": "same train/validation assignment as source scramble; no held-out test rows used",
    }
    dagger_path.with_name("dagger_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return records


def build_examples(dataset_path, dagger_path=None):
    records = _load_teacher_rows(dataset_path)
    dagger_records = _load_teacher_rows(dagger_path) if dagger_path else []
    records.extend(dagger_records)
    examples = []
    seen_cfop_split = defaultdict(set)
    for row in records:
        start = apply_sequence(SOLVED_STATE, parse_moves(row["scramble"]))
        if row["scramble_id"] in seen_cfop_split[row["split"]]:
            raise ValueError("Duplicate CFOP trajectory in split")
        seen_cfop_split[row["split"]].add(row["scramble_id"])
        phase_rows, final_state = _trajectory_examples(
            start, row["phases"], row["data_source"], row["split"], row["scramble_id"],
            row.get("prefix_history", ()),
        )
        if not cube_solved(final_state):
            raise ValueError("Synthetic CFOP path did not solve {}".format(row["scramble_id"]))
        examples.extend(phase_rows)
    examples.extend(_zbll_examples())
    train = [row for row in examples if row["split"] == "train"]
    validation = [row for row in examples if row["split"] == "validation"]
    test = [row for row in examples if row["split"] == "test"]
    if not train or not validation or not test:
        raise ValueError("Training, validation, and ZBLL pose-test data are required")
    return train, validation, test


def _tensors(rows):
    return TensorDataset(
        torch.stack([row["features"] for row in rows]),
        torch.tensor([row["label"] for row in rows], dtype=torch.long),
        torch.tensor([row["value"] for row in rows], dtype=torch.float32),
    )


def _score(model, rows, device):
    model.eval()
    metrics = {}
    with torch.inference_mode():
        for phase in PHASES:
            selected = [row for row in rows if row["phase"] == phase]
            if not selected:
                continue
            features = torch.stack([row["features"] for row in selected]).to(device)
            labels = torch.tensor([row["label"] for row in selected], dtype=torch.long, device=device)
            values = torch.tensor([row["value"] for row in selected], dtype=torch.float32, device=device)
            logits, value_estimates = model(features)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            top = logits.topk(3, dim=1).indices
            metrics[phase.lower()] = {
                "samples": len(selected),
                "cross_entropy": float(loss),
                "top1_accuracy": float((top[:, 0] == labels).float().mean()),
                "top3_accuracy": float((top == labels[:, None]).any(dim=1).float().mean()),
                "value_mae_moves": float((value_estimates - values).abs().mean()),
            }
    metrics["macro_cross_entropy"] = sum(item["cross_entropy"] for item in metrics.values()) / max(1, len(metrics))
    return metrics


def _balanced_sampler(rows, seed):
    counts = Counter(row["phase"] for row in rows)
    weights = torch.tensor([1.0 / counts[row["phase"]] for row in rows], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(rows), replacement=True,
                                 generator=torch.Generator().manual_seed(seed))


def train(dataset_path=DEFAULT_DATA, checkpoint_path=DEFAULT_CHECKPOINT, metrics_path=DEFAULT_METRICS,
          epochs=20, seed=73, device_name="auto", train_solves=160, validation_solves=32,
          regenerate=False):
    dataset_path = Path(dataset_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if regenerate or not dataset_path.exists():
        records = teacher_request({
            "mode": "generate", "train": train_solves, "validation": validation_solves, "seed": seed,
        })
        dataset_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        manifest = {
            "sample_kind": "synthetic", "data_source": "synthetic:cfop-teacher:@moishy/cfop@0.4.0",
            "seed": seed, "train_solves": train_solves, "validation_solves": validation_solves,
            "scramble_generator": "25 random outer turns; fixed white/Down Cross; solver-grouped split",
            "reference": "@moishy/cfop 0.4.0; every trajectory replay-verified in its cubing-core and CubeMind simulators",
        }
        dataset_path.with_name("manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    dagger_path = dataset_path.with_name("dagger_teacher.jsonl")
    if regenerate or not dagger_path.exists():
        _generate_dagger(dataset_path, dagger_path, seed)
    dagger_records = _load_teacher_rows(dagger_path)
    train_rows, validation_rows, test_rows = build_examples(dataset_path, dagger_path)
    set_seed(seed)
    device = choose_device(device_name)
    torch.set_num_threads(2)
    model = SolverPolicyValue().to(device)
    train_data = _tensors(train_rows)
    train_loader = DataLoader(
        train_data, batch_size=512, sampler=_balanced_sampler(train_rows, seed), num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    best_loss, best_state, best_epoch, history, stale = float("inf"), None, 0, [], 0
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        total_loss = count = 0
        for features, labels, values in train_loader:
            features, labels, values = features.to(device), labels.to(device), values.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, value_estimates = model(features)
            policy_loss = torch.nn.functional.cross_entropy(logits, labels)
            value_loss = torch.nn.functional.smooth_l1_loss(value_estimates, values)
            loss = policy_loss + 0.1 * value_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * labels.size(0)
            count += labels.size(0)
        val_metrics = _score(model, validation_rows, device)
        record = {
            "epoch": epoch, "train_loss": total_loss / max(1, count),
            "validation_macro_cross_entropy": val_metrics["macro_cross_entropy"],
            "validation_by_phase": {key: val_metrics[key] for key in val_metrics if key != "macro_cross_entropy"},
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        if record["validation_macro_cross_entropy"] < best_loss:
            best_loss, best_epoch, stale = record["validation_macro_cross_entropy"], epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 5:
                break
    model.load_state_dict(best_state)
    zbll_data_hash = hashlib.sha256(ZBLL_PATH.read_bytes()).hexdigest()
    hardware = {"python": platform.python_version(), "platform": platform.platform(), "torch": str(torch.__version__)}
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": 1, "feature_size": SOLVER_FEATURE_SIZE,
        "actions": len(SIMULATOR_MOVE_TOKENS), "history_length": 8, "phase_names": PHASES,
        "model_state": model.state_dict(), "seed": seed, "best_epoch": best_epoch,
        "sample_kind": "synthetic", "data_source": SOURCE,
        "cfop_data_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "dagger_data_sha256": hashlib.sha256(dagger_path.read_bytes()).hexdigest(),
        "zbll_data_source": ZBLL_SOURCE, "zbll_algorithms_sha256": zbll_data_hash,
        "hardware": hardware,
        "training_config": {"epochs": epochs, "batch_size": 512, "policy_loss": "cross_entropy",
                            "value_loss": "0.1 * smooth_l1 remaining moves", "balanced_by_phase": True},
    }, checkpoint_path)
    result = {
        "task": "phase-aware synthetic CFOP policy/value learning",
        "sample_kind": "synthetic", "data_source": SOURCE,
        "expert_rows_used": 0,
        "cfop_teacher_source": "@moishy/cfop 0.4.0 (MIT); deterministic reference targets only",
        "recovery_strategy": "DAgger-style synthetic relabeling from project Cross-MLP search states",
        "dagger_trajectories": len(dagger_records),
        "dagger_source": DAGGER_SOURCE,
        "zbll_algorithm_source": "MIT Alg-Trainer database; four AUF poses per known case, train/validation/test by pose",
        "device": str(device), "seed": seed, "best_epoch": best_epoch,
        "hardware": hardware,
        "train_trajectories": len({row["trajectory_id"] for row in train_rows}),
        "validation_trajectories": len({row["trajectory_id"] for row in validation_rows}),
        "train_actions": len(train_rows), "validation_actions": len(validation_rows),
        "zbll_held_out_auf_test_actions": len(test_rows),
        "train_actions_by_phase": dict(Counter(row["phase"] for row in train_rows)),
        "validation_actions_by_phase": dict(Counter(row["phase"] for row in validation_rows)),
        "validation": _score(model, validation_rows, device),
        "zbll_held_out_auf_test": _score(model, test_rows, device),
        "history": history, "checkpoint": str(checkpoint_path),
        "cfop_dataset": str(dataset_path),
        "dagger_dataset": str(dagger_path),
        "human_execution_or_recognition_time": "not measured",
    }
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--train-solves", type=int, default=160)
    parser.add_argument("--validation-solves", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args(argv)
    train(args.dataset, args.checkpoint, args.metrics, args.epochs, args.seed, args.device,
          args.train_solves, args.validation_solves, args.regenerate)


if __name__ == "__main__":
    main()
