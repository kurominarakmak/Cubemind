"""Train and evaluate human-only, synthetic-only, and mixed neural ZBLS policies."""

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from cubemind.environment.cube import (
    COLORS, SIMULATOR_MOVE_TOKENS, SOLVED_STATE, apply_move, apply_sequence,
    cross_complete, f2l_slot_count, inverse_sequence, last_layer_edges_oriented,
    normalize_cross_frame, parse_moves, solved_pair_mask,
)
from cubemind.last_layer import _rotation_paths
from cubemind.models.zbls_policy import ZBLS_FEATURE_SIZE, ZBLSPolicyValue
from cubemind.training.engine import choose_device, set_seed
from cubemind.zbls import encode_zbls_features, goal_view, zbls_complete

ROOT = Path(__file__).resolve().parents[1]
ZBLS_DATABASE = ROOT / "data/algorithms/zbls_cases.json"
RECO_DIR = ROOT / "data/processed/reco_cfop"
ARTIFACT_DIR = ROOT / "artifacts/v2"
DATA_DIR = ROOT / "data/processed/zbls_v2"
MOVE_TO_ID = {move: index for index, move in enumerate(SIMULATOR_MOVE_TOKENS)}
COLOR_TO_ID = {name: index for index, name in enumerate(
    ("yellow", "red", "green", "white", "orange", "blue"))}
HUMAN_SOURCE = "expert:reco.nz:explicit-ZBLS-annotation"
SYNTHETIC_SOURCE = "algorithm_synthetic:CubeRoot-ZBLS"


def _rotated_color_map(cross_color):
    for rotation in _rotation_paths():
        centers = apply_sequence(SOLVED_STATE, rotation)
        if centers[COLORS["D"] * 9 + 4] == cross_color:
            return {face: centers[face * 9 + 4] for face in range(6)}
    raise ValueError("No proper cube rotation places selected color on Down")


def _recolor(state, mapping):
    return tuple(mapping[color] for color in state)


def _move_style(move):
    base = move.rstrip("'2")
    if base in "xyz":
        return "rotation"
    if base in "MES":
        return "slice"
    if base.islower():
        return "lowercase"
    if base.endswith("w"):
        return "wide"
    return "outer"


def _move_before_rotation(move, rotation):
    probe = apply_sequence(SOLVED_STATE, "R U F D B L")
    expected = apply_sequence(probe, list(rotation) + [move])
    matches = [candidate for candidate in SIMULATOR_MOVE_TOKENS
               if apply_sequence(probe, [candidate] + list(rotation)) == expected]
    if not matches:
        raise ValueError("Cannot express {} under rotation {}".format(move, rotation))
    same_style = [candidate for candidate in matches if _move_style(candidate) == _move_style(move)]
    return (same_style or matches)[0]


def strip_regrips(sequence):
    """Move whole-cube rotations to the end and omit them from the executable target."""
    result, orientation = [], []
    for move in parse_moves(sequence):
        if move[0] in "xyz":
            orientation.append(move)
        else:
            result.append(_move_before_rotation(move, orientation))
    return result


def _example_rows(start, moves, cross_color, protected_mask, target_pair, split,
                  source, trajectory_id):
    state, history, rows = tuple(start), [], []
    for index, move in enumerate(parse_moves(moves)):
        view = goal_view(state, cross_color)
        features = encode_zbls_features(
            state, history, protected_mask, target_pair, cross_color,
            len(history), "cpu",
        )
        rows.append({
            "features": features,
            "label": MOVE_TO_ID[move],
            "remaining_moves": len(parse_moves(moves)) - index - 1,
            "eo_label": float(last_layer_edges_oriented(view)),
            "split": split,
            "source": source,
            "trajectory_id": trajectory_id,
        })
        state = apply_move(state, move)
        history.append(MOVE_TO_ID[move])
    if not zbls_complete(state, cross_color):
        return []
    return rows


def build_synthetic(seed=17):
    database = json.loads(ZBLS_DATABASE.read_text(encoding="utf-8"))
    cases = database["cases"]
    rng = random.Random(seed)
    case_ids = [case["case_id"] for case in cases]
    rng.shuffle(case_ids)
    train_end, val_end = int(0.70 * len(cases)), int(0.85 * len(cases))
    split_by_case = {case_id: ("train" if i < train_end else
                               "validation" if i < val_end else "test")
                     for i, case_id in enumerate(case_ids)}
    cases_by_id = {case["case_id"]: case for case in cases}
    rows = {name: [] for name in ("train", "validation", "test")}
    seen = set()
    for case_id in sorted(cases_by_id):
        case, split = cases_by_id[case_id], split_by_case[case_id]
        start = apply_sequence(SOLVED_STATE, case["setup"])
        if not cross_complete(start) or f2l_slot_count(start) != 3:
            raise ValueError("Synthetic ZBLS case has an invalid start: {}".format(case_id))
        algorithm = parse_moves(case["algorithms"][0])
        if not zbls_complete(apply_sequence(start, algorithm), COLORS["D"]):
            raise ValueError("Synthetic ZBLS algorithm failed replay: {}".format(case_id))
        for cross_color in range(6):
            color_start = _recolor(start, _rotated_color_map(cross_color))
            for auf_index, auf in enumerate(([], ["U"], ["U2"], ["U'"])):
                yaw = ([], ["y"], ["y2"], ["y'"])[(int(hashlib.sha256(case_id.encode()).hexdigest()[:8], 16) + auf_index) % 4]
                transform = list(auf) + list(yaw)
                presented = apply_sequence(color_start, transform)
                target = strip_regrips(inverse_sequence(transform) + algorithm)
                view = goal_view(presented, cross_color)
                if not cross_complete(view):
                    raise ValueError("Color/orientation transform broke Cross: {}".format(case_id))
                mask = sum((1 << i) for i, solved in enumerate(solved_pair_mask(view)) if solved)
                if bin(mask).count("1") != 3:
                    raise ValueError("Color/orientation transform broke three-slot start: {}".format(case_id))
                target_pair = next(i for i in range(4) if not mask & (1 << i))
                key = (case_id, cross_color, auf_index, tuple(yaw))
                if key in seen:
                    raise ValueError("Duplicate synthetic ZBLS presentation")
                seen.add(key)
                trajectory = _example_rows(
                    presented, target, cross_color, mask, target_pair, split,
                    SYNTHETIC_SOURCE, "{}:{}:{}:{}".format(case_id, cross_color, auf_index, "".join(yaw)),
                )
                if not trajectory:
                    raise ValueError("Transformed ZBLS target failed replay: {}".format(case_id))
                rows[split].extend(trajectory)
    return rows, {"case_count": len(cases), "split_by_case": split_by_case,
                  "presentation_count": len(seen),
                  "split_case_counts": {name: len({row["trajectory_id"].split(":")[0] for row in items})
                                        for name, items in rows.items()}}


def _human_rows(split):
    path = RECO_DIR / (split + ".jsonl")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped = defaultdict(list)
    for row in records:
        if row.get("phase") == "ZBLS" and row.get("phase_annotation") == "4th pair/ZBLS":
            grouped[row["solve_id"]].append(row)
    examples, trajectories, rejected = [], 0, 0
    for solve_id, sequence in grouped.items():
        sequence.sort(key=lambda row: row["move_count"])
        chunks, chunk = [], []
        for row in sequence:
            if row.get("next_expert_move") not in MOVE_TO_ID:
                if chunk:
                    chunks.append(chunk)
                    chunk = []
                continue
            state, move = tuple(row["cube_state"]), row["next_expert_move"]
            if chunk and apply_move(tuple(chunk[-1]["cube_state"]), chunk[-1]["next_expert_move"]) != state:
                chunks.append(chunk)
                chunk = []
            chunk.append({**row, "cube_state": state, "next_expert_move": move})
        if chunk:
            chunks.append(chunk)
        for run_index, run in enumerate(chunks):
            start_index = next((i for i, row in enumerate(run)
                                if row.get("cross_color") in COLOR_TO_ID
                                and cross_complete(goal_view(row["cube_state"], COLOR_TO_ID[row["cross_color"]]))
                                and f2l_slot_count(goal_view(row["cube_state"], COLOR_TO_ID[row["cross_color"]])) == 3), None)
            if start_index is None:
                rejected += 1
                continue
            first = run[start_index]
            color_name = first.get("cross_color")
            color_id = COLOR_TO_ID.get(color_name)
            if color_id is None:
                rejected += 1
                continue
            start = first["cube_state"]
            view = goal_view(start, color_id)
            mask = sum((1 << i) for i, solved in enumerate(solved_pair_mask(view)) if solved)
            target_pair = next(i for i in range(4) if not mask & (1 << i))
            state = start
            teacher_moves = []
            reached = zbls_complete(state, color_id)
            for row in run[start_index:]:
                if reached:
                    break
                if tuple(row["cube_state"]) != state:
                    break
                move = row["next_expert_move"]
                teacher_moves.append(move)
                state = apply_move(state, move)
                reached = zbls_complete(state, color_id)
            if not reached:
                rejected += 1
                continue
            moves = strip_regrips(teacher_moves)
            trajectory = _example_rows(
                start, moves, color_id, mask, target_pair, split, HUMAN_SOURCE,
                "reco:{}:{}:{}".format(solve_id, first["move_count"], run_index),
            )
            if trajectory:
                examples.extend(trajectory)
                trajectories += 1
            else:
                rejected += 1
    return examples, {"trajectories": trajectories, "action_rows": len(examples), "rejected_runs": rejected}


def _metrics(model, rows, device, batch_size=512):
    if not rows:
        return {"samples": 0, "top1_accuracy": None, "top3_accuracy": None,
                "eo_accuracy": None, "value_mae": None}
    correct, top3, eo_correct, absolute_error, count = 0, 0, 0, 0.0, 0
    model.eval()
    with torch.inference_mode():
        for begin in range(0, len(rows), batch_size):
            batch = rows[begin:begin + batch_size]
            features = torch.stack([row["features"] for row in batch]).to(device)
            labels = torch.tensor([row["label"] for row in batch], dtype=torch.long, device=device)
            targets = torch.tensor([row["remaining_moves"] for row in batch], dtype=torch.float32, device=device)
            eo_targets = torch.tensor([row["eo_label"] for row in batch], dtype=torch.float32, device=device)
            logits, values, eo_logits = model(features)
            top = logits.topk(3, dim=1).indices
            correct += int((top[:, 0] == labels).sum())
            top3 += int((top == labels[:, None]).any(dim=1).sum())
            eo_correct += int(((eo_logits >= 0) == (eo_targets >= 0.5)).sum())
            absolute_error += float((values - targets).abs().sum())
            count += len(batch)
    return {"samples": count, "top1_accuracy": correct / count,
            "top3_accuracy": top3 / count, "eo_accuracy": eo_correct / count,
            "value_mae": absolute_error / count}


def _source_balanced_loader(rows, batch_size, seed):
    counts = Counter(row["source"] for row in rows)
    weights = torch.tensor([1.0 / counts[row["source"]] for row in rows], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(rows), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    features = torch.stack([row["features"] for row in rows])
    labels = torch.tensor([row["label"] for row in rows], dtype=torch.long)
    values = torch.tensor([row["remaining_moves"] for row in rows], dtype=torch.float32)
    eo = torch.tensor([row["eo_label"] for row in rows], dtype=torch.float32)
    return DataLoader(TensorDataset(features, labels, values, eo), batch_size=batch_size,
                      sampler=sampler, num_workers=0)


def _fit(kind, train_rows, validation_rows, seed, epochs, device):
    set_seed(seed)
    model = ZBLSPolicyValue().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    loader = _source_balanced_loader(train_rows, 256, seed)
    best_loss, best_state, best_epoch, history = float("inf"), None, 0, []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        totals = [0.0, 0.0, 0.0, 0]
        for features, labels, values, eo in loader:
            features, labels = features.to(device), labels.to(device)
            values, eo = values.to(device), eo.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, value_output, eo_logits = model(features)
            loss = (F.cross_entropy(logits, labels)
                    + 0.05 * F.smooth_l1_loss(value_output, values)
                    + 0.1 * F.binary_cross_entropy_with_logits(eo_logits, eo))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = len(labels)
            totals[0] += float(loss.detach()) * n
            totals[1] += int((logits.argmax(dim=1) == labels).sum())
            totals[2] += float((value_output.detach() - values).abs().sum())
            totals[3] += n
        model.eval()
        source_losses = []
        with torch.inference_mode():
            for source in sorted({row["source"] for row in validation_rows}):
                rows = [row for row in validation_rows if row["source"] == source]
                loss_sum, count = 0.0, 0
                for begin in range(0, len(rows), 512):
                    batch = rows[begin:begin + 512]
                    features = torch.stack([row["features"] for row in batch]).to(device)
                    labels = torch.tensor([row["label"] for row in batch], dtype=torch.long, device=device)
                    values = torch.tensor([row["remaining_moves"] for row in batch], dtype=torch.float32, device=device)
                    eo = torch.tensor([row["eo_label"] for row in batch], dtype=torch.float32, device=device)
                    logits, value_output, eo_logits = model(features)
                    loss_sum += float((F.cross_entropy(logits, labels, reduction="sum")
                                      + 0.05 * F.smooth_l1_loss(value_output, values, reduction="sum")
                                      + 0.1 * F.binary_cross_entropy_with_logits(eo_logits, eo, reduction="sum")))
                    count += len(batch)
                source_losses.append(loss_sum / count)
        validation_loss = sum(source_losses) / max(1, len(source_losses))
        record = {"epoch": epoch, "train_loss": totals[0] / totals[3],
                  "train_top1": totals[1] / totals[3], "train_value_mae": totals[2] / totals[3],
                  "validation_balanced_loss": validation_loss,
                  "seconds": time.perf_counter() - started}
        history.append(record)
        print(json.dumps({"model": kind, **record}), flush=True)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
    model.load_state_dict(best_state)
    return model, best_epoch, history, best_loss


def train_all(seed=17, epochs=10, device_name="auto"):
    torch.set_num_threads(2)
    device = choose_device(device_name)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    synthetic, synthetic_manifest = build_synthetic(seed)
    human = {split: _human_rows(split) for split in ("train", "validation", "test")}
    if not human["train"][0] or not human["validation"][0]:
        raise ValueError("Explicit human ZBLS demonstrations are too sparse to train and validate")
    configurations = {
        "synthetic": (synthetic["train"], synthetic["validation"], {
            "synthetic_test": synthetic["test"]}),
        "human": (human["train"][0], human["validation"][0], {
            "human_test": human["test"][0]}),
        "mixed": (synthetic["train"] + human["train"][0],
                  synthetic["validation"] + human["validation"][0], {
                      "synthetic_test": synthetic["test"], "human_test": human["test"][0]}),
    }
    report = {
        "task": "goal-conditioned neural ZBLS action policy",
        "training_from_scratch": True,
        "initialization": "PyTorch random initialization; no pretrained encoder",
        "device": str(device), "seed": seed, "epochs_max": epochs,
        "split_policy": {
            "synthetic": "case-disjoint 70/15/15 split by canonical CubeRoot case_id",
            "human": "existing solver-disjoint reco.nz train/validation/test partitions",
        },
        "synthetic": synthetic_manifest,
        "human": {split: manifest for split, (_rows, manifest) in human.items()},
        "models": {},
    }
    for offset, (kind, (train_rows, val_rows, tests)) in enumerate(configurations.items()):
        if not train_rows or not val_rows:
            raise ValueError("{} ZBLS model lacks train or validation examples".format(kind))
        model, best_epoch, history, val_loss = _fit(kind, train_rows, val_rows,
                                                     seed + offset, epochs, device)
        test_metrics = {name: _metrics(model, rows, device) for name, rows in tests.items()}
        checkpoint_path = ARTIFACT_DIR / ("zbls-{}-policy.pt".format(kind))
        torch.save({
            "task": "neural_zbls_policy", "feature_size": ZBLS_FEATURE_SIZE,
            "actions": len(SIMULATOR_MOVE_TOKENS), "model_state": model.state_dict(),
            "seed": seed + offset, "best_epoch": best_epoch,
            "training_sources": sorted({row["source"] for row in train_rows}),
            "training_sample_counts": dict(Counter(row["source"] for row in train_rows)),
            "initialization": "random", "history": history,
        }, checkpoint_path)
        report["models"][kind] = {
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "training_rows": len(train_rows),
            "training_rows_by_source": dict(Counter(row["source"] for row in train_rows)),
            "validation_rows": len(val_rows), "best_epoch": best_epoch,
            "validation_loss": val_loss, "history": history, "test": test_metrics,
        }
        print(json.dumps({"model": kind, "best_epoch": best_epoch,
                          "test": test_metrics}, indent=2), flush=True)
    report["algorithm_reference"] = {
        "cases": synthetic_manifest["case_count"],
        "role": "offline teacher and exact replay reference only; never used by neural inference",
    }
    report_path = ARTIFACT_DIR / "zbls-training-evaluation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1, "report": str(report_path.relative_to(ROOT)),
        "synthetic_source": SYNTHETIC_SOURCE, "human_source": HUMAN_SOURCE,
        "raw_reconstructions_saved": False,
        "training_split_counts": {name: len(rows) for name, rows in synthetic.items()},
        "checkpoint_hashes": {name: value["checkpoint_sha256"] for name, value in report["models"].items()},
    }
    (ARTIFACT_DIR / "zbls-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("epochs must be positive")
    train_all(args.seed, args.epochs, args.device)


if __name__ == "__main__":
    main()
