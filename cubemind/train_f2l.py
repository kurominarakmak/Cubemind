"""Train and compare context-aware/state-only F2L action imitation policies."""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from cubemind.data.reco import COLOR_NAME, OUTPUT_PATH, prepare_f2l
from cubemind.environment.cube import SIMULATOR_MOVE_TOKENS
from cubemind.models.f2l import F2LPolicy
from cubemind.training.engine import choose_device, set_seed

ROOT = Path(__file__).resolve().parents[1]
FULL_CHECKPOINT = ROOT / "artifacts" / "reco-cfop-context.pt"
METRICS_PATH = ROOT / "artifacts" / "reco-cfop-eval.json"
HISTORY_LENGTH = 8
PHASES = ("CROSS", "CROSS+1", "F2L", "ZBLS")
STRATEGIES = ("standard F2L", "cross+1", "pseudo-slotting", "multi-pair annotation", "zbls")
MOVE_TO_ID = {move: index for index, move in enumerate(SIMULATOR_MOVE_TOKENS)}
COLOR_TO_ID = {name: color_id for color_id, name in COLOR_NAME.items()}
CONTEXT_SIZE = len(PHASES) + len(STRATEGIES) + 6 + 4
FULL_INPUT_SIZE = 54 * 6 + HISTORY_LENGTH * len(SIMULATOR_MOVE_TOKENS) + CONTEXT_SIZE


def _read_rows(data_dir):
    splits = {}
    for split in ("train", "validation", "test"):
        path = Path(data_dir) / (split + ".jsonl")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            raise ValueError("No {} F2L samples in {}".format(split, path))
        splits[split] = rows
    solver_splits = {}
    for split, rows in splits.items():
        for row in rows:
            if row.get("sample_kind") != "expert" or row.get("data_source") != "reco.nz":
                raise ValueError("F2L human data must remain separate and source-labeled")
            previous = solver_splits.setdefault(row["solver_id"], split)
            if previous != split:
                raise ValueError("Solver {} crosses held-out splits".format(row["solver_id"]))
            if row["move_id"] not in range(len(SIMULATOR_MOVE_TOKENS)) or row["next_expert_move"] != SIMULATOR_MOVE_TOKENS[row["move_id"]]:
                raise ValueError("Invalid expert action label in {}".format(row["solve_id"]))
    return splits


def _one_hot(values, classes):
    values = torch.tensor(values, dtype=torch.long)
    return torch.nn.functional.one_hot(values, classes).float()


def encode_features(rows, context=True):
    states = torch.tensor([row["cube_state"] for row in rows], dtype=torch.long)
    state_features = torch.nn.functional.one_hot(states, num_classes=6).float().flatten(1)
    if not context:
        return state_features
    history = torch.zeros((len(rows), HISTORY_LENGTH, len(SIMULATOR_MOVE_TOKENS)), dtype=torch.float32)
    for row_index, row in enumerate(rows):
        values = row.get("move_history", [])[-HISTORY_LENGTH:]
        for offset, move_id in enumerate(values, HISTORY_LENGTH - len(values)):
            history[row_index, offset, move_id] = 1.0
    phase_ids = [PHASES.index(row["phase"]) for row in rows]
    strategy_ids = [STRATEGIES.index(row["strategy_context"]) for row in rows]
    colors = [COLOR_TO_ID[row["cross_color"]] for row in rows]
    scalars = torch.tensor([[
        float(row["last_layer_edges_oriented"]),
        float(row["cross_complete"]),
        float(row["f2l_slot_count"]) / 4,
        min(float(row["move_count"]) / 60, 1.0),
    ] for row in rows], dtype=torch.float32)
    return torch.cat((state_features, history.flatten(1), _one_hot(phase_ids, len(PHASES)),
                      _one_hot(strategy_ids, len(STRATEGIES)), _one_hot(colors, 6), scalars), dim=1)


def score(model, rows, device, context=True):
    features = encode_features(rows, context).to(device)
    labels = torch.tensor([row["move_id"] for row in rows], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        logits = model(features)
        loss = torch.nn.functional.cross_entropy(logits, labels).item()
        ordering = logits.topk(5, dim=1).indices
        metrics = {"samples": len(rows), "cross_entropy": loss}
        for k in (1, 3, 5):
            metrics["top{}_accuracy".format(k)] = float((ordering[:, :k] == labels[:, None]).any(dim=1).float().mean())
        for phase in PHASES:
            indices = [i for i, row in enumerate(rows) if row["phase"] == phase]
            if indices:
                metrics["{}_top1_accuracy".format(phase.lower())] = float(
                    (ordering[indices, :1].squeeze(1) == labels[indices]).float().mean())
    return metrics


def _train_policy(train_rows, validation_rows, context, seed, epochs, device):
    set_seed(seed)
    model = F2LPolicy(FULL_INPUT_SIZE if context else 54 * 6, len(SIMULATOR_MOVE_TOKENS)).to(device)
    train_features = encode_features(train_rows, context)
    train_labels = torch.tensor([row["move_id"] for row in train_rows], dtype=torch.long)
    train_loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=128, shuffle=True,
                              num_workers=0, generator=torch.Generator().manual_seed(seed))
    validation_features = encode_features(validation_rows, context).to(device)
    validation_labels = torch.tensor([row["move_id"] for row in validation_rows], dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    best_loss, best_state, best_epoch, history = float("inf"), None, 0, []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(features), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(validation_features)
            val_loss = float(torch.nn.functional.cross_entropy(logits, validation_labels))
            val_accuracy = float((logits.argmax(dim=1) == validation_labels).float().mean())
        history.append({"epoch": epoch, "validation_loss": val_loss,
                        "validation_accuracy": val_accuracy, "seconds": time.perf_counter() - started})
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_epoch, history


def train(data_dir=OUTPUT_PATH, checkpoint_path=FULL_CHECKPOINT, metrics_path=METRICS_PATH,
          epochs=20, seed=17, device_name="auto", raw_path=Path("data/raw/reco_nz/cfop_3x3.jsonl")):
    data_dir = Path(data_dir)
    if not (data_dir / "train.jsonl").exists():
        raw = [json.loads(line) for line in Path(raw_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        prepare_f2l(raw, data_dir, seed)
    splits = _read_rows(data_dir)
    device = choose_device(device_name)
    torch.set_num_threads(2)
    model, best_epoch, history = _train_policy(splits["train"], splits["validation"], True, seed, epochs, device)
    state_only, state_epoch, state_history = _train_policy(
        splits["train"], splits["validation"], False, seed, epochs, device)
    full_metrics = score(model, splits["test"], device)
    state_metrics = score(state_only, splits["test"], device, context=False)
    label_counts = Counter(row["move_id"] for row in splits["train"])
    frequency_order = [move for move, _count in label_counts.most_common()]
    labels = [row["move_id"] for row in splits["test"]]
    baseline = {
        "top1_accuracy": sum(label == frequency_order[0] for label in labels) / len(labels),
        "top3_accuracy": sum(label in frequency_order[:3] for label in labels) / len(labels),
        "top5_accuracy": sum(label in frequency_order[:5] for label in labels) / len(labels),
    }
    sample = encode_features(splits["test"][:min(128, len(splits["test"]))], True).to(device)
    with torch.no_grad():
        for _ in range(5):
            model(sample)
        _sync(device)
        started = time.perf_counter()
        for _ in range(30):
            model(sample)
        _sync(device)
        latency = (time.perf_counter() - started) * 1000 / (30 * len(sample))
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": 1, "model_config": {"input_size": FULL_INPUT_SIZE,
               "actions": len(SIMULATOR_MOVE_TOKENS), "history_length": HISTORY_LENGTH},
               "model_state": model.state_dict(), "seed": seed, "best_epoch": best_epoch,
               "feature_schema": ["54 facelets", "last 8 actions", "F2L/ZBLS phase", "pair annotation",
                                  "selected cross color", "current EO", "move count", "slot count", "cross status"],
               "data_source": "reco.nz 3x3 CFOP reconstructions", "train_samples": len(splits["train"]),
               "sample_kind": "expert"}, checkpoint_path)
    result = {
        "task": "expert CFOP Cross/F2L/ZBLS next-action imitation", "sample_kind": "expert", "data_source": "reco.nz",
        "event_filter": "3x3 CFOP; OH excluded", "split_policy": "held out by solver_id",
        "device": str(device), "seed": seed, "train_solvers": len({r["solver_id"] for r in splits["train"]}),
        "test_solvers": len({r["solver_id"] for r in splits["test"]}),
        "test_solves": len({r["solve_id"] for r in splits["test"]}),
        "test_samples": len(splits["test"]), "full_context_model": full_metrics,
        "state_only_model": state_metrics, "frequency_baseline": baseline,
        "full_context_ms_per_sample": latency, "best_epoch": best_epoch,
        "state_only_best_epoch": state_epoch, "history": history,
        "state_only_history": state_history,
        "move_count_interpretation": "move count is an input feature; no move-count optimality claim is made",
        "human_timing": "not measured",
    }
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--checkpoint", type=Path, default=FULL_CHECKPOINT)
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/reco_nz/cfop_3x3.jsonl"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    train(args.data_dir, args.checkpoint, args.metrics, args.epochs, args.seed, args.device, args.raw)


if __name__ == "__main__":
    main()
