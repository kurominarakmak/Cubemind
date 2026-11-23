"""Train/evaluate a random-init ZBLL classifier on simulator-generated cases."""

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from cubemind.environment.cube import SOLVED_STATE, apply_sequence
from cubemind.last_layer import (
    ZBLL_PATH,
    _rotation_paths,
    recognize_zbll_case,
    verify_zbll_database,
    zbll_case_state,
)
from cubemind.models.zbll import ZBLLClassifier
from cubemind.training.engine import choose_device, set_seed

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "zbll_algorithm_synthetic"
CHECKPOINT = ROOT / "artifacts" / "zbll-classifier.pt"
METRICS = ROOT / "artifacts" / "zbll-classifier-eval.json"


def prepare_data(output=DATA_DIR, seed=17):
    """Create disjoint pose splits; all 493 labels occur in every split."""
    verification = verify_zbll_database()
    database = json.loads(ZBLL_PATH.read_text(encoding="utf-8"))
    rotations = _rotation_paths()
    transform_ids = list(range(4 * len(rotations)))
    random.Random(seed).shuffle(transform_ids)
    train_ids = set(transform_ids[:int(len(transform_ids) * 0.8)])
    validation_ids = set(transform_ids[int(len(transform_ids) * 0.8):int(len(transform_ids) * 0.9)])
    splits = {"train": [], "validation": [], "test": []}
    seen = set()
    for label, case in enumerate(database["cases"]):
        base = zbll_case_state(case["case_id"])
        for auf_index in range(4):
            auf = ["", "U", "U2", "U'"][auf_index]
            for rotation_index, rotation in enumerate(rotations):
                state = tuple(apply_sequence(base, ([auf] if auf else []) + list(rotation)))
                state_key = (case["case_id"], state)
                if state_key in seen:
                    continue
                seen.add(state_key)
                transform_id = auf_index * len(rotations) + rotation_index
                split = "train" if transform_id in train_ids else (
                    "validation" if transform_id in validation_ids else "test")
                splits[split].append({
                    "cube_state": list(state), "case_id": case["case_id"], "label_id": label,
                    "sample_kind": "algorithm_synthetic", "data_source": "MIT Alg-Trainer",
                    "presentation_id": transform_id,
                })
    if any(not rows for rows in splits.values()):
        raise AssertionError("Each ZBLL data split must have samples")
    for rows in splits.values():
        if {row["case_id"] for row in rows} != {case["case_id"] for case in database["cases"]}:
            raise AssertionError("Every split must cover all ZBLL labels")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1, "seed": seed, "source": database["source"],
        "generator": "inverse primary algorithm, then AUF and one of 24 cube orientations",
        "split_policy": "held out presentation transforms; same labels occur in all splits",
        "exact_state_deduplication": True, "verification": verification,
        "splits": {},
    }
    all_states = {}
    for name, rows in splits.items():
        path = output / (name + ".jsonl")
        path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
        all_states[name] = {tuple(row["cube_state"]) for row in rows}
        manifest["splits"][name] = {
            "samples": len(rows), "cases": len({row["case_id"] for row in rows}),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    if any(all_states[a] & all_states[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise AssertionError("Exact cube states must not cross ZBLL splits")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return splits, manifest


def _score(model, rows, device, batch_size=256):
    states = torch.tensor([row["cube_state"] for row in rows], dtype=torch.long)
    labels = torch.tensor([row["label_id"] for row in rows], dtype=torch.long)
    loader = DataLoader(TensorDataset(states, labels), batch_size=batch_size, shuffle=False)
    totals = np.zeros(3, dtype=np.int64)
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch_states, batch_labels in loader:
            batch_states, batch_labels = batch_states.to(device), batch_labels.to(device)
            logits = model(batch_states)
            total_loss += float(torch.nn.functional.cross_entropy(logits, batch_labels, reduction="sum"))
            order = logits.topk(5, dim=1).indices
            for i, k in enumerate((1, 3, 5)):
                totals[i] += int((order[:, :k] == batch_labels[:, None]).any(dim=1).sum())
    return {"samples": len(rows), "cross_entropy": total_loss / len(rows),
            "top1_accuracy": totals[0] / len(rows), "top3_accuracy": totals[1] / len(rows),
            "top5_accuracy": totals[2] / len(rows)}


def train(output=CHECKPOINT, metrics_path=METRICS, data_dir=DATA_DIR, epochs=15, seed=17, device_name="auto"):
    set_seed(seed)
    device = choose_device(device_name)
    torch.set_num_threads(2)
    splits, manifest = prepare_data(data_dir, seed)
    tensors = {name: TensorDataset(
        torch.tensor([row["cube_state"] for row in rows], dtype=torch.long),
        torch.tensor([row["label_id"] for row in rows], dtype=torch.long))
        for name, rows in splits.items()}
    train_loader = DataLoader(tensors["train"], batch_size=256, shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(seed))
    validation_loader = DataLoader(tensors["validation"], batch_size=512, shuffle=False, num_workers=0)
    model = ZBLLClassifier(classes=len(json.loads(ZBLL_PATH.read_text())["cases"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    best_loss, best_state, best_epoch = float("inf"), None, 0
    history = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        train_loss = train_correct = train_count = 0
        for batch_states, batch_labels in train_loader:
            batch_states, batch_labels = batch_states.to(device), batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_states)
            loss = torch.nn.functional.cross_entropy(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.detach()) * batch_labels.size(0)
            train_correct += int((logits.argmax(dim=1) == batch_labels).sum())
            train_count += batch_labels.size(0)
        model.eval()
        val_loss = val_correct = val_count = 0
        with torch.no_grad():
            for batch_states, batch_labels in validation_loader:
                batch_states, batch_labels = batch_states.to(device), batch_labels.to(device)
                logits = model(batch_states)
                val_loss += float(torch.nn.functional.cross_entropy(logits, batch_labels, reduction="sum"))
                val_correct += int((logits.argmax(dim=1) == batch_labels).sum())
                val_count += batch_labels.size(0)
        metric = {"epoch": epoch, "train_loss": train_loss / train_count,
                  "train_accuracy": train_correct / train_count,
                  "validation_loss": val_loss / val_count, "validation_accuracy": val_correct / val_count,
                  "seconds": time.perf_counter() - started}
        history.append(metric)
        print(json.dumps(metric))
        if val_loss < best_loss:
            best_loss, best_state, best_epoch = val_loss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, epoch
    model.load_state_dict(best_state)
    output, metrics_path = Path(output), Path(metrics_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"schema_version": 1, "model_config": {"classes": 493}, "model_state": model.state_dict(),
                  "seed": seed, "best_epoch": best_epoch, "data_source": manifest["source"],
                  "training_data_kind": "algorithm_synthetic", "history": history}
    torch.save(checkpoint, output)
    test_rows = splits["test"]
    test_metrics = _score(model, test_rows, device)
    exact_count = sum(recognize_zbll_case(row["cube_state"]) == row["case_id"] for row in test_rows)
    counts = Counter(row["label_id"] for row in splits["train"])
    majority_id = max(counts, key=counts.get)
    frequency_accuracy = sum(row["label_id"] == majority_id for row in test_rows) / len(test_rows)
    sample = torch.tensor([row["cube_state"] for row in test_rows[:min(128, len(test_rows))]], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(5):
            model(sample)
        _sync(device)
        started = time.perf_counter()
        for _ in range(30):
            model(sample)
        _sync(device)
        neural_latency = (time.perf_counter() - started) * 1000 / (30 * len(sample))
    exact_started = time.perf_counter()
    for row in test_rows[:min(500, len(test_rows))]:
        recognize_zbll_case(row["cube_state"])
    exact_latency = (time.perf_counter() - exact_started) * 1000 / min(500, len(test_rows))
    cpu_model = ZBLLClassifier(classes=len(json.loads(ZBLL_PATH.read_text())["cases"]))
    cpu_model.load_state_dict(model.state_dict())
    cpu_rows = test_rows[:min(128, len(test_rows))]
    cpu_sample = torch.tensor([row["cube_state"] for row in cpu_rows], dtype=torch.long)
    with torch.no_grad():
        for _ in range(5):
            cpu_model(cpu_sample)
        started = time.perf_counter()
        for _ in range(30):
            cpu_model(cpu_sample)
        neural_cpu_latency = (time.perf_counter() - started) * 1000 / (30 * len(cpu_sample))
    result = {
        "task": "493-way ZBLL+PLL case classification", "sample_kind": "algorithm_synthetic",
        "device": str(device), "seed": seed, "best_epoch": best_epoch,
        "split_policy": manifest["split_policy"], "held_out_cases": False,
        "test": test_metrics,
        "exact_deterministic": {"accuracy": exact_count / len(test_rows), "samples": len(test_rows),
                                 "ms_per_sample": exact_latency},
        "training_label_frequency_baseline_top1": frequency_accuracy,
        "neural_device_ms_per_sample": neural_latency, "neural_cpu_ms_per_sample": neural_cpu_latency,
        "latency_batch_size": min(128, len(test_rows)), "history": history,
    }
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
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    args = parser.parse_args(argv)
    train(args.checkpoint, args.metrics, args.data_dir, args.epochs, args.seed, args.device)


if __name__ == "__main__":
    main()
