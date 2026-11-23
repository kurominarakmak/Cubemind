import json
import hashlib
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cubemind.environment.cube import MOVE_TOKENS, apply_move, cross_complete
from cubemind.models import build_model
from cubemind.models.common import encode_history


class CrossDataset(Dataset):
    def __init__(self, path):
        self.path = Path(path)
        self.rows = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for field in ("cube_state", "move_history", "phase", "next_expert_move", "move_id",
                          "solver_id", "solve_id", "scramble_id", "data_source"):
                if field not in row:
                    raise ValueError("{}:{} missing {}".format(self.path, line_number, field))
            if row["phase"] != "CROSS" or len(row["cube_state"]) != 54:
                raise ValueError("{}:{} is not a structured Cross sample".format(self.path, line_number))
            if row["move_id"] not in range(18) or row["next_expert_move"] != MOVE_TOKENS[row["move_id"]]:
                raise ValueError("{}:{} has an invalid move label".format(self.path, line_number))
            self.rows.append(row)
        if not self.rows:
            raise ValueError("No samples in {}".format(self.path))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return torch.tensor(row["cube_state"], dtype=torch.long), row["move_history"], int(row["move_id"])


def collate_cross(batch):
    states, histories, labels = zip(*batch)
    return torch.stack(states), list(histories), torch.tensor(labels, dtype=torch.long)


def load_splits(train_path, validation_path, test_path=None):
    paths = {"train": train_path, "validation": validation_path}
    if test_path:
        paths["test"] = test_path
    datasets = {name: CrossDataset(path) for name, path in paths.items()}
    seen_solve = {}
    seen_scramble = {}
    for name, dataset in datasets.items():
        for row in dataset.rows:
            for key, seen in (("solve_id", seen_solve), ("scramble_id", seen_scramble)):
                group = row[key]
                if group in seen and seen[group] != name:
                    raise ValueError("{} appears in both {} and {} splits".format(group, seen[group], name))
                seen[group] = name
    return datasets


def choose_device(requested="auto"):
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _batch_to_device(batch, max_history, device):
    states, histories, labels = batch
    return (states.to(device), encode_history(histories, max_history, device), labels.to(device))


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def run_epoch(model, loader, optimizer, max_history, device, grad_clip=None):
    training = optimizer is not None
    model.train(training)
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = correct = count = 0
    for batch in loader:
        states, histories, labels = _batch_to_device(batch, max_history, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(states, histories)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                optimizer.step()
        total_loss += float(loss.detach()) * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum())
        count += labels.size(0)
    return {"loss": total_loss / count, "accuracy": correct / count, "samples": count}


def train_model(config, config_path):
    seed = int(config["train"].get("seed", 17))
    set_seed(seed)
    device = choose_device(config["train"].get("device", "auto"))
    torch.set_num_threads(int(config["train"].get("cpu_threads", 2)))
    datasets = load_splits(config["data"]["train"], config["data"]["validation"])
    model_config = config["model"]
    max_history = int(model_config.get("max_history", 8)) if model_config.get("use_history", False) else 0
    batch_size = int(config["train"].get("batch_size", 64))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True,
                              collate_fn=collate_cross, num_workers=0, generator=generator)
    val_loader = DataLoader(datasets["validation"], batch_size=batch_size, shuffle=False,
                            collate_fn=collate_cross, num_workers=0)
    model = build_model(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["train"].get("learning_rate", 0.001)),
                                  weight_decay=float(config["train"].get("weight_decay", 0.0001)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=float(config["train"].get("lr_factor", 0.5)),
        patience=int(config["train"].get("lr_patience", 2)), min_lr=float(config["train"].get("min_learning_rate", 1e-5)))
    output = Path(config["train"].get("checkpoint", "artifacts/{}-best.pt".format(config["name"])))
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(config["train"].get("metrics", str(output.with_suffix(".jsonl"))))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    patience = int(config["train"].get("early_stopping_patience", 5))
    epochs = int(config["train"].get("epochs", 20))
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    with metrics_path.open("w", encoding="utf-8") as log:
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            train_metrics = run_epoch(model, train_loader, optimizer, max_history, device,
                                      config["train"].get("gradient_clip", 1.0))
            validation_metrics = run_epoch(model, val_loader, None, max_history, device)
            scheduler.step(validation_metrics["loss"])
            metrics = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "seconds": time.perf_counter() - started,
            }
            history.append(metrics)
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            print(json.dumps(metrics))
            if validation_metrics["loss"] < best_loss:
                best_loss = validation_metrics["loss"]
                best_epoch = epoch
                stale = 0
                checkpoint = {
                    "schema_version": 1,
                    "name": config["name"],
                    "model_config": model_config,
                    "model_state": model.state_dict(),
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_loss,
                    "best_validation_accuracy": validation_metrics["accuracy"],
                    "config_path": str(config_path),
                    "dataset": {
                        "train_path": str(datasets["train"].path),
                        "train_sha256": hashlib.sha256(datasets["train"].path.read_bytes()).hexdigest(),
                        "train_samples": len(datasets["train"]),
                        "validation_path": str(datasets["validation"].path),
                        "validation_sha256": hashlib.sha256(datasets["validation"].path.read_bytes()).hexdigest(),
                        "validation_samples": len(datasets["validation"]),
                        "sample_kinds": sorted({row.get("sample_kind", "unknown") for row in datasets["train"].rows}),
                    },
                    "history": history,
                }
                torch.save(checkpoint, output)
            else:
                stale += 1
                if stale >= patience:
                    print("Early stopping at epoch {} (best epoch {})".format(epoch, best_epoch))
                    break
    return output


def model_from_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = build_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def score_model(model, rows, max_history, device, batch_size=128):
    dataset = _RowsDataset(rows)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_cross, num_workers=0)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    top = [0, 0, 0]
    count = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            states, histories, labels = _batch_to_device(batch, max_history, device)
            logits = model(states, histories)
            total_loss += float(criterion(logits, labels))
            ordering = logits.topk(5, dim=1).indices
            for index, k in enumerate((1, 3, 5)):
                top[index] += int((ordering[:, :k] == labels[:, None]).any(dim=1).sum())
            count += labels.size(0)
    latency_batch = rows[:min(batch_size, len(rows))]
    latency_states = torch.tensor([row["cube_state"] for row in latency_batch], dtype=torch.long, device=device)
    latency_history = encode_history([row["move_history"] for row in latency_batch], max_history, device)
    with torch.no_grad():
        for _ in range(5):
            model(latency_states, latency_history)
        _sync(device)
        started = time.perf_counter()
        repeats = 20
        for _ in range(repeats):
            model(latency_states, latency_history)
        _sync(device)
        latency = (time.perf_counter() - started) * 1000 / (repeats * len(latency_batch))
    return {
        "samples": count,
        "cross_entropy": total_loss / count,
        "top1_accuracy": top[0] / count,
        "top3_accuracy": top[1] / count,
        "top5_accuracy": top[2] / count,
        "batched_forward_ms_per_sample": latency,
        "latency_batch_size": len(latency_batch),
        "latency_warmup_batches": 5,
        "latency_measurement_batches": 20,
    }


class _RowsDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return torch.tensor(row["cube_state"], dtype=torch.long), row["move_history"], int(row["move_id"])


def rollout(model, initial_state, max_history, device, max_moves=20):
    state = tuple(initial_state)
    history = []
    moves = []
    model.eval()
    for _ in range(max_moves):
        if cross_complete(state):
            break
        states = torch.tensor([state], dtype=torch.long, device=device)
        encoded = encode_history([history], max_history, device)
        with torch.no_grad():
            move_id = int(model(states, encoded).argmax(dim=1).item())
        move = MOVE_TOKENS[move_id]
        state = apply_move(state, move)
        history.append(move_id)
        moves.append(move)
    return moves, state


def beam_search_cross(model, initial_state, max_history, device, beam_width=64, max_moves=10):
    """Search Cross paths in descending accumulated policy probability."""
    if beam_width < 1 or max_moves < 0 or max_history < 0:
        raise ValueError("beam_width must be positive; max_moves and max_history must be non-negative")
    start = tuple(initial_state)
    if cross_complete(start):
        return [], start, 0, True
    frontier = [(start, (), 0.0)]
    nodes_scored = 0
    model.eval()
    for _depth in range(1, max_moves + 1):
        states = torch.tensor([node[0] for node in frontier], dtype=torch.long, device=device)
        histories = [node[1][-max_history:] if max_history else [] for node in frontier]
        encoded = encode_history(histories, max_history, device)
        with torch.no_grad():
            scores = torch.log_softmax(model(states, encoded), dim=1).detach().cpu().tolist()
        nodes_scored += len(frontier)
        candidates = {}
        solutions = []
        for (state, path, score), move_scores in zip(frontier, scores):
            previous_face = MOVE_TOKENS[path[-1]][0] if path else None
            for move_id in sorted(range(len(MOVE_TOKENS)), key=lambda item: (-move_scores[item], item)):
                move = MOVE_TOKENS[move_id]
                if move[0] == previous_face:
                    continue  # Consecutive turns of one face can always be combined.
                next_state = apply_move(state, move)
                next_path = path + (move_id,)
                next_score = score + move_scores[move_id]
                if cross_complete(next_state):
                    solutions.append((next_score, next_path, next_state))
                    continue
                history_key = next_path[-max_history:] if max_history else ()
                key = (next_state, move[0], history_key)
                previous = candidates.get(key)
                if previous is None or next_score > previous[2]:
                    candidates[key] = (next_state, next_path, next_score)
        if solutions:
            _score, path, state = min(solutions, key=lambda item: (-item[0], item[1]))
            return [MOVE_TOKENS[move_id] for move_id in path], state, nodes_scored, True
        # shortcut: fixed-width pruning can discard all solving paths, upgrade if validation warrants wider or heuristic search.
        frontier = sorted(candidates.values(), key=lambda item: (-item[2], item[1]))[:beam_width]
        if not frontier:
            break
    if frontier:
        state, path, _score = frontier[0]
    else:
        state, path = start, ()
    return [MOVE_TOKENS[move_id] for move_id in path], state, nodes_scored, False
