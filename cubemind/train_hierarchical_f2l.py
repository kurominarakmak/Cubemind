"""Train pair-conditioned F2L policy, value, and pair-selection heads."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from cubemind.environment.cube import (
    COLOR_NAMES, SIMULATOR_MOVE_TOKENS, SOLVED_STATE, apply_move,
    apply_sequence, cross_complete, cross_frame_move, f2l_complete, inverse_move,
    normalize_cross_frame, parse_moves,
    solved_pair_mask,
)
from cubemind.models.f2l import F2LPolicy
from cubemind.models.hierarchical_f2l import (
    HIERARCHICAL_F2L_FEATURE_SIZE, HISTORY_LENGTH, HierarchicalF2LPolicy,
    encode_f2l_features,
)
from cubemind.neural_solver import teacher_request
from cubemind.train_f2l import encode_features as encode_expert_features
from cubemind.train_f2l import FULL_INPUT_SIZE as EXPERT_FEATURE_SIZE
from cubemind.training.engine import choose_device, set_seed

ROOT = Path(__file__).resolve().parents[1]
RECO = ROOT / "data" / "processed" / "reco_cfop"
SYNTHETIC = ROOT / "data" / "processed" / "neural_solver"
DATA_DIR = ROOT / "data" / "processed" / "hierarchical_f2l"
CHECKPOINT = ROOT / "artifacts" / "hierarchical-f2l.pt"
METRICS = ROOT / "artifacts" / "hierarchical-f2l-training.json"
SOURCE_TEACHER = "synthetic:cfop-teacher:@moishy/cfop@0.4.0"
SOURCE_DAGGER = "synthetic:dagger-reference-after-project-cross-mlp"
SOURCE_RECOVERY = "synthetic:on-policy-f2l-recovery:@moishy/cfop@0.4.0"
SOURCE_HUMAN = "reco.nz 3x3 CFOP reconstructions"
MOVE_TO_ID = {move: index for index, move in enumerate(SIMULATOR_MOVE_TOKENS)}


def _mask_bits(state):
    return sum((1 << index) for index, solved in enumerate(solved_pair_mask(state)) if solved)


def _popcount(mask):
    return bin(int(mask)).count("1")


def label_trajectory(transitions, source, split, trajectory_id):
    """Derive pair goals only from verified slot changes in a replayed path."""
    action_rows, selector_rows = [], []
    if not transitions or not cross_complete(transitions[0]["state"]):
        return action_rows, selector_rows
    protected = _mask_bits(transitions[0]["state"])
    pending = []
    for transition in transitions:
        state = tuple(transition["state"])
        move = transition["move"]
        child = apply_move(state, move)
        pending.append(transition)
        current_mask = _mask_bits(child)
        added = current_mask & ~protected
        preserved = (current_mask & protected) == protected and cross_complete(child)
        if not added or not preserved:
            continue
        if _popcount(added) == 1:
            target = (added & -added).bit_length() - 1
            selector_rows.append({
                "state": list(pending[0]["state"]),
                "history": list(pending[0]["history"]),
                "protected_mask": protected,
                "target_pair": target,
                "move_count": pending[0]["move_count"],
                "sample_kind": pending[0]["sample_kind"],
                "data_source": source,
                "split": split,
                "trajectory_id": trajectory_id,
            })
            final_index = len(pending) - 1
            for index, item in enumerate(pending):
                action_rows.append({
                    "state": list(item["state"]), "history": list(item["history"]),
                    "protected_mask": protected, "target_pair": target,
                    "move_id": MOVE_TO_ID[item["move"]],
                    "remaining_moves": final_index - index + 1,
                    "move_count": item["move_count"],
                    "sample_kind": item["sample_kind"], "data_source": source,
                    "split": split, "trajectory_id": trajectory_id,
                    "expert_row": item.get("expert_row"),
                })
        protected = current_mask
        pending = []
    return action_rows, selector_rows


def _synthetic_records():
    records = []
    for name, expected_source in (("cfop_teacher.jsonl", SOURCE_TEACHER),
                                  ("dagger_teacher.jsonl", SOURCE_DAGGER)):
        path = SYNTHETIC / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("sample_kind") != "synthetic" or row.get("data_source") != expected_source:
                raise ValueError("Synthetic F2L source labels do not match {}".format(path))
            records.append(row)
    if not records:
        raise ValueError("No verified synthetic CFOP trajectories were found")
    return records


def synthetic_trajectory(record):
    state = apply_sequence(SOLVED_STATE, parse_moves(record["scramble"]))
    cross_moves = parse_moves(record["phases"].get("CROSS", []))
    state = apply_sequence(state, cross_moves)
    if not cross_complete(state):
        raise ValueError("Synthetic teacher Cross failed replay for {}".format(record["scramble_id"]))
    if record.get("prefix_history") is not None:
        history = list(record["prefix_history"])[-HISTORY_LENGTH:]
        move_count = len(record["prefix_history"])
    else:
        history = [MOVE_TO_ID[move] for move in cross_moves][-HISTORY_LENGTH:]
        move_count = len(cross_moves)
    transitions = []
    for index, move in enumerate(parse_moves(record["phases"].get("F2L", []))):
        transitions.append({
            "state": state, "history": list(history), "move": move,
            "move_count": move_count + index, "sample_kind": "synthetic",
        })
        state = apply_move(state, move)
        history = (history + [MOVE_TO_ID[move]])[-HISTORY_LENGTH:]
    if not f2l_complete(state):
        raise ValueError("Synthetic teacher F2L failed replay for {}".format(record["scramble_id"]))
    return transitions


def _human_splits():
    result, solver_split = {}, {}
    for split in ("train", "validation", "test"):
        path = RECO / (split + ".jsonl")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            if row.get("sample_kind") != "expert" or row.get("data_source") != "reco.nz":
                raise ValueError("Human F2L rows must remain source-labeled reco.nz data")
            previous = solver_split.setdefault(row["solver_id"], split)
            if previous != split:
                raise ValueError("A human solver occurs in multiple dataset splits")
        result[split] = rows
    return result


def human_trajectories(rows, split):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("phase") != "F2L" or row.get("cross_color") not in COLOR_NAMES:
            continue
        raw_state = tuple(row["cube_state"])
        centers = tuple(raw_state[face * 9 + 4] for face in range(6))
        color_id = COLOR_NAMES.index(row["cross_color"])
        state = normalize_cross_frame(raw_state, color_id)
        move = row.get("next_expert_move")
        if move not in MOVE_TO_ID or MOVE_TO_ID[move] != row.get("move_id"):
            continue
        if move[0] not in "URFDLB":
            continue
        mapped_move = cross_frame_move(move, centers, color_id)
        grouped[row["solve_id"]].append({
            **row, "cube_state": list(state), "move_history": [],
            "next_expert_move": mapped_move, "move_id": MOVE_TO_ID[mapped_move],
            "raw_expert_row": row,
        })
    action_rows, selector_rows = [], []
    for solve_id, samples in grouped.items():
        samples.sort(key=lambda row: row["move_count"])
        chunk = []
        for row in samples:
            state = tuple(row["cube_state"])
            if chunk:
                previous = chunk[-1]
                expected = apply_move(tuple(previous["cube_state"]), previous["next_expert_move"])
                contiguous = tuple(row["cube_state"]) == expected
                if not contiguous:
                    transitions = []
                    trace_history = []
                    for item in chunk:
                        transitions.append({
                            "state": tuple(item["cube_state"]), "history": trace_history[-HISTORY_LENGTH:],
                            "move": item["next_expert_move"], "move_count": item["move_count"],
                            "sample_kind": "expert", "expert_row": item["raw_expert_row"],
                        })
                        trace_history.append(item["move_id"])
                    actions, selectors = label_trajectory(
                        transitions, SOURCE_HUMAN, split, "reco:{}:{}".format(solve_id, chunk[0]["move_count"]),
                    )
                    action_rows.extend(actions)
                    selector_rows.extend(selectors)
                    chunk = []
            chunk.append(row)
        if chunk:
            transitions = []
            trace_history = []
            for item in chunk:
                transitions.append({
                    "state": tuple(item["cube_state"]), "history": trace_history[-HISTORY_LENGTH:],
                    "move": item["next_expert_move"], "move_count": item["move_count"],
                    "sample_kind": "expert", "expert_row": item["raw_expert_row"],
                })
                trace_history.append(item["move_id"])
            actions, selectors = label_trajectory(
                transitions, SOURCE_HUMAN, split, "reco:{}:{}".format(solve_id, chunk[0]["move_count"]),
            )
            action_rows.extend(actions)
            selector_rows.extend(selectors)
    return action_rows, selector_rows


def _feature_tensor(rows, target, device="cpu"):
    return torch.stack([
        encode_f2l_features(row["state"], row["history"], row["protected_mask"],
                            row["target_pair"] if target else None, row["move_count"], "cpu")
        for row in rows
    ]) if rows else torch.empty((0, HIERARCHICAL_F2L_FEATURE_SIZE), dtype=torch.float32)


def _balanced_weights(rows):
    counts = Counter((row["sample_kind"], row["target_pair"]) for row in rows)
    return torch.tensor([1.0 / math.sqrt(counts[(row["sample_kind"], row["target_pair"])])
                         for row in rows], dtype=torch.double)


def _batch_metrics(model, action_rows, selector_rows, device, batch_size=512):
    model.eval()
    action_features = _feature_tensor(action_rows, True)
    selector_features = _feature_tensor(selector_rows, False)
    action_correct = [0, 0, 0]
    action_count, value_abs, value_count = 0, 0.0, 0
    with torch.inference_mode():
        for start in range(0, len(action_rows), batch_size):
            batch = action_rows[start:start + batch_size]
            features = action_features[start:start + len(batch)].to(device)
            actions, values, _pairs = model(features)
            labels = torch.tensor([row["move_id"] for row in batch], dtype=torch.long, device=device)
            top = actions.topk(min(3, actions.shape[1]), dim=1).indices
            action_correct[0] += int((top[:, 0] == labels).sum())
            action_correct[1] += int((top == labels[:, None]).any(dim=1).sum())
            action_count += len(batch)
            targets = torch.tensor([row["remaining_moves"] for row in batch], dtype=torch.float32, device=device)
            value_abs += float((values - targets).abs().sum())
            value_count += len(batch)
        pair_correct = 0
        pair_count = 0
        for start in range(0, len(selector_rows), batch_size):
            batch = selector_rows[start:start + batch_size]
            features = selector_features[start:start + len(batch)].to(device)
            _actions, _values, pairs = model(features)
            labels = torch.tensor([row["target_pair"] for row in batch], dtype=torch.long, device=device)
            pair_correct += int((pairs.argmax(dim=1) == labels).sum())
            pair_count += len(batch)
    return {
        "action_samples": action_count,
        "action_top1": action_correct[0] / action_count if action_count else None,
        "action_top3": action_correct[1] / action_count if action_count else None,
        "remaining_moves_mae": value_abs / value_count if value_count else None,
        "pair_selection_samples": pair_count,
        "pair_selection_top1": pair_correct / pair_count if pair_count else None,
    }


def _fit(train_actions, train_selectors, validation_actions, validation_selectors,
         seed, epochs, device):
    if not train_actions or not train_selectors or not validation_actions or not validation_selectors:
        raise ValueError("Hierarchical F2L training and validation need action and pair-order labels")
    set_seed(seed)
    model = HierarchicalF2LPolicy().to(device)
    action_features = _feature_tensor(train_actions, True)
    selector_features = _feature_tensor(train_selectors, False)
    action_labels = torch.tensor([row["move_id"] for row in train_actions], dtype=torch.long)
    value_labels = torch.tensor([row["remaining_moves"] for row in train_actions], dtype=torch.float32)
    selector_labels = torch.tensor([row["target_pair"] for row in train_selectors], dtype=torch.long)
    batch_size = 256
    action_sampler = WeightedRandomSampler(_balanced_weights(train_actions), min(len(train_actions), 30000), replacement=True,
                                           generator=torch.Generator().manual_seed(seed))
    selector_sampler = WeightedRandomSampler(_balanced_weights(train_selectors), min(len(train_selectors), 8000), replacement=True,
                                             generator=torch.Generator().manual_seed(seed + 1))
    action_loader = DataLoader(TensorDataset(action_features, action_labels, value_labels),
                               batch_size=batch_size, sampler=action_sampler, num_workers=0)
    selector_loader = DataLoader(TensorDataset(selector_features, selector_labels),
                                 batch_size=batch_size, sampler=selector_sampler, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=0.0001)
    best_state, best_loss, best_epoch, stale, history = None, float("inf"), 0, 0, []
    validation_action_features = _feature_tensor(validation_actions, True).to(device)
    validation_action_labels = torch.tensor([row["move_id"] for row in validation_actions], dtype=torch.long, device=device)
    validation_values = torch.tensor([row["remaining_moves"] for row in validation_actions], dtype=torch.float32, device=device)
    validation_selector_features = _feature_tensor(validation_selectors, False).to(device)
    validation_selector_labels = torch.tensor([row["target_pair"] for row in validation_selectors], dtype=torch.long, device=device)
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        action_batches = iter(action_loader)
        selector_batches = iter(selector_loader)
        for batch in action_batches:
            features, labels, values = (item.to(device) for item in batch)
            optimizer.zero_grad(set_to_none=True)
            logits, estimate, _pairs = model(features)
            loss = F.cross_entropy(logits, labels) + 0.05 * F.smooth_l1_loss(estimate, values)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        for batch in selector_batches:
            features, labels = (item.to(device) for item in batch)
            optimizer.zero_grad(set_to_none=True)
            _actions, _values, logits = model(features)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            actions, values, _pairs = model(validation_action_features)
            _actions, _values, pairs = model(validation_selector_features)
            val_action_loss = float(F.cross_entropy(actions, validation_action_labels))
            val_value_loss = float(F.smooth_l1_loss(values, validation_values))
            val_pair_loss = float(F.cross_entropy(pairs, validation_selector_labels))
        val_loss = val_action_loss + 0.05 * val_value_loss + val_pair_loss
        history.append({
            "epoch": epoch, "validation_action_loss": val_action_loss,
            "validation_value_loss": val_value_loss, "validation_pair_loss": val_pair_loss,
            "seconds": time.perf_counter() - started,
        })
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 4:
                break
    model.load_state_dict(best_state)
    return model, best_epoch, history, best_loss


def _save_rows(path, action_rows, selector_rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ([dict(row, task="action_value") for row in action_rows]
            + [dict(row, task="pair_selection") for row in selector_rows])
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def _collect_recovery(model, records, device, count=24):
    """Relabel bounded F2L states visited by the first learned greedy policy."""
    train_records = [row for row in records if row["split"] == "train"][:count]
    requests, origins = [], []
    for record in train_records:
        start = apply_sequence(SOLVED_STATE, parse_moves(record["scramble"]))
        cross_moves = parse_moves(record["phases"].get("CROSS", []))
        cross_state = apply_sequence(start, cross_moves)
        if not cross_complete(cross_state):
            continue
        prefix_history = ([MOVE_TO_ID[move] for move in cross_moves]
                          if record.get("prefix_history") is None else list(record["prefix_history"]))
        current, current_history = cross_state, list(prefix_history)[-HISTORY_LENGTH:]
        current_count = len(record["phases"].get("CROSS", [])) if record.get("prefix_history") is None else len(record["prefix_history"])
        protected = _mask_bits(current)
        selector_features = encode_f2l_features(current, current_history, protected, None,
                                                 current_count, device).unsqueeze(0)
        with torch.inference_mode():
            _actions, _values, selector_logits = model(selector_features)
        target = next((index for index in selector_logits[0].argsort(descending=True).tolist()
                       if not protected & (1 << index)), None)
        if target is None:
            continue
        prefix_moves, visited = [], {current}
        for _step in range(4):
            features = encode_f2l_features(current, current_history, protected, target,
                                           current_count + len(prefix_moves), device).unsqueeze(0)
            with torch.inference_mode():
                action_logits, _values, _pairs = model(features)
            previous = SIMULATOR_MOVE_TOKENS[current_history[-1]] if current_history else None
            previous_base = previous.rstrip("'2") if previous else None
            next_move = None
            for move_id in action_logits[0].argsort(descending=True).tolist():
                move = SIMULATOR_MOVE_TOKENS[move_id]
                if previous and move == inverse_move(previous):
                    continue
                if previous_base and move.rstrip("'2") == previous_base:
                    continue
                child = apply_move(current, move)
                if child in visited or not cross_complete(child):
                    continue
                next_move = move
                break
            if next_move is None:
                break
            prefix_moves.append(next_move)
            current = apply_move(current, next_move)
            current_history = (current_history + [MOVE_TO_ID[next_move]])[-HISTORY_LENGTH:]
            visited.add(current)
        if not prefix_moves:
            continue
        visited = apply_sequence(cross_state, prefix_moves)
        if not cross_complete(visited) or f2l_complete(visited):
            continue
        composite = " ".join(parse_moves(record["scramble"]) + cross_moves + prefix_moves)
        requests.append(composite)
        origins.append((record, cross_state, prefix_moves, prefix_history))
    if not requests:
        return [], {"requested": 0, "verified": 0, "rejected": []}
    targets = []
    for offset in range(0, len(requests), 8):
        targets.extend(teacher_request({"mode": "solve_many", "scrambles": requests[offset:offset + 8]}))
    action_rows, selector_rows, recovery_records, rejected = [], [], [], []
    for index, (teacher, (origin, _cross_state, prefix_moves, prefix_history)) in enumerate(zip(targets, origins)):
        prefix_state = apply_sequence(_cross_state, prefix_moves)
        reference_cross = parse_moves(teacher["phases"].get("CROSS", []))
        f2l_moves = parse_moves(teacher["phases"].get("F2L", []))
        reference_moves = reference_cross + f2l_moves
        final_state = apply_sequence(prefix_state, reference_moves)
        if (not cross_complete(final_state) or not f2l_complete(final_state)
                or _mask_bits(final_state) & _mask_bits(prefix_state) != _mask_bits(prefix_state)):
            rejected.append(origin["scramble_id"] + ":invalid-f2l-recovery")
            continue
        start_history = (list(prefix_history) + [MOVE_TO_ID[move] for move in prefix_moves])[-HISTORY_LENGTH:]
        history = list(start_history)
        state = prefix_state
        transitions = []
        for move_index, move in enumerate(reference_moves):
            transitions.append({
                "state": state, "history": list(history), "move": move,
                "move_count": len(prefix_history) + len(prefix_moves) + move_index,
                "sample_kind": "synthetic",
            })
            state = apply_move(state, move)
            history = (history + [MOVE_TO_ID[move]])[-HISTORY_LENGTH:]
        actions, selectors = label_trajectory(
            transitions, SOURCE_RECOVERY, "train", "on-policy:{}:{}".format(origin["scramble_id"], index),
        )
        if not actions or not selectors:
            rejected.append(origin["scramble_id"] + ":no-pair-transition")
            continue
        action_rows.extend(actions)
        selector_rows.extend(selectors)
        recovery_records.append({
            "sample_kind": "synthetic", "data_source": SOURCE_RECOVERY,
            "split": "train", "source_scramble_id": origin["scramble_id"],
            "start_state": list(prefix_state), "history": start_history,
            "start_move_count": len(prefix_history) + len(prefix_moves),
            "on_policy_prefix": prefix_moves, "reference_cross": reference_cross,
            "reference_f2l": f2l_moves, "reference_moves": reference_moves,
            "on_policy_scramble": " ".join(parse_moves(origin["scramble"])
                                              + parse_moves(origin["phases"].get("CROSS", []))
                                              + prefix_moves),
            "reference_verified": bool(teacher["reference_verified"]),
        })
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    recovery_path = DATA_DIR / "on_policy_recovery.jsonl"
    recovery_path.write_text("".join(json.dumps(row) + "\n" for row in recovery_records), encoding="utf-8")
    return (action_rows, selector_rows), {"requested": len(requests), "verified": len(recovery_records),
                         "action_samples": len(action_rows), "rejected": rejected,
                         "data_path": str(recovery_path.relative_to(ROOT))}


def _evaluate_old_human_model(action_rows, device):
    rows = [row["expert_row"] for row in action_rows if row.get("expert_row") is not None]
    if not rows:
        return None
    checkpoint_path = ROOT / "artifacts" / "reco-cfop-context.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = F2LPolicy(EXPERT_FEATURE_SIZE, len(SIMULATOR_MOVE_TOKENS)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    features = encode_expert_features(rows, context=True).to(device)
    labels = torch.tensor([row["move_id"] for row in rows], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(features)
        top = logits.topk(3, dim=1).indices
    return {"samples": len(rows), "top1": float((top[:, 0] == labels).float().mean()),
            "top3": float((top == labels[:, None]).any(dim=1).float().mean()),
            "training_samples": checkpoint.get("train_samples"),
            "full_dataset_f2l_top1": 0.6142}


def train(checkpoint_path=CHECKPOINT, metrics_path=METRICS, epochs=12,
          device_name="auto", recovery_count=24):
    device = choose_device(device_name)
    torch.set_num_threads(2)
    human_splits = _human_splits()
    human = {split: human_trajectories(rows, split) for split, rows in human_splits.items()}
    synthetic_records = _synthetic_records()
    synthetic = {"train": ([], []), "validation": ([], [])}
    for record in synthetic_records:
        if record["split"] not in synthetic:
            continue
        source = record["data_source"]
        actions, selectors = label_trajectory(
            synthetic_trajectory(record), source, record["split"], record["scramble_id"],
        )
        synthetic[record["split"]][0].extend(actions)
        synthetic[record["split"]][1].extend(selectors)
    base_train_actions = human["train"][0] + synthetic["train"][0]
    base_train_selectors = human["train"][1] + synthetic["train"][1]
    validation_actions = human["validation"][0] + synthetic["validation"][0]
    validation_selectors = human["validation"][1] + synthetic["validation"][1]
    test_actions, test_selectors = human["test"]
    pre_model, pre_epoch, pre_history, pre_val = _fit(
        base_train_actions, base_train_selectors, validation_actions, validation_selectors,
        79, epochs, device,
    )
    pre_metrics = _batch_metrics(pre_model, validation_actions, validation_selectors, device)
    pre_checkpoint_path = ROOT / "artifacts" / "hierarchical-f2l-pre-recovery.pt"
    torch.save({
        "schema_version": 1, "feature_size": HIERARCHICAL_F2L_FEATURE_SIZE,
        "actions": len(SIMULATOR_MOVE_TOKENS), "model_state": pre_model.state_dict(),
        "seed": 79, "best_epoch": pre_epoch, "sample_kind": "mixed",
        "data_sources": [SOURCE_HUMAN, SOURCE_TEACHER, SOURCE_DAGGER],
        "training_data": {"expert_actions": len(human["train"][0]),
                           "synthetic_actions": len(synthetic["train"][0])},
    }, pre_checkpoint_path)
    (recovery_actions, recovery_selectors), recovery_manifest = _collect_recovery(
        pre_model, [row for row in synthetic_records if row.get("data_source") == SOURCE_TEACHER],
        device, recovery_count,
    ) if recovery_count > 0 else (([], []), {"requested": 0, "verified": 0, "rejected": []})
    final_train_actions = base_train_actions + recovery_actions
    final_train_selectors = base_train_selectors + recovery_selectors
    models = []
    for seed in (79, 80):
        model, best_epoch, history, val_loss = _fit(
            final_train_actions, final_train_selectors,
            validation_actions, validation_selectors, seed, epochs, device,
        )
        models.append((val_loss, seed, model, best_epoch, history))
    _val_loss, seed, model, best_epoch, history = min(models, key=lambda item: item[0])
    validation_metrics = _batch_metrics(model, validation_actions, validation_selectors, device)
    test_metrics = _batch_metrics(model, test_actions, test_selectors, device)
    old_human = _evaluate_old_human_model(test_actions, device)
    source_action_counts = Counter(row["data_source"] for row in final_train_actions)
    source_selector_counts = Counter(row["data_source"] for row in final_train_selectors)
    split_hashes = {
        split: hashlib.sha256("\n".join(sorted(row["trajectory_id"] for row in actions)).encode()).hexdigest()
        for split, (actions, _selectors) in human.items()
    }
    manifest = {
        "task": "hierarchical pair-conditioned F2L policy/value and learned pair-order model",
        "sample_kind": "mixed: expert imitation and synthetic simulator trajectories",
        "data_sources": [SOURCE_HUMAN, SOURCE_TEACHER, SOURCE_DAGGER, SOURCE_RECOVERY],
        "human_filter": "reco.nz 3x3 CFOP; all recorded Cross colors normalized to Down; F2L rows with outer-face actions; replay-derived Cross and pair goals",
        "human_split_policy": "original reco.nz solver-held-out train/validation/test; test used only for evaluation",
        "synthetic_split_policy": "existing cfop_teacher and dagger records keep their train/validation split; on-policy recovery uses train scrambles only",
        "pair_labels": "inferred only when simulator replay shows exactly one new solved slot while protected slots remain solved",
        "synthetic_trajectories": len(synthetic_records),
        "training_action_samples_by_source": dict(source_action_counts),
        "training_pair_selection_samples_by_source": dict(source_selector_counts),
        "validation": validation_metrics,
        "heldout_expert_test": test_metrics,
        "old_reco_model_same_test_rows": old_human,
        "pre_recovery": {"checkpoint": str(pre_checkpoint_path.relative_to(ROOT)),
                         "validation": pre_metrics, "best_epoch": pre_epoch,
                         "validation_loss": pre_val, "history": pre_history},
        "on_policy_recovery": recovery_manifest,
        "final_training_seeds": [{"seed": item[1], "validation_loss": item[0], "best_epoch": item[3]}
                                  for item in models],
        "selected_seed": seed, "best_epoch": best_epoch, "history": history,
        "architecture": {"input_size": HIERARCHICAL_F2L_FEATURE_SIZE,
                         "hidden": [256, 128], "heads": ["72-action policy", "remaining moves", "4-way pair selector"]},
        "training_config": {"optimizer": "AdamW", "batch_size": 256, "epochs_max": epochs,
                            "policy_loss": "cross entropy + 0.05 smooth L1 remaining moves",
                            "selector_loss": "cross entropy", "initialization": "PyTorch random initialization"},
        "expert_split_sha256": split_hashes,
        "device": str(device),
    }
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": 1, "feature_size": HIERARCHICAL_F2L_FEATURE_SIZE,
        "actions": len(SIMULATOR_MOVE_TOKENS), "model_state": model.state_dict(),
        "seed": seed, "best_epoch": best_epoch, "sample_kind": "mixed",
        "data_sources": [SOURCE_HUMAN, SOURCE_TEACHER, SOURCE_DAGGER, SOURCE_RECOVERY],
        "training_action_samples_by_source": dict(source_action_counts),
        "training_pair_selection_samples_by_source": dict(source_selector_counts),
        "training_config": manifest["training_config"],
    }, checkpoint_path)
    metrics_path = Path(metrics_path)
    metrics_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _save_rows(DATA_DIR / "train.jsonl", final_train_actions, final_train_selectors)
    _save_rows(DATA_DIR / "validation.jsonl", validation_actions, validation_selectors)
    print(json.dumps(manifest, indent=2))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--recovery-count", type=int, default=24)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.recovery_count < 0:
        parser.error("epochs must be positive and recovery-count nonnegative")
    train(args.checkpoint, args.metrics, args.epochs, args.device, args.recovery_count)


if __name__ == "__main__":
    main()
