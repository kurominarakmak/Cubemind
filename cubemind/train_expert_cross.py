"""Train a solver-disjoint, from-scratch policy on explicit human Cross actions."""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from cubemind.environment.cube import (
    COLOR_NAMES, MOVE_TOKENS, SIMULATOR_MOVE_TOKENS, cross_frame_move,
    cross_complete, normalize_cross_frame,
)
from cubemind.models.expert_cross import FEATURE_SIZE, ExpertCrossPolicy
from cubemind.train_hierarchical_f2l import _human_splits
from cubemind.training.engine import choose_device, set_seed

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "reco_cfop"
CHECKPOINT = ROOT / "artifacts" / "v2" / "expert-cross-policy.pt"
REPORT = ROOT / "artifacts" / "v2" / "expert-cross-training.json"
MOVE_TO_ID = {move: index for index, move in enumerate(MOVE_TOKENS)}


def _cross_rows(rows):
    result, excluded = [], Counter()
    for row in rows:
        if row.get("phase") != "CROSS":
            continue
        if row.get("sample_kind") != "expert" or row.get("data_source") != "reco.nz":
            raise ValueError("Cross imitation accepts only source-labeled human rows")
        color = row.get("cross_color")
        if color not in COLOR_NAMES:
            excluded["unknown_cross_color"] += 1
            continue
        state = tuple(row["cube_state"])
        centers = tuple(state[face * 9 + 4] for face in range(6))
        color_id = COLOR_NAMES.index(color)
        source_move = row.get("next_expert_move")
        if source_move not in SIMULATOR_MOVE_TOKENS or SIMULATOR_MOVE_TOKENS.index(source_move) != row.get("move_id"):
            raise ValueError("Invalid source move label in human Cross row")
        move = cross_frame_move(source_move, centers, color_id)
        if move not in MOVE_TO_ID:
            excluded["non_outer_turn"] += 1
            continue
        normalized = normalize_cross_frame(state, color_id)
        if cross_complete(normalized):
            excluded["annotation_after_cross_goal"] += 1
            continue
        centers_after_rotation = tuple(normalized[face * 9 + 4] for face in range(6))
        color_relabel = {color_id: face for face, color_id in enumerate(centers_after_rotation)}
        canonical = tuple(color_relabel[sticker] for sticker in normalized)
        result.append({"state": canonical, "label": MOVE_TO_ID[move], "cross_color": color})
    return result, dict(excluded)


def _features(rows):
    states = torch.tensor([row["state"] for row in rows], dtype=torch.long)
    return F.one_hot(states, num_classes=6).float().flatten(1)


def _evaluate(model, rows, device):
    if not rows:
        return {"samples": 0, "top1": None, "top3": None, "cross_entropy": None}
    model.eval()
    x = _features(rows).to(device)
    y = torch.tensor([row["label"] for row in rows], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(x)
        top = logits.topk(3, dim=1).indices
        result = {"samples": len(rows), "top1": float((top[:, :1] == y[:, None]).float().mean()),
                  "top3": float((top == y[:, None]).any(dim=1).float().mean()),
                  "cross_entropy": float(F.cross_entropy(logits, y))}
    result["top1_by_demonstrated_cross_color"] = {
        color: float((top[[i for i, row in enumerate(rows) if row["cross_color"] == color], 0]
                      == y[[i for i, row in enumerate(rows) if row["cross_color"] == color]]).float().mean())
        for color in COLOR_NAMES if any(row["cross_color"] == color for row in rows)
    }
    return result


def _fit(train_rows, validation_rows, seed, epochs, device):
    set_seed(seed)
    model = ExpertCrossPolicy().to(device)
    x = _features(train_rows)
    y = torch.tensor([row["label"] for row in train_rows], dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=512, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    validation_x = _features(validation_rows).to(device)
    validation_y = torch.tensor([row["label"] for row in validation_rows], dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    best_state, best_loss, best_epoch, stale = None, float("inf"), 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(features.to(device)), labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_logits = model(validation_x)
            validation_loss = float(F.cross_entropy(validation_logits, validation_y))
            validation_top1 = float((validation_logits.argmax(1) == validation_y).float().mean())
        history.append({"epoch": epoch, "validation_loss": validation_loss,
                        "validation_top1": validation_top1,
                        "seconds": time.perf_counter() - started})
        if validation_loss < best_loss:
            best_state, best_loss, best_epoch, stale = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }, validation_loss, epoch, 0
        else:
            stale += 1
            if stale >= 4:
                break
    model.load_state_dict(best_state)
    return model, {"seed": seed, "best_epoch": best_epoch, "validation_loss": best_loss,
                   "history": history}


def train(data_dir=PROCESSED, checkpoint_path=CHECKPOINT, report_path=REPORT,
          epochs=20, seeds=(17, 37, 73), device_name="auto"):
    data_dir = Path(data_dir)
    splits = _human_splits() if data_dir.resolve() == PROCESSED.resolve() else {
        name: [json.loads(line) for line in (data_dir / (name + ".jsonl")).read_text().splitlines() if line]
        for name in ("train", "validation", "test")
    }
    solver_splits = {}
    for split, rows in splits.items():
        for row in rows:
            solver_id = row.get("solver_id")
            if solver_id is None or row.get("sample_kind") != "expert" or row.get("data_source") != "reco.nz":
                raise ValueError("Expected solver-labeled, expert-only reco.nz splits")
            if solver_splits.setdefault(solver_id, split) != split:
                raise ValueError("Human Cross evaluation must be held out by solver")
    prepared, exclusions = {}, {}
    for split, rows in splits.items():
        prepared[split], exclusions[split] = _cross_rows(rows)
        if not prepared[split]:
            raise ValueError("No supported human Cross actions in {}".format(split))

    device = choose_device(device_name)
    torch.set_num_threads(2)
    best_model, best_run = None, None
    seed_runs = []
    for seed in seeds:
        model, run = _fit(prepared["train"], prepared["validation"], seed, epochs, device)
        run["validation"] = _evaluate(model, prepared["validation"], device)
        seed_runs.append(run)
        if best_run is None or run["validation_loss"] < best_run["validation_loss"]:
            best_model, best_run = model, run
    train_counts = Counter(row["label"] for row in prepared["train"])
    most_common = {move_id for move_id, _count in train_counts.most_common(3)}
    test_labels = [row["label"] for row in prepared["test"]]
    baseline = {"most_common_action_top1": sum(label == train_counts.most_common(1)[0][0]
                                                for label in test_labels) / len(test_labels),
                "most_common_three_top3": sum(label in most_common for label in test_labels) / len(test_labels)}
    report = {
        "task": "human-only Cross action imitation",
        "data_source": "reco.nz 3x3 CFOP reconstructions",
        "data_split": "existing solver-disjoint train/validation/test; phase CROSS only",
        "label_policy": "explicit Cross phase rows; X-Cross/Cross+1 rows are not inferred as Cross labels",
        "feature_schema": "54 facelets after rotating the demonstrated Cross color to Down; no simulator labels",
        "action_vocabulary": list(MOVE_TOKENS), "excluded_rows": exclusions,
        "split_samples": {split: len(rows) for split, rows in prepared.items()},
        "human_cross_color_counts": {split: dict(Counter(row["cross_color"] for row in rows))
                                     for split, rows in prepared.items()},
        "cross_plus_one_annotations": {split: sum(row.get("phase") == "CROSS+1" for row in rows)
                                       for split, rows in splits.items()},
        "xcross_annotations": {split: sum(row.get("phase") == "CROSS+1"
                                          and row.get("phase_annotation", "").lower() in ("xcross", "x-cross")
                                          for row in rows) for split, rows in splits.items()},
        "majority_action_baseline": baseline,
        "seed_runs": seed_runs,
        "test": _evaluate(best_model, prepared["test"], device),
        "selected_seed": best_run["seed"], "selected_validation_loss": best_run["validation_loss"],
        "initialization": "PyTorch random initialization", "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in best_model.parameters()),
        "human_timing": "not measured",
    }
    split_hashes = {split: hashlib.sha256((data_dir / (split + ".jsonl")).read_bytes()).hexdigest()
                    for split in ("train", "validation", "test")}
    checkpoint_path, report_path = Path(checkpoint_path), Path(report_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"task": "expert_cross_action_policy", "sample_kind": "expert",
                "data_source": "reco.nz 3x3 CFOP reconstructions", "feature_size": FEATURE_SIZE,
                "actions": len(MOVE_TOKENS), "model_state": best_model.state_dict(),
                "seed": best_run["seed"], "best_epoch": best_run["best_epoch"],
                "solver_split_sha256": split_hashes}, checkpoint_path)
    report["checkpoint"] = str(checkpoint_path.relative_to(ROOT))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", default="17,37,73")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    train(args.data_dir, args.checkpoint, args.report, args.epochs,
          tuple(int(seed) for seed in args.seeds.split(",")), args.device)


if __name__ == "__main__":
    main()
