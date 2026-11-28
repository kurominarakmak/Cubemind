"""Train a solver-disjoint preference ranker from replay-matched human OLL/PLL segments."""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from cubemind.algorithm_memory import (
    _home_moves, _oll_complete, algorithm_records, recognize_oll_case,
)
from cubemind.environment.cube import (
    SOLVED_STATE, apply_sequence, f2l_complete, last_layer_edges_oriented,
    normalize_cube_orientation, parse_moves,
    inverse_sequence,
)
from cubemind.last_layer import recognize_zbll_case
from cubemind.models.algorithm_preference import (
    FEATURE_SIZE, AlgorithmPreferenceRanker, algorithm_features, normalize_algorithm,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/v2/algorithm-preference-report.json"
CHECKPOINT = ROOT / "artifacts/v2/algorithm-preference-policy.pt"
RAW = ROOT / "data/raw/reco_nz/cfop_3x3.jsonl"
PROCESSED = ROOT / "data/processed/reco_cfop"


def _split_map():
    result = {}
    for split in ("train", "validation", "test"):
        path = PROCESSED / (split + ".jsonl")
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            result[row["solver_id"]] = split
    return result


def _table():
    grouped = defaultdict(dict)
    for record in algorithm_records():
        if record["family"] not in ("OLL", "PLL"):
            continue
        key = normalize_algorithm(record["moves"])
        current = grouped[(record["family"], record["case_id"])].get(key)
        if current is None or (record["move_metrics"]["stm"], record["algorithm_id"]) < (
                current["move_metrics"]["stm"], current["algorithm_id"]):
            grouped[(record["family"], record["case_id"])][key] = record
    sequence_index = defaultdict(list)
    candidates = {}
    for (family, case_id), variants in grouped.items():
        ordered = sorted(variants.items(), key=lambda item: (
            item[1]["move_metrics"]["stm"], item[1]["move_metrics"]["rotations"], item[1]["algorithm_id"]))
        candidates[(family, case_id)] = ordered
        for key, record in ordered:
            sequence_index[(family, key)].append(case_id)
    return candidates, sequence_index


def _candidate_replays(before, record, family):
    """Replay only the matched source algorithm under allowed setup orientation/AUF."""
    home = _home_moves(before)
    body = tuple(record["moves"])
    if family == "OLL":
        return any(_oll_complete(normalize_cube_orientation(apply_sequence(
            before, home + ("U",) * auf + body))) for auf in range(4))
    for yaw_count in range(4):
        yaw = ("y",) * yaw_count
        undo_yaw = tuple(inverse_sequence(yaw))
        for pre in range(4):
            for post in range(4):
                moves = home + yaw + ("U",) * pre + body + ("U",) * post + undo_yaw
                if normalize_cube_orientation(apply_sequence(before, moves)) == SOLVED_STATE:
                    return True
    return False


def extract_examples():
    split_by_solver = _split_map()
    candidates, sequence_index = _table()
    examples, counts = [], Counter()
    for line in RAW.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        solver = record["solver_id"]
        split = split_by_solver.get(solver)
        if split is None:
            counts["unmapped_solver"] += 1
            continue
        try:
            state = apply_sequence(SOLVED_STATE, parse_moves(record["scramble"]))
        except (TypeError, ValueError):
            counts["invalid_scramble"] += 1
            continue
        for segment in record.get("segments", []):
            phase = str(segment.get("phase", "")).strip().upper()
            moves = tuple(parse_moves(segment.get("moves", "")))
            if phase in ("OLL", "PLL"):
                counts[phase + "_segments"] += 1
                before, key = state, normalize_algorithm(moves)
                after = apply_sequence(before, moves)
                if phase == "OLL":
                    case_id = recognize_oll_case(before)
                    valid = bool(case_id and f2l_complete(after)
                                 and _oll_complete(normalize_cube_orientation(after)))
                else:
                    recognized = recognize_zbll_case(before)
                    valid = bool(f2l_complete(before) and last_layer_edges_oriented(before)
                                 and any(normalize_cube_orientation(apply_sequence(after, ("U",) * auf))
                                         == SOLVED_STATE for auf in range(4)))
                    matches = sequence_index.get((phase, key), [])
                    if recognized and recognized.startswith("PLL-"):
                        matches = [candidate for candidate in matches if candidate == recognized]
                    case_id = matches[0] if len(set(matches)) == 1 else None
                matches = sequence_index.get((phase, key), [])
                if phase == "OLL" and case_id:
                    matches = [candidate for candidate in matches if candidate == case_id]
                if valid and case_id and (phase, case_id) in candidates:
                    rows = candidates[(phase, case_id)]
                    label = next((index for index, (candidate_key, _row) in enumerate(rows)
                                  if candidate_key == key), None)
                    if label is not None and not _candidate_replays(before, rows[label][1], phase):
                        label = None
                    if label is not None and len(rows) > 1:
                        examples.append({"split": split, "solver_id": solver, "family": phase,
                                         "case_id": case_id, "candidate_keys": [x[0] for x in rows],
                                         "candidate_rows": [x[1] for x in rows], "label": label})
                        counts[phase + "_matched_choice"] += 1
                    elif label is not None:
                        counts[phase + "_matched_singleton"] += 1
                    else:
                        counts[phase + "_unmatched_algorithm"] += 1
                elif valid:
                    counts[phase + "_ambiguous_or_unknown_case"] += 1
                else:
                    counts[phase + "_replay_or_case_mismatch"] += 1
            state = apply_sequence(state, moves)
    return examples, counts, candidates


def _tensor_rows(example, case_index):
    return torch.tensor([algorithm_features(row["moves"]) for row in example["candidate_rows"]],
                        dtype=torch.float32), case_index[example["case_id"]]


def _metrics(model, examples, case_index):
    model.eval()
    top1 = top3 = loss_sum = 0.0
    with torch.inference_mode():
        for example in examples:
            features, case_id = _tensor_rows(example, case_index)
            scores = model(torch.full((len(features),), case_id, dtype=torch.long), features)
            label = example["label"]
            top1 += int(int(scores.argmax()) == label)
            top3 += int(label in scores.topk(min(3, len(scores))).indices.tolist())
            loss_sum += float(F.cross_entropy(scores.unsqueeze(0), torch.tensor([label])))
    count = len(examples)
    return {"examples": count, "top1": top1 / count if count else None,
            "top3": top3 / count if count else None,
            "cross_entropy": loss_sum / count if count else None}


def _predictions(model, examples, case_index):
    model.eval()
    result = []
    with torch.inference_mode():
        for example in examples:
            features, case_id = _tensor_rows(example, case_index)
            scores = model(torch.full((len(features),), case_id, dtype=torch.long), features)
            result.append(int(scores.argmax()))
    return result


def _paired_choice_comparison(predictions, baseline, examples):
    model_only = sum(prediction == row["label"] and other != row["label"]
                     for prediction, other, row in zip(predictions, baseline, examples))
    baseline_only = sum(other == row["label"] and prediction != row["label"]
                        for prediction, other, row in zip(predictions, baseline, examples))
    discordant = model_only + baseline_only
    p_value = (min(1.0, 2 * sum(math.comb(discordant, index)
                                for index in range(min(model_only, baseline_only) + 1)
                                ) / (2 ** discordant)) if discordant else 1.0)
    return {"preference_only_correct": model_only, "baseline_only_correct": baseline_only,
            "both_correct": sum(prediction == row["label"] and other == row["label"]
                                for prediction, other, row in zip(predictions, baseline, examples)),
            "neither_correct": sum(prediction != row["label"] and other != row["label"]
                                   for prediction, other, row in zip(predictions, baseline, examples)),
            "exact_mcnemar_two_sided_p": p_value}


def _train_seed(train, validation, case_index, case_count, seed, epochs):
    random.seed(seed)
    torch.manual_seed(seed)
    model = AlgorithmPreferenceRanker(case_count)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    best_loss, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(train)))
        random.shuffle(order)
        for start in range(0, len(order), 64):
            batch = [train[index] for index in order[start:start + 64]]
            feature_rows, case_rows, offsets = [], [], [0]
            labels = []
            for example in batch:
                features, case_id = _tensor_rows(example, case_index)
                feature_rows.append(features)
                case_rows.extend([case_id] * len(features))
                offsets.append(offsets[-1] + len(features))
                labels.append(example["label"])
            features = torch.cat(feature_rows)
            scores = model(torch.tensor(case_rows), features)
            losses = [F.cross_entropy(scores[offsets[i]:offsets[i + 1]].unsqueeze(0),
                                      torch.tensor([labels[i]])) for i in range(len(batch))]
            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        validation_metrics = _metrics(model, validation, case_index)
        if validation_metrics["cross_entropy"] < best_loss:
            best_loss = validation_metrics["cross_entropy"]
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
    model.load_state_dict(best_state)
    return model, best_loss, best_epoch


def train(seeds=(17, 37, 73), epochs=20, output=OUTPUT, checkpoint=CHECKPOINT):
    examples, counts, candidates = extract_examples()
    splits = {name: [row for row in examples if row["split"] == name]
              for name in ("train", "validation", "test")}
    if not splits["train"] or not splits["validation"] or not splits["test"]:
        raise ValueError("The explicit human algorithm labels did not populate all solver-disjoint splits")
    case_ids = sorted({row["case_id"] for row in examples})
    case_index = {case_id: index for index, case_id in enumerate(case_ids)}
    seed_runs, selected = [], None
    for seed in seeds:
        model, validation_loss, epoch = _train_seed(
            splits["train"], splits["validation"], case_index, len(case_ids), seed, epochs)
        seed_runs.append({"seed": seed, "best_epoch": epoch, "validation_loss": validation_loss})
        if selected is None or validation_loss < selected[0]:
            selected = (validation_loss, model, seed, epoch)
    _val_loss, model, selected_seed, best_epoch = selected
    train_frequency = Counter((row["case_id"], row["candidate_keys"][row["label"]])
                              for row in splits["train"])
    baselines = {}
    model_predictions = _predictions(model, splits["test"], case_index)
    for name in ("shortest", "most_frequent"):
        correct = 0
        predictions = []
        for row in splits["test"]:
            if name == "shortest":
                choice = 0
            else:
                choice = max(range(len(row["candidate_keys"])), key=lambda index: (
                    train_frequency[(row["case_id"], row["candidate_keys"][index])], -index))
            predictions.append(choice)
            correct += int(choice == row["label"])
        baselines[name] = {"examples": len(splits["test"]), "top1": correct / len(splits["test"]),
                           "paired_with_preference": _paired_choice_comparison(
                               model_predictions, predictions, splits["test"])}
    report = {
        "task": "human OLL/PLL algorithm choice ranking",
        "data_source": "local reco.nz 3x3 CFOP reconstructions; explicit OLL/PLL segments only",
        "preference_label": "unique normalized table sequence; exact OLL recognition or unique PLL table mapping; the specific matched table algorithm must appear in replay-verified simulator candidates",
        "normalization": "trim leading/trailing U AUF and whole-cube rotations; do not infer unmatched labels",
        "candidate_families": {family: sum(key[0] == family and len(rows) > 1
                                             for key, rows in candidates.items())
                               for family in ("OLL", "PLL")},
        "matched_counts": dict(counts),
        "choice_examples": {split: len(rows) for split, rows in splits.items()},
        "choice_examples_by_family": {
            split: {family: sum(row["family"] == family for row in rows)
                    for family in ("OLL", "PLL")} for split, rows in splits.items()},
        "selected_seed": selected_seed, "selected_epoch": best_epoch, "seed_runs": seed_runs,
        "initialization": "PyTorch random initialization", "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()),
        "test_baselines": baselines,
        "test_preference_model": _metrics(model, splits["test"], case_index),
        "solver_conditioning": "not trained; no solver embeddings",
        "human_execution_or_recognition_time": "not measured",
    }
    checkpoint_path = Path(checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"task": "algorithm_preference_ranker", "model_state": model.state_dict(),
                "case_ids": case_ids, "feature_size": FEATURE_SIZE,
                "selected_seed": selected_seed, "best_epoch": best_epoch}, checkpoint_path)
    report["checkpoint"] = str(checkpoint_path.relative_to(ROOT))
    Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 37, 73])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    print(json.dumps(train(args.seeds, args.epochs, args.output, args.checkpoint), indent=2))
