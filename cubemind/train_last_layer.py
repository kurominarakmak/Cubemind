"""Train source-separated, from-scratch OLL/PLL/ZBLL policy/value models."""

import argparse
import hashlib
import json
import platform
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from cubemind.data.reco import split_by_solver
from cubemind.environment.cube import (
    COLORS, COLOR_NAMES, SIMULATOR_MOVE_TOKENS, SOLVED_STATE, apply_move, apply_sequence,
    inverse_sequence,
)
from cubemind.environment.cube import (
    f2l_complete, parse_moves, zbll_eligible,
)
from cubemind.last_layer import ZBLL_PATH, recognize_zbll_case, zbll_case_state
from cubemind.models.solver import SOLVER_FEATURE_SIZE, SolverPolicyValue
from cubemind.neural_solver import (
    PHASES, cube_solved, encode_features, oll_complete, teacher_request,
)
from cubemind.training.engine import choose_device, set_seed

ROOT = Path(__file__).resolve().parents[1]
RAW_HUMAN = ROOT / "data/raw/reco_nz/cfop_3x3.jsonl"
DATA_DIR = ROOT / "data/processed/last_layer"
DEFAULT_CHECKPOINT = ROOT / "artifacts/last-layer-policy.pt"
DEFAULT_METRICS = ROOT / "artifacts/last-layer-training.json"
LL_PHASES = ("OLL", "PLL", "ZBLL")
CFOP_SOURCE = "synthetic:cfop-teacher:@moishy/cfop@0.4.0"
ZBLL_SOURCE = "synthetic:zbll-algorithms:MIT-Alg-Trainer"


def _human_examples(path=RAW_HUMAN, seed=17):
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [row for row in records if row.get("reuse_permission_verified") is True]
    groups = split_by_solver(records, seed=seed)
    examples = {split: [] for split in ("train", "validation", "test")}
    for split, group in groups.items():
        for record in group:
            state = apply_sequence(SOLVED_STATE, parse_moves(record["scramble"]))
            history, move_count = [], 0
            for segment in record["segments"]:
                source_annotation = segment["phase"].strip().replace("\ufeff", "").upper()
                annotation = source_annotation
                phase = annotation if annotation in LL_PHASES else None
                auf_segment = annotation == "AUF"
                if auf_segment:
                    if f2l_complete(state) and oll_complete(state) and not cube_solved(state):
                        phase = "PLL"
                    elif zbll_eligible(state) and not oll_complete(state) and not cube_solved(state):
                        phase = "ZBLL"
                moves = parse_moves(segment["moves"])
                if phase:
                    valid_start = (
                        phase == "OLL" and f2l_complete(state) and not oll_complete(state)
                        or phase == "PLL" and f2l_complete(state) and oll_complete(state)
                        or phase == "ZBLL" and zbll_eligible(state)
                    )
                    final = apply_sequence(state, moves)
                    valid_end = (
                        phase == "OLL" and f2l_complete(final) and oll_complete(final)
                        or phase == "PLL" and f2l_complete(final) and cube_solved(final)
                        or phase == "ZBLL" and cube_solved(final)
                    )
                    if auf_segment:
                        valid_end = (phase == "PLL" and f2l_complete(final) and oll_complete(final)
                                     or phase == "ZBLL" and zbll_eligible(final) and not cube_solved(final))
                    case_id = recognize_zbll_case(state) if phase == "ZBLL" else None
                    if valid_start and valid_end and (phase != "ZBLL" or case_id is not None):
                        color = COLOR_NAMES.index(record["cross_color"])
                        for offset, move in enumerate(moves):
                            if move not in SIMULATOR_MOVE_TOKENS:
                                raise ValueError("Unsupported expert last-layer move: {}".format(move))
                            examples[split].append({
                                "state": list(state), "history": history[-8:], "move_count": move_count,
                                "phase": phase, "label": SIMULATOR_MOVE_TOKENS.index(move),
                                "value": float(len(moves) - offset - 1), "cross_color": color,
                                "sample_kind": "expert", "data_source": "reco.nz 3x3 CFOP reconstructions",
                                "phase_annotation": source_annotation,
                                "solver_id": record["solver_id"], "scramble_id": record["scramble_id"],
                                "trajectory_id": "{}:{}:{}".format(record["scramble_id"], phase, move_count + offset),
                                "case_id": case_id,
                            })
                            state = apply_move(state, move)
                            history.append(SIMULATOR_MOVE_TOKENS.index(move))
                            move_count += 1
                    else:
                        state = final
                        history.extend(SIMULATOR_MOVE_TOKENS.index(move) for move in moves)
                        move_count += len(moves)
                else:
                    state = apply_sequence(state, moves)
                    history.extend(SIMULATOR_MOVE_TOKENS.index(move) for move in moves)
                    move_count += len(moves)
            if not cube_solved(state):
                raise ValueError("Expert reconstruction failed simulator replay: {}".format(record["scramble_id"]))
    return examples


def _example_rows(start, moves, phase, split, source, trajectory_id, history=(),
                  cross_color=COLORS["D"], case_id=None):
    state, history = tuple(start), list(history)
    parsed = parse_moves(moves)
    rows = []
    for offset, move in enumerate(parsed):
        if move not in SIMULATOR_MOVE_TOKENS:
            raise ValueError("Unsupported synthetic last-layer move: {}".format(move))
        rows.append({
            "state": list(state), "history": history[-8:], "move_count": len(history),
            "phase": phase, "label": SIMULATOR_MOVE_TOKENS.index(move),
            "value": float(len(parsed) - offset - 1), "cross_color": cross_color,
            "sample_kind": "synthetic", "data_source": source, "split": split,
            "trajectory_id": trajectory_id, "case_id": case_id,
        })
        state = apply_move(state, move)
        history.append(SIMULATOR_MOVE_TOKENS.index(move))
    if phase == "OLL" and not (f2l_complete(state) and oll_complete(state)):
        raise ValueError("Synthetic OLL path failed phase replay: {}".format(trajectory_id))
    if phase == "PLL" and not (f2l_complete(state) and cube_solved(state)):
        raise ValueError("Synthetic PLL path failed phase replay: {}".format(trajectory_id))
    if phase == "ZBLL" and not cube_solved(state):
        raise ValueError("Synthetic ZBLL path failed phase replay: {}".format(trajectory_id))
    return rows


def _scramble(length, rng):
    faces, suffixes, moves = "URFDLB", ("", "'", "2"), []
    while len(moves) < length:
        face = rng.choice(faces)
        if moves and moves[-1][0] == face:
            continue
        moves.append(face + rng.choice(suffixes))
    return " ".join(moves)


def _synthetic_cfop_examples(split_counts, seed):
    result = {split: [] for split in split_counts}
    all_scrambles = set()
    for split, count in split_counts.items():
        rng = random.Random(seed + {"train": 0, "validation": 100_000, "test": 200_000}[split])
        scrambles = []
        while len(scrambles) < count:
            value = _scramble(25, rng)
            if value not in all_scrambles:
                all_scrambles.add(value)
                scrambles.append(value)
        for offset in range(0, count, 32):
            batch = scrambles[offset:offset + 32]
            references = teacher_request({"mode": "solve_many", "scrambles": batch})
            if len(references) != len(batch):
                raise ValueError("CFOP teacher returned the wrong number of rows")
            for scramble, row in zip(batch, references):
                if row.get("sample_kind") != "synthetic" or not row.get("reference_verified"):
                    raise ValueError("CFOP training target lacks synthetic provenance or replay verification")
                state = apply_sequence(SOLVED_STATE, parse_moves(scramble))
                history = []
                for phase in ("CROSS", "F2L", "OLL", "PLL"):
                    moves = parse_moves(row["phases"][phase])
                    if phase == "OLL" and not f2l_complete(state):
                        raise ValueError("CFOP teacher did not preserve F2L before OLL")
                    if phase == "PLL" and not (f2l_complete(state) and oll_complete(state)):
                        raise ValueError("CFOP teacher did not preserve OLL before PLL")
                    if phase in LL_PHASES and moves:
                        result[split].extend(_example_rows(
                            state, moves, phase, split, CFOP_SOURCE,
                            hashlib.sha256(scramble.encode("utf-8")).hexdigest()[:16] + ":" + phase,
                            history,
                        ))
                    state = apply_sequence(state, moves)
                    history.extend(SIMULATOR_MOVE_TOKENS.index(move) for move in moves)
                if not cube_solved(state):
                    raise ValueError("CFOP synthetic reference failed CubeMind replay")
        print("generated {} synthetic CFOP {} trajectories".format(count, split), flush=True)
    return result, all_scrambles


def _synthetic_zbll_examples(seed):
    database = json.loads(ZBLL_PATH.read_text(encoding="utf-8"))
    case_ids = sorted(case["case_id"] for case in database["cases"])
    random.Random(seed).shuffle(case_ids)
    train_end = int(len(case_ids) * 0.8)
    val_end = int(len(case_ids) * 0.9)
    case_split = {
        "train": set(case_ids[:train_end]),
        "validation": set(case_ids[train_end:val_end]),
        "test": set(case_ids[val_end:]),
    }
    examples = {split: [] for split in case_split}
    for case in database["cases"]:
        phase_split = next(split for split, identities in case_split.items() if case["case_id"] in identities)
        start = tuple(zbll_case_state(case["case_id"]))
        algorithm = parse_moves(case["algorithms"][0])
        for auf_index, auf in enumerate(([], ["U"], ["U2"], ["U'"])):
            state = apply_sequence(start, auf)
            target = inverse_sequence(auf)
            rows = _example_rows(
                state, target + algorithm, "ZBLL", phase_split, ZBLL_SOURCE,
                "{}-auf-{}".format(case["case_id"], auf_index), case_id=case["case_id"],
            )
            examples[phase_split].extend(rows)
    return examples, {key: sorted(value) for key, value in case_split.items()}


def _save_rows(directory, name, rows):
    path = Path(directory) / (name + ".jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    return path


def generate_data(directory=DATA_DIR, raw_human=RAW_HUMAN, seed=73,
                  train_solves=1200, validation_solves=160, test_solves=160):
    if min(train_solves, validation_solves, test_solves) < 1:
        raise ValueError("Each synthetic split needs at least one solve")
    human = _human_examples(raw_human, seed=17)
    cfop, scrambles = _synthetic_cfop_examples(
        {"train": train_solves, "validation": validation_solves, "test": test_solves}, seed,
    )
    zbll, case_splits = _synthetic_zbll_examples(seed)
    solver_splits = {
        split: {row["solver_id"] for row in rows}
        for split, rows in human.items()
    }
    solver_disjoint = not any(
        solver_splits[left] & solver_splits[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    if not solver_disjoint:
        raise ValueError("Human last-layer split leaks solver identities")
    manifest = {
        "schema_version": 1, "seed": seed, "data_sources_separated": True,
        "human_source": "reco.nz 3x3 CFOP reconstructions; only user-authorized records with verified replay",
        "human_split_policy": "grouped by solver_id with the repository's seed-17 split policy",
        "human_verified_actions_by_split_phase": {
            split: dict(Counter(row["phase"] for row in rows)) for split, rows in human.items()
        },
        "human_auf_actions_by_split_phase": {
            split: dict(Counter(row["phase"] for row in rows if row.get("phase_annotation") == "AUF"))
            for split, rows in human.items()
        },
        "human_solver_counts_by_split": {split: len(solver_splits[split]) for split in solver_splits},
        "human_solver_ids_disjoint": solver_disjoint,
        "synthetic_cfop_source": CFOP_SOURCE,
        "synthetic_cfop_scramble_counts": {"train": train_solves, "validation": validation_solves,
                                            "test": test_solves},
        "synthetic_cfop_scrambles_disjoint": len(scrambles) == train_solves + validation_solves + test_solves,
        "synthetic_cfop_scramble_hashes": sorted(
            hashlib.sha256(scramble.encode("utf-8")).hexdigest() for scramble in scrambles
        ),
        "synthetic_cfop_actions_by_split_phase": {
            split: dict(Counter(row["phase"] for row in rows)) for split, rows in cfop.items()
        },
        "synthetic_zbll_source": ZBLL_SOURCE,
        "synthetic_zbll_case_ids_by_split": case_splits,
        "synthetic_zbll_case_identity_splits_disjoint": not (
            set(case_splits["train"]) & set(case_splits["validation"])
            or set(case_splits["train"]) & set(case_splits["test"])
            or set(case_splits["validation"]) & set(case_splits["test"])
        ),
        "synthetic_zbll_actions_by_split": {split: len(rows) for split, rows in zbll.items()},
        "reco_nz_raw_sha256": hashlib.sha256(Path(raw_human).read_bytes()).hexdigest(),
        "zbll_algorithms_sha256": hashlib.sha256(ZBLL_PATH.read_bytes()).hexdigest(),
        "note": "Human and simulator-generated rows are written to separate files; only training combines eligible train rows in memory.",
    }
    for split in ("train", "validation", "test"):
        _save_rows(directory, "human_" + split, human[split])
        _save_rows(directory, "synthetic_cfop_" + split, cfop[split])
        _save_rows(directory, "synthetic_zbll_" + split, zbll[split])
    Path(directory).mkdir(parents=True, exist_ok=True)
    (Path(directory) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _encode_rows(rows):
    return torch.stack([
        encode_features(row["state"], row["history"], row["phase"], row["move_count"],
                        torch.device("cpu"), row.get("cross_color"))
        for row in rows
    ])


def _features(rows, device):
    return TensorDataset(
        _encode_rows(rows),
        torch.tensor([row["label"] for row in rows], dtype=torch.long),
        torch.tensor([row["value"] for row in rows], dtype=torch.float32),
    )


def _score(model, rows, device, encoded=None):
    model.eval()
    encoded = _encode_rows(rows) if encoded is None else encoded
    output = {}
    with torch.inference_mode():
        for phase in LL_PHASES:
            phase_indices = [index for index, row in enumerate(rows) if row["phase"] == phase]
            if not phase_indices:
                continue
            all_top1 = all_top3 = 0
            source_metrics = {}
            for source in sorted({rows[index]["sample_kind"] for index in phase_indices}):
                subset_indices = [index for index in phase_indices if rows[index]["sample_kind"] == source]
                top1 = top3 = 0
                for offset in range(0, len(subset_indices), 512):
                    indices = subset_indices[offset:offset + 512]
                    features = encoded[indices].to(device)
                    labels = torch.tensor([rows[index]["label"] for index in indices], dtype=torch.long, device=device)
                    logits, _values = model(features)
                    ranked = logits.topk(3, dim=1).indices
                    top1 += int((ranked[:, 0] == labels).sum())
                    top3 += int((ranked == labels[:, None]).any(dim=1).sum())
                source_metrics[source] = {"samples": len(subset_indices), "top1_accuracy": top1 / len(subset_indices),
                                          "top3_accuracy": top3 / len(subset_indices)}
                all_top1 += top1
                all_top3 += top3
            output[phase.lower()] = {
                "samples": len(phase_indices), "top1_accuracy": all_top1 / len(phase_indices),
                "top3_accuracy": all_top3 / len(phase_indices), "by_sample_kind": source_metrics,
            }
    return output


def _balanced_sampler(rows, seed):
    counts = Counter((row["phase"], row["sample_kind"], row.get("data_source")) for row in rows)
    weights = torch.tensor([
        1.0 / counts[(row["phase"], row["sample_kind"], row.get("data_source"))] for row in rows
    ], dtype=torch.double)
    return WeightedRandomSampler(weights, len(rows), replacement=True,
                                 generator=torch.Generator().manual_seed(seed))


def _load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _training_rows(directory, manifest):
    human_train = _load_jsonl(Path(directory) / "human_train.jsonl")
    human_validation = _load_jsonl(Path(directory) / "human_validation.jsonl")
    human_test = _load_jsonl(Path(directory) / "human_test.jsonl")
    synthetic_train = _load_jsonl(Path(directory) / "synthetic_cfop_train.jsonl")
    synthetic_validation = _load_jsonl(Path(directory) / "synthetic_cfop_validation.jsonl")
    synthetic_test = _load_jsonl(Path(directory) / "synthetic_cfop_test.jsonl")
    zbll_train = _load_jsonl(Path(directory) / "synthetic_zbll_train.jsonl")
    zbll_validation = _load_jsonl(Path(directory) / "synthetic_zbll_validation.jsonl")
    zbll_test = _load_jsonl(Path(directory) / "synthetic_zbll_test.jsonl")
    train_cases = set(manifest["synthetic_zbll_case_ids_by_split"]["train"])
    # Keep unseen ZBLL identities out of all gradients; their algorithm rows remain test targets.
    human_train = [row for row in human_train if row["phase"] != "ZBLL" or row["case_id"] in train_cases]
    human_validation = [row for row in human_validation if row["phase"] != "ZBLL" or row["case_id"] in train_cases]
    return (
        human_train + synthetic_train + zbll_train,
        human_validation + synthetic_validation + zbll_validation,
        human_test, synthetic_test + zbll_test,
    )


def train(data_dir=DATA_DIR, checkpoint_path=DEFAULT_CHECKPOINT, metrics_path=DEFAULT_METRICS,
          epochs=24, seeds=(73, 74, 75), device_name="auto", generate=False,
          train_solves=1200, validation_solves=160, test_solves=160):
    if epochs < 7 or not seeds:
        raise ValueError("Use at least seven epochs and one training seed")
    data_dir = Path(data_dir)
    manifest_path = data_dir / "manifest.json"
    if generate or not manifest_path.exists():
        manifest = generate_data(data_dir, seed=seeds[0], train_solves=train_solves,
                                validation_solves=validation_solves, test_solves=test_solves)
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_rows, validation_rows, human_test, synthetic_test = _training_rows(data_dir, manifest)
    if not train_rows or not validation_rows or not synthetic_test:
        raise ValueError("Last-layer train/validation/test splits must be non-empty")
    device = choose_device(device_name)
    torch.set_num_threads(2)
    hardware = {"python": platform.python_version(), "platform": platform.platform(),
                "torch": str(torch.__version__)}
    train_data = _features(train_rows, device)
    validation_features = _encode_rows(validation_rows)
    human_test_features = _encode_rows(human_test)
    synthetic_test_features = _encode_rows(synthetic_test)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    seed_results, best = [], None
    for seed in seeds:
        set_seed(seed)
        model = SolverPolicyValue().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        best_loss, best_state, best_epoch, stale, history = float("inf"), None, 0, 0, []
        for epoch in range(1, epochs + 1):
            active = ("PLL",) if epoch <= 3 else (("OLL", "PLL") if epoch <= 6 else LL_PHASES)
            selected_indices = [index for index, row in enumerate(train_rows) if row["phase"] in active]
            active_rows = [train_rows[index] for index in selected_indices]
            sampler = _balanced_sampler(active_rows, seed + epoch)
            loader = DataLoader(torch.utils.data.Subset(train_data, selected_indices), batch_size=512,
                                sampler=sampler, num_workers=0)
            model.train()
            total_loss = total = 0
            started = time.perf_counter()
            for features, labels, values in loader:
                features, labels, values = features.to(device), labels.to(device), values.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits, value_estimate = model(features)
                policy_loss = torch.nn.functional.cross_entropy(logits, labels)
                value_loss = torch.nn.functional.smooth_l1_loss(value_estimate, values)
                loss = policy_loss + 0.1 * value_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.detach()) * len(labels)
                total += len(labels)
            metrics = _score(model, validation_rows, device, validation_features)
            macro_loss = 0.0
            count = 0
            with torch.inference_mode():
                for phase in LL_PHASES:
                    selected_indices = [index for index, row in enumerate(validation_rows) if row["phase"] == phase]
                    if selected_indices:
                        losses = []
                        for offset in range(0, len(selected_indices), 512):
                            indices = selected_indices[offset:offset + 512]
                            features = validation_features[indices].to(device)
                            labels = torch.tensor([validation_rows[index]["label"] for index in indices],
                                                  dtype=torch.long, device=device)
                            logits, _values = model(features)
                            losses.append(float(torch.nn.functional.cross_entropy(logits, labels, reduction="sum")))
                        macro_loss += sum(losses) / len(selected_indices)
                        count += 1
            macro_loss /= max(1, count)
            record = {"epoch": epoch, "active_phases": list(active), "train_loss": total_loss / max(1, total),
                      "validation_macro_cross_entropy": macro_loss, "validation": metrics,
                      "seconds": time.perf_counter() - started}
            history.append(record)
            if epoch >= 7 and macro_loss < best_loss:
                best_loss, best_epoch, stale = macro_loss, epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            elif epoch >= 7:
                stale += 1
                if stale >= 6:
                    break
        model.load_state_dict(best_state)
        seed_metrics = {
            "seed": seed, "best_epoch": best_epoch, "validation_cross_entropy": best_loss,
            "validation": _score(model, validation_rows, device, validation_features),
            "human_solver_held_out_action_agreement": _score(model, human_test, device, human_test_features),
            "synthetic_held_out_policy_accuracy": _score(model, synthetic_test, device, synthetic_test_features),
            "history": history,
        }
        seed_results.append(seed_metrics)
        seed_checkpoint = checkpoint_path.with_name("{}-seed{}.pt".format(checkpoint_path.stem, seed))
        torch.save({
            "schema_version": 1, "model_state": best_state, "feature_size": SOLVER_FEATURE_SIZE,
            "actions": len(SIMULATOR_MOVE_TOKENS), "phase_names": LL_PHASES,
            "seed": seed, "best_epoch": best_epoch, "sample_kind": "mixed:expert+synthetic",
            "data_sources": ["reco.nz 3x3 CFOP reconstructions", CFOP_SOURCE, ZBLL_SOURCE],
            "data_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "hardware": hardware,
            "training_config": {"epochs": epochs, "curriculum": ["PLL", "OLL+PLL", "OLL+PLL+ZBLL"],
                                "loss": "cross_entropy + 0.1*smooth_l1 remaining moves",
                                "balanced_by_phase_and_source": True},
        }, seed_checkpoint)
        seed_metrics["checkpoint"] = str(seed_checkpoint)
        if best is None or best_loss < best["loss"]:
            best = {"loss": best_loss, "state": best_state, "model": model, "seed": seed,
                    "best_epoch": best_epoch, "seed_metrics": seed_metrics}
        print(json.dumps({key: value for key, value in seed_metrics.items() if key != "history"}), flush=True)

    torch.save({
        "schema_version": 1, "model_state": best["state"], "feature_size": SOLVER_FEATURE_SIZE,
        "actions": len(SIMULATOR_MOVE_TOKENS), "phase_names": LL_PHASES,
        "seed": best["seed"], "best_epoch": best["best_epoch"],
        "sample_kind": "mixed:expert+synthetic",
        "data_sources": ["reco.nz 3x3 CFOP reconstructions", CFOP_SOURCE, ZBLL_SOURCE],
        "data_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "hardware": hardware,
        "training_config": {"epochs": epochs, "curriculum": ["PLL", "OLL+PLL", "OLL+PLL+ZBLL"],
                            "loss": "cross_entropy + 0.1*smooth_l1 remaining moves",
                            "balanced_by_phase_and_source": True},
    }, checkpoint_path)
    report = {
        "task": "expert-aware neural OLL/PLL/ZBLL action and value prediction",
        "model": "random-initialized SolverPolicyValue MLP; 920 inputs, 256/128 hidden units, 72 move logits and value",
        "training_data": {"manifest": str(manifest_path), "human_actions_by_phase": dict(Counter(row["phase"] for row in train_rows if row["sample_kind"] == "expert")),
                          "synthetic_actions_by_phase": dict(Counter(row["phase"] for row in train_rows if row["sample_kind"] == "synthetic")),
                          "human_train_solvers": len({row["solver_id"] for row in train_rows if row.get("solver_id")}),
                          "human_test_solvers": len({row["solver_id"] for row in human_test if row.get("solver_id")}),
                          "test_human_solver_ids_disjoint": not bool(
                              {row["solver_id"] for row in train_rows if row.get("solver_id")}
                              & {row["solver_id"] for row in human_test if row.get("solver_id")}),
                          "synthetic_train_data_separate": True},
        "seed_results": seed_results, "selected_seed": best["seed"],
        "selected_validation": best["seed_metrics"]["validation"],
        "selected_human_solver_held_out_action_agreement": best["seed_metrics"]["human_solver_held_out_action_agreement"],
        "selected_synthetic_held_out_policy_accuracy": best["seed_metrics"]["synthetic_held_out_policy_accuracy"],
        "checkpoint": str(checkpoint_path), "device": str(device),
        "hardware": hardware,
        "human_execution_or_recognition_time": "not measured",
    }
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "seed_results"}, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--seeds", type=int, nargs="+", default=[73, 74, 75])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--train-solves", type=int, default=1200)
    parser.add_argument("--validation-solves", type=int, default=160)
    parser.add_argument("--test-solves", type=int, default=160)
    args = parser.parse_args(argv)
    train(args.data_dir, args.checkpoint, args.metrics, args.epochs, tuple(args.seeds), args.device,
          args.generate, args.train_solves, args.validation_solves, args.test_solves)


if __name__ == "__main__":
    main()
