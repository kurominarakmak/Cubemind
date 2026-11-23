"""Train an expert-only pair selector and calibrate action-policy fusion."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from cubemind.environment.cube import SIMULATOR_MOVE_TOKENS
from cubemind.expert_aware_f2l import (
    ACTION_COUNT, ExpertAwareF2LPolicy, encode_expert_features,
    load_canonical_expert_action_model, load_expert_action_model,
)
from cubemind.hierarchical_f2l import _legal_moves, load_hierarchical_f2l
from cubemind.models.expert_pair import ExpertPairSelector
from cubemind.models.f2l import F2LPolicy
from cubemind.models.hierarchical_f2l import HIERARCHICAL_F2L_FEATURE_SIZE, encode_f2l_features
from cubemind.train_hierarchical_f2l import RECO, SOURCE_HUMAN, human_trajectories, _human_splits
from cubemind.training.engine import choose_device, set_seed

ROOT = Path(__file__).resolve().parents[1]
PAIR_CHECKPOINT = ROOT / "artifacts" / "expert-aware-pair-selector.pt"
REPORT = ROOT / "artifacts" / "expert-aware-f2l-evaluation.json"
ACTION_CHECKPOINT = ROOT / "artifacts" / "reco-cfop-context.pt"
CANONICAL_CHECKPOINT = ROOT / "artifacts" / "canonical-expert-f2l.pt"
PLANNER_CHECKPOINT = ROOT / "artifacts" / "hierarchical-f2l.pt"
TEMPERATURES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _features(rows, selector=False):
    return torch.stack([
        encode_f2l_features(row["state"], row["history"], row["protected_mask"],
                            None if selector else row["target_pair"], row["move_count"])
        for row in rows
    ]) if rows else torch.empty((0, HIERARCHICAL_F2L_FEATURE_SIZE), dtype=torch.float32)


def _human_data():
    splits = _human_splits()
    result = {}
    for name, rows in splits.items():
        actions, selectors = human_trajectories(rows, name)
        result[name] = {"actions": actions, "selectors": selectors}
    return result


def _train_pair_selector(train_rows, validation_rows, seed, epochs, device):
    if not train_rows or not validation_rows:
        raise ValueError("Expert pair-selector training needs train and validation labels")
    set_seed(seed)
    model = ExpertPairSelector().to(device)
    train_x, validation_x = _features(train_rows, True), _features(validation_rows, True).to(device)
    train_y = torch.tensor([row["target_pair"] for row in train_rows], dtype=torch.long)
    validation_y = torch.tensor([row["target_pair"] for row in validation_rows], dtype=torch.long, device=device)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=256, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=0.0001)
    best, best_loss, best_epoch, stale = None, float("inf"), 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(features.to(device)), labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            logits = model(validation_x)
            validation_loss = float(F.cross_entropy(logits, validation_y))
        if validation_loss < best_loss:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale == 4:
                break
    model.load_state_dict(best)
    return model, best_epoch, best_loss


def _train_action_policy(train_rows, validation_rows, seed, epochs, device):
    if not train_rows or not validation_rows:
        raise ValueError("Expert action training needs train and validation labels")
    if any(row.get("sample_kind") != "expert" or row.get("data_source") != SOURCE_HUMAN
           for row in train_rows + validation_rows):
        raise ValueError("Canonical action training accepts only source-labeled human rows")
    set_seed(seed)
    model = F2LPolicy(HIERARCHICAL_F2L_FEATURE_SIZE, ACTION_COUNT).to(device)
    train_x, validation_x = _features(train_rows), _features(validation_rows).to(device)
    train_y = torch.tensor([row["move_id"] for row in train_rows], dtype=torch.long)
    validation_y = torch.tensor([row["move_id"] for row in validation_rows], dtype=torch.long, device=device)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=512, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    best, best_loss, best_epoch, stale = None, float("inf"), 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(features.to(device)), labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_loss = float(F.cross_entropy(model(validation_x), validation_y))
        if validation_loss < best_loss:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale == 4:
                break
    model.load_state_dict(best)
    return model, best_epoch, best_loss


def _pair_metrics(model, rows, device, planner=False):
    if not rows:
        return {"samples": 0, "top1": None, "top2": None, "cross_entropy": None}
    model.eval()
    features = _features(rows, True).to(device)
    labels = torch.tensor([row["target_pair"] for row in rows], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model(features)
        logits = output[2] if planner else output
        top = logits.topk(2, dim=1).indices
        return {
            "samples": len(rows), "top1": float((top[:, 0] == labels).float().mean()),
            "top2": float((top == labels[:, None]).any(dim=1).float().mean()),
            "cross_entropy": float(F.cross_entropy(logits, labels)),
        }


def _action_tensors(rows, expert, planner, device, canonical_expert=False):
    kept = [row for row in rows if row["move_id"] in _legal_moves(row["history"])]
    if not kept:
        raise ValueError("No held-out human actions overlap the planner's legal outer-turn set")
    features = _features(kept, False)
    labels = torch.tensor([row["move_id"] for row in kept], dtype=torch.long, device=device)
    legal = torch.zeros((len(kept), ACTION_COUNT), dtype=torch.bool, device=device)
    for index, row in enumerate(kept):
        legal[index, _legal_moves(row["history"])] = True
    expert_logits, planner_logits = [], []
    with torch.inference_mode():
        for start in range(0, len(kept), 512):
            batch = features[start:start + 512].to(device)
            p_logits, _values, _pairs = planner(batch)
            e_logits = expert(batch if canonical_expert else encode_expert_features(batch))
            expert_logits.append(e_logits)
            planner_logits.append(p_logits)
    return kept, torch.cat(expert_logits), torch.cat(planner_logits), labels, legal


def _masked_log_probs(logits, legal, temperature):
    return F.log_softmax((logits / temperature).masked_fill(~legal, -torch.inf), dim=-1)


def _action_metrics(expert_logits, planner_logits, labels, legal, expert_temperature,
                    planner_temperature, expert_weight):
    expert_logp = _masked_log_probs(expert_logits, legal, expert_temperature)
    planner_logp = _masked_log_probs(planner_logits, legal, planner_temperature)
    if expert_weight <= 0:
        fused = planner_logp
    elif expert_weight >= 1:
        fused = expert_logp
    else:
        fused = F.log_softmax(expert_weight * expert_logp + (1.0 - expert_weight) * planner_logp, dim=-1)
    output = {}
    for name, scores in (("expert", expert_logp), ("planner", planner_logp), ("fused", fused)):
        top = scores.argmax(dim=-1)
        output[name] = {
            "cross_entropy": float(F.nll_loss(scores, labels)),
            "top1": float((top == labels).float().mean()),
            "top3": float((scores.topk(3, dim=-1).indices == labels[:, None]).any(dim=1).float().mean()),
        }
    output["top1_prediction_agreement"] = float(
        (expert_logp.argmax(dim=-1) == planner_logp.argmax(dim=-1)).float().mean())
    return output


def _adaptive_action_metrics(expert_logits, planner_logits, labels, legal,
                             expert_temperature, planner_temperature, scale):
    expert_logp = _masked_log_probs(expert_logits, legal, expert_temperature)
    planner_logp = _masked_log_probs(planner_logits, legal, planner_temperature)
    entropy = -(expert_logp.exp() * expert_logp).sum(-1)
    confidence = (1.0 - entropy / torch.log(legal.sum(-1).float())).clamp(0, 1)
    weight = (scale * confidence).unsqueeze(-1)
    scores = planner_logp.clone()
    mixed = weight[:, 0].gt(0) & weight[:, 0].lt(1)
    scores[mixed] = F.log_softmax(
        weight[mixed] * expert_logp[mixed] + (1.0 - weight[mixed]) * planner_logp[mixed], dim=-1)
    fully_expert = weight[:, 0].ge(1)
    scores[fully_expert] = expert_logp[fully_expert]
    top = scores.argmax(dim=-1)
    return {"cross_entropy": float(F.nll_loss(scores, labels)),
            "top1": float((top == labels).float().mean()),
            "top3": float((scores.topk(3, dim=-1).indices == labels[:, None]).any(dim=1).float().mean())}


def _calibrate(validation_tensors):
    _rows, expert_logits, planner_logits, labels, legal = validation_tensors
    expert_temperature_nll = {str(value): _action_metrics(
        expert_logits, planner_logits, labels, legal, value, 1.0, 1.0)["expert"]["cross_entropy"]
        for value in TEMPERATURES}
    planner_temperature_nll = {str(value): _action_metrics(
        expert_logits, planner_logits, labels, legal, 1.0, value, 0.0)["planner"]["cross_entropy"]
        for value in TEMPERATURES}
    expert_temperature = min(TEMPERATURES, key=lambda value: _action_metrics(
        expert_logits, planner_logits, labels, legal, value, 1.0, 1.0)["expert"]["cross_entropy"])
    planner_temperature = min(TEMPERATURES, key=lambda value: _action_metrics(
        expert_logits, planner_logits, labels, legal, 1.0, value, 0.0)["planner"]["cross_entropy"])
    expert_weight = min(WEIGHTS, key=lambda value: _action_metrics(
        expert_logits, planner_logits, labels, legal, expert_temperature,
        planner_temperature, value)["fused"]["cross_entropy"])
    weight_validation_nll = {str(value): _action_metrics(
        expert_logits, planner_logits, labels, legal, expert_temperature,
        planner_temperature, value)["fused"]["cross_entropy"] for value in WEIGHTS}
    expert_logp = _masked_log_probs(expert_logits, legal, expert_temperature)
    adaptive_losses = []
    for scale in WEIGHTS:
        metrics = _adaptive_action_metrics(expert_logits, planner_logits, labels, legal,
                                           expert_temperature, planner_temperature, scale)
        adaptive_losses.append((metrics["cross_entropy"], scale))
    adaptive_loss, adaptive_weight = min(adaptive_losses)
    return {
        "expert_temperature": expert_temperature, "planner_temperature": planner_temperature,
        "expert_weight": expert_weight,
        "validation_temperature_nll": {"expert": expert_temperature_nll,
                                        "planner": planner_temperature_nll},
        "validation_weight_nll": weight_validation_nll,
        "adaptive_entropy_weight_scale": adaptive_weight,
        "adaptive_entropy_validation_nll": adaptive_loss,
    }


def train(pair_checkpoint=PAIR_CHECKPOINT, report_path=REPORT, action_checkpoint=ACTION_CHECKPOINT,
          planner_checkpoint=PLANNER_CHECKPOINT, canonical_checkpoint=CANONICAL_CHECKPOINT,
          epochs=12, seed=1701, device_name="auto"):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    device = choose_device(device_name)
    torch.set_num_threads(2)
    data = _human_data()
    legacy_expert, _expert_meta = load_expert_action_model(action_checkpoint, device)
    planner, _planner_meta = load_hierarchical_f2l(planner_checkpoint, device)
    canonical_expert, action_epoch, action_validation_loss = _train_action_policy(
        data["train"]["actions"], data["validation"]["actions"], seed, epochs, device,
    )
    selector, best_epoch, best_validation_loss = _train_pair_selector(
        data["train"]["selectors"], data["validation"]["selectors"], seed + 1, epochs, device,
    )
    pair_test = _pair_metrics(selector, data["test"]["selectors"], device)
    validation_legacy = _action_tensors(data["validation"]["actions"], legacy_expert, planner, device)
    test_legacy = _action_tensors(data["test"]["actions"], legacy_expert, planner, device)
    validation_actions = _action_tensors(data["validation"]["actions"], canonical_expert,
                                         planner, device, canonical_expert=True)
    test_actions = _action_tensors(data["test"]["actions"], canonical_expert,
                                   planner, device, canonical_expert=True)
    calibration = _calibrate(validation_actions)
    common = (calibration["expert_temperature"], calibration["planner_temperature"], calibration["expert_weight"])
    validation_metrics = _action_metrics(*validation_actions[1:], *common)
    test_metrics = _action_metrics(*test_actions[1:], *common)
    validation_metrics["adaptive_entropy"] = _adaptive_action_metrics(
        *validation_actions[1:], calibration["expert_temperature"],
        calibration["planner_temperature"], calibration["adaptive_entropy_weight_scale"])
    test_metrics["adaptive_entropy"] = _adaptive_action_metrics(
        *test_actions[1:], calibration["expert_temperature"],
        calibration["planner_temperature"], calibration["adaptive_entropy_weight_scale"])
    validation_selector_metrics = _pair_metrics(selector, data["validation"]["selectors"], device)
    pair_train_metrics = _pair_metrics(selector, data["train"]["selectors"], device)
    planner_pair_validation = _pair_metrics(planner, data["validation"]["selectors"], device, planner=True)
    planner_pair_test = _pair_metrics(planner, data["test"]["selectors"], device, planner=True)
    canonical_checkpoint = Path(canonical_checkpoint)
    canonical_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": 1, "model_config": {"input_size": HIERARCHICAL_F2L_FEATURE_SIZE,
                                                "actions": ACTION_COUNT},
        "feature_size": HIERARCHICAL_F2L_FEATURE_SIZE, "model_state": canonical_expert.state_dict(),
        "seed": seed, "best_epoch": action_epoch, "sample_kind": "expert",
        "data_source": "reco.nz 3x3 CFOP reconstructions",
        "feature_schema": "hierarchical white-Down facelet/history/protected-pair/target-pair schema",
        "train_samples": len(data["train"]["actions"]),
        "validation_samples": len(data["validation"]["actions"]),
        "human_split_sha256": {name: hashlib.sha256((RECO / (name + ".jsonl")).read_bytes()).hexdigest()
                                for name in ("train", "validation", "test")},
    }, canonical_checkpoint)
    pair_checkpoint = Path(pair_checkpoint)
    pair_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    split_hashes = {name: hashlib.sha256((RECO / (name + ".jsonl")).read_bytes()).hexdigest()
                    for name in ("train", "validation", "test")}
    torch.save({
        "schema_version": 1, "feature_size": 916, "actions": 4,
        "model_state": selector.state_dict(), "seed": seed, "best_epoch": best_epoch,
        "sample_kind": "expert", "data_source": "reco.nz 3x3 CFOP reconstructions",
        "label_source": "simulator-verified single-pair slot transitions",
        "train_samples": len(data["train"]["selectors"]),
        "validation_samples": len(data["validation"]["selectors"]), "solver_split_sha256": split_hashes,
    }, pair_checkpoint)
    report = {
        "task": "expert-aware hierarchical F2L action fusion and expert-only pair selector",
        "sample_kind": "random-initialized human-only action and pair-order models; no simulator teacher actions",
        "data_source": "reco.nz 3x3 CFOP reconstructions; human pair labels are simulator-verified",
        "data_split": "solver-disjoint train/validation/test; validation tunes temperatures and fusion weight",
        "seed": seed, "device": str(device),
        "canonical_expert_action": {"train_samples": len(data["train"]["actions"]),
                                     "validation_samples": len(data["validation"]["actions"]),
                                     "best_epoch": action_epoch,
                                     "validation_loss": action_validation_loss,
                                     "initialization": "PyTorch random initialization"},
        "pair_selector_best_epoch": best_epoch,
        "pair_selector_validation_loss": best_validation_loss,
        "pair_selector": {"train": pair_train_metrics,
                          "validation": {"expert_only": validation_selector_metrics,
                                         "mixed_planner": planner_pair_validation},
                          "test": {"expert_only": pair_test, "mixed_planner": planner_pair_test}},
        "action_fusion": {
            "calibration": calibration,
            "validation_action_coverage": {
                "pair_labeled_human_actions": len(data["validation"]["actions"]),
                "legal_outer_turn_actions": len(validation_actions[0]),
                "fraction": len(validation_actions[0]) / max(1, len(data["validation"]["actions"]))},
            "test_action_coverage": {
                "pair_labeled_human_actions": len(data["test"]["actions"]),
                "legal_outer_turn_actions": len(test_actions[0]),
                "fraction": len(test_actions[0]) / max(1, len(data["test"]["actions"]))},
            "validation": validation_metrics, "test": test_metrics,
            "legacy_reco_adapter_compatibility": {
                "validation": _action_metrics(*validation_legacy[1:], 2.0, 1.0, 1.0)["expert"],
                "test": _action_metrics(*test_legacy[1:], 2.0, 1.0, 1.0)["expert"],
            },
            "legacy_reco_adapter_calibration": _calibrate(validation_legacy),
            "action_filter": "human next action must be an outer turn allowed by the planner's history pruning",
        },
        "checkpoints": {"legacy_expert_action": str(action_checkpoint),
                        "canonical_expert_action": str(canonical_checkpoint),
                        "planner": str(planner_checkpoint), "expert_pair_selector": str(pair_checkpoint)},
        "expert_action_checkpoint_sha256": hashlib.sha256(Path(action_checkpoint).read_bytes()).hexdigest(),
        "canonical_expert_checkpoint_sha256": hashlib.sha256(canonical_checkpoint.read_bytes()).hexdigest(),
        "planner_checkpoint_sha256": hashlib.sha256(Path(planner_checkpoint).read_bytes()).hexdigest(),
        "human_split_sha256": split_hashes,
        "execution_time": "not measured; no timing labels are present",
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-checkpoint", type=Path, default=PAIR_CHECKPOINT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--action-checkpoint", type=Path, default=ACTION_CHECKPOINT)
    parser.add_argument("--planner-checkpoint", type=Path, default=PLANNER_CHECKPOINT)
    parser.add_argument("--canonical-checkpoint", type=Path, default=CANONICAL_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    train(args.pair_checkpoint, args.report, args.action_checkpoint, args.planner_checkpoint,
          args.canonical_checkpoint, args.epochs, args.seed, args.device)


if __name__ == "__main__":
    main()
