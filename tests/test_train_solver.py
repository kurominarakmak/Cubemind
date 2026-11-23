import json

import torch

from cubemind.environment.cube import inverse_sequence, parse_moves
from cubemind.train_solver import CFOP_SOURCE, _generate_dagger


def test_dagger_keeps_verified_splits_and_skips_cross_failures(tmp_path, monkeypatch):
    dataset = tmp_path / "teacher.jsonl"
    records = []
    for index in range(60):
        scramble = "R" if index == 0 else " ".join(["U"] * index)
        records.append({
            "scramble_id": "train-{:05d}".format(index), "scramble": scramble,
            "sample_kind": "synthetic", "data_source": CFOP_SOURCE,
            "split": "train", "phases": {}, "reference_verified": True,
        })
    for index in range(10):
        records.append({
            "scramble_id": "validation-{:05d}".format(index),
            "scramble": " ".join(["U'"] * (index + 1)),
            "sample_kind": "synthetic", "data_source": CFOP_SOURCE,
            "split": "validation", "phases": {}, "reference_verified": True,
        })
    dataset.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")

    monkeypatch.setattr("cubemind.train_solver.choose_device", lambda _name: torch.device("cpu"))
    monkeypatch.setattr("cubemind.train_solver.model_from_checkpoint",
                        lambda *_args: (object(), {"model_config": {}}))

    def cross_search(start, _history, _model, _length, _device, _mode,
                     _width, _depth, _nodes, _seen):
        return start, [], [], 1, "search_budget:CROSS"

    monkeypatch.setattr("cubemind.train_solver._solve_cross_model", cross_search)
    teacher_calls = []

    def teacher(request):
        teacher_calls.extend(request["scrambles"])
        return [{
            "reference_verified": True,
            "phases": {"CROSS": [], "F2L": inverse_sequence(parse_moves(scramble)), "OLL": [], "PLL": []},
        } for scramble in request["scrambles"]]

    monkeypatch.setattr("cubemind.train_solver.teacher_request", teacher)
    dagger_path = tmp_path / "dagger.jsonl"
    generated = _generate_dagger(dataset, dagger_path, 79)
    manifest = json.loads((tmp_path / "dagger_manifest.json").read_text(encoding="utf-8"))

    assert len(generated) == 55  # 48 train + 8 validation, less one failed Cross.
    assert len(teacher_calls) == len(generated)
    assert manifest["train_trajectories"] == 47
    assert manifest["validation_trajectories"] == 8
    assert manifest["cross_failures"] == ["train-00000"]
    assert all(row["sample_kind"] == "synthetic" and row["reference_verified"] for row in generated)
    assert generated[0]["scramble"] == "U"
    assert generated[0]["prefix_history"] == []
