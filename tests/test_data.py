import json

import pytest

from cubemind.data.prepare import _examples_from_record, load_expert_records, make_synthetic_records, prepare
from cubemind.training.engine import CrossDataset, load_splits


def test_synthetic_fixture_is_provenance_labeled_and_group_split(tmp_path):
    records = make_synthetic_records(30, seed=3)
    manifest = prepare(tmp_path, records, seed=3)
    assert manifest["sample_kinds"] == ["synthetic"]
    datasets = load_splits(tmp_path / "train.jsonl", tmp_path / "validation.jsonl", tmp_path / "test.jsonl")
    assert all(row["sample_kind"] == "synthetic" for split in datasets.values() for row in split.rows)
    assert all(row["cross_face"] == "D" and row["cross_color"] == "white"
               for split in datasets.values() for row in split.rows)
    assert len({row["scramble_id"] for row in datasets["train"].rows}.intersection(
        row["scramble_id"] for row in datasets["test"].rows)) == 0
    assert CrossDataset(tmp_path / "test.jsonl").rows


def test_expert_import_requires_recorded_reuse_permission(tmp_path):
    record = {
        "event": "3x3",
        "solver_id": "cubing-person",
        "solve_id": "solve-1",
        "scramble_id": "scramble-1",
        "scramble": "R U F",
        "solution_moves": "F' U' R'",
        "solution_scope": "FULL_SOLVE",
        "cross_face": "D",
        "cross_color": "white",
        "solver_cross_color_range": "white_only",
        "data_source": "https://example.invalid/reconstruction",
        "reuse_permission": "permission from rights holder",
        "reuse_permission_verified": False,
    }
    path = tmp_path / "raw.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="permission-verified"):
        load_expert_records(path)
    record["reuse_permission_verified"] = True
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    loaded = load_expert_records(path)
    samples = _examples_from_record(loaded[0])
    assert samples and all(sample["event"] == "3x3" and sample["sample_kind"] == "expert"
                           and sample["solver_cross_color_range"] == "white_only"
                           for sample in samples)
    record["cross_color"] = "yellow"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="white Down Cross"):
        load_expert_records(path)
    record["cross_color"] = "white"
    record["solver_cross_color_range"] = "yellow_only"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="yellow-only"):
        load_expert_records(path)
    record["solver_cross_color_range"] = "white_only"
    record["event"] = "OH"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="only 3x3"):
        load_expert_records(path)
