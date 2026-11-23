import json

from cubemind.train_last_layer import _human_examples, _synthetic_zbll_examples, _training_rows
from cubemind.environment.cube import SIMULATOR_MOVE_TOKENS, SOLVED_STATE, apply_sequence
from cubemind.benchmark_last_layer import _aggregate, _move_counts, _phase_starts_from_data


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_zbll_synthetic_split_holds_out_case_identities():
    rows, splits = _synthetic_zbll_examples(73)
    train, validation, test = (set(splits[name]) for name in ("train", "validation", "test"))
    assert not (train & validation or train & test or validation & test)
    assert all(row["case_id"] in train for row in rows["train"])
    assert all(row["case_id"] in validation for row in rows["validation"])
    assert all(row["case_id"] in test for row in rows["test"])


def test_human_zbll_train_rows_cannot_leak_held_out_case_ids(tmp_path):
    human = {
        "train": [{"phase": "ZBLL", "case_id": "test-case", "solver_id": "solver-a"},
                  {"phase": "PLL", "solver_id": "solver-a"}],
        "validation": [{"phase": "ZBLL", "case_id": "train-case", "solver_id": "solver-b"}],
        "test": [{"phase": "ZBLL", "case_id": "test-case", "solver_id": "solver-c"}],
    }
    for split, rows in human.items():
        _write(tmp_path / ("human_" + split + ".jsonl"), rows)
        _write(tmp_path / ("synthetic_cfop_" + split + ".jsonl"), [])
        _write(tmp_path / ("synthetic_zbll_" + split + ".jsonl"), [])
    manifest = {"synthetic_zbll_case_ids_by_split": {
        "train": ["train-case"], "validation": ["validation-case"], "test": ["test-case"],
    }}

    train, validation, held_out_human, _synthetic_test = _training_rows(tmp_path, manifest)

    assert [row["phase"] for row in train] == ["PLL"]
    assert [row["case_id"] for row in validation] == ["train-case"]
    assert held_out_human[0]["case_id"] == "test-case"
    assert not ({row["solver_id"] for row in train if row.get("solver_id")}
                & {row["solver_id"] for row in held_out_human if row.get("solver_id")})


def test_verified_human_auf_before_pll_is_retained_as_a_pll_action(tmp_path):
    records = [{
        "reuse_permission_verified": True, "scramble": "U", "cross_color": "white",
        "solver_id": "solver-{}".format(name), "solve_id": "solve-{}".format(name),
        "scramble_id": "scramble-{}".format(name), "segments": [{"phase": "AUF", "moves": "U'"}],
    } for name in "abc"]
    path = tmp_path / "human.jsonl"
    _write(path, records)

    examples = _human_examples(path)
    rows = [row for split_rows in examples.values() for row in split_rows]

    assert len(rows) == 3
    assert all(row["phase"] == "PLL" for row in rows)
    assert all(row["phase_annotation"] == "AUF" for row in rows)
    assert all(row["label"] == SIMULATOR_MOVE_TOKENS.index("U'") for row in rows)


def test_last_layer_move_metrics_distinguish_stm_htm_and_rotations():
    assert _move_counts(["R2", "Rw", "U", "M", "x"]) == {
        "stm": 5, "htm": 2, "rotations": 1,
    }


def test_aggregate_does_not_count_non_applicable_zbll_phases_as_failures():
    row = {"verified": True, "stm": 12, "rotations": 0, "states_explored": 10,
           "runtime_seconds": 0.1, "cross_complete": True, "f2l_complete": True,
           "oll_complete": None, "pll_complete": None, "zbll_eligible": True}

    result = _aggregate([row])

    assert result["success_rate"] == 1.0
    assert result["mean_search_nodes"] == 10
    assert result["cross_success_rate"] == 1.0
    assert result["oll_success_rate"] is None
    assert result["pll_success_rate"] is None
    assert result["phase_metric_samples"]["oll_complete"] == 0


def test_controlled_stage_inputs_reuse_first_held_out_trajectory_state(tmp_path):
    oll = apply_sequence(SOLVED_STATE, "R U R' U R U2 R'")
    pll = apply_sequence(SOLVED_STATE, "U")
    _write(tmp_path / "synthetic_cfop_validation.jsonl", [
        {"trajectory_id": "a:OLL", "phase": "OLL", "state": list(oll),
         "history": [1, 2], "move_count": 20},
        {"trajectory_id": "a:OLL", "phase": "OLL", "state": list(SOLVED_STATE),
         "history": [3], "move_count": 21},
        {"trajectory_id": "a:PLL", "phase": "PLL", "state": list(pll),
         "history": [4], "move_count": 17},
    ])

    starts = _phase_starts_from_data(tmp_path, "validation")

    assert starts["OLL"] == [{"trajectory_id": "a:OLL", "state": oll,
                               "history": [1, 2], "move_count": 20}]
    assert starts["PLL"] == [{"trajectory_id": "a:PLL", "state": pll,
                               "history": [4], "move_count": 17}]
