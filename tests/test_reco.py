import json

import pytest

from cubemind.data import reco
from cubemind.data.reco import (
    _parse_reconstruction_moves,
    examples_from_record,
    next_index_page,
    parse_index,
    parse_reconstruction,
)


def test_reco_index_filters_to_3x3_cfop_rows():
    html = """
    <table>
      <tr class='solve-row' data-id='1'><td>1</td><td>3x3</td><td>2.0</td><td>Alex</td><td>CFOP</td><td>2025-01-01</td><td></td><td></td><td>30</td><td>15</td><td>R</td></tr>
      <tr class='solve-row' data-id='2'><td>2</td><td>3x3 OH</td><td>4.0</td><td>Alex</td><td>CFOP</td><td>2025-01-01</td><td></td><td></td><td>40</td><td>10</td><td>R</td></tr>
      <tr class='solve-row' data-id='3'><td>3</td><td>3x3</td><td>3.0</td><td>Sam</td><td>Roux</td><td>2025-01-01</td><td></td><td></td><td>30</td><td>10</td><td>R</td></tr>
    </table>
    """
    rows = parse_index(html)
    assert [row["solve_id"] for row in rows] == ["1"]
    assert [row["solve_id"] for row in parse_index(html, method=None)] == ["1", "3"]


def test_reco_pagination_walks_next_links_and_checkpoints_records(monkeypatch, tmp_path):
    def index_row(solve_id):
        return "<tr class='solve-row' data-id='{}'><td>1</td><td>3x3</td><td>2</td><td>A</td><td>CFOP</td><td>2025-01-01</td><td></td><td></td><td>30</td><td>15</td><td>R</td></tr>".format(solve_id)

    first = "<table>{}</table><div class='pagination'><a href='?page=2'>Next</a></div>".format(index_row("1"))
    second = "<table>{}</table><div class='pagination'><a href='?page=1'>Previous</a></div>".format(index_row("2"))
    detail = "<div id='reconstruction'>U<br><br><br>U' // cross</div>"

    def fake_fetch(url):
        if "/solve/" in url:
            return detail
        return second if "page=2" in url else first

    monkeypatch.setattr(reco, "_fetch", fake_fetch)
    checkpoint = tmp_path / "data" / "partial.jsonl"
    records, rejected = reco.scrape_reconstructions(
        delay_seconds=0, workers=2, checkpoint_path=checkpoint)

    assert next_index_page(first) == 2
    assert next_index_page(second) is None
    assert [record["solve_id"] for record in records] == ["1", "2"]
    assert rejected == []
    assert len(checkpoint.read_text().splitlines()) == 2


def test_reco_page_parser_keeps_phase_annotations_and_validates_replay():
    row = {"solve_id": "test", "solver": "Test Solver", "date": "2025-01-01",
           "result": "2.00", "move_count_source": "2", "tps_source": "1.0", "reconstructor": "Test"}
    html = """
    <div id='reconstruction'>R<br><br><br>x // inspection<br>R' // cross</div>
    """
    record = parse_reconstruction(html, row)
    assert record["event"] == "3x3"
    assert record["solver_cross_color_range"] == "unknown"
    assert record["segments"] == [
        {"phase": "inspection", "moves": "x"},
        {"phase": "cross", "moves": "R'"},
    ]


def test_reco_reconstruction_preserves_non_cfop_method():
    row = {"solve_id": "roux", "solver": "Test Solver", "method": "Roux",
           "date": "2025-01-01", "result": "2.00", "move_count_source": "2",
           "tps_source": "1.0", "reconstructor": "Test"}
    record = parse_reconstruction("<div id='reconstruction'>U<br><br><br>U' // solve</div>", row)
    assert record["method"] == "ROUX"


def test_reco_move_parser_accepts_parentheses_compact_turns_and_three_turn_suffix():
    assert _parse_reconstruction_moves("U'D (R U R')2 L3' R3") == [
        "U'", "D", "R", "U", "R'", "R", "U", "R'", "L", "R'",
    ]


def test_f2l_action_rows_keep_phase_color_and_source_provenance():
    row = {"solve_id": "f2l-test", "solver": "Test Solver", "date": "2025-01-01",
           "result": "2.00", "move_count_source": "1", "tps_source": "1.0", "reconstructor": "Test"}
    record = parse_reconstruction(
        "<div id='reconstruction'>U<br><br><br>U' // 1st/2nd pairs<br></div>", row)
    samples = examples_from_record(record)
    assert len(samples) == 1
    assert samples[0]["phase"] == "F2L"
    assert samples[0]["strategy_context"] == "multi-pair annotation"
    assert samples[0]["data_source"] == "reco.nz"
    assert samples[0]["cross_color"] == "white"


def test_human_solver_splits_reject_solver_overlap(monkeypatch, tmp_path):
    from cubemind.train_hierarchical_f2l import _human_splits

    monkeypatch.setattr("cubemind.train_hierarchical_f2l.RECO", tmp_path)
    row = {"sample_kind": "expert", "data_source": "reco.nz", "solver_id": "same-cuber"}
    (tmp_path / "train.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "validation.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "test.jsonl").write_text("")
    with pytest.raises(ValueError, match="multiple dataset splits"):
        _human_splits()
