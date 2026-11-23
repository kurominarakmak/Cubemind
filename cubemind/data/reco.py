"""Collect 3x3 solve reconstructions and prepare human CFOP actions."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import random
import re
import time
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from cubemind.environment.cube import (
    COLORS,
    FACE_ORDER,
    SIMULATOR_MOVE_TOKENS,
    SOLVED_STATE,
    apply_move,
    apply_sequence,
    cross_complete,
    f2l_slot_count,
    last_layer_edges_oriented,
    normalize_cube_orientation,
    parse_moves,
)

RAW_PATH = Path("data/raw/reco_nz/cfop_3x3.jsonl")
OUTPUT_PATH = Path("data/processed/reco_cfop")
BASE_URL = "https://reco.nz/"
MOVE_TO_ID = {move: index for index, move in enumerate(SIMULATOR_MOVE_TOKENS)}
COLOR_NAME = {0: "yellow", 1: "red", 2: "green", 3: "white", 4: "orange", 5: "blue"}
_RECO_MOVE = re.compile(r"(?:[URFDLB](?:w)?|[urfdlb]|[MES]|[xyz])(?:2'?|3'?|')?")


class _IndexParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.row = None
        self.cell = None
        self.text = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "tr" and "solve-row" in attrs.get("class", ""):
            self.row = {"id": attrs.get("data-id"), "cells": []}
        elif self.row is not None and tag == "td":
            self.cell, self.text = True, []

    def handle_data(self, data):
        if self.cell is not None:
            self.text.append(data)

    def handle_endtag(self, tag):
        if tag == "td" and self.row is not None and self.cell is not None:
            self.row["cells"].append(" ".join("".join(self.text).split()))
            self.cell, self.text = None, []
        elif tag == "tr" and self.row is not None:
            cells = self.row["cells"]
            if len(cells) >= 11 and self.row["id"]:
                self.rows.append({
                    "solve_id": self.row["id"], "event": cells[1], "result": cells[2],
                    "solver": cells[3], "method": cells[4], "date": cells[5],
                    "move_count_source": cells[8], "tps_source": cells[9],
                    "reconstructor": cells[10],
                })
            self.row = None


class _ReconstructionParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "div":
            attrs = dict(attrs)
            if self.depth:
                self.depth += 1
            elif attrs.get("id") == "reconstruction":
                self.depth = 1
        elif tag == "br" and self.depth:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "div" and self.depth:
            self.depth -= 1

    def handle_data(self, data):
        if self.depth:
            self.parts.append(data)


class _NextPageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.href = None
        self.text = []
        self.next_page = None

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.href = dict(attrs).get("href", "")
            self.text = []

    def handle_data(self, data):
        if self.href is not None:
            self.text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self.href is not None:
            if " ".join("".join(self.text).split()).lower().startswith("next"):
                match = re.search(r"[?&]page=(\d+)", self.href)
                if match:
                    self.next_page = int(match.group(1))
            self.href, self.text = None, []


def parse_index(html, method="CFOP"):
    parser = _IndexParser()
    parser.feed(html)
    return [row for row in parser.rows if row["event"] == "3x3"
            and (method is None or row["method"].upper() == method.upper())]


def _parse_reconstruction_moves(sequence):
    """Accept grouped/compact Singmaster notation used by reconstruction pages."""
    source = sequence
    for _ in range(8):
        expanded = re.sub(r"\(([^()]*)\)(\d+)",
                          lambda match: " ".join([match.group(1)] * int(match.group(2))), source)
        if expanded == source:
            break
        source = expanded
    if source.count("(") != source.count(")"):
        raise ValueError("Unbalanced or unsupported grouped move notation")
    source = source.replace("(", " ").replace(")", " ")
    moves = []
    for chunk in source.split():
        tokens = list(_RECO_MOVE.finditer(chunk))
        if "".join(token.group() for token in tokens) != chunk:
            raise ValueError("Moves must use standard face, wide, slice, or rotation notation")
        for token in (match.group() for match in tokens):
            if token.endswith("3'"):
                moves.append(token[:-2])
            else:
                moves.extend(parse_moves([token]))
    return moves


def next_index_page(html):
    parser = _NextPageParser()
    parser.feed(html)
    return parser.next_page


def parse_reconstruction(html, index_row):
    parser = _ReconstructionParser()
    parser.feed(html)
    lines = [line.strip() for line in "".join(parser.parts).splitlines() if line.strip()]
    if not lines:
        raise ValueError("reconstruction page has no move list")
    scramble = " ".join(_parse_reconstruction_moves(lines[0]))
    segments = []
    for line in lines[1:]:
        if "//" not in line:
            continue
        move_text, phase = line.split("//", 1)
        moves = _parse_reconstruction_moves(move_text.strip())
        if moves:
            segments.append({"phase": phase.strip(), "moves": " ".join(moves)})
    if not segments:
        raise ValueError("reconstruction page has no phase-annotated solution moves")
    all_moves = [move for segment in segments for move in parse_moves(segment["moves"])]
    state = apply_sequence(SOLVED_STATE, scramble)
    state_after_inspection = state
    for segment in segments:
        moves = parse_moves(segment["moves"])
        if "inspection" in segment["phase"].lower():
            state_after_inspection = apply_sequence(state_after_inspection, moves)
        state = apply_sequence(state, moves)
    final_state = normalize_cube_orientation(state)
    if final_state != SOLVED_STATE:
        raise ValueError("complete reconstruction does not replay to a solved cube")
    bottom_color = state_after_inspection[COLORS["D"] * 9 + 4]
    scramble_text = " ".join(parse_moves(scramble))
    return {
        "event": "3x3", "method": index_row.get("method", "CFOP").upper(), "solver_id": index_row["solver"],
        "solver_name": index_row["solver"], "solve_id": index_row["solve_id"],
        "scramble_id": hashlib.sha256(scramble_text.encode("utf-8")).hexdigest()[:16],
        "scramble": scramble_text, "segments": segments,
        "solution_moves": " ".join(all_moves), "solution_scope": "FULL_SOLVE",
        "cross_face": "D", "cross_color": COLOR_NAME[bottom_color],
        "solver_cross_color_range": "unknown", "sample_kind": "expert",
        "data_source": "reco.nz", "source_url": BASE_URL + "solve/" + index_row["solve_id"],
        "source_date": index_row["date"], "source_result_seconds": _seconds(index_row["result"]),
        "source_move_count": _integer(index_row["move_count_source"]),
        "source_tps": _number(index_row["tps_source"]), "reconstructor": index_row["reconstructor"],
        "reuse_permission": "User-authorized local research training; raw records are not redistributed.",
        "reuse_permission_verified": True,
    }


def _seconds(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _number(value):
    return _seconds(value)


def _integer(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _fetch(url):
    request = Request(url, headers={"User-Agent": "CubeMind research prototype/0.1"})
    for attempt in range(4):
        try:
            with urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8", errors="replace")
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
        except (URLError, TimeoutError, OSError):
            if attempt == 3:
                raise
        time.sleep(min(0.5 * 2 ** attempt, 4))


def scrape_reconstructions(pages=None, max_solves=None, delay_seconds=0.05, workers=4,
                           cached_records=(), checkpoint_path=None, all_methods=False):
    if pages is not None and pages < 1 or max_solves is not None and max_solves < 1 or workers < 1:
        raise ValueError("pages, max_solves, and workers must be positive when provided")
    filters = {"sort": "result_asc", "3x3": "on", "official": "on", "unofficial": "on"}
    if not all_methods:
        filters["CFOP"] = "on"
    query = urlencode(filters)
    rows, seen = [], set()
    page, pages_read = 1, 0
    while pages is None or pages_read < pages:
        url = BASE_URL + "?" + query + ("&page={}".format(page) if page > 1 else "")
        html = _fetch(url)
        for row in parse_index(html, method=None if all_methods else "CFOP"):
            if row["solve_id"] not in seen:
                seen.add(row["solve_id"])
                rows.append(row)
                if max_solves is not None and len(rows) >= max_solves:
                    break
        pages_read += 1
        if pages_read % 25 == 0:
            print("Index pages read: {} · 3×3 CFOP entries: {}".format(pages_read, len(rows)), flush=True)
        if max_solves is not None and len(rows) >= max_solves:
            break
        following_page = next_index_page(html)
        if following_page is None:
            break
        if following_page <= page:
            raise ValueError("reco.nz pagination did not advance")
        page = following_page
        if delay_seconds:
            time.sleep(delay_seconds)

    records_by_id = {record["solve_id"]: record for record in cached_records}
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    if checkpoint and checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records_by_id[record["solve_id"]] = record
    missing = [row for row in rows if row["solve_id"] not in records_by_id]
    rejected, completed = [], 0
    if checkpoint:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint.open("a", encoding="utf-8") if checkpoint else None
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_load_reconstruction, row): row for row in missing}
            for future in as_completed(futures):
                row = futures[future]
                try:
                    record = future.result()
                    records_by_id[row["solve_id"]] = record
                    if checkpoint_file:
                        checkpoint_file.write(json.dumps(record, separators=(",", ":")) + "\n")
                except (ValueError, URLError, TimeoutError, OSError) as error:
                    rejected.append({"solve_id": row["solve_id"], "reason": str(error)})
                completed += 1
                if checkpoint_file and completed % 50 == 0:
                    checkpoint_file.flush()
                if completed % 250 == 0:
                    print("Reconstruction pages checked: {} / {} · valid cached: {} · rejected: {}".format(
                        completed, len(missing), len(records_by_id), len(rejected)), flush=True)
    finally:
        if checkpoint_file:
            checkpoint_file.flush()
            checkpoint_file.close()
    ordered = [records_by_id[row["solve_id"]] for row in rows if row["solve_id"] in records_by_id]
    return ordered, rejected


def row_url(row):
    return BASE_URL + "solve/" + row["solve_id"]


def _load_reconstruction(row):
    return parse_reconstruction(_fetch(row_url(row)), row)


def _sample_phase(annotation):
    value = annotation.lower()
    if "inspection" in value:
        return None
    if "zbls" in value:
        return "ZBLS"
    if value.strip() == "cross":
        return "CROSS"
    if "xcross" in value or "x-cross" in value or "cross+1" in value:
        return "CROSS+1"
    if any(key in value for key in ("pair", "slot", "f2l", "pseudo", "xcross", "cross+1")):
        return "F2L"
    return None


def _strategy_context(annotation):
    value = annotation.lower()
    if "zbls" in value:
        return "zbls"
    if "pseudo" in value:
        return "pseudo-slotting"
    if "xcross" in value or "cross+1" in value:
        return "cross+1"
    if "1st/2nd" in value or "1st + 2nd" in value or "pair" in value and "pairs" in value:
        return "multi-pair annotation"
    return "standard F2L"


def examples_from_record(record, history_length=8):
    if record.get("event") != "3x3" or record.get("method", "CFOP").upper() != "CFOP":
        raise ValueError("Only 3x3 CFOP records are accepted")
    state = apply_sequence(SOLVED_STATE, record["scramble"])
    history = []
    samples = []
    move_count = 0
    for segment in record["segments"]:
        phase = segment["phase"].strip()
        target_phase = _sample_phase(phase)
        strategy = _strategy_context(phase)
        for move in parse_moves(segment["moves"]):
            move_id = MOVE_TO_ID.get(move)
            if move_id is None:
                raise ValueError("Unsupported demonstration action: {}".format(move))
            if target_phase:
                samples.append({
                    "cube_state": list(state), "move_history": history[-history_length:],
                    "phase": target_phase, "phase_annotation": phase,
                    "strategy_context": strategy, "next_expert_move": move, "move_id": move_id,
                    "last_layer_edges_oriented": last_layer_edges_oriented(state),
                    "cross_complete": cross_complete(state),
                    "f2l_slot_count": f2l_slot_count(state),
                    "move_count": move_count, "cross_color": record["cross_color"],
                    "solver_id": record["solver_id"], "solve_id": record["solve_id"],
                    "scramble_id": record["scramble_id"], "data_source": "reco.nz",
                    "sample_kind": "expert", "source_url": record["source_url"],
                })
            state = apply_move(state, move)
            history.append(move_id)
            move_count += 1
    if normalize_cube_orientation(state) != SOLVED_STATE:
        raise ValueError("Solve {} failed replay verification".format(record["solve_id"]))
    return samples


def split_by_solver(records, seed=17, fractions=(0.8, 0.1, 0.1)):
    if len(records) < 3:
        raise ValueError("At least three valid solves are required")
    by_solver = defaultdict(list)
    seen = set()
    for record in records:
        if record["solve_id"] in seen:
            raise ValueError("Duplicate solve_id: {}".format(record["solve_id"]))
        seen.add(record["solve_id"])
        by_solver[record["solver_id"]].append(record)
    solvers = sorted(by_solver)
    if len(solvers) < 3:
        raise ValueError("At least three distinct solvers are required for solver-held-out splits")
    random.Random(seed).shuffle(solvers)
    train_end = max(1, min(len(solvers) - 2, int(len(solvers) * fractions[0])))
    validation_end = max(train_end + 1, min(len(solvers) - 1, int(len(solvers) * (fractions[0] + fractions[1]))))
    groups = {"train": solvers[:train_end], "validation": solvers[train_end:validation_end],
              "test": solvers[validation_end:]}
    cross_solvers = {
        solver for solver, solver_records in by_solver.items()
        if any(_sample_phase(segment["phase"]) == "CROSS"
               for record in solver_records for segment in record["segments"])
    }
    reserved = set()
    for target in ("validation", "test"):
        if not cross_solvers.intersection(groups[target]):
            donor = next((solver for solver in groups["train"]
                          if solver in cross_solvers and solver not in reserved), None)
            if donor is not None:
                receiver = groups[target][0]
                groups["train"].remove(donor)
                groups[target][0] = donor
                groups["train"].append(receiver)
                reserved.add(donor)
    return {name: [record for solver in values for record in by_solver[solver]] for name, values in groups.items()}


def prepare_f2l(records, output=OUTPUT_PATH, seed=17):
    split_records = split_by_solver(records, seed=seed)
    manifest = {"schema_version": 1, "seed": seed, "data_source": "reco.nz",
                "sample_kind": "expert", "split_policy": "held out by solver_id; no solver crosses splits",
                "event_filter": "3x3 CFOP only; OH excluded", "splits": {}}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, group in split_records.items():
        rows = [sample for record in group for sample in examples_from_record(record)]
        path = output / (name + ".jsonl")
        path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
        manifest["splits"][name] = {
            "solves": len(group), "solvers": len({record["solver_id"] for record in group}),
            "samples": len(rows), "files": path.name,
            "phase_counts": {phase: sum(row["phase"] == phase for row in rows)
                             for phase in ("CROSS", "CROSS+1", "F2L", "ZBLS")},
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--pages", type=int, default=None, help="limit index pages; default follows all pages")
    parser.add_argument("--max-solves", type=int, default=None, help="limit indexed solves; default collects all filtered rows")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--reuse-raw", action="store_true", help="use the existing local reco.nz file instead of fetching")
    parser.add_argument("--all-3x3-methods", action="store_true",
                        help="collect every method in the 3x3 event; do not add non-CFOP records to model data")
    args = parser.parse_args(argv)
    if args.all_3x3_methods and args.raw == RAW_PATH:
        args.raw = Path("data/raw/reco_nz/all_3x3.jsonl")
    if args.reuse_raw and args.raw.exists():
        records = [json.loads(line) for line in args.raw.read_text(encoding="utf-8").splitlines() if line.strip()]
        rejected = []
    else:
        cached = []
        for path in (args.raw, args.raw.with_suffix(args.raw.suffix + ".partial")):
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    try:
                        cached.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if args.all_3x3_methods and RAW_PATH.exists():
            for line in RAW_PATH.read_text(encoding="utf-8").splitlines():
                try:
                    cached.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        checkpoint_path = args.raw.with_suffix(args.raw.suffix + ".partial")
        records, rejected = scrape_reconstructions(
            args.pages, args.max_solves, cached_records=cached, checkpoint_path=checkpoint_path,
            all_methods=args.all_3x3_methods)
        args.raw.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = args.raw.with_suffix(args.raw.suffix + ".tmp")
        temporary_path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in records), encoding="utf-8")
        temporary_path.replace(args.raw)
        checkpoint_path.unlink(missing_ok=True)
    if args.all_3x3_methods:
        method_counts = {}
        for record in records:
            method_counts[record["method"]] = method_counts.get(record["method"], 0) + 1
        manifest = {"data_source": "reco.nz", "sample_kind": "expert",
                    "event_filter": "3x3 all methods; OH and other events excluded",
                    "candidate_index_rows": len(records) + len(rejected),
                    "valid_reconstructions": len(records), "rejected": len(rejected),
                    "method_counts": method_counts, "raw_file": str(args.raw)}
    else:
        manifest = prepare_f2l(records, args.output, args.seed)
    rejection_path = args.raw.with_name(args.raw.stem + "_rejected.jsonl")
    rejection_path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rejected), encoding="utf-8")
    reason_counts = {}
    for item in rejected:
        reason_counts[item["reason"]] = reason_counts.get(item["reason"], 0) + 1
    manifest["collection"] = {"candidate_index_rows": len(records) + len(rejected),
                               "valid_reconstructions": len(records), "rejected": len(rejected),
                               "rejection_reason_counts": reason_counts,
                               "rejection_examples": rejected[:20], "rejection_file": str(rejection_path),
                               "raw_file": str(args.raw)}
    if args.all_3x3_methods:
        (args.raw.parent / "all_3x3_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    else:
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
