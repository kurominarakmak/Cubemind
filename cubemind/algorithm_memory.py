"""Exact, replay-verified retrieval from the checked-in CFOP/ZB algorithm tables."""

import json
from functools import lru_cache
from pathlib import Path

from cubemind.environment.cube import (
    COLORS, SOLVED_STATE, _GEOMETRY, apply_sequence, cross_complete,
    f2l_complete, f2l_slot_count, inverse_sequence, last_layer_edges_oriented,
    normalize_cube_orientation, parse_moves,
)
from cubemind.last_layer import (
    ZBLL_PATH, _rotation_paths, recognize_zbll_case, solution_for_zbll_state,
    zbll_case_state,
    verify_zbll_database,
)

ROOT = Path(__file__).resolve().parents[1]
OLL_PATH = ROOT / "data" / "algorithms" / "oll_cases.json"
PLL_PATH = ROOT / "data" / "algorithms" / "pll_cases.json"
ZBLS_PATH = ROOT / "data" / "algorithms" / "zbls_cases.json"
_UP = (0, 1, 0)
_CORNER_SLOTS = []
_EDGE_SLOTS = []
for _x in (-1, 1):
    for _z in (-1, 1):
        _pos = (_x, 1, _z)
        _indices = {normal: index for index, (pos, normal) in enumerate(_GEOMETRY) if pos == _pos}
        _CORNER_SLOTS.append((_pos, tuple(_indices[normal] for normal in (_UP, (_x, 0, 0), (0, 0, _z)))))
for _pos in ((-1, 1, 0), (1, 1, 0), (0, 1, -1), (0, 1, 1)):
    _indices = {normal: index for index, (pos, normal) in enumerate(_GEOMETRY) if pos == _pos}
    _side = next(normal for normal in _indices if normal != _UP)
    _EDGE_SLOTS.append((_pos, (_indices[_UP], _indices[_side])))
_CORNER_SLOTS.sort()
_EDGE_SLOTS.sort()


def _oll_complete(state):
    return f2l_complete(state) and all(value == state[4] for value in state[:9])


def _oll_pattern(state):
    up = state[COLORS["U"] * 9 + 4]
    corners = []
    for _pos, indices in _CORNER_SLOTS:
        orientations = [index for index, facelet in enumerate(indices) if state[facelet] == up]
        if len(orientations) != 1:
            return None
        corners.append(orientations[0])
    edges = []
    for _pos, (up_facelet, side_facelet) in _EDGE_SLOTS:
        if up not in (state[up_facelet], state[side_facelet]):
            return None
        edges.append(int(state[up_facelet] != up))
    return tuple(corners + edges)


def _canonical_oll_pattern(state):
    patterns = []
    for turns in range(4):
        turned = apply_sequence(state, ["U"] * turns)
        pattern = _oll_pattern(turned)
        if pattern is not None:
            patterns.append(pattern)
    return min(patterns) if patterns else None


@lru_cache(maxsize=1)
def _oll_index():
    data = json.loads(OLL_PATH.read_text(encoding="utf-8"))
    by_key, by_id = {}, {}
    for case in data["cases"]:
        state = normalize_cube_orientation(apply_sequence(
            SOLVED_STATE, inverse_sequence(case["algorithms"][0])))
        if not f2l_complete(state) or _oll_complete(state):
            raise ValueError("{} is not an OLL-only algorithm case".format(case["case_id"]))
        key = _canonical_oll_pattern(state)
        if key is None or key in by_key:
            raise ValueError("OLL case identity collision or invalid orientation: {}".format(case["case_id"]))
        by_key[key] = case
        by_id[case["case_id"]] = case
    if len(by_id) != 57:
        raise ValueError("Expected 57 OLL cases; found {}".format(len(by_id)))
    return by_key, by_id


def recognize_oll_case(state):
    """Return a classic OLL identity, independent of last-layer piece permutation."""
    normalized = normalize_cube_orientation(tuple(state))
    if not f2l_complete(normalized) or _oll_complete(normalized):
        return None
    case = _oll_index()[0].get(_canonical_oll_pattern(normalized))
    return case["case_id"] if case else None


def _home_moves(state):
    target = normalize_cube_orientation(tuple(state))
    for moves in _rotation_paths():
        if apply_sequence(state, moves) == target:
            return tuple(moves)
    raise ValueError("Could not normalize cube orientation")


def oll_candidates(state, case_id=None):
    """Return only OLL table variants that preserve F2L and orient the top face on replay."""
    original = tuple(state)
    case_id = case_id or recognize_oll_case(original)
    if case_id is None or recognize_oll_case(original) != case_id:
        return []
    case = _oll_index()[1][case_id]
    candidates, seen = [], set()
    for auf_count in range(4):
        auf = ("U",) * auf_count
        for algorithm_index, algorithm in enumerate(case["algorithms"]):
            moves = _home_moves(original) + auf + tuple(parse_moves(algorithm))
            if moves in seen:
                continue
            seen.add(moves)
            result = normalize_cube_orientation(apply_sequence(original, moves))
            if _oll_complete(result):
                candidates.append({
                    "case_id": case_id, "algorithm_id": "{}:{}".format(case_id, algorithm_index + 1),
                    "moves": list(moves), "source": "SpeedCubeDB via @moishy/algsets 0.3.3",
                    "license": "MIT", "move_metrics": _move_metrics(moves),
                })
    return sorted(candidates, key=lambda row: (_move_metrics(row["moves"])["stm"],
                                                _move_metrics(row["moves"])["rotations"],
                                                row["algorithm_id"]))


def _move_metrics(sequence):
    moves = parse_moves(sequence)
    return {
        "stm": sum(2 if move[0].islower() or move[1:2] == "w" else 1
                   for move in moves if move[0] not in "xyz"),
        "htm": sum(move[0] in "URFDLB" and move[1:2] != "w" for move in moves),
        "rotations": sum(move[0] in "xyz" for move in moves),
        "slice_moves": sum(move[0] in "MES" for move in moves),
        "wide_moves": sum(move[0].islower() or move[1:2] == "w" for move in moves),
    }


@lru_cache(maxsize=1)
def _verified_pll_variants():
    """Map source PLL variants to canonical case IDs only when simulator replay proves the match."""
    source = json.loads(PLL_PATH.read_text(encoding="utf-8"))
    cases = json.loads(ZBLL_PATH.read_text(encoding="utf-8"))["cases"]
    case_ids = [case["case_id"] for case in cases if case["case_id"].startswith("PLL-")]
    result = {case_id: [] for case_id in case_ids}
    for case_id in case_ids:
        state = zbll_case_state(case_id)
        home = _home_moves(state)
        for source_case in source["cases"]:
            for index, algorithm in enumerate(source_case["algorithms"], 1):
                body = tuple(parse_moves(algorithm))
                for pre_count in range(4):
                    pre = ("U",) * pre_count
                    matched = False
                    for post_count in range(4):
                        moves = home + pre + body + ("U",) * post_count
                        if normalize_cube_orientation(apply_sequence(state, moves)) == SOLVED_STATE:
                            result[case_id].append({
                                "algorithm_id": "{}:{}".format(source_case["case_id"], index),
                                "source_case_id": source_case["case_id"], "algorithm": algorithm,
                                "source": "SpeedCubeDB via @moishy/algsets 0.3.3", "license": "MIT",
                            })
                            matched = True
                            break
                    if matched:
                        break
    return {case_id: tuple(rows) for case_id, rows in result.items()}


@lru_cache(maxsize=1)
def _pll_algorithm_pool():
    rows = {case_id: list(variants) for case_id, variants in _verified_pll_variants().items()}
    for case in json.loads(ZBLL_PATH.read_text(encoding="utf-8"))["cases"]:
        if case["case_id"].startswith("PLL-"):
            rows[case["case_id"]].append({
                "algorithm_id": case["case_id"] + ":primary", "algorithm": case["algorithms"][0],
                "source": "Tao Yu Alg-Trainer", "license": "MIT",
            })
    return {case_id: tuple(algorithms) for case_id, algorithms in rows.items()}


def pll_candidates(state, case_id=None):
    """Match PLL algorithms to the current state and keep only replay-solving variants."""
    original = tuple(state)
    recognized = recognize_zbll_case(original)
    if case_id is not None and (not case_id.startswith("PLL-") or recognized != case_id):
        return []
    candidates, seen = [], set()
    home = _home_moves(original)
    variants = _pll_algorithm_pool()
    case_ids = ((case_id,) if case_id else
                ((recognized,) if recognized and recognized.startswith("PLL-") else tuple(variants)))
    for candidate_case in case_ids:
        for record in variants.get(candidate_case, ()):
            body = tuple(parse_moves(record["algorithm"]))
            for yaw_count in range(4):
                yaw = ("y",) * yaw_count
                undo_yaw = tuple(inverse_sequence(yaw))
                for pre_count in range(4):
                    pre = ("U",) * pre_count
                    for post_count in range(4):
                        moves = home + yaw + pre + body + ("U",) * post_count + undo_yaw
                        if moves in seen:
                            continue
                        seen.add(moves)
                        if normalize_cube_orientation(apply_sequence(original, moves)) == SOLVED_STATE:
                            candidates.append({**record, "case_id": candidate_case,
                                               "orientation_y": yaw_count, "moves": list(moves),
                                               "move_metrics": _move_metrics(moves)})
    if not candidates and case_id:
        moves = solution_for_zbll_state(original, case_id)
        if moves:
            candidates.append({"case_id": case_id, "algorithm_id": case_id + ":primary",
                               "moves": list(moves), "source": "Tao Yu Alg-Trainer", "license": "MIT",
                               "move_metrics": _move_metrics(moves)})
    return sorted(candidates, key=lambda row: (row["move_metrics"]["stm"],
                                                row["move_metrics"]["rotations"],
                                                row["algorithm_id"]))


def verify_pll_database():
    variants = _verified_pll_variants()
    return {"cases": len(variants), "covered_cases": sum(bool(rows) for rows in variants.values()),
            "source_variants": sum(len(case["algorithms"]) for case in
                                   json.loads(PLL_PATH.read_text(encoding="utf-8"))["cases"]),
            "verified_variants": sum(len(rows) for rows in variants.values()),
            "verification": "each variant replayed against its recognized PLL case"}


@lru_cache(maxsize=1)
def algorithm_records():
    """Flatten source entries into stable records; human frequency is deliberately unknown."""
    oll = json.loads(OLL_PATH.read_text(encoding="utf-8"))
    zbll = json.loads(ZBLL_PATH.read_text(encoding="utf-8"))
    zbls = json.loads(ZBLS_PATH.read_text(encoding="utf-8"))
    groups = []
    for case in oll["cases"]:
        for index, moves in enumerate(case["algorithms"], 1):
            groups.append(("OLL", case["case_id"], index, moves,
                           "SpeedCubeDB via @moishy/algsets 0.3.3", "MIT",
                           "F2L complete", "F2L preserved; U face oriented"))
    for case_id, variants in _verified_pll_variants().items():
        for variant in variants:
            groups.append(("PLL", case_id, variant["algorithm_id"], variant["algorithm"],
                           variant["source"], variant["license"],
                           "F2L complete; last-layer edges oriented", "Cube solved up to whole-cube orientation"))
    for case in zbll["cases"]:
        family = "PLL" if case["case_id"].startswith("PLL-") else "ZBLL"
        for index, moves in enumerate(case["algorithms"], 1):
            groups.append((family, case["case_id"], index, moves,
                           "Tao Yu Alg-Trainer", "MIT",
                           "F2L complete; last-layer edges oriented", "Cube solved up to whole-cube orientation"))
    for case in zbls["cases"]:
        for index, moves in enumerate(case["algorithms"], 1):
            groups.append(("ZBLS", case["case_id"], index, moves,
                           "CubeRoot public algorithm API", "not recorded in imported snapshot",
                           "Cross complete; exactly three F2L pairs solved", "F2L complete; last-layer edges oriented"))
    return tuple({
        "family": family, "case_id": case_id,
        "algorithm_id": index if isinstance(index, str) else "{}:{}".format(case_id, index),
        "moves": tuple(parse_moves(moves)), "source": source, "license": license_name,
        "preconditions": preconditions, "postconditions": postconditions,
        "move_metrics": _move_metrics(moves), "observed_human_usage": None,
    } for family, case_id, index, moves, source, license_name, preconditions, postconditions in groups)


def rank_algorithm_records(records, mode="shortest"):
    if mode != "shortest":
        raise ValueError("rank_algorithm_records supports only mode='shortest'; pass a trained preference model to solve_symbolic_last_layer")
    return sorted(records, key=lambda row: (row["move_metrics"]["stm"],
                                            row["move_metrics"]["rotations"],
                                            row["algorithm_id"]))


def solve_symbolic_last_layer(state, strategy="eo-aware", preference_model=None):
    """Resolve an F2L-complete state with deterministic OLL/PLL or ZBLL retrieval."""
    if strategy not in ("oll-pll", "eo-aware"):
        raise ValueError("strategy must be oll-pll or eo-aware")
    original = tuple(state)
    if not f2l_complete(normalize_cube_orientation(original)):
        return {"verified": False, "failure_reason": "requires_f2l_complete",
                "moves": [], "moves_by_phase": {"OLL": [], "PLL": [], "ZBLL": []}}
    for auf in ((), ("U",), ("U2",), ("U'",)):
        if normalize_cube_orientation(apply_sequence(original, auf)) == SOLVED_STATE:
            moves = list(auf)
            return {"verified": True, "failure_reason": None, "moves": moves,
                    "moves_by_phase": {"OLL": [], "PLL": moves, "ZBLL": []},
                    "route": "PLL AUF", "pll_case_id": "solved",
                    "algorithm_source": "deterministic AUF normalization"}
    if strategy == "eo-aware" and last_layer_edges_oriented(normalize_cube_orientation(original)):
        case_id = recognize_zbll_case(original)
        if case_id is not None:
            is_pll = case_id.startswith("PLL-")
            candidates = pll_candidates(original, case_id) if is_pll else []
            if preference_model is not None and candidates:
                from cubemind.models.algorithm_preference import rank_algorithm_candidates
                candidates = rank_algorithm_candidates(candidates, preference_model)
            moves = candidates[0]["moves"] if candidates else solution_for_zbll_state(original, case_id)
            if moves is not None and normalize_cube_orientation(apply_sequence(original, moves)) == SOLVED_STATE:
                phase = "PLL" if is_pll else "ZBLL"
                return {"verified": True, "failure_reason": None, "moves": list(moves),
                        "moves_by_phase": {"OLL": [], "PLL": list(moves) if is_pll else [],
                                           "ZBLL": list(moves) if not is_pll else []},
                        "route": phase, "case_id": case_id,
                        "algorithm_source": (candidates[0]["source"] if candidates else "Tao Yu Alg-Trainer"),
                        "algorithm_candidates": ([dict(row, phase=phase) for row in candidates]
                                                  or [{"case_id": case_id, "algorithm_id": case_id + ":primary",
                                                       "moves": list(moves), "source": "Tao Yu Alg-Trainer",
                                                       "license": "MIT", "phase": phase,
                                                       "move_metrics": _move_metrics(moves)}])}

    moves_by_phase = {"OLL": [], "PLL": [], "ZBLL": []}
    current = original
    oll_case_id = None
    oll_variants = []
    if not _oll_complete(normalize_cube_orientation(current)):
        oll_case_id = recognize_oll_case(current)
        oll_variants = oll_candidates(current, oll_case_id)
        if not oll_variants:
            return {"verified": False, "failure_reason": "unrecognized_or_unverified_oll_case",
                    "moves": [], "moves_by_phase": moves_by_phase, "oll_case_id": oll_case_id}
        if preference_model is not None:
            from cubemind.models.algorithm_preference import rank_algorithm_candidates
            oll_variants = rank_algorithm_candidates(oll_variants, preference_model)
        moves_by_phase["OLL"] = oll_variants[0]["moves"]
        current = apply_sequence(current, moves_by_phase["OLL"])

    for auf in ((), ("U",), ("U2",), ("U'",)):
        if normalize_cube_orientation(apply_sequence(current, auf)) == SOLVED_STATE:
            moves_by_phase["PLL"] = list(auf)
            moves = moves_by_phase["OLL"] + moves_by_phase["PLL"]
            return {"verified": True, "failure_reason": None, "moves": moves,
                    "moves_by_phase": moves_by_phase,
                    "route": "OLL → PLL AUF" if moves_by_phase["OLL"] else "PLL AUF",
                    "oll_case_id": oll_case_id, "pll_case_id": "solved",
                    "algorithm_source": "SpeedCubeDB via @moishy/algsets 0.3.3 + deterministic AUF",
                    "algorithm_candidates": [dict(row, phase="OLL") for row in oll_variants]}

    if normalize_cube_orientation(current) == SOLVED_STATE:
        moves = list(moves_by_phase["OLL"])
        return {"verified": True, "failure_reason": None, "moves": moves,
                "moves_by_phase": moves_by_phase, "route": "OLL skip" if not moves else "OLL",
                "oll_case_id": oll_case_id,
                "algorithm_source": "SpeedCubeDB via @moishy/algsets 0.3.3"}

    pll_case_id = recognize_zbll_case(current)
    pll_variants = pll_candidates(current, pll_case_id if pll_case_id and pll_case_id.startswith("PLL-") else None)
    if pll_case_id is None and pll_variants:
        pll_case_id = pll_variants[0]["case_id"]
    if pll_case_id is None or not pll_case_id.startswith("PLL-"):
        return {"verified": False, "failure_reason": "unrecognized_pll_case",
                "moves": [*moves_by_phase["OLL"]], "moves_by_phase": moves_by_phase,
                "oll_case_id": oll_case_id}
    if not pll_variants:
        pll_variants = pll_candidates(current, pll_case_id)
    if preference_model is not None and pll_variants:
        from cubemind.models.algorithm_preference import rank_algorithm_candidates
        pll_variants = rank_algorithm_candidates(pll_variants, preference_model)
    pll_moves = pll_variants[0]["moves"] if pll_variants else solution_for_zbll_state(current, pll_case_id)
    if pll_moves is None:
        return {"verified": False, "failure_reason": "no_verified_pll_algorithm",
                "moves": list(moves_by_phase["OLL"]), "moves_by_phase": moves_by_phase,
                "oll_case_id": oll_case_id, "pll_case_id": pll_case_id}
    moves_by_phase["PLL"] = list(pll_moves)
    moves = moves_by_phase["OLL"] + moves_by_phase["PLL"]
    verified = normalize_cube_orientation(apply_sequence(original, moves)) == SOLVED_STATE
    return {"verified": verified,
            "failure_reason": None if verified else "symbolic_cfop_replay_failed",
            "moves": moves, "moves_by_phase": moves_by_phase, "route": "OLL → PLL",
            "oll_case_id": oll_case_id, "pll_case_id": pll_case_id,
            "algorithm_source": (pll_variants[0]["source"] if pll_variants else "Tao Yu Alg-Trainer"),
            "algorithm_candidates": ([dict(row, phase="OLL") for row in oll_variants]
                                     + [dict(row, phase="PLL") for row in pll_variants])}


def solve_symbolic_zbls(state, max_candidates=None):
    """Try only source ZBLS algorithms and return candidates that satisfy the exact goal."""
    original = tuple(state)
    if not cross_complete(original) or f2l_slot_count(original) != 3:
        return {"verified": False, "failure_reason": "requires_cross_and_three_f2l_pairs",
                "moves": [], "candidates": [], "states_checked": 0}
    records = rank_algorithm_records([row for row in algorithm_records() if row["family"] == "ZBLS"])
    if max_candidates is not None:
        if type(max_candidates) is not int or max_candidates < 1:
            raise ValueError("max_candidates must be a positive integer")
        records = records[:max_candidates]
    found, checked, seen = [], 0, set()
    y_rotations = ((), ("y",), ("y2",), ("y'",))
    u_aufs = ((), ("U",), ("U2",), ("U'",))
    for record in records:
        for y in y_rotations:
            for pre_auf in u_aufs:
                for post_auf in u_aufs:
                    inverse_y = tuple("y'" if move == "y" else "y" if move == "y'" else move for move in reversed(y))
                    moves = tuple(y) + tuple(pre_auf) + record["moves"] + tuple(post_auf) + inverse_y
                    if moves in seen:
                        continue
                    seen.add(moves)
                    checked += 1
                    final = apply_sequence(original, moves)
                    if f2l_complete(final) and last_layer_edges_oriented(final):
                        found.append({"case_id": record["case_id"], "algorithm_id": record["algorithm_id"],
                                      "moves": list(moves), "source": record["source"],
                                      "license": record["license"],
                                      "move_metrics": _move_metrics(moves)})
    found.sort(key=lambda row: (row["move_metrics"]["stm"], row["move_metrics"]["rotations"],
                                row["algorithm_id"]))
    chosen = found[0] if found else None
    return {"verified": bool(chosen), "failure_reason": None if chosen else "no_verified_zbls_algorithm",
            "moves": chosen["moves"] if chosen else [], "candidates": found[:8],
            "case_id": chosen["case_id"] if chosen else None,
            "algorithm_id": chosen["algorithm_id"] if chosen else None,
            "states_checked": checked,
            "edge_oriented": bool(chosen), "recognition": "per-candidate simulator postcondition verification"}


def verify_oll_database():
    by_key, by_id = _oll_index()
    verified = 0
    for case_id, case in by_id.items():
        state = normalize_cube_orientation(apply_sequence(
            SOLVED_STATE, inverse_sequence(case["algorithms"][0])))
        if recognize_oll_case(state) != case_id:
            raise ValueError("OLL recognition failed for {}".format(case_id))
        for algorithm in case["algorithms"]:
            for auf in ((), ("U",), ("U2",), ("U'",)):
                final = normalize_cube_orientation(apply_sequence(state, tuple(auf) + tuple(parse_moves(algorithm))))
                if _oll_complete(final):
                    verified += 1
                    break
            else:
                raise ValueError("Unverified OLL algorithm for {}: {}".format(case_id, algorithm))
    return {"cases": len(by_id), "algorithms": sum(len(case["algorithms"]) for case in by_id.values()),
            "verified_algorithms": verified, "recognition": "corner/edge orientation pattern normalized over AUF"}


def verify_algorithm_memory():
    oll = verify_oll_database()
    pll = verify_pll_database()
    zbls = verify_zbls_database()
    records = algorithm_records()
    families = {family: sum(row["family"] == family for row in records)
                for family in ("OLL", "PLL", "ZBLL", "ZBLS")}
    return {"OLL": oll, "PLL": pll, "ZBLS": zbls, "other_families": families,
            "ZBLL_PLL": verify_zbll_database(),
            "ZBLS_case_recognition": "not exact; compatible candidates are identified by simulator postcondition replay"}


def verify_zbls_database():
    data = json.loads(ZBLS_PATH.read_text(encoding="utf-8"))
    verified = 0
    for case in data["cases"]:
        state = apply_sequence(SOLVED_STATE, case["setup"])
        if not cross_complete(state) or f2l_slot_count(state) != 3:
            raise ValueError("Invalid ZBLS setup precondition: {}".format(case["case_id"]))
        for algorithm in case["algorithms"]:
            final = apply_sequence(state, algorithm)
            if not f2l_complete(final) or not last_layer_edges_oriented(final):
                raise ValueError("Unverified ZBLS algorithm: {}".format(case["case_id"]))
            verified += 1
    return {"cases": len(data["cases"]), "verified_algorithms": verified,
            "verification": "Cross + exactly three F2L pairs to solved F2L + oriented last-layer edges"}
