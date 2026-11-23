"""Import and simulator-verify CubeRoot's public 3x3 ZBLS case list."""

import hashlib
import json
import re
import urllib.request
from pathlib import Path

from cubemind.environment.cube import (
    SOLVED_STATE,
    apply_sequence,
    cross_complete,
    f2l_complete,
    f2l_slot_count,
    last_layer_edges_oriented,
    normalize_cube_orientation,
)

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://api.cuberoot.me/v1/alg/sets/3x3/zbls"
OUTPUT = ROOT / "data" / "algorithms" / "zbls_cases.json"
_MOVE = re.compile(r"(?:[URFDLB](?:w)?|[urfdlb]|[MES]|[xyz])(?:2'?|3|')?")


def normalize_source_algorithm(source):
    """Convert CubeRoot annotations to turns, expanding grouped repetitions."""
    algorithm = source
    for _ in range(8):
        expanded = re.sub(
            r"\(([^()]*)\)(\d+)",
            lambda match: " ".join([match.group(1)] * int(match.group(2))),
            algorithm,
        )
        if expanded == algorithm:
            break
        algorithm = expanded
    if re.search(r"\([^()]*\)\d+", algorithm):
        raise ValueError("Unsupported nested source repetition")
    algorithm = re.sub(r"\[[^\]]*\]", " ", algorithm)
    algorithm = re.sub(r"[↓↑]", " ", algorithm)
    algorithm = algorithm.replace("(", " ").replace(")", " ")
    algorithm = re.sub(r"=(?=[xyzURFDLB])", "", algorithm)

    moves = []
    for chunk in algorithm.split():
        tokens = list(_MOVE.finditer(chunk))
        if "".join(token.group() for token in tokens) != chunk:
            raise ValueError(f"Unrecognized CubeRoot algorithm token: {chunk}")
        moves.extend(token.group() for token in tokens)
    return " ".join(moves)


def import_cases():
    request = urllib.request.Request(ENDPOINT, headers={"User-Agent": "CubeMind research"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read()
    source = json.loads(raw)
    if source.get("puzzle") != "3x3" or source.get("set") != "zbls":
        raise ValueError("CubeRoot endpoint returned a different algorithm set")

    cases = []
    seen_ids = set()
    for item in source["cases"]:
        case_id = item["id"]
        if case_id in seen_ids:
            raise ValueError(f"Duplicate CubeRoot case ID: {case_id}")
        seen_ids.add(case_id)

        setup = item["setup"]
        state = apply_sequence(SOLVED_STATE, setup)
        if not cross_complete(state):
            raise ValueError(f"CubeRoot setup does not preserve Cross: {case_id}")
        slots = f2l_slot_count(state)
        if slots == 4:
            continue
        if slots != 3:
            raise ValueError(f"CubeRoot setup is not a three-slot ZBLS case: {case_id}")

        alternatives = item["algs"]
        if item.get("oriNames", [None])[0] != "Front Right" or len(alternatives) != 4:
            raise ValueError(f"Unexpected CubeRoot slot bucket order: {case_id}")
        source_algorithm = alternatives[0][0]["alg"]
        algorithm = normalize_source_algorithm(source_algorithm)
        result = normalize_cube_orientation(apply_sequence(state, algorithm))
        if not f2l_complete(result) or not last_layer_edges_oriented(result):
            raise ValueError(f"CubeRoot algorithm failed simulator replay: {case_id}")
        cases.append({
            "case_id": f"CubeRoot-{case_id}",
            "family": item["name"],
            "setup": setup,
            "algorithms": [algorithm],
            "source_algorithm": source_algorithm,
            "source_variant_count": sum(len(bucket) for bucket in alternatives),
            "source_case_id": case_id,
            "source_subgroup": item.get("subgroup"),
        })

    if len(cases) != 302 or len({case["case_id"] for case in cases}) != 302:
        raise ValueError(f"Expected 302 unique three-slot cases; verified {len(cases)}")
    if len({case["algorithms"][0] for case in cases}) != 302:
        raise ValueError("CubeRoot import contains duplicate normalized primary algorithms")
    return {
        "schema_version": 2,
        "dataset": "ZBLS",
        "source": {
            "name": "CubeRoot public algorithm API",
            "page": "https://cuberoot.me/alg/3x3/zbls",
            "endpoint": ENDPOINT,
            "snapshot_scraped_at": source.get("scrapedAt"),
            "snapshot_source_metadata": source.get("source"),
            "retrieved_payload_sha256": hashlib.sha256(raw).hexdigest(),
            "selection": "First algorithm in the Front Right slot bucket per source case",
            "normalization": (
                "Expand parenthesized repetitions; omit bracketed optional tokens and arrow markers; "
                "remove CubeRoot '=' turn decoration; split adjacent Singmaster turns. Original text retained."
            ),
            "verification": (
                "Apply setup to solved 3x3; require Cross plus exactly three F2L slots; "
                "apply normalized algorithm and require solved F2L plus oriented last-layer edges."
            ),
        },
        "cases": cases,
    }


if __name__ == "__main__":
    data = import_cases()
    OUTPUT.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote and verified {len(data['cases'])} cases to {OUTPUT}")
