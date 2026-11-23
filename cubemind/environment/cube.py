"""A deterministic 54-facelet simulator for standard 3×3 Singmaster moves."""

from collections import deque
from functools import lru_cache
import re
from typing import Dict, Iterable, List, Sequence, Tuple, Union

FACE_ORDER = "URFDLB"
COLORS = {face: index for index, face in enumerate(FACE_ORDER)}
# Fixed frame: U yellow, R red, F green, D white, L orange, B blue.
COLOR_NAMES = ("yellow", "red", "green", "white", "orange", "blue")
MOVE_TOKENS = tuple(face + suffix for face in "URFDLB" for suffix in ("", "'", "2"))
_SIMULATOR_BASE_MOVES = tuple("URFDLB") + tuple(face + "w" for face in "URFDLB") + tuple("urfdlb") + tuple("MESxyz")
SIMULATOR_MOVE_TOKENS = tuple(move + suffix for move in _SIMULATOR_BASE_MOVES for suffix in ("", "'", "2"))
_MOVE_RE = re.compile(r"^(?:[URFDLB](?:w)?|[urfdlb]|[MES]|[xyz])(?:2'?|3|')?$")
_VEC = Tuple[int, int, int]
_FACE_NORMALS: Dict[str, _VEC] = {
    "U": (0, 1, 0), "R": (1, 0, 0), "F": (0, 0, 1),
    "D": (0, -1, 0), "L": (-1, 0, 0), "B": (0, 0, -1),
}


def _facelet_geometry() -> Tuple[List[Tuple[_VEC, _VEC]], Dict[Tuple[_VEC, _VEC], int]]:
    geometry: List[Tuple[_VEC, _VEC]] = []
    for face in FACE_ORDER:
        for row in range(3):
            for col in range(3):
                if face == "U":
                    pos = (col - 1, 1, row - 1)
                elif face == "R":
                    pos = (1, 1 - row, 1 - col)
                elif face == "F":
                    pos = (col - 1, 1 - row, 1)
                elif face == "D":
                    pos = (col - 1, -1, 1 - row)
                elif face == "L":
                    pos = (-1, 1 - row, col - 1)
                else:  # B
                    pos = (1 - col, 1 - row, -1)
                geometry.append((pos, _FACE_NORMALS[face]))
    return geometry, {item: index for index, item in enumerate(geometry)}


_GEOMETRY, _GEOMETRY_INDEX = _facelet_geometry()
SOLVED_STATE = tuple(index for index in range(6) for _ in range(9))
_FACELET_CENTER = tuple(
    COLORS[next(face for face, face_normal in _FACE_NORMALS.items() if face_normal == normal)] * 9 + 4
    for _pos, normal in _GEOMETRY
)


def _cross(a: _VEC, b: _VEC) -> _VEC:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a: _VEC, b: _VEC) -> int:
    return sum(x * y for x, y in zip(a, b))


def _rotate(v: _VEC, axis: _VEC, clockwise_quarters: int) -> _VEC:
    q = clockwise_quarters % 4
    if q == 0:
        return v
    if q == 2:
        return tuple(2 * _dot(axis, v) * axis[i] - v[i] for i in range(3))  # type: ignore[return-value]
    perpendicular = _cross(v, axis) if q == 1 else _cross(axis, v)
    parallel = tuple(_dot(axis, v) * axis[i] for i in range(3))
    return tuple(perpendicular[i] + parallel[i] for i in range(3))  # type: ignore[return-value]


def _build_move_permutation(token: str) -> Tuple[int, ...]:
    base = token[:-2] if token.endswith("2'") else token[:-1] if token.endswith(("'", "2")) else token
    turns = 2 if token.endswith(("2", "2'")) else (3 if token.endswith("'") else 1)
    face_axes = {"U": (0, 1, 0), "R": (1, 0, 0), "F": (0, 0, 1),
                 "D": (0, -1, 0), "L": (-1, 0, 0), "B": (0, 0, -1)}
    if base in "xyz":
        axis = {"x": face_axes["R"], "y": face_axes["U"], "z": face_axes["F"]}[base]
        layer = "all"
    elif base in "MES":
        axis = {"M": face_axes["L"], "E": face_axes["D"], "S": face_axes["F"]}[base]
        layer = "middle"
    else:
        face = base[0].upper()
        axis = face_axes[face]
        layer = "wide" if base.islower() or base.endswith("w") else "outer"
    permutation = list(range(54))
    for source, (pos, normal) in enumerate(_GEOMETRY):
        layer_position = _dot(pos, axis)
        selected = (layer == "all" or
                    (layer == "middle" and layer_position == 0) or
                    (layer == "wide" and layer_position >= 0) or
                    (layer == "outer" and layer_position == 1))
        if not selected:
            continue
        new_pos, new_normal = pos, normal
        for _ in range(turns):
            new_pos = _rotate(new_pos, axis, 1)
            new_normal = _rotate(new_normal, axis, 1)
        permutation[source] = _GEOMETRY_INDEX[(new_pos, new_normal)]
    return tuple(permutation)


_MOVE_PERMUTATIONS = {token: _build_move_permutation(token) for token in SIMULATOR_MOVE_TOKENS}


def parse_moves(sequence: Union[str, Iterable[str]]) -> List[str]:
    """Parse standard outer, wide, slice, and cube-rotation Singmaster tokens."""
    tokens = sequence.split() if isinstance(sequence, str) else list(sequence)
    normalized = []
    for token in tokens:
        if not isinstance(token, str) or not _MOVE_RE.fullmatch(token):
            raise ValueError("Moves must use standard face, wide, slice, or rotation notation")
        if token.endswith("2'"):
            normalized.append(token[:-1])
        elif token.endswith("3"):
            normalized.append(token[:-1] + "'")
        else:
            normalized.append(token)
    return normalized


def validate_state(state: Sequence[int]) -> Tuple[int, ...]:
    if len(state) != 54 or any(type(value) is not int or value not in range(6) for value in state):
        raise ValueError("Cube state must contain exactly 54 facelet color IDs in 0..5")
    if any(state.count(color) != 9 for color in range(6)):
        raise ValueError("Cube state must contain nine facelets of each color")
    return tuple(state)


def apply_move(state: Sequence[int], move: str) -> Tuple[int, ...]:
    state = validate_state(state)
    token = parse_moves([move])[0]
    permutation = _MOVE_PERMUTATIONS[token]
    result = [0] * 54
    for source, destination in enumerate(permutation):
        result[destination] = state[source]
    return tuple(result)


def apply_sequence(state: Sequence[int], sequence: Union[str, Iterable[str]]) -> Tuple[int, ...]:
    result = validate_state(state)
    for move in parse_moves(sequence):
        permutation = _MOVE_PERMUTATIONS[move]
        next_state = [0] * 54
        for source, destination in enumerate(permutation):
            next_state[destination] = result[source]
        result = tuple(next_state)
    return result


def inverse_move(move: str) -> str:
    token = parse_moves([move])[0]
    if token.endswith("2"):
        return token
    return token[:-1] if token.endswith("'") else token + "'"


def inverse_sequence(sequence: Union[str, Iterable[str]]) -> List[str]:
    return [inverse_move(move) for move in reversed(parse_moves(sequence))]


@lru_cache(maxsize=1)
def _orientation_homing_sequences() -> Dict[Tuple[int, ...], Tuple[str, ...]]:
    """Map each center orientation to a shortest whole-cube homing sequence."""
    center_indices = tuple(face * 9 + 4 for face in range(6))
    moves = tuple(move for move in SIMULATOR_MOVE_TOKENS if move[0] in "xyz")
    start = SOLVED_STATE
    queue = deque([start])
    paths = {tuple(start[index] for index in center_indices): ()}
    while queue:
        state = queue.popleft()
        path = paths[tuple(state[index] for index in center_indices)]
        for move in moves:
            next_state = apply_move(state, move)
            centers = tuple(next_state[index] for index in center_indices)
            if centers not in paths:
                paths[centers] = path + (move,)
                queue.append(next_state)
    if len(paths) != 24:
        raise AssertionError("Whole-cube rotations must produce 24 center orientations")
    return {centers: tuple(inverse_sequence(path)) for centers, path in paths.items()}


def normalize_cube_orientation(state: Sequence[int]) -> Tuple[int, ...]:
    """Rotate a state back to the fixed U/R/F/D/L/B center frame."""
    state = validate_state(state)
    center_indices = tuple(face * 9 + 4 for face in range(6))
    centers = tuple(state[index] for index in center_indices)
    return apply_sequence(state, _orientation_homing_sequences()[centers])


@lru_cache(maxsize=1)
def _cross_frame_rotations() -> Dict[int, Tuple[str, ...]]:
    center_index = COLORS["D"]
    orientations = _orientation_homing_sequences()
    result = {}
    for color in range(6):
        candidates = sorted(centers for centers in orientations if centers[center_index] == color)
        if not candidates:
            raise AssertionError("Every color must be placeable on Down by cube rotation")
        result[color] = tuple(inverse_sequence(orientations[candidates[0]]))
    return result


def cross_frame_rotation(state: Sequence[int], cross_color: Union[int, str]) -> Tuple[str, ...]:
    """Return cube rotations that place the requested Cross color on Down."""
    state = validate_state(state)
    if isinstance(cross_color, str):
        if cross_color not in COLOR_NAMES:
            raise ValueError("Unknown Cross color: {}".format(cross_color))
        cross_color = COLOR_NAMES.index(cross_color)
    if type(cross_color) is not int or cross_color not in range(6):
        raise ValueError("Cross color must be an ID in 0..5 or a known color name")
    centers = tuple(state[face * 9 + 4] for face in range(6))
    home = _orientation_homing_sequences()[centers]
    return tuple(home) + _cross_frame_rotations()[cross_color]


def normalize_cross_frame(state: Sequence[int], cross_color: Union[int, str]) -> Tuple[int, ...]:
    """Rotate a cube so its selected Cross color is on Down in a fixed frame."""
    state = validate_state(state)
    return apply_sequence(state, cross_frame_rotation(state, cross_color))


@lru_cache(maxsize=144)
def _cross_frame_move_map(centers: Tuple[int, ...], cross_color: int) -> Dict[str, str]:
    rotation = tuple(_orientation_homing_sequences()[centers]) + _cross_frame_rotations()[cross_color]
    canonical_start = apply_sequence(SOLVED_STATE, rotation)
    canonical_moves = {}
    for move in SIMULATOR_MOVE_TOKENS:
        canonical_moves.setdefault(apply_move(canonical_start, move), move)
    result = {}
    for move in SIMULATOR_MOVE_TOKENS:
        rotated_result = apply_sequence(apply_move(SOLVED_STATE, move), rotation)
        try:
            result[move] = canonical_moves[rotated_result]
        except KeyError as error:
            raise AssertionError("Cube rotation must preserve the simulator move vocabulary") from error
    return result


def cross_frame_move(move: str, centers: Sequence[int], cross_color: Union[int, str]) -> str:
    """Express one move in the normalized frame returned by normalize_cross_frame."""
    move = parse_moves([move])[0]
    centers = tuple(centers)
    if len(centers) != 6 or any(value not in range(6) for value in centers):
        raise ValueError("Center orientation must contain six color IDs")
    if isinstance(cross_color, str):
        if cross_color not in COLOR_NAMES:
            raise ValueError("Unknown Cross color: {}".format(cross_color))
        cross_color = COLOR_NAMES.index(cross_color)
    if type(cross_color) is not int or cross_color not in range(6):
        raise ValueError("Cross color must be an ID in 0..5 or a known color name")
    return _cross_frame_move_map(centers, cross_color)[move]


_CROSS_STICKERS = (
    (((0, -1, 1), (0, -1, 0)), ((0, -1, 1), (0, 0, 1))),
    (((1, -1, 0), (0, -1, 0)), ((1, -1, 0), (1, 0, 0))),
    (((0, -1, -1), (0, -1, 0)), ((0, -1, -1), (0, 0, -1))),
    (((-1, -1, 0), (0, -1, 0)), ((-1, -1, 0), (-1, 0, 0))),
)
_CROSS_FACELETS = tuple(tuple(_GEOMETRY_INDEX[item] for item in pair) for pair in _CROSS_STICKERS)
_F2L_SLOT_COORDS = ((1, 1), (-1, 1), (-1, -1), (1, -1))
_POSITION_FACELETS: Dict[_VEC, Tuple[int, ...]] = {}
for _index, (_pos, _normal) in enumerate(_GEOMETRY):
    _POSITION_FACELETS.setdefault(_pos, tuple())
    _POSITION_FACELETS[_pos] += (_index,)
_CORNER_POSITIONS = tuple((pos, indices) for pos, indices in _POSITION_FACELETS.items()
                          if sum(abs(value) for value in pos) == 3)
_EDGE_POSITIONS = tuple((pos, indices) for pos, indices in _POSITION_FACELETS.items()
                        if sum(abs(value) for value in pos) == 2)


def cross_complete(state: Sequence[int]) -> bool:
    state = validate_state(state)
    return all(state[index] == state[_FACELET_CENTER[index]]
               for pair in _CROSS_FACELETS for index in pair)


cross_solved = cross_complete


F2L_PAIR_NAMES = ("FR", "FL", "BL", "BR")
_F2L_SLOT_FACELETS = tuple(
    tuple(index for index, (pos, _normal) in enumerate(_GEOMETRY)
          if pos[0] == x and pos[2] == z and pos[1] in (-1, 0))
    for x, z in _F2L_SLOT_COORDS
)
_TOP_EDGE_UP_FACELETS = tuple(
    index for index, (pos, normal) in enumerate(_GEOMETRY)
    if pos[1] == 1 and sum(abs(value) for value in pos) == 2 and normal == (0, 1, 0)
)


def pair_solved(state: Sequence[int], pair_id: Union[int, str]) -> bool:
    """Return whether one corner and its matching edge occupy the named slot."""
    state = validate_state(state)
    if isinstance(pair_id, str):
        if pair_id not in F2L_PAIR_NAMES:
            raise ValueError("Unknown F2L pair: {}".format(pair_id))
        pair_id = F2L_PAIR_NAMES.index(pair_id)
    if type(pair_id) is not int or pair_id not in range(len(_F2L_SLOT_FACELETS)):
        raise ValueError("F2L pair ID must be 0..3 or one of {}".format(F2L_PAIR_NAMES))
    return all(state[index] == state[_FACELET_CENTER[index]] for index in _F2L_SLOT_FACELETS[pair_id])


def solved_pair_mask(state: Sequence[int]) -> Tuple[bool, bool, bool, bool]:
    """Return solved F2L slots in FR, FL, BL, BR order."""
    state = validate_state(state)
    return tuple(all(state[index] == state[_FACELET_CENTER[index]] for index in slot)
                 for slot in _F2L_SLOT_FACELETS)


def f2l_complete(state: Sequence[int]) -> bool:
    """Return whether Cross and all four corner-edge slots are solved."""
    state = validate_state(state)
    return cross_complete(state) and all(solved_pair_mask(state))


def f2l_slot_count(state: Sequence[int]) -> int:
    """Count solved corner-plus-middle-edge slots after the cross."""
    return sum(solved_pair_mask(state))


def _pair_pattern(state, corner_colors, edge_colors):
    def locate(positions, colors):
        return next(((index, tuple(state[facelet] for facelet in facelets))
                     for index, (_position, facelets) in enumerate(positions)
                     if frozenset(state[facelet] for facelet in facelets) == colors), None)

    corner = locate(_CORNER_POSITIONS, corner_colors)
    edge = locate(_EDGE_POSITIONS, edge_colors)
    if corner is None or edge is None:
        raise ValueError("Cube state does not contain the target F2L pieces")
    return corner + edge


@lru_cache(maxsize=96)
def _f2l_pair_pattern_distances(centers, pair_id):
    x, z = _F2L_SLOT_COORDS[pair_id]
    corner_indices = _POSITION_FACELETS[(x, -1, z)]
    edge_indices = _POSITION_FACELETS[(x, 0, z)]
    corner_colors = frozenset(centers[_FACELET_CENTER[index] // 9] for index in corner_indices)
    edge_colors = frozenset(centers[_FACELET_CENTER[index] // 9] for index in edge_indices)
    solved = tuple(color for color in centers for _ in range(9))
    start = _pair_pattern(solved, corner_colors, edge_colors)
    distances, queue = {start: 0}, deque([(start, solved)])
    while queue:
        pattern, representative = queue.popleft()
        distance = distances[pattern] + 1
        for move in MOVE_TOKENS:
            child = apply_move(representative, move)
            child_pattern = _pair_pattern(child, corner_colors, edge_colors)
            if child_pattern not in distances:
                distances[child_pattern] = distance
                queue.append((child_pattern, child))
    return distances


def f2l_pair_distance(state: Sequence[int], pair_id: Union[int, str]) -> int:
    """Return the exact outer-turn distance for one corner-edge pair, ignoring other pieces."""
    state = validate_state(state)
    if isinstance(pair_id, str):
        if pair_id not in F2L_PAIR_NAMES:
            raise ValueError("Unknown F2L pair: {}".format(pair_id))
        pair_id = F2L_PAIR_NAMES.index(pair_id)
    if type(pair_id) is not int or pair_id not in range(4):
        raise ValueError("F2L pair ID must be 0..3 or one of {}".format(F2L_PAIR_NAMES))
    centers = tuple(state[face * 9 + 4] for face in range(6))
    x, z = _F2L_SLOT_COORDS[pair_id]
    corner_colors = frozenset(centers[_FACELET_CENTER[index] // 9]
                              for index in _POSITION_FACELETS[(x, -1, z)])
    edge_colors = frozenset(centers[_FACELET_CENTER[index] // 9]
                            for index in _POSITION_FACELETS[(x, 0, z)])
    pattern = _pair_pattern(state, corner_colors, edge_colors)
    return _f2l_pair_pattern_distances(centers, pair_id)[pattern]


def last_layer_edges_oriented(state: Sequence[int]) -> bool:
    """Detect orientation relative to the color currently centered on U."""
    state = validate_state(state)
    top_color = state[COLORS["U"] * 9 + 4]
    return all(state[index] == top_color for index in _TOP_EDGE_UP_FACELETS)


def zbll_eligible(state: Sequence[int]) -> bool:
    """Return whether F2L is solved and top edges are oriented in the current frame."""
    state = validate_state(state)
    return f2l_complete(state) and last_layer_edges_oriented(state)


def _edge_slots() -> Tuple[Tuple[int, int], ...]:
    by_position: Dict[_VEC, List[int]] = {}
    for index, (pos, _normal) in enumerate(_GEOMETRY):
        if sum(abs(value) for value in pos) == 2:
            by_position.setdefault(pos, []).append(index)
    return tuple(tuple(sorted(pair)) for _pos, pair in sorted(by_position.items()))


_EDGE_SLOTS = _edge_slots()
_EDGE_SLOT_BY_FACELET = {facelet: slot for slot, pair in enumerate(_EDGE_SLOTS) for facelet in pair}
_EDGE_TRANSITIONS = {}
for _move, _permutation in _MOVE_PERMUTATIONS.items():
    _transitions = []
    for _slot, (_first, _second) in enumerate(_EDGE_SLOTS):
        for _orientation, _source in enumerate((_first, _second)):
            _destination = _permutation[_source]
            _destination_slot = _EDGE_SLOT_BY_FACELET[_destination]
            _destination_orientation = 0 if _EDGE_SLOTS[_destination_slot][0] == _destination else 1
            _transitions.append(_destination_slot * 2 + _destination_orientation)
    _EDGE_TRANSITIONS[_move] = bytes(_transitions)


def cross_key(state: Sequence[int]) -> bytes:
    """Project a facelet state onto the Cross selected by its current Down center."""
    state = validate_state(state)
    down_color = state[COLORS["D"] * 9 + 4]
    goal_colors = ({down_color, state[COLORS[side] * 9 + 4]} for side in "FRBL")
    result = []
    for colors in goal_colors:
        found = None
        for slot, pair in enumerate(_EDGE_SLOTS):
            a, b = pair
            if {state[a], state[b]} == colors:
                down_facelet = a if state[a] == down_color else b
                orientation = 0 if down_facelet == a else 1
                found = slot * 2 + orientation
                break
        if found is None:
            raise ValueError("Cube state does not contain all four down-layer edge cubies")
        result.append(found)
    return bytes(result)


@lru_cache(maxsize=1)
def _cross_policy() -> Tuple[Dict[bytes, str], bytes]:
    goal = cross_key(SOLVED_STATE)
    distance = {goal: 0}
    policy: Dict[bytes, str] = {}
    queue = deque([goal])
    while queue:
        current = queue.popleft()
        next_distance = distance[current] + 1
        for move in MOVE_TOKENS:
            table = _EDGE_TRANSITIONS[move]
            neighbor = bytes(table[code] for code in current)
            if neighbor not in distance:
                distance[neighbor] = next_distance
                policy[neighbor] = inverse_move(move)
                queue.append(neighbor)
    return policy, goal


def shortest_cross_solution(state: Sequence[int]) -> List[str]:
    """Return a shortest move sequence that completes the current Down Cross."""
    key = cross_key(state)
    policy, goal = _cross_policy()
    result = []
    while key != goal:
        move = policy[key]
        result.append(move)
        table = _EDGE_TRANSITIONS[move]
        key = bytes(table[code] for code in key)
    return result


def cross_distance(state: Sequence[int]) -> int:
    """Return the exact outer-turn distance to the Cross for the current center frame."""
    return len(shortest_cross_solution(state))
