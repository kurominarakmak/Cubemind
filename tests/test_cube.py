import pytest

from cubemind.environment.cube import (
    COLOR_NAMES,
    COLORS,
    MOVE_TOKENS,
    F2L_PAIR_NAMES,
    SIMULATOR_MOVE_TOKENS,
    SOLVED_STATE,
    apply_move,
    apply_sequence,
    cross_complete,
    cross_solved,
    cross_frame_move,
    f2l_complete,
    f2l_pair_distance,
    f2l_slot_count,
    pair_solved,
    inverse_move,
    inverse_sequence,
    last_layer_edges_oriented,
    normalize_cube_orientation,
    parse_moves,
    shortest_cross_solution,
    normalize_cross_frame,
    solved_pair_mask,
    zbll_eligible,
    _cross_policy,
)


def test_every_move_has_an_inverse_and_four_quarter_turns_are_identity():
    for move in SIMULATOR_MOVE_TOKENS:
        assert apply_move(apply_move(SOLVED_STATE, move), inverse_move(move)) == SOLVED_STATE
        assert apply_sequence(SOLVED_STATE, [move] * 4) == SOLVED_STATE


def test_scramble_replay_and_inverse():
    scramble = "R U F' D2 L B U2"
    state = apply_sequence(SOLVED_STATE, scramble)
    assert apply_sequence(state, inverse_sequence(scramble)) == SOLVED_STATE


def test_notation_accepts_wide_slice_rotation_and_normalizes_half_turn_prime():
    assert parse_moves("r Rw M E S x y z U2' R3") == ["r", "Rw", "M", "E", "S", "x", "y", "z", "U2", "R'"]


def test_notation_rejects_invalid_moves():
    with pytest.raises(ValueError):
        parse_moves("R4")


def test_wide_turn_matches_outer_turn_plus_middle_slice():
    state = apply_sequence(SOLVED_STATE, "U F R' D2")
    assert apply_move(state, "r") == apply_sequence(state, "R M'")
    assert apply_move(state, "u") == apply_sequence(state, "U E'")
    assert apply_move(state, "f") == apply_sequence(state, "F S")


def test_all_cube_orientations_normalize_back_to_the_fixed_frame():
    for rotation in ("x", "x'", "x2", "y", "y'", "y2", "z", "z'", "z2"):
        state = apply_sequence(SOLVED_STATE, "R U F' {}".format(rotation))
        assert normalize_cube_orientation(state) == apply_sequence(SOLVED_STATE, "R U F'")


def test_shortest_cross_solution_replays_to_cross():
    state = apply_sequence(SOLVED_STATE, "R U F' D2 L B U2 R' F")
    assert not cross_complete(state)
    solution = shortest_cross_solution(state)
    assert cross_complete(apply_sequence(state, solution))


def test_shortest_cross_solution_uses_current_center_frame():
    state = apply_sequence(SOLVED_STATE, "R U F' x y")
    solution = shortest_cross_solution(state)
    assert cross_complete(apply_sequence(state, solution))


def test_solved_cube_has_a_complete_cross():
    assert cross_complete(SOLVED_STATE)
    assert f2l_slot_count(SOLVED_STATE) == 4


def test_last_layer_edge_orientation_requires_solved_f2l_for_zbll_eligibility():
    assert f2l_complete(SOLVED_STATE)
    assert last_layer_edges_oriented(SOLVED_STATE)
    assert zbll_eligible(SOLVED_STATE)
    edge_flip = apply_sequence(SOLVED_STATE, "F R U R' U' F'")
    assert f2l_complete(edge_flip)
    assert not last_layer_edges_oriented(edge_flip)
    assert not zbll_eligible(edge_flip)


def test_layer_detectors_follow_center_colors_after_cube_rotation():
    rotated = apply_sequence(SOLVED_STATE, "x")
    assert cross_complete(rotated)
    assert f2l_complete(rotated)
    assert f2l_slot_count(rotated) == 4
    assert last_layer_edges_oriented(rotated)
    assert zbll_eligible(rotated)


def test_pair_masks_distinguish_an_unslotted_pair_from_three_solved_slots():
    state = apply_sequence(SOLVED_STATE, "U R U' R'")
    assert cross_solved(state)
    assert F2L_PAIR_NAMES == ("FR", "FL", "BL", "BR")
    assert solved_pair_mask(state) == (False, True, True, True)
    assert not pair_solved(state, 0)
    assert not pair_solved(state, "FR")
    assert all(pair_solved(state, name) for name in F2L_PAIR_NAMES[1:])
    assert not f2l_complete(state)
    assert f2l_complete(apply_sequence(state, "R U R' U'"))


def test_pair_detector_tracks_current_center_frame_and_validates_pair_id():
    assert solved_pair_mask(SOLVED_STATE) == (True, True, True, True)
    for rotation in ("x", "y", "z", "x y'", "z2 x'"):
        rotated = apply_sequence(SOLVED_STATE, rotation)
        assert cross_solved(rotated)
        assert f2l_complete(rotated)
        assert solved_pair_mask(rotated) == (True, True, True, True)
    with pytest.raises(ValueError):
        pair_solved(SOLVED_STATE, "XX")


def test_pair_masks_cover_all_sixteen_solved_slot_configurations():
    extracts = ("U R U' R'", "U' L' U L", "U' B' U B", "U B U' B'")
    for expected_bits in range(16):
        state = SOLVED_STATE
        for pair_id, extraction in enumerate(extracts):
            if not expected_bits & (1 << pair_id):
                state = apply_sequence(state, extraction)
        expected = tuple(bool(expected_bits & (1 << pair_id)) for pair_id in range(4))
        assert cross_solved(state)
        assert solved_pair_mask(state) == expected
        assert f2l_slot_count(state) == sum(expected)


def test_pair_distance_is_zero_only_after_the_named_pair_is_solved():
    state = apply_sequence(SOLVED_STATE, "U R U' R'")
    assert f2l_pair_distance(state, "FR") > 0
    assert f2l_pair_distance(apply_sequence(state, "R U R'"), "FR") == 0
    assert f2l_pair_distance(SOLVED_STATE, "FL") == 0


def test_cross_frame_normalization_and_move_mapping_preserve_outer_turns():
    state = apply_sequence(SOLVED_STATE, "R U F' x")
    centers = tuple(state[face * 9 + 4] for face in range(6))
    for color in range(6):
        normalized = normalize_cross_frame(state, color)
        assert normalized[COLORS["D"] * 9 + 4] == color
        for move in MOVE_TOKENS:
            mapped = cross_frame_move(move, centers, color)
            assert apply_move(normalized, mapped) == normalize_cross_frame(apply_move(state, move), color)


def test_fixed_frame_places_the_white_cross_on_down():
    assert COLOR_NAMES[COLORS["D"]] == "white"
    assert COLOR_NAMES[COLORS["U"]] == "yellow"


def test_cross_projection_covers_every_legal_down_edge_configuration():
    # Four distinct edges occupy 12 slots and each has two orientations: 12P4 * 2^4 states.
    policy, _goal = _cross_policy()
    assert len(policy) == 12 * 11 * 10 * 9 * 16 - 1
