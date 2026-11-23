"""Local Streamlit demo for Cross next-move prediction."""

from pathlib import Path
import json
import time

import streamlit as st
import torch

from cubemind.environment.cube import (
    COLORS, MOVE_TOKENS, SOLVED_STATE, apply_move, apply_sequence, cross_complete,
    f2l_complete, last_layer_edges_oriented, parse_moves,
)
from cubemind.last_layer import recognize_zbll_case, solution_for_zbll_state, zbll_case_state
from cubemind.models.common import encode_history
from cubemind.models.zbll import ZBLLClassifier
from cubemind.neural_solver import (
    cube_solved, load_expert_f2l_policy, load_last_layer_checkpoint, load_solver_checkpoint,
    reference_solve, solve_expert_aware_full, solve_neural, _solve_cross_model,
)
from cubemind.expert_aware_f2l import (
    ExpertAwareF2LPolicy, load_canonical_expert_action_model, load_expert_pair_selector,
)
from cubemind.hierarchical_f2l import load_hierarchical_f2l, solve_hierarchical_f2l
from cubemind.training.engine import choose_device, model_from_checkpoint

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FACE_LAYOUT = {"U": (3, 0), "L": (0, 3), "F": (3, 3), "R": (6, 3), "B": (9, 3), "D": (3, 6)}
HEX = {0: "#f2d44e", 1: "#d94841", 2: "#2b9a59", 3: "#f4f2e8", 4: "#f08b32", 5: "#3f69bb"}
COLOR_NAMES = {0: "yellow", 1: "red", 2: "green", 3: "white", 4: "orange", 5: "blue"}


def cube_svg(state):
    size = 28
    margin = 2
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 336 252" role="img" aria-label="Rubik cube net">']
    for face, (start_x, start_y) in FACE_LAYOUT.items():
        face_id = COLORS[face]
        for row in range(3):
            for col in range(3):
                color_id = state[face_id * 9 + row * 3 + col]
                x = (start_x + col) * size
                y = (start_y + row) * size
                label = COLOR_NAMES[color_id]
                parts.append('<rect x="{}" y="{}" width="{}" height="{}" rx="3" fill="{}" stroke="#20272b" stroke-width="2"><title>{} facelet: {}</title></rect>'.format(
                    x + margin, y + margin, size - 2 * margin, size - 2 * margin, HEX[color_id], face, label))
    parts.append("</svg>")
    return "".join(parts)


def move_metrics(moves):
    turns = [move for move in moves if move[0] not in "xyz"]
    return {
        "stm": sum(2 if move[0].islower() or move[1:2] == "w" else 1 for move in turns),
        "htm": sum(move[0] in "URFDLB" and move[1:2] != "w" for move in moves),
        "rotations": sum(move[0] in "xyz" for move in moves),
    }


@st.cache_resource(show_spinner=False)
def load_policy(path, device_name="auto"):
    device = choose_device(device_name)
    model, checkpoint = model_from_checkpoint(path, device)
    history_length = int(checkpoint["model_config"].get("max_history", 8)) if checkpoint["model_config"].get("use_history", False) else 0
    return model, checkpoint, device, history_length


@st.cache_resource(show_spinner=False)
def load_zbll_policy(path, classes):
    device = choose_device("auto")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = ZBLLClassifier(classes).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint, device


@st.cache_resource(show_spinner=False)
def load_full_solver(path):
    device = choose_device("auto")
    model, checkpoint = load_solver_checkpoint(path, device)
    f2l_model, _f2l_checkpoint = load_expert_f2l_policy(
        ARTIFACTS / "reco-cfop-context.pt", model, device,
    )
    return model, checkpoint, device, f2l_model


@st.cache_resource(show_spinner=False)
def load_neural_full_solver(last_layer_path, cross_path, planner_path, expert_path,
                            selector_path, calibration_path):
    device = choose_device("cpu")
    last_layer, last_layer_checkpoint = load_last_layer_checkpoint(last_layer_path, device)
    cross, cross_checkpoint, cross_device, history_length = load_policy(cross_path, "cpu")
    f2l, calibration, _f2l_device = load_expert_aware_f2l_policy(
        planner_path, expert_path, selector_path, calibration_path, "cpu",
    )
    return last_layer, last_layer_checkpoint, cross, history_length, f2l, calibration, device


@st.cache_resource(show_spinner=False)
def load_hierarchical_f2l_policy(path):
    device = choose_device("auto")
    model, checkpoint = load_hierarchical_f2l(path, device)
    return model, checkpoint, device


@st.cache_resource(show_spinner=False)
def load_expert_aware_f2l_policy(planner_path, expert_path, selector_path, calibration_path,
                                 device_name="auto"):
    device = choose_device(device_name)
    planner, _planner_checkpoint = load_hierarchical_f2l(planner_path, device)
    expert, _expert_checkpoint = load_canonical_expert_action_model(expert_path, device)
    selector, _selector_checkpoint = load_expert_pair_selector(selector_path, device)
    calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))["action_fusion"]["calibration"]
    model = ExpertAwareF2LPolicy(
        planner, expert, calibration["expert_weight"], 1.0 - calibration["expert_weight"],
        calibration["expert_temperature"], calibration["planner_temperature"], selector,
        canonical_expert=True,
    ).eval()
    return model, calibration, device


def predict(model, state, history, history_length, device):
    state_tensor = torch.tensor([state], dtype=torch.long, device=device)
    history_tensor = encode_history([history], history_length, device)
    with torch.no_grad():
        logits = model(state_tensor, history_tensor)
        probabilities = torch.softmax(logits[0], dim=0)
    return probabilities.detach().cpu().tolist()


def rollout_states(model, state, history_length, device, max_moves=20):
    states = [tuple(state)]
    moves = []
    history = []
    current = tuple(state)
    for _ in range(max_moves):
        if cross_complete(current):
            break
        probabilities = predict(model, current, history, history_length, device)
        move_id = max(range(len(probabilities)), key=probabilities.__getitem__)
        move = MOVE_TOKENS[move_id]
        current = apply_move(current, move)
        history.append(move_id)
        moves.append(move)
        states.append(current)
    return moves, states


st.set_page_config(page_title="CubeMind — Cross policy", page_icon="🧊", layout="wide")
view = st.radio("Demo", ("Full solver", "Cross policy", "Last-layer recognition"), horizontal=True)

if view == "Full solver":
    st.title("CubeMind · scramble-to-solution experiment")
    st.caption("The new full-solve modes reuse expert-aware Cross/F2L search and add the trained neural Last Layer. Neural Greedy changes only Last-Layer inference; the guided mode uses the same upstream checkpoints and neural-guided Last-Layer search. Deterministic CFOP is a separate reference and never rescues neural results.")
    solver_path = ARTIFACTS / "neural-cfop-policy.pt"
    with st.form("full_solve"):
        scramble = st.text_input("3×3 scramble", "R U F' D2 L B U2 R' F D L2 B'")
        mode = st.selectbox("Solving mode", (
            "Neural Greedy", "Expert-Aware Neural-Guided Full Solve", "Neural-Guided Search",
            "Neural Cross + Hierarchical Neural F2L", "Neural Cross + Expert-Aware F2L",
            "Compare Hierarchical vs Expert-Aware F2L", "Deterministic Reference",
        ))
        max_moves = st.slider("Maximum moves for neural modes", min_value=20, max_value=180, value=120)
        submitted = st.form_submit_button("Solve and verify")
    if submitted:
        st.session_state.pop("full_solve_result", None)
        try:
            scramble_moves = parse_moves(scramble)
            if len(scramble_moves) > 100:
                raise ValueError("Please keep the scramble to 100 moves or fewer")
            start_state = apply_sequence(SOLVED_STATE, scramble_moves)
            if mode == "Deterministic Reference":
                result = reference_solve(scramble)
            elif mode in ("Neural Greedy", "Expert-Aware Neural-Guided Full Solve"):
                last_layer_path = ARTIFACTS / "last-layer-policy.pt"
                if not last_layer_path.exists():
                    raise ValueError("Train the from-scratch Last-Layer model with `python -m cubemind.train_last_layer --generate` first")
                last_layer, checkpoint, cross_model, history_length, expert_f2l, calibration, device = load_neural_full_solver(
                    str(last_layer_path), str(ARTIFACTS / "mlp-state-history.pt"),
                    str(ARTIFACTS / "hierarchical-f2l.pt"),
                    str(ARTIFACTS / "canonical-expert-f2l.pt"),
                    str(ARTIFACTS / "expert-aware-pair-selector.pt"),
                    str(ARTIFACTS / "expert-aware-f2l-evaluation.json"),
                )
                result = solve_expert_aware_full(
                    start_state, last_layer, device, cross_model, history_length, expert_f2l,
                    mode="search", strategy="eo-aware", max_moves=max_moves,
                    last_layer_mode=("greedy" if mode == "Neural Greedy" else "search"),
                )
                result["checkpoint"] = checkpoint.get("best_epoch")
                result["expert_fusion"] = calibration
            elif mode in ("Neural Cross + Hierarchical Neural F2L", "Neural Cross + Expert-Aware F2L",
                          "Compare Hierarchical vs Expert-Aware F2L"):
                model_path = ARTIFACTS / "hierarchical-f2l.pt"
                if not model_path.exists():
                    raise ValueError("Train the hierarchical model with `python -m cubemind.train_hierarchical_f2l` first")
                cross_model, cross_checkpoint, device, cross_history_length = load_policy(
                    str(ARTIFACTS / "mlp-state-history.pt"))
                started = time.perf_counter()
                cross_state, cross_history, cross_moves, cross_nodes, cross_failure = _solve_cross_model(
                    start_state, [], cross_model, cross_history_length, device,
                    "search", 256, 10, 10000, {start_state},
                )
                if cross_complete(cross_state):
                    planner, f2l_checkpoint, _device = load_hierarchical_f2l_policy(str(model_path))
                    expert_config = None
                    compare = mode == "Compare Hierarchical vs Expert-Aware F2L"
                    if mode == "Neural Cross + Expert-Aware F2L" or compare:
                        expert_policy, expert_config, _device = load_expert_aware_f2l_policy(
                            str(model_path), str(ARTIFACTS / "canonical-expert-f2l.pt"),
                            str(ARTIFACTS / "expert-aware-pair-selector.pt"),
                            str(ARTIFACTS / "expert-aware-f2l-evaluation.json"),
                        )
                    else:
                        expert_policy = None

                    def solve_f2l(policy, legal_conditioned=None):
                        solved = solve_hierarchical_f2l(
                            cross_state, policy, device, history=cross_history,
                            mode="search", width=128, branching=6, max_nodes=6000,
                            max_depth=22, max_moves=max_moves, max_actions_per_pair=22,
                            max_order_attempts=24, move_count=len(cross_moves),
                            conditional_legal=(policy is expert_policy if legal_conditioned is None
                                               else legal_conditioned),
                        )
                        all_moves = cross_moves + solved["moves"]
                        final = apply_sequence(start_state, all_moves)
                        verified_result = bool(solved["verified"] and cross_complete(final) and f2l_complete(final))
                        return {
                            "status": "F2L COMPLETE" if verified_result else "FAILED / INCOMPLETE",
                            "moves": all_moves, "last_state": final, "verified": verified_result,
                            "cross_complete": cross_complete(final), "f2l_complete": f2l_complete(final),
                            "moves_by_phase": {"CROSS": cross_moves, "F2L": solved["moves"]},
                            "solved_pair_mask": solved["solved_pair_mask"],
                            "pair_results": solved["pair_results"], "pair_attempts": solved["pair_attempts"],
                            "states_explored": cross_nodes + solved["states_explored"],
                            "runtime_seconds": time.perf_counter() - started,
                            "failure_reason": solved["failure_reason"],
                            "route": ("hierarchical neural F2L with expert action/pair guidance; no deterministic fallback"
                                      if policy is expert_policy else solved["route"]),
                            "checkpoint": f2l_checkpoint.get("best_epoch"),
                            "zbll_eligible_after_f2l": bool(verified_result and last_layer_edges_oriented(final)),
                            "expert_fusion": expert_config,
                        }

                    if compare:
                        baseline_result = solve_f2l(planner, legal_conditioned=True)
                        expert_result = solve_f2l(expert_policy)
                        result = expert_result
                        result["comparison_results"] = {
                            "Hierarchical planner": baseline_result,
                            "Expert-aware planner": expert_result,
                        }
                    else:
                        result = solve_f2l(expert_policy if expert_policy is not None else planner)
                else:
                    result = {
                        "status": "FAILED / INCOMPLETE", "moves": cross_moves,
                        "last_state": cross_state, "verified": False,
                        "cross_complete": cross_complete(cross_state), "f2l_complete": False,
                        "moves_by_phase": {"CROSS": cross_moves, "F2L": []},
                        "solved_pair_mask": (), "pair_results": [], "pair_attempts": [],
                        "states_explored": cross_nodes, "runtime_seconds": time.perf_counter() - started,
                        "failure_reason": cross_failure,
                        "route": "Neural Cross → hierarchical F2L search; no deterministic fallback",
                        "zbll_eligible_after_f2l": False,
                    }
            else:
                if not solver_path.exists():
                    raise ValueError("Train the synthetic CFOP model with `python -m cubemind.train_solver` first")
                model, checkpoint, device, f2l_model = load_full_solver(str(solver_path))
                cross_model, _cross_checkpoint, _cross_device, cross_history_length = load_policy(
                    str(ARTIFACTS / "mlp-state-history.pt"))
                result = solve_neural(start_state, model, device,
                                      mode="search" if mode == "Neural-Guided Search" else "greedy",
                                      max_moves=max_moves, cross_model=cross_model,
                                      cross_history_length=cross_history_length,
                                      f2l_model=f2l_model)
                result["checkpoint"] = checkpoint.get("best_epoch")
            states = [start_state]
            for move in result["moves"]:
                states.append(apply_move(states[-1], move))
            replay_verified = bool(result["verified"])
            is_f2l_mode = mode in (
                "Neural Cross + Hierarchical Neural F2L", "Neural Cross + Expert-Aware F2L",
                "Compare Hierarchical vs Expert-Aware F2L",
            )
            if is_f2l_mode:
                replay_verified = replay_verified and cross_complete(states[-1]) and f2l_complete(states[-1])
            else:
                replay_verified = replay_verified and cube_solved(states[-1])
            result["verified"] = replay_verified
            result["status"] = ("F2L COMPLETE" if is_f2l_mode and replay_verified
                                else "SOLVED" if replay_verified else "FAILED / INCOMPLETE")
            result["last_state"] = states[-1]
            st.session_state["full_solve_result"] = {
                "scramble": scramble, "mode": mode, "result": result, "states": states,
            }
        except (ValueError, RuntimeError, KeyError) as error:
            st.error(str(error))
    solved_run = st.session_state.get("full_solve_result")
    if solved_run:
        result = solved_run["result"]
        st.subheader("Scrambled state")
        st.markdown(cube_svg(solved_run["states"][0]), unsafe_allow_html=True)
        if result["verified"]:
            st.success("F2L COMPLETE · simulator replay verified" if solved_run["mode"] in (
                           "Neural Cross + Hierarchical Neural F2L", "Neural Cross + Expert-Aware F2L",
                           "Compare Hierarchical vs Expert-Aware F2L")
                       else "SOLVED · simulator replay verified")
        else:
            st.error("Status: FAILED / INCOMPLETE")
            st.write("Moves Attempted:", len(result["moves"]))
            st.write("Last Valid State:")
            st.markdown(cube_svg(result["last_state"]), unsafe_allow_html=True)
            st.write("Failure Reason:", result.get("failure_reason") or "solution did not replay to solved")
        st.write("Mode:", solved_run["mode"])
        st.write("Route:", result.get("route") or result.get("last_layer_route", ""))
        if result.get("provenance"):
            st.caption("Solution provenance: {}".format(result["provenance"]))
        if result.get("expert_fusion"):
            mix = result["expert_fusion"]
            st.caption("Expert action weight {:.2f} · planner action weight {:.2f} · validation calibrated".format(
                mix["expert_weight"], 1.0 - mix["expert_weight"]))
        if result.get("comparison_results"):
            st.subheader("Same-scramble F2L comparison")
            for label, compared in result["comparison_results"].items():
                with st.expander(label, expanded=True):
                    st.write("Replay verified:", "YES" if compared["verified"] else "NO")
                    st.write("Cross + F2L moves (STM):", move_metrics(compared["moves"])["stm"])
                    st.write("Runtime:", "{:.3f} ms".format(compared["runtime_seconds"] * 1000))
                    st.write("Search nodes:", compared["states_explored"])
                    st.code(" ".join(compared["moves"]) or "(no moves)")
                    for pair in compared.get("pair_results", []):
                        st.write("{}: {}".format(pair["target_slot"], " ".join(pair["moves"]) or "already solved"))
        for phase, title in (("CROSS", "Cross"), ("F2L", "F2L"), ("ZBLL", "ZBLL"), ("OLL", "OLL"), ("PLL", "PLL")):
            moves = result.get("moves_by_phase", {}).get(phase, [])
            if moves:
                st.markdown("**{}**".format(title))
                st.code(" ".join(moves))
        is_full_neural_mode = solved_run["mode"] in (
            "Neural Greedy", "Expert-Aware Neural-Guided Full Solve",
        )
        if solved_run["mode"] in (
                "Neural Cross + Hierarchical Neural F2L", "Neural Cross + Expert-Aware F2L",
                "Compare Hierarchical vs Expert-Aware F2L") or is_full_neural_mode:
            st.write("Cross Verified:", "YES" if result.get("cross_complete") else "NO")
            st.markdown("**Verified pair insertions**")
            for index, pair in enumerate(result.get("pair_results", []), 1):
                st.markdown("**F2L Pair {} — {}**".format(index, pair["target_slot"]))
                st.code(" ".join(pair["moves"]) or "(already solved)")
                st.write("Verified:", "YES" if pair["verified"] else "NO")
            st.markdown("**F2L slot status**")
            pair_mask = result.get("solved_pair_mask") or (False,) * 4
            for index, pair in enumerate(("FR", "FL", "BL", "BR")):
                solved = bool(pair_mask[index])
                st.write("{}: {}".format(pair, "verified" if solved else "unsolved"))
            if not result["verified"] and result.get("pair_attempts"):
                st.markdown("**Pair search attempts**")
                for attempt in result["pair_attempts"]:
                    st.write("{}: {} nodes · {}".format(
                        attempt["target_slot"], attempt["nodes"],
                        "verified" if attempt["verified"] else attempt.get("failure_reason") or "incomplete",
                    ))
            st.write("F2L Complete:", "YES" if result.get("f2l_complete") else "NO")
        if is_full_neural_mode and result.get("zbll_eligible_after_f2l"):
            f2l_state = dict(result.get("phase_states", [])).get("F2L")
            case_id = recognize_zbll_case(f2l_state) if f2l_state else None
            st.write("Exact ZBLL recognition reference:", case_id or "unrecognized")
            st.caption("Case recognition is shown as a baseline only; the neural solver generates its moves without case-to-algorithm lookup.")
        st.markdown("**Moves through F2L**" if solved_run["mode"] == "Neural Cross + Hierarchical Neural F2L" else "**Full Solution**")
        st.code(" ".join(result["moves"]) or "(no moves)")
        counts = move_metrics(result["moves"])
        st.write("Move Count (STM / HTM):", "{} / {}".format(counts["stm"], counts["htm"]))
        st.write("Cube rotations:", counts["rotations"])
        st.write("Runtime:", "{:.3f} ms".format(result["runtime_seconds"] * 1000))
        st.write("States Explored:", result.get("states_explored", 0))
        st.write("ZBLL eligible after F2L:", "Yes" if result.get("zbll_eligible_after_f2l") else "No")
        st.write("Verification:", result["status"] if result["verified"] else "NOT VERIFIED")
        replay_step = st.slider("Replay solution", min_value=0, max_value=len(solved_run["states"]) - 1, value=0)
        st.markdown(cube_svg(solved_run["states"][replay_step]), unsafe_allow_html=True)
        st.caption("Step {} / {}".format(replay_step, len(solved_run["states"]) - 1))
    st.stop()

if view == "Last-layer recognition":
    st.title("CubeMind · ZBLL case recognition")
    st.caption("Recognition comparison only: exact case and database algorithm lookup are the deterministic reference; the classifier predicts a case label. Full-solve neural modes generate moves with a separate policy and do not use this lookup.")
    database = json.loads((ROOT / "data" / "algorithms" / "zbll_cases.json").read_text(encoding="utf-8"))
    cases = database["cases"]
    case_by_id = {case["case_id"]: case for case in cases}
    with st.form("zbll_prediction"):
        case_id = st.selectbox("Representative case", [case["case_id"] for case in cases])
        scramble = st.text_input("Or enter a complete cube scramble", "")
        submitted = st.form_submit_button("Recognize case")
    if submitted:
        st.session_state.pop("zbll_result", None)
        try:
            state = apply_sequence(SOLVED_STATE, parse_moves(scramble)) if scramble.strip() else zbll_case_state(case_id)
            if len(parse_moves(scramble)) > 100:
                raise ValueError("Please keep the scramble to 100 moves or fewer")
            st.session_state["zbll_result"] = {"state": state, "selected_case": case_id}
        except ValueError as error:
            st.error(str(error))
    result = st.session_state.get("zbll_result")
    if result:
        state = result["state"]
        exact_id = recognize_zbll_case(state)
        st.subheader("Last-layer state")
        st.markdown(cube_svg(state), unsafe_allow_html=True)
        if exact_id is None:
            st.warning("This state is not ZBLL-eligible: it needs solved F2L and oriented upper-layer edges.")
        else:
            algorithm = case_by_id[exact_id]["algorithms"][0]
            verified_solution = solution_for_zbll_state(state, exact_id)
            model_path = ARTIFACTS / "zbll-classifier.pt"
            left, right = st.columns(2)
            with left:
                st.metric("Exact deterministic case", exact_id)
                st.code(algorithm)
            with right:
                if model_path.exists():
                    model, checkpoint, device = load_zbll_policy(str(model_path), len(cases))
                    state_tensor = torch.tensor([state], dtype=torch.long, device=device)
                    with torch.no_grad():
                        probabilities = torch.softmax(model(state_tensor)[0], dim=0)
                    predicted_id = int(probabilities.argmax())
                    st.metric("Neural prediction", cases[predicted_id]["case_id"],
                              "{}% confidence".format(round(float(probabilities[predicted_id]) * 100, 1)))
                    st.code(cases[predicted_id]["algorithms"][0])
                    st.caption("Checkpoint epoch {} · exact match: {}".format(
                        checkpoint["best_epoch"], "yes" if cases[predicted_id]["case_id"] == exact_id else "no"))
                else:
                    st.info("Train the classifier with `python -m cubemind.train_zbll` to show its prediction.")
            st.markdown("**Verified database-baseline solution**")
            if verified_solution:
                st.code(" ".join(verified_solution))
                st.success("Replay verified in the simulator; the final state is solved up to whole-cube orientation.")
            else:
                st.error("The case matched, but no database algorithm replayed successfully for this orientation.")
    st.stop()

st.title("CubeMind · Cross decision demo")
st.caption("The existing Cross policy is a from-scratch model trained on a synthetic shortest-Cross oracle; it is not a human-imitation result.")

checkpoints = sorted(path for path in ARTIFACTS.glob("*.pt")
                     if path.name.startswith(("mlp-", "transformer-")))
if not checkpoints:
    st.warning("No trained checkpoint is available. From the project folder, run `python -m cubemind.train --config configs/mlp.yaml`.")
    st.stop()

checkpoint_names = {path.name: path for path in checkpoints}
with st.form("prediction"):
    scramble = st.text_input("Scramble (outer turns only)", "R U F' D2 L B U2 R' F D L2 B'")
    selected_name = st.selectbox("Policy checkpoint", list(checkpoint_names))
    top_k = st.slider("Top move probabilities", min_value=1, max_value=5, value=3)
    submitted = st.form_submit_button("Predict and roll out")

if submitted:
    st.session_state.pop("result", None)
    try:
        scramble_moves = parse_moves(scramble)
        if len(scramble_moves) > 100:
            raise ValueError("Please keep the scramble to 100 moves or fewer")
        start_state = apply_sequence(SOLVED_STATE, scramble_moves)
        model, checkpoint, device, history_length = load_policy(str(checkpoint_names[selected_name]))
        probabilities = predict(model, start_state, [], history_length, device)
        ranked = sorted(range(18), key=lambda move_id: (-probabilities[move_id], move_id))[:top_k]
        moves, states = rollout_states(model, start_state, history_length, device)
        st.session_state["result"] = {
            "scramble": scramble,
            "checkpoint": checkpoint["name"],
            "probabilities": [(MOVE_TOKENS[i], probabilities[i]) for i in ranked],
            "moves": moves,
            "states": states,
        }
    except ValueError as error:
        st.error(str(error))

result = st.session_state.get("result")
if result:
    st.subheader("Scrambled state")
    st.markdown(cube_svg(apply_sequence(SOLVED_STATE, result["scramble"])), unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown("**Top next-move probabilities**")
        st.dataframe([{"move": move, "probability": round(probability, 4)} for move, probability in result["probabilities"]], hide_index=True)
    with right:
        st.markdown("**Autoregressive Cross attempt**")
        st.code(" ".join(result["moves"]) or "(no moves)")
        st.write("Cross completed:", "Yes" if cross_complete(result["states"][-1]) else "No")
        st.caption("Policy: {}".format(result["checkpoint"]))
    step = st.slider("Replay predicted sequence", min_value=0, max_value=len(result["moves"]), value=0)
    st.markdown(cube_svg(result["states"][step]), unsafe_allow_html=True)
    st.caption("Step {} / {} · Cross {}".format(step, len(result["moves"]), "complete" if cross_complete(result["states"][step]) else "incomplete"))
    st.info("No permission-cleared expert reconstruction is loaded, so the demo cannot show an expert comparison.")
