# Research proposal

## Question

Can small neural models trained from scratch imitate expert 3×3 CFOP move choices, including Cross+1 and F2L actions, and can edge-orientation-aware planning make ZBLL reachable often enough to change solver outcomes?

The project keeps source-separated evidence tracks. Human reco.nz reconstructions supervise demonstrated next actions. Simulator-generated trajectories supply controlled coverage. The algorithm table supports exact recognition and offline teacher labels; it is not evidence of human-like strategy and is not a neural inference-time solver.

## Research foundation

The original synthetic Cross experiment remains the controlled engineering baseline: shortest-Cross BFS generates labels, MLP/Transformer models are randomly initialized, and held-out scramble evaluation separates action accuracy from complete Cross rollout. Its labels are optimal Cross paths, not human choices.

## Human CFOP data

The dedicated CFOP reco.nz sample traverses the 3×3 CFOP index; OH and other methods/events are excluded from model training. The user authorized local research training for this project. The collector preserves scramble/solve/source IDs, solver and reconstructor names, phase annotations, inspection rotations, selected Cross color, and move count. Of 9,288 unique entries in the filtered 3×3 CFOP index, 9,109 passed full simulator replay; the remaining 179 unsupported or non-replayable records are recorded as exclusions. A separate all-method 3×3 crawl found 10,474 entries and 10,274 replay-valid solves. The two crawls returned non-identical CFOP IDs, so the broad scan remains separate from the model corpus. The site-wide ~14,000 count is broader than the project's 3×3 event filter. Raw, rejected, and processed human data remain local and outside the checked-in project.

The from-scratch MLP action model includes current facelets, the previous eight moves, phase, selected Cross color, edge orientation, pair/cross status, move count, and available source strategy annotations. Solver-held-out evaluation uses 9,109 valid solves across 386 solver identities. `1st/2nd pairs`, X-cross, pseudo-slotting, and ZBLS labels are retained when present; unannotated intent is not inferred from moves alone.

The model predicts the next action in the reconstruction. It does not yet plan an entire Cross→F2L route, optimize moves, or model inspection-time recognition. State-only and context-aware policies are compared against a train-frequency baseline by exact next-action agreement.

## Last-layer data and recognition

The algorithm baseline has 472 ZBLL cases plus 21 PLL cases and 302 ZBLS cases. ZBLL/PLL uses the MIT-licensed Alg-Trainer data, which credits ZBLL to Jabari Nuruddin. ZBLS uses the user-selected [CubeRoot 3×3 page](https://cuberoot.me/alg/3x3/zbls) and its [public API](https://api.cuberoot.me/v1/alg/sets/3x3/zbls). The importer preserves source notation and case IDs, normalizes supported notation, and verifies each primary algorithm against its setup in the simulator.

Exact case recognition normalizes whole-cube orientation and U-layer AUF. A lightweight 493-way PyTorch classifier is trained from random initialization on simulator-generated algorithm cases with held-out pose presentations. Exact deterministic recognition is the required baseline. Because the same case IDs occur in each pose split, this evaluation measures pose generalization rather than unseen-case discovery.

## Expert-aware neural Last Layer

The Last-Layer policy/value MLP is initialized randomly and trained for OLL, PLL, and ZBLL actions. Human targets come only from explicit `OLL`, `PLL`, or `ZBLL` reconstruction segments whose preconditions and endpoint are confirmed by CubeMind replay. Human splits are grouped by solver identity. Synthetic CFOP teacher trajectories and ZBLL algorithm trajectories are generated into separate files and combined with eligible expert training rows only at training time. Synthetic ZBLL case identities are split as groups so held-out cases do not enter gradients.

The model predicts simulator moves and remaining demonstration cost from facelets, recent move history, phase, and the recorded Cross color. Neural greedy rollout and neural-guided beam search solve phase subgoals. Exact EO, OLL, solved-cube, and F2L checks validate transitions; exact ZBLL recognition is measured as a baseline but does not select the neural solving algorithm. The external CFOP package remains a separately labeled reference and offline teacher.

## Policy comparison

Compare:

1. Conventional CFOP: Cross → F2L → OLL → PLL.
2. EO-aware CFOP: Cross → F2L → neural ZBLL when eligible, otherwise neural OLL → PLL.

Both neural policies share the existing expert-aware Cross/F2L checkpoints and use identical held-out 3×3 scrambles. Report stage-level and full-solve replay rates, ZBLL eligibility, move counts, rotations, search nodes, compute runtime, and expert-action agreement. Compare greedy with neural-guided search and a matched value-only search ablation. The deterministic solver and exact case recognizer remain explicit baselines. Human execution and recognition times require timed human observations and are not inferred from compute timings.

## Next steps

Expand the solver-held-out reconstruction corpus, obtain more explicit Cross+1 and pair-order annotations, and evaluate repeated seeds before behavioral claims. Full F2L-to-ZBLL planning remains an advanced extension: test pair order, pseudo-slotting, last-pair alternatives, and edge orientation against the measured current pipeline, especially given the gap between controlled ZBLL success and states produced by the full F2L route. A direct one-look recognition study should separately test shape/color pattern cues and algorithm choice against exact case recognition.

The implemented flat end-to-end neural baseline includes greedy rollout, neural-guided search, a separate deterministic reference, and simulator-verified failure reporting. Its first full 100-scramble result was 0/100 neural F2L completions despite 100/100 Cross completion under search; this historical result is retained in its original artifact.

The later Milestone B hierarchical controller addressed the F2L bottleneck with pair-conditioned policy/value/selection heads and verified pair subgoals. It achieved 98/100 Cross+F2L completions on a separate held-out seed and 22/100 under a matched uniform-action ablation. It stops before the Last Layer, and it uses exact projected pair/Cross distances only as search heuristics. See the Milestone B section of `docs/experiment_report.md` for the outcome and remaining limits.

An expert-aware extension trained a canonical-frame action model and pair-order selector from random initialization on solver-disjoint reco.nz demonstrations. The old reco.nz checkpoint was not rewritten; its feature adapter scored 45.90% Top-1 on the held-out legal-action subset, compared with 69.13% for the newly trained canonical expert model and 60.57% for the mixed planner. Pair-order Top-1 was 62.34% versus 56.89%. Validation selected a pure expert action distribution rather than a blend. On 100 new scrambles the expert-aware policy completed F2L on 95/100 (Wilson 95% interval 88.82–97.85%), while both exact-original and legal-normalized planner modes completed 90/100. The paired five-point gain was not statistically significant (exact McNemar p=0.227), and mean move count was slightly higher. In a separate 20-case component ablation, expert action-only finished 19/20, expert pair-order-only 18/20, combined action+pair 20/20, uniform legal action 9/20, and STM order selection 20/20 at higher compute cost. These small ablations support further testing but do not establish that a component reliably improves completion. The historical 98/100 planner result is preserved on its separate seed. Human ratings or timing remain unmeasured.

## Milestones C and D measured outcome

The new Last-Layer policy is a from-scratch 920-input PyTorch action/value MLP trained on separate solver-held-out reco.nz and simulator-generated sources. On the untouched controlled test states, it completed PLL on 158/158, OLL→PLL on 158/158, and eligible ZBLL on 194/200. Exact recognition and the neural case classifier both scored 200/200 on known-case presentations; case classification is not used to choose moves.

On 100 fresh held-out scrambles, expert-aware Cross/F2L plus neural OLL→PLL solved 93/100, while EO-aware ZBLL-or-OLL/PLL routing solved 90/100. The EO-aware result meets the 90% point target, with a Wilson 95% interval of 82.56–94.48%. Only 10/100 expert-aware F2L outputs were ZBLL-eligible and direct ZBLL solved 7/10; controlled-stage ZBLL success was higher at 194/200. This is evidence of a distribution gap between controlled eligible states and states produced by the full solver. Paired outcomes did not establish an EO-aware completion benefit (exact McNemar p=0.25).

Neural-guided search completed 90/100 versus 10/100 for greedy rollout and 0/100 for a value-only matched-budget ablation. This supports the value of learned action guidance in this search setup. It does not show that the ZBLL database improves solving, because the neural solver never selects its move sequence from the table. The current system meets the specified point targets, but confidence intervals remain wide; F2L-to-ZBLL planning and broader human ZBLL coverage remain next research steps. Execution speed, recognition time, fingertricks, and human preference were not measured.
