# CubeMind

**Learning speedcubing decisions from demonstrations, with a verified Cross foundation and 3×3 CFOP/ZB last-layer experiments.**

CubeMind studies next-move imitation with a deterministic 3×3 simulator and PyTorch models trained from random initialization. The original synthetic Cross experiments remain intact. A local reco.nz dataset supports human Cross/F2L and last-layer action learning; the verified ZBLL table remains a recognition and offline-teacher baseline, not an inference-time move lookup.

## What is implemented

- The original shortest-Cross BFS oracle, MLP/Transformer experiments, held-out-scramble evaluation, beam-search ablation, and Cross demo.
- A 54-facelet simulator for outer, wide, slice, and cube-rotation moves. Cross/F2L/edge-orientation checks follow the current center colors, and whole-cube orientation can be normalized deterministically.
- A paginated [reco.nz](https://reco.nz/) collector for the 3×3 event across methods, excluding OH and other events. The latest all-method pass found 10,474 entries and accepted 10,274 after simulator replay. A separate CFOP-only training corpus has 9,109 valid solves and yields 324,097 Cross, Cross+1, F2L, and annotated ZBLS action labels. Raw reconstructions, rejection logs, and processed human rows are local and Git-ignored.
- A from-scratch PyTorch human next-action model with state, recent move history, phase, selected Cross color, edge orientation, move count, and pair/cross status context. Train/validation/test splits hold out solver names. Reco annotations such as `1st/2nd pairs`, X-cross, and pseudo-slotting are preserved where recorded. A separate hierarchical F2L policy/value/pair-order model uses source-tagged human and simulator demonstrations and verifies each pair subgoal in the simulator.
- MIT-licensed Alg-Trainer data for 493 ZBLL cases including 21 PLL cases, plus 302 ZBLS cases imported from CubeRoot's public 3×3 API. Each case keeps one primary algorithm and source notation; all are replay-verified. See [`data/algorithms/README.md`](data/algorithms/README.md).
- An exact ZBLL recognizer that normalizes AUF and whole-cube orientation, plus a randomly initialized PyTorch classifier trained only on simulator-generated algorithm cases. Exact recognition is the accuracy baseline; the neural model is not assumed to be better.
- A paired 3×3 benchmark for conventional CFOP and EO-aware CFOP (Cross → F2L/ZBLS → ZBLL), with replay-verified solves, eligibility rate, STM count, and computational time. Human recognition and execution times are not measured.
- A separate end-to-end CFOP experiment: a from-scratch PyTorch policy/value MLP trained on simulator-verified synthetic CFOP teacher paths, ZBLL algorithm trajectories, and DAgger-style post-Cross states from the project Cross model. It has greedy and neural-guided beam-search modes; the deterministic reference never rescues neural attempts.
- An expert-aware hierarchical F2L extension: canonical-frame action and pair-order models trained from scratch on solver-disjoint reco.nz rows. Validation-only calibration feeds expert scores into the existing pair-conditioned search; historical planner and expert checkpoints remain unchanged.
- A from-scratch OLL/PLL/ZBLL policy/value model trained on replay-confirmed reco.nz last-layer segments plus separately stored simulator-generated CFOP/ZBLL trajectories. It uses neural greedy rollout or neural-guided search; exact case recognition is reported as a reference only, and failed neural routes never fall back to a table or full-cube solver.
- A Streamlit demo with a Cross-policy view and an interactive last-layer view showing the cube, exact case, neural prediction, predicted algorithm, and replay-verified solution.

The user authorized local use of reco.nz 3×3 reconstructions for this community research project. Raw and processed solve data are kept local and are not included in the checked-in project. ZBLL/PLL strings come from the MIT-licensed Alg-Trainer source; ZBLS representatives come from CubeRoot's public algorithm API and are replay-verified. See [dataset provenance](docs/dataset_provenance.md).

## Install

Python 3.9 or newer is supported. Install PyTorch for your platform, then:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,demo]'
npm install
```

Training selects CUDA, Apple Silicon MPS, or CPU automatically. The Node dependency is used only by the deterministic full-CFOP policy benchmark.

## Reproduce

The original synthetic Cross experiment remains separately reproducible:

```bash
python -m cubemind.data.prepare --solves 1200 --seed 17
python -m cubemind.train --config configs/mlp.yaml
python -m cubemind.train --config configs/mlp_state_only.yaml
python -m cubemind.train --config configs/transformer.yaml
python -m cubemind.evaluate --checkpoint artifacts/mlp-state-history.pt --output artifacts/eval-mlp-state-history.json
```

Collect all pages in the current 3×3 CFOP filter and train on the local human data:

```bash
python -m cubemind.data.reco
python -m cubemind.train_f2l --epochs 20
```

To collect all 3×3 methods while keeping the CFOP training data separate:

```bash
python -m cubemind.data.reco --all-3x3-methods
```

Train the ZBLL classifier from synthetic algorithm cases and compare the two CFOP policies:

```bash
python scripts/import_cuberoot_zbls.py
python -m cubemind.train_zbll --epochs 15
npm run benchmark:cfop
streamlit run demo/app.py
pytest
```

Train and evaluate the end-to-end neural solver separately from the human imitation experiment:

```bash
python -m cubemind.train_solver --dataset data/processed/neural_solver/cfop_teacher.jsonl --train-solves 120 --validation-solves 24 --epochs 20 --seed 73 --device auto --regenerate
python -m cubemind.train_solver --dataset data/processed/neural_solver/cfop_teacher.jsonl --train-solves 120 --validation-solves 24 --epochs 20 --seed 79 --device auto
python -m cubemind.benchmark_solver --output artifacts/neural-cfop-benchmark-hybrid-search.json --count 100 --seed 20261124 --device auto --beam-width 4 --branching 3 --max-nodes 500 --max-moves 120
streamlit run demo/app.py
```

The first command creates the seed-73 synthetic CFOP data and DAgger trajectories; the second trains the final random-initialized seed-79 policy/value MLP from those cached rows. Human reco.nz action rows are not used to train this model. The interactive neural solver composes it with the separate reco.nz F2L action model. The checkpoint and training report are `artifacts/neural-cfop-policy.pt` and `artifacts/neural-cfop-training.json`.

Train and benchmark Milestone B's separate all-pair F2L model:

```bash
python -m cubemind.train_hierarchical_f2l --epochs 8 --recovery-count 24 --device auto
python -m cubemind.benchmark_hierarchical_f2l --count 100 --seed 20261127 --device auto --width 128 --branching 6 --max-nodes 6000 --max-depth 22 --output artifacts/hierarchical-f2l-benchmark.json
```

Its report and checkpoint are `artifacts/hierarchical-f2l-training.json` and `artifacts/hierarchical-f2l.pt`. Human reconstructions and simulator-generated teacher/recovery rows retain separate source labels. The reference teacher is used for offline labels and comparison only; neural inference has no deterministic fallback.

Train and evaluate the expert-aware extension using the local solver-disjoint reco.nz splits:

```bash
python -m cubemind.train_expert_aware_f2l --epochs 12 --seed 1701 --device auto
python -m cubemind.benchmark_expert_aware_f2l --count 100 --seed 20261130 --device auto --modes planner_original,planner_legal,combined --width 128 --branching 6 --max-nodes 6000 --max-depth 22 --output artifacts/expert-aware-f2l-benchmark.json
python -m cubemind.benchmark_expert_aware_f2l --count 20 --seed 20261129 --device auto --modes expert_action_only,expert_pair_order,uniform_action,execution_cost --width 128 --branching 6 --max-nodes 6000 --max-depth 22 --output artifacts/expert-aware-f2l-ablation-20-case.json
streamlit run demo/app.py
```

This extension trains only on source-labeled reco.nz rows; pair-order labels require a replay-verified single-slot transition. Temperature and action weights use validation cubers only. The final 100-scramble set is seed 20261130; the 20-case ablation set is separate. The historical 98/100 result remains in `artifacts/hierarchical-f2l-benchmark.json`.

Train and evaluate the expert-aware neural Last Layer and integrated full solver:

```bash
python -m cubemind.train_last_layer --generate --train-solves 1200 --validation-solves 160 --test-solves 160 --epochs 24 --seeds 73 74 75 --device auto
python -m cubemind.benchmark_last_layer --stages-only --stage-split validation --device cpu --beam-width 128 --branching 8 --max-nodes 20000 --action-score-weight 1 --oll-budget 40 --pll-budget 40 --zbll-budget 100 --output artifacts/last-layer-stage-validation-wide-budget.json
python -m cubemind.benchmark_last_layer --count 100 --seed 20261123 --stage-split test --device cpu --beam-width 128 --branching 8 --max-nodes 20000 --action-score-weight 1 --oll-budget 40 --pll-budget 40 --zbll-budget 100 --output artifacts/last-layer-benchmark.json
streamlit run demo/app.py
```

Human reconstructions and synthetic teacher trajectories are stored in separate local JSONL files under `data/processed/last_layer/`. Training splits human examples by solver identity and holds out synthetic ZBLL case identities. The full benchmark excludes raw reco.nz scrambles, prior evaluation scrambles, and generated training scrambles. Neural modes use the existing expert-aware Cross/F2L stack; deterministic CFOP is a separate baseline and never rescues a neural attempt.

`reco.py` follows all index pages by default; `--pages` and `--max-solves` can limit a run. It resumes from local raw/checkpoint records, validates full-solve replay, and writes exclusions to an ignored rejection log. Raw records live under `data/raw/reco_nz/` and solver-held-out action rows under `data/processed/reco_cfop/`. ZBLL case poses are generated under ignored `data/processed/zbll_algorithm_synthetic/`. Human and algorithm-synthetic sources remain separate.

## Results and interpretation

The original Cross report and both reco.nz runs are documented in [`docs/experiment_report.md`](docs/experiment_report.md). On the latest solver-held-out test set (38,057 actions from 1,048 solves and 39 held-out solvers), the context model scored 56.75% Top-1, 82.15% Top-3, and 90.60% Top-5. Its phase Top-1 was 61.42% on F2L and 58.67% on 150 explicitly annotated ZBLS actions. The state-only model scored 49.76% Top-1 and the training-label frequency baseline 18.12%. These measure exact next-action imitation, not full-solve success, human intent, or move optimality. The test partition changed after the corpus expansion, so earlier scores are not paired comparisons.

On the final held-out test set, exact ZBLL recognition and the from-scratch neural case classifier both reached 100% on 200 AUF/orientation presentations of known case identities. This measures pose recognition, not discovery of unseen cases or solving; the classifier is not used to choose neural moves.

The external CFOP package used by the policy benchmark is a deterministic solver baseline, not a neural training-data source. Algorithm source and evaluation limits are documented in the experiment report. No human execution-speed claim is made.

The earlier 100-scramble end-to-end CFOP policy evaluation is `artifacts/neural-cfop-benchmark-hybrid-search.json`. Its two flat neural searches reached Cross on 99/100 but completed F2L on 0/100. The deterministic reference solved 100/100. This result remains in place; it is a different solver from the later hierarchical model.

The separate **Milestone B Cross+F2L controller** stops before OLL/PLL/ZBLL. On 100 held-out scrambles (seed 20261127), hierarchical neural-guided search preserved Cross and completed all four F2L pairs on 98/100; uniform-action search completed 22/100, greedy completed 0/100, and both flat F2L baselines completed 0/100. The neural model reached 58.30% held-out human F2L action Top-1, below the previous reco.nz model's 64.92% on the same rows, so higher exact-move imitation accuracy does not explain its solving result. Successful neural routes averaged 34.72 STM versus 29.93 STM for the deterministic reference on those same cases. See the [Milestone B report](docs/experiment_report.md) and [benchmark JSON](artifacts/hierarchical-f2l-benchmark.json). Human execution and recognition times were not measured.

The subsequent expert-aware extension reached 69.13% held-out human-action Top-1 and 62.34% pair-order Top-1, versus 60.57% and 56.89% for the mixed planner on the same reconstruction rows. Validation selected the expert-only action policy. On 100 new scrambles it completed F2L on 95/100 (Wilson 95% interval 88.82–97.85%), versus 90/100 for both the original and legal-normalized planner modes. The paired five-point gain was not statistically significant (exact p=0.227). Mean STM was 34.40 versus 34.07 for the original planner, so move efficiency did not improve. The historical 98/100 result is preserved on its separate seed. See the [expert-aware report](docs/experiment_report.md#expert-aware-f2l-extension) and [case-level benchmark](artifacts/expert-aware-f2l-benchmark.json). No human execution or recognition time was measured.

The **Milestones C and D Last-Layer extension** uses a from-scratch PyTorch policy/value model trained on solver-held-out reco.nz OLL/PLL/ZBLL actions and separately stored simulator-generated trajectories. On the untouched stage test set it completed PLL on 158/158, OLL→PLL on 158/158, and eligible ZBLL on 194/200 (97%; Wilson 95% interval 93.61–98.62%). On 100 fresh scrambles, expert-aware Cross/F2L plus neural OLL→PLL solved 93/100; the EO-aware route solved 90/100 (82.56–94.48% interval), meeting the 90% point target. Neural-guided search outperformed greedy (90/100 vs 10/100); the matched value-only search ablation solved 0/100. EO awareness did not improve over OLL→PLL on this sample: only 10/100 expert-aware F2L outcomes were ZBLL-eligible and the paired difference was not significant (exact McNemar p=0.25). Every successful row was independently replay-checked from its scramble. See the [Milestone C/D report](docs/experiment_report.md#milestones-c-and-d-neural-last-layer-and-full-cube) and [benchmark JSON](artifacts/last-layer-benchmark.json). Timing is software runtime; human recognition and execution were not measured.

## Research notes

- [Research proposal](docs/research_proposal.md)
- [Experiment report and benchmarks](docs/experiment_report.md)
- [Dataset provenance](docs/dataset_provenance.md)
- [Limitations](docs/limitations.md)
- [Portfolio summary](docs/portfolio_summary.md)

The code is MIT-licensed. The included synthetic Cross fixture has its own CC0 notice. ZBLL data retains the MIT Alg-Trainer attribution/license; ZBLS data records its CubeRoot API source and verified conversion. Reco.nz raw/processed human data and its rejection log are local-only and excluded from the repository archive.
