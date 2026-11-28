# Dataset provenance

**Updated:** 2026-11-28

## Human solve reconstructions

| Field | Value |
| --- | --- |
| Source | [reco.nz](https://reco.nz/), linked from the [speedcubing.com reconstruction hub](https://www.speedcubing.com/Reconstructions.html) |
| User direction | The user authorized local community-research training on reco.nz reconstructions. |
| Filter | All current index pages, unique 3×3 CFOP records only; OH and other methods/events are excluded. |
| Index and replay | 9,288 unique filtered entries; 9,109 complete reconstructions replayed successfully. The remaining 179 entries were retried after adding compact/grouped notation support. |
| Rejections | 13 unsupported notation, 165 full-sequence replay failures, and 1 page with no move list. IDs/reasons are stored locally in `data/raw/reco_nz/cfop_3x3_rejected.jsonl`. |
| Labels | 324,097 state/action rows: 39,164 Cross, 30,619 Cross+1, 253,051 F2L, and 1,263 explicitly annotated ZBLS actions. |
| Split | By solver name: 6,654 train solves / 308 solvers, 1,407 validation / 39, and 1,048 test / 39. No solver crosses a split. |
| Storage | Raw records at `data/raw/reco_nz/` and processed rows at `data/processed/reco_cfop/`; both paths are ignored by Git. |

A separate scan removed the method filter while retaining the 3×3 puzzle filter. It found 10,474 index entries and 10,274 simulator-valid reconstructions (CFOP 9,129; ZB 557; Roux 515; ZZ 45; Petrus 26; CF 2). This is still not the entire site-wide count of roughly 14,000: OH and other events are excluded. The broad scan and the CFOP-only scan returned non-identical CFOP solve IDs (120 found only in the broad scan and 100 found only in the CFOP-only scan), so they remain separate data snapshots; the published human-model metrics use only the explicit CFOP-filtered corpus above.

The raw records preserve the reco.nz solve URL, solver and reconstructor names, scramble, source date/result/move count, phase comments, inspection rotations, solve ID, and scramble ID. Each full sequence is replayed from the stated scramble and must finish solved up to whole-cube orientation. The selected Cross color is recorded per solve. Solver-wide color neutrality is set to unknown unless separately verified. The training run uses only train-split actions for gradient updates; validation selects checkpoints and the solver-held-out test split measures action agreement.

This use is local to the user's authorized research project; the repository does not redistribute raw or processed reconstructions. The model's exact action agreement is not evidence that it inferred an unannotated decision or intention.

## Algorithm data

| File | Cases | Source and validation |
| --- | ---: | --- |
| `data/algorithms/zbll_cases.json` | 472 ZBLL + 21 PLL = 493 | [Tao Yu Alg-Trainer](https://github.com/tao-yu/Alg-Trainer), MIT commit `cb95945c3e4839ee1e4b058d4a79bc022615b220`; upstream credits ZBLL to Jabari Nuruddin. |
| `data/algorithms/zbls_cases.json` | 302 | [CubeRoot 3×3 ZBLS page](https://cuberoot.me/alg/3x3/zbls), imported from the [public API](https://api.cuberoot.me/v1/alg/sets/3x3/zbls). |

`data/algorithms/LICENSE-Alg-Trainer.txt` carries the upstream MIT notice for ZBLL/PLL. All 493 ZBLL/PLL case identities are unique under AUF and whole-cube orientation normalization; each primary algorithm is replayed from its inverse-generated state. The CubeRoot API snapshot reports 305 entries scraped at `2026-05-04T00:13:11Z`; three entries already have all four F2L slots complete, leaving 302 ZBLS cases. The checked-in import stores one first-slot primary per case, its original notation, a payload hash, and source variant counts. Each source setup has a solved Cross and exactly three solved F2L slots; replay verifies complete F2L and oriented upper-layer edges. Recreate it with `python scripts/import_cuberoot_zbls.py`.

Human reconstruction rows are not mixed into the algorithm-generated training set.

## V2 ZBLS model data

The V2 ZBLS policy uses three source-separated configurations. The synthetic dataset contains simulator-generated, replay-verified trajectories derived from the 302 imported CubeRoot cases, with 7,248 presentations split by canonical case identity (211 train / 45 validation / 46 test). The human dataset contains only explicitly annotated, replay-verified reco.nz ZBLS phase segments and inherits the existing solver-disjoint partitions (49 train trajectories / 576 actions, 16 validation / 144, and 9 test / 97). The mixed policy combines training rows from the two sources but keeps their validation and test results separate. No human intent is inferred from an unannotated solve phase, and raw reconstruction rows are not copied into V2 benchmark artifacts.

The V2 ZBLS training run used random initialization and did not reuse pretrained encoders. The synthetic-only, human-only, and mixed checkpoints plus source counts, split identities, configuration, and hashes are local at `artifacts/v2/`. They have not been uploaded as public releases; local research use and public checkpoint redistribution are separate questions.

## Other baselines

The paired full-CFOP benchmark uses [`@moishy/cfop` 0.4.0](https://www.npmjs.com/package/@moishy/cfop), an MIT-licensed deterministic solver, for both policies. The package includes its own CFOP/ZBLS/ZBLL algorithm sets; these are used only inside the solver benchmark and are not used to train CubeMind's neural models. The ZBLL classifier and its held-out test cases use the Alg-Trainer data; the local ZBLS table comes from CubeRoot.

## V2.1 expert Cross labels

The Cross policy uses only explicit `CROSS` actions from the already processed reco.nz CFOP corpus, with its existing solver-disjoint train/validation/test assignments. It excludes post-Cross actions and non-outer-turn labels. The saved manifest records 25,596/4,901/3,996 usable Cross actions across train/validation/test, with 21,703/5,856/3,060 Cross+1 annotations and 17,672/4,053/2,659 X-Cross annotations retained as separate outcome counts. The annotations do not reveal inspection intent or a solver's permitted color profile. Of 5,738 explicit OLL and 6,033 PLL segments, 2,169 choices matched a unique table variant and passed candidate-specific simulator replay (1,515 train, 485 validation, 169 solver-held-out test). Unmatched, ambiguous, and non-replaying segments were excluded. Raw reco.nz rows remain local and Git-ignored.

## Original synthetic Cross corpus

`data/fixture/processed/` is generated by CubeMind's deterministic shortest white-Down-Cross BFS oracle. Rows use `sample_kind: synthetic`, `solver_id: synthetic-bfs-oracle`, `cross_face: D`, `cross_color: white`, and `data_source: synthetic:shortest-cross-bfs-v1`. The fixture remains separate from human and algorithm data and includes its own CC0 notice.
