# Raw solve data

This folder is ignored by Git. Local reco.nz 3×3 reconstructions were collected for this project after the user authorized community research training. Raw records and processed human action rows are local-only and are not part of the checked-in deliverable.

The optional importer accepts one JSON object per line with these fields:

```json
{
  "event": "3x3",
  "solver_id": "stable pseudonymous identifier",
  "solve_id": "unique solve identifier",
  "scramble_id": "stable scramble-group identifier",
  "scramble": "R U F' D2 ...",
  "solution_moves": "... complete solving sequence ...",
  "solution_scope": "FULL_SOLVE",
  "cross_face": "D",
  "cross_color": "white",
  "solver_cross_color_range": "white_only",
  "data_source": "URL or archive identifier",
  "reuse_permission": "license or explicit permission record",
  "reuse_permission_verified": true
}
```

The legacy Cross importer accepts only `event: 3x3` and requires a normalized white Cross on D, in CubeMind's fixed frame (U yellow, R red, F green, D white, L orange, B blue). The default reco.nz pipeline filters the archive to 3×3 CFOP, excludes OH, replays inspection rotations, and retains the selected Cross color per solve. Its action rows feed the CFOP model. The `--all-3x3-methods` option also gathers other solving methods in the 3×3 event, but those records are kept outside the CFOP action-model splits. Solver color-neutrality remains `unknown` unless independently established; one reconstruction does not reveal a cuber's full color range.

The full solution must replay from the stated scramble to solved, allowing a final whole-cube orientation. The reco.nz collector preserves source URLs, solver/reconstructor names, phase annotations, and solve IDs; unsupported notation and failed replays are rejected. Processed samples preserve source, solver, solve, scramble, phase, Cross color, edge orientation, and move-count context. Both raw reconstructions and processed human samples are ignored by Git.

Collect a filtered local sample and prepare solver-held-out Cross/F2L samples with:

```bash
python -m cubemind.data.reco
python -m cubemind.train_f2l --epochs 20
```

The default collector follows all pages for the 3×3 CFOP filter, retries transient network errors, resumes accepted rows from local raw/checkpoint files, and writes rejected solve IDs/reasons to `reco_nz/cfop_3x3_rejected.jsonl`. Use `python -m cubemind.data.reco --all-3x3-methods` to collect the full 3×3 event across methods into `reco_nz/all_3x3.jsonl`; these extra method records remain separate and are not added to the CFOP action-model splits. Optional `--pages` and `--max-solves` flags support smaller test runs.
