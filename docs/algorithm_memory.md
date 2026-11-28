# Verified algorithm memory

The V2.1 symbolic path retrieves known procedures; it does not train a neural model to rediscover them. Every returned sequence is replayed in CubeMind's simulator before selection.

| Family | Case coverage | Stored variants | Recognition/retrieval |
| --- | ---: | ---: | --- |
| OLL | 57 | 240 | Exact orientation pattern, AUF and whole-cube normalized; candidate algorithms are replay checked. |
| PLL | 21 | 104 SpeedCubeDB variants plus 21 Alg-Trainer primaries | Exact ZBLL-table match when available; otherwise match by replay against compatible PLL algorithms and AUF/yaw transforms. |
| ZBLL | 472 | 472 primaries | Existing exact facelet recognizer; algorithms replay verified. |
| ZBLL table's PLL identities | 21 | 21 primaries | Existing exact facelet recognizer; algorithms replay verified. |
| ZBLS | 302 | 302 primaries | Scans the imported set and accepts only Cross + three-pair → F2L + EO replay transitions. There is no exact ZBLS case recognizer yet. |

OLL and PLL alternatives were imported from `@moishy/algsets` 0.3.3, which labels them as SpeedCubeDB data and carries an MIT license. The package notice is in `data/algorithms/LICENSE-Algsets.txt`; canonical PLL identities are assigned only after replay against the existing 21-case table. The 493 ZBLL/PLL primaries retain the Alg-Trainer MIT attribution and license. ZBLS retains its CubeRoot API snapshot and provenance, but the imported snapshot does not record reuse terms; no redistribution claim is made for that source.

`algorithm_records()` returns case/algorithm IDs, moves, source, license status, preconditions, postconditions, move metrics, and an explicit `observed_human_usage: None`. The optional 8,033-parameter OLL/PLL preference ranker was trained from random initialization on 2,169 human choices whose exact table candidate was replay-verified: 1,515 train, 485 validation, and 169 solver-held-out test choices. On the test set it reached 72.19% Top-1 and 94.67% Top-3. It beat the shortest-variant baseline (36.69% Top-1), but not the most-frequent training choice (75.15%; paired exact McNemar p=0.227). This is evidence for a useful signal beyond shortest length, not a demonstrated gain over frequency. It covers only cases represented in training and does not model ZBLL/ZBLS preferences, execution ergonomics, or recognition difficulty. Human execution and recognition time were not measured.

Run the replay audit with:

```bash
python - <<'PY'
from cubemind.algorithm_memory import verify_algorithm_memory
print(verify_algorithm_memory())
PY
```
