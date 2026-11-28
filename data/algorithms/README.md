# Algorithm data

`oll_cases.json` adds 57 OLL identities and 240 algorithm variants; `pll_cases.json` adds 104 PLL variants. Both were extracted from MIT-licensed `@moishy/algsets` 0.3.3, whose entries attribute OLL/PLL algorithms to SpeedCubeDB. The package license notice is retained in `LICENSE-Algsets.txt`. PLL variant-to-case assignments are replay matched against the canonical case table. See [`docs/algorithm_memory.md`](../../docs/algorithm_memory.md) for verification and recognition details.

`zbll_cases.json` contains 472 ZBLL cases plus the 21 PLL cases (493 total). `zbls_cases.json` contains 302 ZBLS cases. Each record keeps one primary algorithm and its source text.

The ZBLL/PLL list is from [Tao Yu's Alg-Trainer](https://github.com/tao-yu/Alg-Trainer), MIT-licensed commit `cb95945c3e4839ee1e4b058d4a79bc022615b220`; upstream credits ZBLL to Jabari Nuruddin. The MIT license is included as `LICENSE-Alg-Trainer.txt`.

The ZBLS set is imported from CubeRoot's [3×3 ZBLS page](https://cuberoot.me/alg/3x3/zbls) using its [public API](https://api.cuberoot.me/v1/alg/sets/3x3/zbls). `python scripts/import_cuberoot_zbls.py` reproduces the conversion and verification. The API snapshot contains 305 entries; three setups already have all four F2L slots complete, leaving 302 ZBLS cases. Case IDs use CubeRoot's stable numeric IDs because display names repeat.

The imported CubeRoot snapshot does not contain a license/reuse field. Its source and payload hash are retained for provenance, but no clear redistribution permission is recorded. `verify_algorithm_memory()` independently replays all 302 setup/algorithm pairs. ZBLS lookup currently scans and replays candidates rather than assigning an exact case identity.

The importer takes the first algorithm in each case's Front Right slot bucket. It retains the raw CubeRoot text and converts its grouped repetitions, optional bracketed moves, arrow decorations, equals-prefixed turns, and adjacent Singmaster tokens. Each normalized ZBLS algorithm is applied to its source setup in CubeMind's simulator; the setup must have a solved Cross and exactly three solved F2L slots, and the algorithm must complete F2L and orient the top edges. Import fails unless all 302 unique cases verify. Other slot orientations and source alternatives are counted but are not selected as primary algorithms.

Every ZBLL/PLL primary algorithm is inverted from solved to produce a case state, then replayed in CubeMind's simulator. All 493 case identities are unique after whole-cube orientation and AUF normalization.

Algorithm-generated cases are synthetic training/evaluation examples. They are kept separate from human reco.nz solve reconstructions.
