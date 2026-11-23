import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { cfop } from "@moishy/cfop";
import { applyAlg, applyMoves, formatMove, isSolved, normalizeOrientation, solvedCube } from "@moishy/cubing-core";

const root = fileURLToPath(new URL("..", import.meta.url));
const outputPath = join(root, "artifacts", "cfop-policy-benchmark.json");
const seed = 20261116;
const solveCount = Number(process.env.CUBEMIND_SOLVES || 20);
const faces = ["U", "R", "F", "D", "L", "B"];
const suffixes = ["", "'", "2"];
let randomState = seed >>> 0;

function random() {
  randomState = (1664525 * randomState + 1013904223) >>> 0;
  return randomState / 2 ** 32;
}

function scramble(length = 25) {
  const moves = [];
  while (moves.length < length) {
    const face = faces[Math.floor(random() * faces.length)];
    if (moves.length && moves.at(-1)[0] === face) continue;
    moves.push(face + suffixes[Math.floor(random() * suffixes.length)]);
  }
  return moves.join(" ");
}

function stmCount(moves) {
  return moves.filter((move) => !["x", "y", "z"].includes(move.family)).length;
}

function edgesOriented(state) {
  return normalizeOrientation(state).eo.every((orientation) => orientation === 0);
}

function inspect(result) {
  let state = applyAlg(solvedCube(), result.scramble);
  state = applyMoves(state, result.orientation);
  let eoAfterF2l = null;
  let eoAfterZbls = null;
  for (const segment of result.segments) {
    state = applyMoves(state, segment.moves);
    if (segment.unitId === "f2l") eoAfterF2l = edgesOriented(state);
    if (segment.unitId === "zbls") eoAfterZbls = edgesOriented(state);
  }
  return {
    solved: result.solved && isSolved(result.finalState),
    eoAfterF2l,
    eoAfterZbls,
    usedZbls: result.segments.some((segment) => segment.unitId === "zbls"),
    usedZbll: result.segments.some((segment) => segment.unitId === "zbll"),
    stm: stmCount(result.solution),
    solution: result.solution.map(formatMove).join(" "),
    selection: result.segments.some((segment) => segment.unitId === "zbls") ? "ZBLS → ZBLL" :
      result.segments.some((segment) => segment.unitId === "zbll") ? "F2L → ZBLL" : "F2L → OLL → PLL",
  };
}

async function runPolicy(scrambles, policy) {
  const rows = [];
  for (const input of scrambles) {
    const started = performance.now();
    try {
      const settings = { colorNeutrality: "fixed", lookahead: { depth: 1 } };
      let result;
      let fallback = false;
      if (policy === "conventional_cfop") {
        result = await cfop.solve(input, settings);
      } else {
        try {
          result = await cfop.solve(input, {
            ...settings,
            replacements: { zbls: { enabled: true, mode: "force" } },
            extras: { zbll: { enabled: true, mode: "force" } },
          });
          if (!result.solved || !result.segments.some((segment) => segment.unitId === "zbll")) {
            result = null;
          }
        } catch {
          result = null;
        }
        if (!result) {
          fallback = true;
          result = await cfop.solve(input, settings);
          const baseline = inspect(result);
          if (baseline.eoAfterF2l) {
            try {
              const zbllResult = await cfop.solve(input, {
                ...settings, extras: { zbll: { enabled: true, mode: "force" } },
              });
              if (zbllResult.solved && zbllResult.segments.some((segment) => segment.unitId === "zbll")) {
                result = zbllResult;
              }
            } catch {
              // If the optional case is uncovered, retain the verified OLL→PLL fallback.
            }
          }
        }
      }
      rows.push({ scramble: input, ...inspect(result), fallback_to_conventional_f2l: fallback,
        computational_ms: performance.now() - started });
    } catch (error) {
      rows.push({ scramble: input, solved: false, error: String(error), computational_ms: performance.now() - started });
    }
  }
  const solved = rows.filter((row) => row.solved);
  const f2lEligible = rows.filter((row) => row.eoAfterF2l === true);
  const zblsEligible = rows.filter((row) => row.eoAfterZbls === true);
  const transitionEligible = rows.filter((row) => policy === "eo_aware"
    ? row.eoAfterZbls === true || row.eoAfterF2l === true
    : row.eoAfterF2l === true);
  return {
    policy,
    solves: rows.length,
    successful_solves: solved.length,
    successful_solve_rate: solved.length / rows.length,
    zbll_eligible_after_f2l: f2lEligible.length,
    zbll_eligible_after_zbls: zblsEligible.length,
    zbll_eligible_at_policy_transition: transitionEligible.length,
    zbll_eligibility_rate: transitionEligible.length / rows.length,
    zbls_used: rows.filter((row) => row.usedZbls).length,
    zbll_used: rows.filter((row) => row.usedZbll).length,
    eo_aware_fallbacks: rows.filter((row) => row.fallback_to_conventional_f2l).length,
    mean_stm_successful: solved.length ? solved.reduce((n, row) => n + row.stm, 0) / solved.length : null,
    median_stm_successful: solved.length ? [...solved].sort((a, b) => a.stm - b.stm)[Math.floor(solved.length / 2)].stm : null,
    mean_computational_ms: rows.reduce((n, row) => n + row.computational_ms, 0) / rows.length,
    human_recognition_or_execution_time: "not measured",
    rows,
  };
}

const scrambles = Array.from({ length: solveCount }, () => scramble());
const conventional = await runPolicy(scrambles, "conventional_cfop");
const eoAware = await runPolicy(scrambles, "eo_aware");
const report = {
  benchmark: "deterministic 3x3 CFOP policy comparison",
  seed,
  scramble_generator: "25 random outer turns; fixed white/Down orientation for both policies",
  solver: "@moishy/cfop 0.4.0; deterministic external CFOP algorithm baseline",
  runtime: { node: process.version, platform: process.platform, architecture: process.arch },
  neural_training_data: "none; expert reconstructions and synthetic algorithm cases are evaluated separately",
  move_count: "STM; whole-cube rotations excluded; wide and slice moves count as one",
  conventional,
  eo_aware: eoAware,
};
await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({
  output: outputPath,
  conventional: Object.fromEntries(Object.entries(conventional).filter(([key]) => key !== "rows")),
  eo_aware: Object.fromEntries(Object.entries(eoAware).filter(([key]) => key !== "rows")),
}, null, 2));
