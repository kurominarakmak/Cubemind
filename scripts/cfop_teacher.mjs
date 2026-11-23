import { cfop } from "@moishy/cfop";
import { applyAlg, applyMoves, formatMove, isSolved, solvedCube } from "@moishy/cubing-core";

const source = "synthetic:cfop-teacher:@moishy/cfop@0.4.0";
let input = "";
for await (const chunk of process.stdin) input += chunk;
const request = JSON.parse(input);
let randomState = Number(request.seed ?? 17) >>> 0;
const faces = ["U", "R", "F", "D", "L", "B"];
const suffixes = ["", "'", "2"];

function random() {
  randomState = (1664525 * randomState + 1013904223) >>> 0;
  return randomState / 2 ** 32;
}

function makeScramble(length = 25) {
  const moves = [];
  while (moves.length < length) {
    const face = faces[Math.floor(random() * faces.length)];
    if (moves.length && moves.at(-1)[0] === face) continue;
    moves.push(face + suffixes[Math.floor(random() * suffixes.length)]);
  }
  return moves.join(" ");
}

async function solve(scramble, scrambleId = "evaluation") {
  const started = performance.now();
  const result = await cfop.solve(scramble, { colorNeutrality: "fixed", lookahead: { depth: 1 } });
  const phases = { CROSS: result.orientation, F2L: [], OLL: [], PLL: [] };
  for (const segment of result.segments) {
    if (segment.unitId === "cross") phases.CROSS.push(...segment.moves);
    else if (segment.unitId.startsWith("f2l")) phases.F2L.push(...segment.moves);
    else if (segment.unitId === "oll") phases.OLL.push(...segment.moves);
    else if (segment.unitId === "pll") phases.PLL.push(...segment.moves);
    else throw new Error(`Unexpected CFOP phase ${segment.unitId}`);
  }
  const moves = Object.values(phases).flat();
  const replay = applyMoves(applyAlg(solvedCube(), scramble), moves);
  if (!result.solved || !isSolved(result.finalState) || !isSolved(replay)) {
    throw new Error(`Reference teacher did not replay to solved for ${scrambleId}`);
  }
  return {
    scramble_id: scrambleId,
    scramble,
    sample_kind: "synthetic",
    data_source: source,
    phases: Object.fromEntries(Object.entries(phases).map(([key, values]) => [key, values.map(formatMove)])),
    reference_verified: true,
    reference_ms: performance.now() - started,
  };
}

let rows;
if (request.mode === "generate") {
  rows = [];
  for (const [split, count] of [["train", Number(request.train)], ["validation", Number(request.validation)]]) {
    if (!Number.isInteger(count) || count < 1) throw new Error("train and validation must be positive integers");
    for (let i = 0; i < count; i++) {
      const scramble = makeScramble();
      const row = await solve(scramble, `${split}-${i.toString().padStart(5, "0")}`);
      row.split = split;
      rows.push(row);
    }
  }
} else if (request.mode === "solve") {
  if (typeof request.scramble !== "string") throw new Error("scramble is required");
  rows = [await solve(request.scramble)];
} else if (request.mode === "solve_many") {
  if (!Array.isArray(request.scrambles)) throw new Error("scrambles must be an array");
  rows = [];
  for (let i = 0; i < request.scrambles.length; i++) rows.push(await solve(request.scrambles[i], `evaluation-${i}`));
} else {
  throw new Error("mode must be generate, solve, or solve_many");
}

process.stdout.write(JSON.stringify(rows));
