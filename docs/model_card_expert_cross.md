# Model card: human Cross action policy

## Intended use

Predict the next outer-face Cross move from a 3×3 facelet state after rotating the demonstrated Cross color to Down. The policy is one learned action component inside a bounded Cross/F2L/last-layer planner; it is not an inspection-intent model or a complete solver by itself.

## Training

- Source: user-authorized local reco.nz 3×3 CFOP reconstructions.
- Labels: explicit `CROSS` segments only. Cross+1 and X-Cross annotations are recorded separately and are not converted into Cross actions.
- Split: existing solver-disjoint train/validation/test partitions; no solver appears in multiple partitions.
- Input: 54 one-hot facelets with centers canonicalized after orienting the demonstrated Cross color Down.
- Output: 18 outer-face turns; slice, wide, rotation, post-Cross, and invalid actions are excluded.
- Initialization: random PyTorch initialization. Three seeds (17, 37, 73); seed 37 selected using validation loss.
- Size: 118,418 parameters.

## Held-out action results

On 3,996 test actions from held-out solvers: 47.37% Top-1, 71.67% Top-3, cross-entropy 1.744. The majority-action baseline is 12.09% Top-1 and 27.08% Top-3. This is next-action agreement, not solve success, expert approval, or human likeness.

## Limits

The checkpoint does not choose a Cross color, predict X-Cross, infer inspection intent, represent pseudo-slot intent, or condition on a cuber's white-only/dual-color/color-neutral capability. The planner can evaluate six color frames, but that compute is not evidence that a human inspected six colors. Execution speed, pauses, recognition time, and TPS were not measured.

Weights and training data remain local. No Hugging Face or GitHub publication was performed.
