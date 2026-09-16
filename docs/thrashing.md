# Thrashing: what we are investigating and how

Some runs of the same solver on the same input finish in seconds and others grind until they hit their
budget. We have seen this with and without batched collapse, with and without the connectivity
constraint. Picking a batch size, a heuristic or a default without understanding *why* would be
guesswork dressed up as tuning, so this page sets out what we are trying to find out, how, and what we
have established so far.

The questions, in the order they need answering:

1. **What exactly causes a thrashing run?** Not "contradictions" — the mechanism: which choice, how far
   back the cause lies, and why recovery fails to make progress.
2. **How does it unfold?** Is it a single deep dead end, or many shallow ones? Does the search return to
   the same region repeatedly?
3. **Are there several distinct causes?** A failure that comes from a global constraint may behave
   nothing like one that comes from a tight local rule set.
4. **What raises or lowers the probability?** Rule-set structure, tile count, constraint type, grid
   size, batch size, cell-selection heuristic, weights.
5. **Can it be prevented outright**, and if not, detected early and recovered from cheaply?

## Prerequisite: reproducibility

Until now the collapse choice used an unseeded RNG (status.md A-6), so **no thrashing run could be
reproduced**. That makes every question above unanswerable: you cannot bisect a failure you cannot
re-run, and you cannot compare two configurations when the only difference you measure is luck.

Seeding the choice is therefore the first piece of work, and it changes what an experiment means: with
a fixed seed, a difference between two configurations is a real difference, and a distribution over
seeds is a real distribution rather than noise.

Propagation itself is already deterministic — it is a monotone fixpoint, so the order restrictions are
applied in does not change the result (Apt's chaotic-iteration theorem; see
[constraints.md](constraints.md)). That means seeding the choice should be sufficient for
reproducibility, and if it is not, the gap is itself a finding worth chasing.

## What we will measure per run

Wall time alone cannot distinguish "slow" from "stuck". Each run should report:

- collapses made, against cells in the grid (the redo factor),
- contradictions, backtracks, and the depth of each backjump,
- **where** contradictions happen: a histogram over cells and over grid layers,
- whether the search makes net progress over time (collapsed cells against iteration count),
- how often the search returns to a region it has already failed in — the failure mode Tessera's author
  reports as the solver "repeatedly exploring the same area over and over".

## The rule sets we need

Our current fixtures are all *adjacency* rules of varying tightness, which tests one axis and hides the
rest. To learn what structure drives thrashing we need rule sets that differ in kind, not just in size:

| Kind | Example | Why it matters |
|---|---|---|
| Adjacency only, loose | permissive 2-tile | control: should never thrash |
| Adjacency only, tight | corridors, coast | isolates constraint density from constraint kind |
| Adjacency, many variants | the 81-variant city | isolates tile count from tightness |
| Surrounding / neighbourhood | rules over a cell's whole neighbourhood, not just faces | propagation radius grows; cones interact sooner |
| Global | connectivity (have) | failures surface far from their cause |
| Counting | "at least three X or Y within 3 cells" | non-local, but bounded radius, unlike connectivity |
| Exclusion at range | "no Z within 2 cells directly above" | directional, bounded, cheap to check |
| Statistical | likelihood rises with nearby X, weighted by distance | affects *choice* rather than legality; may change failure rates without changing the solution set |

The last three do not exist yet. Counting and range-exclusion fit the existing `GlobalConstraint` trait.
The statistical one is different in kind: it biases the weighted choice rather than pruning domains, so
it belongs beside tile weights rather than in the constraint pipeline.

Not every rule set needs to be fast enough for live generation. The point of the zoo is to find which
*properties* predict thrashing, so that the city rule set — which does need to be fast — can be designed
to avoid them.

## Hypotheses to test

Stated so they can be falsified, with the reasoning that suggests them:

- **H1: a contradiction's cause is usually far from where it surfaces.** *Why:* conflict-directed
  backjumping (jumping to the most recent choice adjacent to the failure) took the constrained city from
  1-of-3 runs to 3-of-3, which is evidence that *where* you undo matters. If the cause were local,
  chronological undo would have worked.
- **H2: batching raises the contradiction rate by making choices blind to each other.** *Why:* a batch
  commits several cells before propagating, so a later choice cannot see what an earlier one implied.
  Predicts contradictions rise with batch size and with how close batched cells are.
- **H3: thrashing is a recovery failure, not a contradiction-rate problem.** *Why:* contradictions are
  normal in WFC; runs that finish have them too. Predicts that in a thrashing run the backjump target is
  wrong — the same region fails repeatedly after each recovery.
- **H4: global constraints make each contradiction dramatically more expensive.** *Why:* every violation
  forces a re-fixpoint over the whole grid, and the constrained city went 6.9 s → 305 s at batch 8 while
  batch 2 *improved* it to 0.76 s.
- **H5: tightness matters more than tile count.** *Why:* the 81-variant city is far slower than the
  2-tile permissive grid, but the permissive grid has trivially satisfiable rules; the two differ in both
  respects, so the experiment must separate them.

## Status

- Reproducibility: **not yet** — seeding is the current work.
- Diagnostics: partial (collapses, iterations, backtracks reported under `WFC_REPORT_SEARCH`).
- Rule-set zoo: adjacency and connectivity only.
- Findings: recorded in [solver-fit.md](solver-fit.md) as they are established, with the same
  facts/guesses/unknowns discipline.
