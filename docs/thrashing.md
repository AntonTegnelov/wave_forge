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

## First data: thrashing is not caused by batching

Five samples per batch size, 24x24x8 city, one attempt each, 180 s cap. Times are the successful runs;
"timed out" means the sample never finished inside the cap.

| Batch | Times (s) | Backtracks | Timed out |
|---|---|---|---|
| 1 | 23.1, 26.7, 30.2 | 22, 361, 514 | 2 of 5 |
| 2 | 13.6, 25.8 | 0, 751 | 3 of 5 |
| 4 | 8.1, 9.1, 9.3, 9.7, 69.3 | 15, 16, 26, 75, **4801** | 0 of 5 |
| 8 | 5.8, 6.0, 6.4, 27.1 | 33, 38, 52, **1322** | 1 of 5 |

Three things follow, and the first one overturned our working assumption:

1. **Batch 1 — the plain algorithm — thrashes too**, on two of five samples. Batching is not the cause.
   It changes the frequency and the cost, but the phenomenon is in the base solver.
2. **The failure is bimodal in search effort, not just in time.** Healthy runs finish with 15-75
   backtracks; thrashing runs show 1322 and 4801, and the timeouts are presumably worse. Wall time is a
   symptom; backtrack count is much closer to the disease.
3. **Batching makes healthy runs much faster** (batch 8 reaching 5.8-6.4 s against 23-30 s at batch 1,
   with *fewer* iterations: ~400 against ~3600 for the same ~3300 collapses). The question is not
   whether batching helps but whether thrashing can be removed, after which batching is nearly free.

Incidental but useful: every run collapses about 3300 of 4608 cells, so roughly 1300 cells are decided
by propagation rather than by choice — consistent with the published finding that propagation, not the
entropy heuristic, does WFC's real work.

This is evidence for **H3** (thrashing is a recovery failure) over a simple contradiction-rate story: if
contradictions alone drove it, the distribution would be smooth rather than split into 15-75 and
1322-4801.

## Experimental setup

The 24x24x8 stress city takes 6-30 s per sample and thrashing samples hit their cap, which makes it a
poor instrument for a study that needs hundreds of runs. The **12x12x6 E2E city (864 cells)** asks the
same question at roughly a twentieth of the cost, exhibits the same bimodality, and already reports the
search statistics. It is the primary instrument; the stress city confirms findings rather than
discovering them.

Two environment switches keep experiments honest:

- `WFC_SEED` fixes the collapse choice, so a comparison between configurations is a real comparison and
  a failure can be replayed.
- `WFC_SWEEP` drops the harness to one attempt with a tight iteration budget, so a configuration that
  thrashes reports quickly instead of retrying twenty times. (The first sweep without it spent forty
  minutes in silence before reporting a single number.)

`WFC_REPORT_SEARCH` prints, per run: seed, collapses, cells, iterations, backtracks, how many distinct
cells contradictions surfaced at, how many of those failed more than once, and the deepest and mean
undo. The repeated-failure count is the direct test of H3.

## Status

- Reproducibility: **not yet** — seeding is the current work.
- Diagnostics: partial (collapses, iterations, backtracks reported under `WFC_REPORT_SEARCH`).
- Rule-set zoo: adjacency and connectivity only.
- Findings: recorded in [solver-fit.md](solver-fit.md) as they are established, with the same
  facts/guesses/unknowns discipline.
