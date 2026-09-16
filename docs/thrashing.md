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

The last three do not exist yet. Surveying the code settled where each one has to live:

**Surrounding, counting and range-exclusion fit `GlobalConstraint` unchanged.** The trait hands `apply`
the whole `PossibilityGrid` mutably and asks for the cells it narrowed, so any neighbourhood predicate
is expressible. Soundness, not expressiveness, is the constraint: a bit may only be cleared when *no*
completion of the current possibility sets could make that tile legal. `ConnectivityConstraint` shows
the conservative pattern — reason over what could still be true, and return `Ok(vec![])` when the grid
is not yet decided enough to rule anything out.

**The statistical rule does not fit, and it is worth being exact about why.** `apply` can only remove
possibilities. A likelihood bias removes none, so the only faithful implementation of it as a
constraint is a no-op; making it prune instead would change the solution set, which is precisely what a
rule that shifts *probability* must not do. It also cannot reach the place it would need to: the
collapse choice indexes `tile_weights[tile]` by tile alone, even though the grid and the cell
coordinates are in scope a few lines above. So it needs a choice-time hook — a weight function that
sees the cell and its neighbourhood — rather than a slot in the constraint pipeline.

One hazard applies to every constraint we add. The solver re-runs `apply` until it reports no changes,
and that inner loop has no iteration cap of its own; it does not tick the outer `iterations` counter
either. A constraint that returns a cell it did not actually narrow therefore spins forever, and
`max_iterations` will not rescue the run. Only ever clear bits, and only report a cell when a bit
really was cleared.

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

## Reproducibility: achieved, and it was not only the RNG

Seeding the collapse choice turned out to be necessary but not sufficient. Cell *selection* was
nondeterministic too, below the level the RNG could reach: the entropy shader compare-exchanged the
minimum entropy into one word and then stored the winning index into another as a separate operation,
so the pair could tear, and the tie-break afterwards compared against a stale local copy of the global
minimum. Which cell got selected depended on which workgroup reached the atomic first.

This is a useful reminder of how the earlier measurements went wrong. Seeding alone would have produced
runs that still varied, and the natural conclusion would have been that some deeper nondeterminism was
inherent to GPU solving — when in fact one shader reduction was written incorrectly.

Reducing over a single packed key instead — quantised entropy in the high bits, cell index in the low
bits, one `atomicMin` — makes the winner independent of workgroup order, and orders ties by lowest
index so they break identically every time. Verified on the 864-cell city: seed 12345 gives
`collapses=638 iterations=638 backtracks=1` on three consecutive runs, and seed 999 diverges. That is
one grid and one rule set, so it is evidence rather than proof that the solver is now a pure function
of its seed; the zoo will test it more widely.

**Every batch measurement taken before this point is void**, including the table above. All of it was
gathered through torn selection *and* through a backjump that never escalated, so the numbers describe
two bugs interacting, not batch sizes. They are kept only as a record of what prompted the
investigation.

## Second data: the shape of the tail, measured on 48 seeds

48 seeds on the 864-cell city (12x12x6), batch 1, one attempt each, with the deterministic selection
and the seeded choice in place.

| Seeds | Backtracks | Wall time |
|---|---|---|
| 47 of 48 | 0-4 | 4.4-5.7 s |
| seed 8 | **197** | 7.5 s |

Four things this establishes, and one it does not.

1. **The pathology is exactly reproducible.** Seed 8 replayed three times gives an identical line every
   time: `collapses=591 iterations=785 backtracks=197`, with 98 of the contradictions at the single
   cell (9, 5, 4). Determinism therefore covers the failing path and not merely the healthy one, which
   is what makes the rest of this investigation possible at all.
2. **The distribution is extremely skewed rather than merely noisy.** One seed in 48 is roughly fifty
   times worse than the next worst. This is the shape that makes averages useless here: a mean over
   these 48 runs describes no run that actually happened.
3. **It is one cell, not one region.** 98 of seed 8's 197 backtracks surface at the same coordinate.
   The search is not wandering through a hard neighbourhood; it keeps rebuilding one doomed state.
4. **The escalation never engages.** `max_undo` is at most 2 across all 48 seeds, including the one
   that backtracks 197 times. Two independent causes, both confirmed by reading the code rather than
   inferred from the numbers:
   - the doubling counter is reset unconditionally after every successful propagation, so it cannot
     accumulate across repeated failures separated by a little progress;
   - widening the culprit radius is **backwards**. The search takes the *most recent* choice within the
     radius, so a larger radius can only return an equally recent or more recent choice. Widening makes
     the undo shallower, never deeper, and so cannot reach a cause that lies further back. This was our
     own change, and the measurement is what exposed it.

What it does not establish: **why (9, 5, 4) is unsatisfiable in the first place.** Knowing the search
fails to escape does not tell us what it is failing to escape from. That is the next question.

A weak observation, recorded so it can be tested rather than trusted: of the failure cells reported
across these seeds, the layer `z = 4` appears most often and `z = 5` never. `z = 5` is forced to air and
`z = 0` to street level by `constrain_city`, which would make `z = 4` the layer squeezed between forced
air above and whatever the buildings did below. This is a biased sample — only the top five cells per
seed are reported, and only for seeds that failed at all — so it is a hypothesis about where the rule
set is tightest, not a finding.

## Third data: thrashing is a plateau, not slow progress

Totals cannot separate a slow run from a stuck one, so the solver now reports the series of
`(iteration, collapsed cells, backtracks)` at each backtrack. Seed 8, sampled every fifth backtrack:

| Backtrack | Iteration | Collapsed |
|---|---|---|
| 1 | 533 | 534 |
| 51 | 581 | 533 |
| 101 | 630 | 532 |
| 151 | 680 | 532 |
| 191 | 717 | 529 |
| 196 | 774 | **581** |

**For 190 backtracks and 184 iterations the collapsed count never leaves the band 529-534.** Net
progress over that stretch is zero — very slightly negative, in fact. Then the search escapes and
finishes 591 cells' worth of work within about 57 further iterations.

So "thrashing" is not a slowdown. It is a plateau: the search performs work at a normal rate while
accomplishing nothing, and essentially all the wall-clock cost of a bad run is spent there. Question 2
of this document — is it one deep dead end or many shallow ones — has a clear answer for this seed:
**many shallow ones, all the same one.**

The mechanism follows from the code, and every step of it is confirmed rather than inferred:

1. A contradiction surfaces at (9, 5, 4).
2. Recovery undoes one choice — `max_undo` is 2 at most, mean 1.5 — restoring a state that still
   contains whatever actually causes the contradiction.
3. The restored choice's tile is banned, a cell is re-collapsed, propagation runs, and the identical
   contradiction surfaces again.
4. Repeat, 98 times at that one cell.

The escape appears to be accidental rather than directed. Each backtrack bans one tile at one cell, so
after enough failures the neighbouring domains are drained far enough that a structurally different
completion is finally forced. That is brute-force enumeration at depth one wearing the costume of
backjumping — which is why the escape, when it comes, is abrupt.

Seed 28 is the same trap, survived: three backtracks, all at iteration 532, with collapsed pinned at
533 throughout. It enters the plateau and leaves almost immediately.

One observation held deliberately loose: both seeds stall at roughly 530-534 collapsed cells of 864,
around 62% of the grid. Two samples is not a distribution, and the healthy seeds never contradict at
all, so this is a thread to pull — is there a characteristic density at which this rule set becomes
hard? — rather than a result.

## What the literature says, and what it says about our implementation

Thrashing is the CSP literature's own word for this, and it is old. Mackworth (1977) defines it by
cause, and one of his causes is our symptom exactly: a failure rediscovered over and over, which he
says "cannot be removed by such minor palliatives as reordering the nodes." Dechter and Frost put it
as "rediscovering the same inconsistencies and same partial successes during search."

**Our backjump target rule is right in principle.** Dechter and Frost's Proposition 4: "the latest
variable in its jumpback set is the earliest variable to which it is safe to jump." Jumping *later*
than that re-fails; jumping *earlier* can skip solutions. Prosser and Ginsberg pick the same target,
and CDCL's assertive level is the same idea.

**Our conflict set is not a conflict set.** We approximate it by spatial radius. For a grid whose
constraints are local adjacency that is a defensible proxy for the constraint graph — it is essentially
graph-based backjumping — but propagation destroys the bound. Arc consistency runs to fixpoint, so a
domain wipeout at C can be caused by a collapse arbitrarily far away, transmitted through a chain of
prunings. Distance says where the damage is; it says nothing about when the responsible choice was
made. A real conflict set is accumulated from *which constraint eliminated which value*, not from
proximity.

**The mechanism we are missing is accumulation.** CBJ merges conflict sets as it jumps
(`conf-set[h] <- conf-set[h] union conf-set[i] - {h}`), so depth comes from repeated merged jumps, not
from one wide jump. This resolves what looked like a paradox in our own measurement: taking the most
recent member of a *larger* set can only give a *shallower* jump, and that is true of real CBJ too. It
is harmless there because the sets accumulate. Ours are recomputed from scratch each failure, so the
search cannot climb.

**Prosser describes our bug in a sentence.** Writing about the accumulated set in graph-based
backjumping: *"if `P` was dispensed with, or was reset whenever a successful forward move was made, we
would again have an incomplete algorithm."* Our doubling counter is reset unconditionally after every
successful propagation. Same reset, same consequence.

**Our failure is under-jumping, which is the benign direction.** Under-jumping does not lose solutions;
it re-derives the same dead end. Over-jumping is the one that silently skips solutions. So the 98
repeats are the predicted signature of a target that is too shallow, and not a soundness problem.

### A termination hazard we should check rather than assume

Backtracking WFC is usually argued to terminate because every backtrack removes at least one tile from
some domain, so the search cannot cycle. That argument needs the removals to survive. In our undo path
each history entry snapshots the grid *before* its collapse, and recovery restores a snapshot and then
bans the tile of the restored choice. Undoing several steps therefore restores a snapshot that predates
bans recorded after it, and those bans are discarded with it. At `max_undo` of 1-2 this rarely bites,
but it means the monotone-progress argument does not hold in general, and a genuine cycle is possible
rather than merely slow. This needs a test, not a reassurance.

### What the evidence favours, in order

1. **Restart with a cutoff.** Our distribution is the textbook case: 47 of 48 seeds finish in 0-4
   backtracks. With a per-attempt success probability that high, a short cutoff plus retry has an
   expected cost close to a trivial run, while the bad seed currently pays 197 backtracks. Luby,
   Sinclair and Zuckerman give the theory; Gomes, Selman and Kautz show randomised restarts eliminate
   the heavy right tail and exploit a heavy left tail. Boris the Brave's Tessera ships exactly this as
   a step limit that "will allow some backtracking to occur, but after a fixed amount of computation it
   will automatically retry with a fresh generation." Naive fixed-cutoff restart is incomplete; the
   established fixes are an increasing cutoff or retaining nogoods across restarts (Lecoutre et al.).
2. **Failure-weighted cell selection (dom/wdeg).** Boussemart et al. weight constraints by how often
   they caused a dead end and prefer variables involved in them, explicitly to avoid thrashing. Applied
   here: a cell that has failed repeatedly should be collapsed *earlier*, not left until the frontier
   reaches it. We already record `failures_by_cell`, so the input exists.
3. **Escalation that accumulates with failure count.** Punch Out Model Synthesis raises its erosion
   probability with the number of failed attempts, and states the reason plainly: "Without the erosion,
   block level solvers could perpetually attempt resolution on blocks with identical initial state."
   That is our 98 failures at one cell, and it is the same correction as Prosser's non-reset.
4. **True conflict cells instead of the collapse site.** The shader already records which cell's domain
   emptied and the error already carries it; the search discards it. This is cheap to fix and is a
   prerequisite for any real conflict set.

Two cautions from the same reading. **Chen and van Beek** prove backjumping's value shrinks as
maintained local consistency and fail-first ordering strengthen — WFC maintains arc consistency to
fixpoint with a min-entropy heuristic, the regime where CBJ pays least, so perfecting our backjump has
a lower expected return than its prominence here suggests. And **Baker's "Hazards of Fancy
Backtracking"** finds dynamic backtracking performs badly when combined with dynamic variable ordering,
which min-entropy is; that rules out one of the obvious upgrades.

For context on what others do: Gumin's original WFC does not backtrack at all and restarts globally.
marian42 reports the *opposite* failure to ours — "errors are recognized very late which leads to many
steps being backtracked" — deep undos where ours are too shallow. DeBroglie documents backtracking as
complete but slow and memory-hungry, "generally only appropriate for generating small arrays."

## Status

- Reproducibility: **yes** — seeded choice plus a deterministic selection reduction, verified on the
  864-cell city, and confirmed on the pathological seed as well as healthy ones.
- Diagnostics: partial (collapses, iterations, backtracks reported under `WFC_REPORT_SEARCH`).
- Rule-set zoo: adjacency and connectivity only.
- Findings: recorded in [solver-fit.md](solver-fit.md) as they are established, with the same
  facts/guesses/unknowns discipline.
