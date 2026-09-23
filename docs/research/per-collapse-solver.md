# The per-collapse solver

Wave Forge's first solver kept one grid on the GPU and ran the search on the CPU: for every collapse
it computed entropy, selected a cell, collapsed it and propagated, each step a dispatch followed by a
readback. It was replaced by the block kernel described in [solver.md](../architecture/solver.md) and
deleted in commit `98cc144`. This page keeps what it taught: a short timeline, the claims we made and
then refuted, and the diagnosis, fix and result of its two investigations (speed in
[#7](https://github.com/AntonTegnelov/wave_forge/issues/7), and thrashing).

The full working logs stay in git history, at the last commit that had them:

- [solver-redesign.md](https://github.com/AntonTegnelov/wave_forge/blob/4ad523491a429c1641dc0b61ea593f2ecacf4a67/docs/solver-redesign.md):
  the profile, the micro-benchmarks, the round-trip audit and the leads that became the block kernel.
- [thrashing.md](https://github.com/AntonTegnelov/wave_forge/blob/4ad523491a429c1641dc0b61ea593f2ecacf4a67/docs/thrashing.md):
  the thrashing study, seven rounds of data, and the rule-set zoo.
- [solver-fit.md at 318693e](https://github.com/AntonTegnelov/wave_forge/blob/318693ed5483cefd2ec672aefc72f3da64e2cd81/docs/solver-fit.md):
  the design analysis of CDCL, ghost cells, blocks and parallelism that preceded the block kernel.

The sources quoted here are in [literature.md](literature.md), and the measurement tables in
[measurements.md](measurements.md).

**Protocol, unless a row says otherwise.** Release builds on an RTX 3070 through dozen (Vulkan on
D3D12) in the dev container, the city rule set (81 module variants), and the CPU reference on a
Ryzen 9 5900X. The stress suite ran the 24×24×8 city (4608 cells), the 48×48×10 city and a permissive
2-tile 24³ grid. The thrashing instrument was the 12×12×6 end-to-end city (864 cells), batch 1, one
attempt per seed, with `WFC_SWEEP` capping a run at `cells × 4` iterations (3456).

## Timeline

| When | Step | Commit |
|---|---|---|
| 2026-09-16 | Baseline: city 24×24×8 in 45.3 s (102 cells/s), 48×48×10 in 403.8 s, propagation 76% of the run | `db5d8f0` |
| 2026-09-16 | Sweeping the whole grid on small worklists: measured slower on the city, reverted | `02a3c00` |
| 2026-09-16 | Warm re-measurement: the cost is per-cell work, not the dispatch | `55cf072` |
| 2026-09-16 | Union rule rows instead of testing tile pairs: 45.3 s to 37.0 s | `a20aae0` |
| 2026-09-16 | Restrict neighbours with `atomicAnd`, drop the confirmation sweep: 37.0 s to 25.9 s | `6dc14a3` |
| 2026-09-16 | Deterministic selection (one packed `atomicMin` key) and a seeded choice | `38bb61d`, `70444bd` |
| 2026-09-16 | Thrashing diagnosed; undo escalation fixed: worst of 48 seeds 197 to 11 backtracks | `2c1229c` |
| 2026-09-16 | Rule-set zoo: range exclusion, counting, surrounding and statistical rules | `c788a11` to `ddcbbc7` |
| 2026-09-17 | Round-trip audit and a CPU reference: the GPU loop about 150 times slower than one CPU thread | `dfe5890` |
| 2026-09-17 | Block kernel measured: 0.17 ms per chunk at 256 chunks, 15 times one CPU thread | `e035586` to `5a73162` |
| 2026-09-18 | The per-collapse solver deleted, along with its stress suite, traces and zoo | `98cc144` |

## Refuted claims

Kept visible because each one was believed, written down and acted on. One line each: the claim,
then what refuted it.

1. **"A dispatch costs a fixed 2.5 ms."** A cold-clock artifact: warmed up, interleaved and taken as
   medians, a trivial dispatch costs 0.099 ms (`dispatch_cost_bench`).
2. **"A pass costs the same for 1 cell or 4608, so sweep the whole grid when the worklist is small."**
   Sweeping made the 2-tile grid 15 to 20% faster and the city 17 to 27% slower; the benchmark had
   measured a freshly constrained grid, while a swept city cell unions the rows of every tile it
   still holds (about 373 000 tile visits per sweep at 81 variants against 9000 at two).
3. **"Propagation only clears bits, so racing threads converge and the atomics can go."** True of an
   atomic AND, false of the shader's load, modify and store, which lost restrictions; that race is
   why the host re-ran propagation over every cell after each collapse.
4. **"Batching causes thrashing."** Batch 1 thrashed on 2 of 5 samples of the 24×24×8 city. The whole
   first batch table was later void anyway: it was taken through the two bugs in items 5 and 6.
5. **"Seeding the choice makes a run reproducible."** Selection itself tore: the entropy shader stored
   the minimum and its cell index as two operations, so the winner depended on which workgroup
   reached the atomic first.
6. **"Widening the culprit radius makes the undo deeper."** Taking the most recent choice within a
   larger radius can only return an equally or more recent choice; across 48 seeds the deepest undo
   was 2, including the seed that backtracked 197 times.
7. **"Aiming the backjump at the true conflict cell will fix thrashing."** The conflict cell (10, 6, 4)
   sat at Chebyshev distance 1 from the collapse site (9, 5, 4); what was missing was accumulation
   across failures, not a better-placed target.
8. **"`MAX_UNDO_STEPS` (64) will be the next binding limit at larger sizes."** At 4608 cells the
   deepest undo was 12, and the constant never bounded the conflict-directed jump at all: a
   constrained run later undid 99 steps.
9. **"Deeper undos will cost wall time."** At 4608 cells, eight seeds ran in 20.2 to 23.5 s against
   23.1 to 30.2 s for the three samples that finished before the fix.
10. **"Thrashing concentrates on one cell."** True of seed 8 at 864 cells (195 of 197 failures); at
    4608 cells failures spread over 3 to 20 cells with at most 18 repeats on any one.
11. **"Escalating on a global constraint's reported cell is the same fix one branch over."** Range
    exclusion, which had finished seed 8 in 80 backtracks, stopped finishing; reverting restored it
    exactly.
12. **"What makes a rule hard to search is non-locality."** Counting and surrounding rules are just
    as non-local and cost far less than range exclusion; the variable that tracked cost was how often
    a rule declared failure (see [constraints.md](../architecture/constraints.md#what-a-rule-costs-the-search)).
13. **"Counting makes the search easier."** Worse than the control on 6 of 8 seeds; seed 8 was the
    control's worst seed and counting's best.
14. **"Range exclusion costs ten times the backtracks."** It left 6 of 8 seeds unfinished within the
    budget; seed 8 was the lucky case.
15. **"A counting rule with an identical search line is harmless."** It never fired: `prunes=0` on
    two configurations, because a radius-3 ball almost always holds more candidates than the count.
16. **"Persisting every ban is worth 331× and lands at control speed."** From one seed; over eight
    seeds backtracks ranged 15 to 437, and seed 8 got worse (80 to 97).
17. **"Restart with a cutoff can wait, the tail has collapsed."** Measured on adjacency rules only; on
    range exclusion one seed took 98.8 s and another did not finish at 12.5 times the budget.
18. **"Backjumping pays least in WFC's regime, so perfecting ours is worth little"** (our reading of
    Chen and van Beek). Fixing the escalation was worth 24 times on the bad seed (197 to 8 backtracks).
19. **"Seed 8 takes 189 s."** The timing included the release compile of `wfc-devtools`; the same seed
    on the compiled binary takes 4.4 s.
20. **"Possibility counts that rise after a restore show lost bans."** Any restore returns what
    propagation had removed since, so the first detector measured undo itself; the corrected one
    tracks named `(cell, tile)` bans.
21. **"Every block-kernel step pays a barrier across 256 invocations."** One invocation per workgroup
    is 30 times slower, so per-cell sweep work dominates the step.

## Speed: diagnosis, fix, result

**Diagnosis.** Warm, a propagation pass cost 2.271 ms for a 1-cell worklist, 3.094 ms for 4608 cells
of a fresh grid and 0.605 ms for 4608 collapsed cells (`propagation_bench`). A pass costs what its
cells cost, and one cell costs nearly as much as the grid because it is one thread doing the work
the grid spreads across thousands. The work was `compute_allowed_neighbor_mask`: for every tile still
possible it tested every tile with `check_rule`, about 81 × 81 × 6 ≈ 39 000 bit tests with a division
and a modulo each, for one uncollapsed cell.

**Fix.** Store the rule table row-aligned, one `ceil(num_tiles / 32)`-word mask per `(axis, tile)`,
and union rows: 45.3 s to 37.0 s (124 cells/s). Then replace the read-modify-write with `atomicAnd`,
which made the full-grid confirmation sweeps unnecessary: 37.0 s to 25.9 s (178 cells/s), zero
adjacency violations in the end-to-end test.

**Result, and why it was not enough.** The round-trip audit at `e28ab9d` counted 2 + 2·P blocking
queue drains per collapse (P ≈ 1.8 passes on the city) and three full grid clones. One traced
24×24×8 run (seed 1, cold device, one run) spent 16.2 s propagating, 4.3 s downloading and 3.0 s in
entropy and selection out of 25.8 s, about 7.4 ms per collapse. The CPU reference, a plain
single-threaded solver on the same rules, solved the same grid in 0.14 to 0.16 s on six of eight seeds
and an 8×8×8 chunk in 2.8 to 4.6 ms (same commit, one run per seed). The GPU loop was about 150 times slower than one CPU thread,
waiting on per-collapse synchronisation rather than computing.

The lead that replaced it was a **block-local chunk solver**: one workgroup solves a whole chunk in
workgroup memory inside one dispatch, and parallelism is spent across chunks rather than inside one
propagation (arc consistency is P-complete). The support was our own measurement that 256 dependent
steps are 13 to 18 times cheaper inside one dispatch than as separate dispatches. Measured first at
29.7 ms for one chunk and 0.91 ms per chunk at 256 chunks; with every local minimum within radius 1
collapsing per round and undo to a checkpoint, 0.17 ms per chunk at 256 chunks, about 15 times one
CPU thread in the same run, none failing (`block_solver_bench`, seed 7, 3 warm-ups, median of 5).
The rows are in [measurements.md](measurements.md).

## Thrashing: diagnosis, fix, result

Some seeds finished in seconds and others ground until their budget, with and without batching and
with and without the connectivity constraint.

**Diagnosis.** On 48 seeds of the 864-cell city, 47 finished with 0 to 4 backtracks in 4.4 to 5.7 s
and seed 8 took 197 backtracks in 7.5 s, identically on three replays. Its progress series showed a
plateau: for 190 backtracks and 184 iterations the collapsed count stayed between 529 and 534, then
the search escaped and finished within about 57 iterations. 195 of its 197 contradictions were at one
cell, (10, 6, 4), and 97 of them came from re-propagation inside the recovery itself: restore, ban a
tile, re-propagate and fail at the same cell. Undoing one or two choices never reached the cause,
for two reasons read from the code. The doubling counter was reset after every successful
propagation (the reset Prosser warns makes backjumping incomplete), and widening the culprit radius
could only make the undo shallower.

**Fix.** Undo at least as many steps as the true conflict cell has failed, keep that count across
successful propagations, and drop the radius widening (`2c1229c`).

**Result.**

| Protocol | Before | After |
|---|---|---|
| 864-cell city, seed 8 | 197 backtracks, deepest undo 2 | 8 backtracks, deepest undo 8 |
| 864-cell city, worst of 48 seeds | 197 backtracks | 11 backtracks |
| 864-cell city, corpus wall time | 4.4 to 5.7 s | 4.2 to 5.4 s |
| 24×24×8 city, batch 1 | 3 of 5 finished within 180 s; 22, 361, 514 backtracks (before the selection fix) | 8 of 8 finished; 5 to 79 backtracks, deepest undo 2 to 12 |

The fix held for adjacency rules. The rule-set zoo then held the 864-cell grid, rules, weights and
seeds 1 to 8 fixed and varied only the kind of extra rule; its lessons about rule cost are in
[constraints.md](../architecture/constraints.md#what-a-rule-costs-the-search). Two findings are about
the solver rather than the rules:

- **Bans did not survive undo.** Each history entry snapshotted the grid before its collapse, so
  undoing several steps discarded bans recorded after it. On range exclusion, seed 5, 4955 of 4962
  backtracks revived a ban, 226 314 revivals over 417 distinct bans, and one cell failed 3160 times.
  The usual argument that backtracking WFC terminates (every backtrack removes a tile) did not hold.
- **Perfect ban persistence was an upper bound worth building toward.** `WFC_PERSIST_BANS`, a
  deliberately unsound probe that re-applied every ban after each restore, took range exclusion from
  2 of 8 seeds finishing to 8 of 8 (15 to 437 backtracks, 4.0 to 23.2 s). Conditional nogood
  recording was the planned next step when the solver was deleted.

## What carried over into the block kernel

- **Gather propagation.** Each invocation writes only its own cells, so the fixpoint is race-free
  without atomics per word, the lesson of refuted claim 3.
- **Rule rows as unions.** The compiled `RuleTable` stores one mask row per `(axis, tile)`.
- **Choices that do not depend on scheduling.** Selection breaks ties by a stateless hash rather than
  by which lane wins an atomic, and every choice is `pcg3d(seed, chunk, tries, step)`.
- **Undo that accumulates.** A contradiction at or before the round of the last failure doubles the
  undo; only getting past that round starts again from one.
- **Restarts with a cutoff, and seeds in parallel.** Attempts and a step budget bound a region, and a
  repair runs 32 seeds side by side and keeps the lowest that solves.
- **Search statistics as data.** `RegionStats` reports sweeps, collapses, restarts and backtracks per
  region, so "slow" and "stuck" can be told apart.

Not carried over: global constraints, statistical rules and ban recording. The kernel has no
nogoods; what hosting the first two would take is in [constraints.md](../architecture/constraints.md).

## Left open when the solver was deleted

- Why cell (10, 6, 4) of seed 8 was unsatisfiable in the first place; the fix changed how the search
  escapes, not what it escapes from.
- Whether the run-time distribution is heavy-tailed: one seed in 48 about fifty times worse than the
  next is suggestive, but a tail needs a survival-function plot over many more seeds.
- Whether the city rule set is tightest at `z = 4`, between forced air above and buildings below. The
  sample (the top five failure cells per failing seed) was biased.
- Whether every number above holds on native Vulkan rather than through dozen.
