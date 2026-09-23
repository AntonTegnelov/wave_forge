# Measurements

Every number the project has measured, each with the protocol that produced it. Design docs link
here instead of copying numbers, and [status.md](../plan/status.md) quotes the headline figures.
The solver's design is in [solver.md](../architecture/solver.md), the world's schedule and repairs
in [world.md](../architecture/world.md), and how to measure in
[performance.md](../guides/performance.md). Numbers others measured, read at their sources, are in
[literature.md](literature.md); this page holds only the project's own.

A number describes one build on one machine and driver stack. It becomes a design conclusion only
after it has been repeated with a warm-up and medians over interleaved samples. The most expensive
mistake so far came from skipping that: a cold first measurement said a dispatch costs about 2.5 ms,
and it steered the redesign in the wrong direction for a day (P14 to P21).

## How to read a row

Each era starts with its standing protocol, and a row names only what differs from it. The fields:

- **Device.** "dozen" is the dev container's NVIDIA GeForce RTX 3070 reached through Mesa's dozen
  driver (Vulkan on Direct3D 12, under WSL). Translation distorts dispatch and submission costs, so
  a dozen timing is a property of this stack until it is repeated on native Vulkan.
  "lavapipe" is Mesa's CPU implementation of Vulkan, the device CI runs on; its timings are not
  performance data. CPU timings were taken on an AMD Ryzen 9 5900X (12 cores, 24 threads). How the
  container reaches the GPU is in [environment.md](../guides/environment.md).
- **Build.** Release unless a row says otherwise. The profiles are in
  [performance.md](../guides/performance.md).
- **Rule set.** "The city" is the 81-variant marian42-style module set, three possibility words per
  cell: code in `wfc_devtools::city` until [#52], `examples/city.ron` since. Other rule sets are
  named where they are used.
- **Chunk and halo.** A chunk is 8×8×8 cells of 2 m unless stated. The halo is in cells.
- **Seeds.** The world seed, or the seeds of the solves compared.
- **Source.** The test or bench and its command. Where the code no longer exists, the commit or
  pull request where it can be read.
- **Struck rows** (~~like this~~) were superseded by a later measurement of the same thing, or were
  void. They stay, with a pointer to what replaced them, because the change between two rows is
  evidence too.

Rows carry an ID (P, K, L or E, by era) so that other rows and docs can point at them.

## Eras

| Era | Dates (2026) | What it measured | Code |
|---|---|---|---|
| [The per-collapse solver](#the-per-collapse-solver-deleted) | 09-15 to 09-17 | The first GPU solver: throughput, where its time went, search and thrashing, rule kinds | Deleted in [#25]; read at `746eef2` |
| [The block kernel](#the-block-kernel) | 09-16 to 09-17 | One workgroup solving one chunk per dispatch, against the CPU reference; stitching chunks into a world | `wfc-gpu/tests/block_solver_bench.rs`, `wfc-devtools/tests/cpu_reference.rs` |
| [The streaming library](#the-streaming-library) | 09-17 to 09-23 | Whole worlds through `wave_forge`: streaming, repairs, holes, order independence, determinism across devices | `wfc-devtools/tests/` |
| [The engines](#the-engines) | 09-18 to 09-23 | Godot and Bevy: frame costs, multimesh, colliders, navigation, stages | `wave_forge_godot/`, `wave_forge_bevy/tests/` |

Questions still open against the current code are [at the end](#open-questions).

## The per-collapse solver (deleted)

The first GPU solver collapsed one cell at a time: an entropy pass and cell selection on the device,
the collapse, propagation in passes, and the whole grid downloaded again, with backtracking on the
host. It was deleted in [#25]. Its story and what it taught are in
[per-collapse-solver.md](per-collapse-solver.md). Its harnesses went with it; they can be read at
`746eef2`, the last `develop` commit before the deletion.

**Standing protocol.** dozen, release, the city of that time with its boundary conditions from
`constrain_city`. The stress suite is `wfc-devtools/tests/stress.rs`, one attempt per run. The
864-cell city is the 12×12×6 end-to-end test (`e2e_3d_city.rs`, `small_city`), steered by three
environment switches: `WFC_SEED` fixes the choice, `WFC_SWEEP` drops the harness to one attempt with
a tight iteration budget, and `WFC_REPORT_SEARCH` prints the search line (collapses, iterations,
backtracks, failing cells, undo depth). The rule-set zoo is `wfc-devtools/tests/ruleset_zoo.rs`.
Traces are Chrome traces from `--trace-chrome` ([#11]).

Two caveats apply to the whole era:

1. Every row before the packed-key fix (P28) was taken while cell selection was nondeterministic.
   The entropy shader stored the winning entropy and its index as two separate operations, so the
   pair could tear, and which cell won depended on which workgroup reached the atomic first. This
   affects the connectivity rows (P5) and the batch rows (P27) most. The qualitative comparisons
   stand; the spreads are not repeatable measurements.
2. The redo factor (P13) is a lower bound. Its counter was updated by
   `collapsed_cells.saturating_sub(undone)`, and undoing N choices restores more than N cells because
   propagation had collapsed others for free.

### Baseline ([#18], 2026-09-16)

| ID | Measurement | Result | Protocol |
|---|---|---|---|
| P1 | City 24×24×8 (4 608 cells) | 45.3 s, 102 cells/s | stress suite |
| P2 | City 48×48×10 (23 040 cells) | 403.8 s, 57 cells/s | stress suite; an earlier run exhausted the iteration budget after about 11 000 undos |
| P3 | Permissive 2-tile grid, 24³ (13 824 cells) | 96.0 s, 144 cells/s | stress suite |
| P4 | P1 again, unchanged code | 45.3 s and 48.7 s | backtrack counts differ between runs, so a change on this workload needs repeated runs |
| P5 | Connectivity-constrained 8×8×5 city | 5 s to 136 s across runs; restart-only recovery: 0 of 20 attempts succeed; conflict-directed backjumping: 3 of 3 | end-to-end test; caveat 1 applies |
| P6 | Walkable cells in the largest connected network | 0.3 to 0.85 per run, against 0.6 to 0.8 for marian42's own rules re-simulated (0.14 to 0.32 with his interiors removed) | `disconnected_walkable_cells`, without the connectivity constraint |

### Profile of the loop and the fixes it led to ([#17], [#18], 2026-09-16)

| ID | Measurement | Result | Protocol |
|---|---|---|---|
| P7 | Propagation's share of a city run | 76%: 36.9 s of 48.7 s, over 15 284 passes (2.36 ms each) | trace of one P1 run |
| P8 | The other spans of that run | grid transfers 6.7 s; entropy and selection 3.1 s together | same trace |
| P9 | Propagation passes per collapse | 2.17 | same trace |
| P10 | Full-grid sweeps among the passes | 7 048 of 15 284, the confirmation sweeps the non-atomic shader needed for its fixpoint | same trace, span argument `input_count` |
| P11 | Passes carrying 8 cells or fewer | 70% | same |
| P12 | GPU round-trips per collapse | 5.7 | same trace |
| P13 | Collapses redone by backtracking | 1.53× the minimum (a lower bound, caveat 2) | `select_cell` count against cells |
| ~~P14~~ | ~~Propagation pass, 1-cell worklist, one submit per pass, no readbacks~~ | ~~2.69 ms~~ | `wfc-gpu/tests/propagation_bench.rs`, first variant timed cold; void, the GPU had not raised its clocks. Superseded by P19 |
| ~~P15~~ | ~~Same, all 200 passes in one submit~~ | ~~3.81 ms~~ | same, cold. Superseded by P18 |
| ~~P16~~ | ~~Propagation pass, 4 608-cell worklist, no readbacks~~ | ~~3.13 ms~~ | same, cold. Superseded by P20 |
| P17 | Trivial dispatch (one workgroup, no work) | 0.099 ms | `wfc-gpu/tests/dispatch_cost_bench.rs`, warm-up, interleaved variants, medians |
| P18 | 256 dependent steps as 256 dispatches, against one dispatch that loops | one dispatch is 18× cheaper; 13× with storage reads in every step | same |
| P19 | Propagation pass, 1-cell worklist | 2.271 ms | `propagation_bench.rs`, warm, medians |
| P20 | Propagation pass, 4 608-cell worklist, freshly constrained grid | 3.094 ms | same |
| P21 | Same, every cell collapsed | 0.605 ms | same |
| P22 | Dispatch cost of a whole city solve | 22 332 dispatches, about 2 s of the 48 s run | P17 times the trace's dispatch count |
| P23 | Sweep the whole grid whenever the worklist is under a quarter of it | city 53.0 s (87 cells/s), permissive 81.3 s (170 cells/s), against P1 and P3 | stress suite; reverted |
| P24 | Sweep under a tile-visit budget (about 400 cells at 81 variants) | city 57.6 s (80 cells/s), permissive 76.9 s (180 cells/s) | stress suite; reverted |
| P25 | Rule rows stored word-aligned and unioned per possible tile | city 45.3 s → 37.0 s, 124 cells/s | stress suite |
| P26 | Atomic restriction (`atomicAnd`), confirmation sweeps dropped | city 37.0 s → 25.9 s, 178 cells/s; 0 adjacency violations | stress suite and the end-to-end test |

What P19 to P24 say together: a pass costs what its cells cost, not what its dispatch costs. The same
full-grid pass is 5× cheaper when every cell is collapsed (P20 against P21), and a 1-cell pass costs
nearly as much as a full-grid one because one thread does the work the grid otherwise spreads over
thousands. Enlarging passes to amortise the dispatch helped the 2-tile grid by 15 to 20% and slowed
the 81-variant city by 17 to 27% (P23, P24), because a swept cell unions the rows of every tile still
possible in it.

### Search: determinism, thrashing and the escalation fix ([#17], 2026-09-16)

| ID | Measurement | Result | Protocol |
|---|---|---|---|
| ~~P27~~ | ~~Batch size against thrashing, city 24×24×8, five samples per batch size, 180 s cap~~ | ~~batch 1: 23.1, 26.7, 30.2 s, backtracks 22, 361, 514, 2 of 5 timed out; batch 2: 13.6, 25.8 s, backtracks 0, 751, 3 of 5 timed out; batch 4: 8.1 to 69.3 s, backtracks 15 to 4 801, none timed out; batch 8: 5.8 to 27.1 s, backtracks 33 to 1 322, 1 of 5 timed out~~ | stress suite, one attempt; void: taken through torn selection (caveat 1) and a backjump that never escalated. Superseded by P30 and P37 |
| P28 | The same seed three times, 864-cell city | identical: 638 collapses, 638 iterations, 1 backtrack | seed 12345, after the packed-key fix (entropy and cell index in one key, one `atomicMin`) |
| P29 | A different seed | 621 collapses; the run genuinely diverges | seed 999 |
| P30 | 48 seeds, batch 1 | 47 seeds at 0 to 4 backtracks, 4.4 to 5.7 s; seed 8: 197 backtracks, 7.5 s, 98 failures at one cell | 864-cell city, `WFC_SWEEP`, `WFC_REPORT_SEARCH` |
| P31 | Seed 8 replayed three times | the same search line every time | same |
| P32 | Deepest undo across the 48 seeds | at most 2, seed 8 included | same |
| P33 | Seed 8's plateau | collapsed cells pinned at 529 to 534 for 190 backtracks over 184 iterations | progress series |
| P34 | Seed 8's contradictions by cell | 195 of 197 at one cell, (10, 6, 4); 97 of 197 raised while recovering | conflict diagnostics, `by_source` |
| P35 | Seed 8 after the escalation fix (the real conflict cell drives an escalation that accumulates across progress) | 197 → 8 backtracks; deepest undo 2 → 8 | the 48-seed corpus |
| P36 | The corpus after the fix | worst of 48 seeds: 197 → 11 backtracks; wall time 4.4 to 5.7 s → 4.2 to 5.4 s | same |
| P37 | City 24×24×8 after the fix, 8 seeds | 8 of 8 finish (before: 3 of 5); backtracks 5 to 79 (before: 22, 361, 514 and 2 timeouts); deepest undo 2 to 12 against a cap of 64; 20.2 to 23.5 s, 196 to 229 cells/s; failures at 3 to 20 distinct cells, at most 18 repeats each | stress suite, batch 1, seeded; P26's 178 cells/s used several unseeded attempts, so the throughput comparison is indicative |

### Rule kinds: the zoo ([#17], 2026-09-16)

The zoo holds the 864-cell city's grid, rules, weights and seed fixed and varies only the kind of
rule added. The control (adjacency only) reproduces the end-to-end `small_city` search line exactly.

| ID | Measurement | Result | Protocol |
|---|---|---|---|
| P38 | Seed 8, control and range exclusion ("no Z within 2 cells directly above") | control: 8 backtracks, 2.88 s; range exclusion: 80 backtracks, 6.07 s, 10 distinct failure cells, all 80 through the constraint path with no conflict cell recorded | zoo |
| P39 | Range exclusion with its failing cell fed into the escalation | no longer finishes (iteration budget exhausted); reverting restores exactly 80 backtracks and 808 iterations | zoo; reverted, escalation stays off for constraint failures |
| ~~P40~~ | ~~Counting rule, first configurations (a road within 2 cells of a door; within 3 of a building)~~ | ~~search line byte-identical to the control, 0 prunes~~ | zoo; void: the rule never fired, because it prunes only when the candidates in its ball exactly equal the count. Superseded by P41 |
| P41 | Seed 8 with a prune counter: control, range exclusion, counting (three roads within one cell of a door) | control: 8 backtracks, 2.92 s; range exclusion: 29 prunes, 80 constraint failures, 80 backtracks, 5.39 s; counting: 13 prunes, 0 failures, 1 backtrack, 2.72 s | zoo. An impossible counting configuration (three roads within one cell of every building) ground through 1 911 backtracks and stopped at 174 of 864 cells |
| P42 | 8 seeds per kind | control: 0 to 8 backtracks, 2.56 to 3.04 s. Range exclusion: 6 of 8 never finish; the two that do take 80 and 494. Counting: 1 to 48, worse than the control on 6 of 8. Surrounding: 7 of 8 finish, 3 to 40 backtracks, 2.7 to 5.9 s. Statistical (roads likelier near roads): 7 seeds at 0 and one at 7, 15 → 7 in total, 2.7 to 3.3 s | zoo, `WFC_SWEEP`; the statistical result is weak because every control run is easy |
| P43 | Harness change (`solve_rules` delegating to `solve_rules_with`), control seed 8 | unchanged: 595 collapses, 624 iterations, 8 backtracks | zoo |
| P44 | Backtracks against constraint failures | range exclusion: exactly equal (80/80, 494/494); surrounding: near identity (40/40, 3/3, 6/6, 7/7, 6/6, 5/4, 11/6), with prunes of 9 to 301 unrelated to cost; counting: monotone, backtracks 2 to 4× the failures | P42's runs |
| P45 | Range exclusion at 12.5× the iteration budget, seed 5 | finishes: 98.8 s, 4 962 backtracks, 4 927 constraint failures; worst cell 3 160 failures, of 37 distinct cells; deepest undo 99, above `MAX_UNDO_STEPS` (64) | zoo without `WFC_SWEEP` |
| P46 | Same, seed 1 | still fails at 43 200 iterations, 11 145 backtracks | same |
| P47 | Bans revived by backtracking | control seed 8: 7 of 8 backtracks revive a ban, 26 revivals over 8 bans; range exclusion seed 5: 4 955 of 4 962, 226 314 revivals over 417 bans | `WFC_CHECK_TERMINATION` |
| P48 | Ceiling probe: re-apply every ban forever (unsound) | range exclusion seed 5: 4 962 → 15 backtracks, 96.9 → 3.79 s, failures 4 927 → 15; control seed 8: 8 → 7 backtracks, 2.86 → 3.14 s; range exclusion over 8 seeds: 8 of 8 finish (before: 2 of 8), 15 to 437 backtracks, 4.0 to 23.2 s; seed 7: 494 → 200, seed 8: 80 → 97 | `WFC_PERSIST_BANS` |

### Against a CPU thread ([#21], at `e28ab9d`, 2026-09-16)

| ID | Measurement | Result | Protocol |
|---|---|---|---|
| P49 | GPU loop, city 24×24×8 | 25.8 s: propagation 16.2 s, downloads 4.3 s, entropy and selection 3.0 s; about 7.4 ms per collapse | seed 1, one traced run, cold device, so indicative |
| P50 | Blocking drains per collapse | 2 + 2P, where P is about 1.8 propagation passes, plus 3 full grid clones | read from `wfc-gpu/src/gpu/accelerator.rs` at `e28ab9d` |
| P51 | The same grid on one CPU thread | 0.14 to 0.16 s (K3): the GPU loop was about 150× slower | CPU reference |

## The block kernel

One workgroup solves one chunk inside a single dispatch, with the chunk's domains in workgroup
memory: sweep propagation with change epochs, every local minimum within a radius collapsing per
round, and checkpoint undo. The design is in [solver.md](../architecture/solver.md). It was built as
a benchmark in [#21], and moved into `wfc-gpu` as the library's solver in [#23].

**Standing protocol.** dozen, release, the city, one 8×8×8 chunk with a halo of 1, seed 7, 3
warm-up dispatches and the median of 5, kernels compiled before timing. CPU rows are the CPU
reference on a Ryzen 9 5900X, one thread unless stated, in the same run as the GPU rows they are
compared with. Commands:

```text
cargo test -p wfc-gpu --release --test block_solver_bench -- --ignored --nocapture --test-threads=1
cargo test -p wfc-devtools --release --test cpu_reference -- --ignored --nocapture
```

The bench's stitching and streaming tests moved to `wfc-devtools/tests/streaming.rs` in [#24] and
[#25]; rows that name them refer to the bench at the commit given.

### The CPU reference, first taken (at `e28ab9d`, 2026-09-16)

A deliberately plain single-threaded solver on the same rules: a stack for propagation, a full scan
for selection, marian42's undo doubling. It held two `u64` words per cell here.

| ID | Measurement | Result | Protocol |
|---|---|---|---|
| K1 | 8×8×8, 8 seeds | 2.8 to 4.6 ms per chunk, 0 to 179 backtracks | `cpu_reference.rs`, one run per seed |
| K2 | 12×12×6, 8 seeds | 8.3 to 9.6 ms | same |
| K3 | 24×24×8 | 0.14 to 0.16 s on the 6 seeds that finish; 2 thrash under its naive undo | same |
| K4 | 48×48×10 | 1.65 to 1.79 s on 4 seeds; 4 thrash | same; full-scan selection is quadratic in cells |

### The prototype ([#21], 2026-09-17)

| ID | Measurement | Result | Protocol |
|---|---|---|---|
| K5 | Propagation only, one chunk | the same fixpoint as the CPU reference (412 of 512 cells narrowed), 8 sweeps, 0 readbacks | dozen granted 32 KiB of workgroup storage |
| K6 | Full solve of one chunk, restart on contradiction | valid (0 violations) and bit-identical for the same seed; seed 1: 1 127 collapses, 2 restarts, 3 555 sweeps; seed 2: 372 collapses, 0 restarts | pcg3d hash for the choice |
| K7 | Sweep counts across builds | vary by one or two for the same seed (3 555 against 3 554); collapses and restarts identical | epochs are read while other lanes write them, so how many sweeps a change takes to be noticed depends on scheduling; the fixpoint does not |
| K8 | 1 chunk per dispatch | 29.7 ms, 41 µs per collapse, 3.2 sweeps per collapse; one CPU thread does the chunk in 2.7 ms | at `a22601b` with the bench of `e035586`, restart-only, one cell per round; CPU median of 16 seeds |
| K9 | 16, 64 and 256 chunks per dispatch | 108, 109 and 232 ms: 1.7 ms per chunk at 64 and 0.91 ms at 256 (564k cells/s, 3.0× one CPU thread); every chunk valid, restarts median 1, max 8 | same |
| K10 | 1 chunk at 1, 4, 16, 64 and 256 invocations per workgroup | 875, 294, 106, 49 and 29 ms: 445, 146, 52, 24 and 12.7 µs per step | after `e035586` |
| K11 | µs per step of the slowest chunk, 1 → 64 chunks | flat: 51.7 → 51.5 at 16 invocations, 23.8 → 23.4 at 64, 12.6 → 11.9 at 256; 23.6 at 256 chunks of 256 invocations | same |
| K12 | Slowest chunk against the mean, 256 chunks | 9 902 against 2 619 steps (3.8×); restarts max 8, median 1 | same |
| K13 | 256 chunks at 1 invocation | the device was removed by the Windows timeout (about 2 s) | since then the bench grows a chunk count only while 4× the last dispatch stays under 600 ms |
| K14 | Every local minimum within radius r collapses per round, 256 chunks, restart-only recovery | r=0: 0.93 ms per chunk, median 1 restart; r=1: 0.50 ms but 73 of 256 chunks fail (median 35 restarts); r=2: 0.79 ms, median 10 restarts, 2 fail; r=3 (64 chunks): 2.4 ms, median 16. radius 1 is 1.86× faster than radius 0 but fails 73 of 256 | cap of 64 attempts |
| K15 | Sweeps per collapse, same runs | 3.25 at r=0; 0.63, 1.21 and 1.73 at r=1, 2 and 3 | same |
| K16 | Undo: restore the checkpoint from before the failing round, doubling on repeated failure, 256 chunks | r=0: 0.27 ms per chunk (restart-only 0.92); r=1: 0.17 ms, 15.5× one CPU thread, 0 of 256 fail; r=2: 0.21 ms; r=3: 0.26 ms | 32 checkpoints per chunk in a storage buffer; CPU 2.64 ms per chunk |
| K17 | Slowest against mean steps, r=1 with undo | 1 915 against 456 (restart-only at r=0: 9 917 against 2 620) | same |
| K18 | Noise in K14 to K17 | single rows at 2 to 9 times the typical µs per step (112 and 39 µs against 12 to 15) despite medians of 5 | treat one row as indicative |
| K19 | A chunk reported as solved with 2 empty cells | undo restored checkpoint slot k after a deeper stretch of the same attempt had overwritten it at round k + 32 | found by `every_reported_success_is_a_valid_chunk` (64 chunks); fixed by undoing only to rounds nothing can have overwritten, and a restored empty cell now fails the chunk loudly |
| K20 | The CPU reference on all 24 threads, 256 chunks | 150 ms, 0.59 ms per chunk; 3 of 256 seeds thrash to the 20 000-backtrack cap (their time included); the kernel took 45 ms for the same 256 at r=1 with undo | median of 3 |
| K21 | 1 to 4 chunks per dispatch | µs per step bimodal across runs: r=2 without undo took 9.4 ms, then 29.7 ms, for the same 625 steps; r=1 with undo ran at 100 µs per step twice, against 12 µs at 16 chunks | small dispatches are not a stable measurement on this stack |

Stitching an 8×8-chunk world (64×64×8 cells), r=1 with undo. Every border contradiction the kernel
reported was checked by the CPU reference propagating the same domains, which empties a cell too.

| ID | Measurement | Result | Protocol |
|---|---|---|---|
| K22 | Chunks that fail against fixed neighbours, halo 0 | checkerboard: 29 of 32 second-pass chunks; diagonal waves (N-WFC's order): 28 of 63 | bench schedule tests |
| K23 | Where they empty | at street level (z 0 or 1) on the chunk's face, across from a fixed road, door or building tile | first contradiction of each pass, printed with its fixed neighbours |
| K24 | A halo solved and then discarded, 1 cell and 2 cells | checkerboard: 10 and 11 of 32 fail; diagonal: 3 and 3 of 63 | halo cells inside solved chunks pinned to their tiles |
| K25 | Seam violations between decided cells | 0 in every schedule | same |
| K26 | Each failed chunk repaired alone with its halo released (rewriting the neighbours' cells it covers), halo 1 then wider | the world completes with 0 seam violations: checkerboard 10 of 10 repaired (one needed halo 2), diagonal 4 of 4 at halo 1 | `*_with_repair_completes_the_world`; the isometric render shows no chunk grid |
| ~~K27~~ | ~~Live streaming, a 24×8-chunk world (192×64×8 cells), view radius 4 chunks, walking 1.4 m/s, 0.5 s ticks~~ | ~~192 chunks (98 304 cells) in 1.78 s of dispatches; median tick 43 ms, p90 141 ms, busiest 854 ms (the initial 40-chunk view fill); 10 chunks repaired; 0 seam violations; world complete~~ | bench `live_streaming_keeps_ahead_of_a_walking_player` at [#21]. Superseded by K34 |

### The model move ([#22], 2026-09-17)

| ID | Measurement | Result | Protocol |
|---|---|---|---|
| K28 | The CPU reference on `u32` words, pcg3d choice and integer weights | 8×8×8: 128k to 164k cells/s (before: 120k to 190k on `[u64; 2]`); 24×24×8: 23k to 29k (before: 29k to 35k); its naive undo now thrashes on 1 of 8 seeds at 8×8×8 and 2 of 8 at 24×24×8 | `cpu_reference.rs` at [#22], one run per seed; the choice rule changed, so trajectories and tails are not comparable with K1 to K4 |
| K29 | The block kernel on the new model types | unchanged: the fixpoint narrows the same 412 of 512 cells, the same seeds give the same collapses and restarts | bench correctness tests |

### The kernel as a library ([#23], 2026-09-17)

| ID | Measurement | Result | Protocol |
|---|---|---|---|
| K30 | 256 chunks per dispatch, r=1 with undo | 0.183 ms per chunk (46.8 ms), 21 µs per step of the slowest chunk, none failing; r=2: 0.19 ms; r=0: 0.34 ms. The CPU reference: 3.9 ms per chunk on one thread, 0.79 ms on 24 | bench at [#23] |
| K31 | The same kernel with the mask words in a loop instead of written out | 0.453 ms per chunk, 51 µs per step: 2.5× slower | same build and seeds; a mask indexed by a loop variable lands in scratch memory. The generated mask type restored K30 exactly |
| K32 | Compiling one kernel specialisation | about 4 s; ten took 43 s | `warm`; a dispatch that compiles first looked like a 4.8 s solve |
| K33 | Workgroup memory for a halo of 3 at 81 tiles | 38 288 B needed against dozen's 32 768 B, so the repair ladder stops at halo 2 | the bench's repair ladder at [#23] |
| ~~K34~~ | ~~Live streaming, the walk of K27~~ | ~~192 chunks (98 304 cells) in 1.11 s of dispatches: median tick 32 ms, p90 66 ms, busiest 443 ms (the initial 40-chunk fill), 88k cells/s while generating; 3 of 192 chunks (1.6%) could not be placed after repairs at halo 1 and 2; 14 repaired; 0 seam violations~~ | bench at [#23]. Superseded by L1, the same walk through the library's scheduler |
| K35 | Stitching an 8×8-chunk world on the library solver | a checkerboard without a halo leaves 30 of 64 chunks unsolved; with halo 1 and repair, both the checkerboard (14 repairs) and the diagonal order (2 repairs) complete it, 0 violations | bench schedule tests at [#23] |

## The streaming library

`wave_forge::WorldGenerator` solves chunks around focus points in parity batches, with a halo
discarded after the first parity and repairs for chunks that fixed borders left unsolvable. The
design is in [world.md](../architecture/world.md).

**Standing protocol.** dozen, release, the city, 8×8×8 chunks of 2 m cells, halo 1, radius 1 with
undo, kernels compiled before timing. The tests, all in `wfc-devtools/tests/`:

- `streaming.rs::live_streaming_keeps_ahead_of_a_walking_player` ("live streaming"): a 24×8-chunk
  world (192×64×8 cells), world seed 11, a focus walking 1.4 m/s with a view radius of 4 chunks,
  0.5 s ticks inside a 500 ms budget.
- `streaming.rs::a_world_asked_for_at_once_comes_out_seamless` ("stitching"): an 8×8-chunk world,
  asked for all at once.
- `game_session.rs` ("the game session"): an unbounded city, world seed 11, played in wall-clock
  time through a `Worker` by a 60 Hz frame loop; a 776 m route in about 230 s, walking 1.4 m/s and
  running 4.2 m/s, generating 3 chunks out, a view of 30 m, an eviction margin of 1.
- `hole_census.rs` ("the census"): unbounded worlds with one focus of radius 8, world seeds 11, 23,
  47, 101 and 977; every chunk given up on is solved again from its neighbours' final tiles.
- `order_diff.rs`: a 4×4-chunk city, world seeds 8 and 11, generated all at once, chunk by chunk in
  raster order, and in reverse raster order.

```text
cargo test -p wfc-devtools --release --test streaming -- --ignored --nocapture --test-threads=1
cargo test -p wfc-devtools --release --test game_session -- --ignored --nocapture --test-threads=1
cargo test -p wfc-devtools --release --test hole_census -- --ignored --nocapture --test-threads=1
cargo test -p wfc-devtools --release --test order_diff
```

### The facade ([#24] and [#25], 2026-09-17 to 09-18)

| ID | Measurement | Result | Protocol |
|---|---|---|---|
| ~~L1~~ | ~~Live streaming~~ | ~~186 chunks (95 232 cells) in 1.05 s of dispatches: median tick 47 ms, p90 61 ms, busiest 322 ms (the initial 44-chunk fill), 86k cells/s while generating; 19 chunks repaired, 107 rewritten by repairs, 0 seam violations; 6 of 186 (3.2%) no repair could place~~ | at [#24]. The wanted set is closed over face neighbours, so every second-parity chunk has four fixed faces where the bench's frontier often left one or two free: that is the price of tiles that do not depend on where the player came from, against K34's 1.6%. Superseded by L4 |
| L2 | A repair given a stream of its own (the batch seed keyed by the repair's halo) | unplaceable 8 of 184 → 6 of 186; repairs 17 → 19 | live streaming, two runs either side of the change |
| ~~L3~~ | ~~Stitching~~ | ~~61 of 64 chunks in 0.31 s of dispatches over 14 batches; 5 repaired, 30 rewritten by repairs, 3 unplaceable, 0 seam violations~~ | at [#24]. Superseded by L17 |
| ~~L4~~ | ~~Live streaming, after the per-collapse solver was deleted~~ | ~~median tick 50 ms~~ | at [#25]. Superseded by L15 |

### The game session ([#37], 2026-09-22)

| ID | Measurement | Result | Protocol |
|---|---|---|---|
| ~~L5~~ | ~~Generation against the player~~ | ~~0 chunk-frames late within 30 m of the player at 1.4 and 4.2 m/s; main thread per frame p50 0.001 ms, p99 0.006 ms, max 0.18 ms; 432 chunks solved in 167 batches, 6.2 s in the solver; at most 70 chunks held at once~~ | 14 kernels compiled in 60 s before play. Superseded by L16 |
| ~~L6~~ | ~~What the player saw~~ | ~~13 holes in view, two of them within 30 m of the start; 18 of 432 chunks unplaceable (4.2%); 33 repaired, 181 rewritten by repairs; 0 rule violations inside chunks and across 1 197 seam checks; 23 chunks walked back to, all identical, 44 not comparable because a repair or a failed neighbour touched them~~ | same session. Superseded by L16 |
| L7 | Walkability of the generated city | walkable faces continued: 0.810 inside chunks, 0.794 across seams; the largest network held 0.40 of the walkable cells at the end | same session |
| L8 | Kernels warmed only for capacity 1 and one parity at one halo | other batch sizes and repair halos compiled in the middle of the walk, about 4 s each, longer than a running player's lead | same session; every batch shape is warmed since [#54] |

### Devices and CI ([#50], [#51], 2026-09-22)

| ID | Measurement | Result | Protocol |
|---|---|---|---|
| L9 | The kernel on lavapipe from Mesa 22.3.6 (Debian 12) | the driver drops long dispatches: the workgroup stops at the first backtrack or a few hundred steps in, the dispatch completes without an error, and the host decoded the zeroed status as solved. Mesa 25.0.7's lavapipe runs the same kernel to the end | `the_result_does_not_depend_on_invocations_per_workgroup`, found by tracing each lane's phase and step; now a `NoReport` error (`wfc-gpu/tests/dropped_dispatch.rs`) |
| L10 | Time of the GPU tests by device | `block_solver` tests: 13 s on dozen, most of it compiling kernels; 0.7 s on Mesa 25's lavapipe with 4 cores | `wfc-gpu/tests/block_solver.rs`; build profile not recorded |

### No chunk left unplaced ([#54], 2026-09-23)

| ID | Measurement | Result | Protocol |
|---|---|---|---|
| L11 | Chunks a streamed city gave up on, each solved again from its neighbours' final tiles | 20 of 301 given up on, all exhausted. With the halo released and a first attempt's budget (64 attempts, 50 000 steps), 20 of 20 solve with some of 16 seeds. With the neighbours fixed, 11 of 20 solve at halo 0 and 6 of 20 at halo 1, where 9 and 14 contradict before any choice, mostly in a corner column or a diagonal halo cell | the census, world seed 11 only, 16 seeds per problem |
| L12 | Those 20 as a repair of 32 seeds side by side, per budget (attempts / steps) | 8 / 5 000: 19 of 20, at most 15 seeds needed; 16 / 10 000: 20 of 20, dispatch median 21 ms, max 37 ms; 32 / 20 000: 20 of 20, median 29 ms, max 50 ms; 64 / 50 000: 20 of 20, median 42 ms, max 79 ms | same run; every seed of a dispatch runs to its own end, so the slowest sets the time |
| ~~L13~~ | ~~Repairs of 32 seeds at 32 / 20 000, five worlds~~ | ~~0 of 1 605 chunks given up on; 280 repairs, 1 874 chunks rewritten by them, 10.0 s in the solver~~ | the census, the generator's own repairs. Superseded by L14 |
| ~~L14~~ | ~~Same, the second parity solved without a halo~~ | ~~176 repairs (−37%), 1 265 rewritten, 6.5 s in the solver (−35%), 0 given up on. The first parity without a halo instead: about 690 repairs and 2 chunks given up on; the first parity at halo 2: 175 repairs, 6.8 s~~ | same. Superseded by L20 |
| ~~L15~~ | ~~Live streaming with both~~ | ~~0 of 192 could not be placed, 20 repaired; median tick 56 ms, p90 141 ms, busiest 216 to 451 ms~~ | three runs; the p90 rose from 61 ms because a repair dispatch waits for its slowest seed. Superseded by L19 |
| ~~L16~~ | ~~The game session with both~~ | ~~0 holes in view, 0 of 455 chunks given up on, 45 repaired, 266 rewritten; 0 chunk-frames late; main thread p99 0.004 ms; 10.8 s in the solver over the 230 s walk; 27 chunks walked back to, all identical; 0 violations across 1 592 seam checks~~ | Superseded by L21 |
| L17 | Stitching with both | 0 of 64 given up on, 5 repaired, 0.37 s of dispatches | the test asserts 0 given up on since this change |

### Repairs stop early ([#55], 2026-09-23)

A seed of a repair stops once a lower seed has solved, so the winner is the one every seed running
to its end would give.

| ID | Measurement | Result | Protocol |
|---|---|---|---|
| L18 | Repair time in the live walk (20 repairs) | 600 ms → 255 ms (30 → 12.8 ms each); repairs had been 43% of the walk's solver time (600 of 1 390 ms) | live streaming, `GeneratorStats::repair_ms` and `repair_batches` |
| ~~L19~~ | ~~Live streaming~~ | ~~median tick 40 ms, p90 69 ms, busiest 167 to 433 ms; 0 of 192 could not be placed~~ | three runs. Superseded by L29 |
| ~~L20~~ | ~~Five census worlds~~ | ~~the same repairs (30, 35, 31, 36 and 44), rewriting the same chunks as without stopping, in 3.6 s of solver time instead of 6.5 s; 0 of 1 605 given up on~~ | Superseded by L27 |
| ~~L21~~ | ~~The game session~~ | ~~0 holes in view, 0 of 455 given up on, 45 repairs in 1.75 s of 9.97 s in the solver, 0 chunk-frames late, main thread p99 0.004 ms~~ | Superseded by L28 |

### The same world on two vendors ([#57], 2026-09-23)

| ID | Measurement | Result | Protocol |
|---|---|---|---|
| L22 | A 4×4-chunk city whose three portfolio repairs rewrite all 16 chunks, compared tile for tile with a recorded fixture | identical on the RTX 3070 through dozen (Mesa 26.2.2), twice, and on lavapipe (Mesa 25.0.7, LLVM 15); CI runs it on lavapipe (Mesa 25.2.8) with every pull request | `wfc-devtools/tests/golden_world.rs`, world seed 8, fixture `wfc-devtools/tests/fixtures/golden_city.txt`; re-recorded in [#78], where one cell changed because repairs run in class order |

### Order independence ([#77] and [#78], 2026-09-23)

| ID | Measurement | Result | Protocol |
|---|---|---|---|
| L23 | `order_diff` with repairs off | all three orders give the same world for seeds 8 and 11 | asserted since [#77] |
| ~~L24~~ | ~~`order_diff` with repairs on~~ | ~~12 of 16 chunks differ between orders for seed 8, 6 of 16 for seed 11~~ | opt-in measurement at [#77] (`-- --ignored --nocapture`). Superseded by L26 |
| L25 | Cost of `order_diff` | 72 s on dozen; the golden world takes 3.5 s on CI's lavapipe | at [#77] |
| L26 | `order_diff` with repair classes: a repair waits until every chunk it can see has had its first attempt, lowest class first | the city is tile for tile the same all at once, in raster order and in reverse raster order, for seeds 8 and 11, repairs included | asserted since [#78] |
| L27 | Five census worlds | 0 of 1 772 given up on (more chunks than before, because repairs have neighbours generated that no focus asked for); 181 repairs (was 176); 4.0 s in the solver (was 3.6 s) | the census at [#78] |
| L28 | The game session | 0 holes in view, 0 given up on, 58 repairs, 0 chunk-frames late, main thread p99 0.005 ms, 9.1 s in the solver; at most 94 chunks held; 42 chunks walked back to, all identical; 0 violations across 2 454 seam checks | at [#78] |
| L29 | Live streaming | 0 of 192 could not be placed, 24 repairs in 347 ms; median tick 53 to 54 ms (was 40), p90 128 to 130 ms (was 69); every tick of play inside the 500 ms budget; the initial 44-chunk fill 493 ms | three runs at [#78]. Repairs now wait for their neighbourhoods, so more of the work lands in fewer ticks: the price of order independence |

### Towns on the stage runtime ([#82], 2026-09-23)

| ID | Measurement | Result | Protocol |
|---|---|---|---|
| L30 | Four city towns of 6 to 9 chunks, each a bounded world solved whole on its site | 0 adjacency violations across their seams; identical generated all at once and one chunk at a time | `wfc-devtools/tests/towns.rs` on dozen, and on lavapipe in CI |

## The engines

No frame cost has been measured in Bevy yet; its rows are correctness checks with counts.

**Standing protocol for Godot.** Godot 4.7.2, headless, the extension built in release, the
generator on a wgpu device of its own through dozen. `wave_forge_godot/verify.sh release` runs two
scripts:

- `godot/verify.gd` ("the Godot check"): the band rule set `godot/rules.ron` (water, sand, grass,
  forest), 8×3 chunks of 8×8×8 cells of 2 units, a focus running along the strip and back at 4.2
  units/s without waiting for generation, view radius 2, colliders and navigation within 1 chunk
  (9 chunks, 512 boxes each), Jolt physics, `Engine.max_fps` 60. Its bars: Godot's slowest frame
  under 8 ms, the node's own process under 2 ms at the 99th percentile.
- `godot/verify_stages.gd` ("the stages check"): the valley pack `godot/valley.world.ron` with towns
  from `godot/city.ron`, radius 3. Since [#88] it also builds the ground from `level` and bodies
  within 2 chunks (the ground's height map, a floor slab under every street-level town cell and a
  box for every town cell above), and walks a capsule (radius 0.4, height 1.5, gravity 20) at
  4 units/s from open ground straight through a town. Its bars: the node's own process under 2 ms
  at the 99th percentile and 8 ms at worst; the walker's feet never more than 0.3 below the ground's
  surface, nor above it while standing.

How frame time is read: Godot's `Performance.TIME_PROCESS` is the slowest frame of the last second,
not the last frame's time (`main.cpp` keeps the maximum and publishes it once a second, 4.7.2). A
percentile taken over it once per frame is the slowest frame repeated. Since `1ed29f8` ([#62]) the
node times its own `process` every frame and `stats()` reports its median, 99th percentile and
maximum; "Godot's slowest frame" is the maximum of `TIME_PROCESS`, which that reading leaves valid.

### Godot: the extension ([#27], [#37], [#51], [#58]; 2026-09-18 to 09-23)

| ID | Measurement | Result | Protocol |
|---|---|---|---|
| E1 | Main loop during the strip walk | 144 frames per second throughout, including about 4 s of kernel compilation | at [#27], no frame cap, the focus paced by generation |
| ~~E2~~ | ~~Godot's process time~~ | ~~p99 0.22 ms, 0 late frames~~ | the Godot check at [#37]; void: a percentile over `TIME_PROCESS`. Superseded by E15 |
| E3 | The Godot check on lavapipe (Mesa 25, 4 cores) | loads in 1.1 s; runs the strip with 0 late frames | before [#51], the CI runner's software device |
| ~~E4~~ | ~~Same, process time~~ | ~~p99 0.9 ms, max 0.9 ms~~ | void: the equal p99 and maximum are the `TIME_PROCESS` reading. Not measured again on lavapipe |
| E5 | Freeing a node set to start on ready while it is still building | Godot's main thread froze for 40 s (process time max 40 178 ms): `Worker` joined its thread on drop | the Godot check at [#58]; fixed, the drop no longer waits |
| E6 | The Godot check after that fix | loads in 13.7 s, most of it compiling kernels; process time max 0.37 ms; 0 late frames | at [#58]; the first-run figure user story P3 quotes |

### Godot: drawing a chunk ([#59] and `966a9bf`, 2026-09-23)

Protocol: `wave_forge_godot/render_city.sh` under `xvfb`, the Compatibility renderer on Mesa's D3D12
OpenGL driver (Mesa 22.3.6, "D3D12 (NVIDIA GeForce RTX 3070)"), the city with its exported glTF
models, chunks of 8×8×8, time on Godot's thread per chunk.

| ID | Measurement | Result | Protocol |
|---|---|---|---|
| E7 | Server path: `instance_sets` and `RenderingServer`, no nodes | p50 6.9 ms a chunk, of which `instance_sets` 0.09 ms | 13 chunks, `render_city.gd server` |
| E8 | Node path: GDScript placing nodes | p50 7.5 ms a chunk | 13 chunks, `render_city.gd nodes` |
| E9 | Cost per multimesh | `multimesh_allocate_data` about 280 µs; creating one, setting its buffer and creating its instance about 6 µs each | same runs |
| E10 | A pool of 25 multimeshes of 512 instances, refilled per chunk with padded buffers and `multimesh_set_visible_instances` | 13.9 ms a chunk against 7.3 ms for fresh multimeshes; every buffer write costs a few hundred µs, more for one a draw has used | 20 chunks, a pooled variant of `render_city.gd` that is not in the repository, recorded in `966a9bf` |

### Godot: colliders ([#61], 2026-09-23)

| ID | Measurement | Result | Protocol |
|---|---|---|---|
| E11 | Building one chunk's colliders | `PhysicsServer3D` with every shape added before `body_set_space`: 0.12 ms on Jolt, 0.16 ms on Godot Physics. The body in the space first: 3.1 ms on Jolt, which rebuilds the compound per shape, 0.08 ms on Godot Physics. `StaticBody3D` and `CollisionShape3D` nodes: 1.0 ms on Jolt, 0.68 ms on Godot Physics | headless, 20 chunks of 200 boxes, a timing script that is not in the repository |
| E12 | The Godot check with colliders | Godot's process time at most 1.4 ms; 0 late frames | at [#61] |

### Godot: navigation ([#62], 2026-09-23)

Protocol: the Godot check, an agent of radius 0.5, height 1.5 and maximum climb 0.25, the map's
default cell of 0.25, one region baked per chunk within 1 chunk of the player.

| ID | Measurement | Result | Protocol |
|---|---|---|---|
| ~~E13~~ | ~~Baking from the node's own source (its shapes and its neighbours' out to two cells, a border of 4 units)~~ | ~~bakes 33 ms median, 49 ms max, off Godot's thread; preparing one costs Godot's thread 0.60 ms median, 2.6 ms max~~ | at `748a34b`. Superseded by E15 |
| E14 | Several bakes prepared in one frame, as the player crosses into a chunk | one frame of 15 ms | at `748a34b`; since then at most one bake is prepared per frame, nearest first |
| E15 | Baking from the library's NavSource, with Recast's tile padding as the border (the agent's radius in whole cells and three more: 1.25 units) | a bake takes 33 ms from asking to its mesh being in place, median and maximum alike; preparing one costs Godot's thread 0.53 ms median, 2.0 ms max; putting a finished mesh in its region 0.002 ms; the 9 chunks around the player hold 8 082 polygons; a path across two seams is 32.0 long for a straight line of 32.0. Godot's slowest frame 3.4 ms, the node's own time 0.56 ms per frame at the 99th percentile | at `a3f11b4` ([#62]) |
| E16 | Handing the source over with `append_arrays` instead of `set_vertices` and `set_indices` | preparing a bake costs 0.78 ms median | at [#62]; `append_arrays` copies the arrays again and rewrites every index |
| E17 | Neighbouring bakes' border vertices, unsnapped | the map merges them at a cell of 0.25 and at 0.2 | at [#62] |

### Godot: frame costs since ([#75], [#78], [#79], [#85]; 2026-09-23)

| ID | Measurement | Result | Protocol |
|---|---|---|---|
| E18 | The Godot check | Godot's slowest frame 5.6 ms, the node's p99 0.56 ms; path 32.0 for 32.0 | at [#75] |
| E19 | The Godot check, six runs | 5 runs with the slowest frame at 2.6 to 4.1 ms; one run at 12.2 ms, inside the node's own process, unexplained | at [#78] |
| E20 | The node's slowest frame broken down by `stats()`, four runs | 2.6 to 4.7 ms: bodies for 4 to 9 chunks at about 0.3 ms each (the collider radius of 1 bounds them at 9), one bake prepared at 1.5 to 1.9 ms, signals under 0.2 ms. Both parts are bounded, so a slower frame points outside the node | at [#79] |
| E21 | The stages check | 49 chunks with towns arrived in 16 s; the node's own process p99 0.02 ms, max 4.8 ms | at [#85], the Godot check passing alongside |

### Godot: ground and a walk through a town ([#88], 2026-09-23)

| ID | Measurement | Result | Protocol |
|---|---|---|---|
| E24 | The stages check with ground and bodies, four runs | 49 chunks arrived in 16.7 to 18.4 s; 49 chunks with ground and 25 with bodies; the node's own process p99 0.09 to 0.17 ms, max 3.9 to 5.4 ms | the stages check at [#88]; the Godot check alongside: slowest frame 3.3 ms, node p99 0.61 ms |
| E25 | The walk through the town of region (-2, -2), 56 units | crossed in 14.2 s; the feet at most 0.00 below the ground's surface and at most 0.05 above it while standing | same runs |
| E26 | The same walk with every body removed | the walker sank through the ground in the first frames: feet at 53.39, the surface at 53.75 | a one-off change to the check, to show it detects falling through |
| E27 | The stages check with a Rules stage (`cover`) and the slowest-frame breakdown, three runs | the node's own process p99 0.12 to 0.21 ms, max 4.7 to 5.2 ms; the slowest frame was always the one that emitted 2 745 to 2 841 `stage_ready` signals at once, in 3.3 to 3.8 ms, after the town search over 25×25 chunks; grounds and bodies under 0.01 ms in that frame | the stages check at [#91], radius 12 for the search |
| E28 | The same check once before the breakdown existed | the node's own process max 9.64 ms, over the 8 ms bar; what the frame held was not recorded. Emitting every event of a drain in one frame is unbounded, which is the likeliest cause | at [#91]; the bound on signals per frame is [#108] |
| E29 | Each stage of the valley pack on the stages' thread, three runs | per product: hills 0.023 ms, ground 0.019 ms, towns 0.007 ms, level 0.002 ms, trees 0.005 ms, cover 0.003 ms; city (towns solved with WFC on the GPU) 334 to 338 ms, 16.3 to 16.6 s for 49 products, the slowest 8.5 to 8.8 s. In all, the field, site and scatter stages took 72 ms of the 16.5 s | the stages check at [#87], release, dozen on the RTX 3070; the search and the view together generate 1 684 hills and 49 city products. The slowest city product is the first town, which most likely includes compiling its kernels ([#111]) |
| E30 | The same runs' slowest frame | one of the three runs failed the 8 ms bar: its slowest frame emitted 2 833 signals in 6.65 ms (max 8.03 ms); the others were 4.4 and 4.7 ms | [#108] |
| E31 | The stages check with at most 256 signals per frame, ten runs | every run passed; the node's own process p99 0.23 to 0.28 ms, max 1.77 to 2.92 ms, against 4.4 to 9.6 ms before; the slowest frame emitted 256 signals in 0.24 to 0.42 ms, and in seven runs also built 47 grounds (1.0 to 1.2 ms) and 25 bodies (0.4 to 0.5 ms). In the other three, 2.4 to 2.9 ms of that frame lies outside the breakdown, most likely draining the worker's products, which the breakdown does not time | the stages check at [#108], release, dozen on the RTX 3070 |

### Bevy ([#26], 2026-09-18)

Protocol: `wave_forge_bevy/tests`, Bevy 0.20, the generator on Bevy's own `RenderDevice` through
dozen, `#[ignore]`d tests run with `-- --ignored`.

| ID | Measurement | Result | Protocol |
|---|---|---|---|
| E22 | A headless app with the real `DefaultPlugins` | generates a 2×2-chunk city on the device Bevy created over 22 frames, nothing failing | `real_render_plugin.rs` |
| E23 | A 4×4-chunk city through the plugin against the library on a device of its own | the same tiles, cell for cell: 16 chunks, 4 batches, 2 repairs, 0 rule violations | `shared_device.rs` |

## Open questions

Measurements the current code still waits for.

- **What the numbers are on native Vulkan and a desktop.** Every GPU timing here goes through dozen,
  which distorts dispatch and submission cost, and small dispatches are not even stable on it (K21).
  The same measurement decides whether Godot should generate on its own `RenderingDevice` instead of
  a device of its own, which cannot be measured here: dozen lacks `VK_KHR_swapchain`, and Godot
  creates no `RenderingDevice` without it
  ([#39](https://github.com/AntonTegnelov/wave_forge/issues/39)).
- **What a kernel step's floor consists of.** At one chunk, µs per step fits about 11 µs plus
  0.85 µs per cell an invocation owns (K10), and the slowest chunk's step cost stays flat from 1 to
  64 chunks (K11), which rules out barriers as the main cost. The 11 µs is a fit to five points on
  one stack, not a measurement of what it is.
- **How much of the kernel's lead over the CPU is real.** The CPU reference is deliberately naive:
  it scans the whole chunk per collapse, clones the grid per collapse and thrashes under its undo on
  3 of 256 seeds (K20). An incremental selection and checkpoint undo would plausibly cut its time
  several fold, so the kernel's 3.3× over 24 threads (K20) is an upper bound on the GPU's advantage.
- **What a Prior computed upstream does to the solve.** Filtering domains before solving (driven
  WFC) removes collapses the kernel would otherwise pay for, and Rules stages are meant to compute
  such Priors ([stages.md](../architecture/stages.md)). Neither the speed-up nor the cost in quality
  has been measured.
- **Whether the city can be streaming-clean.** 181 repairs over 1 772 chunks (L27): about one chunk
  in ten needs a repair, which rewrites its neighbours. A module set that is sub-complete in N-WFC's
  sense would need none; whether the city's can be made so is untried.
- **Whether connectivity can be enforced across chunks.** A global constraint was only ever measured
  on the deleted solver (P5), and a block scheme cannot express a constraint larger than a block.
  Per-chunk connectivity with boundary contracts might compose; nothing has been tried
  ([constraints.md](../architecture/constraints.md)).
- **The 12.2 ms Godot frame** (E19). It has not recurred in the runs since the breakdown (E20)
  exists to explain it.
- **One merged mesh per chunk against one multimesh per module**, and whether Forward+ allocates
  multimeshes as slowly as Compatibility does on Mesa's OpenGL-on-Direct3D 12 (E7 to E10). Both need
  a desktop.
- **Start time with cached kernels.** The Godot check loads in 13.7 s here, most of it compiling
  kernels (E6); user story P3 asks for under 5 s on a reference desktop with kernels cached.
- **Frame cost in Bevy.** Nothing has been timed there yet.

[#11]: https://github.com/AntonTegnelov/wave_forge/pull/11
[#17]: https://github.com/AntonTegnelov/wave_forge/pull/17
[#18]: https://github.com/AntonTegnelov/wave_forge/pull/18
[#21]: https://github.com/AntonTegnelov/wave_forge/pull/21
[#22]: https://github.com/AntonTegnelov/wave_forge/pull/22
[#23]: https://github.com/AntonTegnelov/wave_forge/pull/23
[#24]: https://github.com/AntonTegnelov/wave_forge/pull/24
[#25]: https://github.com/AntonTegnelov/wave_forge/pull/25
[#26]: https://github.com/AntonTegnelov/wave_forge/pull/26
[#27]: https://github.com/AntonTegnelov/wave_forge/pull/27
[#37]: https://github.com/AntonTegnelov/wave_forge/pull/37
[#50]: https://github.com/AntonTegnelov/wave_forge/pull/50
[#51]: https://github.com/AntonTegnelov/wave_forge/pull/51
[#52]: https://github.com/AntonTegnelov/wave_forge/pull/52
[#54]: https://github.com/AntonTegnelov/wave_forge/pull/54
[#55]: https://github.com/AntonTegnelov/wave_forge/pull/55
[#57]: https://github.com/AntonTegnelov/wave_forge/pull/57
[#58]: https://github.com/AntonTegnelov/wave_forge/pull/58
[#59]: https://github.com/AntonTegnelov/wave_forge/pull/59
[#61]: https://github.com/AntonTegnelov/wave_forge/pull/61
[#62]: https://github.com/AntonTegnelov/wave_forge/pull/62
[#75]: https://github.com/AntonTegnelov/wave_forge/pull/75
[#77]: https://github.com/AntonTegnelov/wave_forge/pull/77
[#78]: https://github.com/AntonTegnelov/wave_forge/pull/78
[#79]: https://github.com/AntonTegnelov/wave_forge/pull/79
[#82]: https://github.com/AntonTegnelov/wave_forge/pull/82
[#85]: https://github.com/AntonTegnelov/wave_forge/pull/85
[#87]: https://github.com/AntonTegnelov/wave_forge/issues/87
[#88]: https://github.com/AntonTegnelov/wave_forge/issues/88
[#91]: https://github.com/AntonTegnelov/wave_forge/issues/91
[#108]: https://github.com/AntonTegnelov/wave_forge/issues/108
[#111]: https://github.com/AntonTegnelov/wave_forge/issues/111
