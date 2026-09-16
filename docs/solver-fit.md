# What we know about making this solver fast

A knowledge base for future optimisation passes. Everything here is sorted by **how well we know it**,
because the most expensive mistakes in this project so far came from treating a plausible inference as
a measured fact. One early conclusion ("a dispatch costs ~2.5 ms") was a cold-clock measurement
artifact that survived two commits and pointed the redesign in the wrong direction for a day.

Three categories, used strictly:

- **Facts** — measured by us, or quoted from a source we read. Reproducible.
- **Educated guesses** — inferences, each with the reasoning that produced it, so it can be attacked.
- **Unknowns** — things we did not measure, could not find, or that nobody appears to have settled.

An important standing caveat on every number we produced: all of it is one machine, one GPU
(RTX 3070), through **dozen**, a Vulkan-on-D3D12 translation layer. Translation distorts exactly the
quantities we care about most — dispatch and submission overhead. Numbers should be re-measured on
native Vulkan before any of them is treated as a property of the algorithm rather than of this setup.

See [solver-redesign.md](solver-redesign.md) for the chronological measurement history,
[performance.md](performance.md) for method, and [constraints.md](constraints.md) for constraint
machinery.

---

## Facts: measured by us

Release builds, RTX 3070 via dozen, city rule set (81 module variants) unless stated.

| Measurement | Value | How |
|---|---|---|
| City 24×24×8 baseline | 45.3 s, 102 cells/s | stress suite |
| Same, after unioning rule rows | 37.0 s, 124 cells/s | stress suite |
| Same, after atomic restriction (no confirmation sweep) | **25.9 s, 178 cells/s** | stress suite |
| City 48×48×10 | 403.8 s, 57 cells/s | stress suite |
| Permissive 2-tile 24³ | 96.0 s, 144 cells/s | stress suite |
| Propagation share of a city run | 76% (36.9 s of 48.7 s) | Chrome trace |
| Propagation passes per collapse | 2.17 | Chrome trace |
| **Full-grid sweeps** among passes | 7048 of 15284 | trace, `input_count` arg |
| Passes carrying ≤8 cells | 70% | trace, `input_count` arg |
| GPU round-trips per collapse | 5.7 | trace |
| Collapses redone by backtracking | 1.53× the minimum | trace, `select_cell` count vs cells |
| Warm dispatch, trivial work | 0.099 ms | `dispatch_cost_bench` |
| 256 dependent steps: N dispatches vs one looping dispatch | 13–18× cheaper inside one | `dispatch_cost_bench` |
| Propagation pass, 1-cell worklist | 2.271 ms | `propagation_bench` |
| Propagation pass, 4608-cell worklist, fresh grid | 3.094 ms | `propagation_bench` |
| Propagation pass, 4608-cell worklist, **all cells collapsed** | **0.605 ms** | `propagation_bench` |
| Connectivity-constrained 8×8×5 city | 5 s to 136 s across runs | E2E test |
| Same, with restart-only recovery | 0 of 20 attempts succeeded | E2E test |
| Same, with conflict-directed backjumping | 3 of 3 runs succeeded | E2E test |
| City run-to-run variance (unchanged code) | 45.3 s and 48.7 s | stress suite |

Two derived facts worth stating separately because they are load-bearing:

- **Cost tracks per-cell work, not dispatch count.** The same full-grid pass costs 5× less when every
  cell is collapsed, and a 1-cell pass costs nearly as much as a 4608-cell one (one thread doing the
  work the grid otherwise spreads over thousands).
- **Enlarging passes is not a win.** Sweeping the grid whenever the worklist was small made the 2-tile
  grid 15–20% faster and the 81-variant city 17–27% *slower*. Reverted.

## Facts: measured by others, read at the source

- **WFC barely searches.** Karth & Smith ran 48×48 WFC scenarios through clingo and observed **zero
  conflicts** under three variable-selection policies including heuristics-off. Their conclusion:
  WFC's strength "comes from constraint propagation removing bad choices from variable domains before
  they are considered for assignment rather than the entropy heuristic".
- **A global constraint inverts that.** Same paper: with one global constraint added, restart-on-conflict
  (WFC's own policy) "cannot find a solution within the one-minute timeout window", while ordinary
  backtracking "quickly resolved" it.
- **SAT encodings of binary CSPs lose to native arc consistency.** Gent measured establishing arc
  consistency through SAT at up to ~30× slower, and solving ~5–6× slower, than MAC on the original
  problem. Unit propagation on the *direct* encoding is strictly weaker than arc consistency (Walsh).
- **Support encodings are large.** The support encoding materialises the adjacency relation per edge.
  For our 24×24×8 grid that is ~2.1 M clauses / ~171 M literals at 81 variants (≈6.6 M / ≈1.7 G at 256).
  Our packed bitset stores the table once: 4.8 KB at 81 variants, 48 KB at 256.
- **Nogoods from strong propagators go saturated.** Katsirelos & Bacchus: for large-arity constraints
  "the resulting s-nogood will typically include all or almost all of the decision assignments", which
  "is the reason why nogood learning tends not to help GAC much".
- **The heuristic outweighs the learning, in the one large head-to-head.** Over 1064 CSP instances:
  dom/ddeg 365 timeouts → dom/wdeg 140; adding restarts *and* nogood recording on top moved 140 → 121.
- **Nogood propagation can cost more than it saves.** A reproduction of nogood-recording-from-restarts
  measured search tree down to 72% but "time is not saved … due to the overhead of nogood propagation".
- **Arc consistency is P-complete** (Kasif 1990), i.e. "inherently sequential in the worst case".
- **Propagation is confluent.** Constraint propagation is a monotone, inflationary fixpoint; Apt's
  chaotic-iteration theorem gives order-independence provided no operator is "indefinitely neglected".
  Parallel or redundant propagation cannot change the result.
- **Halo width `r·t` requires a bounded radius.** The communication-avoiding literature states the
  condition explicitly and describes the unbounded case: "every processor needs all n rows", the scheme
  degenerating to communicating every step.
- **Block schemes that work, and their guarantees.** Merrell's modify-in-blocks (revert block on
  failure, keep previous contents); Boris the Brave's four-layer offset scheme (deterministic
  regardless of traversal order, "each layer 4 block needs a total of 12 blocks from earlier layers
  evaluated", fall back to the earlier layer on failure, ~4× redundant work); N-WFC (seam-only
  constraints, diagonal order, no backtracking *if* the tileset is sub-complete); POMS (boundary erosion
  on failure).
- **Block schemes cannot express constraints larger than a block.** Stated by every block-based source.
- **Two published parallel WFC attempts lost.** A CUDA WFC measured 26–103× *slower* than its CPU
  baseline; a parallel CPU implementation found "the sequential queue-based algorithm outperformed all
  parallel implementations".
- **Mallob's clause sharing is worth ~15.6× on unsatisfiable instances and ~4.1× on satisfiable ones**,
  with satisfiable scaling stalling past 96 cores and slowdowns on instances a sequential solver
  finishes in under 1.4 s.
- **The domino problem is undecidable** (Berger 1966), so no method with bounded lookahead can decide in
  general whether a partial tiling extends to an infinite one.

## Educated guesses (with the reasoning)

Each of these is an inference. The reasoning is given so a future pass can check whether it still holds.

1. ~~**The full-grid confirmation sweeps are the largest remaining propagation cost.**~~ **Confirmed
   and now a fact:** replacing the non-atomic read-modify-write with `atomicAnd` made the sweeps
   unnecessary and took the city from 37.0 s to 25.9 s (124 to 178 cells/s), with the E2E still
   reporting zero adjacency violations. Recorded here as a worked example of the method: the guess named
   the mechanism, the prediction was falsifiable, and the measurement settled it.
2. **Per-cell propagation cost is roughly proportional to (remaining possibilities × words per cell).**
   *Reasoning:* the collapsed-grid pass is 5× cheaper than the fresh-grid one, and the kernel's inner
   loop iterates over set bits. Not directly measured as a curve, only at the two endpoints.
3. **A block-local solver would amortise dispatch cost well.**
   *Reasoning:* a workgroup can loop with `workgroupBarrier` 13–18× cheaper than separate dispatches
   (measured), and an 8³ block at 81 variants needs ~6 KB of workgroup memory against a 16 KB limit
   (arithmetic). Nobody has measured an actual block-local WFC kernel, ours or anyone else's.
4. **Live generation needs a different unit of work, not a faster kernel.**
   *Reasoning:* we are at ~8 ms per cell; a playable budget is a chunk of hundreds of cells in ~10 ms.
   That is three to four orders of magnitude, which no constant-factor kernel win reaches. Block
   decomposition plus a coarse pass that pre-filters domains is the only structure in the literature
   that changes the exponent rather than the constant.
5. **Driven WFC (pre-filtering domains from a coarse pass) is the cheapest large win available.**
   *Reasoning:* Boris states filtering domains before solving is "basically free" and marian42's city
   constrains only every 8th tile; every domain bit removed before solving is a collapse we never pay
   for. Untested here, and the quality cost is unknown.
6. **AC-4 support counters are a bad trade at our variant count.**
   *Reasoning:* counters cost cells × 6 × variants entries (~16 M at 81 variants, ~50 M at 256) against
   ~32 B/cell for the bitset, and their own proponent says they work poorly with backtracking. This is
   arithmetic plus a quoted caveat, not a measurement of both designs.
7. **Our connectivity constraint cannot be enforced across streamed blocks.**
   *Reasoning:* it is a whole-grid property, and block schemes forbid supra-block constraints. Follows
   from the block-scheme fact above, but we have not tried and failed — we have not tried.
8. **The dozen translation layer inflates our dispatch and submission costs.**
   *Reasoning:* published kernel-launch overheads are microseconds; we measure 0.099 ms warm for a
   trivial dispatch, and batching dispatches into one submit made things *worse*, which is atypical.
   Not verified against native Vulkan.

## Unknowns

Honest gaps. Several of these are where the next big win probably hides.

- **Whether GPU CDCL can be made to work.** What we know is that no competitive implementation was
  published as of this survey, and two measured ports were much slower. That is evidence about the
  state of the art, **not** a proof of impossibility. Warp-level primitives, subgroup operations, or a
  redesigned conflict-analysis scheme could change it. If someone publishes one, revisit.
- **Whether correct parallel collapse is achievable.** We found no published algorithm. Propagation is
  provably confluent; *choices* are not, and deciding whether two cells' propagation cones intersect
  looks as hard as propagating. "Looks as hard" is not a proof, and a conservative sufficient condition
  (e.g. a distance bound that is safe for a *specific* tileset) may well exist.
- **What our numbers look like on native Vulkan.** Everything above is through a translation layer.
- **The real shape of per-cell cost.** We have two points, not a curve; we do not know whether cost is
  linear in popcount, in words touched, or dominated by memory latency.
- **Whether a bitplane/SoA possibility layout helps us.** The LBM literature reports 5× for
  memory-bound neighbour sweeps, but our passes may not be memory-bound. Unmeasured.
- **Whether our tileset could be made sub-complete** in N-WFC's sense, which would remove backtracking
  entirely. This is a tileset-design question we have not attempted. POMS notes sub-completeness "may be
  difficult for tile sets in the wild".
- **How much of the run is redone work at larger sizes.** Measured 1.53× at 24×24×8; unknown at
  48×48×10 and beyond, where one run burned ~11 000 undos.
- **Whether dom/wdeg helps *us*.** The CSP literature's numbers are from binary CSP benchmarks, not
  from WFC with a global constraint. Unmeasured here.
- **What the quality cost of cheaper cell-selection is.** Merrell reports scanline order beating
  min-entropy at scale for *failure rate*; Boris says cell choice "is not usually an important
  decision" but "can have interesting effects on quality". We have not looked at either for our city.
- **Whether the connectivity constraint can be decomposed** — e.g. per-block connectivity plus boundary
  contracts that compose into global connectivity. No source addresses it; it may be possible for a
  restricted tileset.

## What this implies for the next passes

In rough order of expected value, with the basis for each:

1. ~~Remove the full-grid confirmation sweeps via atomic restriction~~ **done, 1.43x**.
2. Stop moving the whole grid per collapse *(fact: 6.7 s of 48.7 s in transfers, plus CPU packing that
   the GPU spans do not show: `upload_grid` tests every tile bit of every cell and allocates per cell)*.
3. Coarse pass pre-filtering domains, i.e. driven WFC *(guess 5)*.
4. Block-local solve as the unit of work, for both streaming and latency *(guesses 3 and 4)*.
5. dom/wdeg failure weighting to attack the 1.53× redo and the 5–136 s variance *(fact about others'
   measurements; unknown for us)*.

Do not spend effort on: SAT/CDCL encodings of pairwise adjacency, AC-4 counters at our variant count,
clause-sharing portfolios, or Morton/space-filling layouts — each is covered above with the evidence.
