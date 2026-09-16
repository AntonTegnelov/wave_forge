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
| Same seed three times, 864-cell city | identical: 638 collapses, 1 backtrack | `WFC_SEED`, after the packed-key fix |
| A different seed (999) | 621 collapses — genuinely diverges | `WFC_SEED` |
| 48-seed corpus, 864-cell city, batch 1 | 47 seeds at 0–4 backtracks, 4.4–5.7 s | `WFC_SWEEP`, `WFC_REPORT_SEARCH` |
| Worst seed of the 48 (seed 8) | **197 backtracks**, 7.5 s, 98 failures at one cell | same |
| Seed 8 replayed three times | identical line every time | same |
| Deepest undo, across all 48 seeds | **≤ 2**, including the 197-backtrack seed | same |
| Seed 8 plateau | collapsed pinned to 529–534 for 190 backtracks / 184 iterations | progress series |
| Seed 8 contradictions, by cell | **195 of 197 at one cell**, (10, 6, 4) | conflict diagnostics |
| Of those, raised while *recovering* | 97 of 197 | `by_source` |
| Seed 8 after the escalation fix | **197 → 8 backtracks**, deepest undo 2 → 8 | 48-seed corpus |
| Worst of 48 seeds after the fix | **197 → 11 backtracks** | same |
| Corpus wall time after the fix | 4.4–5.7 s → 4.2–5.4 s, no regression | same |
| 24×24×8 stress city after the fix | **8 of 8 finished**, was 3 of 5 | 8 seeds, batch 1 |
| Same, backtracks | 5–79, was 22/361/514 plus 2 timeouts | same |
| Same, deepest undo | 2–12, against a cap of 64 | same |
| Same, run time | 20.2–23.5 s, 196–229 cells/s | same |
| Failure spread at 4608 cells | 3–20 distinct cells, ≤18 repeats each | same |
| Zoo control (adjacency only), 8 seeds | 0–8 backtracks, 2.56–3.04 s | rule-set zoo, 864 cells |
| Zoo range exclusion, 8 seeds | **6 of 8 never finish**; the two that do: 80 and 494 backtracks | same |
| Zoo counting, 8 seeds | 1–48 backtracks, worse than control on 6 of 8 | same |
| Zoo surrounding, seeds 1/5/8 | 11, 6, **did not finish**; prunes 142 and 9 | same |
| Same, failure count vs cost | failures 6 and 6 on both finishing seeds, prunes differ 16× | out-of-sample check |
| Backtracks vs constraint failures | range exclusion: **exactly equal** (80/80, 494/494) | same |
| Same, counting | monotone, 2–4× the failure count | same |
| Range exclusion at 12.5× budget, seed 5 | **finishes**: 98.8 s, 4962 backtracks, 4927 constraint failures | no `WFC_SWEEP` |
| Same, seed 1 | still fails at 43 200 iterations, 11 145 backtracks | same |
| Worst single cell, seed 5 | **3160 failures**, of 37 distinct cells | same |
| Deepest undo observed | **99**, above `MAX_UNDO_STEPS` (64) | same |
| Bans revived, control seed 8 | 7 of 8 backtracks; 26 revivals over 8 bans | `WFC_CHECK_TERMINATION` |
| Bans revived, range exclusion seed 5 | 4955 of 4962 backtracks; **226 314 revivals over 417 bans** | same |
| Ceiling probe, range exclusion seed 5 | 4962 → **15** backtracks, 96.9 → **3.79 s**, failures 4927 → 15 | `WFC_PERSIST_BANS` (unsound probe) |
| Same, control seed 8 | 8 → 7 backtracks, 2.86 → 3.14 s (no effect) | same |
| Same, 8-seed sweep | **8 of 8 finish** (was 2 of 8); backtracks 15–437, 4.0–23.2 s | same |
| Same, seeds that finished under both | seed 7: 494 → 200; seed 8: **80 → 97 (worse)** | same |

**Two corrections to the rows above.** Every row measured before the packed-key fix was taken while
cell *selection* was nondeterministic: the entropy shader stored the winning entropy and its index as
two separate operations and the pair could tear, so which cell got selected depended on which workgroup
reached the atomic first. That affects the connectivity-constrained rows in particular — the 5–136 s
spread, the 0-of-20 restart result and the 3-of-3 backjumping result all predate it. They are kept
because the *qualitative* comparisons still stand, but the spreads are not repeatable measurements and
should not be quoted as such. Separately, the **1.53× redo figure is approximate**: it derives from a
counter updated by `collapsed_cells.saturating_sub(undone)`, and undoing N choices restores more than N
cells because propagation had collapsed others for free. The counter therefore drifts upward after
backtracking, so the true redo factor is somewhat higher than reported.

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
- **Thrashing is the literature's own term for our symptom.** Mackworth (1977) defines it by cause, one
  of which is a failure rediscovered repeatedly, and warns it "cannot be removed by such minor
  palliatives as reordering the nodes." Dechter & Frost: "rediscovering the same inconsistencies and
  same partial successes during search."
- **The correct backjump target is the most recent member of the conflict set.** Dechter & Frost,
  Proposition 4: "the latest variable in its jumpback set is the earliest variable to which it is safe
  to jump." Jumping later re-fails; jumping earlier can skip solutions. Prosser, Ginsberg and CDCL's
  assertive level all choose the same target.
- **CBJ gets its depth from accumulation, not from one wide jump.** Prosser merges sets on each jump
  (`conf-set[h] ← conf-set[h] ∪ conf-set[i] − {h}`), and contrasts this with Gaschnig's BJ, which jumps
  once then steps chronologically. Taking the most recent member of a *larger* set gives a *shallower*
  jump in real CBJ too; it is harmless only because the sets accumulate.
- **Resetting the accumulated set on progress destroys completeness.** Prosser, verbatim: "if `P` was
  dispensed with, or was reset whenever a successful forward move was made, we would again have an
  incomplete algorithm."
- **Under-jumping re-fails; over-jumping loses solutions.** The two error directions are not symmetric
  (Dechter & Frost, Propositions 2 and 4). Ours is the benign direction.
- **Backjumping pays *least* in exactly our regime.** Chen & van Beek (JAIR 2001): "as the level of
  local consistency that is maintained in the backtracking search is increased, the less that
  backjumping will be an improvement." WFC maintains arc consistency to fixpoint with a fail-first
  heuristic. They do refute the stronger claim that CBJ becomes useless.
- **Dynamic backtracking combines badly with dynamic variable ordering.** Baker (AAAI 1994): "worse by a
  factor exponential in the size of the problem", and explicitly not an overhead effect — "the effective
  search space itself becomes larger". min-entropy is a dynamic variable ordering, so this rules the
  technique out for us.
- **Restarts eliminate the right tail and exploit the left one.** Gomes, Selman & Kautz: randomised
  rapid restarts "provably eliminate heavy-tails to the right of the median" and exploit "a
  non-negligible chance of very short runs". Luby, Sinclair & Zuckerman give the optimal fixed cutoff
  when the distribution is known. Naive fixed-cutoff restart is incomplete; the fixes are an increasing
  cutoff or retaining nogoods across restarts (Lecoutre et al., IJCAI 2007).
- **Failure-weighted variable ordering is the literature's anti-thrashing heuristic.** Boussemart et al.
  (ECAI 2004) weight constraints by dead-ends caused and prefer variables involved in them; dom/wdeg is
  the default in solvers such as Choco.
- **Escalation must accumulate with failure count.** POMS raises erosion probability with the number of
  failed attempts, because "without the erosion, block level solvers could perpetually attempt
  resolution on blocks with identical initial state."
- **Our exact failure mode is named in the WFC world.** Boris the Brave calls it stalling: the algorithm
  "doesn't know to backtrack out of it, instead repeatedly exploring variations of the partial
  solution" — "rabbit holes". Tessera ships a step limit that "will allow some backtracking to occur,
  but after a fixed amount of computation it will automatically retry with a fresh generation."
- **marian42's failure is the opposite of ours.** He reports "errors are recognized very late which
  leads to many steps being backtracked" — undos too deep, where ours are too shallow.
- **Gumin's original WFC does not backtrack at all**, restarting globally instead (Karth & Smith).

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
8. ~~**The escape from a thrashing plateau is accidental rather than directed.**~~ **Confirmed as a
   description of the old recovery, and now obsolete.** It predicted the abrupt escape correctly, and
   the escalation fix removed the plateau it described (seed 8: 197 backtracks to 8). Kept as a worked
   example: this guess is what identified that recovery was enumerating tile bans at depth one, which
   is what pointed at the reset-on-progress as the mechanism worth attacking.
9. **Restart with a cutoff is the highest-value fix available for our measured distribution.**
   *Reasoning:* 47 of 48 seeds finish in 0–4 backtracks, so a short cutoff has a very high success
   probability per attempt and expected cost near a trivial run, while the bad seed pays 197 backtracks.
   Luby's analysis favours a fixed cutoff when the distribution is this well characterised, and Chen &
   van Beek argue the alternative (perfecting the backjump) pays least in our regime. Not yet measured
   *here*: we have not implemented it, and the cutoff value is unchosen.
   **Update, after the escalation fix:** the tail this argument rests on has largely collapsed — the
   worst of 48 seeds is now 11 backtracks rather than 197 — so the urgency is much lower, and note that
   Chen & van Beek's prediction (backjumping pays least in our regime) is the one thing here the
   measurement went *against*: fixing the backjump was worth 24× on the bad seed. The reasoning still
   holds for whatever tail remains at larger grid sizes, which is unmeasured.
10. **The `z = 4` layer is where our city rule set is tightest.**
   *Reasoning:* failure cells reported across the corpus fall most often on `z = 4` and never on
   `z = 5`; `constrain_city` forces air at the top layer and street level at the bottom, which would
   squeeze `z = 4` between forced air and whatever the buildings did below. The sample is biased (top
   five cells per seed, failing seeds only), so this is a hypothesis about the rule set, not a finding.
11. **The dozen translation layer inflates our dispatch and submission costs.**
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
- **Whether our runtime distribution is genuinely heavy-tailed.** The shape is suggestive — one seed in
  48 roughly fifty times worse than the next — but **one outlier is not a measurement of a tail**. The
  recognised diagnostic is a log-log plot of the survival function showing linear decay plus a tail
  index below 1, over many more seeds and ideally many runs on the same hard instance. Gomes et al. were
  careful about this distinction and found domains with no heavy tails at all.
- **Why (9, 5, 4) is unsatisfiable in the first place.** We know the search cannot escape it and we know
  two reasons why. We do not know what it is failing to escape *from*. Fixing the recovery machinery
  could well turn seed 8 into a faster 197 backtracks without touching the cause.
- **Whether the cell the GPU reports as the contradiction site is stable.** `propagate.wgsl` records it
  with a plain `atomicStore`, not a min or a CAS, so with several workgroups failing in one pass it is
  last-writer-wins: *a* contradicting cell, not a canonical one. Whether it reproduces across runs of a
  fixed seed is being measured now, and it matters because a backjump target that jitters is not a
  target.
- ~~**Whether our undo can genuinely cycle rather than merely stall.**~~ **Partly answered, and the
  answer is bad.** Measured with `WFC_CHECK_TERMINATION`: bans do not survive. The control revives a
  forbidden tile on 7 of 8 backtracks; a constrained run revives 226 314 times over only 417 distinct
  bans, so the same small set is discarded and re-derived thousands of times. The monotone-progress
  argument for termination therefore **does not hold**. What remains genuinely unknown is whether an
  actual cycle occurs, which needs the same full state to recur and has not been demonstrated — so this
  removes the termination guarantee without proving non-termination. Not unsoundness either: a ban is
  valid only in the context that produced it.
- **Whether the connectivity constraint can be decomposed** — e.g. per-block connectivity plus boundary
  contracts that compose into global connectivity. No source addresses it; it may be possible for a
  restricted tileset.

## What this implies for the next passes

In rough order of expected value, with the basis for each:

1. ~~Remove the full-grid confirmation sweeps via atomic restriction~~ **done, 1.43x**.
2. Stop moving the whole grid per collapse *(fact: 6.7 s of 48.7 s in transfers, plus CPU packing that
   the GPU spans do not show: `upload_grid` tests every tile bit of every cell and allocates per cell)*.
3. ~~Aim the backjump at the cell that actually failed rather than the cell we chose to collapse~~
   **done, 24× on the pathological seed**: the true conflict cell now drives an escalation that
   accumulates across progress, and the radius widening — which could only ever make undos shallower —
   is gone. Seed 8 fell from 197 backtracks to 8, the worst of 48 from 197 to 11, at no cost to the
   healthy seeds. Verified on the 864-cell city only; the 24×24×8 stress city is next.
4. Restart with a cutoff, with an increasing cutoff or retained nogoods for completeness *(guess 9)* —
   ~~demoted~~ **re-promoted, and now the top open item.** The demotion was measured on adjacency rules
   only, where the tail had collapsed to 11 backtracks across 48 seeds. On a constraint-heavy rule set
   the tail is alive and severe: range exclusion takes 4962 backtracks and 98.8 s on one seed and does
   not finish at all on another, even at 12.5× the iteration budget. That is the distribution Luby and
   Gomes et al. describe, and a short cutoff with retained nogoods is the recognised remedy. Correct
   for what it was measured on, wrong as a general conclusion.
5. **Conditional nogood recording — keep each ban with the assignments that justify it.** This is now
   the best-evidenced item on the list.
   **Decided approach:** start with the *cheap* version — record the history depth at which each ban
   was made, and apply it only while the choices below that depth are unchanged. It is sound, fits
   inside `accelerator.rs`, and can be measured against the ceiling in one sitting. Its weakness is
   known in advance: it forgets a ban whenever the search diverges below that depth, so it will recover
   some fraction of the 8-of-8 ceiling rather than all of it.
   **Held in reserve:** the full version, carrying real propagation antecedents so a ban survives as
   long as its actual cause does. Much closer to CDCL and to the ceiling, but it needs
   `direct_strategy` to report which cells caused each wipeout — a GPU-side change plus plumbing.
   Return to it if the cheap version measures short.
   The criterion for choosing between them is the least work that reaches **live city generation**,
   not the most complete algorithm; if that judgement turns out wrong, adjust rather than persist. Bans do not survive our undo: a constrained run revives them
   226 314 times over 417 distinct bans, so the same small set is discarded and re-derived thousands of
   times, and the usual monotone-progress argument for termination does not hold. An unsound upper-bound
   probe that re-applies every ban (`WFC_PERSIST_BANS`) takes range exclusion from **2 of 8 seeds
   finishing to 8 of 8**. A correct implementation applies a ban only while its condition holds, so it
   will recover some fraction of that, not all of it — but the ceiling is high enough to justify
   building. It is also the only mechanism the literature offers for *preventing* a conflict being
   re-derived rather than bounding the cost of re-deriving it.
6. **Recovery for global-constraint failures, which currently has none that works.** Range exclusion
   fails 4927 times in one run, 3160 of them at a single cell, on the one path where undo escalation is
   deliberately disabled — enabling it there was measured and made things strictly worse (a 6.07 s run
   stopped finishing). Item 5 is the most promising route to this, since the probe's gain comes from the
   search ceasing to re-enter those states at all: constraint failures fall from 4927 to 15 on the seed
   measured in isolation.
5. Coarse pass pre-filtering domains, i.e. driven WFC *(guess 5)*.
6. Block-local solve as the unit of work, for both streaming and latency *(guesses 3 and 4)*.
7. dom/wdeg failure weighting to attack the redo factor and the run-to-run variance *(facts about
   others' measurements, and Boussemart et al.'s anti-thrashing rationale; unknown for us)*. We already
   record `failures_by_cell`, so the input exists.

Do not spend effort on: SAT/CDCL encodings of pairwise adjacency, AC-4 counters at our variant count,
clause-sharing portfolios, or Morton/space-filling layouts — each is covered above with the evidence.
