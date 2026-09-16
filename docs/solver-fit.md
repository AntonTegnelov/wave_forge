# What fits Wave Forge: CDCL, ghost cells, blocks, and parallelism

This answers a specific question: given everything available — CDCL/SAT, overlapping blocks with ghost
cells, arc consistency on boundaries, GPU constraint solving — what should *this* project build, and
does changing the algorithm change how parallel it can be?

It is written after a literature survey (sources at the end) and against our own measurements, which
matter more than the literature wherever the two disagree. See [solver-redesign.md](solver-redesign.md)
for the measurement history and [constraints.md](constraints.md) for the constraint machinery.

## What we measured

| Fact | Number |
|---|---|
| Propagation share of a city solve | 76% |
| Cost of a dispatch (warm, trivial work) | 0.099 ms |
| Propagation pass, fresh grid vs every cell collapsed | 3.09 ms vs 0.605 ms |
| 256 dependent steps: separate dispatches vs one looping dispatch | 13-18x cheaper inside one |
| Collapses redone because of backtracking | 1.53x the minimum |
| City 24x24x8 after unioning rule rows | 45.3 s -> 37.0 s (102 -> 124 cells/s) |

The load-bearing conclusion: **cost tracks the work each cell needs, not the number of dispatches.**
An early "2.5 ms per dispatch" figure was a cold-clock artifact and is corrected in
[solver-redesign.md](solver-redesign.md).

## CDCL and SAT: no, and the reason is structural

**WFC barely searches.** Karth and Smith ran 48x48 WFC scenarios through clingo and measured **zero
conflicts** under three different variable-selection policies, including heuristics disabled. Their
conclusion is that WFC's strength "comes from constraint propagation removing bad choices from
variable domains before they are considered for assignment rather than the entropy heuristic". Clause
learning learns from conflicts; on a workload with no conflicts there is nothing to learn.

**The encoding would cost about four orders of magnitude.** Unit propagation on the *direct* encoding
is strictly weaker than arc consistency (Walsh), so it would downgrade what we already do. Matching our
propagation needs the *support* encoding, which materialises the adjacency table per edge: roughly 171
million literals at 81 variants and 1.7 billion at 256. Our packed bitset stores that table **once** —
4.8 KB at 81 variants, 48 KB at 256 — and shares it across all 12864 neighbour pairs of a 24x24x8 grid.
That gap is structural to CNF, not an implementation detail. Gent measured the resulting race directly:
establishing arc consistency through SAT was up to ~30x slower, and solving ~5-6x slower, than doing it
natively.

**Learning from strong propagators produces useless nogoods.** Katsirelos and Bacchus: when a
constraint's arity is large "the resulting s-nogood will typically include all or almost all of the
decision assignments", which "is the reason why nogood learning tends not to help GAC much". Our
connectivity constraint is exactly such a large-arity global.

**And CDCL is the worst possible fit for a GPU.** No published work has a competitive CDCL loop on a
GPU; one measured port ran 50-250x slower than CPU MiniSat. Watched literals are mutable per-clause
state that breaks under concurrent threads, conflict analysis is described in the literature as
"intrinsically serial", and propagation's memory access "can only be determined at run time".

**What we take instead.** Two cheap pieces of the same family:

- **Conflict-directed backjumping**, which we already have. This is not optional: with a global
  constraint, restart-on-contradiction could not solve at all (clingo timed out at 60 s where
  backtracking "quickly resolved" it), and our own constrained city went from 1-of-3 runs to 3-of-3
  when backjumping replaced restarts.
- **Failure weighting (dom/wdeg)**, not yet built. In the one large head-to-head, switching to dom/wdeg
  cut timeouts from 365 to 140 and average time from 125 s to 48 s, while adding restarts *and* nogoods
  on top moved 140 to 121. The heuristic is the 80/20; the learning is the 20/80. It costs one counter
  per cell, updated where we already detect contradictions.

## Ghost cells: the stencil theory does not transfer, but blocks still do

Halo width `r·t` (radius times deferred steps) is necessary and sufficient **only when the per-step
propagation radius is statically bounded**. Ours is not: a single collapse can cascade arbitrarily far.
The communication-avoiding literature is explicit about that case — when the dependency cone is not
locally bounded, "every processor needs all n rows", the halo inflates to the whole domain, and the
scheme degenerates to communicating every step. The documented responses are to accept that
degeneration, to split the operator so long-range coupling is handled separately, or to drop redundancy
and schedule around true dependencies. None of them is "pick a wider halo".

What *does* work is a different idea that shares the word "block": **decompose into blocks that are
solved independently, with boundaries pinned rather than exchanged.** Three published schemes:

| Scheme | Start state | Blocks | On failure |
|---|---|---|---|
| Merrell, modify-in-blocks | fully resolved (trivial model) | 10³, fixed sweep, 50% overlap | revert block, keep old contents |
| Punch-Out Model Synthesis | indeterminate | randomised centres, size from correlation length | erode boundaries probabilistically |
| Boris the Brave, infinite blocks | known-good background | 4 layers, each offset 50% on alternating axes | fall back to the earlier layer |

Boris's scheme is the one that matches our streaming goal, because its guarantees are the ones an
infinite city needs: lazy, **deterministic regardless of the player's path**, constant work per block
("each layer 4 block needs a total of 12 blocks from earlier layers evaluated"), and a defined failure
path. It costs about 4x redundant work. Extending it to 3D is ours to design — the published scheme
cycles two axes and we have three.

Two warnings, both from practitioners who shipped this:

- marian42 built our exact history-undo scheme for an infinite city, **abandoned it**, and shipped
  offset chunks with fallback-to-patch instead.
- Every block-based source states the same hard limit: "any patterns or special constraints that are
  naturally larger than the size of a block are simply impossible".

## Does changing the algorithm change parallelisability?

Yes — decisively, and not in the direction one would hope.

- **Parallel propagation has a theoretical ceiling.** Establishing arc consistency is P-complete, so it
  is "inherently sequential in the worst case". Not fatal, but it means propagation will not scale by
  throwing cores or lanes at it.
- **Parallel *collapse* is unpublished.** No peer-reviewed work implements simultaneous collapse of
  multiple cells. Propagation is confluent — a monotone fixpoint, order-independent given fairness — so
  propagating in parallel is provably safe. *Choices* are not confluent, and deciding whether two cells'
  propagation cones intersect is as hard as doing the propagation.
- **Every published attempt to parallelise WFC itself lost.** The one CUDA WFC ran 26-103x slower than
  its CPU baseline; a parallel CPU implementation found that "the sequential queue-based algorithm
  outperformed all parallel implementations".
- **Blocks are the one parallel axis with evidence behind them** — and they only work because pinned
  boundaries make blocks genuinely independent, which is a property of the decomposition, not of the
  solver.

So: the algorithm choice determines the unit of parallelism. Cell-level gives us confluent propagation
and nothing else. Block-level gives us real independence, at the cost of forbidding constraints larger
than a block.

## Portfolios and clause sharing: also no

Mallob/MallobSat wins the SAT competition's cloud track by sharing learned clauses across thousands of
cores, so it is worth being explicit about why it does not apply. Its regime differs from ours on every
axis that matters: hundreds to thousands of cores across many machines, problems that run for minutes
to hours, *unsatisfiable* instances (clause sharing is worth ~15.6x there against ~4.1x on satisfiable
ones, and satisfiable scaling stalls past 96 cores), learned clauses that must be discovered before
they can be shared, and deliberate nondeterminism as the engine of diversity. Their own measurements
show slowdowns on instances a sequential solver finishes in under 1.4 s, and their predecessor was
statistically indistinguishable from an ordinary shared-memory portfolio at 8-16 cores.

We are satisfiable by construction, latency-bound at tens of milliseconds, on one GPU and one CPU, and
we want reproducible output. Two things do transfer:

- **Parallelise across regions, not within a region.** Solving k formulas with p/k workers each is more
  efficient than solving one with p; their cooperative efficiency is ~16% at 384 cores. For us that
  means threads should own *different* blocks rather than cooperate on one — which is also what the
  block decomposition above wants.
- **If we build a nogood store, bound it by volume and forget entries.** Budgeting by volume with
  adaptive quality beat fixed length thresholds, and ageing entries out (15 s) beat keeping them
  forever. Their scoring-policy ablation barely moved, so it is not worth over-engineering.

## The recommendation

1. **Keep the architecture: CPU owns the search, GPU does propagation.** This is what every proven
   system in the survey does, including the one GPU-assisted SAT solver that beat a competition-winning
   baseline. Device-resident search is unproven, and WGSL cannot express the persistent kernels it would
   need.
2. **Keep optimising the kernel, guided by the per-cell cost model.** Unioning rule rows bought 18%.
   The same measurement says the next candidates are skipping cells that are already collapsed (a
   collapsed grid is 5x cheaper to sweep) and a bitplane/SoA possibility layout. Note **AC-4 support
   counters are contraindicated for us**: at 256 variants they cost roughly 50x the memory of the bitset
   wave, and their author says they work poorly with backtracking.
3. **Attack search cost with dom/wdeg weighting**, then re-measure the 1.53x redo and the 5-136 s spread
   on the constrained city. Cheap, device-side, no new dispatches.
4. **Adopt block decomposition for streaming**, Boris-style, extended to 3D. Our own measurement makes
   this more attractive than it is for a CPU implementation: a workgroup can loop over a block in
   workgroup memory 13-18x cheaper than issuing separate dispatches, and an 8³ block at 81 variants is
   ~6 KB against a 16 KB limit. Blocks are simultaneously the unit of work, of parallelism, and of
   streaming.
5. **Solve global connectivity above the blocks, not inside them.** A coarse pass decides road and path
   topology for the region; blocks then fill in detail with their boundaries pinned, and the
   per-block connectivity constraint keeps each block locally walkable. This is the only approach with
   support in the literature: Boris's "decide it elsewhere", marian42's pre-generated patches, and
   Caves of Qud's post-pass that adds doors after generation. The alternative — a truly global
   connectivity constraint over a streamed world — is unsolved, and the domino problem says no local
   method can decide extendability in general.

**What would change this recommendation:** if measurement shows the kernel is no longer the bottleneck
after items 2 and 3, the case for block-local solving becomes purely about streaming rather than
performance. And if a tileset can be made *sub-complete* in N-WFC's sense — every legal boundary always
extendable — then blocks need no backtracking at all, which would be worth more than any of the above.
That is a tileset-design question, and per Boris "robustness for Wave Function Collapse boils down to
the tileset".

## Sources

marian42 ([article](https://marian42.de/article/wfc/), [infinite](https://marian42.de/article/infinite-wfc/)) ·
Boris the Brave ([infinite blocks](https://www.boristhebrave.com/2021/11/08/infinite-modifying-in-blocks/),
[editable WFC](https://www.boristhebrave.com/2022/04/25/editable-wfc/),
[arc consistency](https://www.boristhebrave.com/2021/08/30/arc-consistency-explained/),
[table constraints](https://www.boristhebrave.com/2021/08/30/advanced-table-constraints/),
[tile generators](https://www.boristhebrave.com/2021/10/31/constraint-based-tile-generators/),
[driven WFC](https://www.boristhebrave.com/2021/06/06/driven-wavefunctioncollapse/)) ·
Karth & Smith, [WFC is Constraint Solving in the Wild](https://escholarship.org/uc/item/1f29235t) (FDG 2017) and IEEE ToG 2022 ·
Merrell, [model synthesis vs WFC](https://paulmerrell.org/wp-content/uploads/2021/07/comparison.pdf) ·
[N-WFC](https://arxiv.org/abs/2308.07307) · [Punch-Out Model Synthesis](https://arxiv.org/abs/2501.14786) ·
Walsh, SAT v CSP (CP 2000) · Gent, [Arc Consistency in SAT](https://sites.cs.st-andrews.ac.uk/people/ipg1/papers/ipgECAI.pdf) (ECAI 2002) ·
Katsirelos & Bacchus, Generalized NoGoods (AAAI 2005) · Boussemart et al., dom/wdeg (ECAI 2004) ·
Lecoutre et al., [Nogood Recording from Restarts](https://www.ijcai.org/Proceedings/07/Papers/019.pdf) (IJCAI 2007) ·
Kasif, AC is P-complete (AI 1990) · Gent et al., [Parallel Constraint Solving](https://arxiv.org/abs/1803.10981) (TPLP 2018) ·
Katsirelos et al., [Barriers to Parallelizing SAT](https://ojs.aaai.org/index.php/AAAI/article/view/8660) (AAAI 2013) ·
Prevot, Soos & Meel, [GpuShareSat](https://arxiv.org/abs/2012.03119) (SAT 2021) ·
Osama, Wijs & Biere, ParaFROST (TACAS 2021) · [cuWaveFunctionCollapse](https://github.com/Chocomunk/cuWaveFunctionCollapse) ·
Meng & Skadron, ghost zones (IJPP 2011) · Demmel et al., avoiding communication (IPDPS 2008) ·
Carson, communication-avoiding Krylov (PhD 2015) · Apt, [Essence of Constraint Propagation](https://arxiv.org/abs/cs/9811024)
