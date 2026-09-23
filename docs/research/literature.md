# Solver literature

The sources behind the solver's design, one entry per source: what it claims, and whether we rely on
that claim, tested it, or ruled something out with it. Quoted passages are as the notes that first
read each source recorded them. Where our own measurement agreed or disagreed, the entry says so and
points at [measurements.md](measurements.md) or [per-collapse-solver.md](per-collapse-solver.md),
which hold the protocols.

The research behind world generation (layers, stages, placement, the games surveyed) is in
[worldgen-survey.md](worldgen-survey.md).

## Wave Function Collapse and model synthesis

- **Gumin, WaveFunctionCollapse** ([repository](https://github.com/mxgmn/WaveFunctionCollapse)). The
  original algorithm. It does not backtrack; on a contradiction it restarts globally.
- **Karth and Smith, "WaveFunctionCollapse is Constraint Solving in the Wild"** (FDG 2017, and IEEE
  ToG 2022; [paper](https://escholarship.org/uc/item/1f29235t)). 48×48 scenarios run through clingo
  showed zero conflicts under three variable-selection policies, heuristics off included; WFC's
  strength "comes from constraint propagation removing bad choices from variable domains before they
  are considered for assignment rather than the entropy heuristic". With one global constraint added,
  restart-on-conflict "cannot find a solution within the one-minute timeout window" while ordinary
  backtracking "quickly resolved" it. **Agrees with us:** a 24×24×8 city decides about 1300 of 4608
  cells by propagation, and our connectivity-constrained city went from 0 of 20 attempts under
  restarts to 3 of 3 runs under backjumping.
- **Merrell, model synthesis compared with WFC**
  ([paper](https://paulmerrell.org/wp-content/uploads/2021/07/comparison.pdf)). WFC fails 98 to 100% of
  attempts at 200², with lowest-entropy ordering implicated, while model synthesis succeeds in seconds;
  scanline order beats lowest entropy at scale on failure rate. Modify-in-blocks starts from a fully
  resolved trivial model, solves 10³ blocks in a fixed sweep with 50% overlap, and on failure reverts
  the block and keeps its old contents. **We use** the revert-on-failure idea: a repair is a block
  modified with its halo released. Scanline order is untested here.
- **marian42, "Infinite procedurally generated city with the Wave Function Collapse algorithm"**
  ([article](https://marian42.de/article/wfc/), [infinite WFC](https://marian42.de/article/infinite-wfc/),
  [code](https://github.com/marian42/wavefunctioncollapse)). Modules described by face connectors, a
  walkable-neighbour rule, and a city walkable by design with no global constraint (verified by
  re-simulating his rules, see [constraints.md](../architecture/constraints.md)). He built a
  history-undo solver for an infinite city, abandoned it, and shipped offset chunks with a fallback,
  reporting that "errors are recognized very late which leads to many steps being backtracked": undos
  too deep, where ours were too shallow.
- **N-WFC** ([arXiv:2308.07307](https://arxiv.org/abs/2308.07307)). Sub-grids that constrain only
  their seams, solved in diagonal order, need no backtracking if the tile set is sub-complete.
  **Tested:** our city is not sub-complete; under diagonal order with faces fixed, 28 of 63 chunks had
  unsatisfiable borders, 3 of 63 with a one-cell halo.
- **Punch-Out Model Synthesis** ([arXiv:2501.14786](https://arxiv.org/abs/2501.14786)). Blocks at
  randomised centres, and boundary erosion whose probability rises with failed attempts, because
  "Without the erosion, block level solvers could perpetually attempt resolution on blocks with
  identical initial state." Notes that sub-completeness "may be difficult for tile sets in the wild".
  **Agrees with us:** the same correction as the escalation fix in
  [per-collapse-solver.md](per-collapse-solver.md).
- **Boris the Brave, blog posts on constraint-based generation**
  ([infinite modifying in blocks](https://www.boristhebrave.com/2021/11/08/infinite-modifying-in-blocks/),
  [editable WFC](https://www.boristhebrave.com/2022/04/25/editable-wfc/),
  [arc consistency explained](https://www.boristhebrave.com/2021/08/30/arc-consistency-explained/),
  [advanced table constraints](https://www.boristhebrave.com/2021/08/30/advanced-table-constraints/),
  [constraint-based tile generators](https://www.boristhebrave.com/2021/10/31/constraint-based-tile-generators/),
  [driven WFC](https://www.boristhebrave.com/2021/06/06/driven-wavefunctioncollapse/)). Infinite blocks:
  four layers offset on alternating axes, deterministic regardless of traversal order, "each layer 4
  block needs a total of 12 blocks from earlier layers evaluated", about 4× redundant work, and a
  fallback to the earlier layer on failure. Driven WFC: filtering domains before solving is "basically
  free". Generators: "robustness for Wave Function Collapse boils down to the tileset", and cell choice
  "is not usually an important decision" but "can have interesting effects on quality". **We use**
  the priors that driven WFC describes; the four-layer scheme was set aside for the parity schedule in
  [world.md](../architecture/world.md).
- **Boris the Brave, DeBroglie and Tessera** ([DeBroglie](https://github.com/BorisTheBrave/DeBroglie)).
  DeBroglie's path constraint is the model for our connectivity constraint; it documents backtracking
  as complete but slow and memory-hungry, "generally only appropriate for generating small arrays".
  Tessera ships a step limit that "will allow some backtracking to occur, but after a fixed amount of
  computation it will automatically retry with a fresh generation". Boris names our thrashing
  "stalling": the algorithm "doesn't know to backtrack out of it, instead repeatedly exploring
  variations of the partial solution". **We use** the step limit and retry: a region's attempts and
  step budget.
- **cuWaveFunctionCollapse** ([repository](https://github.com/Chocomunk/cuWaveFunctionCollapse)). A
  CUDA WFC measured 26 to 103 times slower than its CPU baseline.
- **A parallel CPU WFC from CMU.** Found that "the sequential queue-based algorithm outperformed all
  parallel implementations". With the entry above, the reason we expected no free win from the GPU and
  moved parallelism to whole chunks.

## Constraint propagation

- **Apt, "The Essence of Constraint Propagation"** ([arXiv:cs/9811024](https://arxiv.org/abs/cs/9811024)).
  Propagation is a monotone, inflationary fixpoint; the chaotic-iteration theorem makes the result
  independent of order provided no operator is "indefinitely neglected". **We rely on it:** the
  kernel's gather sweep reaches the same fixpoint whatever order its lanes run in, and the CPU
  reference reaches the same fixpoint as the kernel.
- **Kasif, arc consistency is P-complete** (Artificial Intelligence, 1990). "Inherently sequential in
  the worst case". **We rely on it:** parallelism is spent across chunks, not inside one propagation.
- **RTAC** ([arXiv:2407.11388](https://arxiv.org/abs/2407.11388)). The fixpoint needs only about 3.5 to
  4.8 whole-network sweeps even at density 1.0, against tens of thousands of AC-3 revisions. Supports
  sweep propagation over a worklist.
- **Walsh, "SAT v CSP"** (CP 2000). Unit propagation on the direct encoding is strictly weaker than
  arc consistency.
- **Gent, "Arc Consistency in SAT"** (ECAI 2002;
  [paper](https://sites.cs.st-andrews.ac.uk/people/ipg1/papers/ipgECAI.pdf)). Establishing arc
  consistency through SAT was up to about 30 times slower, and solving about 5 to 6 times slower,
  than doing it natively. With our arithmetic (the support encoding of a 24×24×8 grid is about
  171 million literals at 81 variants, against a 4.8 KB rule table), the reason SAT and CDCL
  encodings of adjacency are ruled out.
- **AC-4 support counters** (as in fast-wfc, marian42 and DeBroglie). Counters cost cells × 6 ×
  variants entries, about 16 million at 81 variants, and their proponent says they
  work poorly with backtracking. Ruled out at our variant count by arithmetic, not by measurement.
- **Berger, the domino problem** (1966). Undecidable, so no method with bounded lookahead can decide in
  general whether a partial tiling extends to an infinite one. Why a streamed world repairs rather
  than guarantees.
- **Communication-avoiding methods** (Demmel et al., IPDPS 2008; Carson, PhD thesis 2015; Meng and
  Skadron, ghost zones, IJPP 2011). A halo of width `r·t` works only when the per-step radius is
  bounded; otherwise "every processor needs all n rows". Why our halo is a margin that is solved and
  discarded, not a proof of independence.

## Search, backjumping and thrashing

- **Mackworth, "Consistency in networks of relations"** (1977). Defines thrashing by cause, one cause
  being a failure rediscovered over and over, which "cannot be removed by such minor palliatives as
  reordering the nodes."
- **Dechter and Frost, backjump-based backtracking.** Thrashing is "rediscovering the same
  inconsistencies and same partial successes during search." Proposition 4: "the latest variable in
  its jumpback set is the earliest variable to which it is safe to jump"; jumping later re-fails,
  jumping earlier can skip solutions (Propositions 2 and 4). Ginsberg and CDCL's assertive level pick
  the same target. **Agrees with us:** the per-collapse solver under-jumped and re-failed at one cell
  195 times out of 197.
- **Prosser, conflict-directed backjumping.** CBJ merges conflict sets on each jump
  (`conf-set[h] ← conf-set[h] ∪ conf-set[i] − {h}`), so depth comes from accumulation, unlike
  Gaschnig's backjumping, which jumps once and then steps back chronologically. On graph-based
  backjumping: "if `P` was dispensed with, or was reset whenever a successful forward move was made, we
  would again have an incomplete algorithm." **Tested:** the per-collapse solver had exactly that
  reset; removing it took the worst of 48 seeds from 197 backtracks to 11. The block kernel's undo
  keeps doubling until the search gets past the failing round.
- **Chen and van Beek** (JAIR 2001). "As the level of local consistency that is maintained in the
  backtracking search is increased, the less that backjumping will be an improvement"; they refute
  the stronger claim that it becomes useless. **Disagreed with us:** fixing our backjump was worth 24
  times on the bad seed.
- **Baker, "The Hazards of Fancy Backtracking"** (AAAI 1994). Dynamic backtracking with dynamic
  variable ordering is "worse by a factor exponential in the size of the problem", because "the
  effective search space itself becomes larger". Lowest-count selection is a dynamic ordering, so
  dynamic backtracking is ruled out.
- **Katsirelos and Bacchus, "Generalized NoGoods"** (AAAI 2005). For large-arity constraints "the
  resulting s-nogood will typically include all or almost all of the decision assignments", which "is
  the reason why nogood learning tends not to help GAC much". Why nogoods from a global constraint
  would be weak.
- **Lecoutre et al., "Nogood Recording from Restarts"**
  (IJCAI 2007; [paper](https://www.ijcai.org/Proceedings/07/Papers/019.pdf)). A fixed-cutoff restart is
  incomplete; an increasing cutoff or nogoods retained across restarts restore completeness. A
  reproduction we read measured the search tree down to 72% but "time is not saved … due to the
  overhead of nogood propagation" (our notes did not record which reproduction).
- **Boussemart et al., dom/wdeg** (ECAI 2004). Weight constraints by the dead ends they caused and
  prefer variables involved in them, explicitly against thrashing; the default in solvers such as
  Choco. In a head-to-head over 1064 CSP instances, dom/ddeg to dom/wdeg cut timeouts from 365 to 140
  and mean time from 125 s to 48 s, and adding restarts and nogoods moved 140 to 121 (our notes did
  not record which paper ran it). Untested here.
- **Gomes, Selman and Kautz, heavy-tailed run times.** Randomised rapid restarts "provably eliminate
  heavy-tails to the right of the median" and exploit "a non-negligible chance of very short runs";
  some domains show no heavy tail at all, which a survival-function plot over many runs tells apart.
- **Luby, Sinclair and Zuckerman, optimal restart cutoffs.** The optimal fixed cutoff when the
  run-time distribution is known, and a universal sequence when it is not.
- **Parallel Luby Restarts.** Each restart index paired with a deterministic seed; 88% efficiency on
  32 cores on Magic Square and super-linear speedups on heavy-tailed Quasigroup Completion instances,
  for solvers without nogood learning. **We use it:** a repair runs 32 seeds side by side and keeps the
  lowest that solves.
- **Luby, parallel maximal independent set.** Each node joins when its random key is a local minimum
  among its neighbours. **We use** the same rule for selection: every local minimum of (count, hash)
  within a radius collapses in one round.

## GPU and parallel solving

- **Atos** ([arXiv:2112.00132](https://arxiv.org/abs/2112.00132)). On small frontiers "fixed costs
  (the cost of the global synchronization barrier plus the kernel launch cost) dominate the overall
  processing cost"; persistent kernels pulling from a device queue give 3.44× geomean on BFS.
  **Tested:** our apparent fixed cost of 2.5 ms per dispatch was a cold-clock artifact; warm, a
  dispatch costs 0.099 ms, and the cost was per-cell work.
- **Gunrock.** 122 516 MTEPS on a scale-free graph against 85 MTEPS on a road network. A voxel grid is
  mesh-like and high-diameter, so frontiers stay small and iterations many.
- **Turbo** (AAAI-26). A fully GPU-resident constraint solver that uses "a propagation loop similar to
  AC1" and full recomputation instead of a worklist, because view-based propagators cause "uncoalesced
  memory accesses, load imbalance, thread divergence". Worse than OR-Tools on 58% of instances on an
  H100. Supports sweep propagation, and tempers the expectation of a GPU win.
- **Gent et al., "A Review of Literature on Parallel Constraint Solving"**
  (TPLP 2018; [arXiv:1803.10981](https://arxiv.org/abs/1803.10981)). "GPUs are not a silver bullet,
  and direct ports of existing algorithms to a GPU architecture often perform poorly."
- **CDCL on a GPU** (surveyed from Katsirelos et al., "Resolution and Parallelizability: Barriers to
  the Efficient Parallelization of SAT Solvers", [AAAI 2013](https://ojs.aaai.org/index.php/AAAI/article/view/8660);
  Osama, Wijs and Biere, ParaFROST, TACAS 2021; and related work). No competitive GPU CDCL loop was
  published as of the survey, one measured port ran 50 to 250 times slower than CPU MiniSat, and
  conflict analysis is described as "intrinsically serial". Evidence about the state of the art, not
  a proof of impossibility.
- **Prevot, Soos and Meel, GpuShareSat** (SAT 2021; [arXiv:2012.03119](https://arxiv.org/abs/2012.03119)).
  The one GPU-assisted SAT solver in the survey that beat a competition-winning baseline keeps every
  dependent decision on the CPU and uses the GPU as an asynchronous bulk service. The alternative we
  weighed before choosing a device-resident block kernel.
- **Mallob and MallobSat.** Clause sharing is worth about 15.6× on unsatisfiable instances and 4.1× on
  satisfiable ones, satisfiable scaling stalls past 96 cores, and instances a sequential solver
  finishes in under 1.4 s slow down. Cooperative efficiency is about 16% at 384 cores, so solving
  different problems beats cooperating on one; a nogood store should be bounded by volume and age
  entries out. Why our parallelism is across chunks and seeds.
- **Santi, Tardivo, Dovier and Formisano, GPU-accelerated Compact-Table** (arXiv 2507.18413). Mean
  speedups of 2.88 and 4.35 on two large-table families, with an RTX 4090 "roughly 45%" utilised
  because "the amount of work offloaded to the GPU is often not enough"; transfers reach 50 to 80% of
  kernel time, and a finer division of work "often results in performance degradation". Their unit of
  work is far larger than one WFC collapse, which is why a GPU needs whole chunks.
- **Xiao and Feng, inter-block GPU barriers.** Hand-rolled global barriers regress past about 18
  blocks even in CUDA.
- **WGSL** ([specification](https://www.w3.org/TR/WGSL/)). Only workgroup-scoped barriers, no
  device-side enqueue, and no forward-progress guarantee between workgroups. With the entry above, why
  a persistent kernel is not an option and a region lives inside one workgroup.
- **Jarzynski and Olano, "Hash Functions for GPU Rendering"** (JCGT 2020;
  [paper](https://jcgt.org/published/0009/03/02/)). The source of `pcg3d`, the stateless hash behind
  every choice.
- **Structure-of-arrays layouts in lattice Boltzmann codes.** About 5× on memory-bound neighbour
  sweeps. Untested here; our sweeps were not shown to be memory-bound.
