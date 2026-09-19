# Constraints: what WFC can and cannot express

Wave Function Collapse only ever compares **two neighbouring cells**. Everything it enforces has to
be expressible as "this tile may sit next to that tile along this axis". That is a surprisingly
strong language, and most of what looks like global structure in a generated world is really local
rules plus weights. It is also a hard limit: some properties are invisible to adjacency, and you
either design around them or enforce them outside the rules.

This page records what we learned building the city module set (`wfc-devtools/src/city.rs`), why the
rule set looks the way it does, and when a global constraint is worth its cost.

## What adjacency can express

A rule set is a set of allowed `(axis, tile, neighbour)` triples, so a rule can only talk about one
face. Writing those triples by hand does not scale, so modules describe their six faces with
**connectors** and the triples are derived (`wfc-rules/src/modules.rs`), the way marian42 does it.
That gives, for free:

- **Continuity.** A road face only fits another road face, so roads never stop mid-cell.
- **Support.** Vertical connectors say what may rest on what: our street-level modules have a
  `BEDROCK` underside that nothing provides, so they can only exist on the bottom layer, and a roof
  needs a building below it.
- **Orientation.** Rotated variants are generated per prototype and deduplicated, and oriented
  vertical connectors force a stair and the headroom above it to face the same way.
- **Local walkability.** A face can be marked walkable, and a face can *require* its opposite face to
  be walkable (marian42's `EnforceWalkableNeighbor`). This is the single most useful trick we found:
  it turns "a path may not end at a wall or in mid-air" into plain adjacency, checked by the solver
  for free rather than by a later pass.
- **Material boundaries.** Cell-centred modules share a face between two different materials (a
  facade and open air), so different connectors may be declared compatible (`ModuleSet::connect`).
  marian42 avoids this by putting modules on grid corners instead.

Two further tools sit next to the rules:

- **Weights** shape *how much* of each thing appears, not what is legal. Note they apply per rotated
  variant, so a prototype with four rotations weighs four times its number. Getting this wrong once
  filled our sky with walkways.
- **Boundary constraints** pin what lies outside the world, which adjacency cannot see: our bottom
  layer is street level, the top layer is air, and the side borders of a bounded world ban paths that
  would point out of it. They are expressed as a `Prior` (`city_prior` in `wfc-devtools/src/city.rs`),
  which is also how a layer below WFC will say "this area is water". marian42 applies the same idea
  through a 1×H×1 wrap-around default column.

## What adjacency cannot express

Anything about a **whole configuration** rather than a pair:

- **Connectivity:** "every walkable cell can be reached from every other". A closed ring of walkways
  in mid-air is locally perfect everywhere. So is a bridge between two buildings nobody can enter.
- **Counting:** "at least one door per building", "exactly three towers".
- **Distance and shape:** "no dead end longer than three cells", "squares are at least 3×3".

The honest way to state the limit: adjacency constrains the *local* neighbourhood, so any property
that survives every local check but fails globally needs something else.

## Designing for connectivity instead of enforcing it

marian42's city is the reference for "you can walk everywhere", and it has **no** global constraint:
`EnforceWalkway` in his code is dead. We verified where the connectivity comes from by re-simulating
his rules, in the order the findings ranked:

1. **Buildings you can walk through.** 38 interior modules connect to the outside through door and
   tunnel connectors. Solid mass drops to about 9% of cells, and the largest connected walkable group
   goes from 0.14 to 0.61 of walkable cells. This dominates everything else.
2. **Multi-cell pieces that cannot be broken.** Stairs and bridges span two slots joined by
   connectors nothing else uses, with the walkable-neighbour rule on every exit.
3. **The walking surface lives in the open cell** above invisible solid blocks, so walkable area
   follows the top of the solid mass and is contiguous by construction.
4. **A short column** (6 layers) and boundary constraints top and bottom.
5. **Weights** that keep dead-end-prone pieces (bridges, tunnels) rare.

Our city keeps buildings solid, so it uses arcades and upper-storey passages for point 1, stairs with
headroom for point 2, walkable flat roofs with railings for point 3, and the same 6-layer column.
`city::disconnected_walkable_cells` measures the result: the share of walkable cells in the largest
network, currently about 0.3 to 0.85 per run. marian42's own rules score 0.6 to 0.8 with hollow
buildings, and 0.14 to 0.32 with them removed.

**Rules of thumb we ended up with:**

- Prefer a connector split over a later check. If a piece must meet something specific, say it in the
  faces.
- Anchor things that can otherwise float. A stair that may start in mid-air will start in mid-air.
- Every "surface" needs an edge treatment (railings) so that its border is legal without being
  walkable, or the surface will either sprawl or contradict.
- Measure the global property you care about even when you do not enforce it. A number per run turns
  "looks disconnected" into a comparison.

## Future capabilities: global constraints and statistical rules

Neither is in the library today. Both were built against the old per-collapse solver, measured, and
left out of the chunk solver so that the first Godot and Bevy integrations have less to carry. What
follows is what they mean, what they cost when we ran them, and what the chunk kernel would need to
host them, so that bringing either back starts from evidence rather than from scratch.

### A global constraint (the path constraint)

When a global property must hold, the WFC-friendly way is a constraint that runs **between**
propagation steps, prunes possibilities and reports failure, following Boris the Brave's path
constraint for DeBroglie. We implemented it as a `ConnectivityConstraint` behind a `GlobalConstraint`
trait:

1. Build the graph of cells that could still connect, where an edge exists if *some* remaining tile
   in one cell links to *some* remaining tile in the other.
2. If cells that can only hold network tiles fall into different components, no later choice can join
   them: report a contradiction so the run recovers.
3. Cells outside the component that holds the network can never join it, so ban network tiles there.
4. Cells whose removal would split the network (articulation points, found with Tarjan's algorithm)
   must stay passable, so ban tiles that are not.

Because the graph only loses edges as possibilities shrink, a fully collapsed grid that passes every
step is connected. The solver applied it after every propagation, uploaded what it changed and
propagated again until it changed nothing.

**It only works with backtracking.** A constraint that prunes hard turns unlikely layouts into
contradictions. With restart-on-failure, the fully constrained city never finished: 20 of 20 attempts
on an 8x8x5 grid ended in a contradiction, because every island the module set would have produced
becomes a failure. The fix was not to weaken the requirement but to recover from the failure. The
lesson generalises: **a global constraint is only as usable as the solver's ability to take a choice
back.**

**Undo the cause, not the most recent choice.** Plain chronological backtracking was not enough: it
solved one of three runs, and the other two burned 16000 iterations and ~11000 undos without
converging. A connectivity violation surfaces long after the choice that caused it, when some earlier
decision has already sealed a region off, so undoing the last 1-64 collapses usually retries the same
dead end. Once failures carried the cell where they surfaced and the solver jumped back to the most
recent choice adjacent to it, the same test went from one of three runs to three of three, in 5 to
136 seconds on an 8x8x5 grid.

**What it cost.** CPU work proportional to the grid on every observation, the grid on the CPU, and
search, since each violation costs the collapses that are undone. Run times varied by more than an
order of magnitude for the same 8x8x5 city, because a run either walks into few conflicts or into
many. It was by far the heaviest workload we ran.

**What the chunk kernel would need.** The kernel solves a whole region inside one dispatch, so there
is no "between propagation steps" for a host pass to run in. Two shapes fit the design:

- A **host pass between dispatches**: solve a chunk, run the constraint over the result, and re-solve
  the chunk with what it banned added to the prior. That reuses the repair machinery (a chunk solved
  again with a tighter prior) and keeps the constraint on the CPU where Tarjan's algorithm belongs,
  at the cost of whole-chunk restarts instead of fine-grained undo.
- A **connectivity check inside the kernel**, over the region's own cells, as another reason to
  restore a checkpoint. Cheap enough only if the check is local: a flood fill per round over a whole
  region would dwarf the propagation it sits between.

Either way the property is only global within a chunk. Connectivity *across* chunks is a different
problem, and the honest answer for a streamed world is a skeleton laid out before WFC runs (see
"Alternatives we did not need" below, and Phase 2 in [architecture.md](architecture.md#7-phase-2-layered-generation-design-constraints-to-keep-in-mind-now)).

**When to reach for one:** a property that must hold every time (a guaranteed path from spawn to
exit), or one that design alone cannot approximate. For "usually connected", designing the module set
is cheaper and scales better.

### A statistical rule (cell-aware weights)

Every rule kind above removes possibilities. A statistical rule ("a shop becomes more likely the more
shops are nearby, and nearer ones count for more") removes none: it shifts probability mass inside
the set of tiles that were already legal. That is why it cannot be a global constraint, whose `apply`
may only clear bits; a likelihood rule there could only return "I cleared nothing", and making it
prune to express a preference would change which outputs are valid, which is exactly what a rule
about likelihood must not do.

It hooks the collapse choice instead. We had it as a `TileWeighting` trait: the weight of a tile
became a function of the cell and its neighbourhood rather than of the tile alone, with an
implementation that added `strength / distance` for every decided attractor within a Chebyshev
radius.

**What the chunk kernel would need.** The kernel's choice is `hash % total` over a weight table in a
storage buffer, shared by every region of a batch. A neighbourhood-dependent weight means recomputing
the table per cell as its neighbours are decided, which is a scan per collapse in the middle of the
hot loop, so the shape that fits is **a weight table per chunk**, computed on the host from what the
neighbours already hold and uploaded with the batch. That expresses "more shops in this part of town"
at chunk granularity, which is the granularity a streamed world thinks in anyway, and it keeps the
kernel's inner loop untouched. Weights must stay integers ([architecture.md §3.2](architecture.md#32-tiles-and-compiled-rules)).

## Alternatives we did not need

- **Post-processing repair:** generate, find the islands, patch them. Cheap, but the patch is not
  guaranteed to be legal by the rules.
- **Reject and retry:** keep generating until the metric passes. Simple, but the cost grows with grid
  size, and it hides how bad the rule set is.
- **Hierarchical generation:** lay out a connected skeleton (streets, stairs) first, then fill in with
  WFC constrained to it. The most promising route for large worlds, and a natural fit for the
  chunk-based solving the library now does: a skeleton is a prior.

## See also

- [architecture.md](architecture.md) for where rules, weights and the solver live.
- [testing.md](testing.md) for the city test and its connectivity report.
- [thrashing.md](thrashing.md) for the measurements behind the recovery this page keeps referring to.
- marian42, ["Infinite procedurally generated city with the Wave Function Collapse algorithm"](https://marian42.de/article/wfc/).
- Boris the Brave, "Path constraints" and the [DeBroglie](https://github.com/BorisTheBrave/DeBroglie) constraint set.
