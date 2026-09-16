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
- **Boundary constraints** pin what lies outside the grid, which adjacency cannot see: our bottom
  layer is street level, the top layer is air, and the side borders ban paths that would point out of
  the grid. marian42 applies the same idea through a 1×H×1 wrap-around default column.

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
his rules (see the ranked findings in [status.md](status.md) and the notes below):

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

## Global constraints: the path constraint

When a global property must hold, the WFC-friendly way is a constraint that runs **between**
propagation steps, prunes possibilities and reports failure, following Boris the Brave's path
constraint for DeBroglie. `wfc-core/src/constraint.rs` implements this as
`ConnectivityConstraint`, behind the `GlobalConstraint` trait:

1. Build the graph of cells that could still connect, where an edge exists if *some* remaining tile
   in one cell links to *some* remaining tile in the other.
2. If cells that can only hold network tiles fall into different components, no later choice can join
   them: report a contradiction so the run restarts.
3. Cells outside the component that holds the network can never join it, so ban network tiles there.
4. Cells whose removal would split the network (articulation points, found with Tarjan's algorithm)
   must stay passable, so ban tiles that are not.

Because the graph only loses edges as possibilities shrink, a fully collapsed grid that passes every
step is connected. The solver applies it after every propagation, uploads what it changed and
propagates again until it changes nothing (`GpuAccelerator::with_global_constraint`).

**Why it needs backtracking.** A constraint that prunes hard turns unlikely layouts into
contradictions. With restart-on-failure, the fully constrained city never finished: 20 of 20 attempts
on an 8x8x5 grid ended in a contradiction, because every island the module set would have produced
becomes a failure. The fix is not to weaken the requirement but to recover from the failure: the
solver keeps the grid state before every collapse and, on any contradiction, undoes an exponentially
growing number of choices and forbids the choice it came back to (A-9, modelled on marian42's
history). The lesson generalises: **a global constraint is only as usable as the solver's ability to
take a choice back.**

**Undo the cause, not the most recent choice.** Plain chronological backtracking was not enough: it
solved one of three runs, and the other two burned 16000 iterations and ~11000 undos without
converging. A connectivity violation surfaces long after the choice that caused it, when some earlier
decision has already sealed a region off, so undoing the last 1-64 collapses usually retries the same
dead end. Failures therefore carry the cell where they surfaced, and the solver jumps back to the most
recent choice adjacent to it, falling back to the doubling step count when no such choice exists. That
change alone took the same test from one of three runs to three of three, in 5 to 136 seconds on an
8x8x5 grid.

**What it costs.** CPU work proportional to the grid on every observation; the grid on the CPU (which
today's run loop already needs, but a device-resident solver would not); and search, since each
violation costs the collapses that are undone. Run times vary by more than an order of magnitude (5
to 136 seconds for the same 8x8x5 city) because a run either walks into few conflicts or into many.
It is by far the heaviest workload we run, which also makes it a useful benchmark for the solver
redesign. The default city
test deliberately runs without it, so the module set is still measured on its own.

**When to reach for one:** a property that must hold every time (a guaranteed path from spawn to
exit), or one that design alone cannot approximate. For "usually connected", designing the module set
is cheaper and scales better.

## Alternatives we did not need

- **Post-processing repair:** generate, find the islands, patch them. Cheap, but the patch is not
  guaranteed to be legal by the rules.
- **Reject and retry:** keep generating until the metric passes. Simple, but the cost grows with grid
  size, and it hides how bad the rule set is.
- **Hierarchical generation:** lay out a connected skeleton (streets, stairs) first, then fill in with
  WFC constrained to it. The most promising route for large worlds, and a natural fit for the
  region-based solving in [roadmap.md](roadmap.md).

## See also

- [architecture.md](architecture.md) for where rules, weights and the solver live.
- [testing.md](testing.md) for the city test and its connectivity report.
- marian42, ["Infinite procedurally generated city with the Wave Function Collapse algorithm"](https://marian42.de/article/wfc/).
- Boris the Brave, "Path constraints" and the [DeBroglie](https://github.com/BorisTheBrave/DeBroglie) constraint set.
