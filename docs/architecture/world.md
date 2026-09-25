# The streamed world

How solved regions become a world that is generated around a player, stays seamless, and comes out
the same whatever path the player took. This is `WorldGenerator` in `src/generator.rs` and the
schedule in `src/scheduler.rs`. The solver underneath is in [solver.md](solver.md); the numbers
behind each choice are in [measurements.md](../research/measurements.md).

## Chunks and the halo

A world is a lattice of chunks of one shape, inside a `WorldExtent` that may be unbounded along any
axis. A chunk is solved as a **region**: itself, widened by a *halo* of cells that is solved and
then thrown away.

**Why a halo:** a chunk solved with free faces can leave border tiles that no row of neighbours can
complete. Each tile on a free face only needs *some* neighbour, one at a time, while a neighbouring
chunk has to supply a whole consistent row. Solving one cell beyond the chunk and discarding it
proves that a completion exists without committing to it. It took the chunks that could not be
placed from 29 of 32 to 10 of 32 under a checkerboard order, and to 3 of 63 under a diagonal one. A
second cell of halo did not help further.

## The schedule

Two rules shape every batch, and between them they are the whole scheduler:

- **Chunks that share a face never ride in one dispatch.** Each would read the other's cells, so a
  batch takes one parity of the lattice: `(x + y + z) & 1`.
- **A chunk waits for the face neighbours it reads.** Parity 0 goes first and reads nothing; parity
  1 follows and reads all of its face neighbours.

The set of wanted chunks is **closed** under the face neighbours of its parity-1 members. Without
that, a chunk at the edge of the view would be solved against fewer fixed faces than the same chunk
in the middle, and its tiles would depend on where the player happened to stand.

**Only the first parity gets a halo.** Its halo is what leaves the neighbours room to complete its
borders. A second-parity chunk is solved after all its face neighbours, so a halo could only pin
their cells as they are; what it would add is the diagonal corner cells, squeezed between two fixed
neighbours, which fail chunks that would otherwise solve. Solving the second parity without a halo
cut repairs on the city from 280 to 176 over five worlds and solver time by a third; dropping the
first parity's halo instead multiplied repairs by four and left chunks unplaced.

## Repairs

A chunk whose batch returned a status other than `Solved` is **repaired**: solved again alone, with
its halo *released*, so it may rewrite the neighbouring cells the halo covers. If that fails, a
wider halo is tried, up to the widest that fits both the device's workgroup memory and the rule
below that keeps repairs of one class apart.

**A repair is a portfolio.** It solves the same region once per seed of `RepairPolicy::seeds` (32
by default) in one dispatch and keeps the lowest seed that solved, so the outcome depends only on the
configuration. A seed stops as soon as a lower one has solved, since it can no longer be the one
kept, so a repair costs about as long as the seeds up to its winner. The seeds are hashed from the
world seed, the halo and their position, all different from the first attempt's sequence.

**Why many seeds:** a chunk that exhausted its first attempt almost always has an arrangement; it
just was not found from that sequence of choices. A census that solved again the exact problem of
every chunk five streamed city worlds had given up on, with the halo released, found arrangements
for every one of them. The earlier conclusion, that such chunks needed a different module set,
came from one seed per repair and was wrong.

A chunk that still will not solve is reported once as `ChunkEvent::Failed` and left alone, because
solving it again would fail the same way.

### Repair classes

A repair rewrites cells of already solved neighbours, which on its own would make the world depend
on generation order: what a repair sees, and so what it writes, would depend on which neighbours
happened to exist. Two rules make it a pure function of the configuration.

- **A repair waits for everything it can see.** It runs only once every chunk within one of it
  (Chebyshev distance, diagonals included) has had its first attempt; the scheduler generates any
  that nobody asked for.
- **Repairs go by class.** A chunk's class is the parity of each coordinate,
  `(x&1) | (y&1)<<1 | (z&1)<<2`. Two chunks within one of each other are in different classes, so
  repairs of one class are at least two chunks apart. A repair also waits until every failed
  neighbour of a lower class has been repaired.

A repair's halo is kept below half a chunk (`2 * halo + 2 <= chunk size`, so at most 3 cells for
chunks of 8), so two repairs of one class neither read nor write each other's cells. What a repair
sees is then the same in any order, and so is what it writes.

Waiting has a reach. A second-parity chunk at the edge of a request waits for its diagonal
neighbours' first attempts and their face neighbours, one of which may itself wait for a repair that
has to see its own diagonals. So in a world one chunk tall, nothing an active repair needs lies more
than `REPAIR_REACH` (3) chunks beyond what was asked for, and those chunks are kept while an active
repair needs them. A repair is **active** if its chunk is wanted, or if an active repair waits on
it. A tick with nothing it can start while repairs are active is a scheduler bug and panics.

## What determinism means here

For a fixed rule set, prior and configuration:

- **The same requests give the same world** on any backend, any number of threads, and any solver
  invocation count. Every choice is a hash of the world seed, the chunk's coordinate and where the
  solve had got to ([solver.md](solver.md#seeds-and-hashing)).
- **A chunk evicted and asked for again comes back tile for tile, repairs included**, with nothing
  saved: a game walking away and back sees what it saw
  ([Regenerating exactly](#regenerating-exactly)). An 8×4-chunk city evicted beyond every cut across
  it and asked for again is the city generated once, and so is one walked along and back while
  evicting behind the focus (`wfc-devtools/tests/partial_eviction.rs`, seeds 8 and 11).
- **With repairs, the world is the same in any generation order.** A 4×4-chunk city comes out tile
  for tile the same generated all at once or chunk by chunk in either raster order, repairs included
  (`wfc-devtools/tests/order_diff.rs`, seeds 8 and 11). Every chunk a repair rewrote is reported as
  an `Updated` event and counted in `GeneratorStats`.
- **The same world on different GPUs.** A 4×4-chunk city recorded on an NVIDIA RTX 3070 through
  dozen is tile for tile the same on Mesa's lavapipe, which CI checks on every pull request
  (`wfc-devtools/tests/golden_world.rs`).

One limit remains: **worlds more than one chunk tall.** There, a first-parity repair also reaches
corner chunks of the other parity, which it does not wait for, because they may be waiting for it.

A rule set is **streaming-clean** when no chunk ever needs a repair; then the limit does not apply.
The city is not: about one chunk in ten is repaired.

## Regenerating exactly

Solving an evicted chunk again against the neighbours that stayed would fit them but change it. The
waits above hold every operation in a fresh world to one order around each chunk, and each reads its
neighbours at a fixed point of it: a first-parity first attempt reads nothing, a second-parity one
reads its face neighbours after the first-parity repairs, and a repair reads them after the first
attempts and the lower classes. A neighbour that stayed has moved on since, rewritten by repairs that
came later. So regeneration reads the past, in the manner of Boris the Brave's
[Infinite Modifying in Blocks](https://www.boristhebrave.com/2021/11/08/infinite-modifying-in-blocks/),
where every block is a fixed function of a bounded set of earlier layers.

- **Phases.** Around one chunk a fresh world runs first-parity first attempts, then first-parity
  repairs by class, then second-parity first attempts, then second-parity repairs by class. The
  scheduler's waits keep every operation within one chunk of another in that order.
- **Views.** The generator keeps, for each chunk in memory that a repair has rewritten, its tiles
  after each phase that wrote it. An operation reads every neighbour as it was before its own phase,
  so it reads what it read in a fresh world however much has been evicted and generated again since.
- **Writes by phase.** A repair writes the neighbours in its region that have not yet passed its
  phase. Repairs of one phase are two chunks apart and write different cells of a chunk between
  them, so those join one version. A neighbour that has passed the phase holds the write already,
  and what the repair decided has to equal it; if it did not, the model of the schedule would be
  wrong, and the generator panics rather than go on.
- **Replays.** A repair of a chunk that stayed rewrote the neighbours around it, and a neighbour
  generated again has lost that. So when a chunk enters memory, every repair of a chunk in memory
  within one of it that comes later in the order runs again, with the halo it placed its chunk
  with, and writes only what has not passed its phase; a second-parity first attempt waits for the
  replays on its face neighbours. A replay with nothing left to write is dropped. `GeneratorStats`
  counts them as `replayed`.

In a fresh world every view is the current tiles and no replay can be queued, since each repair
waits for the chunks around it: the golden world and the order-independence tests pass unchanged.
The pasts are kept only for chunks in memory and go with them. Generating an evicted part of the
city again costs about what generating it the first time did
([measurements.md](../research/measurements.md#regenerating-exactly-188-2026-09-25)). A chunk put
back with `import` is taken as it is, visible to every phase. The limit of worlds more than one
chunk tall stays.

**A frozen world** keeps instead. Given a store (`WorldGenerator::with_store`, the `FrozenStore`
interface frozen stages use, [packs.md](../reference/packs.md#persistence-and-saves)), a world puts
every chunk it evicts in the store as it is, and a chunk it needs that the store has comes back from
it, taken as it is, rather than being generated. So what a player saw survives a change of the rule
set or the prior, as long as the tile indices mean the same tiles; a chunk never generated before is
generated against the stored ones like any other. The store is asked about a chunk once while it
stays needed. A store that fails keeps the chunk in memory, and the next `tick` or `poll` reports the
failure. `GeneratorStats` counts the chunks it brought back as `restored`
(`wfc-devtools/tests/frozen_world.rs`).

## Streaming

`WorldGenerator::request` takes focus points (a chunk and a radius). `tick` starts the next batch:
the nearest eligible chunks of one parity or, when nothing else can start, the lowest-class repair
whose neighbourhood is complete. `poll` commits whatever has finished and returns `ChunkEvent`s
(`Updated`, `Failed`, and `Evicted` from a `Worker`); `wait` and `run_until_idle` block instead, for tests and tools.
`evict_outside` drops chunks beyond a margin, keeping what active repairs need, and hands them back;
asked for again they come back the same, so a game need not keep them. `import` puts a chunk back as
it is.

A `Worker` runs the same generator on a thread of its own for engines that cannot poll
([overview.md](overview.md#threads)).

This keeps well ahead of a walking player: a 24×8-chunk city generated in front of a focus walking at
1.4 m/s fits the 0.5 s tick with room to spare, and a run through an unbounded city in wall-clock
time never had a chunk near the player still generating
([measurements.md](../research/measurements.md) has the current figures and their protocol).

## Memory

Memory is bounded by what is requested plus `REPAIR_REACH`, not by what was ever generated.
`evict_outside` is the game's lever, and the chunk store holds only tiles, one `u16` per cell. The
game-session test asserts the bound over a whole walk
([testing.md](../guides/testing.md)).
