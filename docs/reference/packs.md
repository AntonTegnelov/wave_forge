# Packs

The pack format and the stage runtime as they are built, in `src/stages/` and `src/towns.rs`. The
design they are a first part of, and the stage kinds still to come, are in
[stages.md](../architecture/stages.md). This page changes in the same pull request as the code it
describes.

## A pack file

A pack is RON, conventionally `*.world.ron`: a version, a list of named stages and, optionally, a
list of named tables of facts ([below](#tables-of-facts)). A stage reads
other stages by name, and how far it reads follows from its parameters, so a pack never states a
reach by hand. The order of the list is the order a stack view shows; a stage may read any other,
earlier or later.

```ron
(
    version: 1,
    stages: [
        (name: "hills", kind: Field(Mul(Noise(frequency: 0.012, octaves: 4), Constant(48.0)))),
        (name: "ground", kind: Blur(input: "hills", radius: 2)),
        (name: "towns", kind: Sites(height: "ground", region: 6, size: (2, 3), chance: 0.8)),
        (name: "level", kind: Flatten(height: "ground", sites: "towns", blend: 8)),
        (name: "city", kind: Solve(sites: "towns", rules: "city",
            bottom: Some(Tagged("street_level")), top: Some(Named("air")))),
        (name: "trees", kind: Scatter(kind: "tree", height: "level", spacing: 3, chance: 0.9,
            between: Some((6.0, 40.0)), max_slope: Some(1.2), avoid: Some(("towns", 4)), apart: 3)),
    ],
)
```

That is the valley test pack, `examples/valley.world.ron`: rolling ground, towns on levelled sites
built from the city module set, and trees on the ground between them.

## Loading

`Pack::parse(text)` reads a pack, and `Pack::from_file(PackFile)` checks one built in code. Loading
refuses, each time with a `PackError` that names the stage:

| Error | When |
|---|---|
| `Syntax` | the text is not a pack (RON errors, unknown fields) |
| `Version` | the version is not `PACK_VERSION` (1); there are no migrations yet |
| `DuplicateName` | two stages share a name |
| `UnknownInput` | a stage reads a name no stage has |
| `Invalid` | a parameter is out of range, or a stage reads an input of the wrong type (a field where it needs sites, say) |
| `Cycle` | stages read each other in a cycle |
| `DuplicateTable` | two tables share a name |
| `InvalidTable` | a table's parameter or expression is wrong, or its parents lead back to it |

`Pack::reach(target, chunk_size)` reports, for every stage `target` depends on, how many WFC cells
beyond a column of `target` it has to be generated: the largest sum of reaches along any path, each
in its reader's columns times the reader's scale, plus one column of an input read between its
columns. `Pack::scale(stage)` gives a stage's scale.
`Pack::stage_names` and `Pack::kind` describe the stages to a tool or an engine.

## World bound

A pack may give the world an edge:

```ron
bound: Some(Disk(centre: (0.0, 0.0), radius: 110.0)),
```

or `Some(Rect(min: (x, y), max: (x, y)))`, in cells. The runtime never asks for a chunk of a target
stage that lies wholly outside the bound, so a finite world costs nothing beyond its edge, and an
engine draws its own sea there; a target chunk inside the bound still reads its inputs within its
reach, beyond the edge too. A chunk meets the bound when any of its area, up to its far edge, lies
inside. The pack shapes the coast itself, falling into the sea before the edge as the ring world
does. Loading refuses a bound that holds nothing.

`Runtime::request_bound(targets)` asks for the targets in every chunk the bound meets: how a finite
world computes its region jobs and location tables before play. A bounded world keeps every region
and location table it has computed, since it has few; without a bound it fails with
`StageError::Unbounded`. `Pack::bound` gives the bound to an engine.

## Water

A pack may declare the water it shares:

```ron
water: Some((level: 0.05)),
```

`level` is the sea's height in cells of height: below it the ground is under water. With
`lakes: Some("lakes")` it also names a [Lakes](#lakes) stage, whose surface raises the water inland.
Every stage that reads water reads this one, so they agree: a Scatter stage's `water` measures depth
below the sea or a lake, whichever is higher, and rivers run down to the sea or into a lake.
`Pack::water` gives it to an engine, which draws its water there. Loading refuses a level that is
not a number, lakes that name no Lakes stage, and a Scatter stage's `water`, a Rivers stage or a
Lakes stage in a pack that declares no water.

## Parameters

`params: {"land": (default: 0.45, range: (0.0, 1.0))}` declares numbers a user tunes without
editing the pack, which stages read with `Param("land")`: a preset's handful of sliders, say
(docs/architecture/stages.md, "Four tiers over one pack"). The default has to lie in the range, and
the range is what the pack promises gives a sound world: `tests/presets.rs` sweeps every preset's
ranges over a grid of values and seeds. A stage may read only parameters the pack declares.

`Runtime::set_params(&values)` sets the parameters named in `values`, the others keeping theirs,
and drops every product that read one whose value changed, with what was generated from it, as
`request` does; `Runtime::params()` gives the values now. A name the pack does not declare, or a
value outside its range, fails with `StageError::Param` and changes nothing.

`examples/presets/islands.world.ron` is the first preset: islands whose `land` runs from a sea of
islets to land with a few lakes, whose `roughness` runs from gentle to rugged, and whose `trees` run
from bare to dense woods.

## Edits

What a player changes in a world is a log, `Edits`, which a game keeps and saves beside its facts:
a world is a function of the pack, the seed, the facts and the edits.

- `Edit::Remove { point, at }` takes away a point of a Scatter, Embed, Deposit or Spawn stage, a
  felled tree say, by its positional id (`PointId`, from its `InstanceId`) and where it stood.
- `Edit::Move { point, from, to, turn }` stands it at `to`, turned to `turn`. It stays in the chunk
  it was generated in, so an engine finds it there wherever it now stands.
- `Edit::Raise { stage, column, by }` adds `by` to a field stage's value at one of its columns:
  ground raised or dug. Raises of one column add up, and a column is shared by the chunks on either
  side of a border, so they never disagree.
- `Edit::Dig { stage, at, radius }` digs a ball of `radius` cells out of a [Volume](#volume) or
  [Carve](#carve) stage around `at`, in cells along the lattice's x and y and up, as a Carve stage
  carves a room: a voxel within a cell of the ball, or inside it, keeps the lower of its value and
  how far outside the ball it is. `Edit::Fill { stage, at, radius }` fills one in, each such voxel
  keeping the higher of its value and how far inside the ball it is. Digs and fills apply in the
  log's order, so a fill after a dig closes the hole and a dig after a fill opens it.

`Runtime::set_edits(&edits)` gives a runtime the log, and every product is edited as it is
generated: a field's columns raised, a volume's voxels dug and filled, a Scatter stage's points
removed or moved. So an edit survives
eviction and regeneration, and a sample holds a raise as its chunk does. A new log dirties only the
chunks whose edits changed, a raise its column's chunk, a point the chunk it stood in and a dig or
fill the chunks within a cell of its ball, every later dig and fill's too since their order counts,
and
everything that reads them within its reach, then returns the drops as `request` does. Staleness is
kept per chunk, so a raise regenerates a reader's neighbouring chunks when their reach covers the
raised chunk, even where it does not cover the column. An edit that raises a stage that is no
field, digs or fills one that is no volume, has a ball without a positive radius, or names a point
no point stage placed, fails with `StageError::Edit` and changes nothing.
`Edits::to_ron` and `from_ron` save and load the log.

`stages::brushes::stroke(&runtime, &brush, &path)` gives the edits a brush makes dragged along
`path`, points in cells, so an editor's brush and a player's tool paint the same way in every
engine; the caller appends them to the log, and undoes a stroke by taking them out again.

| Brush | Edits |
|---|---|
| `Raise { stage, radius, strength }` | a raise of every column of a field stage within `radius` of the path by `strength` times how near it lies, 1 on the path falling smoothly to 0 at the radius; a negative strength lowers |
| `Smooth { stage, radius, strength }` | a raise of each such column toward the average of the nine around it, by `strength` (0 to 1) times how near it lies, read with `Runtime::sample` |
| `Dig { stage, radius }`, `Fill { stage, radius }` | balls of `radius` dug out of or filled into a volume along the path, half a radius apart |
| `Remove { stages, radius }` | the removal of every point of the point stages `stages` within `radius` of the path, in the chunks the runtime holds |

A stroke without a path, a radius that is not positive, a smoothing strength outside 0 to 1, or a
stage of the wrong kind fails with `StageError::Edit`.

## Persistence and saves

A stage may declare how a save keeps it, with `persist`:

- `Pure`, the default: regenerated from the pack whenever it is needed, with the player's edits
  replayed on it.
- `Frozen`: each chunk is kept as it was first generated, before edits, and reused from then on,
  even after the pack changes. Edits still apply to it. Locations a player has seen, say.
- `Ephemeral`: regenerated like a pure stage, but its edits are never saved, so cut grass grows
  back after a load. Clutter, say.

```ron
(name: "locations", persist: Frozen, kind: Locations(...)),
(name: "grass", persist: Ephemeral, kind: Scatter(kind: "grass", height: "terrain", spacing: 2)),
```

`Runtime::save()` gives a `Save`: the Wave Forge version that made it (`generator`), the pack's
digest (`pack`, from `Pack::digest`, a hash of the pack as it was read), the edits less those of
ephemeral stages, and every frozen chunk the runtime has generated. `Save::to_ron` and `from_ron`
write and read it. `Runtime::load(&save)` brings a world back: it sets the save's edits and puts
the frozen chunks of every stage this pack freezes in place, even if the pack has changed since, so
a frozen chunk never generated before follows the new pack and one generated before stays as it
was. A frozen chunk of a stage the pack no longer has, or no longer freezes, is left out. It
returns the drops as `request` does, and fails as `set_edits` does, changing nothing. A game
compares `generator` and `pack` with its own to decide what to tell a player; the runtime never
refuses a save for them. The facts are the game's to save beside it.

Without a store, a runtime holds every frozen chunk it has generated for as long as it lives, and a
save holds them all, so freezing suits stages with few products, like locations. With one,
`Runtime::with_store(store)`, a frozen chunk the request no longer needs leaves memory for the store
and comes back from it, unchanged, when a request needs it again, so a walk across an infinite world
holds a bounded number of them (`tests/frozen_store.rs`). A store is anything that implements
`FrozenStore`, which keeps bytes by stage name and chunk; `DirectoryStore::new(path)` keeps a file
per chunk under a directory. A save then holds only the frozen chunks in memory, and the game keeps
the store with its saves; a runtime of a changed pack given the same store reads a stored chunk as
it was first generated.

## Levels

A stage may declare a `scale`: how many WFC cells one of its columns spans along each axis, 1 by
default. `(name: "biome", scale: 8, kind: Rules(...))` makes a coarse stage, a world map's say. Its
chunks have as many columns as any other stage's, so each covers `scale` times as much ground, and a
coarse world needs few of them. A coarse stage's chunks are in its own lattice: `field("biome",
chunk)` takes the coarse chunk's coordinate.

Data flows only from coarse to fine. A stage reads stages as coarse as itself or coarser, by a whole
factor, and loading refuses anything else. A fine stage reads a coarser field between its columns,
linearly from the four around its column's centre, and a coarser category from the column its own
lies in. Positions in expressions (`X`, `Y`, `Distance`, `Angle`, and noise) are in WFC cells at
every scale, so a formula means the same at any scale. Field, Blur, Rules, Region and Lakes stages
can be coarse; Sites, Flatten, Solve and Scatter work on the WFC lattice, at scale 1, and may read coarser
fields.

## Stages

## Tables of facts

A table holds rows, each with an id and a value per column. Stages read tables; tables never read
stages. A **given** table's rows come from the game at run time, a history it simulated say. A
**generated** table's rows are computed from the seed, once per row of its parent table or once in
all, which describes hierarchies like a galaxy's sectors, systems and bodies.

```ron
tables: [
    (name: "villages", kind: Given(columns: [("population", Number), ("culture", Names(["river", "hill"]))])),
    (name: "sectors", kind: Generated(count: Constant(64.0), columns: [("mass", Floor(Random(0.0, 100000.0)))])),
    (name: "systems", kind: Generated(parent: Some("sectors"), count: Floor(Random(1.0, 12.0)), columns: [
        ("mass", Share("mass")),
        ("radius", Random(4.0, 40.0)),
    ])),
],
```

- A given column holds a `Number` or one of a list of `Names`, which expressions read as its index.
- A generated table's `count` and columns are expressions. They read what a stage's expressions
  cannot, and nothing a stage's can:

| Expression | Value | Where |
|---|---|---|
| `Parent("column")` | the parent row's value | count and columns |
| `Random(low, high)` | a number from `low` up to `high`, from the row's hash stream and its column's; every `Random` of one column draws the same number, spread over its own range | count and columns |
| `Index`, `Count` | the row's place among its parent's children, from 0, and how many there are | columns |
| `Share("column")` | the row's part of the parent's whole-number value, split by hashed weights into whole numbers that add up to it exactly | columns |

- The count is rounded down and must lie from 0 to `MAX_CHILDREN` (65 536), and a shared value
  must be a whole number from 0 to `MAX_SHARED` (2^24, where every whole number is exact in an
  `f32`). Either failing is a `StageError::Table` naming the table and the parent row.
- **Ids.** A given row's `RowId` is the game's own id; a generated row's is its parent's id followed
  by its index, or its index alone without a parent. Adding a given row, or changing one parent's
  children, never changes another row's id or values.
- A position is two number columns, and a stage that reads positions names them (TableSites'
  `at`). Curves in tables are not built yet ([#98](https://github.com/AntonTegnelov/wave_forge/issues/98)).

`Facts::new(pack, seed)` computes every generated table, with the given ones empty.
`facts.give(table, rows)` replaces a given table's rows, each a `GivenRow` of the game's id and a
`Value` per column, and recomputes every table below it; it refuses a missing or unknown column, a
number that is not finite, a name not in its column's list, two rows of one id and a generated
table, and changes nothing when it does. `facts.table(name)` reads a table's rows in the order of
their ids. `Facts` is cheap to clone: a game keeps its own and hands a copy to each runtime.

A stage reads a table through the row its runtime is focused on: `Row("systems", "radius")` in a
Field expression or a Rules condition is that row's value. `runtime.set_facts(facts)` gives a
runtime its facts, and `runtime.focus("systems", id)` focuses it on a row, which is how one surface
pack serves every planet. Both drop what the change makes stale and return it as `request` does,
and the request generates it again. A row added, removed or changed stales the chunks its site
covers, before and after, in a TableSites stage; a focused row whose values changed stales every
chunk of the stages that read it. A reader's chunk is stale when its reach covers a stale chunk of
an input, and a town or a region goes with any stale chunk it covers, so burning one village
solves its town again and nothing else.
A stage reading a row with none focused fails with `StageError::NoFocus`. A table's rows become
sites through a [TableSites](#tablesites) stage, and a town's rule set can follow a names column of
its row ([Solve](#solve)). Roads from a table's curves are not built yet
([#98](https://github.com/AntonTegnelov/wave_forge/issues/98)).

## Stages

Every stage works on a two-dimensional lattice of cell columns at its scale; a Volume stage gives
each column levels too. Every stage produces one of eight types (a field, a volume, categories,
sites, tiles, points, curves or stamps), and loading refuses a stage that reads one type as another.

| Kind | Produces | Reads, and how far |
|---|---|---|
| `Field` | Field | the fields named by `Input` and the categories named by `Is`, 0 cells; the categories a `Match` names, `blend` cells |
| `Volume` | Volume | as `Field` |
| `Carve` | Volume | a volume, 0 cells; a curves stage and, for curves on the ground plane, a height field, `max_radius + 1` cells; an Assemble or Cave stage, 1 cell; a sites stage or an Assemble stage to level, 1 cell |
| `Top` | Field | a volume of its own scale, 0 cells |
| `Aquifer` | Volume | a volume, and what its materials read, 0 cells |
| `Rules` | Categories | what its conditions read, 0 cells |
| `Nearest` | Categories | what its climate reads, 0 cells |
| `Blur` | Field | one field, `radius` cells |
| `Delta` | Field | one field, `radius` cells |
| `Area` | Categories | one Rules stage, `distance` cells |
| `Sites` | Sites | a height field, `region` chunks |
| `TableSites` | Sites | a table's rows; a height field, `max_size` chunks |
| `Locations` | Sites | a height field, and what its kinds' conditions read, `region` chunks |
| `TableCurves` | Curves | a table's rows |
| `Rivers` | Curves | a height field, `region - 1` chunks |
| `Network` | Curves | a height field and a sites stage, `region - 1` chunks |
| `Apply` | Field | a height field and a Region or TableCurves stage, `max_radius + blend` cells |
| `Flatten` | Field | a height field, 0 cells; a sites stage or an Assemble stage, `blend` cells |
| `Solve` | Tiles | a Sites or TableSites stage, 0 cells |
| `Assemble` | Stamps | a sites stage, 0 cells |
| `Cave` | Stamps | nothing |
| `Tunnels` | Curves | a Cave stage, 0 cells |
| `Deposit` | Points | a Cave stage, 0 cells; a volume, `region - 1` chunks |
| `Spawn` | Points | a Cave stage, 0 cells |
| `Scatter` | Points | a height field, `apart` cells (one more with `max_slope`); a sites stage or an Assemble stage, `apart + margin` cells |
| `Embed` | Points | a volume, and what its conditions read, 0 cells |

### Field

`Field(expr)`: a value per cell column, from an expression at that column alone. Coordinates are
in cells, measured from the world's origin to the column's centre, along the lattice's x and y.

| Expression | Value |
|---|---|
| `Constant(v)` | `v` |
| `Noise(frequency: f, octaves: n)` | fractal value noise in 0..1: `n` layers (1 to 16), the first with `f` lattice points per cell, each next at twice the frequency and half the weight |
| `Noise(frequency: f, octaves: n, name: "hills")` | the same, from the stream named `hills`: identical in every stage that names it |
| `FastNoise("hills")` | the noise the pack's `noises` names `hills`, as Godot's `FastNoiseLite.get_noise_2d` gives it at the column's centre ([below](#godots-noise)) |
| `Input("name")` | another field's value at the same column |
| `X`, `Y` | the column's centre |
| `Distance((x, y))` | the distance from a point to the column's centre |
| `Angle((x, y))` | the direction from a point to the column's centre, as a fraction of a turn from +x towards +y, in 0..1 |
| `Add(a, b)`, `Sub(a, b)`, `Mul(a, b)`, `Min(a, b)`, `Max(a, b)` | the arithmetic |
| `Abs(a)`, `Floor(a)`, `Sin(a)` | the absolute value, the largest whole number not above it, the sine of an angle in radians |
| `Is("stage", ["name", ...])` | 1 where a Rules stage's category is one of the names, 0 elsewhere |
| `Match(input: "stage", cases: [("name", expr), ...], otherwise: expr, blend: b)` | one expression per category of a Rules stage, blended where categories meet (below) |
| `Clamp(a, low, high)` | `a` held between the bounds |
| `Smoothstep(low, high, a)` | 0 at or below `low`, 1 at or above `high`, and a smooth step between |
| `Remap(a, (from_low, from_high), (to_low, to_high))` | `a` mapped linearly from one range onto the other, not clamped |
| `Curve(a, [(x, y), ...])` | a piecewise-linear curve through points in increasing x, level beyond its ends |
| `Row("table", "column")` | a column of the row the runtime is focused on in a table ([Tables of facts](#tables-of-facts)) |
| `Param("name")` | the value of one of the pack's parameters ([Parameters](#parameters)) |
| `Select(when: Less(a, b), then: c, otherwise: d)` | `c` where `a < b`, else `d`; `Greater(a, b)` compares the other way, and `Between(a, low, high)` holds where `a` is in the range, both ends included |

An unnamed `Noise` draws from its stage's own stream, keyed by the world seed, the stage's name and
the octave, so every unnamed noise of one stage is the same function; name them to make them
independent. Loading refuses a curve of fewer than two points or with x not increasing, a clamp
whose low bound is above its high one, a smoothstep with equal edges, a remap from a range of one
value, and numbers that are not finite.

`examples/rings.world.ron` puts these together: an island's height from products of named noises,
flattened near the centre with a smoothstep over the distance from it and a rim noise, and falling
into the sea past its edge with a clamped remap of that distance.

A `Match` weighs every category within `blend` cells of the column (at most `MAX_BLEND`, 32) by a
tent, `(b + 1 - |dx|) * (b + 1 - |dy|)`, and blends the expressions of the categories it finds by
those weights; a category no case names takes `otherwise`. Moving one column moves at most
`2 / (b + 1)` of the weight, so a step between neighbouring columns is at most the largest step of
any case plus the spread between the cases divided by `b + 1`, across chunk seams too. With a blend
of 0 each column takes its own category's expression. Loading refuses a blend over the limit, two
cases for one category, and a case for a category the rules do not give. The ring world's `terrain`
stage shapes each biome's ground from the island's height this way.

### Godot's noise

A pack may name noises configured as Godot's `FastNoiseLite` resource is, and Field and Rules
expressions and Scatter conditions read them with `FastNoise(name)`:

```ron
noises: {
    "hills": (noise_type: Perlin, seed: 77, frequency: 0.03, fractal_octaves: 3),
},
```

A noise's properties are the resource's, under the same names and with the same defaults, and any
left out takes Godot's default: `noise_type` (`Simplex`, `SimplexSmooth`, `Cellular`, `Perlin`,
`ValueCubic`, `Value`), `seed`, `frequency`, `offset`, the `fractal_` properties, the `cellular_`
properties and the `domain_warp_` properties. `wave_forge::noise::NoiseConfig::sample` and
`sample_3d` are a port of FastNoiseLite 1.1.0, the version Godot bundles, and give exactly what Godot
4.7's `get_noise_2d` and `get_noise_3d` give: `tests/fastnoise.rs` compares 2 560 samples of each
over 80 configurations that Godot computed, with no tolerance. A noise keeps its own seed, as a resource does, so the world's seed does not change it.

`Runtime::with_noise(name, config)` replaces a named noise, which is how an engine hands in a
resource; the Godot node's `noises` does it ([godot.md](godot.md#waveforgestages)). Loading refuses
a `FastNoise` of a name the pack's `noises` does not have, and `with_noise` of one fails with
`StageError::UnknownNoise`. A Field reads a noise in 2D and a [Volume](#volume) in 3D.

### Volume

`Volume(density: expr, bottom: b, top: t)`: a value per voxel, from an expression at that voxel
alone, for overhangs and caves ([#71](https://github.com/AntonTegnelov/wave_forge/issues/71)). Each
column has levels from `b` up to but not including `t`, and a level is as tall as a column is wide,
so at `scale: 4` a voxel is a cube of four cells a side. The expression is a [Field](#field)'s, with
one more leaf and one difference:

| Expression | Value in a volume |
|---|---|
| `Z` | the height of the voxel's centre in cells, `(level + 0.5) × scale`; loading refuses it anywhere else |
| `FastNoise("name")` | Godot's `get_noise_3d` at the voxel's centre, with the lattice's x as Godot's x, the height as Godot's y and the lattice's y as Godot's z, as a Godot game samples the same resource at that point of its Y-up world |

Every other leaf reads the voxel's column: `Noise` stays the library's 2D value noise, and `Input`
reads a field at the column, so `Sub(Input("height"), Z)` is a volume that is solid below a height
field and empty above it. By convention a voxel whose value is above zero is solid.

`Volume(..., materials: Some((rules: [(category: "grass", when: [...])], otherwise: "stone")))`
gives every voxel a material as well: the first rule whose conditions all hold at the voxel, or
`otherwise`, read as a [Rules](#rules) stage reads a column but with `Z` too, so
`Greater(Z, Sub(Input("height"), Constant(1.0)))` is the top voxel of ground under a height field.
The materials are the stage's categories, in the order a Rules stage's are.
`Runtime::volume(stage, chunk)` gives a chunk's `Volume`: its `size` in columns and levels, its
`bottom` level, its values level by level, each row by row with x fastest, and its `materials` in
the same order, empty without materials. [Carve](#carve), [Top](#top), [Aquifer](#aquifer) and
[Embed](#embed) stages read a volume; nothing else does.

`wave_forge::volume_mesh(chunk, volume, voxel_size)` gives a chunk's surface, where its values
cross zero, as one `VolumeMesh` for drawing and for a trimesh collider alike, since a height field
cannot hold an overhang. `voxel_size` is a cell's size times the stage's scale. It is a naive surface
net: a vertex in every cube of eight voxel centres that the surface passes through, and a quad for
every edge between two voxels it crosses, facing from the solid voxel to the empty one and split
along whichever diagonal keeps both triangles facing that way. A chunk reads the volumes of its
eight neighbours as well, so it has no surface until all nine have arrived, the same chunks as its
ground ([`ground_readers`](#ground)); chunks then meet without a seam, and four chunks mesh the same
triangles as one twice as wide. Positions are relative to the chunk's corner on the ground plane,
with the height absolute, in a Y-up engine's axes; normals follow the values' gradient and agree
with the triangles wherever features are wider than a voxel. With materials, each vertex takes the
material of the solid voxel of its cube nearest it, so the chunks on either side of a seam agree. The surface is left open where it
would reach below the lowest level or above the highest, so a volume solid along its lowest level
and empty along its highest is closed. `wave_forge::SurfaceWorker` builds surfaces on a thread of its own from the nine volumes around a
chunk, shared rather than copied, and hands back only the answer to a chunk's newest request, so an
engine's thread only draws them. Godot draws it and collides with it through
`volume_stage` ([godot.md](godot.md#waveforgestages)), and Bevy builds it with `.with_volume`
([bevy.md](bevy.md#packs-of-stages)).

### Carve

`Carve(volume: "rock", tunnels: Some((curves: "tunnels", height: Some("ground"), depth: 6.0, max_radius: 3)), rooms: Some("dungeon"))`:
the volume `rock` with tunnels and rooms carved out of it, empty inside them and unchanged elsewhere
([#71](https://github.com/AntonTegnelov/wave_forge/issues/71)). Every curve of `tunnels.curves`, a
[Region](#region), [Rivers](#rivers), [Network](#network) or [TableCurves](#tablecurves) stage, is a
tube around its centre line whose radius is the curve's value there, of at most `max_radius`
cells. A curve with heights of its own, a region job's worm tunnel say, runs through 3D space, its
centre `depth` cells (default 0) below its heights. A curve on the ground plane has none, so its
centre runs `depth` cells below the field `height` at the nearest point of the centre line; loading
refuses tunnels along a Rivers, Network or TableCurves stage without `height`, and a region job's
curve without heights where there is none fails with `StageError::Curve`.
Every piece of `rooms`, an [Assemble](#assemble) or [Cave](#cave) stage, is a box from its
footprint's floor up its height in cells, so a dungeon grown with a negative `lift` is a set of
rooms under its entrance.

A voxel within a cell of a tunnel or room, or inside one, keeps the lower of its value and how far
it is outside the nearest, in cells: negative inside, so empty there, and exact wherever a surface
reads it; every other voxel keeps its value. The lower of the two does not depend on the order
tunnels and rooms are carved in, so a carve is the same in any order, and a carve keeps its
volume's materials. It works at the WFC lattice's scale on a volume of that scale; a tunnel wider
than `max_radius` fails with `StageError::Curve`. Engines draw it as they draw a Volume stage.

`level: Some((sites: "outposts", depth: 3, clear: 5))` levels the ground of every site or piece of
a sites or Assemble stage before any tunnel or room is carved: in its footprint, the `depth` cells
under its height are filled solid and the `clear` cells above it emptied, each voxel within a cell
of either box keeping the higher of its value and how far inside the slab it is, or the lower of
its value and how far outside the air it is. So a building stands on flat ground under open sky
however the volume's terrain runs, and a tunnel or room still cuts through the slab.

### Top

`Top(volume: "caves")`: a field of the height, in cells, of the top of a Volume or Carve stage of the
same scale in each column: where its values cross zero going up from its highest solid voxel, or
the volume's bottom for a column with none. A volume of `Sub(Input("height"), Z)` has `height` as
its top. It is what things stand on over a volume, a [Scatter](#scatter) stage's `height` say.

### Aquifer

`Aquifer(volume: "caves", cell: (16, 12), level: (-40.0, -4.0), materials: Some(...))`: fluid in the
empty space of a Volume or Carve stage, as Minecraft's aquifers decide it, so caves below one pool's
level are not all flooded. Space is cut into cells `cell.0` columns wide and `cell.1` cells tall.
Each cell has a centre at a hashed place inside it and a level hashed from `level.0` up to
`level.1` cells, and a voxel belongs to the pool of the nearest centre among its own cell and the
26 around it. A voxel's value is the lower of how far it lies under its pool's level and how empty
the volume is there, so it is above zero, fluid, only in empty space under its pool's level. Open
air under a pool's level fills too, so a pack keeps `level.1` below the lowest open ground it
wants dry.

With `materials`, each voxel takes one by rules as a [Volume](#volume) stage's, read at its column
with `Z` as its pool's level rather than its own height, so rules on `Z` alone give a pool one
fluid: `(rules: [(category: "lava", when: [Less(Z, Constant(-30.0))])], otherwise: "water")` makes
the deepest pools lava. The product has its volume's size and bottom; it works at the WFC lattice's
scale on a volume of that scale.

Pools of two levels meet at a vertical face where their cells meet, which Minecraft walls off with
stone and an Aquifer stage does not ([#236](https://github.com/AntonTegnelov/wave_forge/issues/236)).
Godot draws it beside the rock as `fluid_stage`, see-through
and glowing by material, and Bevy builds its surface with `.with_fluid`
([godot.md](godot.md#waveforgestages), [bevy.md](bevy.md#packs-of-stages)); neither collides with
it.

### Rules

`Rules(rules: [(category: "sea", when: [Less(Input("height"), Constant(0.05))]), ...], otherwise:
"grassland")`: a category per column, the first rule whose conditions all hold there, or
`otherwise`. The conditions are the ones `Select` takes, over any expression. The stage's categories
are the names its rules give in the order they first appear, then `otherwise`'s, at most
`MAX_CATEGORIES` (256); `StageKind::categories` lists them, and a chunk's product (`Categories`) holds
one index into them per column.

A field reads categories only through `Is`, and a category test names categories of a Rules stage:
loading refuses reading a category stage as a field, a field as categories, and a name the rules do
not give. `examples/rings.world.ron` sorts an island into ten biomes by height, distance from the
centre with a wobble around it, and noise.

### Nearest

`Nearest(climate: [Input("heat"), Input("wet")], biomes: [(category: "tundra", point: [0.1, 0.2]),
...])`: a category per column, the biome whose `point` lies nearest the column's climate, the values
of the `climate` expressions there in their order, measured straight across that space; of equally
near biomes the first listed. This is how Minecraft's multi-noise biome source picks biomes from six
climate noises. Each point has one value per climate expression, and the biomes' names, each listed
once, are the stage's categories in their order, which a field reads through `Is` as it reads a
Rules stage's. A Nearest stage can be sampled without chunks.

### Blur

`Blur(input: "field", radius: r)`: the input averaged over the square of `r` cells around each
column.

### Delta

`Delta(input: "field", radius: r)`: the highest value of the input less its lowest over the square
of `r` cells around each column: how uneven the ground is there, what Valheim's location table calls
terrain delta. A Rules condition or a Select over it keeps a location or a plant off steep ground.

### Area

`Area(input: "biome", distance: d)`: where each column lies in its category of a Rules stage. Its
own categories are `median` and `edge`, in that order: `edge` where any of the eight columns `d`
cells away, along the axes and the diagonals, has another category than the column, `median`
elsewhere. That is Valheim's biome area, which keeps some locations to a biome's middle and puts
others on its border. A field reads it as any Rules stage's categories, `Is("area", ["edge"])`, and
loading refuses a distance of 0.

### Sites

`Sites(height: "field", region: r, size: (min, max), chance: c)`: settlement footprints,
rectangles of whole chunks between `min` and `max` chunks on a side, at most one per square region
of `r` × `r` chunks. A region has a site with probability `c`, decided by an integer test on the
stage's hash stream, and the site is kept at least one chunk inside its region, so two sites are
always two chunks apart. Each site's height is the mean of the height field over its footprint's
centre and inner corners.

A chunk's product lists the sites that overlap it. A `Site` has its `id`, `SiteId::Region` with
the region that owns it, its footprint `min..max` in chunks, and its `height`.

A region's site, its size and its place are a hash of the world's seed and the region, and its
height is a sample of the height field, so a site can be found without generating a chunk.
`Runtime::locate(stage, at, within)` gives the site nearest `at`, in cells on the lattice's plane,
measured to its footprint, searching regions up to `within` regions from `at`'s own; it is the site
generation places there, as `tests/locate.rs` checks. Only Sites stages can be located
(`StageError::NotLocated`).

### Locations

A location table: sites of several kinds, placed once per square region of `region` chunks.

```ron
(name: "places", kind: Locations(height: "ground", region: 24, kinds: [
    (name: "altar", priority: 10, quota: 3, apart: 48.0, tries: 60,
        when: [Greater(Is("biome", ["woods"]), Constant(0.5))]),
    (name: "trader", priority: 5, quota: 1, size: 2),
])),
```

Kinds are placed in order of `priority`, highest first, and by name on a tie. A kind tries `tries`
(default 20, at most `MAX_TRIES`, 1 024) hashed footprints of `size` chunks a side (default 1), each
one chunk inside the region, and keeps one that:

- comes within a chunk of no site the region has placed already, of any kind;
- lies at least `apart` cells (default 0) from every site of its own kind in the region, centre to
  centre;
- meets its `when` conditions, which a Rules stage takes, at the footprint's centre;

until it has `quota` of them. A site's height is found as a Sites stage finds it. Keeping a chunk
from the region's edge means sites of neighbouring regions never meet, so every region is placed
alone, whatever order chunks are asked for in, and kept while any chunk of it is needed. A site's
`kind` names its kind, and its id is `SiteId::Location` with its region and its place in the order
the region placed its sites; Flatten, Solve and Scatter's `avoid` read these sites as any others.

A site of a location table has a name for a game to show, `Site::name()`: never a finished string,
but the translation key `wf-place-<kind>`, the kind's name with its underscores as hyphens (a
`stone_circle` is `wf-place-stone-circle`), and the arguments `region_x`, `region_y` and `index` a
translation may use to tell sites of a kind apart. A Sites or TableSites stage's site has no name;
a row's name is the game's.

`Runtime::location_log(stage, chunk)` gives a line per kind for the region the chunk lies in, such
as `altar: placed 2 of 3; refused 5 crowded, 1 near its kind, 12 failing its conditions`. A quota is
per region, so what a world holds grows with the world; a finite world placed as one region holds
exactly its quotas, and 'unique' is a quota of 1. Loading refuses a quota or a size of 0, a region
too small to keep a site one chunk inside it, tries outside 1 to 1 024, a negative distance, two
kinds of one name, a kind's name that could not be a key (it starts with a lowercase letter and
holds only lowercase ASCII letters, digits, `_` and `-`), and conditions that read what a stage
cannot.

`examples/rings.world.ron` places shrines per ring: two in the woods at least 48 cells apart on
gentle ground, one in the peaks and one on the grassland, in every region of 24 chunks, and after
them a crypt in the woods.

### TableSites

`TableSites(table: "villages", height: "field", at: ("x", "y"), size: "size", max_size: m)`: a site
for every row of a table ([Tables of facts](#tables-of-facts)), where a history put its villages
say. A row's site is a square of whole chunks around the chunk holding its position, which the
columns `at` give in WFC cells, as many chunks on a side as its `size` column says, a whole number
from 1 to `m`; for an even size the extra chunk lies towards +x and +y. Its height is found as a
Sites stage finds it, and its `id` is `SiteId::Row` with the row's id.

`Runtime::set_facts` refuses rows whose position is not finite, whose size is not a whole number
from 1 to `m`, or whose site comes within a chunk of another row's, naming both rows: Flatten and
Solve rely on sites keeping a chunk apart, as a Sites stage's do. A stage reading a TableSites stage
before the runtime has facts fails with `StageError::NoFacts`. Loading refuses a table or a column
the pack does not have.

### Flatten

`Flatten(height: "field", sites: "sites", blend: b)`: the height field levelled to each nearby
site's height inside its footprint and blended back to the field over `b` cells around it. This is
the adapted field of the base, sites, adapted pattern ([stages.md](../architecture/stages.md#the-execution-contract)).
`sites` may name an [Assemble](#assemble) stage instead, and then each piece's footprint is levelled
to its floor: a village of pieces sits on sloping ground.

### Solve

`Solve(sites: "sites", rules: "name", bottom: selector, top: selector)`: a town on each site. A town
is a bounded WFC world of the named rule set, the size of the site's footprint in chunks, solved
whole from a seed of the site's own, with the module set's boundary rules at its sides. `bottom`
and `top` restrict its lowest and highest layers:

- `Tagged("tag")`: tiles carrying that tag;
- `Named("tile")`: tiles of that name.

Over a TableSites stage, `by: Some(("fate", [("burned", "ruins"), ...]))` chooses each town's rule
set by a names column of its site's row: a row whose `fate` is `burned` gets a town of `ruins`, and a
name the list does not give gets `rules`. Loading refuses `by` over a Sites stage, over a column that
does not hold names, and a name the column does not list. A row that changes its name is a new fact,
so its town is solved again with the other rule set.

A chunk's product is its part of the town (`TownChunk`: the site's id, its levelled height, and
the chunk's tiles, x fastest, then y, then z), or nothing outside every site. A town is solved once,
when its first chunk is needed, and kept while a chunk it covers is. Towns are solved on a thread
of their own, in the order they are asked for, so every other stage goes on generating meanwhile; a
chunk in a site arrives once its town is back, a chunk outside every site at once, and
`run_until_idle` waits for towns only when nothing else is left. A town compiles every kernel its
world runs at once before it starts ([measurements.md](../research/measurements.md) E41, E42). A
solver that panics stops generation with `StageError::TownsStopped`.

### Assemble

`Assemble(sites: "sites", start: "entry", pieces: [...], max: m, ...)`: pieces grown on each site
from connectors, as Minecraft's jigsaw villages and Valheim's dungeons are. A piece is a box of cells
a prefab fills, with doors on its sides:

```ron
(name: "street", size: (1, 4, 1), weight: 3, doors: [
    (at: (0, 0, 0), facing: South, kind: "street"),
    (at: (0, 3, 0), facing: North, kind: "street"),
    (at: (0, 1, 0), facing: East, kind: "house"),
]),
```

`size` is cells along the lattice's x and y and levels upward; a door opens from the cell `at` (x,
y, level) on the side it faces, North toward +y and East toward +x; two doors of one `kind` join
when they face each other from neighbouring cells on one level. `weight` (default 1) is how often a
piece is drawn, and `end: true` marks a piece that only closes doors.

An assembly grows inside its site's footprint:

1. the piece named `start` stands at the footprint's centre, at a hashed quarter turn;
2. each open door, in the order doors opened, draws a growing piece by weight among those with a
   door of its kind, and one of those doors, turns the piece so the doors face each other, and keeps
   it if it stays inside the footprint and clear of every piece placed, `tries` draws at most
   (default 20);
3. growth stops at `max` pieces, and each door still open is closed by the first end piece of its
   kind that fits;
4. an assembly of fewer than `min` growing pieces (default 0) is grown again from a new hash
   stream, `rerolls` times at most (default 0), and then fails with `StageError::Assemble`, naming
   the stage and the site.

`kinds: ["crypt"]` grows only on sites of those kinds of a location table, and `lift: l` raises
every piece `l` cells above its site, a dungeon above its entrance. An assembly is seeded from the
site's id and grown once, when its first chunk is needed, and kept while a chunk it covers is, so a
chunk finds the pieces overlapping it without growing the rest, and every piece is the same in any
order.

A chunk's product is the pieces whose footprint overlaps it (`Stamp`): a positional `InstanceId`
(the chunk and cell of the footprint's centre, 15 bits of the stage's salt, and the piece's place in
its assembly's growth), the site's id, the piece's name, a `position` in cells (the footprint's
centre along x and y, and its floor: the site's height plus `lift` plus its level), a `turn` as a
fraction of a whole turn, 0, 0.25, 0.5 or 0.75, from +x toward +y, and the columns it covers from
`min` up to but not including `max`. `Stamp::y_up_basis` gives the turn in a Y-up engine's axes: an
engine places a scene of the piece, authored at turn 0 with its footprint centred on its origin and
its floor at the origin's height, at `position` with that basis, and its doors then open where the
assembly joined them. Scatter's `avoid` keeps a margin from pieces as from sites.

Loading refuses two pieces of one name, a box of no cells, a growing piece of weight 0, a door that
is not on the side it faces, a `start` that no piece is named or that is an end piece, `min` above
`max`, `max` outside 1 to 1 024, rerolls above 1 024, tries outside 1 to 1 024, and a lift that is not
finite. Its ids share the Scatter stages' salts, so no Assemble stage shares a salt with a Scatter
stage either.

`examples/rings.world.ron` grows a dungeon of rooms, corridors and turns 40 cells above every crypt
of its location table, capping its dead ends, rerolled until it has eight pieces.

The runtime solves towns through the `TownSolver` trait. `WfcTowns::new(chunk).with_rules(name,
rule_file, build_solver)` is the implementation over any `Solver`, one per rule set;
`towns::gpu_solver` builds a GPU solver on a device of its own for it, and
`towns::gpu_solver_cached(rules, dir)` one that keeps its compiled kernels in `dir` across runs;
`WfcTowns::solver(name)` shows a rule set's solver, for example what compiling has cost it; `town_prior` builds a bounded
town's prior, which the city's own prior (`wfc_devtools::city::city_prior`) uses too. A pack with a
Solve stage needs `Runtime::with_towns`, or generating fails with `StageError::NoTownSolver`.

### Cave

`Cave(region: 8, depth: (-30.0, -10.0), patterns: [Linear, Star, Hub], count: (5, 7), apart: 12.0, rooms: [...])`:
a cave level per square region of `region` chunks, planned once for the whole region, as a mission
of a mining game is ([#220](https://github.com/AntonTegnelov/wave_forge/issues/220)). A plan
draws a pattern from `patterns` and a room count from `count.0` to `count.1` (at most 64), then
places each room, drawn by weight from `rooms: [(name: "cavern", size: (7, 7, 5), weight: 2)]`,
with its floor at a hashed height from `depth.0` up to `depth.1` cells. The first room lies in the
middle half of the region; every other lies from `apart` to twice `apart` cells from the room it
links to on the ground plane, at least `apart` cells from every room placed and wholly inside the
region, each room trying `tries` places at most (default 20). A plan that cannot place every room
is drawn again from a new hash stream, `rerolls` times at most (default 0), and then fails with
`StageError::RegionRejected`, a line per attempt. The pattern links the rooms:

| Pattern | Links |
|---|---|
| `Linear` | each room to the one placed before it, a chain |
| `Star` | every room to the first |
| `Hub` | the next three rooms to the first, and every later one to the room three before it, so three branches |

A chunk's product is the rooms overlapping it, as stamps with positional ids, the region as their
site, their floor's height and no turn. A [Carve](#carve) stage carves them as it carves an
Assemble stage's rooms, and a [Tunnels](#tunnels) stage joins the linked ones. The plan depends on
the seed, the stage and the region alone, so a level is the same in any order and a bounded pack
([World bound](#world-bound)) makes one region the whole level. It works at the WFC lattice's
scale. Engines bind scenes to its rooms by name, as to an Assemble stage's pieces
([godot.md](godot.md#scenes), [bevy.md](bevy.md#packs-of-stages)).

### Tunnels

`Tunnels(cave: "level", noise: "wander", radius: (1.5, 2.5), step: 2, wander: 1.0)`: a curve for
each pair of rooms the Cave stage `cave` links, from one room's centre to the other's, with a
height per point. Each is the cheapest path found by A* over a grid of `step` cells (default 2)
filling the cave's region from a cell under its lowest floor to a cell over its highest ceiling,
between neighbouring grid points in all 26 directions, where a step costs its length times one
plus `wander` (default 1) times the named noise at its middle, mapped from -1..1 to 0..1 and
sampled in 3D as a volume reads `FastNoise`. So a tunnel bends toward where the noise is low, and
with `wander: 0.0` runs as straight as the grid allows. A tunnel's radius, its curve's value at
every point, is hashed per tunnel from `radius.0` up to `radius.1` cells. A chunk's product is the
tunnels passing through it, which a [Carve](#carve) stage carves along their heights. Loading
refuses a `cave` that is no Cave stage and a noise the pack does not declare.

### Deposit

`Deposit(kind: "gold", cave: "level", volume: "caves", total: 40, depth: 2, apart: 2.0)`: exactly
`total` points of `kind` (1 to 4096) per level of the Cave stage `cave`, in the rock around its
rooms, so a level's resources meet its quota whatever its shape. Candidates are drawn by hash, each
at the centre of a voxel in a hashed room's shell, less than `depth` cells (default 1) outside its
box. A candidate is kept where the Volume or Carve stage `volume` is solid at that voxel, inside
the region, outside every room and at least `apart` cells (default 0) from every point kept, until
`total` are kept; a level whose `tries` candidates (default 4096) run out first fails with
`StageError::RegionRejected`. A chunk's product is the points whose column lies in it, with
positional ids, so `Edit::Remove` mines one for good. It reads the whole region's volume, so every
chunk of a Deposit stage waits for the volume across its region: a bounded level's cost, heavy for
a region of many chunks. A dig in that volume would place the level's deposits again, so a pack
whose player digs gives the digs a stage of their own after it, `Carve(volume: "caves")` with
nothing to carve, and the Deposit stage reads the one before.

### Spawn

`Spawn(cave: "level", budget: 10, kinds: [(kind: "grunt", cost: 1, weight: 4), (kind: "brute", cost: 7)])`:
points standing on the floors of the Cave stage `cave`'s rooms, each room spending its own
`budget` (at most 1000). Kinds are drawn by weight (default 1) among those whose `cost` (from 1)
is no more than what the room has left, each at a hashed place on the room's floor, until the room
can afford none. So what a room spends is at most its budget, and less than the cheapest kind
short of it: with a kind that costs 1, exactly its budget. A chunk's product is the points whose
column lies in it, with positional ids, so `Edit::Remove` keeps a killed one gone.

### Scatter

`Scatter(kind: "tree", height: "field", spacing: s, ...)`: points of `kind` standing on the height
field, made by a chain of modifiers applied in this order. Every field after `spacing` is optional.

1. **Candidates.** `count: (low, high)` candidates per square block of `s` cells (default one), a
   hashed number in the range for each block, each at a hashed column inside it with a hashed
   priority, and kept with probability `chance` (default 1, compared as an integer). With
   `group: Some((size: (low, high), radius: r))`, each candidate is the first point of a group of
   that many points scattered within `r` cells of it.
2. **Tests at each point's column**, the candidate's and each group member's own:
   - the height within `between`;
   - the slope at most `max_slope`, in height per cell;
   - every condition of `when`, which takes the conditions a Rules stage takes over any
     expression: a biome with `Greater(Is("biome", ["woods"]), Constant(0.5))`, an altitude, a mask
     field, a terrain delta, a biome area;
   - with `water: Some((depth: (low, high)))`, the ground between `low` and `high` cells below the
     pack's water ([Water](#water)), a lake's surface where there is one; with `float: true` in it,
     the point stands on the water's surface rather than on the ground under it, a lily or a boat;
   - at least `margin` cells from every site, or every piece of an Assemble stage, with
     `avoid: Some(("sites", margin))`;
   - at least a clearance from every point of the Scatter stages `block` names, with
     `block: [("rocks", 2.5)]`. Those stages are placed first, so where two kinds would overlap
     the one blocked gives way, the same whatever order chunks are asked for in.
3. **Spacing.** A candidate that passes its tests is kept unless a passing candidate of higher
   priority lies closer than `apart` cells. Candidates are judged by their own tests, never by
   whether spacing kept them, so the decision agrees across chunk seams. A kept candidate's group
   members skip this test and pass or fail on their own.
4. **Attributes.** `scale: (low, high)` (default 1), a `tilt: (low, high)` in degrees from the
   vertical in a hashed direction, and with probability `align` (default 0) standing along the
   ground's normal instead.

Loading refuses counts and group sizes outside 1 to 255, a negative radius, a tilt outside 0 to
180 degrees, an `align` outside 0 to 1, a scale that is not a positive range, an empty depth range,
and conditions that read what a stage cannot. The reach grows with `apart`, the group's radius and
what the conditions read.

A `Point` has a positional `InstanceId`, a `kind`, a `position` in cells (x and y on the ground, z
the field's value), a `turn` about its up as a fraction of a whole turn, a `scale`, and its `up`, a
unit vector along the lattice's x, y and height. `Point::y_up_basis` gives its rotation and scale
in a Y-up engine's axes. The id holds the candidate's chunk, 15 bits of the stage's salt, the
candidate's column, and a slot of the candidate's place in its block times 256 plus the member's
place in its group, so a group member standing in the next chunk still has an id of its own, and
one candidate per block with no group gives the ids a Scatter stage always gave. Loading refuses
two Scatter, Embed or Assemble stages whose ids would carry the same 15 bits of salt; renaming one
fixes it.

`Runtime::scatter_report(stage, chunk)` says what became of every candidate whose column lies in
a chunk, a group's first point for a group, as a `Judgement`: where it stood, and whether it became
a point or which modifier rejected it first, as a `Rejection`:
- `Chance`, `Height`, `Slope`, `Condition(n)` (the `n`th condition of `when`), `Water`, `Sites`
  or `Blocked` for its own tests;
- `Spacing` for a candidate that passed but lay too near one of higher priority.

It decides as generating does, from the inputs the runtime holds, so the kept candidates are
exactly the chunk's points, and it is what a viewer colours candidates by when a rule places
nothing (N5).

`examples/rings.world.ron` scatters ore rocks in the middle of the woods on gentle ground, ore veins
high in the peaks along the slope, and groves of three to six birches on the grassland.

### Embed

`Embed(kind: "iron", volume: "rock", spacing: 4, count: (2, 4), between: (-16.0, 20.0), when: [...])`:
points of `kind` inside a Volume or Carve stage's rock, ore say. Each square block of `spacing`
columns has `count` candidates (default one), each at a hashed place in the block and a hashed
height from `between.0` up to `between.1` cells. A candidate is kept where the volume's value at its
height, between the voxels below and above it as its surface runs, is above zero, so it lies inside
the drawn rock, and where every condition of `when` holds; the conditions are a Rules stage's, with
`Z` read as the point's height, so `Less(Z, Sub(Input("height"), Constant(4.0)))` keeps ore at
least four cells down. A chunk's product is the points whose column lies in it, with positional ids
as a Scatter stage's, so `Edit::Remove` mines one for good, and engines bind scenes to their kind as
to a Scatter point's. It works at the WFC lattice's scale on a volume of that scale.

### Region

`Region(job: "rivers", region: 4, halo: 1, inputs: ["height"], budget: 8)`: curves computed once per
square region of `region` chunks by a region job, Rust code the game gives the runtime with
`Runtime::with_region_job(name, job)`. A chunk's product is the region's curves that pass through
it. `halo` (default 0) and `budget` (default 1) are optional, and the stage reads its inputs as far
as `region - 1 + halo` chunks from any of its chunks.

A job implements `stages::regions::RegionJob`: `run(&RegionInput) -> Attempt`. Through the input it
reads its fields (`field(stage, x, y)`, refused beyond the region and its halo), the region's
columns, its own hash stream (`hash(purpose)`, which changes with the retry index), and
`edge_hash(edge, purpose)`, which the neighbour across that edge computes too. That is how two
regions agree on what crosses between them, a river's crossing point say, without reading each
other. `Attempt::Rejected(reason)` makes the runtime try again with the next retry index; when the
budget is spent, generation fails with `StageError::RegionRejected`, carrying every reason. A
missing job fails with `StageError::NoRegionJob`. A computed region is kept while any chunk of it
is needed, so a finite world is one region computed once.

A `Curve` has a positional id, `CurveId::Region` with its region and index, points in world
columns, one value per point, which an [Apply](#apply) stage reads as its radius, and optionally a
height in cells per point for a curve through 3D space, which a [Carve](#carve) stage's tunnels
follow. A Bevy game
registers its own jobs in the runtime it builds; a Godot game, which cannot, uses the built-in
[Rivers](#rivers) stage.

### Rivers

`Rivers(height: "terrain", region: 16, sources: 3, width: (0.6, 2.0), step: 2)`: rivers down a
height field, a region job built into the library, so a pack names it without Rust. In every square
region of `region` chunks, each of `sources` rivers (1 to `MAX_SOURCES`, 64) starts at the highest
of a few hashed columns of the region and steps `step` cells (default 1) at a time to the lowest of
the eight columns around it, until it reaches a height below the pack's water ([Water](#water)), a
lake of the pack's water, a hollow where no step goes lower, or the region's edge. Its values, the radius an Apply stage carves
by, grow from `width.0` at its source to `width.1` at its mouth (default 1 to 3). A river never
leaves its region, so regions never read each other, and rivers are the same in any order; a river
that reaches its region's edge stops there, so large regions suit an island whose rivers run to its
coast. The ring world's rivers run from its high ground to the sea and are carved into its `ground`.

### Lakes

`Lakes(height: "terrain", region: 16, min_columns: 4)`: water standing in the hollows of a height
field, a region job built into the library. In every square region of `region` chunks, a priority
flood (Barnes, Lehman and Mulla, 2014) starts from the region's edge columns and from the columns
under the pack's water, where water drains away, and raises every other column to the lowest height
its water could spill over. A column raised above its ground and above the sea is under a lake, and
a connected set of such columns, of at least `min_columns` (default 4), is a lake; a smaller one is
left dry. A lake never reaches its region's edge, since the edge drains, so regions never read each
other and lakes are the same in any order; a hollow across a region border stays dry.

The product is a field of the water's surface: a lake's level over its columns, the ground's height
everywhere else, so an expression can read it like any field and never meets a missing value. A
Lakes stage may be coarse ([Levels](#levels)), filling hollows over a world map. The pack's water
names it to join the rest of the water; loading refuses a region of 0 chunks or lakes of 0 columns.

### Network

`Network(sites: "towns", height: "ground", region: 12, width: 1.5, climb: 4.0, dry: Some(0.05))`:
paths between sites over a height field, a region job built into the library. In every square
region of `region` chunks, the sites of the sites stage `sites` whose centre lies in the region are
joined by a minimum spanning tree over the distances between their centres, grown from the first
site by id, and each edge becomes the cheapest path between the two centres by A* over the region's
columns: a step to one of the eight neighbours costs its length plus `climb` (default 4) times the
height it climbs or falls, so a road goes round a ridge through a gap rather than over it. Columns
below `dry` are never crossed, and an edge with no dry path is left out. A path is cut where it
enters either site's footprint, and its values are `width` (default 1.5), the radius an Apply stage
levels or carves by. A path never leaves its region, so regions never read each other and paths are
the same in any order; sites of different regions are not joined, so a network's region should hold
several of its sites' regions. Loading refuses a region of 0 chunks, a negative width or climb and a
`dry` that is not finite.

### TableCurves

`TableCurves(table: "roads", from: ("x0", "y0"), to: ("x1", "y1"), radius: "width")`: a straight
curve for every row of a table ([Tables of facts](#tables-of-facts)), from the point the `from`
columns give to the one the `to` columns give, in WFC cells, with the radius its `radius` column
gives at both ends. A history lays its roads this way, a row per stretch between two villages. A
curve's id is `CurveId::Row` with its row's id, and a chunk's product is the curves that pass
through it. `Runtime::set_facts` refuses a row whose points are not finite, or whose radius is
negative or beyond the `max_radius` of an Apply stage that draws it, naming the row.

### Apply

`Apply(height: "field", curves: "roads", max_radius: r, blend: b, profile: Level)`: the height field
with the curves of a Region or TableCurves stage drawn into it. At each column, the curve segment
that weighs most decides: within its radius, interpolated along it from the curve's values and at
most `r` cells, it weighs 1, and it falls to 0 over `b` cells beyond (`blend`, default 0) with a
smooth step. The column takes the profile's height by that weight:

- `Level`: the field's height at the column of the nearest point of the curve's centre line, so a
  road lies level across its width and follows the ground along it;
- `Carve(depth)`: that height lowered by `depth`, a river's bed.

Where curves overlap, the one that weighs most wins, and the first by id on a tie, so a column never
depends on the order curves arrive in. A curve whose radius is beyond `r` fails generation with
`StageError::Curve`; a table's rows are refused before that, when the facts are given. Apply works on
the WFC lattice, at scale 1, like Flatten.

## The runtime

```rust
let pack = Arc::new(Pack::parse(&text)?);
let mut runtime = Runtime::new(pack, seed, [8, 8])
    .with_towns(Box::new(towns))?;          // only for packs with a Solve stage
let dropped = runtime.request(&[FocusPoint::new(chunk, 3)], &["level", "city", "trees"])?;
runtime.run_until_idle()?;                  // or step(budget) between frames
let trees = runtime.points("trees", chunk);
```

- `Runtime::request(focus, targets)` works out, from the targets backwards, which chunks of every
  stage the request needs, replaces the previous request, and returns what it dropped as
  `(stage, chunk)`. `request_each(focus, [(stage, Some(radius)), ...])` gives targets a radius of
  their own around every focus point, a target with `None` keeping the focus point's: ground far
  out, locations nearer and clutter nearest. `request_bound(targets)` asks for everything inside
  the world's bound ([World bound](#world-bound)).
- `run_until_idle` generates what is missing, stage by stage with inputs first, nearest chunk
  first; `step(budget)` generates at most `budget` products, so a caller can take new requests in
  between; `is_idle` says whether anything is left.
- `sample(stage, at)` gives a stage's value at a point in WFC cells, and `atlas(stage, min, size)`
  its values over an area of its own columns, row by row with x fastest, without generating any
  chunk: exactly what the chunks would hold. Field, Rules, Nearest, Blur, Delta and Area stages whose
  inputs are too can be sampled; the others need neighbouring chunks and fail with `StageError::NotSampled`. Sampling
  takes `&self` and holds no products, so a game can build a runtime just to sample, on any thread:
  an atlas of 256 by 256 world tiles takes 50 ms in release on the dev container. This is how a
  history the game simulates reads the world before play.
- `timings()` reports what each stage has cost since the runtime was made, in the pack's order: a
  `StageTiming` of `products`, `ms` in all and `slowest_ms` (a Solve stage's includes its towns).
- `product`, `field`, `categories`, `curves`, `sites`, `tiles` and `points` read what a stage holds for a chunk; `held`
  counts products held.
- A stage reads its inputs only through a `FieldView` bounded by its reach. A read outside it
  returns `StageError::OutOfReach`, naming the stage and the reach it would have needed.
- `StageWorker::spawn(build)` runs a runtime on a thread of its own, built there by the closure
  (a town solver may own a device that belongs to its thread). `request` sends a new request;
  `drain` returns `StageEvent::Generated` and `StageEvent::Dropped` events and keeps every product
  shared for reading; `timings` reports each stage's cost as of the last drain; `failure` reports
  why the thread stopped, if it did. `set_facts` and `focus` send a runtime's facts and focus to
  the thread; what they make stale arrives as drops.

**Named hash streams.** Every random decision draws from `pcg3d` keyed by the world seed and an
FNV-1a salt of the stage's name, so adding, removing or reordering stages changes no other stage.

**Order independence** is checked by `tests/stages.rs`: a six-stage pack comes out bit for bit the
same over a 4×4-chunk area asked for all at once and one chunk at a time in either raster order.

## Ground

`wave_forge::ground(chunk, field, cell_size)` builds a chunk's ground from a height field stage:
a `GroundMesh` with a vertex above every column's centre plus the first column of the +x and +y
neighbours, so neighbouring chunks share their edge vertices exactly. Normals come from central
differences, which at an edge read the neighbour, so shading is continuous across chunks.
Positions are relative to the chunk's corner on the ground plane with heights absolute, and
`heights` holds the same grid for a height-field collider. All of it is in a Y-up engine's axes.

The triangles come in `levels` of detail, finest first. A level with a `step` of *s* triangulates
every *s*-th vertex along both axes, and there is one for every power of two that divides the
chunk's columns both ways: chunks of 8 by 8 columns have steps 1, 2, 4 and 8, and chunks with an odd
number of columns only full detail. A level's `error` is how far, in engine units, its surface is
at most above or below the grid's, which is what an engine weighs to choose a level by distance.
Neighbours at different levels share only the coarser one's edge vertices, so every level also
hangs a skirt: after the grid, `positions` holds a copy of each edge vertex, lower by the largest
error any level has along the chunk's edges plus a tenth of a cell's height, with its edge vertex's
normal. A level's `indices` hold its surface, facing up (counter-clockwise seen from +y), then its
skirt, facing out of the chunk. A coarser level's vertices are all a finer one's, so two levels
part along an edge by at most the coarser one's error there, which the skirt covers; the tenth of
a cell covers the cracks rasterisation opens where one side of an edge has vertices the other
lacks.

A chunk's ground reads the fields of the eight chunks around it, so `ground` returns `None` until
all nine have arrived, and the ground of a view reaches one chunk less than its fields.
`ground_readers(chunk)` lists the chunks whose ground may have become buildable when that chunk's
field arrives.

`wave_forge::ground_height(at, columns, field, cell_size)` gives the height, in engine units, of the
ground's surface at full detail above any point of the ground plane: on the same triangles, between
the four column centres around it. A field holds one height per column centre, so this, not a
column's value, is what a game stands a player or an object on.

`wave_forge::ground_materials(chunk, categories)` gives the category of every vertex of the same
grid from a Rules, Area or Nearest stage at the height field's scale, the vertices along the +x and +y edges
taking the neighbours' first columns as their heights do: the materials an engine's ground shader
tells apart. It returns `None` until the categories of the chunk and of the chunks beyond its +x
edge, its +y edge and its +x+y corner have arrived.

### Far ground

`wave_forge::far_ground(chunk, scale, field, cell_size, near)` builds the ground beyond the near
ground from a coarse height field stage ([Levels](#levels)): one `FarGround` mesh per chunk of it,
covering its `scale` by `scale` chunks of the WFC lattice. It has a vertex at every lattice chunk's
corner, where a near ground's first vertex is, with the height a fine stage reading the coarse field
there gets, so each lattice chunk is one square with exactly a near ground's outline, and two
neighbouring coarse chunks meet exactly. `near` gives the near ground drawn on a lattice chunk, if
any: its square is left out, so the two never overlap. Along every edge a far square shares with a
near ground a wall stands, from the higher of the two edges to below the lower by as much as the
near ground's skirt hangs, so no view sees between them at any level of detail the near ground is
drawn at; a square beside a near ground fans out from its centre through the wall's feet, so the
two share their vertices and no crack opens. Positions are relative to the corner of the first
lattice chunk the coarse chunk covers, with heights absolute; the walls face both ways. It returns
`None` until the coarse fields of the chunk and the eight around it have arrived, so the far ground
reaches one coarse chunk less than its field.

## In the engines

- Godot: the `WaveForgeStages` node ([godot.md](godot.md#waveforgestages)).
- Bevy: `WaveForgeStagesPlugin` ([bevy.md](bevy.md#packs-of-stages)).
