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

## Edits

What a player changes in a world is a log, `Edits`, which a game keeps and saves beside its facts:
a world is a function of the pack, the seed, the facts and the edits.

- `Edit::Remove { point, at }` takes away a Scatter stage's point, a felled tree say, by its
  positional id (`PointId`, from its `InstanceId`) and where it stood.
- `Edit::Move { point, from, to, turn }` stands it at `to`, turned to `turn`. It stays in the chunk
  it was generated in, so an engine finds it there wherever it now stands.
- `Edit::Raise { stage, column, by }` adds `by` to a field stage's value at one of its columns:
  ground raised or dug. Raises of one column add up, and a column is shared by the chunks on either
  side of a border, so they never disagree.

`Runtime::set_edits(&edits)` gives a runtime the log, and every product is edited as it is
generated: a field's columns raised, a Scatter stage's points removed or moved. So an edit survives
eviction and regeneration, and a sample holds a raise as its chunk does. A new log dirties only the
chunks whose edits changed, a raise its column's chunk and a point the chunk it stood in, and
everything that reads them within its reach, then returns the drops as `request` does. Staleness is
kept per chunk, so a raise regenerates a reader's neighbouring chunks when their reach covers the
raised chunk, even where it does not cover the column. An edit that raises a stage that is no
field, or names a point no Scatter stage placed, fails with `StageError::Edit` and changes nothing.
`Edits::to_ron` and `from_ron` save and load the log.

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
every scale, so a formula means the same at any scale. Field, Blur, Rules and Region stages can be
coarse; Sites, Flatten, Solve and Scatter work on the WFC lattice, at scale 1, and may read coarser
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

Every stage works on a two-dimensional lattice of cell columns at its scale. Every stage produces one of six types (a field, categories, sites, tiles, points or curves), and loading refuses a stage that
reads one type as another.

| Kind | Produces | Reads, and how far |
|---|---|---|
| `Field` | Field | the fields named by `Input` and the categories named by `Is`, 0 cells; the categories a `Match` names, `blend` cells |
| `Rules` | Categories | what its conditions read, 0 cells |
| `Blur` | Field | one field, `radius` cells |
| `Delta` | Field | one field, `radius` cells |
| `Area` | Categories | one Rules stage, `distance` cells |
| `Sites` | Sites | a height field, `region` chunks |
| `TableSites` | Sites | a table's rows; a height field, `max_size` chunks |
| `Locations` | Sites | a height field, and what its kinds' conditions read, `region` chunks |
| `TableCurves` | Curves | a table's rows |
| `Apply` | Field | a height field and a Region or TableCurves stage, `max_radius + blend` cells |
| `Flatten` | Field | a height field, 0 cells; a Sites stage, `blend` cells |
| `Solve` | Tiles | a Sites or TableSites stage, 0 cells |
| `Scatter` | Points | a height field, `apart` cells (one more with `max_slope`); a Sites stage, `apart + margin` cells |

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
properties and the `domain_warp_` properties. `wave_forge::noise::NoiseConfig::sample` is a port of
FastNoiseLite 1.1.0, the version Godot bundles, and gives exactly what Godot 4.7's `get_noise_2d`
gives: `tests/fastnoise.rs` compares 2 560 samples over 80 configurations that Godot computed, with
no tolerance. A noise keeps its own seed, as a resource does, so the world's seed does not change it.

`Runtime::with_noise(name, config)` replaces a named noise, which is how an engine hands in a
resource; the Godot node's `noises` does it ([godot.md](godot.md#waveforgestages)). Loading refuses
a `FastNoise` of a name the pack's `noises` does not have, and `with_noise` of one fails with
`StageError::UnknownNoise`. Only 2D noise exists; 3D noise comes with density volumes
([#71](https://github.com/AntonTegnelov/wave_forge/issues/71)).

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

`Runtime::location_log(stage, chunk)` gives a line per kind for the region the chunk lies in, such
as `altar: placed 2 of 3; refused 5 crowded, 1 near its kind, 12 failing its conditions`. A quota is
per region, so what a world holds grows with the world; a finite world placed as one region holds
exactly its quotas, and 'unique' is a quota of 1. Loading refuses a quota or a size of 0, a region
too small to keep a site one chunk inside it, tries outside 1 to 1 024, a negative distance, two
kinds of one name, and conditions that read what a stage cannot.

`examples/rings.world.ron` places shrines per ring: two in the woods at least 48 cells apart on
gentle ground, one in the peaks and one on the grassland, in every region of 24 chunks.

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
when its first chunk is needed, and kept while a chunk it covers is.

The runtime solves towns through the `TownSolver` trait. `WfcTowns::new(chunk).with_rules(name,
rule_file, build_solver)` is the implementation over any `Solver`, one per rule set;
`towns::gpu_solver` builds a GPU solver on a device of its own for it, and
`towns::gpu_solver_cached(rules, dir)` one that keeps its compiled kernels in `dir` across runs;
`WfcTowns::solver(name)` shows a rule set's solver, for example what compiling has cost it; `town_prior` builds a bounded
town's prior, which the city's own prior (`wfc_devtools::city::city_prior`) uses too. A pack with a
Solve stage needs `Runtime::with_towns`, or generating fails with `StageError::NoTownSolver`.

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
   - with `water: Some((level: w, depth: (low, high)))`, the ground between `low` and `high` cells
     below `w`;
   - at least `margin` cells from every site, with `avoid: Some(("sites", margin))`;
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
one candidate per block with no group gives the ids a Scatter stage always gave. Loading checks that
no two Scatter stages share a salt.

`examples/rings.world.ron` scatters ore rocks in the middle of the woods on gentle ground, ore veins
high in the peaks along the slope, and groves of three to six birches on the grassland.

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
columns and one value per point, which an [Apply](#apply) stage reads as its radius. There is no
built-in job yet. Godot games reach Region stages once there are built-in jobs; a Bevy game
registers its own in the runtime it builds.

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
  chunk: exactly what the chunks would hold. Field, Rules, Blur, Delta and Area stages whose
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
Triangles face up (counter-clockwise seen from +y), positions are relative to the chunk's corner
on the ground plane with heights absolute, and `heights` holds the same grid for a height-field
collider. All of it is in a Y-up engine's axes.

A chunk's ground reads the fields of the eight chunks around it, so `ground` returns `None` until
all nine have arrived, and the ground of a view reaches one chunk less than its fields.
`ground_readers(chunk)` lists the chunks whose ground may have become buildable when that chunk's
field arrives.

## In the engines

- Godot: the `WaveForgeStages` node ([godot.md](godot.md#waveforgestages)).
- Bevy: `WaveForgeStagesPlugin` ([bevy.md](bevy.md#packs-of-stages)).
