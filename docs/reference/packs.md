# Packs

The pack format and the stage runtime as they are built, in `src/stages/` and `src/towns.rs`. The
design they are a first part of, and the stage kinds still to come, are in
[stages.md](../architecture/stages.md). This page changes in the same pull request as the code it
describes.

## A pack file

A pack is RON, conventionally `*.world.ron`: a version and a list of named stages. A stage reads
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

`Pack::reach(target, chunk_size)` reports, for every stage `target` depends on, how many cells
beyond a column of `target` it has to be generated: the largest sum of reaches along any path.
`Pack::stage_names` and `Pack::kind` describe the stages to a tool or an engine.

## Stages

All stages work on one lattice for now: one value per cell column on the WFC chunk lattice, two
dimensional, in cells. Every stage produces one of five types (a field, categories, sites, tiles or points), and loading refuses a stage that
reads one type as another.

| Kind | Produces | Reads, and how far |
|---|---|---|
| `Field` | Field | the fields named by `Input` and the categories named by `Is`, 0 cells; the categories a `Match` names, `blend` cells |
| `Rules` | Categories | what its conditions read, 0 cells |
| `Blur` | Field | one field, `radius` cells |
| `Sites` | Sites | a height field, `region` chunks |
| `Flatten` | Field | a height field, 0 cells; a Sites stage, `blend` cells |
| `Solve` | Tiles | a Sites stage, 0 cells |
| `Scatter` | Points | a height field, `apart` cells (one more with `max_slope`); a Sites stage, `apart + margin` cells |

### Field

`Field(expr)`: a value per cell column, from an expression at that column alone. Coordinates are
in cells, measured from the world's origin to the column's centre, along the lattice's x and y.

| Expression | Value |
|---|---|
| `Constant(v)` | `v` |
| `Noise(frequency: f, octaves: n)` | fractal value noise in 0..1: `n` layers (1 to 16), the first with `f` lattice points per cell, each next at twice the frequency and half the weight |
| `Noise(frequency: f, octaves: n, name: "hills")` | the same, from the stream named `hills`: identical in every stage that names it |
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

### Sites

`Sites(height: "field", region: r, size: (min, max), chance: c)`: settlement footprints,
rectangles of whole chunks between `min` and `max` chunks on a side, at most one per square region
of `r` × `r` chunks. A region has a site with probability `c`, decided by an integer test on the
stage's hash stream, and the site is kept at least one chunk inside its region, so two sites are
always two chunks apart. Each site's height is the mean of the height field over its footprint's
centre and inner corners.

A chunk's product lists the sites that overlap it. A `Site` has its `region` (which names it), its
footprint `min..max` in chunks, and its `height`.

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

A chunk's product is its part of the town (`TownChunk`: the site's region, its levelled height, and
the chunk's tiles, x fastest, then y, then z), or nothing outside every site. A town is solved once,
when its first chunk is needed, and kept while a chunk of its region is.

The runtime solves towns through the `TownSolver` trait. `WfcTowns::new(chunk).with_rules(name,
rule_file, build_solver)` is the implementation over any `Solver`, one per rule set;
`towns::gpu_solver` builds a GPU solver on a device of its own for it, and
`towns::gpu_solver_cached(rules, dir)` one that keeps its compiled kernels in `dir` across runs;
`WfcTowns::solver(name)` shows a rule set's solver, for example what compiling has cost it; `town_prior` builds a bounded
town's prior, which the city's own prior (`wfc_devtools::city::city_prior`) uses too. A pack with a
Solve stage needs `Runtime::with_towns`, or generating fails with `StageError::NoTownSolver`.

### Scatter

`Scatter(kind: "tree", height: "field", spacing: s, chance, between, max_slope, avoid, apart)`:
points of `kind` standing on the height field.

- One candidate per square block of `s` cells, at a hashed column inside it, with a hashed
  priority.
- A candidate passes its own tests: kept with probability `chance` (default 1, compared as an
  integer), height within `between`, slope at most `max_slope` in height per cell, and at least
  `margin` cells from every site when `avoid: Some(("sites", margin))` is set.
- A passing candidate is kept unless a passing candidate of higher priority lies closer than
  `apart` cells. Neighbours are judged by their own tests, never by whether spacing kept them, so
  the decision agrees across chunk seams.

A `Point` has a positional `InstanceId` (its chunk, 15 bits of the stage's salt and its column), a
`kind`, a `position` in cells (x and y on the ground, z the field's value) and a `turn` about the
vertical as a fraction of a whole turn. Loading checks that no two Scatter stages share a salt.

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

A `Curve` has a positional id (its region and index), points in world columns and one value per
point. There is no built-in job yet: rivers that carve the ground are
[#98](https://github.com/AntonTegnelov/wave_forge/issues/98). Godot games reach Region stages
once there are built-in jobs; a Bevy game registers its own in the runtime it builds.

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
  `(stage, chunk)`.
- `run_until_idle` generates what is missing, stage by stage with inputs first, nearest chunk
  first; `step(budget)` generates at most `budget` products, so a caller can take new requests in
  between; `is_idle` says whether anything is left.
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
  why the thread stopped, if it did.

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
