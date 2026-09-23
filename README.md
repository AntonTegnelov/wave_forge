# Wave Forge

A procedural world generator built to run inside games at run time and scale to large, complex worlds. It is a **Rust library**, a **Bevy plugin** and a **Godot GDExtension**, all under the MIT license.

Worlds are generated in chunks around the player, on the GPU, and come out the same for the same seed whatever order they were generated in. Generation is a **pack of stages**: height fields, settlement sites, levelled ground, towns built with GPU **wave function collapse**, and scattered objects, each reading the others within a declared reach, in the spirit of [LayerProcGen](https://github.com/runevision/LayerProcGen). The reasoning is in [docs/product/vision.md](docs/product/vision.md).

## Status

**In development, not published.** The library streams an infinite WFC city around moving focus points, and generates a first valley with towns from a pack of stages, and island biomes by rules. A Godot extension and a Bevy plugin serve both to a game and are checked in real engines. Biomes, rivers, dungeons, edits and ground meshes are still to come. See [docs/plan/status.md](docs/plan/status.md) for what works and its limits, and [docs/plan/roadmap.md](docs/plan/roadmap.md) for what comes next.

## Try it

Requires Rust 1.98.1 (pinned via `rust-toolchain.toml`) and a Vulkan, Metal or DirectX 12 device. A recent Mesa lavapipe (25 or later) works as a software device for correctness, not for speed ([docs/guides/environment.md](docs/guides/environment.md)).

From a game, the library generates chunks around wherever the player is:

```rust
use wave_forge::{Builder, ChunkCoord, FocusPoint, Prior, Ruleset};

let ruleset = Ruleset::new(&rules, &weights)?;     // adjacency and weights, from a rule file
let mut world = Builder::new(ruleset, Prior::open(tiles)).seed(7).build()?;
world.request(&[FocusPoint::new(ChunkCoord::new(0, 0, 0), 4)]);
world.tick()?;                          // starts one batch, never blocks
for event in world.poll()? { /* a chunk's tiles are ready, or it could not be placed */ }
```

A pack of stages runs the same way through `wave_forge::stages::Runtime`; the format is in [docs/reference/packs.md](docs/reference/packs.md), and `examples/valley.world.ron` is a complete one.

`examples/history` is a Godot project that puts a history into a world with GDScript alone: a toy
history of a continent's rivers, villages and roads, given to the stages as tables of facts, and a
scene to walk through the result.

To look at a rule set by hand, the developer CLI generates one chunk and writes each cell's tile index to `output.txt`:

```bash
cargo run -p wfc-devtools --release --bin wave-forge -- --rule-file examples/simple-pattern.ron --width 8 --height 8 --depth 8
```

## Documentation

Start at [docs/README.md](docs/README.md): it says what to read for using Wave Forge, contributing to it, or following where it is going.

## License

[MIT](LICENSE)
