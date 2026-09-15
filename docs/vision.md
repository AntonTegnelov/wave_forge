# Vision

Wave Forge is a procedural world generator built to run **inside games, at runtime**, and to scale to massive and complex worlds. Everything else in the project follows from that sentence, so this document explains what it means and why it forces the design choices described in [architecture.md](architecture.md).

## What we are building

A generator that produces 2D and 3D worlds from rules, delivered in three forms that share one implementation:

| Deliverable | Audience | Why it exists |
|---|---|---|
| **Rust library** | Any Rust program, custom engines, tools | The universal core. Everything else is a thin wrapper around it, so there is exactly one place where generation logic lives and is tested. |
| **Bevy plugin** | Bevy games | Bevy is the main Rust game engine; a plugin lets generation plug into its ECS, asset and task systems instead of fighting them. |
| **Godot GDExtension** (published on the Godot Asset Library) | Godot games | Godot has the largest open-source engine audience. GDExtension lets native Rust code run inside Godot without forking the engine, and the Asset Library is where Godot users look for tools. |

Everything is released under the **MIT license**, so it can ship inside commercial and open-source games alike, including through the Godot Asset Library. That also constrains dependencies: every crate we depend on must be MIT-compatible.

## Priorities, in order

When goals conflict, earlier items win.

1. **Runtime performance and scalability.** Generation happens while a player is playing. Slow generation means visible pop-in, frame drops or loading screens, and a generator that cannot scale limits the worlds a game can offer. This is the number one priority.
2. **Correct, predictable output.** A fast generator that produces invalid or non-reproducible worlds is useless to a game. (See *Determinism* below for why reproducibility follows from the runtime use case.)
3. **Easy integration into engines.** The value is realised inside Bevy, Godot and other engines, so the public API has to fit how engines work: incremental, chunked, off the main thread.
4. **Binary size.** Games ship to players and stores; bloat matters, but less than speed.
5. **Compile time and developer ergonomics.** Last. We accept complex code (static dispatch, GPU kernels, SIMD) when it buys runtime speed, and pay for it with good documentation of *why* the code looks the way it does.

## How we get speed: parallelism, chosen by measurement

The main lever is parallelization, applied in this order of preference:

1. **GPU compute** wherever the work is parallel enough and large enough to outweigh upload, dispatch and readback overhead.
2. **CPU threads** for parallel work where GPU overhead is too high.
3. **SIMD** where even threading overhead is too high (small, hot, data-parallel loops such as bitset operations).
4. **Single-threaded code** only when nothing else works.

These are *assumptions to test, not rules to follow blindly.* Real measurements on real workloads decide where each piece of work runs, and the architecture must make it cheap to move work between tiers when the numbers say so.

## Scale, with realistic bounds

Scalability does not mean unbounded. A game never needs to generate an entire universe at once; even a space game with many galaxies only needs what is around the player. Wave function collapse in particular becomes impractical on humongous single grids (memory grows with cells × tiles, and the chance of an unrecoverable contradiction grows with size). So the design target is:

- **Bounded working sets:** generate in chunks/regions of a bounded size, never one giant grid.
- **Continuous, on-demand generation:** keep generating what is near the player as they move, and let far-away regions stay ungenerated.
- **Seamless borders:** neighbouring chunks must agree at their edges.

### Determinism

On-demand generation only works if regenerating a region produces the same result as the first time (the player walks away and comes back, a save is loaded, or several multiplayer clients generate the same area). Therefore generation must be **deterministic for a given world seed, region and rule set**, independent of thread count, GPU vendor or generation order. This is a direct consequence of the runtime use case and constrains everything from random number generation to parallel tie-breaking.

## Phases

### Phase 1: standalone 2D and 3D wave function collapse

A pure WFC terrain generator for square (2D) and cubic (3D) grids, usable as a Rust library, fast enough for runtime use, with the testing and inspection tooling needed to develop it (see [roadmap.md](roadmap.md)). Inspiration for the kind of output we want to reach: Marian Kleineberg's infinite WFC city ([article](https://marian42.de/article/wfc/), [code](https://github.com/marian42/wavefunctioncollapse)).

### Phase 2: layered world generation

WFC is excellent for structured, locally-constrained content (cities, buildings, dungeons) and poor at large-scale natural shapes. Real worlds need several techniques combined, in the spirit of [LayerProcGen](https://github.com/runevision/LayerProcGen): generation organised into **layers**, where each layer works on bounded regions and can read the results of the layers beneath it. Examples of the intended combination:

- noise-based generation for the broad landscape,
- Fractal Jittered Voronoi Partitions for coastlines and region boundaries,
- WFC for cities placed into that landscape.

The system has to support **layering, blending between techniques, and multiple passes**. Phase 1 must not paint us into a corner here: WFC has to be able to accept constraints produced by other layers (pre-decided cells, masks, border conditions) rather than only starting from a blank grid.

### Engine packaging

The Bevy plugin and Godot GDExtension wrap the library once its API has settled. Their exact timing is an open decision; see [roadmap.md](roadmap.md).

## Non-goals

- **Not a game engine or renderer.** Wave Forge produces data (which tile goes where); engines turn it into meshes, scenes and gameplay. Any rendering in this repository (PNG exports, orthographic views) is **developer tooling** to inspect results, not a shipped feature.
- **Not an editor.** Authoring tools may come later, but the product is the runtime generator.
- **No unbounded generation.** We design for bounded regions around a focus, not whole universes.
