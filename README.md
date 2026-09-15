# Wave Forge

A procedural world generator built to run inside games at runtime and scale to massive, complex worlds. It will ship as a **Rust library**, a **Bevy plugin** and a **Godot GDExtension** (via the Godot Asset Library), all under the MIT license.

Phase 1 is a standalone, GPU-accelerated **wave function collapse** generator for 2D and 3D grids. Phase 2 extends it into layered world generation that combines WFC with other techniques (noise landscapes, Voronoi-based coastlines) in the spirit of [LayerProcGen](https://github.com/runevision/LayerProcGen). The reasoning is in [docs/vision.md](docs/vision.md).

## Status

**Early development, not ready for use.** The GPU WFC solver works end to end for small 3D grids through a development CLI, but there is no library API yet, and the solver is slow for large grids and not yet deterministic. See [docs/status.md](docs/status.md) for exactly what works, what doesn't, and the plan to align the code with the architecture.

## Try it

Requires Rust 1.98.1 (pinned via `rust-toolchain.toml`) and a Vulkan, Metal or DirectX 12 capable device. A software Vulkan driver such as Mesa llvmpipe also works.

```bash
cargo run --release -- --rule-file examples/simple-pattern.ron --width 8 --height 8 --depth 8
cargo run --release -- --help
```

The output file (`output.txt` by default) lists the chosen tile index for every cell: one line per row, blank lines between Z layers.

## Documentation

- [Vision](docs/vision.md): goals, priorities, phases
- [Architecture](docs/architecture.md): target design and why
- [Status](docs/status.md): current state and alignment tasks
- [Roadmap](docs/roadmap.md): what comes next
- [Development guide](docs/development.md): building, testing, conventions

## License

[MIT](LICENSE)
