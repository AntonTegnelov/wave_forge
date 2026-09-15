# Roadmap

Phases come from [vision.md](vision.md). Tasks are the alignment tasks (A-n) in [status.md](status.md#alignment-tasks) plus new work. A phase is done when its exit criteria hold, not when its task list is empty.

## Phase 0: revive the project (done)

Make the abandoned code base build and actually produce valid output again, and bring dependencies and toolchain up to date. Done in [#3](https://github.com/AntonTegnelov/wave_forge/issues/3) and [#4](https://github.com/AntonTegnelov/wave_forge/issues/4).

## Phase 1: standalone 2D and 3D WFC generator

**Goal:** a correct, deterministic, fast WFC library for 2D and 3D grids that other code can call.

Suggested order, and why:

1. **Tooling and tests first** ([#6](https://github.com/AntonTegnelov/wave_forge/issues/6); A-16, A-15 instrumentation). The later steps rewrite core data structures and move work between CPU and GPU; without end-to-end tests, image-based inspection and timelines, regressions in a parallel GPU program are nearly impossible to find.
2. **Profile, then optimise** ([#7](https://github.com/AntonTegnelov/wave_forge/issues/7); A-10, A-11, A-12, A-4). Measure where time goes on realistic rule sets before choosing what runs on GPU, CPU threads or SIMD.
3. **Model and correctness gaps** (A-2, A-3, A-5, A-6, A-7, A-8, A-9). Some of these (storage layout, RNG) are best done together with step 2 because they touch the same code.
4. **Public API and workspace structure** (A-1, A-14). Settle the crate layout and the library facade once the internals have stabilised, so the API reflects what the implementation can do efficiently.
5. **Regions and streaming** (A-13).

**Exit criteria:**

- 2D and 3D generation through a documented library API, with no CLI or engine required.
- Same seed and inputs produce identical output across runs and thread counts.
- E2E tests: a 2D tile set rendered to PNG, and a small 3D city comparable in spirit to [marian42's WFC city](https://marian42.de/article/wfc/).
- Benchmarks on reference rule sets with results recorded in the repository, and each CPU/GPU/SIMD placement decision backed by a measurement.
- Streaming generation of regions around a moving focus point with seamless borders.

## Engine integrations

**Goal:** a Bevy plugin and a Godot GDExtension published to the Godot Asset Library, both thin wrappers around the library.

**Open decision:** build a minimal integration early (after Phase 1 step 4) to validate that the API fits real engine scheduling, or wait until Phase 1 is complete. Integrating early costs maintenance while the API still changes but catches API mistakes sooner.

## Phase 2: layered world generation

**Goal:** LayerProcGen-style layers combining techniques: noise landscapes, Fractal Jittered Voronoi Partition coastlines and WFC cities, with layering, blending and multiple passes. See [architecture.md §7](architecture.md#7-phase-2-layered-generation-design-constraints-to-keep-in-mind-now) for the constraints Phase 1 must respect.

Detailed planning waits until Phase 1 has shown what the region, seed and constraint APIs look like in practice.
