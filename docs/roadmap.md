# Roadmap

Phases come from [vision.md](vision.md). Tasks are the alignment tasks (A-n) in [status.md](status.md#alignment-tasks) plus new work. A phase is done when its exit criteria hold, not when its task list is empty.

## Phase 0: revive the project (done)

Make the abandoned code base build and actually produce valid output again, and bring dependencies and toolchain up to date. Done in [#3](https://github.com/AntonTegnelov/wave_forge/issues/3) and [#4](https://github.com/AntonTegnelov/wave_forge/issues/4).

## Phase 1: standalone 2D and 3D WFC generator

**Goal:** a correct, deterministic, fast WFC library for 2D and 3D grids that other code can call.

Order, and where it stands:

1. **Tooling and tests first** ([#6](https://github.com/AntonTegnelov/wave_forge/issues/6)). Done. The later steps rewrote the core data structures and moved all the work onto the device; without end-to-end tests, image-based inspection and invariant checks, regressions in a parallel GPU program are nearly impossible to find.
2. **Profile, then redesign the solver** ([#7](https://github.com/AntonTegnelov/wave_forge/issues/7); A-10, A-11, A-12, A-4, A-6, A-9). Done. Measurement chose the design: one workgroup solves one region per dispatch, with everything resident on the device ([solver-redesign.md](solver-redesign.md), [solver-fit.md](solver-fit.md)).
3. **Chunks and streaming** (A-13). Done, together with step 2: the streaming suite generates a city in front of a walking player and the world comes out seamless.
4. **Public API and workspace structure** (A-1, A-14). Done: the root package is the `wave_forge` library, the model and the two seams are separate crates, and the CLI is a developer tool.
5. **Model and correctness gaps that remain** (A-2 topology, A-3 symmetry in rule files, A-16 golden images). These are the only Phase 1 items left, and none of them blocks an integration.

**Exit criteria:**

- ~~2D and 3D generation through a documented library API, with no CLI or engine required.~~ Done.
- ~~Same seed and inputs produce identical output across runs and thread counts.~~ Done, and tested by comparing whole worlds.
- ~~E2E tests: a 2D tile set rendered to PNG, and a small 3D city comparable in spirit to [marian42's WFC city](https://marian42.de/article/wfc/).~~ Done, both through the library.
- ~~Benchmarks on reference rule sets with results recorded in the repository, and each CPU/GPU/SIMD placement decision backed by a measurement.~~ Done ([solver-fit.md](solver-fit.md)).
- ~~Streaming generation of regions around a moving focus point with seamless borders.~~ Done, within a 0.5 s tick budget.

What is left of Phase 1 is A-2, A-3 and A-16. 2D works as a world one cell deep, so the topology
abstraction is a cost question rather than a capability one.

## Engine integrations

**Goal:** a Bevy plugin and a Godot GDExtension published to the Godot Asset Library, both thin wrappers around the library.

Both exist and are verified against a real engine; neither is published. The library API was
designed against what each engine actually owns, and that is what each integration turned out to
need ([architecture.md §5.1](architecture.md#51-the-solver-seam)):

- **Bevy** is done (`wave_forge_bevy`): the plugin builds a generator on Bevy's own device with
  `build_on`, holds it as a resource, and asks, starts and collects from two systems per frame. It
  tracks Bevy 0.20 because that is the first release on wgpu 30, which is what makes sharing the
  device a matter of handing two resources over. The one wrinkle worth knowing: a kernel
  specialisation takes seconds to compile, so `WaveForgePlugin::warm` exists to do it while a game
  loads.
- **Godot** is done as far as a game is concerned (`wave_forge_godot`), on the second of the two
  shapes we had in mind: a `Worker` owns the generator on a thread with a wgpu device of its own, and
  `_process` drains its events into signals. Nothing on Godot's side waits for the device, and
  nothing on the generating thread touches a Godot object, so the extension needs none of
  godot-rust's thread-safety features. `verify.sh` drives it in a real Godot and checks what it
  produced.

  **Still open: generating on Godot's own `RenderingDevice`.** It would avoid a second device and its
  allocator, which is the reason the `ComputeBackend` seam exists. It needs the kernel's WGSL
  translated to SPIR-V (naga can, either at runtime or as a build-time matrix) and Godot's compute
  API driven from a worker thread, with `is_done` as `is_task_completed` on a `WorkerThreadPool`
  task. What it *cannot* have here is a measurement: this dev container reaches its GPU through
  Mesa's dozen, which does not expose `VK_KHR_swapchain`, and Godot refuses to create any
  `RenderingDevice` without it. Godot does run on the software Vulkan device (lavapipe), which is
  enough to check that such a backend produces correct worlds but says nothing about whether it is
  faster than a device of its own. The comparison belongs on a desktop Godot.

## Phase 2: layered world generation

**Goal:** LayerProcGen-style layers combining techniques: noise landscapes, Fractal Jittered Voronoi Partition coastlines and WFC cities, with layering, blending and multiple passes. See [architecture.md §7](architecture.md#7-phase-2-layered-generation-design-constraints-to-keep-in-mind-now) for the constraints Phase 1 had to respect.

Phase 1 answered what a layer will be handed: a chunk lattice with deterministic per-chunk seeds, and
constraints as a `Prior` on starting domains. Detailed planning waits until an integration has shown
what a game actually asks for.
