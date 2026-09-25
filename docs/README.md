# Wave Forge documentation

Wave Forge generates game worlds while the game runs. A world is made of chunks generated around the
player on the GPU, and it comes out the same for a seed whatever order its chunks were generated in.
Generation is a pack of stages (height fields, sites, levelled ground, scattered objects, and towns
built with wave function collapse), each reading the others within a declared reach. It is a Rust
library with a Godot extension and a Bevy plugin, under the MIT license, and it is not published
yet.

## Where to start

**To understand the project,** read these in order. Together they answer what it is, how it is built
and why, where it stands, and where it is going.

1. [product/vision.md](product/vision.md): what Wave Forge is for, its priorities, and what it will
   not do.
2. [architecture/overview.md](architecture/overview.md): the principles, the crates and seams, and
   the two ways a world is generated.
3. [plan/status.md](plan/status.md): what works today, the headline numbers, and the known limits.
4. [plan/roadmap.md](plan/roadmap.md): the phases, the open work in order, and the games that
   consume Wave Forge.

**To use it in a game,** read the reference for your engine
([godot.md](reference/godot.md) or [bevy.md](reference/bevy.md)), then [packs.md](reference/packs.md)
to write a pack of stages.

**To work on it,** read [contributing.md](guides/contributing.md) and
[environment.md](guides/environment.md), then the architecture page for the area you are changing,
then [testing.md](guides/testing.md).

## Every document

Each topic has one home; other documents link to it rather than repeating it.

### product: why, for whom, and the done gate

| Document | Owns |
|---|---|
| [vision.md](product/vision.md) | purpose, audiences, priorities, phases, scope and non-goals |
| [user-stories.md](product/user-stories.md) | the requirements as stories (G games, N newcomers, P performance), and the verification gate before anything counts as done or published |

### architecture: the design and why

| Document | Owns |
|---|---|
| [overview.md](architecture/overview.md) | principles, crates, seams, data flow, threads, errors |
| [solver.md](architecture/solver.md) | the model (rules, domains, prior, hashing), the block kernel, the solver and backend seams, GPU specifics, rejected alternatives |
| [world.md](architecture/world.md) | chunks, the halo, the parity schedule, repairs and repair classes, the determinism contract, streaming |
| [stages.md](architecture/stages.md) | generation as a pack of stages: the execution contract, data types, stage kinds, how WFC joins, tiers, scope |
| [engine-integration.md](architecture/engine-integration.md) | products, how each engine takes them, where kernels run, content classes, engine features, configuration |
| [constraints.md](architecture/constraints.md) | what adjacency rules can and cannot express, designing for connectivity, what a rule costs the search |

### reference: what is built

| Document | Owns |
|---|---|
| [packs.md](reference/packs.md) | the pack format, every stage kind as built, the stage runtime |
| [godot.md](reference/godot.md) | `WaveForgeWorld` and `WaveForgeStages`: properties, functions, signals, `stats()` |
| [bevy.md](reference/bevy.md) | `WaveForgePlugin`, `WaveForgeSolverPlugin`, `WaveForgeStagesPlugin` |

### plan: where it stands and where it is going

| Document | Owns |
|---|---|
| [status.md](plan/status.md) | what works by area, headline numbers, known limits |
| [roadmap.md](plan/roadmap.md) | phases, the order of open work, games and repositories, the release gate, deferred work |
| [story-coverage.md](plan/story-coverage.md) | per user story, what exists and what is missing, with issues |

### guides: how to work on it

| Document | Owns |
|---|---|
| [contributing.md](guides/contributing.md) | the workflow, the done gate, human-only publishing, documentation conventions |
| [environment.md](guides/environment.md) | toolchain, build output, the dev container's GPU and drivers, lavapipe, CI, Godot |
| [testing.md](guides/testing.md) | test layers, running them, end-to-end tests, benchmarks, the game session, rendering tools, engine checks |
| [debugging.md](guides/debugging.md) | reproducing failures, invariant oracles, the counters a solve returns, debugging stages and the Godot node |
| [performance.md](guides/performance.md) | method, where work runs, memory, code priorities, build profiles, profiling |
| [desktop-measurements.md](guides/desktop-measurements.md) | the script that takes the measurements a desktop owes, and how to run it on Windows |

### research: the evidence

| Document | Owns |
|---|---|
| [measurements.md](research/measurements.md) | every measurement with its protocol, by era |
| [literature.md](research/literature.md) | the solver and WFC literature we rely on or tested |
| [worldgen-survey.md](research/worldgen-survey.md) | how engines, libraries and eight games structure generation |
| [per-collapse-solver.md](research/per-collapse-solver.md) | the solver that was replaced: what it measured, what was refuted, what carried over |

The dev container's setup and operations are in [.devcontainer/README.md](../.devcontainer/README.md).
