# Project TODO: Multithreaded 3D GPU Terrain Generation (WFC) - Refined

This list outlines the major components and tasks, broken down by proposed crates within a Rust workspace (`wave_forge/Cargo.toml` would define the workspace members).

## Workspace Setup

- [x] Configure the main `wave_forge/Cargo.toml` to define workspace members (`wfc-core`, `wfc-rules`, `wfc-gpu`, `wave-forge-app`).

## Crate: `wfc-core` (Core Logic - No I/O or GPU specifics)

_Purpose: Defines the fundamental data structures and the platform-agnostic WFC algorithm._

- **File:** `wfc-core/src/tile.rs`
  - [x] Define `TileId` (enum or simple struct).
  - [x] Define `TileSet` struct to hold tile information (e.g., symmetry, weight).
- **File:** `wfc-core/src/rules.rs`
  - [x] Define `AdjacencyRules` struct/trait to represent constraints between `TileId`s.
  - [x] _Implement using a flattened, indexed structure suitable for efficient CPU lookup and GPU buffer packing._
- **File:** `wfc-core/src/grid.rs`
  - [x] Define `Grid<T>` struct (3D) to store cell states (e.g., `T = bitvec::BitVec` for possibilities, `f32` for entropy). (Requires `bitvec` dependency).
  - [x] _Use bitsets (`bitvec` crate) for cell possibilities to optimize memory and allow fast bitwise operations._
  - [x] Implement methods for grid initialization, getting/setting cell states.
- **File:** `wfc-core/src/propagator.rs`
  - [x] Define `ConstraintPropagator` trait (for DI).
    - Method: `propagate(&mut Grid<bitvec::BitVec>, updated_coords: Vec<(usize, usize, usize)>) -> Result<(), PropagationError>`
  - [x] Implement a basic CPU-based `ConstraintPropagator`.
    - [x] **Parallelism (CPU):** Explore using `rayon` to parallelize the processing of constraints for updated cells. _Requires careful synchronization or atomic operations to prevent data races when multiple threads update the same neighboring cell._ Consider strategies like chunking updates or using thread-safe data structures.
- **File:** `wfc-core/src/entropy.rs`
  - [x] Define `EntropyCalculator` trait (for DI).
    - Method: `calculate_entropy(&Grid<bitvec::BitVec>) -> Grid<f32>`
    - Method: `find_lowest_entropy(&Grid<f32>) -> Option<(usize, usize, usize)>`
  - [x] Implement a basic CPU-based `EntropyCalculator`.
    - [x] **Parallelism (CPU) - Calculation:** Utilize `rayon` to parallelize entropy calculation across grid cells or chunks. Ensure thread-safe access if necessary (likely read-only access to the main grid suffices here).
    - [x] **Parallelism (CPU) - Finding Minimum:** Use `rayon`'s parallel iterators and reduction operations (e.g., `min_by_key`) to efficiently find the cell with the lowest entropy from the calculated entropy grid.
- **File:** `wfc-core/src/runner.rs` or `src/lib.rs`
  - [x] Implement the main WFC `run` function.
    - **DI:** Takes `&mut Grid`, `&TileSet`, `&AdjacencyRules`, `impl ConstraintPropagator`, `impl EntropyCalculator` as input.
    - [x] Implement core loop: find lowest entropy -> collapse -> propagate.
    - [x] Handle collapse logic (choosing a tile based on weights).
    - [x] Handle potential contradictions/failures returned by propagation. (Simple failure return for now).
    - [x] Add progress reporting support via optional callback parameter:
      ```rust
      pub fn run<P, E>(
          grid: &mut Grid<bitvec::BitVec>,
          tileset: &TileSet,
          rules: &AdjacencyRules,
          propagator: P,
          entropy_calculator: E,
          progress_callback: Option<Box<dyn Fn(ProgressInfo) + Send + Sync>>,
      ) -> Result<(), WfcError>
      ```
    - [x] Create `ProgressInfo` struct with iteration count, collapsed cells count, etc.
- **File:** `wfc-core/src/lib.rs`
  - [x] Define public API / module structure.
- **Testing:** `wfc-core/tests/`
  - [x] Add unit tests for grid manipulation.
  - [x] Add unit tests for CPU entropy calculation.
  - [x] Add unit tests for CPU propagation logic.
  - [x] Add integration tests for the core WFC `run` function (using CPU components).

## Crate: `wfc-rules` (Rule Loading & Parsing)

_Purpose: Handles loading tile set definitions and adjacency rules from external files._

- **File:** `wfc-rules/src/lib.rs`
  - [x] Define error types for loading/parsing.
- **File:** `wfc-rules/src/loader.rs`
  - [x] Implement `load_from_file(path: &Path) -> Result<(TileSet, AdjacencyRules), LoadError>` function.
  - [x] Choose file format (RON). Add necessary dependencies (`ron`).
- **File:** `wfc-rules/src/formats/mod.rs`, `wfc-rules/src/formats/ron_format.rs` (example)
  - [x] Define structs matching the chosen file format (`#[derive(Serialize, Deserialize)]`).
  - [x] Implement parsing logic to transform file format structs into `wfc-core` structs (`TileSet`, flattened/indexed `AdjacencyRules`).
- **Testing:** `wfc-rules/tests/`
  - [x] Add tests for loading valid rule files.
  - [x] Add tests for handling invalid or malformed rule files.

## Crate: `wfc-gpu` (GPU Acceleration via Compute Shaders)

_Purpose: Implements GPU-accelerated versions of entropy calculation and propagation._

- **File:** `wfc-gpu/src/lib.rs`
  - [x] Set up `wgpu` basics (Instance, Adapter, Device, Queue). Needs `wgpu` dependency.
  - [x] Define error types.
- **File:** `wfc-gpu/src/shaders/`
  - [x] Write `entropy.wgsl` compute shader.
  - [/] Write `propagate.wgsl` compute shader. (Initial structure created)
- **File:** `wfc-gpu/src/pipeline.rs`
  - [ ] Implement shader loading and compute pipeline creation.
- **File:** `wfc-gpu/src/buffers.rs`
  - [ ] Implement logic to create and manage GPU buffers for grid state (bitsets), rules (flattened/indexed), entropy etc.
  - [ ] Implement CPU <-> GPU data transfer logic (staging buffers). _Design to keep grid state and rules primarily on GPU, minimizing transfers during the main loop._
- **File:** `wfc-gpu/src/accelerator.rs`
  - [ ] Define `GpuAccelerator` struct holding `wgpu` state, pipelines, buffers.
  - [ ] Implement `wfc-core::EntropyCalculator` trait using the entropy compute shader.
  - [ ] Implement `wfc-core::ConstraintPropagator` trait using the propagation compute shader.
- **Testing:** `wfc-gpu/tests/` (May require specific setup or be harder to unit test)
  - [ ] Add tests for buffer creation and data transfer.
  - [ ] (Optional/Difficult) Add tests comparing GPU vs CPU results for simple cases.

## Crate: `wave-forge-app` (Main Binary - Orchestrator)

_Purpose: Ties everything together, handles user input (CLI), manages threading, and outputs results._

- **File:** `wave-forge-app/src/config.rs`

  - [ ] Define `AppConfig` struct for generation parameters (size, rule file path, use_gpu, seed, output_path).
  - [ ] Implement command-line argument parsing using `clap`.
  - [ ] Add configuration options for benchmarking, progress reporting, and visualization:

    ```rust
    pub struct AppConfig {
        // Existing fields...
        pub benchmark_mode: bool,
        pub report_progress_interval: Option<Duration>,
        pub visualization_mode: VisualizationMode,
    }

    pub enum VisualizationMode {
        None,
        Terminal,
        Simple2D,
    }
    ```

- **File:** `wave-forge-app/src/benchmark.rs`
  - [ ] Implement benchmarking infrastructure:
    - [ ] Add function to run both CPU and GPU implementations with identical input
    - [ ] Record metrics: total runtime, memory usage, time per step, etc.
    - [ ] Implement comparison and reporting of benchmark results
    - [ ] Support different grid sizes and complexity levels for comprehensive testing
    - [ ] Add option to output results in CSV format for further analysis
- **File:** `wave-forge-app/src/progress.rs`
  - [ ] Implement progress reporting system:
    - [ ] Define `ProgressReporter` trait with methods for reporting status updates
    - [ ] Implement console reporter with percentage complete, ETA, and current operation
    - [ ] Support throttling of updates to avoid performance impact
    - [ ] Add statistics tracking (contradictions, backtracks, collapsed cells)
- **File:** `wave-forge-app/src/visualization.rs`
  - [ ] Implement simple visualization modes:
    - [ ] Define `Visualizer` trait with methods to display current WFC state
    - [ ] Implement terminal-based visualization using ASCII/Unicode characters
    - [ ] Add simple 2D slice visualization of the 3D grid state
    - [ ] Support toggling visualization on/off during runtime
    - [ ] Implement color coding for entropy levels and cell states
    - [ ] Add ability to focus on specific layers/slices of the 3D grid
- **File:** `wave-forge-app/src/main.rs`
  - [ ] Parse arguments into `AppConfig`.
  - [ ] Load rules using `wfc-rules::loader::load_from_file`.
  - [ ] Initialize `wfc-core::Grid`.
  - [ ] **DI / Loose Coupling:** Based on `AppConfig`, instantiate the required components. The `wfc-core::runner::run` function only depends on the `EntropyCalculator` and `ConstraintPropagator` traits, ensuring it's decoupled from the specific implementation (CPU or GPU):
    - If `use_gpu`: Initialize `wfc_gpu::GpuAccelerator`. Create trait objects (e.g., `Box<dyn EntropyCalculator>`) or use generics (`impl EntropyCalculator`) pointing to the GPU implementations provided by `wfc-gpu`.
    - Else: Create trait objects or use generics pointing to the default CPU implementations provided by `wfc-core`.
  - [ ] **Threading:** Use `rayon` to run them in parallel.
    - [ ] Parallelize parts of a _single_ large grid generation
  - [ ] If benchmark mode is enabled, run benchmarks using the benchmark infrastructure.
  - [ ] Set up progress reporting based on configuration.
  - [ ] Initialize visualizer if visualization mode is enabled.
  - [ ] Call `wfc-core::runner::run` with the initialized grid, injected components, and progress callback.
  - [ ] Handle results (success or failure).
- **File:** `wave-forge-app/src/output.rs` (Simple Output)
  - [ ] Implement function to save the final collapsed `Grid<TileId>` to a simple format (e.g., text, basic binary, RON).
- **Testing:** `wave-forge-app/tests/`
  - [ ] Add integration tests for running the app with basic configs (CPU and GPU if possible).
  - [ ] Add tests for argument parsing.
  - [ ] Add tests for benchmark comparisons.
  - [ ] Add tests for progress reporting.
  - [ ] Add tests for visualization rendering.

## General / Ongoing

- [ ] Refine error handling across all crates. _Strategy: Use `thiserror` for library crates (`wfc-core`, `wfc-rules`, `wfc-gpu`) and `anyhow` for the application (`wave-forge-app`)._
- [ ] Add documentation comments (`///`) to public APIs.
- [ ] Set up basic logging (`log`, `env_logger`).
- [ ] Profile and optimize bottlenecks (CPU and GPU).
- [ ] **Thread Safety (CPU):** Rigorously verify thread safety in `rayon`-based parallel implementations within `wfc-core` (e.g., avoiding data races on shared structures like `Grid`, ensuring proper synchronization if needed, handling error aggregation from parallel tasks).
- [ ] Add visualization toggle to CLI interface.
- [ ] Implement configurable logging levels for progress reporting.
- [ ] Add benchmark results to documentation.
- [ ] Create sample configurations for different visualization modes.
