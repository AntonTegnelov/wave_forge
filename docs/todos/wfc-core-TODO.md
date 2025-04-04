# WFC Core Module TODO List

## Rules Integration

- [x] Migrate from internal rules implementation to wfc-rules module
  - [x] Identify and document all uses of internal `AdjacencyRules` in the codebase
  - [x] Create direct mapping between wfc-rules types and internal usage requirements
    - _Analysis: `wfc-rules` currently parses external formats (e.g., RON) into intermediate structs (`RonTileData`, `RonRuleFile`), validates them, and then directly constructs `wfc_core::TileSet` (from `Vec<f32>` weights) and `wfc_core::AdjacencyRules` (from `usize`, `usize`, and a flattened `Vec<bool>`)._
    - _Requirement: `wfc-rules` needs to define its *own* `TileSet` and `AdjacencyRules` types. The loading functions should return these new types. `wfc-core` (propagator, runner) must then be updated to use these `wfc-rules` types._
    - _Data Mapping: The core data (`Vec<f32>` for tileset weights, flattened `Vec<bool>` for adjacency rules) remains the same, but the struct definitions will move from `wfc-core` to `wfc-rules`._
  - [x] Replace internal rules implementation with wfc-rules in one commit
  - [x] Update all affected tests to use wfc-rules
  - [x] Verify all tests pass with the new implementation
  - [x] Remove the internal rules implementation code completely
  - [x] Update documentation to reflect the new dependency on wfc-rules module

## Missing Features

- [x] Implement tile symmetry handling (rotation/reflection support in TileSet)
- [x] Add rule generation helpers based on symmetry/transformation principles
- [x] Support serialization of grid states for saving/loading
- [x] Create checkpoint system to pause/resume WFC algorithm execution
- [x] Enhance visualization hooks beyond simple progress callbacks
- [x] Add boundary condition options (periodic, fixed, etc.) -> Parameter added, implementation needed in propagator

## Architecture Improvements

- [x] Refactor `AdjacencyRules` to use more memory-efficient storage for sparse rule sets
- [x] Simplify error propagation in `runner.rs` with better Result chaining
- [x] Break down the `run` function in runner.rs to improve separation of concerns
- [x] Make hard-coded values configurable (iteration limits, contradiction handling)
- [x] Create builder pattern for configurating WFC algorithm parameters

## Code Quality Enhancements

- [x] Add property-based tests (e.g., using proptest) for rule consistency
- [x] Implement fuzzing for edge case discovery (Setup done, needs targets implemented/run)

## Core Logic

- [x] Add boundary condition options (periodic, fixed, etc.) -> Parameter added, implementation needed in propagator
- [x] Implement tile transformations (rotation, reflection) in `TileSet` and rule generation. (Note: `Transformation::combine` logic for flips is incomplete)
- [x] Add support for weighted tile selection during collapse.
- [x] Implement entropy selection strategies (FirstMinimum, RandomLowest) for `CpuEntropyCalculator`. (GPU/Hilbert still TODO).
- [ ] Consider alternative propagation algorithms (e.g., AC-4). (Deferred)
- [x] Add checkpointing/resuming functionality to the `runner`. (Basic iteration-based implemented)
- [x] Add progress reporting callback to the `runner`. (Implemented)
- [x] Refactor `runner` for clarity and potential parallelization (internal steps).
- [/] Add comprehensive unit tests for `propagator`, `entropy`, `runner`. (Added some runner tests; propagator tests still problematic)
- [x] Add integration tests for the overall WFC process.
