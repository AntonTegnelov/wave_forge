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

## GPU Migration Plan

- [x] Phase 1: Complete GPU implementation core components

  - [x] Refactor GPU Accelerator to delegate WFC steps to dedicated structs
    - [x] Create dedicated `GpuEntropyCalculator` implementation in wfc-gpu
    - [x] Create dedicated `GpuConstraintPropagator` implementation in wfc-gpu
  - [x] Implement GPU-based constraint propagation algorithm
  - [x] Optimize grid representation for GPU memory layout (coalescence)
  - [x] Add comprehensive benchmarking tools to compare CPU vs GPU performance

- [ ] Phase 2: Performance optimization

  - [x] Implement batched update processing for constraint propagation
  - [x] Minimize CPU-GPU data transfers during algorithm steps
  - [ ] Optimize workgroup sizes and memory access patterns in shaders
  - [ ] Add shader specialization for different grid sizes and tile counts

- [ ] Phase 3: Transition plan

  - [ ] Deprecate CPU implementations with warning messages
  - [ ] Provide migration guide documentation
  - [ ] Update all examples to use GPU implementation by default
  - [ ] Create final benchmark suite demonstrating performance gains

- [ ] Phase 4: Cleanup
  - [ ] Remove CPU-specific implementations and refactor core API
  - [ ] Consolidate all computation code into the GPU-based module
  - [ ] Remove CPU-only optimizations and simplify codebase

## Missing Features

- [ ] Implement tile symmetry handling (rotation/reflection support in `TileSet`)
- [ ] Add rule generation helpers based on symmetry/transformation principles
- [ ] Support serialization of grid states for saving/loading
- [ ] Create checkpoint system to pause/resume WFC algorithm execution
- [ ] Enhance visualization hooks beyond simple progress callbacks
- [ ] Add boundary condition options (periodic, fixed, etc.)

## Architecture Improvements

- [ ] Refactor `AdjacencyRules` to use more memory-efficient storage for sparse rule sets
- [ ] Simplify error propagation in `runner.rs` with better Result chaining
- [ ] Break down the `run` function in runner.rs to improve separation of concerns
- [ ] Make hard-coded values configurable (iteration limits, contradiction handling)
- [ ] Create builder pattern for configurating WFC algorithm parameters

## Code Quality Enhancements

- [ ] Add property-based tests (e.g., using proptest) for rule consistency
- [ ] Implement fuzzing for edge case discovery
