# WFC Core Module TODO List

## GPU Migration Plan

- [ ] Design GPU data structures for efficient representation of the WFC state

  - [ ] Create `GpuGrid` trait and `wgpu`-based implementation optimized for GPU memory layout
  - [ ] Design shader-compatible representation of possibility states (likely using uint arrays)
  - [ ] Implement buffer management system for grid data transfer

- [ ] Core algorithm GPU implementation

  - [ ] Implement `GpuEntropyCalculator` using compute shaders
  - [ ] Implement `GpuConstraintPropagator` using compute shaders for massive parallel propagation
  - [ ] Develop atomic operations for parallel grid updates

- [ ] Integration and compatibility

  - [ ] Create dispatcher to manage shader pipelines and buffer synchronization
  - [ ] Implement host-device memory management for efficient data transfer
  - [ ] Develop fallback mechanism for systems without compatible GPU hardware

- [ ] Migration and benchmarking
  - [ ] Create benchmarking suite to compare CPU vs GPU performance across grid sizes
  - [ ] Implement phased deprecation of CPU-based implementations
  - [ ] Document minimum GPU requirements and feature detection

## Remaining Features

- [ ] Implement tile symmetry handling (rotation/reflection support in `TileSet`)
- [ ] Add rule generation helpers based on symmetry/transformation principles
- [ ] Support serialization of grid states for saving/loading
- [ ] Create checkpoint system to pause/resume WFC algorithm execution
- [ ] Enhance visualization hooks beyond simple progress callbacks
- [ ] Add boundary condition options (periodic, fixed, etc.)

## Architecture Improvements

- [ ] Refactor `AdjacencyRules` to use GPU-compatible storage for sparse rule sets
- [ ] Adapt runner architecture for async GPU execution model
- [ ] Break down the `run` function in runner.rs to improve separation of concerns
- [ ] Make hard-coded values configurable (iteration limits, contradiction handling)
- [ ] Create builder pattern for configurating WFC algorithm parameters

## Code Quality Enhancements

- [ ] Add property-based tests (e.g., using proptest) for rule consistency
- [ ] Implement fuzzing for edge case discovery
- [ ] Create shader unit tests and validation tools for GPU code
