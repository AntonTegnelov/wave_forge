# WFC Core Module TODO List

## GPU Migration Plan

- [ ] Phase 1: Complete GPU implementation core components

  - [ ] Create dedicated `GpuEntropyCalculator` implementation in wfc-gpu
  - [ ] Implement GPU-based constraint propagation algorithm
  - [ ] Optimize grid representation for GPU memory layout (coalescence)
  - [ ] Add comprehensive benchmarking tools to compare CPU vs GPU performance

- [ ] Phase 2: Performance optimization

  - [ ] Implement batched update processing for constraint propagation
  - [ ] Minimize CPU-GPU data transfers during algorithm steps
  - [ ] Optimize workgroup sizes and memory access patterns in shaders
  - [ ] Add shader specialization for different grid sizes and tile counts

- [ ] Phase 3: API enhancement

  - [ ] Create unified factory for CPU/GPU backends (runtime selection)
  - [ ] Add graceful fallback to CPU implementation when GPU unavailable
  - [ ] Implement multi-GPU support for large grids
  - [ ] Add configurable precision options (float vs double for entropy)

- [ ] Phase 4: Transition plan

  - [ ] Deprecate CPU implementations with warning messages
  - [ ] Provide migration guide documentation
  - [ ] Update all examples to use GPU implementation by default
  - [ ] Create final benchmark suite demonstrating performance gains

- [ ] Phase 5: Cleanup
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
