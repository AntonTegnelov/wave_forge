# WFC Core Module TODO List

## Performance Improvements

- [ ] Implement Shannon entropy calculation to replace simple counting in `CpuEntropyCalculator`
- [ ] Optimize grid access patterns for better cache locality in 3D grid operations
- [ ] Benchmark and optimize `ParallelConstraintPropagator` batch sizes for different grid dimensions
- [ ] Implement GPU-accelerated versions of both entropy calculation and constraint propagation
- [ ] Add compile-time optimizations for common use cases (e.g., 2D grids)

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
- [ ] Add abstraction layer for different grid dimensions (optimize 2D vs 3D)

## Code Quality Enhancements

- [ ] Add property-based tests (e.g., using proptest) for rule consistency
- [ ] Create benchmarking suite for performance regression testing
- [ ] Implement fuzzing for edge case discovery
- [ ] Add more examples showcasing different use cases

## Documentation

- [ ] Create visual documentation of algorithm steps
- [ ] Document performance characteristics and memory usage
- [ ] Add more detailed tutorials beyond code comments

## Priority Tasks (Start Here)

- [ ] Implement Shannon entropy for more accurate cell selection
- [ ] Add tile symmetry support
- [ ] Create memory-efficient rule storage for complex rulesets
- [ ] Implement serialization support for grid states
