# WFC-GPU Implementation TODOs

This document outlines improvement opportunities for the GPU-accelerated Wave Function Collapse implementation.

## Performance & Scalability

- [x] **Implement multi-pass propagation**: Current implementation only performs a single propagation pass which may not fully resolve all constraints. Add support for iterative propagation with a work queue.
- [x] **Dynamic workgroup sizing**: Replace hard-coded workgroup size (64) with dynamic sizing based on GPU capabilities for optimal performance across hardware.
- [x] **Pipeline and shader caching**: Add mechanism to cache compiled shaders and pipelines for faster reinitialization.
- [x] **Optimize buffer synchronization**: Reduce CPU-GPU synchronization points and minimize buffer copies.
- [ ] **Parallel subgrid processing**: Add support for dividing large grids into subgrids that can be processed independently.
- [x] **Early termination**: Add mechanism to stop propagation early if further passes yield minimal changes.

## Robustness

- [x] **Improve asynchronous handling**: Replace blocking `pollster::block_on()` calls with proper async/await patterns.
- [x] **Enhanced error recovery**: Implement recovery mechanisms for non-fatal GPU errors instead of terminating.
- [ ] **Hardware feature validation**: Add explicit checks for required GPU features before attempting to use them.
- [ ] **Shader fallbacks**: Provide simplified shader implementations for hardware that doesn't support atomics or other advanced features.
- [ ] **Timeout handling improvements**: Replace fixed timeouts with adaptive timeouts based on grid size and complexity.
- [ ] **Reduce verbose logging**: Remove excessive debug logging in performance-critical paths.

## Features

- [ ] **Support larger tile sets**: Extend beyond the current 128 tile limit with dynamic array sizing in shaders.
- [ ] **Advanced adjacency rules**: Support for weighted adjacency rules and more complex constraint types.
- [ ] **Debug visualization**: Add ability to visualize propagation steps, entropy heatmaps, and contradictions.
- [ ] **Progressive results**: Allow retrieving partial/intermediate results before algorithm completion.
- [ ] **Backtracking support**: Implement backtracking capability when contradictions are found.
- [ ] **Custom entropy heuristics**: Allow custom entropy calculation strategies beyond Shannon entropy.

## Architecture

- [ ] **Separation of GPU synchronization and algorithm logic**: Refactor to better separate these concerns.
- [ ] **Dynamic buffer management**: Implement resizable buffers based on runtime requirements.
- [ ] **Proper resource cleanup**: Ensure all GPU resources are properly released using RAII patterns.
- [ ] **Abstract hardware specifics**: Create abstraction layers to handle different GPU backends/capabilities.
- [ ] **Reduce Arc nesting**: Simplify ownership model where excessive Arc wrapping occurs.
- [ ] **Modularize shader code**: Split large shader functions into more manageable, testable pieces.

## Testing & Documentation

- [ ] **Shader validation**: Add validation tests for WGSL shaders.
- [ ] **Buffer allocation strategy documentation**: Document and explain buffer creation logic.
- [ ] **Numerical stability analysis**: Assess and document entropy calculation numerical stability.

## Specific Shader Improvements

### entropy.wgsl

- [ ] **Improved Shannon entropy calculation**: Enhance numerical stability for edge cases.
- [ ] **Better atomicMinF32Index validation**: Ensure correct handling of race conditions.
- [ ] **Configurable tie-breaking**: Allow different strategies for breaking entropy ties.

### propagate.wgsl

- [ ] **Support for multiple propagation passes**: Enhance to support iterative constraint propagation.
- [ ] **Improved contradiction handling**: Track and report multiple contradictions for better debugging.
- [ ] **Memory access optimization**: Reduce atomic operations and memory access patterns.
