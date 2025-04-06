# WFC-GPU Refactoring Plan

This document outlines a structured plan to refactor the wfc-gpu module, addressing code duplication and architectural issues while preserving the improvements from the original TODO list.

## 1. Shader Modularization

- [ ] **Create truly modular shader components**:

  - [ ] Break down shaders into even smaller, single-purpose files
  - [ ] Organize shader code by functionality (e.g., entropy calculation, propagation logic, utility functions)
  - [ ] Create clear interfaces between shader components

- [ ] **Build shader assembly system**:

  - [ ] Develop a shader preprocessing pipeline that assembles complete shaders from modular components
  - [ ] Use conditional includes based on feature flags and hardware capabilities
  - [ ] Generate optimal variants automatically at build time instead of maintaining separate versions

- [ ] **Eliminate duplication across shader files**:
  - [ ] Ensure each function exists in exactly one place
  - [ ] Create a dependency graph for shader components
  - [ ] Standardize interfaces between shader modules

## 2. Clear Responsibility Boundaries

- [ ] **Buffer management vs. synchronization**:

  - [ ] Move all buffer data transfer operations to `GpuSynchronizer`
  - [ ] Limit `GpuBuffers` to storage and creation of buffers
  - [ ] Create clear documentation of responsibility boundaries

- [ ] **Entropy calculation**:

  - [ ] Centralize entropy calculation in `GpuEntropyCalculator`
  - [ ] Remove duplicate implementation from `GpuAccelerator`
  - [ ] Make `GpuAccelerator` delegate to specialized components

- [ ] **Pipeline management**:
  - [ ] Move shader loading/compilation from `pipeline.rs` to `shaders.rs`
  - [ ] Make `pipeline.rs` focus only on pipeline creation and binding

## 3. Simplify Complex Structures

- [ ] **Break up `GpuBuffers`**:

  - [ ] Create specialized buffer groups (`GridBuffers`, `WorklistBuffers`, `EntropyBuffers`)
  - [ ] Define clear interfaces between buffer groups
  - [ ] Implement a facade pattern if needed for backward compatibility

- [ ] **Revise `GpuAccelerator`**:
  - [ ] Convert to a coordinator that delegates to specialized components
  - [ ] Remove direct implementation of traits, use composition instead
  - [ ] Make the ownership model clearer with fewer nested `Arc`s

## 4. Advanced Shader Management

- [ ] **Implement shader component system in `shaders.rs`**:

  - [ ] Create a component registry for shader modules
  - [ ] Develop an intelligent shader assembly system
  - [ ] Support conditional compilation based on feature requirements

- [ ] **Feature-based shader optimization**:

  - [ ] Create a capability detection system that maps hardware features to shader requirements
  - [ ] Implement automatic selection of optimal shader components based on detected capabilities
  - [ ] Build verification tools to ensure correctness of assembled shaders

- [ ] **Build-time shader generation**:
  - [ ] Add support for pre-generating optimized shader variants during build
  - [ ] Create shader caching to avoid redundant processing
  - [ ] Develop tools for shader variant testing and verification

## 5. Unify Testing Strategy

- [ ] **Consolidate test modules**:

  - [ ] Create a unified test framework in `test_utils.rs`
  - [ ] Extract common test setup code to reduce duplication
  - [ ] Organize tests by concern rather than by source file

- [ ] **Improve shader testing**:
  - [ ] Create component-level tests for shader modules
  - [ ] Add validation tests for assembled shaders
  - [ ] Create a shader sandbox environment for testing isolated components

## 6. Consistent Error Handling

- [ ] **Unify error types**:

  - [ ] Consolidate `GpuError` and `BackendError` into a cohesive system
  - [ ] Use consistent error propagation patterns throughout
  - [ ] Ensure proper context is maintained in error messages

- [ ] **Error recovery improvements**:
  - [ ] Standardize use of `error_recovery.rs` across all operations
  - [ ] Add structured recovery strategies for different error types

## 7. Reduce Redundancy in Core Algorithm

- [ ] **Extract core algorithm logic**:

  - [ ] Create a clear separation between algorithm logic and GPU implementation
  - [ ] Ensure implementation follows the same pattern for CPU and GPU versions

- [ ] **Standardize workload division**:
  - [ ] Unify the subgrid handling approach
  - [ ] Create consistent patterns for parallel execution

## 8. Documentation & API Design

- [ ] **Create clear API boundaries**:

  - [ ] Define public vs internal APIs
  - [ ] Document integration points clearly
  - [ ] Add more comprehensive examples

- [ ] **Technical documentation**:
  - [ ] Document buffer lifecycle and ownership
  - [ ] Add architecture diagrams showing component relationships
  - [ ] Create shader reference documentation
  - [ ] Document the shader component system and assembly process

## Implementation Strategy

1. **Start with low-risk changes**:

   - Begin with creating the shader component system
   - Build and test individual shader modules in isolation

2. **Progressive refinement**:

   - Implement the shader assembly system
   - Gradually migrate to the modular approach while maintaining compatibility

3. **Measure impact**:

   - Create benchmarks to verify performance is maintained or improved
   - Add metrics to track shader compilation time and runtime performance

4. **Preserve existing improvements**:
   - Ensure all checked items in original TODO list remain implemented
   - Validate that advanced features still work correctly with the new modular shader system
