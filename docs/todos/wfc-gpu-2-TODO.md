# WFC-GPU Refactoring Plan

This document outlines a structured plan to refactor the wfc-gpu module, addressing code duplication and architectural issues while preserving the improvements from the original TODO list.

## 1. Shader Consolidation

- [ ] **Unify shader variants**: Consolidate duplicate shader files (entropy/propagate variants) into single files with feature toggles

  - [ ] Create a shader preprocessing system that handles feature flags
  - [ ] Replace `_fallback` and `_modular` variants with conditional compilation in core files
  - [ ] Maintain a clear mapping between feature flags and hardware capabilities

- [ ] **Modularize shader code**: Ensure proper code reuse across shaders
  - [ ] Extract common utilities into properly imported modules
  - [ ] Standardize include mechanism for shader code
  - [ ] Remove duplicated functions like `count_ones` across files

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

## 4. Centralize Shader Management

- [ ] **Activate `shaders.rs`**:

  - [ ] Move shader loading logic from `pipeline.rs` to `shaders.rs`
  - [ ] Create a shader registry system for managing variants
  - [ ] Implement shader preprocessing (constants, includes) in a centralized way

- [ ] **Shader feature detection**:
  - [ ] Create a system to determine optimal shader variant based on hardware capabilities
  - [ ] Consolidate feature detection code currently spread across modules

## 5. Unify Testing Strategy

- [ ] **Consolidate test modules**:

  - [ ] Create a unified test framework in `test_utils.rs`
  - [ ] Extract common test setup code to reduce duplication
  - [ ] Organize tests by concern rather than by source file

- [ ] **Improve shader testing**:
  - [ ] Merge `shader_validation_tests.rs` with proper tests in `shaders.rs`
  - [ ] Create more comprehensive shader validation tests

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

## Implementation Strategy

1. **Start with low-risk changes**:

   - Begin with shader consolidation and modularization
   - Improve tests to catch regressions early

2. **Progressive refinement**:

   - Implement changes to one component at a time
   - Maintain backward compatibility during transition

3. **Measure impact**:

   - Create benchmarks to verify performance is maintained or improved
   - Add metrics to track memory usage and resource allocation

4. **Preserve existing improvements**:
   - Ensure all checked items in original TODO list remain implemented
   - Validate that advanced features still work correctly
