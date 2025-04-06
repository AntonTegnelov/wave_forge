# WFC-GPU Refactoring Plan

This document outlines a specific plan to refactor the wfc-gpu module, addressing code duplication and architectural issues while preserving the improvements from the original TODO list.

## 1. Shader Modularization

- [ ] **Create truly modular shader components**:

  - [ ] Create new directory structure in `wfc-gpu/src/shaders/`:
    - `wfc-gpu/src/shaders/components/` - Base components
    - `wfc-gpu/src/shaders/features/` - Feature-specific implementations
    - `wfc-gpu/src/shaders/variants/` - Generated shader variants (build outputs)
  - [ ] Extract from existing shaders into component files:
    - [ ] `components/entropy_calculation.wgsl` (from entropy\*.wgsl)
    - [ ] `components/worklist_management.wgsl` (from propagate\*.wgsl)
    - [ ] `components/cell_collapse.wgsl` (new file)
    - [ ] `components/contradiction_detection.wgsl` (from propagate\*.wgsl)
  - [ ] Keep but refactor core utility files:
    - [ ] `utils.wgsl` - Keep only truly generic utilities
    - [ ] `coords.wgsl` - Coordinate system operations
    - [ ] `rules.wgsl` - Adjacency rule handling

- [ ] **Build shader assembly system**:

  - [ ] Create new files:
    - [ ] `wfc-gpu/src/shader_compiler.rs` - Shader preprocessing & assembly system
    - [ ] `wfc-gpu/src/shader_registry.rs` - Registry for shader components & features
  - [ ] Create build script:
    - [ ] `wfc-gpu/build.rs` - Pre-build shader generation
  - [ ] Modify `shaders.rs`:
    - [ ] Expand from 12 lines to a full shader management system
    - [ ] Implement the shader variant loading interface

- [ ] **Eliminate duplication across shader files**:
  - [ ] Eventually remove redundant files:
    - [ ] `entropy_fallback.wgsl`, `entropy_modular.wgsl` (after component extraction)
    - [ ] `propagate_fallback.wgsl`, `propagate_modular.wgsl` (after component extraction)
  - [ ] Create new component registry:
    - [ ] `wfc-gpu/src/shaders/components/registry.json` - Component metadata & dependencies

## 2. Clear Responsibility Boundaries

- [ ] **Buffer management vs. synchronization**:

  - [ ] Modify `sync.rs`:
    - [ ] Move all data transfer methods from `GpuBuffers` to `GpuSynchronizer`
    - [ ] Add new transfer methods for remaining buffer types
  - [ ] Modify `buffers.rs`:
    - [ ] Remove data transfer methods
    - [ ] Refocus on buffer creation and management
  - [ ] Create new documentation file:
    - [ ] `wfc-gpu/docs/buffer_lifecycle.md` - Explain buffer ownership & synchronization

- [ ] **Entropy calculation**:

  - [ ] Modify `entropy.rs`:
    - [ ] Enhance `GpuEntropyCalculator` to handle all entropy calculation logic
  - [ ] Modify `accelerator.rs`:
    - [ ] Remove direct entropy calculation implementation
    - [ ] Change `EntropyCalculator` trait impl to delegate to `GpuEntropyCalculator`

- [ ] **Pipeline management**:
  - [ ] Modify `pipeline.rs`:
    - [ ] Remove shader loading code
    - [ ] Focus only on pipeline creation and binding
  - [ ] Expand `shaders.rs`:
    - [ ] Add shader loading & preprocessing logic from `pipeline.rs`
    - [ ] Implement shader variant management

## 3. Simplify Complex Structures

- [ ] **Break up `GpuBuffers`**:

  - [ ] Create new files:
    - [ ] `wfc-gpu/src/buffers/grid_buffers.rs` - Grid state buffers
    - [ ] `wfc-gpu/src/buffers/worklist_buffers.rs` - Propagation worklist buffers
    - [ ] `wfc-gpu/src/buffers/entropy_buffers.rs` - Entropy calculation buffers
    - [ ] `wfc-gpu/src/buffers/rule_buffers.rs` - Adjacency rule buffers
    - [ ] `wfc-gpu/src/buffers/mod.rs` - Facade implementation & common utilities
  - [ ] Modify existing `buffers.rs`:
    - [ ] Move code to appropriate new files
    - [ ] Convert to a facade for backward compatibility

- [ ] **Revise `GpuAccelerator`**:
  - [ ] Modify `accelerator.rs`:
    - [ ] Convert to use composition rather than direct implementation
    - [ ] Create delegation methods for all WFC operations
    - [ ] Reduce Arc nesting
  - [ ] Create new coordination files:
    - [ ] `wfc-gpu/src/coordination/mod.rs` - Operational coordination interfaces
    - [ ] `wfc-gpu/src/coordination/propagation.rs` - Propagation strategy coordination

## 4. Advanced Shader Management

- [ ] **Implement shader component system in `shaders.rs`**:

  - [ ] Expand `shaders.rs` to include:
    - [ ] Component registry system
    - [ ] Dependency resolution
    - [ ] Feature flag handling
  - [ ] Create new JSON schema files:
    - [ ] `wfc-gpu/src/shaders/schemas/component.json` - Shader component metadata schema
    - [ ] `wfc-gpu/src/shaders/schemas/feature.json` - Feature capability flags schema

- [ ] **Feature-based shader optimization**:

  - [ ] Create new files:
    - [ ] `wfc-gpu/src/shader_features.rs` - Hardware capability detection
    - [ ] `wfc-gpu/src/features/atomics.rs` - Atomics feature detection & handling
    - [ ] `wfc-gpu/src/features/workgroups.rs` - Workgroup size optimization
  - [ ] Modify `backend.rs`:
    - [ ] Add capability reporting methods
    - [ ] Standardize feature detection across backends

- [ ] **Build-time shader generation**:
  - [ ] Create new build time tools:
    - [ ] `wfc-gpu/tools/shader_optimizer.rs` - Shader optimization tool
    - [ ] `wfc-gpu/tools/shader_validator.rs` - Shader validation tool
  - [ ] Implement caching in `build.rs`:
    - [ ] Shader hash-based caching
    - [ ] Incremental rebuilds of changed components only

## 5. Unify Testing Strategy

- [ ] **Consolidate test modules**:

  - [ ] Enhance `test_utils.rs`:
    - [ ] Move common test setup code from module-specific tests
    - [ ] Create standardized test fixtures
  - [ ] Move module-specific tests:
    - [ ] From `tests.rs` to appropriate module test submodules
    - [ ] From `shader_validation_tests.rs` to `shaders.rs`

- [ ] **Improve shader testing**:
  - [ ] Create new test files:
    - [ ] `wfc-gpu/tests/shaders/component_tests.rs` - Test individual shader components
    - [ ] `wfc-gpu/tests/shaders/variant_tests.rs` - Test assembled shader variants
    - [ ] `wfc-gpu/tests/shaders/sandbox.rs` - Isolated shader testing environment

## 6. Consistent Error Handling

- [ ] **Unify error types**:

  - [ ] Create new error handling module:
    - [ ] `wfc-gpu/src/error/mod.rs` - Unified error system
    - [ ] `wfc-gpu/src/error/gpu_error.rs` - Enhanced GPU errors
    - [ ] `wfc-gpu/src/error/io_error.rs` - File and resource errors
  - [ ] Modify existing error definitions:
    - [ ] `lib.rs` - Update `GpuError` to use new system
    - [ ] `backend.rs` - Update `BackendError` to use new system

- [ ] **Error recovery improvements**:
  - [ ] Enhance `error_recovery.rs`:
    - [ ] Add specialized recovery strategies for common errors
    - [ ] Create standardized recovery interfaces
  - [ ] Add error recovery support to all operation modules:
    - [ ] `accelerator.rs`
    - [ ] `propagator.rs`
    - [ ] `sync.rs`
    - [ ] All new buffer modules

## 7. Reduce Redundancy in Core Algorithm

- [ ] **Extract core algorithm logic**:

  - [ ] Create new algorithm abstraction files:
    - [ ] `wfc-gpu/src/algorithm/propagator_strategy.rs` - Core propagation logic
    - [ ] `wfc-gpu/src/algorithm/entropy_strategy.rs` - Core entropy calculation
  - [ ] Update implementation files:
    - [ ] `propagator.rs` - Use strategy pattern
    - [ ] `entropy.rs` - Use strategy pattern

- [ ] **Standardize workload division**:
  - [ ] Enhance `subgrid.rs`:
    - [ ] Improve subgrid strategy implementation
    - [ ] Add better coordination with main algorithm
  - [ ] Create new file:
    - [ ] `wfc-gpu/src/parallelism.rs` - Parallel execution strategies

## 8. Documentation & API Design

- [ ] **Create clear API boundaries**:

  - [ ] Create documentation files:
    - [ ] `wfc-gpu/docs/api_boundaries.md` - Public vs private API
    - [ ] `wfc-gpu/docs/integration_guide.md` - Integration points
  - [ ] Add examples directory:
    - [ ] `wfc-gpu/examples/basic_usage.rs` - Simple usage example
    - [ ] `wfc-gpu/examples/advanced_features.rs` - Advanced features example

- [ ] **Technical documentation**:
  - [ ] Create documentation files:
    - [ ] `wfc-gpu/docs/buffer_lifecycle.md` - Buffer management documentation
    - [ ] `wfc-gpu/docs/shader_components.md` - Shader component system
  - [ ] Create diagrams:
    - [ ] `wfc-gpu/docs/diagrams/architecture.svg` - Overall architecture
    - [ ] `wfc-gpu/docs/diagrams/buffer_flow.svg` - Buffer lifecycle

## Implementation Strategy

1. **Start with low-risk changes**:

   - Begin with creating the shader component system:
     - Create basic component files in `wfc-gpu/src/shaders/components/`
     - Implement minimal shader registry in `shader_registry.rs`

2. **Progressive refinement**:

   - Implement buffer management changes:
     - Create new buffer module structure
     - Migrate buffer creation code first, then synchronization

3. **Measure impact**:

   - Create benchmarks:
     - `wfc-gpu/benches/propagation_bench.rs`
     - `wfc-gpu/benches/shader_compilation_bench.rs`

4. **Preserve existing improvements**:
   - Create regression test suite:
     - `wfc-gpu/tests/regression/features.rs` - Tests for all features from original TODO
