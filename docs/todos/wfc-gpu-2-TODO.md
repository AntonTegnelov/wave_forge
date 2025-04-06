# WFC-GPU Refactoring Plan

This document outlines a specific plan to refactor the wfc-gpu module, addressing code duplication and architectural issues while preserving the improvements from the original TODO list.

## 1. Shader Modularization

- [x] **Create truly modular shader components**:

  - [x] **Create new directory structure in `wfc-gpu/src/shaders/`**:
    - [x] `wfc-gpu/src/shaders/components/` - Base components
    - [x] `wfc-gpu/src/shaders/features/` - Feature-specific implementations
    - [x] `wfc-gpu/src/shaders/variants/` - Generated shader variants (build outputs)
  - [x] **Extract from existing shaders into component files**:
    - [x] `components/entropy_calculation.wgsl` (from entropy\*.wgsl)
    - [x] `components/worklist_management.wgsl` (from propagate\*.wgsl)
    - [x] `components/cell_collapse.wgsl` (new file)
    - [x] `components/contradiction_detection.wgsl` (from propagate\*.wgsl)
  - [x] **Keep but refactor core utility files**:
    - [x] `utils.wgsl` - Keep only truly generic utilities
    - [x] `coords.wgsl` - Coordinate system operations
    - [x] `rules.wgsl` - Adjacency rule handling
  - [x] **Update files to reference new components**:
    - [x] Update `pipeline.rs`: Add module component loading
    - [x] Update `shaders.rs`: Reference new component paths

- [x] **Build shader assembly system**:

  - [x] Create new files:
    - [x] `wfc-gpu/src/shader_compiler.rs` - Shader preprocessing & assembly system
    - [x] `wfc-gpu/src/shader_registry.rs` - Registry for shader components & features
  - [x] Create build script:
    - [x] `wfc-gpu/build.rs` - Pre-build shader generation
  - [x] Modify `shaders.rs`:
    - [x] Expand from 12 lines to a full shader management system
    - [x] Implement the shader variant loading interface
  - [x] **Update files to use new shader system**:
    - [x] Update `lib.rs`: Add new module imports
    - [x] Update `pipeline.rs`: Use new shader compiler
    - [x] Update `accelerator.rs`: Reference shader registry for feature detection

- [ ] **Eliminate duplication across shader files**:
  - [x] Eventually remove redundant files:
    - [x] `entropy_fallback.wgsl`, `entropy_modular.wgsl` (after component extraction)
    - [x] `propagate_fallback.wgsl`, `propagate_modular.wgsl` (after component extraction)
  - [x] Create new component registry:
    - [x] `wfc-gpu/src/shaders/components/registry.json` - Component metadata & dependencies
  - [ ] **Update shader loading paths**:
    - [x] Update `pipeline.rs`: Remove direct inclusion of shader files
    - [x] Update build scripts to use component registry

## 2. Clear Responsibility Boundaries

- [ ] **Buffer management vs. synchronization**:

  - [x] Modify `sync.rs`: Move all data transfer methods from `GpuBuffers` to `GpuSynchronizer`
  - [x] Modify `sync.rs`: Add new transfer methods for remaining buffer types
  - [x] Modify `buffers.rs`: Remove data transfer methods
  - [x] Create new documentation file:
    - [x] `wfc-gpu/docs/buffer_lifecycle.md` - Explain buffer ownership & synchronization
  - [ ] **Update files to use new buffer management methods**:
    - [x] Update `accelerator.rs`: Use `GpuSynchronizer` for all data transfers (Partially done, `run_with_callback` needs manual fixes)
    - [x] Update `propagator.rs`: Ensure correct use of `GpuSynchronizer`
    - [x] Update `entropy.rs`: Ensure correct use of `GpuSynchronizer` (GPU dispatch in `calculate_entropy` still TODO)

- [ ] **Entropy calculation**:

  - [ ] Modify `entropy.rs`:
    - [ ] Enhance `GpuEntropyCalculator` to handle all entropy calculation logic
  - [ ] Modify `accelerator.rs`:
    - [ ] Remove direct entropy calculation implementation
    - [ ] Change `EntropyCalculator` trait impl to delegate to `GpuEntropyCalculator`
  - [ ] **Update files relying on entropy calculation**:
    - [ ] Update `tests.rs`: Use centralized entropy calculation
    - [ ] Update any code in `propagator.rs` that touches entropy calculation

- [ ] **Pipeline management**:
  - [ ] Modify `pipeline.rs`:
    - [ ] Remove shader loading code
    - [ ] Focus only on pipeline creation and binding
  - [ ] Expand `shaders.rs`:
    - [ ] Add shader loading & preprocessing logic from `pipeline.rs`
    - [ ] Implement shader variant management
  - [ ] **Update files using the pipeline management**:
    - [ ] Update `accelerator.rs`: Use new shader loading interfaces
    - [ ] Update `tests.rs`: Use new shader loading methods
    - [ ] Update `debug_viz.rs`: If it uses shaders

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
  - [ ] **Update files using GpuBuffers**:
    - [ ] Update `accelerator.rs`: Import from new buffer modules
    - [ ] Update `propagator.rs`: Use specific buffer modules
    - [ ] Update `entropy.rs`: Use entropy-specific buffers
    - [ ] Update `sync.rs`: Reference new buffer module structure
    - [ ] Update all tests and validation code

- [ ] **Revise `GpuAccelerator`**:
  - [ ] Modify `accelerator.rs`:
    - [ ] Convert to use composition rather than direct implementation
    - [ ] Create delegation methods for all WFC operations
    - [ ] Reduce Arc nesting
  - [ ] Create new coordination files:
    - [ ] `wfc-gpu/src/coordination/mod.rs` - Operational coordination interfaces
    - [ ] `wfc-gpu/src/coordination/propagation.rs` - Propagation strategy coordination
  - [ ] **Update files using GpuAccelerator**:
    - [ ] Update `lib.rs`: Export new coordinator traits
    - [ ] Update `tests.rs`: Use new coordinator interfaces
    - [ ] Update integration test files to use new abstractions

## 4. Advanced Shader Management

- [ ] **Implement shader component system in `shaders.rs`**:

  - [ ] Expand `shaders.rs` to include:
    - [ ] Component registry system
    - [ ] Dependency resolution
    - [ ] Feature flag handling
  - [ ] Create new JSON schema files:
    - [ ] `wfc-gpu/src/shaders/schemas/component.json` - Shader component metadata schema
    - [ ] `wfc-gpu/src/shaders/schemas/feature.json` - Feature capability flags schema
  - [ ] **Update files using shader management**:
    - [ ] Update `pipeline.rs`: Use component registry
    - [ ] Update `build.rs`: Use schema validation
    - [ ] Update any test files using shaders

- [ ] **Feature-based shader optimization**:

  - [ ] Create new files:
    - [ ] `wfc-gpu/src/shader_features.rs` - Hardware capability detection
    - [ ] `wfc-gpu/src/features/atomics.rs` - Atomics feature detection & handling
    - [ ] `wfc-gpu/src/features/workgroups.rs` - Workgroup size optimization
  - [ ] Modify `backend.rs`:
    - [ ] Add capability reporting methods
    - [ ] Standardize feature detection across backends
  - [ ] **Update files using feature detection**:
    - [ ] Update `accelerator.rs`: Use feature detection for initialization
    - [ ] Update `pipeline.rs`: Select shader variants based on features
    - [ ] Update `shader_compiler.rs`: Include feature-specific code

- [ ] **Build-time shader generation**:
  - [ ] Create new build time tools:
    - [ ] `wfc-gpu/tools/shader_optimizer.rs` - Shader optimization tool
    - [ ] `wfc-gpu/tools/shader_validator.rs` - Shader validation tool
  - [ ] Implement caching in `build.rs`:
    - [ ] Shader hash-based caching
    - [ ] Incremental rebuilds of changed components only
  - [ ] **Update project configuration**:
    - [ ] Update `Cargo.toml`: Add build script and build dependencies
    - [ ] Update `.gitignore`: Exclude generated shader variant files
    - [ ] Update CI workflow files to handle shader generation

## 5. Unify Testing Strategy

- [ ] **Consolidate test modules**:

  - [ ] Enhance `test_utils.rs`:
    - [ ] Move common test setup code from module-specific tests
    - [ ] Create standardized test fixtures
  - [ ] Move module-specific tests:
    - [ ] From `tests.rs` to appropriate module test submodules
    - [ ] From `shader_validation_tests.rs` to `shaders.rs`
  - [ ] **Update test importing files**:
    - [ ] Update `lib.rs`: Change test module references
    - [ ] Update each module file to include its own tests

- [ ] **Improve shader testing**:
  - [ ] Create new test files:
    - [ ] `wfc-gpu/tests/shaders/component_tests.rs` - Test individual shader components
    - [ ] `wfc-gpu/tests/shaders/variant_tests.rs` - Test assembled shader variants
    - [ ] `wfc-gpu/tests/shaders/sandbox.rs` - Isolated shader testing environment
  - [ ] **Update existing shader tests**:
    - [ ] Update `shader_validation_tests.rs` (before moving to `shaders.rs`)
    - [ ] Update any shader tests in `pipeline.rs`
    - [ ] Update `Cargo.toml` to include new test directories

## 6. Consistent Error Handling

- [ ] **Unify error types**:

  - [ ] Create new error handling module:
    - [ ] `wfc-gpu/src/error/mod.rs` - Unified error system
    - [ ] `wfc-gpu/src/error/gpu_error.rs` - Enhanced GPU errors
    - [ ] `wfc-gpu/src/error/io_error.rs` - File and resource errors
  - [ ] Modify existing error definitions:
    - [ ] `lib.rs` - Update `GpuError` to use new system
    - [ ] `backend.rs` - Update `BackendError` to use new system
  - [ ] **Update files using error types**:
    - [ ] Update all modules using `GpuError` or `BackendError`
    - [ ] Update `accelerator.rs`, `propagator.rs`, `sync.rs`, `entropy.rs`
    - [ ] Update all buffer modules to use new error system

- [ ] **Error recovery improvements**:
  - [ ] Enhance `error_recovery.rs`:
    - [ ] Add specialized recovery strategies for common errors
    - [ ] Create standardized recovery interfaces
  - [ ] Add error recovery support to all operation modules:
    - [ ] `accelerator.rs`
    - [ ] `propagator.rs`
    - [ ] `sync.rs`
    - [ ] All new buffer modules
  - [ ] **Update error-generating methods**:
    - [ ] Update buffer methods to use recovery strategies
    - [ ] Update shader loading to use recovery
    - [ ] Update propagation to handle recoverable errors

## 7. Reduce Redundancy in Core Algorithm

- [ ] **Extract core algorithm logic**:

  - [ ] Create new algorithm abstraction files:
    - [ ] `wfc-gpu/src/algorithm/propagator_strategy.rs` - Core propagation logic
    - [ ] `wfc-gpu/src/algorithm/entropy_strategy.rs` - Core entropy calculation
  - [ ] Update implementation files:
    - [ ] `propagator.rs` - Use strategy pattern
    - [ ] `entropy.rs` - Use strategy pattern
  - [ ] **Update files using algorithm logic**:
    - [ ] Update `accelerator.rs`: Use strategy interfaces
    - [ ] Update any test files that directly use algorithm logic
    - [ ] Update `debug_viz.rs` if it interacts with algorithm

- [ ] **Standardize workload division**:
  - [ ] Enhance `subgrid.rs`:
    - [ ] Improve subgrid strategy implementation
    - [ ] Add better coordination with main algorithm
  - [ ] Create new file:
    - [ ] `wfc-gpu/src/parallelism.rs` - Parallel execution strategies
  - [ ] **Update files using workload division**:
    - [ ] Update `accelerator.rs`: Use parallelism strategies
    - [ ] Update `propagator.rs`: Implement parallelism interfaces
    - [ ] Update benchmark files to test different workload strategies

## 8. Documentation & API Design

- [ ] **Create clear API boundaries**:

  - [ ] Create documentation files:
    - [ ] `wfc-gpu/docs/api_boundaries.md` - Public vs private API
    - [ ] `wfc-gpu/docs/integration_guide.md` - Integration points
  - [ ] Add examples directory:
    - [ ] `wfc-gpu/examples/basic_usage.rs` - Simple usage example
    - [ ] `wfc-gpu/examples/advanced_features.rs` - Advanced features example
  - [ ] **Update exports and visibility**:
    - [ ] Update `lib.rs`: Adjust public exports
    - [ ] Update module files to use correct visibility modifiers
    - [ ] Ensure backward compatibility for existing users

- [ ] **Technical documentation**:
  - [ ] Create documentation files:
    - [ ] `wfc-gpu/docs/buffer_lifecycle.md` - Buffer management documentation
    - [ ] `wfc-gpu/docs/shader_components.md` - Shader component system
  - [ ] Create diagrams:
    - [ ] `wfc-gpu/docs/diagrams/architecture.svg` - Overall architecture
    - [ ] `wfc-gpu/docs/diagrams/buffer_flow.svg` - Buffer lifecycle
  - [ ] **Add internal documentation**:
    - [ ] Add rustdoc comments to all public APIs
    - [ ] Add architecture explanation to module headers
    - [ ] Cross-reference between related components

## Implementation Strategy

1. **Start with low-risk changes**:

   - Begin with creating the shader component system:
     - Create basic component files in `wfc-gpu/src/shaders/components/`
     - Implement minimal shader registry in `shader_registry.rs`
   - **Integration points**:
     - Update `pipeline.rs` to use new component files
     - Update `shaders.rs` with registry hooks

2. **Progressive refinement**:

   - Implement buffer management changes:
     - Create new buffer module structure
     - Migrate buffer creation code first, then synchronization
   - **Integration points**:
     - Update `accelerator.rs` to use new buffer modules
     - Update `sync.rs` to work with new buffer organization
     - Adjust `propagator.rs` and `entropy.rs` to use specialized buffers

3. **Measure impact**:

   - Create benchmarks:
     - `wfc-gpu/benches/propagation_bench.rs`
     - `wfc-gpu/benches/shader_compilation_bench.rs`
   - **Integration points**:
     - Add benchmark hooks in key operations
     - Create performance tracking mechanisms
     - Update `Cargo.toml` to include benchmark harness

4. **Preserve existing improvements**:
   - Create regression test suite:
     - `wfc-gpu/tests/regression/features.rs` - Tests for all features from original TODO
   - **Integration points**:
     - Add test hooks to verify all existing functionality
     - Ensure backward compatibility for public APIs
     - Update documentation to explain both old and new approaches during transition

- [ ] Implement GPU entropy calculation in entropy.rs
  - [x] Dispatch compute shader in calculate_entropy (in GpuAccelerator, needs import fixes)
  - [ ] Download entropy grid results
