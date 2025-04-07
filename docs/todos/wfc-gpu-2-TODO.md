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

- [x] **Eliminate duplication across shader files**:
  - [x] Eventually remove redundant files:
    - [x] `entropy_fallback.wgsl`, `entropy_modular.wgsl` (after component extraction)
    - [x] `propagate_fallback.wgsl`, `propagate_modular.wgsl` (after component extraction)
  - [x] Create new component registry:
    - [x] `wfc-gpu/src/shaders/components/registry.json` - Component metadata & dependencies
  - [x] **Update shader loading paths**:
    - [x] Update `pipeline.rs`: Remove direct inclusion of shader files
    - [x] Update build scripts to use component registry

## 2. Clear Responsibility Boundaries

- [x] **Buffer management vs. synchronization**:

  - [x] Modify `sync.rs`: Move all data transfer methods from `GpuBuffers` to `GpuSynchronizer`
  - [x] Modify `sync.rs`: Add new transfer methods for remaining buffer types
  - [x] Modify `buffers.rs`: Remove data transfer methods
  - [x] Create new documentation file:
    - [x] `wfc-gpu/docs/buffer_lifecycle.md` - Explain buffer ownership & synchronization
  - [x] **Update files to use new buffer management methods**:
    - [x] Update `accelerator.rs`: Use `GpuSynchronizer` for all data transfers (Partially done, `run_with_callback` needs manual fixes)
    - [x] Update `propagator.rs`: Access buffers via new structs (e.g., `buffers.worklist_buffers`)
    - [x] Update `entropy.rs`: Access buffers via new structs (e.g., `buffers.entropy_buffers`)
    - [x] Update `sync.rs`: Reference new buffer module structure
    - [x] Update all tests and validation code

- [x] **Entropy calculation**:

  - [x] Modify `entropy.rs`:
    - [x] Enhance `GpuEntropyCalculator` to handle all entropy calculation logic
  - [x] Modify `accelerator.rs`:
    - [x] Remove direct entropy calculation implementation
    - [x] Change `EntropyCalculator` trait impl to delegate to `GpuEntropyCalculator`
  - [x] **Update files relying on entropy calculation**:
    - [x] Update `tests.rs`: Use centralized entropy calculation
    - [x] Update any code in `propagator.rs` that touches entropy calculation

- [x] **Pipeline management**:
  - [x] Modify `pipeline.rs`:
    - [x] Remove shader loading code
    - [x] Focus only on pipeline creation and binding
  - [x] Expand `shaders.rs`:
    - [x] Add shader loading & preprocessing logic from `pipeline.rs`
    - [x] Implement shader variant management
  - [x] **Update files using the pipeline management**:
    - [x] Update `accelerator.rs`: Use new shader loading interfaces
    - [x] Update `tests.rs`: Use new shader loading methods
    - [x] Update `debug_viz.rs`: If it uses shaders

## 3. Simplify Complex Structures

- [x] **Break up `GpuBuffers`**:

  - [x] Create new files:
    - [x] `wfc-gpu/src/buffers/grid_buffers.rs` - Grid state buffers
    - [x] `wfc-gpu/src/buffers/worklist_buffers.rs` - Propagation worklist buffers
    - [x] `wfc-gpu/src/buffers/entropy_buffers.rs` - Entropy calculation buffers
    - [x] `wfc-gpu/src/buffers/rule_buffers.rs` - Adjacency rule buffers
    - [x] `wfc-gpu/src/buffers/mod.rs` - Facade implementation & common utilities
  - [x] Modify existing `buffers.rs`:
    - [x] Move code to appropriate new files
  - [x] **Update files using GpuBuffers**:
    - [x] Update `accelerator.rs`: Import from new buffer modules
    - [x] Update `propagator.rs`: Access buffers via new structs (e.g., `buffers.worklist_buffers`)
    - [x] Update `entropy.rs`: Access buffers via new structs (e.g., `buffers.entropy_buffers`)
    - [x] Update `sync.rs`: Reference new buffer module structure
    - [x] Update all tests and validation code

- [x] **Revise `GpuAccelerator`**:
  - [x] Modify `accelerator.rs`:
    - [x] Convert to use composition rather than direct implementation
    - [x] Create delegation methods for all WFC operations
    - [x] Reduce Arc nesting
  - [ ] Create new coordination files:
    - [x] `wfc-gpu/src/coordination/mod.rs` - Operational coordination interfaces
    - [ ] `wfc-gpu/src/coordination/propagation.rs` - Propagation strategy coordination
  - [ ] **Update files using GpuAccelerator**:
    - [ ] Update `accelerator.rs`: Use new shader loading interfaces
    - [ ] Update `tests.rs`: Use new shader loading methods
    - [ ] Update `debug_viz.rs`: If it uses shaders
