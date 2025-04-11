# WFC-GPU Architecture Alignment Plan

This document outlines a specific plan to align the wfc-gpu module with its intended architecture, addressing the gaps identified in the code review while building on the refactoring work already completed.

## 1. Complete Algorithm Strategy Pattern Implementation

- [x] **Enhance Entropy Calculation Strategies**:

  - [x] **Create strategy interface for entropy calculations**:
    - [x] `wfc-gpu/src/entropy/entropy_strategy.rs` - Define core entropy calculation strategies
    - [x] Extract existing entropy implementations into separate strategy classes
    - [x] Implement Shannon, Count, CountSimple, and WeightedCount strategies
  - [x] **Modify entropy calculator to use strategies**:
    - [x] Update `entropy/calculator.rs` to delegate to specific strategies
    - [x] Add factory method to create appropriate strategy based on configuration
  - [x] **Update files to use enhanced entropy strategies**:
    - [x] Update `gpu/accelerator.rs` to configure specific entropy strategies
    - [x] Update `coordination/mod.rs` to work with strategy pattern

  **Implementation Details**:

  - **Files to modify**:

    - `wfc-gpu/src/entropy/calculator.rs`:

      - Refactor to use strategy pattern - extract heuristic-specific logic (~150 lines) to strategy implementations
      - Modify `GpuEntropyCalculator::calculate_entropy_async` to delegate to selected strategy
      - Add factory method for strategy creation
      - Update `GpuEntropyCalculator::with_heuristic` to accept a strategy instance

    - `wfc-gpu/src/lib.rs`:

      - Add re-export for the new entropy strategy types
      - Update the module references to reflect new structure

    - `wfc-gpu/src/gpu/accelerator.rs`:

      - Update `GpuAccelerator::new` (~line 124) to configure the entropy calculator with appropriate strategy
      - Add a method to customize entropy strategy after initialization
      - Update imports to use entropy strategy types

    - `wfc-gpu/src/buffers/entropy_buffers.rs`:
      - Add buffer specializations for different entropy heuristics if needed
      - Ensure buffer layouts align with strategy implementations

  - **New directories/files**:
    - ~~`wfc-gpu/src/algorithm/` - Create new directory~~ (Now exists at `wfc-gpu/src/entropy/`)
    - ~~`wfc-gpu/src/algorithm/mod.rs` - Register submodules~~ (Now at `wfc-gpu/src/entropy/mod.rs`)
    - ~~`wfc-gpu/src/algorithm/entropy_strategy.rs` - Create strategy interface and implementations~~ (Now at `wfc-gpu/src/entropy/entropy_strategy.rs`)

- [x] **Complete Propagation Strategy Pattern**:

  - [x] **Create dedicated strategy files**:
    - [x] `wfc-gpu/src/propagator/propagator_strategy.rs` - Core propagation strategies
    - [x] Extract direct propagation logic from `propagator/gpu_constraint_propagator.rs`
    - [x] Extract subgrid propagation logic into its own strategy
  - [x] **Implement additional propagation strategies**:
    - [x] `DirectPropagationStrategy` - Standard propagation approach
    - [x] `SubgridPropagationStrategy` - For large grid optimization
    - [x] `AdaptivePropagationStrategy` - Selects optimal strategy based on grid size
  - [x] **Update propagator to use strategy pattern**:
    - [x] Modify `propagator/gpu_constraint_propagator.rs` to delegate to strategies
    - [x] Reduce file size by extracting strategy implementations
    - [x] Update strategy selection based on configuration

  **Implementation Details**:

  - **Files to modify**:

    - `wfc-gpu/src/propagator/gpu_constraint_propagator.rs` (~29KB, 783 lines):

      - Reduce file size by ~60% by extracting strategy-specific code
      - Convert remaining code to use the Facade/Strategy pattern
      - Modify `GpuConstraintPropagator` to delegate to strategy implementations
      - Update public API to maintain backward compatibility
      - Add factory methods for strategy selection

    - `wfc-gpu/src/coordination/propagation.rs`:

      - Update to work with the new strategy pattern
      - Modify `PropagationCoordinator` interfaces to accept strategy configurations
      - Ensure `DirectPropagationCoordinator` and `SubgridPropagationCoordinator` select appropriate strategies

    - `wfc-gpu/src/utils/subgrid.rs`:

      - Extract subgrid-specific propagation logic to `propagator/propagator_strategy.rs` in `SubgridPropagationStrategy`
      - Update `SubgridConfig` to work with strategy pattern

    - `wfc-gpu/src/gpu/accelerator.rs`:
      - Update `GpuAccelerator` to configure propagator with the right strategy
      - Add methods to customize propagation strategy after initialization
      - Update imports to use propagator strategy types

  - **Dependencies**:
    - This change must be coordinated with shader changes to ensure WGSL code matches the strategy implementations
    - Ensure performance testing before/after conversion to validate no regressions

- [x] **Add Algorithm Coordination Strategies**:

  - [x] **Create coordination strategy interface**:
    - [x] `wfc-gpu/src/coordination/strategy.rs` - Define coordination strategies
    - [x] `wfc-gpu/src/coordination/entropy.rs` - Add entropy coordination

  **Implementation Details**:

  - **Files to modify**:

    - `wfc-gpu/src/coordination/mod.rs`:

      - [x] Formalize the `CoordinationStrategy` trait (currently just a placeholder on line 47)
      - [x] Update `WfcCoordinator` to leverage the strategy pattern
      - [x] Add new coordinator implementations using the strategy pattern
      - [x] Implement strategy factory methods

    - `wfc-gpu/src/coordination/propagation.rs`:

      - [x] Update to use the new coordination strategy interfaces
      - [x] Implement specific strategy for propagation coordination

    - `wfc-gpu/src/gpu/accelerator.rs`:
      - [x] Update to configure the right coordination strategy
      - [x] Add method to select coordination strategy

  - **New files**:

    - [x] `wfc-gpu/src/coordination/strategy.rs` - Core coordination strategy interfaces
    - [x] `wfc-gpu/src/coordination/entropy.rs` - Entropy-specific coordination

  - **Dependencies**:
    - [x] Must be implemented after entropy and propagation strategies are completed
    - [x] Coordination strategies should compose existing strategies rather than duplicate logic

## 2. Enhance Shader Component System

- [x] **Improve shader component modularity**:

  - [x] **Add more fine-grained shader components**:
    - [x] `shader/shaders/components/entropy/shannon.wgsl` - Shannon entropy specific code
    - [x] `shader/shaders/components/entropy/count_based.wgsl` - Count-based entropy
    - [x] `shader/shaders/components/propagation/direct.wgsl` - Direct propagation
    - [x] `shader/shaders/components/propagation/subgrid.wgsl` - Subgrid propagation
  - [x] **Update registry to track component dependencies**:
    - [x] Enhance component metadata to include explicit dependencies
    - [x] Add version information to components
    - [x] Add capability requirements to components

  **Implementation Details**:

  - **Files to modify**:

    - `wfc-gpu/src/shader/shaders/components/registry.json`:

      - [x] Expand to include explicit dependencies between components
      - [x] Add version fields for all components
      - [x] Add GPU capability requirements

    - `wfc-gpu/src/shader/shader_registry.rs`:

      - [x] Update to parse and validate enhanced component metadata
      - [x] Add dependency resolution and version checks
      - [x] Modify component loading to handle nested structure

    - `wfc-gpu/src/shader/shader_compiler.rs`:

      - [x] Update to assemble modular components based on dependencies
      - [x] Add support for conditional compilation based on features
      - [x] Ensure robust error reporting for missing dependencies

    - `wfc-gpu/src/shader/shaders.rs`:

      - [x] Update shader loading code to work with the enhanced component system
      - [x] Add versioning support for shader components

    - `wfc-gpu/build.rs`:
      - [x] Update build process to validate shader component dependencies
      - [x] Add component validation during build

  - **New files**:

    - [x] Multiple new WGSL component files in specialized subdirectories:
      - [x] `wfc-gpu/src/shader/shaders/components/entropy/`
      - [x] `wfc-gpu/src/shader/shaders/components/propagation/`
      - [x] `wfc-gpu/src/shader/shaders/components/entropy/shannon.wgsl`, etc.

  - **Dependencies**:
    - [x] Must be coordinated with algorithm strategy implementations to ensure shader components match algorithm implementations

- [x] **Create dedicated feature detection files**:

  - [x] `wfc-gpu/src/gpu/features/mod.rs` - Feature detection system
  - [x] `wfc-gpu/src/gpu/features/atomics.rs` - Atomics support detection
  - [x] `wfc-gpu/src/gpu/features/workgroups.rs` - Workgroup capabilities

- [x] **Enhance shader compilation with feature flags**:

  - [x] Modify `shader/shader_compiler.rs` to use feature detection
  - [x] Add conditional compilation in shaders based on features
  - [x] Implement fallback paths for less capable hardware

  **Implementation Details**:

  - **Files to modify**:

    - `wfc-gpu/src/gpu/backend.rs`:

      - [ ] Enhance feature detection and reporting
      - [ ] Add methods to query specific capability groups
      - [ ] Update to report comprehensive feature information

    - `wfc-gpu/src/shader/shader_compiler.rs`:

      - [x] Update to use feature flags during shader compilation
      - [x] Add conditional path selection based on available features
      - [x] Implement fallback mechanism for missing features

    - `wfc-gpu/src/gpu/accelerator.rs`:
      - [ ] Update initialization to detect and configure for hardware capabilities
      - [ ] Add adaptive behavior based on detected features

  - **New files/directories**:

    - [ ] `wfc-gpu/src/gpu/features/` - New directory
    - [ ] `wfc-gpu/src/gpu/features/mod.rs` - Feature detection system
    - [ ] `wfc-gpu/src/gpu/features/atomics.rs` - Atomics support
    - [ ] `wfc-gpu/src/gpu/features/workgroups.rs` - Workgroup capabilities

  - **Dependencies**:
    - [x] Must be implemented before shader optimizations
    - [x] Requires coordination with backend changes

- [x] **Build advanced shader optimization system**:

  - [x] **Create shader optimization tools**:
    - [x] `wfc-gpu/tools/shader_optimizer.rs` - Shader optimization tool
    - [x] `wfc-gpu/tools/shader_validator.rs` - Shader validation tool
  - [x] **Enhance build script**:
    - [x] Update `build.rs` for shader optimization
    - [x] Add validation during build process
    - [x] Generate optimized variants for common configurations

  **Implementation Details**:

  - **Files to modify**:

    - `wfc-gpu/build.rs`:

      - [x] Enhance to run shader optimization during build
      - [x] Add validation step for shader components
      - [x] Generate optimized variants for common configurations
      - [x] Add proper error reporting during build

    - `wfc-gpu/src/shader/shaders.rs`:
      - [x] Update to load optimized shader variants
      - [x] Add fallback mechanism for missing variants
      - [x] Improve error reporting for shader loading

  - **New files/directories**:

    - [x] `wfc-gpu/tools/` - New directory for build tools
    - [x] `wfc-gpu/tools/shader_optimizer.rs` - Optimization tool
    - [x] `wfc-gpu/tools/shader_validator.rs` - Validation tool
    - [x] `wfc-gpu/src/shader/shaders/variants/` - Directory for generated shader variants

  - **Dependencies**:
    - [x] Requires feature detection system to be implemented first
    - [x] Optimization should happen after shader component system is enhanced

## 3. Decouple Error Handling and Recovery

- [x] **Create dedicated error recovery system**:

  - [x] **Refine error types**:
    - [x] `wfc-gpu/src/utils/error/mod.rs` - Unified error system
    - [x] `wfc-gpu/src/utils/error/gpu_error.rs` - Enhanced GPU errors
    - [x] `wfc-gpu/src/utils/error/io_error.rs` - File and resource errors
  - [x] **Implement recovery strategies**:
    - [x] `wfc-gpu/src/utils/error_recovery/strategies.rs` - Error recovery strategies
    - [x] Create retry strategy for transient GPU errors
    - [x] Create fallback strategy for hardware limitations
    - [x] Create graceful degradation for recoverable errors

  **Implementation Details**:

  - **Files to modify**:

    - `wfc-gpu/src/utils/error_recovery.rs` (21KB, 583 lines):

      - [x] Extract types to dedicated module files
      - [x] Move `GpuError` enum to `error/gpu_error.rs`
      - [x] Move recovery logic to strategy-based implementation
      - [x] Update to use the Strategy pattern

    - `wfc-gpu/src/lib.rs`:

      - [x] Update module declarations and re-exports
      - [x] Add the new error module to the module tree

    - Files with error handling code (e.g., `gpu/accelerator.rs`, `propagator/gpu_constraint_propagator.rs`, `entropy/calculator.rs`):
      - [x] Update to use the new error types
      - [x] Replace direct error handling with strategy-based recovery

  - **New files/directories**:

    - [x] `wfc-gpu/src/utils/error/` - New directory
    - [x] `wfc-gpu/src/utils/error/mod.rs` - Main error module
    - [x] `wfc-gpu/src/utils/error/gpu_error.rs` - GPU-specific errors
    - [x] `wfc-gpu/src/utils/error/io_error.rs` - I/O and resource errors
    - [x] `wfc-gpu/src/utils/error_recovery/mod.rs` - Refactored from existing file
    - [x] `wfc-gpu/src/utils/error_recovery/strategies.rs` - Recovery strategies

  - **Dependencies**:
    - [ ] Error handling changes should be made before other major refactoring
    - [ ] All other modules will need to be updated to use the new error types

- [x] **Enhance error reporting**:

  - [x] **Improve diagnostic information**:
    - [x] Add location context to errors
    - [x] Capture GPU state for debugging
    - [x] Add suggested solutions to common errors
  - [x] **Update error interfaces**:
    - [x] Make errors more actionable by user code
    - [x] Ensure consistent error types across the library
    - [x] Add recovery hooks for user-defined recovery logic

  **Implementation Details**:

  - **Files to modify**:

    - `wfc-gpu/src/utils/error_recovery.rs` and/or new `error` module files:

      - [x] Add context fields to error types (location, state info)
      - [x] Add helper methods for diagnostic reporting
      - [x] Implement user-defined recovery hooks

    - Error-producing code (most files):

      - [x] Update error creation to include context information
      - [x] Use consistent error mapping conventions
      - [x] Capture relevant state information at error sites

    - `wfc-gpu/src/gpu/accelerator.rs`:
      - [x] Add methods for registering user-defined recovery hooks
      - [x] Improve error reporting to application code

  - **Specific inconsistencies to address**:
    - [x] In `entropy/calculator.rs`, `From<GpuError> for CoreEntropyError` conversion uses different mapping than in other files
    - [x] `gpu/accelerator.rs` uses both `GpuError` and `WfcError` for similar failures
    - [ ] `propagator/gpu_constraint_propagator.rs` uses custom error conversions
    - [ ] `utils/error_recovery.rs` inconsistently handles buffers across error types

## 4. Hardware Abstraction Improvements

- [ ] **Enhance backend abstraction**:

  - [ ] **Create adapter system for different backends**:
    - [ ] `wfc-gpu/src/gpu/backend/wgpu.rs` - WGPU-specific implementation
    - [ ] `wfc-gpu/src/gpu/backend/mock.rs` - Mock backend for testing
    - [ ] Extract backend interface from implementation
  - [ ] **Add capability negotiation**:
    - [ ] Create explicit capability reporting
    - [ ] Implement capability queries for high-level code
    - [ ] Add adaptive behavior based on capabilities

  **Implementation Details**:

  - **Files to modify**:

    - `wfc-gpu/src/gpu/backend.rs` (17KB, 516 lines):

      - [ ] Extract interface to separate file
      - [ ] Move WGPU implementation to separate file
      - [ ] Add capability negotiation system
      - [ ] Remove direct wgpu dependencies from interface

    - `wfc-gpu/src/lib.rs`:

      - [ ] Update module declarations for backend
      - [ ] Update re-exports for backend types

    - `wfc-gpu/src/gpu/accelerator.rs`:
      - [ ] Update to use backend abstraction for device creation
      - [ ] Add checks for required capabilities
      - [ ] Add adaptive behavior based on capabilities

  - **New files/directories**:

    - [ ] `wfc-gpu/src/gpu/backend/` - New directory
    - [ ] `wfc-gpu/src/gpu/backend/mod.rs` - Main backend module with traits
    - [ ] `wfc-gpu/src/gpu/backend/wgpu.rs` - WGPU implementation
    - [ ] `wfc-gpu/src/gpu/backend/mock.rs` - Mock implementation for testing

  - **Dependencies**:
    - [ ] Should be implemented alongside or after feature detection
    - [ ] Will significantly impact many other modules that currently use wgpu directly

## 5. Testing and Documentation

- [ ] **Enhance test coverage**:

  - [ ] **Improve unit test organization**:

    - [ ] `wfc-gpu/src/tests/entropy_strategies.rs` - Unit tests for entropy strategies
    - [ ] `wfc-gpu/src/tests/propagation_strategies.rs` - Unit tests for propagation strategies
    - [ ] `wfc-gpu/src/tests/error_recovery.rs` - Tests for error recovery system

  - [ ] **Add integration tests for coordinator strategies**:
    - [ ] `
