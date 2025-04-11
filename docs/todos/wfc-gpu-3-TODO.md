# WFC-GPU Architecture Alignment Plan

This document outlines a specific plan to align the wfc-gpu module with its intended architecture, addressing the gaps identified in the code review while building on the refactoring work already completed.

## 1. Complete Algorithm Strategy Pattern Implementation

- [ ] **Enhance Entropy Calculation Strategies**:

  - [ ] **Create strategy interface for entropy calculations**:
    - [x] `wfc-gpu/src/algorithm/entropy_strategy.rs` - Define core entropy calculation strategies
    - [x] Extract existing entropy implementations into separate strategy classes
    - [x] Implement Shannon, Count, CountSimple, and WeightedCount strategies
  - [ ] **Modify entropy calculator to use strategies**:
    - [x] Update `entropy.rs` to delegate to specific strategies
    - [x] Add factory method to create appropriate strategy based on configuration
  - [ ] **Update files to use enhanced entropy strategies**:
    - [x] Update `accelerator.rs` to configure specific entropy strategies
    - [x] Update `coordination/mod.rs` to work with strategy pattern

  **Implementation Details**:

  - **Files to modify**:

    - `wfc-gpu/src/entropy.rs`:

      - Refactor to use strategy pattern - extract heuristic-specific logic (~150 lines) to strategy implementations
      - Modify `GpuEntropyCalculator::calculate_entropy_async` to delegate to selected strategy
      - Add factory method for strategy creation
      - Update `GpuEntropyCalculator::with_heuristic` to accept a strategy instance

    - `wfc-gpu/src/lib.rs`:

      - Add re-export for the new entropy strategy types
      - Add the new `algorithm` module to the module tree

    - `wfc-gpu/src/accelerator.rs`:

      - Update `GpuAccelerator::new` (~line 124) to configure the entropy calculator with appropriate strategy
      - Add a method to customize entropy strategy after initialization

    - `wfc-gpu/src/buffers/entropy_buffers.rs`:
      - Add buffer specializations for different entropy heuristics if needed
      - Ensure buffer layouts align with strategy implementations

  - **New directories/files**:
    - `wfc-gpu/src/algorithm/` - Create new directory
    - `wfc-gpu/src/algorithm/mod.rs` - Register submodules
    - `wfc-gpu/src/algorithm/entropy_strategy.rs` - Create strategy interface and implementations

- [ ] **Complete Propagation Strategy Pattern**:

  - [ ] **Create dedicated strategy files**:
    - [ ] `wfc-gpu/src/algorithm/propagator_strategy.rs` - Core propagation strategies
    - [ ] Extract direct propagation logic from `propagator.rs`
    - [ ] Extract subgrid propagation logic into its own strategy
  - [ ] **Implement additional propagation strategies**:
    - [ ] `DirectPropagationStrategy` - Standard propagation approach
    - [ ] `SubgridPropagationStrategy` - For large grid optimization
    - [ ] `AdaptivePropagationStrategy` - Selects optimal strategy based on grid size
  - [ ] **Update propagator to use strategy pattern**:
    - [ ] Modify `propagator.rs` to delegate to strategies
    - [ ] Reduce file size by extracting strategy implementations
    - [ ] Update strategy selection based on configuration

  **Implementation Details**:

  - **Files to modify**:

    - `wfc-gpu/src/propagator.rs` (~36KB, 923 lines):

      - Reduce file size by ~60% by extracting strategy-specific code
      - Convert remaining code to use the Facade/Strategy pattern
      - Modify `GpuConstraintPropagator` to delegate to strategy implementations
      - Update public API to maintain backward compatibility
      - Add factory methods for strategy selection

    - `wfc-gpu/src/coordination/propagation.rs`:

      - Update to work with the new strategy pattern
      - Modify `PropagationCoordinator` interfaces to accept strategy configurations
      - Ensure `DirectPropagationCoordinator` and `SubgridPropagationCoordinator` select appropriate strategies

    - `wfc-gpu/src/subgrid.rs`:

      - Extract subgrid-specific propagation logic to `SubgridPropagationStrategy`
      - Update `SubgridConfig` to work with strategy pattern

    - `wfc-gpu/src/accelerator.rs`:
      - Update `GpuAccelerator` to configure propagator with the right strategy
      - Add methods to customize propagation strategy after initialization

  - **Dependencies**:
    - This change must be coordinated with shader changes to ensure WGSL code matches the strategy implementations
    - Ensure performance testing before/after conversion to validate no regressions

- [ ] **Add Algorithm Coordination Strategies**:

  - [ ] **Create coordination strategy interface**:
    - [ ] `wfc-gpu/src/coordination/strategy.rs` - Define coordination strategies
    - [ ] `wfc-gpu/src/coordination/entropy.rs` - Add entropy coordination
  - [ ] **Implement coordination strategies**:
    - [ ] Standard coordination strategy (current implementation)
    - [ ] Parallel coordination strategy (for multi-GPU or large grids)
    - [ ] Add explicit factory methods for creating different strategies

  **Implementation Details**:

  - **Files to modify**:

    - `wfc-gpu/src/coordination/mod.rs`:

      - Formalize the `CoordinationStrategy` trait (currently just a placeholder on line 47)
      - Update `WfcCoordinator` to leverage the strategy pattern
      - Add new coordinator implementations using the strategy pattern
      - Implement strategy factory methods

    - `wfc-gpu/src/coordination/propagation.rs`:

      - Update to use the new coordination strategy interfaces
      - Implement specific strategy for propagation coordination

    - `wfc-gpu/src/accelerator.rs`:
      - Update to configure the right coordination strategy
      - Add method to select coordination strategy

  - **New files**:

    - `wfc-gpu/src/coordination/strategy.rs` - Core coordination strategy interfaces
    - `wfc-gpu/src/coordination/entropy.rs` - Entropy-specific coordination

  - **Dependencies**:
    - Must be implemented after entropy and propagation strategies are completed
    - Coordination strategies should compose existing strategies rather than duplicate logic

## 2. Enhance Shader Component System

- [ ] **Improve shader component modularity**:

  - [ ] **Add more fine-grained shader components**:
    - [ ] `components/entropy/shannon.wgsl` - Shannon entropy specific code
    - [ ] `components/entropy/count_based.wgsl` - Count-based entropy
    - [ ] `components/propagation/direct.wgsl` - Direct propagation
    - [ ] `components/propagation/subgrid.wgsl` - Subgrid propagation
  - [ ] **Update registry to track component dependencies**:
    - [ ] Enhance component metadata to include explicit dependencies
    - [ ] Add version information to components
    - [ ] Add capability requirements to components

  **Implementation Details**:

  - **Files to modify**:

    - `wfc-gpu/src/shaders/components/registry.json`:

      - Expand to include explicit dependencies between components
      - Add version fields for all components
      - Add GPU capability requirements

    - `wfc-gpu/src/shader_registry.rs`:

      - Update to parse and validate enhanced component metadata
      - Add dependency resolution and version checks
      - Modify component loading to handle nested structure

    - `wfc-gpu/src/shader_compiler.rs`:

      - Update to assemble modular components based on dependencies
      - Add support for conditional compilation based on features
      - Ensure robust error reporting for missing dependencies

    - `wfc-gpu/src/shaders.rs`:

      - Update shader loading code to work with the enhanced component system
      - Add versioning support for shader components

    - `wfc-gpu/build.rs`:
      - Update build process to validate shader component dependencies
      - Add component validation during build

  - **New files**:

    - Multiple new WGSL component files in specialized subdirectories:
      - `wfc-gpu/src/shaders/components/entropy/`
      - `wfc-gpu/src/shaders/components/propagation/`
      - `wfc-gpu/src/shaders/components/entropy/shannon.wgsl`, etc.

  - **Dependencies**:
    - Must be coordinated with algorithm strategy implementations to ensure shader components match algorithm implementations

- [ ] **Implement comprehensive feature detection**:

  - [ ] **Create dedicated feature detection files**:
    - [ ] `wfc-gpu/src/features/mod.rs` - Feature detection system
    - [ ] `wfc-gpu/src/features/atomics.rs` - Atomics support detection
    - [ ] `wfc-gpu/src/features/workgroups.rs` - Workgroup capabilities
  - [ ] **Enhance shader compilation with feature flags**:
    - [ ] Modify `shader_compiler.rs` to use feature detection
    - [ ] Add conditional compilation in shaders based on features
    - [ ] Implement fallback paths for less capable hardware

  **Implementation Details**:

  - **Files to modify**:

    - `wfc-gpu/src/backend.rs`:

      - Enhance feature detection and reporting
      - Add methods to query specific capability groups
      - Update to report comprehensive feature information

    - `wfc-gpu/src/shader_compiler.rs`:

      - Update to use feature flags during shader compilation
      - Add conditional path selection based on available features
      - Implement fallback mechanism for missing features

    - `wfc-gpu/src/accelerator.rs`:
      - Update initialization to detect and configure for hardware capabilities
      - Add adaptive behavior based on detected features

  - **New files/directories**:

    - `wfc-gpu/src/features/` - New directory
    - `wfc-gpu/src/features/mod.rs` - Feature detection system
    - `wfc-gpu/src/features/atomics.rs` - Atomics support
    - `wfc-gpu/src/features/workgroups.rs` - Workgroup capabilities

  - **Dependencies**:
    - Must be implemented before shader optimizations
    - Requires coordination with backend changes

- [ ] **Build advanced shader optimization system**:

  - [ ] **Create shader optimization tools**:
    - [ ] `wfc-gpu/tools/shader_optimizer.rs` - Shader optimization tool
    - [ ] `wfc-gpu/tools/shader_validator.rs` - Shader validation tool
  - [ ] **Enhance build script**:
    - [ ] Update `build.rs` for shader optimization
    - [ ] Add validation during build process
    - [ ] Generate optimized variants for common configurations

  **Implementation Details**:

  - **Files to modify**:

    - `wfc-gpu/build.rs`:

      - Enhance to run shader optimization during build
      - Add validation step for shader components
      - Generate optimized variants for common configurations
      - Add proper error reporting during build

    - `wfc-gpu/src/shaders.rs`:
      - Update to load optimized shader variants
      - Add fallback mechanism for missing variants
      - Improve error reporting for shader loading

  - **New files/directories**:

    - `wfc-gpu/tools/` - New directory for build tools
    - `wfc-gpu/tools/shader_optimizer.rs` - Optimization tool
    - `wfc-gpu/tools/shader_validator.rs` - Validation tool
    - `wfc-gpu/src/shaders/variants/` - Directory for generated shader variants

  - **Dependencies**:
    - Requires feature detection system to be implemented first
    - Optimization should happen after shader component system is enhanced

## 3. Decouple Error Handling and Recovery

- [ ] **Create dedicated error recovery system**:

  - [ ] **Refine error types**:
    - [ ] `wfc-gpu/src/error/mod.rs` - Unified error system
    - [ ] `wfc-gpu/src/error/gpu_error.rs` - Enhanced GPU errors
    - [ ] `wfc-gpu/src/error/io_error.rs` - File and resource errors
  - [ ] **Implement recovery strategies**:
    - [ ] `wfc-gpu/src/error_recovery/strategies.rs` - Error recovery strategies
    - [ ] Create retry strategy for transient GPU errors
    - [ ] Create fallback strategy for hardware limitations
    - [ ] Create graceful degradation for recoverable errors

  **Implementation Details**:

  - **Files to modify**:

    - `wfc-gpu/src/error_recovery.rs` (21KB, 583 lines):

      - Extract types to dedicated module files
      - Move `GpuError` enum to `error/gpu_error.rs`
      - Move recovery logic to strategy-based implementation
      - Update to use the Strategy pattern

    - `wfc-gpu/src/lib.rs`:

      - Update module declarations and re-exports
      - Add the new error module to the module tree

    - Files with error handling code (e.g., `accelerator.rs`, `propagator.rs`, `entropy.rs`):
      - Update to use the new error types
      - Replace direct error handling with strategy-based recovery

  - **New files/directories**:

    - `wfc-gpu/src/error/` - New directory
    - `wfc-gpu/src/error/mod.rs` - Main error module
    - `wfc-gpu/src/error/gpu_error.rs` - GPU-specific errors
    - `wfc-gpu/src/error/io_error.rs` - I/O and resource errors
    - `wfc-gpu/src/error_recovery/mod.rs` - Refactored from existing file
    - `wfc-gpu/src/error_recovery/strategies.rs` - Recovery strategies

  - **Dependencies**:
    - Error handling changes should be made before other major refactoring
    - All other modules will need to be updated to use the new error types

- [ ] **Enhance error reporting**:

  - [ ] **Improve diagnostic information**:
    - [ ] Add location context to errors
    - [ ] Capture GPU state for debugging
    - [ ] Add suggested solutions to common errors
  - [ ] **Update error interfaces**:
    - [ ] Make errors more actionable by user code
    - [ ] Ensure consistent error types across the library
    - [ ] Add recovery hooks for user-defined recovery logic

  **Implementation Details**:

  - **Files to modify**:

    - `wfc-gpu/src/error_recovery.rs` and/or new `error` module files:

      - Add context fields to error types (location, state info)
      - Add helper methods for diagnostic reporting
      - Implement user-defined recovery hooks

    - Error-producing code (most files):

      - Update error creation to include context information
      - Use consistent error mapping conventions
      - Capture relevant state information at error sites

    - `wfc-gpu/src/accelerator.rs`:
      - Add methods for registering user-defined recovery hooks
      - Improve error reporting to application code

  - **Specific inconsistencies to address**:
    - In `entropy.rs`, `From<GpuError> for CoreEntropyError` conversion uses different mapping than in other files
    - `accelerator.rs` uses both `GpuError` and `WfcError` for similar failures
    - `propagator.rs` uses custom error conversions
    - `error_recovery.rs` inconsistently handles buffers across error types

## 4. Hardware Abstraction Improvements

- [ ] **Enhance backend abstraction**:

  - [ ] **Create adapter system for different backends**:
    - [ ] `wfc-gpu/src/backend/wgpu.rs` - WGPU-specific implementation
    - [ ] `wfc-gpu/src/backend/mock.rs` - Mock backend for testing
    - [ ] Extract backend interface from implementation
  - [ ] **Add capability negotiation**:
    - [ ] Create explicit capability reporting
    - [ ] Implement capability queries for high-level code
    - [ ] Add adaptive behavior based on capabilities

  **Implementation Details**:

  - **Files to modify**:

    - `wfc-gpu/src/backend.rs` (17KB, 516 lines):

      - Extract interface to separate file
      - Move WGPU implementation to separate file
      - Add capability negotiation system
      - Remove direct wgpu dependencies from interface

    - `wfc-gpu/src/lib.rs`:

      - Update module declarations for backend
      - Update re-exports for backend types

    - `wfc-gpu/src/accelerator.rs`:
      - Update to use backend abstraction for device creation
      - Add checks for required capabilities
      - Add adaptive behavior based on capabilities

  - **New files/directories**:

    - `wfc-gpu/src/backend/` - New directory
    - `wfc-gpu/src/backend/mod.rs` - Main backend module with traits
    - `wfc-gpu/src/backend/wgpu.rs` - WGPU implementation
    - `wfc-gpu/src/backend/mock.rs` - Mock implementation for testing

  - **Dependencies**:
    - Should be implemented alongside or after feature detection
    - Will significantly impact many other modules that currently use wgpu directly

## 5. Testing and Documentation

- [ ] **Enhance test coverage**:

  - [ ] **Improve unit test organization**:

    - [ ] `wfc-gpu/src/tests/entropy_strategies.rs` - Unit tests for entropy strategies
    - [ ] `wfc-gpu/src/tests/propagation_strategies.rs` - Unit tests for propagation strategies
    - [ ] `wfc-gpu/src/tests/error_recovery.rs` - Tests for error recovery system

  - [ ] **Add integration tests for coordinator strategies**:
    - [ ] `wfc-gpu/tests/integration/coordinator_tests.rs` - Tests for coordination strategies
    - [ ] `wfc-gpu/tests/integration/strategy_composition_tests.rs` - Tests for strategy composition

  **Implementation Details**:

  - **Files to modify**:

    - `wfc-gpu/src/tests.rs` (7.9KB, 229 lines):

      - Refactor into multiple files with focused tests
      - Add tests for new components and strategies
      - Add error handling testing

    - `wfc-gpu/src/test_utils.rs`:
      - Enhance with mock implementations
      - Add test helpers for strategy testing
      - Add fixtures for common test cases

- [ ] **Update documentation**:

  - [ ] **Update API documentation**:

    - [ ] Document strategy interfaces
    - [ ] Update error handling documentation

  - [ ] **Create architecture documentation**:
    - [ ] `wfc-gpu/docs/architecture.md` - Updated architecture guide
    - [ ] `wfc-gpu/docs/error-handling.md` - Error handling guide
    - [ ] `wfc-gpu/docs/customization.md` - Strategy customization guide

  **Implementation Details**:

  - **Files to modify**:
    - Code documentation in most files
    - Update inline documentation to reflect new architecture
  - **New files**:
    - Architecture and usage documentation files

## 6. Public API Refactoring

- [ ] **Restrict Module Visibility in lib.rs**:

  - [ ] **Change most module declarations to pub(crate)**:
    - [ ] `pub(crate) mod buffers;`
    - [ ] `pub(crate) mod coordination;`
    - [ ] `pub(crate) mod entropy;`
    - [ ] `pub(crate) mod pipeline;`
    - [ ] `pub(crate) mod propagator;`
    - [ ] `pub(crate) mod shader_compiler;`
    - [ ] `pub(crate) mod shader_registry;`
    - [ ] `pub(crate) mod shaders;`
    - [ ] `pub(crate) mod subgrid;`
    - [ ] `pub(crate) mod sync;`
    - [ ] `pub(crate) mod backend;`
  - [ ] **Keep necessary modules public**:
    - [ ] `pub mod accelerator;` (Contains the main entry point)
    - [ ] `pub mod error_recovery;` (Contains the public error types)
    - [ ] `pub mod debug_viz;` (If debug visualization is intended as a public feature)

  **Implementation Details**:

  - **Files to modify**:
    - `wfc-gpu/src/lib.rs`:
      - Update all module declarations as specified above
      - Adjust pub use statements to only expose the intended public API
      - Ensure backward compatibility with existing client code

- [ ] **Adjust pub use statements in lib.rs**:

  - [ ] **Keep essential public re-exports**:
    - [ ] `pub use accelerator::GpuAccelerator;`
    - [ ] `pub use error_recovery::{GpuError, GridCoord};`
    - [ ] `pub use accelerator::{GridDefinition, GridStats, WfcRunResult};`
    - [ ] `pub use subgrid::SubgridConfig;` (If needed for configuration)
    - [ ] `pub use debug_viz::{DebugVisualizationConfig, DebugVisualizer, VisualizationType};` (If debug viz is public)
  - [ ] **Remove internal implementation re-exports**:
    - [ ] Remove re-exports of `GpuBuffers`, `ComputePipelines`, `GpuConstraintPropagator`, `GpuSynchronizer`, etc.
    - [ ] Remove re-exports of coordination types
    - [ ] Remove re-exports of internal buffer types

  **Implementation Details**:

  - **Files to modify**:
    - `wfc-gpu/src/lib.rs`:
      - Audit all `pub use` statements against the desired public API
      - Remove any that expose internal implementation details
      - Add `#[doc(hidden)]` to types that must remain public for technical reasons but shouldn't be in public docs

- [ ] **Restrict Visibility within Individual Modules**:

  - [ ] **Modify accelerator.rs**:

    - [ ] Make `AcceleratorInstance` `pub(crate)`
    - [ ] Make methods exposing internal types (`backend()`, `pipelines()`, `buffers()`) `pub(crate)`
    - [ ] Keep public: `GpuAccelerator`, `GridDefinition`, `GridStats`, `WfcRunResult`, core methods

  - [ ] **Modify buffers/ module and submodules**:

    - [ ] Change visibility of all public structs (`GpuBuffers`, `EntropyBuffers`, etc.) to `pub(crate)`
    - [ ] Change visibility of all public enums and functions to `pub(crate)`
    - [ ] Keep `DynamicBufferConfig` public only if part of the public API

  - [ ] **Modify coordination/ module and submodules**:

    - [ ] Change visibility of all public structs/traits/enums to `pub(crate)`

  - [ ] **Modify debug_viz.rs**:

    - [ ] If public: Retain public visibility only for config/control types
    - [ ] Make `DebugSnapshot` `pub(crate)`

  - [ ] **Modify error_recovery.rs**:

    - [ ] Keep `GpuError` and `GridCoord` public
    - [ ] Make `ErrorSeverity`, `GpuErrorRecovery`, `AdaptiveTimeoutConfig`, `RecoverableGpuOp` `pub(crate)`

  - [ ] **Modify other modules (entropy.rs, pipeline.rs, propagator.rs, etc.)**:
    - [ ] Change all public structs, traits, enums, and functions to `pub(crate)`
    - [ ] Exception: Keep `SubgridConfig` public (if needed in public API)

  **Implementation Details**:

  - **Files to modify**:
    - All source files in the `wfc-gpu` module
    - For each file:
      - Audit all `pub` declarations
      - Change to `pub(crate)` for implementation details
      - Keep public only types/functions intended for the public API
      - Ensure changes maintain backward compatibility with existing code

- [ ] **Add Public API Design Documentation**:

  - [ ] **Create public API documentation**:
    - [ ] `wfc-gpu/docs/public_api.md` - Document the public API design and usage
    - [ ] Add examples of proper API usage
  - [ ] **Update existing documentation**:
    - [ ] Update docstrings to reflect the new public/private distinction
    - [ ] Mark internal-use-only APIs with appropriate documentation annotations
