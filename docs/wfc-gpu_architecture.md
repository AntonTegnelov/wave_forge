# WFC-GPU Architecture

This document describes the architecture of the WFC-GPU module. It serves as a blueprint for the ideal structure and provides guidance for maintaining and extending the codebase.

## 1. High-Level Architecture

The WFC-GPU module provides a GPU-accelerated implementation of the Wave Function Collapse algorithm using WGPU. The architecture follows a layered approach with clear boundaries between components:

```
┌───────────────────────────────────────────────────────────────┐
│                      Public API Layer                         │
│                                                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────┐  │
│  │  GpuAccelerator │  │ EntropyStrategy │  │   Propagator  │  │
│  └─────────────────┘  └─────────────────┘  └───────────────┘  │
└───────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────────────────────────────────────────┐
│                    Coordination Layer                         │
│                                                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────┐  │
│  │   Coordination  │  │  Parallelism    │  │ ErrorRecovery │  │
│  └─────────────────┘  └─────────────────┘  └───────────────┘  │
└───────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────────────────────────────────────────┐
│                    Implementation Layer                       │
│                                                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────┐  │
│  │    Buffers      │  │ GpuSynchronizer │  │   Pipeline    │  │
│  └─────────────────┘  └─────────────────┘  └───────────────┘  │
└───────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────────────────────────────────────────┐
│                     Hardware Layer                            │
│                                                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────┐  │
│  │   GpuBackend    │  │ ShaderRegistry  │  │FeatureDetection│  │
│  └─────────────────┘  └─────────────────┘  └───────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

### Core Design Philosophy

The WFC-GPU module follows these key principles:

1. **Separation of Concerns**: Each component has a single responsibility
2. **Composition over Inheritance**: Components use delegation rather than inheritance
3. **Feature Detection**: Adapt to hardware capabilities seamlessly
4. **Error Recovery**: Handle GPU errors gracefully with recovery mechanisms
5. **Modular Shaders**: Use modular shader components assembled as needed

## 2. Core Components

### 2.1 Public API Layer

#### GpuAccelerator

The main entry point for users, providing a high-level interface to the Wave Function Collapse algorithm.

- **Responsibility**: Coordinate all WFC operations using specialized components
- **Design**: Uses composition to delegate operations to specialized components
- **Extension Point**: Supports plugins for custom entropy calculation and propagation strategies
- **Files**: `gpu/accelerator.rs` - Main accelerator implementation using composition pattern

#### EntropyCalculator

Calculates entropy for cells in the grid and selects the next cell to collapse.

- **Responsibility**: Entropy calculation and cell selection
- **Interface**: Delegates to appropriate entropy strategy implementation
- **Extension Point**: Supports custom entropy heuristics
- **Files**:
  - `entropy/calculator.rs` - Main entropy calculation implementation
  - `entropy/entropy_strategy.rs` - Core entropy calculation strategies

#### Propagator

Propagates constraints after cell collapse by updating neighboring cells.

- **Responsibility**: Constraint propagation following the WFC algorithm
- **Implementation**: Uses specialized propagation strategies
- **Extension Point**: Supports custom propagation approaches and optimization techniques
- **Files**:
  - `propagator/gpu_constraint_propagator.rs` - Main propagator implementation
  - `propagator/propagator_strategy.rs` - Core propagation strategies

### 2.2 Coordination Layer

#### Coordination

Manages the high-level flow of the algorithm and coordinates different operations.

- **Responsibility**: Schedule and manage the different WFC phases
- **Implementation**: Uses strategy pattern for algorithm variants
- **Files**:
  - `coordination/mod.rs` - Core coordination interfaces
  - `coordination/propagation.rs` - Propagation coordination
  - `coordination/entropy.rs` - Entropy calculation coordination

#### Parallelism

Handles distribution of work across multiple GPU workgroups or subgrids.

- **Responsibility**: Efficient parallelization of WFC operations
- **Implementation**: Subgrid decomposition and parallel execution
- **Files**:
  - `utils/subgrid.rs` - Subgrid processing utilities

#### Error Recovery

Provides robust error handling with recovery mechanisms for GPU operations.

- **Responsibility**: Handle GPU errors gracefully, allowing for recovery
- **Implementation**: Strategy pattern for different types of errors
- **Files**:
  - `utils/error_recovery.rs` - Error recovery mechanisms

### 2.3 Implementation Layer

#### Buffers

Manages GPU memory buffers for different data types used in the algorithm.

- **Responsibility**: Create, manage, and cleanup GPU buffers
- **Implementation**: Specialized buffer types for different data
- **Files**:
  - `buffers/mod.rs` - Buffer management facade
  - `buffers/grid_buffers.rs` - Grid state buffers
  - `buffers/worklist_buffers.rs` - Propagation worklist buffers
  - `buffers/entropy_buffers.rs` - Entropy calculation buffers
  - `buffers/rule_buffers.rs` - Adjacency rule buffers

#### GpuSynchronizer

Handles synchronized data transfer between CPU and GPU.

- **Responsibility**: Manage CPU-GPU data synchronization
- **Implementation**: Optimized transfer operations with retry capabilities
- **Files**: `gpu/sync.rs` - Data transfer methods

#### Pipeline

Creates and manages WGPU compute pipelines for shader execution.

- **Responsibility**: Compute pipeline creation and management
- **Implementation**: Creates pipelines from shader components
- **Files**: `shader/pipeline.rs` - Pipeline creation and binding

### 2.4 Hardware Layer

#### GpuBackend

Abstracts away the underlying GPU API details.

- **Responsibility**: Hardware abstraction and feature reporting
- **Implementation**: Adapts to different GPU backends (primarily WGPU)
- **Files**: `gpu/backend.rs` - Backend implementation

#### ShaderRegistry

Manages shader components and their assembly into complete shaders.

- **Responsibility**: Shader component registration and assembly
- **Implementation**: Component-based shader management system
- **Files**:
  - `shader/shader_registry.rs` - Registry for shader components
  - `shader/shader_compiler.rs` - Shader preprocessing & assembly
  - `shader/shaders.rs` - Core shader functionality

#### Feature Detection

Detects GPU hardware capabilities and adapts the implementation accordingly.

- **Responsibility**: GPU capability detection and adaptation
- **Implementation**: Feature detection and feature flag management
- **Files**:
  - `gpu/backend.rs` - Hardware capability detection

## 3. Shader Component System

The shader system is built around modular components that are assembled into complete shaders based on hardware capabilities and requirements.

### 3.1 Component Architecture

```
┌───────────────────────────────────────────────────────────┐
│                  Component Registry                       │
│                                                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────┐  │
│  │   Base Components│  │Feature Components│  │  Variants│  │
│  └─────────────────┘  └─────────────────┘  └───────────┘  │
└───────────────────────────────────────────────────────────┘
                         │
┌───────────────────────────────────────────────────────────┐
│                  Shader Compiler                          │
│                                                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────┐  │
│  │ Component Loader│  │ Dependency Resolver│ │Assembler │  │
│  └─────────────────┘  └─────────────────┘  └───────────┘  │
└───────────────────────────────────────────────────────────┘
                         │
┌───────────────────────────────────────────────────────────┐
│                  Feature Detection                        │
│                                                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────┐  │
│  │ Capability Query│  │ Feature Mapping │  │Constraints│  │
│  └─────────────────┘  └─────────────────┘  └───────────┘  │
└───────────────────────────────────────────────────────────┘
```

### 3.2 Shader Components

Shader code is organized into highly modular components:

- **Base Components**: Core algorithm logic (entropy, propagation)
- **Utility Components**: Common utilities (coords, math, bit operations)
- **Feature Components**: Hardware-specific implementations (atomics, workgroups)
- **Generated Variants**: Complete shaders assembled from components

#### Structure

```
wfc-gpu/src/shader/
├── shaders/           # Core shader components
│   ├── entropy_calculation.wgsl
│   ├── worklist_management.wgsl
│   ├── cell_collapse.wgsl
│   ├── contradiction_detection.wgsl
├── pipeline.rs        # Pipeline creation
├── shader_compiler.rs # Shader compilation
├── shader_registry.rs # Component registry
└── mod.rs             # Module exports
```

### 3.4 Build Process

The build process generates optimized shader variants:

- **Build Script**: `wfc-gpu/build.rs` handles pre-build shader generation

## 4. Buffer Management

The buffer system handles GPU memory allocation and management with specialized components for different data types.

### 4.1 Buffer Organization

```
┌───────────────────────────────────────────────────────────┐
│                     Buffer Modules                        │
│                                                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────┐  │
│  │   Grid Buffers  │  │ Entropy Buffers │  │ Worklists │  │
│  └─────────────────┘  └─────────────────┘  └───────────┘  │
└───────────────────────────────────────────────────────────┘
                         │
┌───────────────────────────────────────────────────────────┐
│                     Buffer Facade                         │
│                                                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────┐  │
│  │  Creation/Sizing│  │Dynamic Resizing │  │Management │  │
│  └─────────────────┘  └─────────────────┘  └───────────┘  │
└───────────────────────────────────────────────────────────┘
                         │
┌───────────────────────────────────────────────────────────┐
│                   GPU Synchronizer                        │
│                                                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────┐  │
│  │  Data Transfer  │  │ Synchronization │  │ Recovery  │  │
│  └─────────────────┘  └─────────────────┘  └───────────┘  │
└───────────────────────────────────────────────────────────┘
```

### 4.2 Buffer Types

- **Grid Buffers**: Store cell possibility states - `buffers/grid_buffers.rs`
- **Entropy Buffers**: Store entropy values and min-entropy information - `buffers/entropy_buffers.rs`
- **Worklist Buffers**: Manage lists of cells to update during propagation - `buffers/worklist_buffers.rs`
- **Rule Buffers**: Store adjacency rules and constraints - `buffers/rule_buffers.rs`

### 4.3 Synchronization Responsibilities

The `GpuSynchronizer` (`gpu/sync.rs`) handles all data transfer between CPU and GPU, providing:

- **Uploading**: CPU data → GPU buffers
- **Downloading**: GPU buffers → CPU data
- **Synchronization**: Waiting for GPU operations to complete
- **Recovery**: Handling transfer errors with retry mechanisms

## 5. Algorithm Implementation

The core WFC algorithm is implemented as a set of interchangeable strategies for different aspects of the algorithm.

### 5.1 Entropy Calculation Strategies

```
┌───────────────────────────────────────────────────────────┐
│                   Entropy Calculator                       │
│                                                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────┐  │
│  │Shannon Entropy  │  │   Count-based   │  │  Weighted │  │
│  └─────────────────┘  └─────────────────┘  └───────────┘  │
└───────────────────────────────────────────────────────────┘
```

Strategies follow the Strategy pattern and include:

- **Shannon Entropy**: Classic entropy calculation
- **Count-based**: Simpler heuristic based on possibility count
- **Weighted**: Takes tile weights into account

Implemented in `entropy/entropy_strategy.rs`.

### 5.2 Propagation Strategies

```
┌───────────────────────────────────────────────────────────┐
│                  Propagation Strategies                    │
│                                                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────┐  │
│  │ Direct Propagation│  │Subgrid Propagation│ │Optimized │  │
│  └─────────────────┘  └─────────────────┘  └───────────┘  │
└───────────────────────────────────────────────────────────┘
```

Strategies include:

- **Direct Propagation**: Standard propagation algorithm
- **Subgrid Propagation**: Divides grid into manageable chunks
- **Adaptive Strategies**: Hardware-specific optimizations

Implemented in `propagator/propagator_strategy.rs`.

### 5.3 Coordination

The algorithm coordinator manages the overall flow:

1. Calculate entropy for all cells
2. Select cell with minimum entropy
3. Collapse the selected cell
4. Propagate constraints
5. Repeat until completion or contradiction

Implemented in `coordination/mod.rs` and coordination submodules.

### 5.4 Algorithm Implementation Files

- `propagator/propagator_strategy.rs` - Core propagation logic
- `entropy/entropy_strategy.rs` - Core entropy calculation
- `coordination/mod.rs` - Operational coordination interfaces
- `coordination/propagation.rs` - Propagation strategy coordination
- `coordination/entropy.rs` - Entropy calculation coordination

## 6. Extension Guidelines

To extend the WFC-GPU library, follow these guidelines:

1. **Adding new entropy strategies**:

   - Implement the `EntropyStrategy` trait in `entropy/entropy_strategy.rs`
   - Register with `EntropyStrategyFactory`

2. **Adding new propagation strategies**:

   - Implement the `PropagationStrategy` trait in `propagator/propagator_strategy.rs`
   - Register with `PropagationStrategyFactory`

3. **Adding new shader components**:

   - Add component to appropriate location in `shader/` directory
   - Register with `ShaderRegistry`

4. **Adding new buffer types**:
   - Add new module in `buffers/` directory
   - Implement buffer creation, resizing, and synchronization

## 7. Testing Strategy

### 7.1 Testing Structure

Tests are organized into multiple categories:

- **Unit Tests**: Tests for individual components
  - Located within each module
  - Focus on testing isolated behavior
- **Integration Tests**:

  - Located in `wfc-gpu/tests/` directory
  - Test interactions between components

- **Shader Tests**:
  - Validate shader compilation and execution
  - Test shader components

## 8. Common Pitfalls and Best Practices

### 8.1 Common Pitfalls

- **Error Handling**: Don't ignore GPU errors; use the recovery system
- **Buffer Management**: Don't create new buffer types without synchronization support
- **Shader Components**: Don't duplicate functionality across shader components
- **Performance Optimization**: Don't optimize prematurely; measure first

### 8.2 Best Practices

- **Use Composition**: Prefer delegation to inheritance
- **Feature Detection**: Always check hardware capabilities before using features
- **Buffer Lifecycle**: Follow the documented buffer lifecycle
- **Error Recovery**: Use standardized error recovery mechanisms
- **Testing**: Write tests for both components and integration

## 9. Integration Example

Here's an example of how the components work together:

```rust
// Create a GpuAccelerator with default settings
let mut accelerator = GpuAccelerator::new(&initial_grid, &rules, BoundaryCondition::Periodic, None).await?;

// Configure with specific strategies
accelerator
    .with_entropy_heuristic(EntropyHeuristicType::Shannon)
    .with_parallel_subgrid_processing(SubgridConfig::default())
    .enable_default_debug_visualization();

// Run the algorithm with progress callback
let result = accelerator.run_with_callback(
    &mut grid,
    &rules,
    |progress| {
        println!("Progress: {}%", progress * 100.0);
        true // continue
    }
).await?;
```

The internals coordinate automatically:

1. `GpuAccelerator` delegates to the right components
2. The shader system assembles optimal shader variants
3. Buffers are managed automatically
4. Error recovery handles any GPU issues

By maintaining this architecture, the codebase remains clean, maintainable, and highly extensible without reimplementing existing functionality.
