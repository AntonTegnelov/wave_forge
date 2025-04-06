# WFC-GPU Architecture

This document describes the architecture of the WFC-GPU module after completing the refactoring outlined in the `wfc-gpu-2-TODO.md` document. It serves as a blueprint for the ideal structure and provides guidance for maintaining and extending the codebase.

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

#### EntropyCalculator

Calculates entropy for cells in the grid and selects the next cell to collapse.

- **Responsibility**: Entropy calculation and cell selection
- **Interface**: Delegates to appropriate entropy strategy implementation
- **Extension Point**: Supports custom entropy heuristics

#### Propagator

Propagates constraints after cell collapse by updating neighboring cells.

- **Responsibility**: Constraint propagation following the WFC algorithm
- **Implementation**: Uses specialized propagation strategies
- **Extension Point**: Supports custom propagation approaches and optimization techniques

### 2.2 Coordination Layer

#### Coordination

Manages the high-level flow of the algorithm and coordinates different operations.

- **Responsibility**: Schedule and manage the different WFC phases
- **Implementation**: Uses strategy pattern for algorithm variants
- **Files**:
  - `coordination/mod.rs` - Core coordination interfaces
  - `coordination/propagation.rs` - Propagation coordination

#### Parallelism

Handles distribution of work across multiple GPU workgroups or subgrids.

- **Responsibility**: Efficient parallelization of WFC operations
- **Implementation**: Subgrid decomposition and parallel execution
- **Files**:
  - `parallelism.rs` - Parallel execution strategies
  - `subgrid.rs` - Subgrid processing implementation

#### Error Recovery

Provides robust error handling with recovery mechanisms for GPU operations.

- **Responsibility**: Handle GPU errors gracefully, allowing for recovery
- **Implementation**: Strategy pattern for different types of errors
- **Files**:
  - `error/mod.rs` - Unified error system
  - `error_recovery.rs` - Recovery strategies

### 2.3 Implementation Layer

#### Buffers

Manages GPU memory buffers for different data types used in the algorithm.

- **Responsibility**: Create, manage, and cleanup GPU buffers
- **Implementation**: Specialized buffer types for different data
- **Files**:
  - `buffers/mod.rs` - Facade implementation & common utilities
  - `buffers/grid_buffers.rs` - Grid state buffers
  - `buffers/worklist_buffers.rs` - Propagation worklist buffers
  - `buffers/entropy_buffers.rs` - Entropy calculation buffers
  - `buffers/rule_buffers.rs` - Adjacency rule buffers

#### GpuSynchronizer

Handles synchronized data transfer between CPU and GPU.

- **Responsibility**: Manage CPU-GPU data synchronization
- **Implementation**: Optimized transfer operations with retry capabilities
- **Files**: `sync.rs`

#### Pipeline

Creates and manages WGPU compute pipelines for shader execution.

- **Responsibility**: Compute pipeline creation and management
- **Implementation**: Creates pipelines from shader components
- **Files**: `pipeline.rs`

### 2.4 Hardware Layer

#### GpuBackend

Abstracts away the underlying GPU API details.

- **Responsibility**: Hardware abstraction and feature reporting
- **Implementation**: Adapts to different GPU backends (primarily WGPU)
- **Files**: `backend.rs`

#### ShaderRegistry

Manages shader components and their assembly into complete shaders.

- **Responsibility**: Shader component registration and assembly
- **Implementation**: Component-based shader management system
- **Files**:
  - `shaders.rs` - Main shader management
  - `shader_compiler.rs` - Shader preprocessing
  - `shader_registry.rs` - Component registry

#### Feature Detection

Detects GPU hardware capabilities and adapts the implementation accordingly.

- **Responsibility**: GPU capability detection and adaptation
- **Implementation**: Feature detection and feature flag management
- **Files**:
  - `shader_features.rs` - Hardware capability detection
  - `features/atomics.rs` - Atomics support
  - `features/workgroups.rs` - Workgroup optimization

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

### 3.3 Directory Structure

```
wfc-gpu/src/shaders/
├── components/           # Core shader components
│   ├── entropy_calculation.wgsl
│   ├── worklist_management.wgsl
│   ├── cell_collapse.wgsl
│   ├── contradiction_detection.wgsl
│   └── registry.json     # Component metadata & dependencies
├── features/             # Feature-specific implementations
│   ├── atomics.wgsl
│   ├── no_atomics.wgsl
│   └── workgroup_opt.wgsl
├── variants/             # Generated shader variants (build outputs)
├── schemas/              # JSON schemas for metadata
├── utils.wgsl            # Common utilities
├── coords.wgsl           # Coordinate system operations
└── rules.wgsl            # Adjacency rule handling
```

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

- **Grid Buffers**: Store cell possibility states
- **Entropy Buffers**: Store entropy values and min-entropy information
- **Worklist Buffers**: Manage lists of cells to update during propagation
- **Rule Buffers**: Store adjacency rules and constraints

### 4.3 Synchronization Responsibilities

The `GpuSynchronizer` handles all data transfer between CPU and GPU, providing:

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
- **Optimized Strategies**: Hardware-specific optimizations

### 5.3 Coordination

The algorithm coordinator manages the overall flow:

1. Calculate entropy for all cells
2. Select cell with minimum entropy
3. Collapse the selected cell
4. Propagate constraints
5. Repeat until completion or contradiction

## 6. Extension Guidelines

### 6.1 Adding a New Entropy Heuristic

1. Create a new implementation in `algorithm/entropy_strategy.rs`
2. Register the strategy in `entropy.rs`
3. Add selection code in `GpuEntropyCalculator`
4. Update shader component in `shaders/components/entropy_calculation.wgsl`

### 6.2 Adding a New Shader Component

1. Create the component file in `shaders/components/`
2. Update component metadata in `registry.json`
3. Add feature detection if needed in `shader_features.rs`
4. Update the shader compiler to use the new component

### 6.3 Supporting a New Hardware Feature

1. Add detection code in `features/` directory
2. Create feature-specific shader components
3. Update shader assembly to conditionally include the feature
4. Add fallback implementation for hardware without the feature

### 6.4 Adding GPU Buffer Types

1. Create a new buffer module in `buffers/`
2. Implement creation and management logic
3. Add synchronization methods to `GpuSynchronizer`
4. Update the buffer facade to expose the new buffer type

## 7. Common Pitfalls and Best Practices

### 7.1 Common Pitfalls

- **Error Handling**: Don't ignore GPU errors; use the recovery system
- **Buffer Management**: Don't create new buffer types without synchronization support
- **Shader Components**: Don't duplicate functionality across shader components
- **Performance Optimization**: Don't optimize prematurely; measure first

### 7.2 Best Practices

- **Use Composition**: Prefer delegation to inheritance
- **Feature Detection**: Always check hardware capabilities before using features
- **Buffer Lifecycle**: Follow the documented buffer lifecycle
- **Error Recovery**: Use standardized error recovery mechanisms
- **Testing**: Write tests for both components and integration

## 8. Integration Example

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
