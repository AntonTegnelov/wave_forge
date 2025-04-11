# WFC-GPU Strategy Customization Guide

This guide explains how to customize and extend the algorithmic strategies used in the Wave Function Collapse GPU implementation.

## Overview of Strategy Pattern

The WFC-GPU module employs the Strategy pattern to make algorithmic components customizable and extensible. This pattern allows you to:

1. Swap implementations of key algorithms without changing the overall structure
2. Create custom implementations tailored to specific use cases
3. Combine different strategies to optimize for particular scenarios

## Available Strategy Types

### Entropy Calculation Strategies

Entropy strategies determine how to calculate entropy and select the next cell to collapse:

```rust
// Example: Using a custom entropy strategy
let accelerator = GpuAccelerator::new(&grid_def, &rules, BoundaryCondition::Periodic, None)
    .await?
    .with_entropy_heuristic(EntropyHeuristicType::Shannon);

// Or with a fully custom implementation
let custom_strategy = MyCustomEntropyStrategy::new(weights);
let accelerator = accelerator.with_custom_entropy_strategy(custom_strategy);
```

#### Built-in Entropy Strategies

| Strategy        | Description                             | Best Use Case                               |
| --------------- | --------------------------------------- | ------------------------------------------- |
| `Shannon`       | Classic Shannon entropy calculation     | General purpose, balances speed and quality |
| `CountBased`    | Simple count of remaining possibilities | Better performance, slightly lower quality  |
| `WeightedCount` | Count-based with tile weights           | When tiles have varying frequencies         |

### Propagation Strategies

Propagation strategies control how constraints are propagated through the grid:

```rust
// Example: Using a built-in propagation strategy
let accelerator = GpuAccelerator::new(&grid_def, &rules, BoundaryCondition::Periodic, None)
    .await?
    .with_propagation_strategy(PropagationStrategyType::Subgrid);

// With custom configuration
let config = SubgridConfig::new()
    .with_size(64, 64)
    .with_overlap(4);
let accelerator = accelerator.with_subgrid_propagation(config);
```

#### Built-in Propagation Strategies

| Strategy   | Description                         | Best Use Case                  |
| ---------- | ----------------------------------- | ------------------------------ |
| `Direct`   | Standard propagation algorithm      | Small to medium grids (≤ 512²) |
| `Subgrid`  | Divides grid into manageable chunks | Large grids (> 512²)           |
| `Adaptive` | Selects strategy based on grid size | General purpose                |

## Creating Custom Strategies

### Custom Entropy Strategy

To create a custom entropy strategy, implement the `EntropyStrategy` trait:

```rust
use wfc_gpu::algorithm::{EntropyStrategy, EntropyContext, MinEntropy};

pub struct MyCustomEntropyStrategy {
    // Custom fields
}

impl EntropyStrategy for MyCustomEntropyStrategy {
    fn calculate_entropy(
        &self,
        ctx: &EntropyContext,
    ) -> Result<MinEntropy, GpuError> {
        // Implement your custom entropy calculation
        // ...

        Ok(min_entropy)
    }

    fn buffer_requirements(&self) -> EntropyBufferRequirements {
        // Specify buffer needs
        EntropyBufferRequirements::default()
            .with_custom_buffer("my_parameter", BufferUsage::STORAGE)
    }
}
```

### Custom Propagation Strategy

To create a custom propagation strategy, implement the `PropagationStrategy` trait:

```rust
use wfc_gpu::algorithm::{PropagationStrategy, PropagationContext};

pub struct MyCustomPropagationStrategy {
    // Custom fields
}

impl PropagationStrategy for MyCustomPropagationStrategy {
    fn propagate(
        &self,
        ctx: &PropagationContext,
    ) -> Result<PropagationResult, GpuError> {
        // Implement your custom propagation logic
        // ...

        Ok(result)
    }

    fn buffer_requirements(&self) -> PropagationBufferRequirements {
        // Specify buffer needs
        PropagationBufferRequirements::default()
            .with_worklist_size(1024)
    }
}
```

## Working with Custom Shaders

When implementing custom strategies, you may need to provide custom shader components:

1. Create shader component files in the appropriate directories:

   - Entropy components: `src/shaders/components/entropy/`
   - Propagation components: `src/shaders/components/propagation/`

2. Register your components in the shader registry.

3. Ensure your Rust implementation correctly selects and configures the shader components.

## Performance Considerations

When implementing custom strategies:

- **GPU Buffers**: Minimize data transfers between CPU and GPU
- **Shader Complexity**: Complex shaders may not perform well on all hardware
- **Memory Usage**: Be mindful of GPU memory constraints
- **Workgroup Size**: Tune workgroup sizes for your target hardware

## Debugging Custom Strategies

The WFC-GPU module provides debugging tools for custom strategies:

```rust
// Enable debug visualization for your strategy
let accelerator = accelerator
    .with_debug_visualization(VisualizationType::EntropyHeatmap)
    .with_debug_callback(|snapshot| {
        // Process debug data
        println!("Min entropy: {}", snapshot.min_entropy);
        true // Continue execution
    });
```

## Examples

### Example 1: Custom entropy strategy with weights

```rust
let weights = vec![0.1, 0.2, 0.3, 0.4]; // Custom weights for 4 tile types
let strategy = WeightedEntropyStrategy::new(weights);
let accelerator = GpuAccelerator::new(&grid, &rules, BoundaryCondition::Periodic, None)
    .await?
    .with_custom_entropy_strategy(strategy);
```

### Example 2: Combining strategies for large grids

```rust
// Configure for large grid processing
let entropy_strategy = EntropyHeuristicType::CountBased; // Faster entropy calculation
let subgrid_config = SubgridConfig::new()
    .with_size(128, 128)
    .with_overlap(8);

let accelerator = GpuAccelerator::new(&grid, &rules, BoundaryCondition::Periodic, None)
    .await?
    .with_entropy_heuristic(entropy_strategy)
    .with_subgrid_propagation(subgrid_config);
```
