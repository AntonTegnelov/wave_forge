# WFC-GPU Public API Documentation

This document provides a comprehensive guide to the public API of the `wfc-gpu` module, explaining its design philosophy, entry points, configuration options, and usage patterns.

## Design Philosophy

The `wfc-gpu` module provides GPU-accelerated Wave Function Collapse (WFC) algorithm implementations. Its public API is designed with the following principles:

1. **Minimal Surface Area**: We expose only what users need to utilize the library effectively, hiding internal implementation details.
2. **Stability**: Public APIs maintain backward compatibility when possible, with clear deprecation paths when changes are needed.
3. **Configurability**: Users can customize algorithm behavior without needing to understand internal implementations.
4. **Error Robustness**: APIs provide meaningful errors and recovery options for GPU-related failures.

## Primary Entry Points

### `GpuAccelerator`

The main entry point for all GPU-accelerated WFC operations.

```rust
let accelerator = GpuAccelerator::new(
    backend_type,
    adapter_options,
    device_options
)?;

let result = accelerator.run_wfc(
    grid_definition,
    initial_state,
    config
)?;
```

#### Core Configuration Types

- `GridDefinition`: Defines the dimensions and structure of the WFC grid
- `WfcConfig`: Contains algorithm configuration options
- `SubgridConfig`: (Optional) Configuration for large grid optimizations

### Primary Public Types

The following types are part of the public API:

| Type                       | Description                                    | Example Usage                                               |
| -------------------------- | ---------------------------------------------- | ----------------------------------------------------------- |
| `GpuAccelerator`           | Main entry point for GPU-accelerated WFC       | `GpuAccelerator::new(...)`                                  |
| `GridDefinition`           | Defines grid dimensions and characteristics    | `GridDefinition::new(width, height, pattern_size)`          |
| `GridStats`                | Contains statistics about the solution process | `let steps_taken = result.stats.iteration_count;`           |
| `WfcRunResult`             | Result of running the WFC algorithm            | `let result = accelerator.run_wfc(...)?;`                   |
| `GpuError`                 | Error type for GPU-related failures            | `match err { GpuError::OutOfMemory => {...} }`              |
| `SubgridConfig`            | Configuration for large grid optimizations     | `SubgridConfig::with_size(32, 32)`                          |
| `DebugVisualizationConfig` | Configuration for debug visualization          | `DebugVisualizationConfig::new(VisualizationType::Entropy)` |

## Configuration Options

### Entropy Calculation Strategies

The `wfc-gpu` module supports multiple entropy calculation strategies:

```rust
// Using Shannon entropy (default)
let config = WfcConfig::default()
    .with_entropy_heuristic(EntropyHeuristic::Shannon);

// Using count-based entropy for better performance
let config = WfcConfig::default()
    .with_entropy_heuristic(EntropyHeuristic::Count);
```

### Propagation Strategies

Different propagation strategies can be selected based on grid size and performance needs:

```rust
// Using direct propagation (default for small/medium grids)
let config = WfcConfig::default()
    .with_propagation_strategy(PropagationStrategy::Direct);

// Using subgrid propagation for large grids
let config = WfcConfig::default()
    .with_propagation_strategy(PropagationStrategy::Subgrid)
    .with_subgrid_config(SubgridConfig::with_size(32, 32));
```

## Complete Usage Example

Here's a complete example of using the `wfc-gpu` module to solve a WFC problem:

```rust
use wfc_gpu::{
    GpuAccelerator, GridDefinition, WfcConfig, EntropyHeuristic,
    PropagationStrategy, SubgridConfig
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the GPU accelerator
    let accelerator = GpuAccelerator::new_default()?;

    // Define the grid
    let grid_def = GridDefinition::new(
        128,  // width
        128,  // height
        3     // pattern size
    );

    // Create algorithm configuration
    let config = WfcConfig::default()
        .with_entropy_heuristic(EntropyHeuristic::Shannon)
        .with_propagation_strategy(PropagationStrategy::Direct)
        .with_max_iterations(10000);

    // Run the WFC algorithm
    let result = accelerator.run_wfc(grid_def, None, config)?;

    // Access the solution
    println!("Solution found in {} iterations", result.stats.iteration_count);
    let solution = result.solution;

    Ok(())
}
```

## Error Handling

The `wfc-gpu` module uses a specialized error type `GpuError` for all GPU-related errors:

```rust
use wfc_gpu::{GpuAccelerator, GpuError};

fn main() {
    let result = GpuAccelerator::new_default().and_then(|accelerator| {
        // Use the accelerator
        accelerator.run_wfc(...)
    });

    match result {
        Ok(solution) => {
            // Process solution
        },
        Err(GpuError::DeviceLost(e)) => {
            eprintln!("GPU device lost: {}", e);
            // Attempt recovery
        },
        Err(GpuError::OutOfMemory) => {
            eprintln!("GPU out of memory");
            // Try with smaller grid or different config
        },
        Err(e) => {
            eprintln!("Other GPU error: {}", e);
        }
    }
}
```

## Advanced Usage

### Debug Visualization

The module includes optional debug visualization capabilities:

```rust
use wfc_gpu::{
    GpuAccelerator, DebugVisualizationConfig, VisualizationType
};

// Create visualization config
let viz_config = DebugVisualizationConfig::new(VisualizationType::Entropy);

// Enable visualization on the accelerator
let accelerator = GpuAccelerator::new_default()?
    .with_debug_visualization(viz_config);

// Run with visualization
let result = accelerator.run_wfc_with_visualization(
    grid_def,
    None,
    config,
    |snapshot| {
        // Process snapshot
        let entropy_map = snapshot.entropy_map();
        // Render or save entropy map
    }
)?;
```

### Performance Considerations

- For grids larger than 512×512, consider using `PropagationStrategy::Subgrid`
- `EntropyHeuristic::Count` provides better performance at the cost of solution quality
- Set appropriate `max_iterations` to prevent infinite loops on unsolvable inputs

## Stability Guarantees

- Types and functions marked with `#[non_exhaustive]` may have fields or variants added in minor releases
- Public API methods without the `#[doc(hidden)]` attribute are considered stable
- Internal types re-exported from `lib.rs` with `#[doc(hidden)]` are not part of the stable API

## Version Compatibility

This API documentation applies to `wfc-gpu` version 0.3.0 and later. For earlier versions, consult the documentation included with that release.
