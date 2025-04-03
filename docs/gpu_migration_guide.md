# Migrating from wfc-core CPU to wfc-gpu

This guide explains how to transition your code if you were previously using the (now removed or deprecated) CPU-based components of `wfc-core` to the new GPU-accelerated implementation provided by the `wfc-gpu` crate.

## Background

To significantly improve performance, the core computationally intensive parts of the Wave Function Collapse algorithm (entropy calculation and constraint propagation) have been moved to GPU compute shaders.

- The CPU-based `ConstraintPropagator` implementations (like `CpuConstraintPropagator`, `ParallelConstraintPropagator`) have been **removed** from `wfc-core`.
- The CPU-based `SimpleEntropyCalculator` has been **deprecated** in `wfc-core` and will be removed in a future version.

The recommended approach is now to use the `GpuAccelerator` provided by the `wfc-gpu` crate.

## Steps to Migrate

1.  **Add Dependency:** Add `wfc-gpu` to your `Cargo.toml`:

    ```toml
    [dependencies]
    # ... other dependencies
    wfc-core = { path = "../wfc-core" } # Or version = "..."
    wfc-gpu = { path = "../wfc-gpu" }   # Or version = "..."
    wfc-rules = { path = "../wfc-rules" } # Likely needed
    # Add async runtime like tokio if not already present
    tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
    pollster = "0.3" # Often needed to block on async GPU setup
    ```

2.  **Initialize `GpuAccelerator`:** Replace the creation of `SimpleEntropyCalculator` and the CPU `ConstraintPropagator` with the initialization of `GpuAccelerator`. This requires an `async` context because GPU setup is asynchronous.

    ```rust
    use wfc_core::grid::PossibilityGrid;
    use wfc_gpu::accelerator::GpuAccelerator;
    use wfc_rules::{AdjacencyRules, TileSet};
    use anyhow::Result; // Or your preferred error type

    async fn setup_gpu(initial_grid: &PossibilityGrid, rules: &AdjacencyRules) -> Result<GpuAccelerator> {
        // GpuAccelerator::new handles wgpu instance, adapter, device, queue,
        // pipeline compilation, and buffer creation.
        let accelerator = GpuAccelerator::new(initial_grid, rules).await?;
        Ok(accelerator)
    }

    // Example usage within a main async function or using pollster::block_on
    // let initial_grid = /* ... create grid ... */;
    // let rules = /* ... load rules ... */;
    // let accelerator = pollster::block_on(setup_gpu(&initial_grid, &rules)).expect("Failed to initialize GPU accelerator");
    ```

3.  **Update `WfcRunner` Call:** Pass the components obtained from the `GpuAccelerator` to the `WfcRunner::new` function (or the `wfc-core::runner::run` function).

    ```rust
    use wfc_core::runner::WfcRunner;
    use std::sync::{Arc, atomic::AtomicBool};

    // Assuming 'accelerator' is your initialized GpuAccelerator
    // Assuming 'rules' and 'tileset' are loaded
    // Assuming 'grid' is your mutable PossibilityGrid

    let entropy_calc = Box::new(accelerator.entropy_calculator());
    let propagator = Box::new(accelerator.constraint_propagator());
    let shutdown_signal = Arc::new(AtomicBool::new(false)); // Example shutdown signal

    let mut runner = WfcRunner::new(
        rules.clone(),      // Rules might need to be cloned or Arc'd
        tileset.clone(),    // TileSet might need to be cloned or Arc'd
        entropy_calc,
        propagator,
        None,               // Optional progress callback
        None,               // Optional RNG seed
    );

    // Run the algorithm
    // let result = runner.run(&mut grid);
    // Or using the standalone function:
    // let result = wfc_core::runner::run(
    //     &mut grid,
    //     &tileset,
    //     &rules,
    //     propagator,      // Note: Pass the propagator itself, run consumes it
    //     entropy_calc,    // Note: Pass the calculator itself, run consumes it
    //     None,            // Optional progress callback
    //     shutdown_signal, // Example shutdown signal
    // );
    ```

## System Requirements

- A GPU compatible with one of `wgpu`'s supported backends (Vulkan, Metal, DX12, potentially OpenGL/WebGL via ANGLE).
- Up-to-date graphics drivers.
- Necessary system libraries for Vulkan/DX12/Metal if not already present.

## Performance

Using the `wfc-gpu` backend is expected to provide significant performance improvements over the old CPU implementations, especially for larger grids. Please refer to the benchmark results (TODO: Add link/reference when available) for quantitative comparisons.
