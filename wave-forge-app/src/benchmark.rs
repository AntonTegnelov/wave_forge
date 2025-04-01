use std::time::{Duration, Instant};
use wfc_core::{
    // Need the CPU implementations for comparison
    entropy::CpuEntropyCalculator,
    grid::PossibilityGrid,
    propagator::CpuConstraintPropagator,
    rules::AdjacencyRules,
    runner::run, // WfcError is re-exported by wfc_core::WfcError directly
    TileSet,     // Import directly from wfc_core
    WfcError,    // Import directly from wfc_core
};

// Only include GPU-specific code when the 'gpu' feature is enabled
#[cfg(feature = "gpu")]
use wfc_gpu::accelerator::GpuAccelerator;

// Use anyhow for application-level errors
use anyhow::Error;

/// Structure to hold benchmark results for a single run (CPU or GPU).
///
/// Contains timing information, grid parameters, and the final result of the WFC run.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Identifier for the implementation used ("CPU" or "GPU").
    pub implementation: String,
    /// Width of the grid used in the benchmark.
    pub grid_width: usize,
    /// Height of the grid used in the benchmark.
    pub grid_height: usize,
    /// Depth of the grid used in the benchmark.
    pub grid_depth: usize,
    /// Number of unique tiles used in the benchmark.
    pub num_tiles: usize,
    /// Total wall-clock time taken for the WFC algorithm to complete or error out.
    pub total_time: Duration,
    /// The result of the WFC run (`Ok(())` on success, `Err(WfcError)` on failure).
    pub wfc_result: Result<(), WfcError>,
    // TODO: Add more metrics like time per step, memory usage, contradictions etc.
}

/// Runs the WFC algorithm using the specified implementation (CPU or GPU)
/// and collects timing and result information.
///
/// This function executes the core WFC logic using either the CPU or GPU backend
/// based on the `implementation` parameter and records the execution time.
///
/// **Note:** Currently, the GPU execution path is skipped due to an ownership
/// conflict in the underlying `run` function signature when using `GpuAccelerator`.
/// It will return an `InternalError` if "GPU" is specified when the `gpu` feature is enabled.
///
/// # Arguments
///
/// * `implementation` - A string slice indicating the backend to use ("CPU" or "GPU").
/// * `grid` - A mutable reference to the `PossibilityGrid` to run the algorithm on.
/// * `tileset` - A reference to the `TileSet` containing tile information (e.g., weights).
/// * `rules` - A reference to the `AdjacencyRules` defining constraints.
///
/// # Returns
///
/// * `Ok(BenchmarkResult)` containing the details of the benchmark run.
/// * `Err(Error)` if an unknown implementation is specified, if the GPU feature is required
///   but not enabled, or if GPU initialization fails (if implemented).
///
pub async fn run_single_benchmark(
    implementation: &str, // "CPU" or "GPU"
    grid: &mut PossibilityGrid,
    tileset: &TileSet,
    rules: &AdjacencyRules,
    // TODO: Add other parameters like seed if necessary
) -> Result<BenchmarkResult, Error> {
    let start_time = Instant::now();

    let wfc_result = match implementation {
        "CPU" => {
            log::info!("Running CPU Benchmark...");
            let propagator = CpuConstraintPropagator::new();
            let entropy_calculator = CpuEntropyCalculator::new();
            // Run with owned components
            run(grid, tileset, rules, propagator, entropy_calculator, None)
        }
        "GPU" => {
            // This block is only compiled if 'gpu' feature is enabled
            #[cfg(feature = "gpu")]
            {
                log::info!("Running GPU Benchmark...");
                // Ensure grid possibilities are reset or cloned if necessary before running GPU
                #[allow(unused_variables)]
                let gpu_accelerator = GpuAccelerator::new(grid, rules)
                    .await
                    .map_err(|e| anyhow::anyhow!("GPU initialization failed: {}", e))?;
                // TODO: Ownership conflict! GpuAccelerator implements both traits,
                //       but run() takes ownership, and GpuAccelerator is not Clone.
                //       Requires refactoring GpuAccelerator or run() signature.
                // run(grid, tileset, rules, gpu_accelerator, gpu_accelerator, None) // This won't compile
                log::warn!("GPU benchmark run skipped due to ownership conflict.");
                Err(WfcError::InternalError("GPU benchmark skipped".to_string()))
                // Placeholder
            }
            // If 'gpu' feature is not enabled, this case should not be reachable
            // or should return an appropriate error.
            #[cfg(not(feature = "gpu"))]
            {
                log::error!("GPU benchmark requested but GPU feature is not enabled!");
                return Err(anyhow::anyhow!("GPU feature not enabled"));
            }
        }
        _ => {
            return Err(anyhow::anyhow!(
                "Unknown implementation type: {}",
                implementation
            ))
        }
    };

    let total_time = start_time.elapsed();

    Ok(BenchmarkResult {
        implementation: implementation.to_string(),
        grid_width: grid.width,
        grid_height: grid.height,
        grid_depth: grid.depth,
        num_tiles: rules.num_tiles(),
        total_time,
        wfc_result,
    })
}

/// Runs both CPU and GPU benchmarks for the same initial configuration
/// and returns the results for comparison.
///
/// This function clones the initial grid state to ensure both CPU and GPU runs
/// start from the exact same conditions.
///
/// **Note:** This function is only available when the `gpu` feature is enabled.
/// Currently, the GPU execution path within `run_single_benchmark` is skipped,
/// so the GPU result will contain an `InternalError`.
///
/// # Arguments
///
/// * `initial_grid` - A reference to the initial `PossibilityGrid` state before collapse.
/// * `tileset` - A reference to the `TileSet`.
/// * `rules` - A reference to the `AdjacencyRules`.
///
/// # Returns
///
/// * `Ok((BenchmarkResult, BenchmarkResult))` - A tuple containing the CPU result and the GPU result.
/// * `Err(Error)` - If either the CPU or GPU benchmark run encounters an error (e.g., setup fails).
// This function should only be available if the GPU feature is enabled,
// as it explicitly compares CPU and GPU.
#[cfg(feature = "gpu")]
pub async fn compare_implementations(
    initial_grid: &PossibilityGrid, // Need initial state to clone for both runs
    tileset: &TileSet,
    rules: &AdjacencyRules,
) -> Result<(BenchmarkResult, BenchmarkResult), Error> {
    log::info!("Starting CPU vs GPU benchmark comparison...");

    // Clone the grid for each run to ensure identical starting conditions
    let mut cpu_grid = initial_grid.clone();
    let mut gpu_grid = initial_grid.clone();

    let cpu_result = run_single_benchmark("CPU", &mut cpu_grid, tileset, rules).await?;
    log::info!("CPU benchmark completed in {:?}", cpu_result.total_time);

    let gpu_result = run_single_benchmark("GPU", &mut gpu_grid, tileset, rules).await?;
    log::info!("GPU benchmark completed in {:?}", gpu_result.total_time);

    // Optional: Compare results (did both succeed/fail similarly?)
    match (&cpu_result.wfc_result, &gpu_result.wfc_result) {
        (Ok(_), Ok(_)) => log::info!("Both CPU and GPU WFC runs completed successfully."),
        (Err(e_cpu), Err(e_gpu)) => log::warn!(
            "Both CPU and GPU WFC runs failed: CPU({:?}), GPU({:?})",
            e_cpu,
            e_gpu
        ),
        (Ok(_), Err(e_gpu)) => log::warn!("CPU succeeded, but GPU failed: {:?}", e_gpu),
        (Err(e_cpu), Ok(_)) => log::warn!("GPU succeeded, but CPU failed: {:?}", e_cpu),
    }

    Ok((cpu_result, gpu_result))
}

/// Formats and prints the comparison results of CPU and GPU benchmarks to the console.
///
/// Displays grid dimensions, tile count, timing for each implementation, and calculates
/// the speedup factor.
///
/// **Note:** This function is only available when the `gpu` feature is enabled.
///
/// # Arguments
///
/// * `cpu_result` - The `BenchmarkResult` from the CPU run.
/// * `gpu_result` - The `BenchmarkResult` from the GPU run.
#[cfg(feature = "gpu")] // Only relevant if GPU results exist
pub fn report_comparison(cpu_result: &BenchmarkResult, gpu_result: &BenchmarkResult) {
    println!("\n--- Benchmark Comparison ---");
    println!(
        "Grid Dimensions: {}x{}x{}",
        cpu_result.grid_width, cpu_result.grid_height, cpu_result.grid_depth
    );
    println!("Number of Tiles: {}", cpu_result.num_tiles);
    println!("\nImplementation | Total Time | Result");
    println!("----------------|------------|--------");
    println!(
        "CPU             | {:<10?} | {:?}",
        cpu_result.total_time,
        cpu_result.wfc_result.is_ok() // Simple OK/Err indicator
    );
    println!(
        "GPU             | {:<10?} | {:?}",
        gpu_result.total_time,
        gpu_result.wfc_result.is_ok()
    );

    // Calculate speedup
    if gpu_result.total_time > Duration::ZERO {
        let speedup = cpu_result.total_time.as_secs_f64() / gpu_result.total_time.as_secs_f64();
        println!("\nGPU Speedup: {:.2}x", speedup);
    } else {
        println!("\nGPU Speedup: N/A (GPU time was zero)");
    }
    println!("----------------------------\n");
}

/// Formats and prints the results of a single benchmark run to the console.
///
/// Used when only one implementation (typically CPU) was run, or when not comparing.
/// Displays grid dimensions, tile count, implementation name, timing, and success/failure.
///
/// # Arguments
///
/// * `result` - The `BenchmarkResult` from the single run.
// Function to report results when only CPU is run (e.g., no GPU feature)
pub fn report_single_result(result: &BenchmarkResult) {
    println!("\n--- Benchmark Result ---");
    println!(
        "Grid Dimensions: {}x{}x{}",
        result.grid_width, result.grid_height, result.grid_depth
    );
    println!("Number of Tiles: {}", result.num_tiles);
    println!("\nImplementation | Total Time | Result");
    println!("----------------|------------|--------");
    println!(
        "{}         | {:<10?} | {:?}",
        result.implementation,
        result.total_time,
        result.wfc_result.is_ok()
    );
    println!("----------------------------\n");
}

// TODO: Implement CSV output function
// TODO: Implement function to run benchmarks for various grid sizes/complexities

#[cfg(test)]
mod tests {
    use super::*;
    use wfc_core::{grid::PossibilityGrid, rules::AdjacencyRules, tile::TileSet};

    // Helper to create simple rules: Tile 0 adjacent to itself
    fn create_simple_rules_and_tileset() -> (TileSet, AdjacencyRules) {
        let num_tiles = 1;
        let num_axes = 6;
        let weights = vec![1.0];
        let tileset = TileSet::new(weights).unwrap();

        let mut allowed = vec![false; num_axes * num_tiles * num_tiles];
        // Allow Tile 0 <-> Tile 0 on all axes
        for axis in 0..num_axes {
            let index = axis * num_tiles * num_tiles + 0 * num_tiles + 0;
            allowed[index] = true;
        }
        let rules = AdjacencyRules::new(num_tiles, num_axes, allowed);
        (tileset, rules)
    }

    #[tokio::test]
    async fn test_cpu_benchmark_run_basic() {
        let (tileset, rules) = create_simple_rules_and_tileset();
        let mut grid = PossibilityGrid::new(3, 3, 3, tileset.weights.len());

        let result = run_single_benchmark("CPU", &mut grid, &tileset, &rules)
            .await
            .expect("CPU benchmark failed to run");

        assert_eq!(result.implementation, "CPU");
        assert!(result.total_time > Duration::ZERO);
        assert!(result.wfc_result.is_ok()); // Expect success for simple case
    }

    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn test_gpu_benchmark_run_skipped() {
        // This test verifies the current workaround where GPU runs are skipped.
        let (tileset, rules) = create_simple_rules_and_tileset();
        let mut grid = PossibilityGrid::new(3, 3, 3, tileset.weights.len());

        let result = run_single_benchmark("GPU", &mut grid, &tileset, &rules)
            .await
            .expect("GPU benchmark (skipped case) failed to run");

        assert_eq!(result.implementation, "GPU");
        // Time might be very small, but check > 0
        assert!(result.total_time > Duration::ZERO);
        // Expect specific error due to skip
        assert!(matches!(result.wfc_result, Err(WfcError::InternalError(_))));
    }

    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn test_compare_implementations_basic() {
        // This test also relies on the GPU skip workaround
        let (tileset, rules) = create_simple_rules_and_tileset();
        let grid = PossibilityGrid::new(3, 3, 3, tileset.weights.len());

        let result = compare_implementations(&grid, &tileset, &rules).await;

        // Expect the comparison function itself to succeed, even if GPU part returns error
        assert!(result.is_ok());

        if let Ok((cpu_res, gpu_res)) = result {
            assert_eq!(cpu_res.implementation, "CPU");
            assert!(cpu_res.wfc_result.is_ok());
            assert_eq!(gpu_res.implementation, "GPU");
            assert!(matches!(
                gpu_res.wfc_result,
                Err(WfcError::InternalError(_))
            ));
        }
    }
}
