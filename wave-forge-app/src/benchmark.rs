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

/// Structure to hold benchmark results for a single run (CPU or GPU)
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub implementation: String, // "CPU" or "GPU"
    pub grid_width: usize,
    pub grid_height: usize,
    pub grid_depth: usize,
    pub num_tiles: usize,
    pub total_time: Duration,
    pub wfc_result: Result<(), WfcError>, // Store if WFC succeeded or failed
                                          // TODO: Add more metrics like time per step, memory usage, contradictions etc.
}

/// Runs the WFC algorithm using the specified implementation (CPU or GPU)
/// and collects timing information.
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
            let mut propagator = CpuConstraintPropagator::new();
            let entropy_calculator = CpuEntropyCalculator::new();
            // Use references now (assuming run signature is fixed manually)
            run(
                grid,
                tileset,
                rules,
                &mut propagator,
                &entropy_calculator,
                None,
            )
        }
        "GPU" => {
            // This block is only compiled if 'gpu' feature is enabled
            #[cfg(feature = "gpu")]
            {
                log::info!("Running GPU Benchmark...");
                // Ensure grid possibilities are reset or cloned if necessary before running GPU
                let mut gpu_accelerator = GpuAccelerator::new(grid, rules).await?;
                // Use references now (assuming run signature is fixed manually)
                run(
                    grid,
                    tileset,
                    rules,
                    &mut gpu_accelerator,
                    &gpu_accelerator,
                    None,
                )
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
/// and returns the results.
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

/// Formats and prints the comparison results to the console.
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
