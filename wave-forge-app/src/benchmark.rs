//! Benchmarking utilities for comparing WFC implementations (CPU vs GPU).

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use wfc_core::{
    // Need the CPU implementations for comparison
    entropy::CpuEntropyCalculator,
    grid::PossibilityGrid,
    propagator::CpuConstraintPropagator,
    rules::AdjacencyRules,
    runner::run,
    ProgressInfo, // Import ProgressInfo directly from wfc_core
    TileSet,      // Import directly from wfc_core
    WfcError,     // Import directly from wfc_core
};

// Only include GPU-specific code when the 'gpu' feature is enabled
#[cfg(feature = "gpu")]
use wfc_gpu::accelerator::GpuAccelerator;

// Use anyhow for application-level errors
use anyhow::Error;
use std::fs::File;
use std::path::Path;

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
    /// Number of iterations completed before finishing or failing. `None` if run failed very early or callback wasn't invoked.
    pub iterations: Option<usize>,
    /// Number of cells collapsed before finishing or failing. `None` if run failed very early or callback wasn't invoked.
    pub collapsed_cells: Option<usize>,
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
    let latest_progress = Arc::new(Mutex::new(None::<ProgressInfo>));

    // Prepare the progress callback
    let progress_callback = {
        let progress_clone = Arc::clone(&latest_progress);
        let callback = move |info: ProgressInfo| {
            // TODO: Potentially throttle this write if it becomes a bottleneck
            let mut progress_guard = progress_clone.lock().unwrap(); // Using unwrap as poison error is critical
            *progress_guard = Some(info);
        };
        // Box the closure and cast it to the dynamic trait object
        Some(Box::new(callback) as Box<dyn Fn(ProgressInfo) + Send + Sync>)
    };

    let wfc_result = match implementation {
        "CPU" => {
            log::info!("Running CPU Benchmark...");
            let propagator = CpuConstraintPropagator::new();
            let entropy_calculator = CpuEntropyCalculator::new();
            // Run with owned components and progress callback
            run(
                grid,
                tileset,
                rules,
                propagator,
                entropy_calculator,
                progress_callback,
            )
        }
        "GPU" => {
            // This block is only compiled if 'gpu' feature is enabled
            #[cfg(feature = "gpu")]
            {
                log::info!("Running GPU Benchmark...");
                let gpu_accelerator = GpuAccelerator::new(grid, rules)
                    .await
                    .map_err(|e| anyhow::anyhow!("GPU initialization failed: {}", e))?;

                // Now we can clone the accelerator
                let propagator = gpu_accelerator.clone();
                let entropy_calc = gpu_accelerator; // Use original for the second owned param

                run(
                    grid,
                    tileset,
                    rules,
                    propagator,
                    entropy_calc,
                    progress_callback,
                )
            }
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

    // Retrieve the last captured progress info
    let final_progress = latest_progress.lock().unwrap().clone(); // Clone the Option<ProgressInfo>

    Ok(BenchmarkResult {
        implementation: implementation.to_string(),
        grid_width: grid.width,
        grid_height: grid.height,
        grid_depth: grid.depth,
        num_tiles: rules.num_tiles(),
        total_time,
        wfc_result,
        iterations: final_progress.as_ref().map(|p| p.iteration),
        collapsed_cells: final_progress.as_ref().map(|p| p.collapsed_cells),
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
    println!("\nImplementation | Total Time | Iterations | Collapsed Cells | Result");
    println!("----------------|------------|------------|-----------------|--------");
    println!(
        "CPU             | {:<10?} | {:<10} | {:<15} | {:?}",
        cpu_result.total_time,
        cpu_result
            .iterations
            .map_or_else(|| "N/A".to_string(), |i| i.to_string()),
        cpu_result
            .collapsed_cells
            .map_or_else(|| "N/A".to_string(), |c| c.to_string()),
        cpu_result.wfc_result.is_ok() // Simple OK/Err indicator
    );
    println!(
        "GPU             | {:<10?} | {:<10} | {:<15} | {:?}",
        gpu_result.total_time,
        gpu_result
            .iterations
            .map_or_else(|| "N/A".to_string(), |i| i.to_string()),
        gpu_result
            .collapsed_cells
            .map_or_else(|| "N/A".to_string(), |c| c.to_string()),
        gpu_result.wfc_result.is_ok()
    );

    // Calculate speedup
    if gpu_result.total_time > Duration::ZERO
        && cpu_result.total_time > Duration::ZERO
        && gpu_result.wfc_result.is_ok()
        && cpu_result.wfc_result.is_ok()
    {
        // Only report speedup if both finished successfully and times are non-zero
        let speedup = cpu_result.total_time.as_secs_f64() / gpu_result.total_time.as_secs_f64();
        println!("\nGPU Speedup: {:.2}x", speedup);
    } else if gpu_result.wfc_result.is_err() {
        println!("\nGPU Speedup: N/A (GPU run failed or was skipped)");
    } else {
        println!("\nGPU Speedup: N/A (Timing data invalid or CPU run failed)");
    }
    println!("----------------------------------------------------------------");
    // Adjust separator
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
    println!("\nImplementation | Total Time | Iterations | Collapsed Cells | Result");
    println!("----------------|------------|------------|-----------------|--------");
    println!(
        "{:<15} | {:<10?} | {:<10} | {:<15} | {:?}",
        result.implementation,
        result.total_time,
        result
            .iterations
            .map_or_else(|| "N/A".to_string(), |i| i.to_string()),
        result
            .collapsed_cells
            .map_or_else(|| "N/A".to_string(), |c| c.to_string()),
        result.wfc_result.is_ok()
    );
    println!("----------------------------------------------------------------");
    // Adjust separator
}

/// Writes a collection of benchmark results to a CSV file.
///
/// Creates a CSV file at the specified path and writes the header row followed by
/// data rows for each `BenchmarkResult` provided.
///
/// # Arguments
///
/// * `results` - A slice of tuples, where each tuple contains the grid dimensions
///               and a `Result` containing either the benchmark data (`BenchmarkTuple`)
///               or an error that occurred during the benchmark run.
///               `BenchmarkTuple` varies depending on whether the `gpu` feature is enabled.
/// * `filepath` - The `Path` where the CSV file should be created.
///
/// # Returns
///
/// * `Ok(())` if the CSV file was written successfully.
/// * `Err(anyhow::Error)` if there was an error creating the file or writing to it.
// Conditionally define the type alias based on the feature flag
#[cfg(feature = "gpu")]
type BenchmarkTuple = (BenchmarkResult, BenchmarkResult);
#[cfg(not(feature = "gpu"))]
type BenchmarkTuple = BenchmarkResult;

/// Type alias for a tuple containing grid dimensions and benchmark result
pub type BenchmarkResultTuple = ((usize, usize, usize), Result<BenchmarkTuple, anyhow::Error>);

pub fn write_results_to_csv(
    results: &[BenchmarkResultTuple],
    filepath: &Path,
) -> Result<(), anyhow::Error> {
    log::info!("Writing benchmark results to CSV: {:?}", filepath);
    let file =
        File::create(filepath).map_err(|e| anyhow::anyhow!("Failed to create CSV file: {}", e))?;
    let mut writer = csv::Writer::from_writer(file);

    // Write header row - adjust based on features
    #[cfg(feature = "gpu")]
    writer.write_record([
        "Width",
        "Height",
        "Depth",
        "Num Tiles",
        "Implementation",
        "Total Time (ms)",
        "Iterations",
        "Collapsed Cells",
        "Result",
    ])?;
    #[cfg(not(feature = "gpu"))]
    writer.write_record([
        "Width",
        "Height",
        "Depth",
        "Num Tiles",
        "Implementation",
        "Total Time (ms)",
        "Iterations",
        "Collapsed Cells",
        "Result",
    ])?;

    // Write data rows
    for ((w, h, d), result_item) in results {
        match result_item {
            #[cfg(feature = "gpu")]
            Ok((cpu_result, gpu_result)) => {
                // Write CPU row
                writer.write_record(&[
                    w.to_string(),
                    h.to_string(),
                    d.to_string(),
                    cpu_result.num_tiles.to_string(),
                    cpu_result.implementation.clone(),
                    cpu_result.total_time.as_millis().to_string(),
                    cpu_result
                        .iterations
                        .map_or_else(|| "N/A".to_string(), |i| i.to_string()),
                    cpu_result
                        .collapsed_cells
                        .map_or_else(|| "N/A".to_string(), |c| c.to_string()),
                    if cpu_result.wfc_result.is_ok() {
                        "Ok"
                    } else {
                        "Fail"
                    }
                    .to_string(),
                ])?;
                // Write GPU row
                writer.write_record(&[
                    w.to_string(),
                    h.to_string(),
                    d.to_string(),
                    gpu_result.num_tiles.to_string(),
                    gpu_result.implementation.clone(),
                    gpu_result.total_time.as_millis().to_string(),
                    gpu_result
                        .iterations
                        .map_or_else(|| "N/A".to_string(), |i| i.to_string()),
                    gpu_result
                        .collapsed_cells
                        .map_or_else(|| "N/A".to_string(), |c| c.to_string()),
                    if gpu_result.wfc_result.is_ok() {
                        "Ok"
                    } else {
                        "Fail"
                    }
                    .to_string(),
                ])?;
            }
            #[cfg(not(feature = "gpu"))]
            Ok(cpu_result) => {
                writer.write_record(&[
                    w.to_string(),
                    h.to_string(),
                    d.to_string(),
                    cpu_result.num_tiles.to_string(),
                    cpu_result.implementation.clone(),
                    cpu_result.total_time.as_millis().to_string(),
                    cpu_result
                        .iterations
                        .map_or_else(|| "N/A".to_string(), |i| i.to_string()),
                    cpu_result
                        .collapsed_cells
                        .map_or_else(|| "N/A".to_string(), |c| c.to_string()),
                    if cpu_result.wfc_result.is_ok() {
                        "Ok"
                    } else {
                        "Fail"
                    }
                    .to_string(),
                ])?;
            }
            Err(e) => {
                // Write an error row
                writer.write_record(&[
                    w.to_string(),
                    h.to_string(),
                    d.to_string(),
                    "N/A".to_string(),
                    "Error".to_string(),
                    "N/A".to_string(),
                    "N/A".to_string(),
                    "N/A".to_string(),
                    format!("Benchmark Error: {}", e),
                ])?;
            }
        }
    }

    writer.flush()?;
    log::info!("Benchmark results successfully written to CSV.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    // Correct imports for tests
    use std::fs;
    use tempfile::tempdir;
    use wfc_core::{AdjacencyRules, PossibilityGrid, PropagationError, TileSet, WfcError}; // Added WfcError & PropagationError // For creating temporary directory for CSV test

    // Helper function for creating simple rules/tileset with 2 tiles
    fn create_simple_rules_and_tileset() -> (TileSet, AdjacencyRules) {
        let num_tiles = 2;
        let num_axes = 6; // 3 dimensions * 2 directions per dimension
        let tileset = TileSet::new(vec![1.0, 1.0]).expect("Failed to create simple tileset"); // 2 tiles, equal weight

        // Create the allowed vector
        let mut allowed = vec![false; num_axes * num_tiles * num_tiles];
        let tile0 = 0;
        let tile1 = 1;

        // Helper closure for setting rules
        let mut set_rule = |t1: usize, t2: usize, axis: usize| {
            let index = axis * num_tiles * num_tiles + t1 * num_tiles + t2;
            if index < allowed.len() {
                allowed[index] = true;
            }
        };

        // Define rules:
        for axis in 0..num_axes {
            set_rule(tile0, tile0, axis); // Tile 0 adjacent to Tile 0 (all axes)
            set_rule(tile1, tile1, axis); // Tile 1 adjacent to Tile 1 (all axes)
        }
        // Specific rules: Tile 0 below Tile 1 (+Y = axis 2), Tile 1 above Tile 0 (-Y = axis 3)
        let axis_pos_y = 2;
        let axis_neg_y = 3;
        set_rule(tile0, tile1, axis_pos_y); // Tile 0 -> Tile 1 (+Y)
        set_rule(tile1, tile0, axis_neg_y); // Tile 1 -> Tile 0 (-Y)

        let rules = AdjacencyRules::new(num_tiles, num_axes, allowed);
        (tileset, rules)
    }

    #[tokio::test]
    async fn test_cpu_benchmark_run_basic() {
        let (tileset, rules) = create_simple_rules_and_tileset();
        // Create grid with 2 tiles, so it doesn't start fully collapsed
        let mut grid = PossibilityGrid::new(2, 2, 2, rules.num_tiles()); // Small grid, 2 tiles
        assert_eq!(rules.num_tiles(), 2, "Test setup should use 2 tiles");

        let result = run_single_benchmark("CPU", &mut grid, &tileset, &rules)
            .await
            .expect("CPU benchmark failed unexpectedly");

        assert_eq!(result.implementation, "CPU");
        assert_eq!(result.grid_width, 2);
        assert_eq!(result.grid_height, 2);
        assert_eq!(result.grid_depth, 2);
        assert_eq!(result.num_tiles, 2); // Verify 2 tiles used
        assert!(result.total_time > Duration::ZERO);
        // Check if result is Ok. With these simple rules, it should succeed.
        assert!(
            result.wfc_result.is_ok(),
            "WFC run failed: {:?}",
            result.wfc_result
        );
        // Check if metrics were captured. Now the callback should be called.
        assert!(result.iterations.is_some(), "Iterations should be Some");
        assert!(
            result.iterations.unwrap() >= 1,
            "Should run at least 1 iteration"
        );
        assert!(
            result.collapsed_cells.is_some(),
            "Collapsed cells should be Some"
        );
        assert_eq!(
            result.collapsed_cells.unwrap(),
            2 * 2 * 2,
            "All cells should be collapsed"
        ); // 8 cells in the grid
    }

    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn test_gpu_benchmark_run_basic() {
        let (tileset, rules) = create_simple_rules_and_tileset();
        let mut grid = PossibilityGrid::new(2, 2, 2, rules.num_tiles());
        assert_eq!(rules.num_tiles(), 2, "Test setup should use 2 tiles");

        // Expect the GPU run to succeed or fail normally, not with the specific InternalError
        let result = run_single_benchmark("GPU", &mut grid, &tileset, &rules)
            .await
            .expect("GPU benchmark function failed");

        assert_eq!(result.implementation, "GPU");
        assert_eq!(result.num_tiles, 2);
        // Check if result is Ok or a valid WfcError (Contradiction, etc.)
        // It should NOT be the specific InternalError("GPU benchmark skipped...")
        match &result.wfc_result {
            Ok(_) => (),
            Err(WfcError::InternalError(s)) if s.contains("skipped") => {
                panic!(
                    "GPU benchmark run was unexpectedly skipped: {:?}",
                    result.wfc_result
                );
            }
            Err(_) => (), // Allow other WfcErrors like Contradiction
        }
        // Check if metrics were captured (if run succeeded)
        if result.wfc_result.is_ok() {
            assert!(result.iterations.is_some());
            assert!(result.iterations.unwrap() >= 1);
            assert!(result.collapsed_cells.is_some());
            assert_eq!(result.collapsed_cells.unwrap(), 2 * 2 * 2);
        } else {
            // If failed, metrics might or might not be None
        }
    }

    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn test_compare_implementations_basic() {
        let (tileset, rules) = create_simple_rules_and_tileset();
        let initial_grid = PossibilityGrid::new(2, 2, 2, rules.num_tiles());
        assert_eq!(rules.num_tiles(), 2, "Test setup should use 2 tiles");

        let result = compare_implementations(&initial_grid, &tileset, &rules).await;
        assert!(result.is_ok());

        if let Ok((cpu_result, gpu_result)) = result {
            // CPU result checks
            assert_eq!(cpu_result.implementation, "CPU");
            assert_eq!(cpu_result.num_tiles, 2);
            assert!(cpu_result.total_time > Duration::ZERO);
            assert!(
                cpu_result.wfc_result.is_ok(),
                "CPU WFC run failed: {:?}",
                cpu_result.wfc_result
            );
            assert!(cpu_result.iterations.is_some());
            assert!(cpu_result.iterations.unwrap() >= 1);
            assert!(cpu_result.collapsed_cells.is_some());
            assert_eq!(cpu_result.collapsed_cells.unwrap(), 2 * 2 * 2);

            // GPU result checks (expect normal run)
            assert_eq!(gpu_result.implementation, "GPU");
            assert_eq!(gpu_result.num_tiles, 2);
            match &gpu_result.wfc_result {
                Ok(_) => (),
                Err(WfcError::InternalError(s)) if s.contains("skipped") => {
                    panic!(
                        "GPU benchmark run was unexpectedly skipped: {:?}",
                        gpu_result.wfc_result
                    );
                }
                Err(_) => (), // Allow other WfcErrors
            }
            if gpu_result.wfc_result.is_ok() {
                assert!(gpu_result.iterations.is_some());
                assert!(gpu_result.iterations.unwrap() >= 1);
                assert!(gpu_result.collapsed_cells.is_some());
                assert_eq!(gpu_result.collapsed_cells.unwrap(), 2 * 2 * 2);
            }
        }
    }

    #[test]
    fn test_report_single_result_formatting() {
        // Simple test to ensure report function doesn't panic
        let result_ok = BenchmarkResult {
            implementation: "CPU".to_string(),
            grid_width: 1,
            grid_height: 1,
            grid_depth: 1,
            num_tiles: 1,
            total_time: Duration::from_millis(100),
            wfc_result: Ok(()),
            iterations: Some(5),
            collapsed_cells: Some(1),
        };
        report_single_result(&result_ok); // Just call it

        let result_fail = BenchmarkResult {
            implementation: "CPU".to_string(),
            grid_width: 1,
            grid_height: 1,
            grid_depth: 1,
            num_tiles: 1,
            total_time: Duration::from_millis(50),
            wfc_result: Err(WfcError::PropagationError(PropagationError::Contradiction(
                0, 0, 0,
            ))),
            iterations: Some(2),
            collapsed_cells: Some(0),
        };
        report_single_result(&result_fail);

        let result_none = BenchmarkResult {
            implementation: "CPU".to_string(),
            grid_width: 1,
            grid_height: 1,
            grid_depth: 1,
            num_tiles: 1,
            total_time: Duration::from_millis(1),
            wfc_result: Err(WfcError::InternalError("Setup failed".into())),
            iterations: None,
            collapsed_cells: None,
        };
        report_single_result(&result_none);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_report_comparison_formatting() {
        // Simple test to ensure report function doesn't panic
        let cpu_result = BenchmarkResult {
            implementation: "CPU".to_string(),
            grid_width: 1,
            grid_height: 1,
            grid_depth: 1,
            num_tiles: 1,
            total_time: Duration::from_millis(100),
            wfc_result: Ok(()),
            iterations: Some(10),
            collapsed_cells: Some(1),
        };
        let gpu_result_skipped = BenchmarkResult {
            implementation: "GPU".to_string(),
            grid_width: 1,
            grid_height: 1,
            grid_depth: 1,
            num_tiles: 1,
            total_time: Duration::from_millis(1),
            wfc_result: Err(WfcError::InternalError("Skipped".into())),
            iterations: None,
            collapsed_cells: None,
        };
        report_comparison(&cpu_result, &gpu_result_skipped); // Call with skipped GPU

        // Example with hypothetical successful GPU run (if skipping is fixed later)
        let gpu_result_success = BenchmarkResult {
            implementation: "GPU".to_string(),
            grid_width: 1,
            grid_height: 1,
            grid_depth: 1,
            num_tiles: 1,
            total_time: Duration::from_millis(20),
            wfc_result: Ok(()),
            iterations: Some(10),
            collapsed_cells: Some(1),
        };
        report_comparison(&cpu_result, &gpu_result_success);
    }

    // This test is specifically for the case when only CPU results are expected,
    // i.e., when the 'gpu' feature is NOT enabled.
    #[cfg(not(feature = "gpu"))]
    #[test]
    fn test_write_results_to_csv_cpu_only() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("cpu_results.csv");

        let cpu_result_ok = BenchmarkResult {
            implementation: "CPU".to_string(),
            grid_width: 8,
            grid_height: 8,
            grid_depth: 8,
            num_tiles: 5,
            total_time: Duration::from_millis(1234),
            wfc_result: Ok(()),
            iterations: Some(100),
            collapsed_cells: Some(512),
        };
        let cpu_result_fail = BenchmarkResult {
            implementation: "CPU".to_string(),
            grid_width: 4,
            grid_height: 4,
            grid_depth: 4,
            num_tiles: 3,
            total_time: Duration::from_millis(567),
            wfc_result: Err(WfcError::Contradiction(0, 0, 0)),
            iterations: Some(50),
            collapsed_cells: Some(60),
        };

        let results: Vec<BenchmarkResultTuple> = vec![
            ((8, 8, 8), Ok(cpu_result_ok)),
            ((4, 4, 4), Ok(cpu_result_fail)),
            ((2, 2, 2), Err(anyhow::anyhow!("Setup failed"))),
        ];

        let write_result = write_results_to_csv(&results, &file_path);
        assert!(write_result.is_ok());

        // Verify file content
        let content = fs::read_to_string(&file_path).expect("Failed to read test CSV");
        let expected_header = "Width,Height,Depth,Num Tiles,Implementation,Total Time (ms),Iterations,Collapsed Cells,Result";
        let expected_row1 = "8,8,8,5,CPU,1234,100,512,Ok";
        let expected_row2 = "4,4,4,3,CPU,567,50,60,Fail";
        let expected_row3 = "2,2,2,N/A,Error,N/A,N/A,N/A,Benchmark Error: Setup failed";

        assert!(content.contains(expected_header));
        assert!(content.contains(expected_row1));
        assert!(content.contains(expected_row2));
        assert!(content.contains(expected_row3));
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_write_results_to_csv_with_gpu() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("gpu_results.csv");

        let cpu_result = BenchmarkResult {
            implementation: "CPU".to_string(),
            grid_width: 8,
            grid_height: 8,
            grid_depth: 8,
            num_tiles: 5,
            total_time: Duration::from_millis(2000),
            wfc_result: Ok(()),
            iterations: Some(100),
            collapsed_cells: Some(512),
        };
        // Simulate a successful (hypothetical) GPU run
        let gpu_result = BenchmarkResult {
            implementation: "GPU".to_string(),
            grid_width: 8,
            grid_height: 8,
            grid_depth: 8,
            num_tiles: 5,
            total_time: Duration::from_millis(500),
            wfc_result: Ok(()),
            iterations: Some(100),
            collapsed_cells: Some(512),
        };
        // Simulate a failed GPU run
        let gpu_result_fail = BenchmarkResult {
            implementation: "GPU".to_string(),
            grid_width: 4,
            grid_height: 4,
            grid_depth: 4,
            num_tiles: 3,
            total_time: Duration::from_millis(100),
            wfc_result: Err(WfcError::InternalError("Skipped".into())),
            iterations: None,
            collapsed_cells: None,
        };
        let cpu_result_for_fail = BenchmarkResult {
            implementation: "CPU".to_string(),
            grid_width: 4,
            grid_height: 4,
            grid_depth: 4,
            num_tiles: 3,
            total_time: Duration::from_millis(800),
            wfc_result: Ok(()),
            iterations: Some(60),
            collapsed_cells: Some(64),
        };

        let results: Vec<BenchmarkResultTuple> = vec![
            ((8, 8, 8), Ok((cpu_result.clone(), gpu_result.clone()))),
            (
                (4, 4, 4),
                Ok((cpu_result_for_fail.clone(), gpu_result_fail.clone())),
            ),
            ((2, 2, 2), Err(anyhow::anyhow!("Setup failed"))),
        ];

        let write_result = write_results_to_csv(&results, &file_path);
        assert!(write_result.is_ok());

        // Verify file content
        let content = fs::read_to_string(&file_path).expect("Failed to read test CSV");
        println!("CSV Content:\n{}", content); // Print for debugging if needed

        let expected_header = "Width,Height,Depth,Num Tiles,Implementation,Total Time (ms),Iterations,Collapsed Cells,Result";
        let expected_cpu_row1 = "8,8,8,5,CPU,2000,100,512,Ok";
        let expected_gpu_row1 = "8,8,8,5,GPU,500,100,512,Ok";
        let expected_cpu_row2 = "4,4,4,3,CPU,800,60,64,Ok";
        let expected_gpu_row2 = "4,4,4,3,GPU,100,N/A,N/A,Fail";
        let expected_err_row = "2,2,2,N/A,Error,N/A,N/A,N/A,Benchmark Error: Setup failed";

        assert!(content.contains(expected_header));
        assert!(content.contains(expected_cpu_row1));
        assert!(content.contains(expected_gpu_row1));
        assert!(content.contains(expected_cpu_row2));
        assert!(content.contains(expected_gpu_row2));
        assert!(content.contains(expected_err_row));
    }
}
