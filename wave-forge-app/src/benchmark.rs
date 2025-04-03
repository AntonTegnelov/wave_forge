//! Benchmarking utilities for WFC GPU performance.
//! Now focuses solely on GPU performance.

use crate::error::AppError;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use wfc_core::{
    entropy::cpu::CpuEntropyCalculator, entropy::SelectionStrategy, grid::PossibilityGrid,
    propagator::cpu::CpuConstraintPropagator, run, runner::WfcConfig, BoundaryMode,
    ConstraintPropagator, EntropyCalculator, ProgressInfo, WfcError,
};
use wfc_rules::{AdjacencyRules, TileId, TileSet, TileSetError, Transformation};

// GPU implementation is now mandatory for benchmarks
use wfc_gpu::accelerator::GpuAccelerator;
use wfc_gpu::{entropy::GpuEntropyCalculator, propagator::GpuConstraintPropagator};

// Use anyhow for application-level errors
use anyhow::{Error, Result};
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::profiler::{print_profiler_summary, ProfileMetric, Profiler};
use csv;
use log; // Import AppError
use pollster; // Add pollster import

/// Represents the aggregated results of a single benchmark run.
#[derive(Debug)]
pub struct BenchmarkResult {
    /// Width of the grid used.
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
    pub iterations: Option<u64>,
    /// Number of cells collapsed before finishing or failing. `None` if run failed very early or callback wasn't invoked.
    pub collapsed_cells: Option<usize>,
    /// Profiling data per code section, if collected
    pub profile_metrics: Option<HashMap<String, ProfileMetric>>,
    /// Memory usage in bytes (if available)
    pub memory_usage: Option<usize>,
}

/// Structure to hold aggregated results for a complete benchmark scenario (multiple runs).
#[derive(Debug)]
pub struct BenchmarkScenarioResult {
    pub rule_file: PathBuf,
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub num_tiles: usize,
    pub runs: usize,
    pub successful_runs: usize,
    pub failed_runs: usize,
    pub avg_total_time_ms: Option<f64>,
    pub median_total_time_ms: Option<f64>,
    pub stddev_total_time_ms: Option<f64>,
}

/// Runs the WFC algorithm using the GPU implementation
/// and collects timing and result information.
///
/// # Arguments
///
/// * `grid` - A mutable reference to the `PossibilityGrid` to run the algorithm on.
/// * `tileset` - A reference to the `TileSet` containing tile information.
/// * `rules` - A reference to the `AdjacencyRules` defining constraints.
///
/// # Returns
///
/// * `Ok(BenchmarkResult)` containing the details of the benchmark run.
/// * `Err(Error)` if GPU initialization or the WFC run fails.
///
pub async fn run_single_benchmark(
    grid: &mut PossibilityGrid,
    tileset: &TileSet,
    rules: &AdjacencyRules,
) -> Result<BenchmarkResult, AppError> {
    // Create a profiler for this benchmark run
    let profiler = Profiler::new("GPU");

    // Start overall timing
    let _overall_guard = profiler.profile("total_execution");
    let start_time = Instant::now();
    let latest_progress = Arc::new(Mutex::new(None::<ProgressInfo>));

    // Prepare the progress callback
    let progress_callback = {
        let progress_clone = Arc::clone(&latest_progress);
        let callback = move |info: ProgressInfo| {
            let mut progress_guard = progress_clone.lock().unwrap();
            *progress_guard = Some(info);
        };
        Some(Box::new(callback) as Box<dyn Fn(ProgressInfo) + Send + Sync>)
    };

    // Get initial memory usage (if supported on platform)
    let initial_memory_res = get_memory_usage().map_err(|e| AppError::Anyhow(e));

    // GPU Path Only
    log::info!("Running GPU Benchmark...");

    // Profile GPU initialization
    let gpu_accelerator = {
        let _guard = profiler.profile("gpu_accelerator_init");
        GpuAccelerator::new(grid, rules, BoundaryMode::Periodic).await?
    };

    // --- Setup based on execution mode ---
    let boundary_mode = BoundaryMode::Periodic; // Or get from args/config
    let tileset_arc = Arc::new(tileset.clone()); // Clone tileset into Arc for CPU calc
    let strategy = SelectionStrategy::FirstMinimum; // Default strategy

    let (propagator, entropy_calc): (
        Box<dyn ConstraintPropagator + Send + Sync>,
        Box<dyn EntropyCalculator + Send + Sync>,
    ) = match ExecutionMode::Gpu {
        ExecutionMode::Cpu => {
            log::info!("Using CPU propagator and entropy calculator.");
            let prop = CpuConstraintPropagator::new(boundary_mode);
            let calc = CpuEntropyCalculator::new(tileset_arc.clone(), strategy); // Pass args
            (Box::new(prop), Box::new(calc))
        }
        ExecutionMode::Gpu => {
            log::info!("Using GPU accelerator for propagation and entropy calculation.");
            // Block on the async GpuAccelerator::new function
            let gpu_accelerator = pollster::block_on(GpuAccelerator::new(
                grid,                   // Pass grid reference
                rules,                  // Pass rules reference
                BoundaryMode::Periodic, // Pass boundary mode
            ))
            .map_err(|e| anyhow::anyhow!("Failed to create GPU accelerator: {}", e))?;

            // Wrap the accelerator in Arc for sharing
            let accelerator_arc = Arc::new(gpu_accelerator);

            // Use the same Arc<GpuAccelerator> for both traits
            (Box::new(accelerator_arc.clone()), Box::new(accelerator_arc))
        }
    };

    // Prepare configuration for the runner
    let config = WfcConfig {
        boundary_mode: BoundaryMode::Periodic, // Set the determined boundary mode
        progress_callback: None,               // No progress for benchmark
        shutdown_signal: Arc::new(AtomicBool::new(false)), // Dummy signal
        // Set other fields to None or defaults as appropriate for benchmark
        initial_checkpoint: None,
        checkpoint_interval: None,
        checkpoint_path: None,
        max_iterations: None,
    };

    log::info!("Running WFC ({:?})...", ExecutionMode::Gpu);

    // Profile the actual WFC run
    let _run_guard = profiler.profile("gpu_wfc_run");

    let wfc_result = run(grid, tileset, rules, propagator, entropy_calc, &config);

    let total_time = start_time.elapsed();

    // Get final memory usage (if supported)
    let final_memory_res = get_memory_usage().map_err(|e| AppError::Anyhow(e));

    // Calculate memory usage only if both results are Ok
    let memory_usage = match (initial_memory_res, final_memory_res) {
        (Ok(initial), Ok(final_memory_val)) => {
            if final_memory_val >= initial {
                Some(final_memory_val - initial)
            } else {
                Some(0) // Handle potential decrease
            }
        }
        _ => None, // One or both memory reads failed
    };

    // Retrieve the last captured progress info
    let final_progress = latest_progress.lock().unwrap().clone();

    // Print profiling results
    print_profiler_summary(&profiler);

    Ok(BenchmarkResult {
        grid_width: grid.width,
        grid_height: grid.height,
        grid_depth: grid.depth,
        num_tiles: rules.num_tiles(),
        total_time,
        wfc_result: wfc_result.map_err(|e| e.into()),
        iterations: final_progress.as_ref().map(|p| p.iterations),
        collapsed_cells: final_progress.as_ref().map(|p| p.collapsed_cells),
        profile_metrics: Some(profiler.get_metrics()),
        memory_usage,
    })
}

/// Attempts to get the current memory usage of the process.
/// (Platform-specific implementation - remains the same)
fn get_memory_usage() -> Result<usize, Error> {
    #[cfg(target_os = "linux")]
    {
        use std::fs::File;
        use std::io::Read;

        let mut status = String::new();
        File::open("/proc/self/status")?.read_to_string(&mut status)?;

        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let Ok(kb) = parts[1].parse::<usize>() {
                        return Ok(kb * 1024); // Convert KB to bytes
                    }
                }
            }
        }
        Err(anyhow::anyhow!("Could not determine memory usage"))
    }

    #[cfg(target_os = "macos")]
    {
        use std::process::Command;

        let output = Command::new("ps")
            .args(&["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()?;

        if output.status.success() {
            let rss = String::from_utf8_lossy(&output.stdout)
                .trim()
                .parse::<usize>()?;
            return Ok(rss * 1024);
        }

        Err(anyhow::anyhow!("Could not determine memory usage"))
    }

    #[cfg(target_os = "windows")]
    {
        log::debug!("Memory usage tracking not fully implemented on Windows");
        Ok(0)
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        Err(anyhow::anyhow!(
            "Memory usage tracking not supported on this platform"
        ))
    }
}

/// Writes the collected aggregated benchmark scenario results to a CSV file.
///
/// # Arguments
///
/// * `scenario_results` - A slice of `BenchmarkScenarioResult` containing the aggregated results.
/// * `path` - The path to the output CSV file.
///
/// # Returns
///
/// * `Ok(())` if writing to CSV is successful.
/// * `Err(Error)` if there is an error creating the file or writing the data.
pub fn write_scenario_results_to_csv(
    scenario_results: &[BenchmarkScenarioResult],
    path: &Path,
) -> Result<(), Error> {
    let file = File::create(path)?;
    let mut wtr = csv::Writer::from_writer(file);

    // Write header row matching console output
    wtr.write_record([
        "Rule File",
        "Width",
        "Height",
        "Depth",
        "Num Tiles",
        "Total Runs",
        "Successful Runs",
        "Failed Runs",
        "Avg Time (ms)",
        "Median Time (ms)",
        "Std Dev Time (ms)",
    ])?;

    for scenario in scenario_results {
        wtr.write_record([
            scenario
                .rule_file
                .file_name()
                .map(|n| n.to_string_lossy())
                .unwrap_or_else(|| scenario.rule_file.to_string_lossy())
                .to_string(),
            scenario.width.to_string(),
            scenario.height.to_string(),
            scenario.depth.to_string(),
            scenario.num_tiles.to_string(),
            scenario.runs.to_string(),
            scenario.successful_runs.to_string(),
            scenario.failed_runs.to_string(),
            scenario
                .avg_total_time_ms
                .map(|t| format!("{:.6}", t))
                .unwrap_or_else(|| "".to_string()),
            scenario
                .median_total_time_ms
                .map(|t| format!("{:.6}", t))
                .unwrap_or_else(|| "".to_string()),
            scenario
                .stddev_total_time_ms
                .map(|t| format!("{:.6}", t))
                .unwrap_or_else(|| "".to_string()),
        ])?;
    }

    wtr.flush()?;
    Ok(())
}

// TODO: Add tests for GPU-only benchmarking functions if possible, although
// this requires a WGPU context which is hard in unit tests.
