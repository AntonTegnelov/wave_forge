//! Benchmarking utilities for WFC GPU performance.
//! Now focuses solely on GPU performance.

use crate::error::AppError;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use wfc_core::{
    // Core components needed
    grid::PossibilityGrid,
    runner::run,
    ProgressInfo,
    WfcError,
};
use wfc_rules::{AdjacencyRules, TileSet}; // Use wfc-rules types

// GPU implementation is now mandatory for benchmarks
use wfc_gpu::accelerator::GpuAccelerator;
use wfc_gpu::{entropy::GpuEntropyCalculator, propagator::GpuConstraintPropagator}; // Corrected import paths

// Use anyhow for application-level errors
use anyhow::{Error, Result};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::atomic::AtomicBool;

use crate::profiler::{print_profiler_summary, ProfileMetric, Profiler};
use log; // Import AppError

/// Structure to hold benchmark results for a single GPU run.
///
/// Contains timing information, grid parameters, and the final result of the WFC run.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
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
    /// Profiling data per code section, if collected
    pub profile_metrics: Option<HashMap<String, ProfileMetric>>,
    /// Memory usage in bytes (if available)
    pub memory_usage: Option<usize>,
}

/// Type alias for storing benchmark results along with grid dimensions.
pub type BenchmarkResultTuple = ((usize, usize, usize), Result<BenchmarkResult, Error>);

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
        GpuAccelerator::new(grid, rules).await?
    };

    // Create the specific propagator and calculator instances using resources from the accelerator
    let propagator = GpuConstraintPropagator::new(
        gpu_accelerator.device(),
        gpu_accelerator.queue(),
        gpu_accelerator.pipelines(),
        gpu_accelerator.buffers(),
        gpu_accelerator.grid_dims(),
    );
    let entropy_calc = GpuEntropyCalculator::new(
        gpu_accelerator.device(),
        gpu_accelerator.queue(),
        (*gpu_accelerator.pipelines()).clone(),
        (*gpu_accelerator.buffers()).clone(),
        gpu_accelerator.grid_dims(),
    );

    let shutdown_signal = Arc::new(AtomicBool::new(false)); // Create default signal

    // Profile the actual WFC run
    let _run_guard = profiler.profile("gpu_wfc_run");

    let wfc_result = run(
        grid,
        tileset,
        rules,
        propagator,
        entropy_calc,
        progress_callback,
        shutdown_signal.clone(), // Pass the signal
    );

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
        iterations: final_progress.as_ref().map(|p| p.iteration),
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

/// Writes the collected benchmark results to a CSV file.
/// (Adapted for GPU-only results)
///
/// # Arguments
///
/// * `results` - A slice of `BenchmarkResultTuple` containing the results to write.
/// * `path` - The path to the output CSV file.
///
/// # Returns
///
/// * `Ok(())` if writing to CSV is successful.
/// * `Err(Error)` if there is an error creating the file or writing the data.
pub fn write_results_to_csv(results: &[BenchmarkResultTuple], path: &Path) -> Result<(), Error> {
    let file = File::create(path)?;
    let mut wtr = csv::Writer::from_writer(file);

    // Write header row
    wtr.write_record([
        "Width",
        "Height",
        "Depth",
        "Num Tiles",
        "Total Time (ms)",
        "Iterations",
        "Collapsed Cells",
        "Result",
        "Memory Usage (bytes)",
    ])?;

    // Write data rows
    for ((w, h, d), result_item) in results {
        match result_item {
            Ok(result) => {
                wtr.write_record([
                    w.to_string(),
                    h.to_string(),
                    d.to_string(),
                    result.num_tiles.to_string(),
                    result.total_time.as_millis().to_string(),
                    result.iterations.map_or("".to_string(), |i| i.to_string()),
                    result
                        .collapsed_cells
                        .map_or("".to_string(), |c| c.to_string()),
                    if result.wfc_result.is_ok() {
                        "Ok".to_string()
                    } else {
                        format!("Fail({:?})", result.wfc_result.clone().err().unwrap())
                    },
                    result
                        .memory_usage
                        .map_or("".to_string(), |m| m.to_string()),
                ])?;
            }
            Err(e) => {
                // Write a row indicating the error for this size
                wtr.write_record([
                    w.to_string(),
                    h.to_string(),
                    d.to_string(),
                    "N/A".to_string(), // Num tiles unknown if setup failed
                    "N/A".to_string(),
                    "N/A".to_string(),
                    "N/A".to_string(),
                    format!("Error({})", e),
                    "N/A".to_string(),
                ])?;
            }
        }
    }

    wtr.flush()?;
    log::info!("Benchmark results written to: {:?}", path);
    Ok(())
}

// TODO: Add tests for GPU-only benchmarking functions if possible, although
// this requires a WGPU context which is hard in unit tests.
