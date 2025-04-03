//! Benchmarking utilities for WFC GPU performance.
//! Now focuses solely on GPU performance.

use crate::profiler::{print_profiler_summary, ProfileMetric, Profiler};
use crate::progress::ConsoleProgressReporter;
use crate::{
    config::{AppConfig, CliExecutionMode, VisualizationMode},
    error::AppError,
};
use anyhow::{Error, Result as AnyhowResult};
use csv;
use log::{debug, info, warn};
use std::{
    collections::HashMap,
    fs::File,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    }, // Consolidated sync imports
    time::{Duration, Instant},
};
use wfc_core::{
    entropy::cpu::CpuEntropyCalculator,
    entropy::{EntropyCalculator, SelectionStrategy},
    grid::PossibilityGrid,
    propagator::{ConstraintPropagator, CpuConstraintPropagator},
    runner::{self, WfcConfig},
    BoundaryMode, ExecutionMode, ProgressInfo, WfcError,
};
use wfc_gpu::accelerator::GpuAccelerator;
use wfc_rules::{AdjacencyRules, TileSet};

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

/// Runs a single WFC benchmark based on the provided configuration.
/// Handles both CPU and GPU paths.
async fn run_single_wfc_benchmark(
    config: &AppConfig,
    tileset: &TileSet,
    rules: &AdjacencyRules,
    gpu_accelerator_arc: Option<Arc<GpuAccelerator>>,
) -> Result<BenchmarkResult, AppError> {
    let core_execution_mode: ExecutionMode = config.execution_mode.into();
    let core_boundary_mode: BoundaryMode = config.boundary_mode.into();

    let profiler = Profiler::new(&format!("{:?}", core_execution_mode));
    let _overall_guard = profiler.profile("total_benchmark_run");

    let mut grid =
        PossibilityGrid::new(config.width, config.height, config.depth, rules.num_tiles());
    let total_cells = grid.width * grid.height * grid.depth;

    let latest_progress: Arc<Mutex<Option<ProgressInfo>>> = Arc::new(Mutex::new(None));
    let progress_callback = {
        let progress_clone = Arc::clone(&latest_progress);
        let callback: runner::ProgressCallback = Box::new(move |info| {
            let mut progress_guard = progress_clone.lock().unwrap();
            *progress_guard = Some(info);
            Ok(())
        });
        Some(callback)
    };

    let start_time = Instant::now();
    let initial_memory_res = get_memory_usage().map_err(|e| AppError::Anyhow(e));

    let tileset_arc = Arc::new(tileset.clone());
    let (propagator, entropy_calc): (
        Box<dyn ConstraintPropagator + Send + Sync>,
        Box<dyn EntropyCalculator + Send + Sync>,
    ) = match core_execution_mode {
        ExecutionMode::Cpu => {
            let _guard = profiler.profile("cpu_component_setup");
            (
                Box::new(CpuConstraintPropagator::new(core_boundary_mode)),
                Box::new(CpuEntropyCalculator::new(
                    tileset_arc,
                    SelectionStrategy::FirstMinimum,
                )),
            )
        }
        ExecutionMode::Gpu => {
            let _guard = profiler.profile("gpu_component_setup");
            let accelerator_arc = match gpu_accelerator_arc {
                Some(arc) => arc.clone(),
                None => {
                    return Err(AppError::Anyhow(anyhow::anyhow!(
                        "GPU accelerator Arc missing in GPU benchmark mode"
                    )));
                }
            };
            (
                Box::new((*accelerator_arc).clone()),
                Box::new((*accelerator_arc).clone()),
            )
        }
    };

    let wfc_config = WfcConfig {
        boundary_mode: core_boundary_mode,
        progress_callback,
        shutdown_signal: Arc::new(AtomicBool::new(false)),
        initial_checkpoint: None,
        checkpoint_interval: None,
        checkpoint_path: None,
        max_iterations: config.max_iterations,
        seed: config.seed,
    };

    log::info!("Running WFC Benchmark ({:?})...", core_execution_mode);
    let _run_guard = profiler.profile("wfc_run");
    let wfc_result = runner::run(
        &mut grid,
        tileset,
        rules,
        propagator,
        entropy_calc,
        &wfc_config,
    );
    let duration = start_time.elapsed();
    drop(_run_guard);

    let final_memory_res = get_memory_usage().map_err(|e| AppError::Anyhow(e));
    let memory_usage = match (initial_memory_res, final_memory_res) {
        (Ok(initial), Ok(final_val)) => Some(final_val.saturating_sub(initial)),
        _ => None,
    };

    let final_progress = latest_progress.lock().unwrap().clone();
    print_profiler_summary(&profiler);

    Ok(BenchmarkResult {
        grid_width: config.width,
        grid_height: config.height,
        grid_depth: config.depth,
        num_tiles: rules.num_tiles(),
        total_time: duration,
        wfc_result,
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
