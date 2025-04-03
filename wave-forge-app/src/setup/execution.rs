//! Handles the core execution logic for standard and benchmark modes.

use crate::benchmark::{self, BenchmarkResult, BenchmarkScenarioResult};
use crate::config::{AppConfig, ExecutionMode, ProgressLogLevel, RunConfig};
use crate::error::AppError;
use crate::output;
use crate::setup::results::ExecutionResult;
use crate::setup::visualization::VizMessage;
use crate::state::AppState;
use anyhow::{Context, Result};
use log;
use log::{debug, info, warn};
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use wfc_core::grid::PossibilityGrid;
use wfc_core::runner;
use wfc_core::{
    entropy::{cpu::CpuEntropyCalculator, EntropyCalculator, SelectionStrategy},
    propagator::{cpu::CpuConstraintPropagator, ConstraintPropagator},
    runner::{self, ProgressCallback, WfcConfig},
    BoundaryMode, ExecutionMode, ProgressInfo, WfcError,
};
use wfc_gpu::accelerator::GpuAccelerator;
use wfc_gpu::entropy::GpuEntropyCalculator;
use wfc_gpu::propagator::GpuConstraintPropagator;
use wfc_gpu::GpuError;
use wfc_rules::loader::load_from_file;
use wfc_rules::{AdjacencyRules, TileSet};
use wgpu::Instance;

// Helper function to parse "WxHxD" strings
fn parse_dimension_string(dim_str: &str) -> Result<(usize, usize, usize), AppError> {
    let parts: Vec<&str> = dim_str.split('x').collect();
    if parts.len() != 3 {
        return Err(AppError::Config(format!(
            "Invalid dimension string format: '{}'. Expected WxHxD.",
            dim_str
        )));
    }
    let w = parts[0].parse::<usize>().map_err(|_| {
        AppError::Config(format!("Invalid width in dimension string: '{}'", dim_str))
    })?;
    let h = parts[1].parse::<usize>().map_err(|_| {
        AppError::Config(format!("Invalid height in dimension string: '{}'", dim_str))
    })?;
    let d = parts[2].parse::<usize>().map_err(|_| {
        AppError::Config(format!("Invalid depth in dimension string: '{}'", dim_str))
    })?;
    Ok((w, h, d))
}

// --- Statistics Helper Functions ---
fn calculate_median(data: &mut [f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = data.len() / 2;
    if data.len() % 2 == 0 {
        Some((data[mid - 1] + data[mid]) / 2.0)
    } else {
        Some(data[mid])
    }
}

fn calculate_std_dev(data: &[f64], mean: f64) -> Option<f64> {
    let n = data.len();
    if n < 2 {
        // Standard deviation requires at least 2 data points
        return None;
    }
    let variance = data
        .iter()
        .map(|value| {
            let diff = mean - value;
            diff * diff
        })
        .sum::<f64>()
        / (n - 1) as f64; // Use n-1 for sample standard deviation
    Some(variance.sqrt())
}
// --- End Statistics Helper Functions ---

// Structure to hold results for a complete scenario (multiple runs)
// #[derive(Debug)]
// struct BenchmarkScenarioResult { ... } // REMOVED

pub async fn run_benchmark_mode(
    config: &AppConfig,
    _initial_tileset: &TileSet,
    _initial_rules: &AdjacencyRules,
    _grid: &mut PossibilityGrid,
    _viz_tx: &Option<Sender<VizMessage>>,
    _snapshot_handle: &mut Option<thread::JoinHandle<()>>,
    _shutdown_signal: Arc<AtomicBool>,
) -> Result<(), AppError> {
    log::info!(
        "Benchmark mode enabled (GPU only). Runs per scenario: {}",
        config.benchmark_runs_per_scenario
    );
    if config.benchmark_runs_per_scenario == 0 {
        log::warn!("Benchmark runs per scenario is 0, no benchmarks will be executed.");
        return Ok(());
    }

    // 1. Get GPU Adapter Info
    let instance = Instance::default();
    let adapter_info = match instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
    {
        Some(adapter) => adapter.get_info(),
        None => {
            return Err(GpuError::AdapterRequestFailed.into());
        }
    };
    log::info!(
        "Using GPU: {} ({:?})",
        adapter_info.name,
        adapter_info.backend
    );

    // 2. Determine Benchmark Scenarios
    let rule_files_to_run = if config.benchmark_rule_files.is_empty() {
        log::info!(
            "No specific benchmark rule files provided, using default: {:?}",
            config.rule_file
        );
        vec![config.rule_file.clone()]
    } else {
        log::info!(
            "Using benchmark rule files: {:?}",
            config.benchmark_rule_files
        );
        config.benchmark_rule_files.clone()
    };

    let dimensions_to_run: Vec<(usize, usize, usize)> = if config.benchmark_grid_sizes.is_empty() {
        log::info!(
            "No specific benchmark grid sizes provided, using default: {}x{}x{}",
            config.width,
            config.height,
            config.depth
        );
        vec![(config.width, config.height, config.depth)]
    } else {
        log::info!(
            "Using benchmark grid sizes: {:?}",
            config.benchmark_grid_sizes
        );
        config
            .benchmark_grid_sizes
            .iter()
            .map(|s| parse_dimension_string(s))
            .collect::<Result<Vec<_>, _>>()?
    };

    // 3. Run Benchmarks
    let mut all_scenario_results: Vec<BenchmarkScenarioResult> = Vec::new();

    for rule_file_path in &rule_files_to_run {
        log::info!("Loading rules from: {:?}", rule_file_path);
        let (tileset, rules) = match load_from_file(rule_file_path) {
            Ok(data) => data,
            Err(e) => {
                log::error!(
                    "Failed to load rule file {:?}: {}. Skipping scenario.",
                    rule_file_path,
                    e
                );
                continue;
            }
        };
        log::info!(
            "Rules loaded: {} tiles, {} axes",
            tileset.weights.len(),
            rules.num_axes()
        );

        for &(width, height, depth) in &dimensions_to_run {
            log::info!(
                "Starting benchmark scenario: Rule='{:?}', Size={}x{}x{}, Runs={}",
                rule_file_path.file_name().unwrap_or_default(),
                width,
                height,
                depth,
                config.benchmark_runs_per_scenario
            );

            let mut scenario_run_results: Vec<Result<BenchmarkResult, AppError>> = Vec::new();
            let mut successful_times_ms: Vec<f64> = Vec::new();

            for run_index in 0..config.benchmark_runs_per_scenario {
                log::info!(
                    "  Run {}/{} for scenario (Rule='{:?}', Size={}x{}x{})...",
                    run_index + 1,
                    config.benchmark_runs_per_scenario,
                    rule_file_path.file_name().unwrap_or_default(),
                    width,
                    height,
                    depth
                );
                let mut bench_grid =
                    PossibilityGrid::new(width, height, depth, tileset.weights.len());

                let result =
                    benchmark::run_single_benchmark(&mut bench_grid, &tileset, &rules).await;

                match &result {
                    Ok(bench_res) => {
                        log::info!(
                            "  Run {} completed successfully in {:.3} ms",
                            run_index + 1,
                            bench_res.total_time.as_secs_f64() * 1000.0
                        );
                        successful_times_ms.push(bench_res.total_time.as_secs_f64() * 1000.0);
                    }
                    Err(e) => {
                        log::error!("  Run {} failed: {}", run_index + 1, e);
                    }
                }
                scenario_run_results.push(result);
            }

            let successful_runs = successful_times_ms.len();
            let failed_runs = config.benchmark_runs_per_scenario - successful_runs;
            let avg_total_time_ms = if successful_runs > 0 {
                Some(successful_times_ms.iter().sum::<f64>() / successful_runs as f64)
            } else {
                None
            };
            let mut sorted_times = successful_times_ms.clone();
            let median_total_time_ms = calculate_median(&mut sorted_times);
            let stddev_total_time_ms =
                avg_total_time_ms.and_then(|avg| calculate_std_dev(&successful_times_ms, avg));

            all_scenario_results.push(BenchmarkScenarioResult {
                rule_file: rule_file_path.clone(),
                width,
                height,
                depth,
                num_tiles: tileset.weights.len(),
                runs: config.benchmark_runs_per_scenario,
                successful_runs,
                failed_runs,
                avg_total_time_ms,
                median_total_time_ms,
                stddev_total_time_ms,
            });
        }
    }

    // 4. Report Summary
    println!("\n--- GPU Benchmark Suite Summary ---");
    println!("GPU: {} ({:?})", adapter_info.name, adapter_info.backend);
    println!("-------------------------------------------------------------------------------------------------------------------------------");
    println!("Rule File             | Size (WxHxD) | Tiles | Runs | Success | Failed | Avg Time (ms) | Median (ms) | Std Dev (ms) | Notes");
    println!("----------------------|--------------|-------|------|---------|--------|---------------|-------------|--------------|-------");

    for scenario_res in &all_scenario_results {
        let rule_name = scenario_res
            .rule_file
            .file_name()
            .map(|n| n.to_string_lossy())
            .unwrap_or_else(|| scenario_res.rule_file.to_string_lossy());
        let size_str = format!(
            "{}x{}x{}",
            scenario_res.width, scenario_res.height, scenario_res.depth
        );
        let avg_time_str = scenario_res
            .avg_total_time_ms
            .map(|t| format!("{:.3}", t))
            .unwrap_or_else(|| "N/A".to_string());
        let median_time_str = scenario_res
            .median_total_time_ms
            .map(|t| format!("{:.3}", t))
            .unwrap_or_else(|| "N/A".to_string());
        let stddev_time_str = scenario_res
            .stddev_total_time_ms
            .map(|t| format!("{:.3}", t))
            .unwrap_or_else(|| "N/A".to_string());

        println!(
            "{:<21} | {:<12} | {:<5} | {:<4} | {:<7} | {:<6} | {:<13} | {:<11} | {:<12} |",
            rule_name,
            size_str,
            scenario_res.num_tiles,
            scenario_res.runs,
            scenario_res.successful_runs,
            scenario_res.failed_runs,
            avg_time_str,
            median_time_str,
            stddev_time_str,
        );
    }
    println!("-------------------------------------------------------------------------------------------------------------------------------");

    // 5. Write to CSV if requested
    if let Some(csv_path) = &config.benchmark_csv_output {
        log::info!("Writing benchmark suite results to {:?}", csv_path);
        // Use the new function from benchmark module
        match benchmark::write_scenario_results_to_csv(&all_scenario_results, csv_path) {
            Ok(_) => log::info!("Benchmark results successfully written to {:?}", csv_path),
            Err(e) => {
                // Log the error but don't stop the main function from returning Ok
                log::error!("Failed to write benchmark suite results to CSV: {}", e);
            }
        }
    }

    Ok(())
}

pub async fn run_standard_mode(
    config: &AppConfig,
    tileset: &TileSet,
    rules: &AdjacencyRules,
    grid: &mut PossibilityGrid, // Mut ref needed for runner::run
    viz_tx: &Option<Sender<VizMessage>>,
    snapshot_handle: &mut Option<thread::JoinHandle<()>>, // Needs to be mutable to assign
    shutdown_signal: Arc<AtomicBool>,
) -> Result<(), AppError> {
    log::info!("Running WFC (GPU only)...");

    // --- Setup Progress Log File ---
    let progress_log_writer = if let Some(path) = &config.progress_log_file {
        log::info!("Opening progress log file: {:?}", path);
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("Failed to open progress log file: {:?}", path))?;
        Some(Arc::new(Mutex::new(BufWriter::new(file)))) // Wrap in Arc<Mutex> for thread safety
    } else {
        None
    };

    let grid_snapshot = Arc::new(Mutex::new(grid.clone())); // Keep snapshot for Viz

    // Setup visualization snapshot thread
    *snapshot_handle = if let Some(tx) = viz_tx {
        let grid_snapshot_for_viz = Arc::clone(&grid_snapshot);
        let tx_clone = tx.clone();
        let viz_interval = config
            .report_progress_interval
            .map(|d| {
                if d < Duration::from_millis(500) {
                    d
                } else {
                    d / 2
                }
            })
            .unwrap_or_else(|| Duration::from_millis(500));
        log::info!(
            "Starting visualization update thread with interval: {:?}",
            viz_interval
        );
        let handle = thread::spawn(move || {
            let mut last_update = Instant::now();
            loop {
                if last_update.elapsed() >= viz_interval {
                    let guard = grid_snapshot_for_viz
                        .lock()
                        .expect("Visualization mutex poisoned");
                    let grid_clone = guard.clone();
                    drop(guard); // Release lock before sending
                    if tx_clone
                        .send(VizMessage::UpdateGrid(Box::new(grid_clone)))
                        .is_err()
                    {
                        log::error!("Viz channel closed, stopping update thread.");
                        break;
                    }
                    last_update = Instant::now();
                }
                thread::sleep(Duration::from_millis(50));
            }
            log::info!("Visualization update thread terminated");
        });
        Some(handle)
    } else {
        None
    };

    // Setup progress reporting
    let last_report_time = Arc::new(Mutex::new(Instant::now()));
    let report_interval = config.report_progress_interval;
    let progress_log_level = config.progress_log_level.clone(); // Clone here
    let progress_log_writer_clone = progress_log_writer.clone(); // Clone Arc for closure
    let progress_callback: Option<Box<dyn Fn(ProgressInfo) + Send + Sync>> = if let Some(interval) =
        report_interval
    {
        let last_report_time_clone = Arc::clone(&last_report_time);
        Some(Box::new(move |info: ProgressInfo| {
            let now = Instant::now();
            let should_report = {
                let mut last_time = last_report_time_clone
                    .lock()
                    .expect("Progress mutex poisoned");
                if now.duration_since(*last_time) >= interval {
                    *last_time = now;
                    true
                } else {
                    false
                }
            };

            if should_report {
                let elapsed_secs = info.elapsed_time.as_secs_f32();
                let collapse_rate = if elapsed_secs > 0.0 {
                    info.collapsed_cells as f32 / elapsed_secs
                } else {
                    0.0
                };
                let percentage = if info.total_cells > 0 {
                    (info.collapsed_cells as f32 / info.total_cells as f32) * 100.0
                } else {
                    100.0 // Avoid division by zero if grid is empty
                };
                let msg = format!(
                    "Progress: Iter {}, Collapsed {}/{} ({:.1}%), Elapsed: {:.2?}, Rate: {:.1} cells/s",
                    info.iterations, info.collapsed_cells, info.total_cells, percentage, info.elapsed_time, collapse_rate
                );
                // Use the cloned progress_log_level
                match progress_log_level {
                    ProgressLogLevel::Trace => log::trace!("{}", msg),
                    ProgressLogLevel::Debug => log::debug!("{}", msg),
                    ProgressLogLevel::Info => log::info!("{}", msg),
                    ProgressLogLevel::Warn => log::warn!("{}", msg),
                }

                // Write to progress log file if enabled
                if let Some(writer_arc) = &progress_log_writer_clone {
                    if let Ok(mut writer_guard) = writer_arc.lock() {
                        if let Err(e) = writeln!(writer_guard, "{}", msg) {
                            log::error!("Failed to write to progress log file: {}", e);
                        }
                    } else {
                        log::error!("Progress log file mutex poisoned!");
                    }
                }
            }
        }))
    } else {
        None
    };

    // Initialize GPU - explicit match
    let gpu_accelerator = {
        let grid_guard = grid_snapshot.lock().expect("GPU init mutex poisoned"); // Use snapshot for GPU init
        GpuAccelerator::new(&grid_guard, rules).await?
    };

    // Now create the specific propagator and calculator
    let tileset_arc = Arc::new(tileset.clone()); // Clone for CPU calculator
    let strategy = SelectionStrategy::FirstMinimum; // Default strategy

    // Create the appropriate propagator and entropy calculator based on mode
    let (propagator, entropy_calc): (
        Box<dyn ConstraintPropagator + Send + Sync>,
        Box<dyn EntropyCalculator + Send + Sync>,
    ) = match mode {
        ExecutionMode::Cpu => (
            Box::new(CpuConstraintPropagator::new(boundary_mode)),
            Box::new(CpuEntropyCalculator::new(tileset_arc.clone(), strategy)), // Pass args
        ),
        ExecutionMode::Gpu => {
            let gpu_accelerator = pollster::block_on(
                GpuAccelerator::new(&grid_guard, &rules, boundary_mode), // Args: &Grid, &Rules, BoundaryMode
            )?;
            let accelerator_arc = Arc::new(gpu_accelerator);
            (Box::new(accelerator_arc.clone()), Box::new(accelerator_arc))
        }
    };

    // Run WFC
    log::info!("Running WFC on GPU...");
    // Clone the grid state *before* passing it to the runner
    let mut runner_grid = {
        let grid_guard = grid_snapshot
            .lock()
            .expect("Runner grid clone mutex poisoned");
        grid_guard.clone()
    };

    let wfc_result = runner::run(
        &mut runner_grid, // Pass the mutable clone
        tileset,
        rules,
        propagator,
        entropy_calc,
        BoundaryMode::Periodic, // Using Periodic boundary mode as default
        progress_callback.map(|cb| {
            Box::new(move |info: ProgressInfo| -> Result<(), WfcError> {
                cb(info);
                Ok(())
            }) as Box<dyn Fn(ProgressInfo) -> Result<(), WfcError> + Send + Sync>
        }),
        shutdown_signal,
        None, // No initial checkpoint
        None, // No checkpoint interval
        None, // No checkpoint path
        None, // No max iterations
    );

    match wfc_result {
        Ok(_) => {
            log::info!("GPU WFC completed successfully.");
            // Update the original grid/snapshot with the final successful state
            {
                let mut final_grid_guard = grid_snapshot
                    .lock()
                    .expect("Final grid update mutex poisoned");
                *final_grid_guard = runner_grid; // Overwrite shared grid with runner's result
                                                 // Send final update AFTER updating the shared state
                if let Some(tx) = viz_tx {
                    let _ = tx.send(VizMessage::UpdateGrid(Box::new(final_grid_guard.clone())));
                }
            }
            // Save grid (use the updated shared grid for saving)
            {
                let grid_to_save = grid_snapshot.lock().expect("Save grid mutex poisoned");
                output::save_grid_to_file(&grid_to_save, config.output_path.as_path())?
            }
        }
        Err(e) => {
            log::error!("GPU WFC failed: {}", e);
            return Err(e.into());
        }
    };

    Ok(())
}

pub fn run_wfc_interactive(
    app_state: Arc<Mutex<AppState>>,
    run_config: RunConfig,
    progress_callback: Option<Box<dyn Fn(ProgressInfo) -> Result<(), WfcError> + Send + Sync>>,
    shutdown_signal: Arc<AtomicBool>,
) -> Result<ExecutionResult, AppError> {
    info!("Entering run_wfc_interactive...");
    let start_time = Instant::now();

    let state_guard = app_state.lock().expect("AppState lock poisoned");
    let grid_lock = state_guard
        .possibility_grid
        .as_ref()
        .ok_or(AppError::Config("WFC grid not initialized".to_string()))?
        .clone();
    let tileset = state_guard
        .tileset
        .as_ref()
        .ok_or(AppError::Config("WFC tileset not initialized".to_string()))?
        .clone();
    let rules = state_guard
        .rules
        .as_ref()
        .ok_or(AppError::Config("WFC rules not initialized".to_string()))?
        .clone();
    drop(state_guard); // Release AppState lock

    let mut grid_guard = grid_lock.write().expect("Grid write lock failed");
    let boundary_mode = run_config.boundary_mode;
    let tileset_arc = Arc::new(tileset.clone());
    let strategy = run_config.selection_strategy;
    let mode = run_config.mode;

    // Create the appropriate propagator and entropy calculator based on mode
    let (propagator, entropy_calc): (
        Box<dyn ConstraintPropagator + Send + Sync>,
        Box<dyn EntropyCalculator + Send + Sync>,
    ) = match mode {
        ExecutionMode::Cpu => (
            Box::new(CpuConstraintPropagator::new(boundary_mode)),
            Box::new(CpuEntropyCalculator::new(tileset_arc.clone(), strategy)),
        ),
        ExecutionMode::Gpu => {
            let gpu_accelerator = pollster::block_on(
                GpuAccelerator::new(&grid_guard, &rules, boundary_mode), // Ensure boundary_mode is passed
            )?;
            let accelerator_arc = Arc::new(gpu_accelerator);
            (Box::new(accelerator_arc.clone()), Box::new(accelerator_arc))
        }
    };

    // Prepare configuration for the runner
    let config = WfcConfig {
        boundary_mode,
        progress_callback,
        shutdown_signal,
        initial_checkpoint: None, // Assuming no checkpoint load here
        checkpoint_interval: run_config.checkpoint_interval,
        checkpoint_path: run_config.checkpoint_path.clone(),
        max_iterations: run_config.max_iterations,
    };

    // Run WFC core logic
    info!("Starting WFC run...");
    let wfc_result = run(
        &mut grid_guard, // Arg 1: &mut Grid
        &tileset,        // Arg 2: &TileSet
        &rules,          // Arg 3: &Rules
        propagator,      // Arg 4: Box<dyn Propagator>
        entropy_calc,    // Arg 5: Box<dyn Calculator>
        &config,         // Arg 6: &Config
    );
    drop(grid_guard); // Release grid lock after run completes

    let elapsed = start_time.elapsed();
    info!(
        "WFC run finished in {:?}. Result: {:?}",
        elapsed, wfc_result
    );

    match wfc_result {
        Ok(_) => Ok(ExecutionResult::Success {
            duration: elapsed,
            // Add final grid state or other metrics if needed
        }),
        Err(e) => Ok(ExecutionResult::Failure {
            duration: elapsed,
            error: e,
        }),
    }
}

#[must_use]
pub async fn setup_and_run_wfc(
    config: AppConfig,
    tileset: TileSet,
    rules: AdjacencyRules,
) -> Result<PossibilityGrid, AppError> {
    // ... conversions, grid init, gpu init ...

    // --- Select WFC Components (CPU or GPU) ---
    // ... (remains same) ...

    // --- Setup Visualization (if enabled) ---
    // ... (remains same) ...

    // --- Setup Progress Reporting ---
    // ... (remains same) ...

    // --- Prepare WFC Runner Configuration ---
    let shutdown_signal = Arc::new(AtomicBool::new(false));

    let progress_callback: Option<ProgressCallback> =
        progress_reporter.as_ref().map(|reporter_arc| {
            let reporter = reporter_arc.clone();
            // Use ProgressInfo directly here
            let callback: ProgressCallback = Box::new(move |info: ProgressInfo| {
                if let Err(e) = reporter.report(&info) {
                    log::error!("Progress reporting failed: {}", e);
                }
                Ok(())
            });
            callback
        });

    // ... wfc_config setup ...
    // ... Run WFC Algorithm ...

    // --- Handle Result ---
    let final_grid = match wfc_result {
        Ok(()) => {
            // ... (success path) ...
        }
        Err(e) => {
            // Use WfcError directly here
            log::error!("WFC failed: {}", e);
            print_profiler_summary(&profiler);
            return Err(AppError::WfcCore(e));
        }
    };

    // --- Output Saving ---
    let output_guard = profiler.profile("output_saving");
    info!("Saving result to {:?}...", config.output_path);
    // Pass &PathBuf directly to save_grid_to_file
    save_grid_to_file(&final_grid, &config.output_path).context("Failed to save output grid")?;
    drop(output_guard);
    print_profiler_summary(&profiler);

    // --- Visualization Loop ---
    // ... (remains same) ...

    Ok(final_grid)
}
