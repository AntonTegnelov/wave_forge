//! Handles the core execution logic for standard and benchmark modes.

use crate::{
    benchmark::{self, BenchmarkResult, BenchmarkScenarioResult},
    config::{AppConfig, ProgressLogLevel},
    error::AppError,
    output,
    setup::visualization::VizMessage,
};
use anyhow::Result;
use log::{error, info};
use std::{
    fs::OpenOptions,
    io::{BufWriter, Write},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::Sender,
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};
use wfc_core::{
    entropy::EntropyHeuristicType, grid::PossibilityGrid, BoundaryCondition, ProgressInfo, WfcError,
};
use wfc_gpu::{
    gpu::accelerator::GpuAccelerator,
    utils::{error::gpu_error::GpuError, subgrid::SubgridConfig},
};
use wfc_rules::{loader::load_from_file, AdjacencyRules, TileSet};
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

// Update the WfcProgressCallback definition to match its usage
// Define the two different callback types
/// Type alias for a callback function that reports progress information
type ProgressCallback = Option<Box<dyn Fn(ProgressInfo) -> Result<(), WfcError> + Send + Sync>>;

/// Type alias for a callback function that can access grid state during WFC execution
type WfcProgressCallback =
    Option<Box<dyn Fn(&PossibilityGrid, u64) -> Result<(), WfcError> + Send + Sync>>;

pub async fn run_benchmark_mode(
    config: &AppConfig,
    _viz_tx: &Option<Sender<VizMessage>>,
    _snapshot_handle: &mut Option<thread::JoinHandle<()>>,
    _shutdown_signal: Arc<AtomicBool>,
) -> Result<(), AppError> {
    log::info!(
        "Benchmark mode enabled. Runs per scenario: {}",
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
        Ok(adapter) => adapter.get_info(),
        Err(_) => {
            return Err(GpuError::adapter_request_failed(Default::default()).into());
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
            let mut successful_iterations: Vec<u64> = Vec::new();
            let mut successful_collapsed_cells: Vec<usize> = Vec::new();

            // Prepare GPU Accelerator *once* per scenario if possible
            let scenario_grid = PossibilityGrid::new(width, height, depth, tileset.weights.len());
            let core_boundary_mode: BoundaryCondition = config.boundary_mode.into();

            // Initialize the GPU accelerator without mutex
            let accelerator_res = GpuAccelerator::new(
                &scenario_grid,
                &rules,
                core_boundary_mode,
                EntropyHeuristicType::Count,
                Some(SubgridConfig {
                    max_subgrid_size: 8,
                    overlap_size: 2,
                    min_size: 4,
                }),
            )
            .await;

            // Store the accelerator directly, not Arc
            let accelerator = match accelerator_res {
                Ok(mut acc) => {
                    // Configure propagation with 5000 iterations
                    acc.with_subgrid_propagation(5000, 8);
                    acc
                }
                Err(e) => {
                    log::error!("Failed to initialize GPU accelerator for scenario {:?} ({}x{}x{}): {}. Skipping scenario.",
                        rule_file_path.file_name().unwrap_or_default(), width, height, depth, e);
                    continue; // Skip this scenario if GPU init fails
                }
            };

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

                // Create Arc wrapper for the accelerator clone (still necessary for the function signature)
                let accelerator_for_run = Arc::new(accelerator.clone());

                // Run the benchmark with the provided accelerator
                let result = benchmark::run_single_wfc_benchmark(
                    config,
                    &tileset,
                    &rules,
                    Some(accelerator_for_run),
                )
                .await;

                match &result {
                    Ok(bench_res) => {
                        log::info!(
                            "  Run {} completed successfully in {:.3} ms ({} iterations)",
                            run_index + 1,
                            bench_res.total_time.as_secs_f64() * 1000.0,
                            bench_res.iterations.unwrap_or(0)
                        );
                        successful_times_ms.push(bench_res.total_time.as_secs_f64() * 1000.0);
                        if let Some(iters) = bench_res.iterations {
                            successful_iterations.push(iters);
                        }
                        if let Some(cells) = bench_res.collapsed_cells {
                            successful_collapsed_cells.push(cells);
                        }
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
        match benchmark::write_scenario_results_to_csv(&all_scenario_results, csv_path) {
            Ok(_) => log::info!("Benchmark results successfully written to {:?}", csv_path),
            Err(e) => {
                log::error!("Failed to write benchmark suite results to CSV: {}", e);
            }
        }
    }

    Ok(())
}

pub async fn run_standard_mode(
    config: &AppConfig,
    _tileset: &TileSet,
    rules: &AdjacencyRules,
    grid: &mut PossibilityGrid,
    viz_tx: &Option<Sender<VizMessage>>,
    snapshot_handle: &mut Option<thread::JoinHandle<()>>,
    shutdown_signal: Arc<AtomicBool>,
) -> Result<(), AppError> {
    log::info!("Running WFC standard mode...");

    // --- Setup Progress Log File ---
    let progress_log_writer = if let Some(path) = &config.progress_log_file {
        log::info!("Opening progress log file: {:?}", path);
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| {
                AppError::Anyhow(anyhow::anyhow!(
                    "Failed to open progress log file: {:?}: {}",
                    path,
                    e
                ))
            })?;
        Some(Arc::new(Mutex::new(BufWriter::new(file)))) // Wrap in Arc<Mutex> for thread safety
    } else {
        None
    };

    let grid_snapshot = Arc::new(Mutex::new(grid.clone())); // Snapshot for Viz & GPU init

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
    let progress_log_level = config.progress_log_level.clone();
    let progress_log_writer_clone = progress_log_writer.clone();
    let _progress_callback: ProgressCallback = if let Some(interval) = report_interval {
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
            Ok(())
        }))
    } else {
        None
    };

    // --- Initialize GPU Accelerator ---
    let core_boundary_mode: BoundaryCondition = config.boundary_mode.into();

    // Get grid data from the snapshot, but don't keep the lock during async operation
    let grid_for_accelerator = {
        let grid_guard = grid_snapshot
            .lock()
            .expect("Standard grid lock failed for GPU init");
        grid_guard.clone()
    };

    // Initialize the GPU accelerator using the cloned grid
    let accelerator_res = GpuAccelerator::new(
        &grid_for_accelerator,
        rules,
        core_boundary_mode,
        EntropyHeuristicType::Count,
        Some(SubgridConfig {
            max_subgrid_size: 8,
            overlap_size: 2,
            min_size: 4,
        }),
    )
    .await;

    let mut gpu_accelerator = match accelerator_res {
        Ok(mut acc) => {
            // Configure propagation with 5000 iterations
            acc.with_subgrid_propagation(5000, 8);
            acc
        }
        Err(e) => {
            error!("Failed to initialize GPU accelerator: {}", e);
            return Err(AppError::GpuError(GpuError::other(
                format!("GPU accelerator initialization failed: {}", e),
                Default::default(),
            )));
        }
    };

    // Create a clone of the GPU accelerator for progressive results callback
    let gpu_accelerator_clone = gpu_accelerator.clone();

    // Setup progressive results callback if visualization is enabled
    let _progressive_results_callback: WfcProgressCallback = if viz_tx.is_some() {
        let grid_snapshot_clone = Arc::clone(&grid_snapshot);
        let gpu_clone = gpu_accelerator_clone;

        // Create callback of the right type
        Some(Box::new(
            move |_grid: &PossibilityGrid, iteration: u64| -> Result<(), WfcError> {
                // Fetch the latest state directly from GPU
                if iteration % 10 == 0 {
                    // Only process every 10th iteration to reduce overhead
                    match pollster::block_on(gpu_clone.get_intermediate_result()) {
                        Ok(latest_grid) => {
                            // Update the grid snapshot with the latest state from GPU
                            if let Ok(mut guard) = grid_snapshot_clone.lock() {
                                *guard = latest_grid;
                            }
                        }
                        Err(e) => {
                            log::warn!("Failed to get intermediate result from GPU: {}", e);
                        }
                    }
                }
                Ok(())
            },
        ))
    } else {
        None
    };

    // --- Run WFC using the runner ---
    log::info!("Starting WFC core algorithm on GPU...");

    // Get a clone of the grid from the snapshot
    let runner_grid = {
        let grid_guard = grid_snapshot
            .lock()
            .expect("Runner grid clone mutex poisoned");
        grid_guard.clone()
    };

    // Clone the shutdown signal for use in the closure
    let shutdown_signal_clone = shutdown_signal.clone();

    // Execute using the GPU accelerator
    let wfc_run_result = gpu_accelerator
        .run_with_callback(
            &runner_grid,
            rules,
            config.max_iterations.unwrap_or(u64::MAX),
            move |_info: ProgressInfo| -> Result<bool, anyhow::Error> {
                // Simply continue unless shutdown is requested
                let continue_execution = !shutdown_signal_clone.load(Ordering::SeqCst);
                Ok(continue_execution)
            },
            None, // No tokio shutdown signal
        )
        .await;

    log::info!("WFC core algorithm finished.");

    // --- Process Result ---
    match wfc_run_result {
        Ok(final_grid) => {
            info!("WFC completed successfully.");
            // Update the original grid with the final state from runner_grid
            *grid = final_grid;

            // --- Save Output ---
            if !config.output_path.as_os_str().is_empty() {
                info!("Saving final grid to: {}", config.output_path.display());
                // Use the output module function
                if let Err(e) = output::save_grid_to_file(grid, &config.output_path) {
                    error!("Failed to save output grid: {}", e);
                    // Use the renamed SaveError variant
                    return Err(AppError::SaveError(e));
                } else {
                    info!("Output grid saved successfully.");
                }
            } else {
                info!("Output path not specified, skipping save.");
            }
            Ok(())
        }
        Err(e) => {
            error!("WFC failed: {}", e);
            // Check if the error was due to cancellation
            if shutdown_signal.load(Ordering::SeqCst) {
                Err(AppError::Cancelled)
            } else {
                // Convert wfc_gpu::WfcError to AppError
                Err(AppError::GpuError(GpuError::other(
                    format!("GPU WFC execution failed: {}", e),
                    Default::default(),
                )))
            }
        }
    }
}
