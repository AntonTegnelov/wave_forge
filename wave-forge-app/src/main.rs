//! # Wave Forge Application
//!
//! This is the main executable crate for the Wave Forge project.
//! It provides a command-line interface to:
//! - Run the Wave Function Collapse (WFC) algorithm on specified rule files and grid dimensions.
//! - Optionally use GPU acceleration (if the `gpu` feature is enabled).
//! - Benchmark CPU vs GPU performance (if the `gpu` feature is enabled).
//! - Configure output paths, visualization modes, and progress reporting.

// wave-forge-app/src/main.rs

pub mod benchmark;
pub mod config;
pub mod output;
pub mod visualization;

use anyhow::Result;
use clap::Parser;
use config::AppConfig;
use config::VisualizationMode;
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use visualization::{TerminalVisualizer, Visualizer};
use wfc_core::grid::PossibilityGrid;
use wfc_rules::loader::load_from_file;

// Helper enum for visualization control messages
enum VizMessage {
    UpdateGrid(Box<PossibilityGrid>),
    Finished,
}

/// The main entry point for the Wave Forge application.
///
/// Parses command-line arguments using `clap`, initializes logging, loads WFC rules,
/// sets up the grid, and then either runs the benchmarking suite or the standard
/// WFC algorithm based on the provided configuration.
///
/// Handles both CPU and GPU execution paths (conditional on the `gpu` feature).
/// Orchestrates progress reporting and final output saving.
///
/// Uses `tokio` for the async runtime, primarily for the asynchronous GPU initialization.
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging (using env_logger)
    env_logger::init();

    // Parse command-line arguments
    let config = AppConfig::parse();

    log::info!("Wave Forge App Starting");
    log::debug!("Loaded Config: {:?}", config);

    // --- Initialize Visualizer in a separate thread if configured ---
    let viz_tx = match config.visualization_mode {
        VisualizationMode::None => None,
        _ => {
            // Create a channel to send grid snapshots for visualization
            let (tx, rx) = mpsc::channel();

            // Start visualization in a separate thread
            let viz_mode = config.visualization_mode.clone();
            let grid_width = config.width;
            let grid_height = config.height;
            let toggle_key = config.visualization_toggle_key;

            log::info!("Starting visualization thread with mode: {:?}", viz_mode);
            log::info!("Visualization toggle key: '{}'", toggle_key);

            thread::spawn(move || {
                let mut visualizer: Box<dyn Visualizer> = match viz_mode {
                    VisualizationMode::Terminal => {
                        Box::new(TerminalVisualizer::with_toggle_key(toggle_key))
                    }
                    VisualizationMode::Simple2D => {
                        match visualization::Simple2DVisualizer::new(
                            &format!("Wave Forge - {}x{}", grid_width, grid_height),
                            grid_width,
                            grid_height,
                            toggle_key,
                        ) {
                            Ok(viz) => Box::new(viz),
                            Err(e) => {
                                log::error!("Failed to create Simple2DVisualizer: {}", e);
                                Box::new(TerminalVisualizer::with_toggle_key(toggle_key))
                                // Fallback to terminal
                            }
                        }
                    }
                    VisualizationMode::None => unreachable!(),
                };

                log::info!("Visualization thread started");

                // Process incoming grid snapshots
                let mut running = true;
                while running {
                    match rx.recv() {
                        Ok(VizMessage::UpdateGrid(grid)) => {
                            // Process input to handle toggle requests
                            if let Ok(continue_viz) = visualizer.process_input() {
                                if !continue_viz {
                                    log::info!("Visualization stopped by user input");
                                    running = false;
                                    continue;
                                }

                                // Only display if visualization is enabled
                                if visualizer.is_enabled() {
                                    if let Err(e) = visualizer.display_state(&grid) {
                                        log::error!("Failed to display grid: {}", e);
                                    }
                                }
                            }
                        }
                        Ok(VizMessage::Finished) => {
                            log::info!("Visualization finished signal received");
                            running = false;
                        }
                        Err(e) => {
                            log::error!("Error receiving visualization update: {}", e);
                            running = false;
                        }
                    }

                    // Small sleep to prevent busy-waiting
                    thread::sleep(Duration::from_millis(10));
                }

                log::info!("Visualization thread terminated");
            });

            Some(tx)
        }
    };
    // --- End Visualizer Initialization ---

    println!("Wave Forge App");
    println!("Config: {:?}", config);

    // Load rules and tileset
    log::info!("Loading rules from: {:?}", config.rule_file);
    let (tileset, rules) = load_from_file(&config.rule_file).map_err(|e| anyhow::anyhow!(e))?;
    log::info!(
        "Rules loaded: {} tiles, {} axes",
        tileset.weights.len(),
        rules.num_axes()
    );

    // Initialize grid
    log::info!(
        "Initializing grid: {}x{}x{}",
        config.width,
        config.height,
        config.depth
    );
    let mut grid = PossibilityGrid::new(
        config.width,
        config.height,
        config.depth,
        tileset.weights.len(),
    );

    // Initial visualization of the empty grid
    if let Some(tx) = &viz_tx {
        log::info!("Sending initial grid state to visualization thread");
        if let Err(e) = tx.send(VizMessage::UpdateGrid(Box::new(grid.clone()))) {
            log::error!("Failed to send initial grid state: {}", e);
        }
    }

    if config.benchmark_mode {
        log::info!("Benchmark mode enabled.");

        // Define benchmark scenarios (dimensions)
        let benchmark_dimensions = [
            (8, 8, 8), // Small
            (16, 16, 16), // Medium
                       // Add more sizes as needed, e.g.:
                       // (32, 16, 8), // Larger, non-cubic
                       // (32, 32, 32), // Large
        ];
        log::info!("Running benchmarks for sizes: {:?}", benchmark_dimensions);

        // Store results for final report
        // Adjusted to handle the conditional compilation of GPU results
        #[cfg(feature = "gpu")]
        type BenchmarkTuple = (benchmark::BenchmarkResult, benchmark::BenchmarkResult);
        #[cfg(not(feature = "gpu"))]
        type BenchmarkTuple = benchmark::BenchmarkResult;
        let mut all_results: Vec<benchmark::BenchmarkResultTuple> = Vec::new();

        #[cfg(feature = "gpu")]
        {
            for &(width, height, depth) in &benchmark_dimensions {
                log::info!(
                    "Starting benchmark for size: {}x{}x{}",
                    width,
                    height,
                    depth
                );

                // Create a new initial grid for this size
                let initial_grid_for_bench =
                    PossibilityGrid::new(width, height, depth, tileset.weights.len());

                // Run comparison benchmark for this size
                let result =
                    benchmark::compare_implementations(&initial_grid_for_bench, &tileset, &rules)
                        .await;

                // Store the result (or error) for this size
                all_results.push(((width, height, depth), result.map_err(anyhow::Error::from)));
            }
        }
        // If only CPU feature is enabled, run only CPU benchmarks
        #[cfg(not(feature = "gpu"))]
        {
            log::warn!("GPU feature not enabled, running CPU benchmarks only.");
            for &(width, height, depth) in &benchmark_dimensions {
                log::info!(
                    "Starting CPU benchmark for size: {}x{}x{}",
                    width,
                    height,
                    depth
                );
                let mut cpu_grid =
                    PossibilityGrid::new(width, height, depth, tileset.weights.len());
                let result =
                    benchmark::run_single_benchmark("CPU", &mut cpu_grid, &tileset, &rules).await;
                all_results.push(((width, height, depth), result));
            }
        }

        // --- Report Summary --- (Moved reporting after all runs)
        println!("\n--- Benchmark Suite Summary ---");
        println!(
            "Rule File: {:?}",
            config.rule_file.file_name().unwrap_or_default()
        );
        println!("Num Tiles: {}", tileset.weights.len());
        println!("-------------------------------------------------------------------------------------------");
        // Adjust header based on features
        #[cfg(feature = "gpu")]
        println!("Size (WxHxD)    | Impl | Total Time | Iterations | Collapsed Cells | Result   | Speedup (vs CPU)");
        #[cfg(not(feature = "gpu"))]
        println!("Size (WxHxD)    | Impl | Total Time | Iterations | Collapsed Cells | Result");
        println!("----------------|------|------------|------------|-----------------|----------|-----------------"); // Keep separator wide enough for GPU case

        for ((w, h, d), result_item) in &all_results {
            let size_str = format!("{}x{}x{}", w, h, d);
            match result_item {
                #[cfg(feature = "gpu")]
                Ok((cpu_result, gpu_result)) => {
                    // Print CPU result
                    println!(
                        "{:<15} | CPU  | {:<10?} | {:<10} | {:<15} | {:<8} | {:<15}",
                        size_str,
                        cpu_result.total_time,
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
                        },
                        "-" // Placeholder for speedup column
                    );
                    // Calculate speedup directly here
                    let speedup_str = if gpu_result.total_time > Duration::ZERO
                        && cpu_result.total_time > Duration::ZERO
                        && gpu_result.wfc_result.is_ok()
                        && cpu_result.wfc_result.is_ok()
                    {
                        format!(
                            "{:.2}x",
                            cpu_result.total_time.as_secs_f64()
                                / gpu_result.total_time.as_secs_f64()
                        )
                    } else if gpu_result.wfc_result.is_err() {
                        "N/A (GPU Fail)".to_string()
                    } else if cpu_result.wfc_result.is_err() {
                        "N/A (CPU Fail)".to_string()
                    } else {
                        "N/A".to_string()
                    };

                    // Print GPU result
                    println!(
                        "{:<15} | GPU  | {:<10?} | {:<10} | {:<15} | {:<8} | {:<15}",
                        "", // Don't repeat size
                        gpu_result.total_time,
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
                        },
                        speedup_str
                    );
                }
                #[cfg(not(feature = "gpu"))]
                Ok(cpu_result) => {
                    // Print CPU result only
                    println!(
                        "{:<15} | CPU  | {:<10?} | {:<10} | {:<15} | {:<8}",
                        size_str,
                        cpu_result.total_time,
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
                        },
                    );
                }
                Err(e) => {
                    println!("{:<15} | Both | Error running benchmark: {} |", size_str, e);
                }
            }
            println!("-------------------------------------------------------------------------------------------");
            // Keep separator wide enough for GPU case
        }

        // --- Write to CSV if requested ---
        if let Some(csv_path) = &config.benchmark_csv_output {
            if let Err(e) = benchmark::write_results_to_csv(&all_results, csv_path) {
                // Log the error but don't necessarily stop the whole app
                log::error!("Failed to write benchmark results to CSV: {}", e);
            }
        }
    } else {
        log::info!("Running standard WFC...");

        let use_gpu = !config.cpu_only && cfg!(feature = "gpu");

        // Create a thread-safe grid reference for the visualization thread
        let grid_snapshot = Arc::new(Mutex::new(grid.clone()));
        let grid_snapshot_for_viz = Arc::clone(&grid_snapshot);

        // Create a visualization update thread if needed
        let _viz_handle = if let Some(tx) = &viz_tx {
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
                .unwrap_or_else(|| Duration::from_millis(500)); // Default to 500ms

            log::info!(
                "Starting visualization update thread with interval: {:?}",
                viz_interval
            );

            let handle = thread::spawn(move || {
                let mut last_update = Instant::now();

                loop {
                    // Check if we should update
                    if last_update.elapsed() >= viz_interval {
                        // Get a snapshot of the current grid
                        let grid_clone = {
                            let grid_guard = grid_snapshot_for_viz.lock().unwrap();
                            grid_guard.clone()
                        };

                        // Send it to the visualization thread
                        match tx_clone.send(VizMessage::UpdateGrid(Box::new(grid_clone))) {
                            Ok(_) => {
                                last_update = Instant::now();
                            }
                            Err(e) => {
                                log::error!(
                                    "Failed to send grid update to visualization thread: {}",
                                    e
                                );
                                break; // Exit if the receiver is gone
                            }
                        }
                    }

                    // Sleep briefly to avoid busy waiting
                    thread::sleep(Duration::from_millis(50));
                }

                log::info!("Visualization update thread terminated");
            });

            Some(handle)
        } else {
            None
        };

        // --- Progress Reporting Setup ---
        let last_report_time = Arc::new(Mutex::new(Instant::now()));
        let report_interval = config.report_progress_interval;
        let progress_log_level = config.progress_log_level.clone();

        log::info!("Progress reporting interval: {:?}", report_interval);
        log::info!("Progress log level: {:?}", progress_log_level);

        let progress_callback: Option<Box<dyn Fn(wfc_core::ProgressInfo) + Send + Sync>> =
            if let Some(interval) = report_interval {
                let last_report_time_clone = Arc::clone(&last_report_time);

                Some(Box::new(move |info: wfc_core::ProgressInfo| {
                    let now = Instant::now();

                    // We can't access 'grid' directly in the closure
                    // Since we can't update the grid snapshot from ProgressInfo (doesn't have grid)
                    // This feature will be handled by the separate visualization thread instead

                    // Progress reporting logic
                    let mut last_time = last_report_time_clone.lock().unwrap();
                    if now.duration_since(*last_time) >= interval {
                        let percentage = if info.total_cells > 0 {
                            (info.collapsed_cells as f32 / info.total_cells as f32) * 100.0
                        } else {
                            100.0 // Grid is empty, consider it 100% done?
                        };

                        // Log with the configured level
                        let progress_msg = format!(
                            "Progress: Iteration {}, Collapsed {}/{} ({:.1}%)                    ",
                            info.iteration, info.collapsed_cells, info.total_cells, percentage
                        );

                        match progress_log_level {
                            config::ProgressLogLevel::Trace => log::trace!("{}", progress_msg),
                            config::ProgressLogLevel::Debug => log::debug!("{}", progress_msg),
                            config::ProgressLogLevel::Info => log::info!("{}", progress_msg),
                            config::ProgressLogLevel::Warn => log::warn!("{}", progress_msg),
                        }

                        *last_time = now; // Reset timer
                    }
                }))
            } else {
                None
            };
        // --- End Progress Reporting Setup ---

        if use_gpu {
            #[cfg(feature = "gpu")]
            {
                log::info!("Initializing GPU Accelerator...");
                match wfc_gpu::accelerator::GpuAccelerator::new(&grid, &rules).await {
                    Ok(gpu_accelerator) => {
                        log::info!("Running WFC on GPU...");
                        // Clone the accelerator for the two trait parameters
                        let propagator = gpu_accelerator.clone();
                        let entropy_calc = gpu_accelerator; // Use original
                        match wfc_core::runner::run(
                            &mut grid,
                            &tileset,
                            &rules,
                            propagator,   // Passes ownership of clone
                            entropy_calc, // Passes ownership of original
                            progress_callback,
                        ) {
                            Ok(_) => {
                                log::info!("GPU WFC completed successfully.");

                                // Stop the visualization update thread if it exists
                                if let Some(handle) = _viz_handle {
                                    // Just let it terminate on its own
                                }

                                // Final visualization
                                if let Some(tx) = &viz_tx {
                                    // Send final grid state
                                    if let Err(e) =
                                        tx.send(VizMessage::UpdateGrid(Box::new(grid.clone())))
                                    {
                                        log::error!("Failed to send final grid state: {}", e);
                                    }
                                    // Signal visualization is finished
                                    if let Err(e) = tx.send(VizMessage::Finished) {
                                        log::error!(
                                            "Failed to send visualization finished signal: {}",
                                            e
                                        );
                                    }
                                }

                                // Save the grid using the passed config
                                if let Err(e) =
                                    output::save_grid_to_file(&grid, config.output_path.as_path())
                                {
                                    log::error!("Failed to save grid: {}", e);
                                    return Err(e);
                                }
                                Ok(())
                            }
                            Err(e) => {
                                log::error!("GPU WFC failed: {}", e);
                                // Still notify visualization to finish
                                if let Some(tx) = &viz_tx {
                                    let _ = tx.send(VizMessage::Finished);
                                }
                                Err(anyhow::anyhow!(e)) // Convert WfcError to anyhow::Error
                            }
                        }?; // Propagate error from run
                    }
                    Err(e) => {
                        log::error!(
                            "Failed to initialize GPU Accelerator: {}. Falling back to CPU.",
                            e
                        );
                        // Fallback to CPU if GPU initialization fails
                        run_cpu(
                            &mut grid,
                            &tileset,
                            &rules,
                            progress_callback,
                            &config,
                            viz_tx.clone(),
                        )?;
                    }
                }
            }
            #[cfg(not(feature = "gpu"))]
            {
                log::error!("GPU mode selected but GPU feature not compiled. Using CPU.");
                run_cpu(
                    &mut grid,
                    &tileset,
                    &rules,
                    progress_callback,
                    &config,
                    viz_tx.clone(),
                )?;
            }
        } else {
            log::info!("Running WFC on CPU...");
            run_cpu(
                &mut grid,
                &tileset,
                &rules,
                progress_callback,
                &config,
                viz_tx.clone(),
            )?;
        }
    }

    log::info!("Wave Forge App Finished.");
    Ok(())
}

/// Helper function to execute the Wave Function Collapse algorithm using the CPU implementation.
///
/// This function is called when GPU acceleration is disabled, unavailable, or fails to initialize.
/// It instantiates the appropriate CPU-based `ConstraintPropagator` and `EntropyCalculator`,
/// with parallel execution for larger grids, then calls the core `wfc_core::runner::run` function.
///
/// After successful completion, it saves the resulting grid to the specified output path.
/// Handles potential errors during the WFC run or file saving.
///
/// # Arguments
///
/// * `grid` - A mutable reference to the `PossibilityGrid` to be collapsed.
/// * `tileset` - A reference to the loaded `TileSet`.
/// * `rules` - A reference to the loaded `AdjacencyRules`.
/// * `progress_callback` - An optional callback function for reporting progress.
/// * `config` - A reference to the application configuration (`AppConfig`) for output settings.
/// * `viz_tx` - Optional sender channel for visualization updates.
///
/// # Returns
///
/// * `Ok(())` if the WFC run completes successfully and the output is saved.
/// * `Err(anyhow::Error)` if the WFC run fails or saving the output fails.
fn run_cpu(
    grid: &mut PossibilityGrid,
    tileset: &wfc_core::TileSet,
    rules: &wfc_core::rules::AdjacencyRules,
    progress_callback: Option<Box<dyn Fn(wfc_core::ProgressInfo) + Send + Sync>>,
    config: &AppConfig,
    viz_tx: Option<Sender<VizMessage>>,
) -> Result<(), anyhow::Error> {
    // Choose the appropriate propagator based on grid size
    let large_grid_threshold = 16; // Threshold to switch to parallel propagator
    let total_cells = grid.width * grid.height * grid.depth;
    let is_large_grid = grid.width >= large_grid_threshold
        || grid.height >= large_grid_threshold
        || grid.depth >= large_grid_threshold
        || total_cells >= large_grid_threshold.pow(3);

    log::info!(
        "CPU grid size: {}x{}x{} ({} cells). {}",
        grid.width,
        grid.height,
        grid.depth,
        total_cells,
        if is_large_grid {
            "Using parallel propagator for large grid"
        } else {
            "Using standard propagator"
        }
    );

    let run_result = if is_large_grid {
        // For large grids, use parallel propagator
        let propagator = wfc_core::ParallelConstraintPropagator::new();
        let entropy_calculator = wfc_core::entropy::CpuEntropyCalculator::new();

        // Run with owned components
        wfc_core::runner::run(
            grid,
            tileset,
            rules,
            propagator,         // Pass ownership
            entropy_calculator, // Pass ownership
            progress_callback,
        )
    } else {
        // For smaller grids, use standard single-threaded propagator
        let propagator = wfc_core::propagator::CpuConstraintPropagator::new();
        let entropy_calculator = wfc_core::entropy::CpuEntropyCalculator::new();

        // Run with owned components
        wfc_core::runner::run(
            grid,
            tileset,
            rules,
            propagator,         // Pass ownership
            entropy_calculator, // Pass ownership
            progress_callback,
        )
    };

    // Process result
    match run_result {
        Ok(_) => {
            log::info!("CPU WFC completed successfully.");

            // Final visualization
            if let Some(tx) = &viz_tx {
                // Send final grid state
                if let Err(e) = tx.send(VizMessage::UpdateGrid(Box::new(grid.clone()))) {
                    log::error!("Failed to send final grid state: {}", e);
                }
                // Signal visualization is finished
                if let Err(e) = tx.send(VizMessage::Finished) {
                    log::error!("Failed to send visualization finished signal: {}", e);
                }
            }

            // Save the grid using the passed config
            if let Err(e) = output::save_grid_to_file(grid, config.output_path.as_path()) {
                log::error!("Failed to save grid: {}", e);
                // Decide whether to return error or just log
                return Err(e);
            }
            Ok(())
        }
        Err(e) => {
            log::error!("CPU WFC failed: {}", e);
            // Still notify visualization to finish
            if let Some(tx) = &viz_tx {
                let _ = tx.send(VizMessage::Finished);
            }
            Err(anyhow::anyhow!(e)) // Convert WfcError to anyhow::Error
        }
    }
}
