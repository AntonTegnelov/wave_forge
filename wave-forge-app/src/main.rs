//! # Wave Forge Application
//!
//! This is the main executable crate for the Wave Forge project.
//! It provides a command-line interface to:
//! - Run the Wave Function Collapse (WFC) algorithm on specified rule files and grid dimensions
//!   using GPU acceleration.
//! - Benchmark GPU WFC performance.
//! - Configure output paths, visualization modes, and progress reporting.

// wave-forge-app/src/main.rs

pub mod benchmark;
pub mod config;
pub mod output;
pub mod profiler;
pub mod progress;
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
}

/// Initializes the logger with the appropriate configuration based on the application settings.
///
/// This function configures the log levels for the different components of the application:
/// - The progress reports use the level specified in `config.progress_log_level`
/// - Other module levels can be controlled via the RUST_LOG environment variable
///
/// If RUST_LOG is set, it is respected and combined with the progress log level configuration.
/// If no RUST_LOG is provided, a default configuration is used.
///
/// # Arguments
///
/// * `config` - The application configuration containing the progress log level setting
fn init_logger(config: &AppConfig) {
    use env_logger::{Builder, Env};
    use log::LevelFilter;

    // Convert ProgressLogLevel to log::LevelFilter
    let progress_level = match config.progress_log_level {
        config::ProgressLogLevel::Trace => LevelFilter::Trace,
        config::ProgressLogLevel::Debug => LevelFilter::Debug,
        config::ProgressLogLevel::Info => LevelFilter::Info,
        config::ProgressLogLevel::Warn => LevelFilter::Warn,
    };

    // Convert GlobalLogLevel to log::LevelFilter
    let global_level = match config.global_log_level {
        config::GlobalLogLevel::Trace => LevelFilter::Trace,
        config::GlobalLogLevel::Debug => LevelFilter::Debug,
        config::GlobalLogLevel::Info => LevelFilter::Info,
        config::GlobalLogLevel::Warn => LevelFilter::Warn,
        config::GlobalLogLevel::Error => LevelFilter::Error,
    };

    // Start with the environment configuration, but allow for override
    let env = Env::default().filter_or("RUST_LOG", "info");

    // Create a new builder
    let mut builder = Builder::from_env(env);

    // Set the default global log level
    builder.filter_level(global_level);

    // Set the progress module's log level specifically (prioritize this over global level)
    builder.filter_module("wave_forge_app::progress", progress_level);

    // Initialize the logger
    builder.init();

    log::debug!(
        "Logger initialized with global log level: {:?}, progress log level: {:?}",
        config.global_log_level,
        config.progress_log_level
    );
}

/// The main entry point for the Wave Forge application.
///
/// Parses command-line arguments using `clap`, initializes logging, loads WFC rules,
/// sets up the grid, and then either runs the benchmarking suite or the standard
/// WFC algorithm based on the provided configuration.
///
/// Uses GPU acceleration exclusively via the `wfc-gpu` crate.
/// Orchestrates progress reporting and final output saving.
///
/// Uses `tokio` for the async runtime, primarily for the asynchronous GPU initialization.
#[tokio::main]
async fn main() -> Result<()> {
    // Parse command-line arguments first (we need the config for logger setup)
    let config = AppConfig::parse();

    // Initialize logging with the configured log level
    init_logger(&config);

    log::info!("Wave Forge App Starting");
    log::debug!("Loaded Config: {:?}", config);

    // --- Initialize Visualizer in a separate thread if configured ---
    let mut main_viz_handle: Option<thread::JoinHandle<()>> = None;
    let viz_tx: Option<Sender<VizMessage>> = match config.visualization_mode {
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

            let handle = thread::spawn(move || {
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

            main_viz_handle = Some(handle);
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

    // Declare handles *before* the benchmark/run split
    let mut snapshot_handle: Option<thread::JoinHandle<()>> = None;

    if config.benchmark_mode {
        log::info!("Benchmark mode enabled (GPU only).");

        // Define benchmark scenarios (dimensions)
        let benchmark_dimensions = [
            (8, 8, 8),    // Small
            (16, 16, 16), // Medium
        ];
        log::info!("Running benchmarks for sizes: {:?}", benchmark_dimensions);

        // Store results for final report (simplified)
        let mut benchmark_results: Vec<benchmark::BenchmarkResultTuple> = Vec::new();

        for &(width, height, depth) in &benchmark_dimensions {
            log::info!(
                "Starting GPU benchmark for size: {}x{}x{}",
                width,
                height,
                depth
            );
            let mut gpu_grid = PossibilityGrid::new(width, height, depth, tileset.weights.len());
            let result = benchmark::run_single_benchmark(&mut gpu_grid, &tileset, &rules).await;
            benchmark_results.push(((width, height, depth), result.map_err(anyhow::Error::from)));
        }

        // --- Report Summary ---
        println!("\n--- GPU Benchmark Summary ---");
        println!(
            "Rule File: {:?}",
            config.rule_file.file_name().unwrap_or_default()
        );
        println!("Num Tiles: {}", tileset.weights.len());
        println!("------------------------------------------------------------");
        println!("Size (WxHxD)    | Total Time | Iterations | Collapsed Cells | Result");
        println!("----------------|------------|------------|-----------------|----------");

        for ((w, h, d), result_item) in &benchmark_results {
            let size_str = format!("{}x{}x{}", w, h, d);
            match result_item {
                Ok(gpu_result) => {
                    println!(
                        "{:<15} | {:<10?} | {:<10} | {:<15} | {:<8}",
                        size_str,
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
                        }
                    );
                }
                Err(e) => {
                    println!("{:<15} | Error running benchmark: {} |", size_str, e);
                }
            }
        }
        println!("------------------------------------------------------------");

        // --- Write to CSV if requested ---
        if let Some(csv_path) = &config.benchmark_csv_output {
            if let Err(e) = benchmark::write_results_to_csv(&benchmark_results, csv_path) {
                log::error!("Failed to write benchmark results to CSV: {}", e);
            }
        }

        // Setup visualization thread (needs Arc/Mutex for grid)
        let grid_snapshot = Arc::new(Mutex::new(grid.clone()));
        if let Some(tx) = &viz_tx {
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
                        let grid_clone = {
                            grid_snapshot_for_viz
                                .lock()
                                .expect("Visualization mutex poisoned")
                                .clone()
                        };
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
            snapshot_handle = Some(handle);
        } else {
            // Ensure snapshot_handle is None if viz is off
            snapshot_handle = None;
        };

        // Setup progress reporting (needs Arc/Mutex for timer)
        let last_report_time = Arc::new(Mutex::new(Instant::now()));
        let report_interval = config.report_progress_interval;
        let progress_log_level = config.progress_log_level.clone();
        let progress_callback: Option<Box<dyn Fn(wfc_core::ProgressInfo) + Send + Sync>> =
            if let Some(interval) = report_interval {
                let last_report_time_clone = Arc::clone(&last_report_time);
                Some(Box::new(move |info: wfc_core::ProgressInfo| {
                    let now = Instant::now();
                    let mut last_time = last_report_time_clone
                        .lock()
                        .expect("Progress mutex poisoned");
                    if now.duration_since(*last_time) >= interval {
                        let percentage = if info.total_cells > 0 {
                            (info.collapsed_cells as f32 / info.total_cells as f32) * 100.0
                        } else {
                            100.0
                        };
                        let msg = format!(
                            "Progress: Iter {}, Collapsed {}/{} ({:.1}%)",
                            info.iteration, info.collapsed_cells, info.total_cells, percentage
                        );
                        match progress_log_level {
                            config::ProgressLogLevel::Trace => log::trace!("{}", msg),
                            config::ProgressLogLevel::Debug => log::debug!("{}", msg),
                            config::ProgressLogLevel::Info => log::info!("{}", msg),
                            config::ProgressLogLevel::Warn => log::warn!("{}", msg),
                        }
                        *last_time = now;
                    }
                }))
            } else {
                None
            };

        // Initialize GPU
        log::info!("Initializing GPU Accelerator...");
        match wfc_gpu::accelerator::GpuAccelerator::new(
            &grid_snapshot.lock().expect("GPU init mutex poisoned"),
            &rules,
        )
        .await
        {
            Ok(gpu_accelerator) => {
                log::info!("Running WFC on GPU...");
                let propagator = gpu_accelerator.clone();
                let entropy_calc = gpu_accelerator;
                // Run the WFC algorithm
                match wfc_core::runner::run(
                    &mut grid,
                    &tileset,
                    &rules,
                    propagator,
                    entropy_calc,
                    progress_callback,
                ) {
                    Ok(_) => {
                        log::info!("GPU WFC completed successfully.");
                        // Final visualization update
                        if let Some(tx) = &viz_tx {
                            let _ = tx.send(VizMessage::UpdateGrid(Box::new(grid.clone())));
                        }
                        // Save grid
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
                        Err(anyhow::anyhow!(e))
                    }
                }?
            }
            Err(e) => {
                log::error!(
                    "CRITICAL: Failed to initialize GPU Accelerator: {}. Cannot continue.",
                    e
                );
                return Err(anyhow::anyhow!("GPU Initialization Failed: {}", e));
            }
        }
    } else {
        log::info!("Running WFC (GPU only)...");

        // GPU is mandatory
        // Setup visualization thread (needs Arc/Mutex for grid)
        let grid_snapshot = Arc::new(Mutex::new(grid.clone()));
        if let Some(tx) = &viz_tx {
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
                        let grid_clone = {
                            grid_snapshot_for_viz
                                .lock()
                                .expect("Visualization mutex poisoned")
                                .clone()
                        };
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
            snapshot_handle = Some(handle);
        } else {
            // Ensure snapshot_handle is None if viz is off
            snapshot_handle = None;
        };

        // Setup progress reporting (needs Arc/Mutex for timer)
        let last_report_time = Arc::new(Mutex::new(Instant::now()));
        let report_interval = config.report_progress_interval;
        let progress_log_level = config.progress_log_level.clone();
        let progress_callback: Option<Box<dyn Fn(wfc_core::ProgressInfo) + Send + Sync>> =
            if let Some(interval) = report_interval {
                let last_report_time_clone = Arc::clone(&last_report_time);
                Some(Box::new(move |info: wfc_core::ProgressInfo| {
                    let now = Instant::now();
                    let mut last_time = last_report_time_clone
                        .lock()
                        .expect("Progress mutex poisoned");
                    if now.duration_since(*last_time) >= interval {
                        let percentage = if info.total_cells > 0 {
                            (info.collapsed_cells as f32 / info.total_cells as f32) * 100.0
                        } else {
                            100.0
                        };
                        let msg = format!(
                            "Progress: Iter {}, Collapsed {}/{} ({:.1}%)",
                            info.iteration, info.collapsed_cells, info.total_cells, percentage
                        );
                        match progress_log_level {
                            config::ProgressLogLevel::Trace => log::trace!("{}", msg),
                            config::ProgressLogLevel::Debug => log::debug!("{}", msg),
                            config::ProgressLogLevel::Info => log::info!("{}", msg),
                            config::ProgressLogLevel::Warn => log::warn!("{}", msg),
                        }
                        *last_time = now;
                    }
                }))
            } else {
                None
            };

        // Initialize GPU
        log::info!("Initializing GPU Accelerator...");
        match wfc_gpu::accelerator::GpuAccelerator::new(
            &grid_snapshot.lock().expect("GPU init mutex poisoned"),
            &rules,
        )
        .await
        {
            Ok(gpu_accelerator) => {
                log::info!("Running WFC on GPU...");
                let propagator = gpu_accelerator.clone();
                let entropy_calc = gpu_accelerator;
                // Run the WFC algorithm
                match wfc_core::runner::run(
                    &mut grid,
                    &tileset,
                    &rules,
                    propagator,
                    entropy_calc,
                    progress_callback,
                ) {
                    Ok(_) => {
                        log::info!("GPU WFC completed successfully.");
                        // Final visualization update
                        if let Some(tx) = &viz_tx {
                            let _ = tx.send(VizMessage::UpdateGrid(Box::new(grid.clone())));
                        }
                        // Save grid
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
                        Err(anyhow::anyhow!(e))
                    }
                }?
            }
            Err(e) => {
                log::error!(
                    "CRITICAL: Failed to initialize GPU Accelerator: {}. Cannot continue.",
                    e
                );
                return Err(anyhow::anyhow!("GPU Initialization Failed: {}", e));
            }
        }
    }

    log::info!("Wave Forge App Finished.");

    // --- Cleanup ---
    log::debug!("Dropping visualization sender...");
    drop(viz_tx);

    if let Some(handle) = snapshot_handle {
        log::debug!("Joining snapshot thread...");
        if let Err(e) = handle.join() {
            log::error!("Error joining snapshot thread: {:?}", e);
        }
    }

    if let Some(handle) = main_viz_handle {
        log::debug!("Joining main visualization thread...");
        if let Err(e) = handle.join() {
            log::error!("Error joining main visualization thread: {:?}", e);
        }
    }

    Ok(())
}
