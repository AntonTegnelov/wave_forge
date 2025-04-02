//! Handles the core execution logic for standard and benchmark modes.

use crate::benchmark::{self, BenchmarkResultTuple};
use crate::config::{AppConfig, ProgressLogLevel};
use crate::output;
use crate::setup::visualization::VizMessage;
use anyhow::{Context, Result};
use log;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use wfc_core::grid::PossibilityGrid;
use wfc_core::rules::AdjacencyRules;
use wfc_core::runner;
use wfc_core::{ProgressInfo, TileSet};
use wfc_gpu::accelerator::GpuAccelerator;

pub async fn run_benchmark_mode(
    config: &AppConfig,
    tileset: &TileSet,
    rules: &AdjacencyRules,
    grid: &mut PossibilityGrid, // Grid is modified by benchmark?
    viz_tx: &Option<Sender<VizMessage>>,
    snapshot_handle: &mut Option<thread::JoinHandle<()>>, // Needs to be mutable to assign
) -> Result<()> {
    log::info!("Benchmark mode enabled (GPU only).");

    // Define benchmark scenarios (dimensions)
    let benchmark_dimensions = [
        (8, 8, 8),    // Small
        (16, 16, 16), // Medium
    ];
    log::info!("Running benchmarks for sizes: {:?}", benchmark_dimensions);

    // Store results for final report (simplified)
    let mut benchmark_results: Vec<BenchmarkResultTuple> = Vec::new();

    for &(width, height, depth) in &benchmark_dimensions {
        log::info!(
            "Starting GPU benchmark for size: {}x{}x{}",
            width,
            height,
            depth
        );
        // Need a *new* grid for each benchmark run to start fresh
        let mut bench_grid = PossibilityGrid::new(width, height, depth, tileset.weights.len());
        let result = benchmark::run_single_benchmark(&mut bench_grid, tileset, rules).await;
        benchmark_results.push(((width, height, depth), result.map_err(anyhow::Error::from)));
    }

    // Report Summary (Remains the same)
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

    // Write to CSV if requested
    if let Some(csv_path) = &config.benchmark_csv_output {
        if let Err(e) = benchmark::write_results_to_csv(&benchmark_results, csv_path) {
            log::error!("Failed to write benchmark results to CSV: {}", e);
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
) -> Result<()> {
    log::info!("Running WFC (GPU only)...");

    // GPU is mandatory
    let grid_snapshot = Arc::new(Mutex::new(grid.clone())); // Snapshot for GPU/Viz

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
    let progress_callback: Option<Box<dyn Fn(ProgressInfo) + Send + Sync>> =
        if let Some(interval) = report_interval {
            let last_report_time_clone = Arc::clone(&last_report_time);
            Some(Box::new(move |info: ProgressInfo| {
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
                    // Use the cloned progress_log_level
                    match progress_log_level {
                        ProgressLogLevel::Trace => log::trace!("{}", msg),
                        ProgressLogLevel::Debug => log::debug!("{}", msg),
                        ProgressLogLevel::Info => log::info!("{}", msg),
                        ProgressLogLevel::Warn => log::warn!("{}", msg),
                    }
                    *last_time = now;
                }
            }))
        } else {
            None
        };

    // Initialize GPU
    log::info!("Initializing GPU Accelerator...");
    // Need to lock the original grid_snapshot for init, not the grid passed in?
    let gpu_accelerator = {
        let grid_guard = grid_snapshot.lock().expect("GPU init mutex poisoned");
        GpuAccelerator::new(&grid_guard, rules).await?
        // lock is dropped here
    };

    // Run WFC
    log::info!("Running WFC on GPU...");
    let propagator = gpu_accelerator.clone();
    let entropy_calc = gpu_accelerator;

    match runner::run(
        grid,
        tileset,
        rules,
        propagator,
        entropy_calc,
        progress_callback,
    ) {
        Ok(_) => {
            log::info!("GPU WFC completed successfully.");
            // Final visualization update
            if let Some(tx) = viz_tx {
                let final_grid_guard = grid_snapshot.lock().expect("Final viz mutex poisoned");
                let _ = tx.send(VizMessage::UpdateGrid(Box::new(final_grid_guard.clone())));
            }
            // Save grid
            output::save_grid_to_file(grid, config.output_path.as_path())?
        }
        Err(e) => {
            log::error!("GPU WFC failed: {}", e);
            // Convert WfcError to anyhow::Error
            return Err(anyhow::anyhow!(e));
        }
    }

    Ok(())
}
