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
pub mod logging;
pub mod output;
pub mod profiler;
pub mod progress;
pub mod setup;
pub mod visualization;

use anyhow::Result;
use clap::Parser;
use config::AppConfig;
use logging::init_logger;
use setup::visualization::{setup_visualization, VizMessage};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use visualization::{TerminalVisualizer, Visualizer};
use wfc_core::grid::PossibilityGrid;
use wfc_rules::loader::load_from_file;

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
    // --- Setup Shutdown Signal ---
    let shutdown_signal = Arc::new(AtomicBool::new(false));
    let signal_handler_shutdown = shutdown_signal.clone();

    tokio::spawn(async move {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
        log::warn!("Ctrl+C received, initiating graceful shutdown...");
        signal_handler_shutdown.store(true, Ordering::Relaxed);
    });

    // Parse command-line arguments first (we need the config for logger setup)
    let config = AppConfig::parse();

    // Initialize logging with the configured log level
    init_logger(&config);

    log::info!("Wave Forge App Starting");
    log::debug!("Loaded Config: {:?}", config);

    // --- Initialize Visualizer in a separate thread if configured ---
    let (viz_tx, main_viz_handle) = setup_visualization(&config);

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

    // Call the appropriate execution function
    if config.benchmark_mode {
        setup::execution::run_benchmark_mode(
            &config,
            &tileset,
            &rules,
            &mut grid,
            &viz_tx,
            &mut snapshot_handle,
            shutdown_signal.clone(),
        )
        .await?;
    } else {
        setup::execution::run_standard_mode(
            &config,
            &tileset,
            &rules,
            &mut grid,
            &viz_tx,
            &mut snapshot_handle,
            shutdown_signal.clone(),
        )
        .await?;
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
