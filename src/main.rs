//! # Wave Forge Application Main Module
//!
//! Contains the main application logic.

use crate::setup::visualization;
use crate::setup::visualization::VizMessage;
use crate::AppConfig;
use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use figment::{
    providers::{Env, Format, Serialized, Toml},
    Figment,
};
use log;
use logging;
use setup::execution;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use wfc_core::grid::PossibilityGrid;
use wfc_rules::loader::load_from_file;

/// thin wrapper around the atual entry point since main can´t be async
pub fn main() -> anyhow::Result<()> {
    match tokio::runtime::Runtime::new() {
        Ok(rt) => rt.block_on(run_app()),
        Err(e) => Err(anyhow::anyhow!("Failed to create Tokio runtime: {}", e)),
    }
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
pub async fn run_app() -> Result<()> {
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

    // 1. Load Configuration (Figment + Clap)
    // Order: Defaults (optional) < TOML File < Env Vars < CLI Args
    let config: AppConfig = Figment::new()
        // Optional: Add Serialized::defaults(AppConfig::default()) here if you have defaults
        .merge(Toml::file("WaveForge.toml").nested())
        .merge(Env::prefixed("WAVEFORGE_").split("__"))
        .merge(Serialized::defaults(AppConfig::parse())) // Parse clap args and merge last
        .extract()
        .context("Failed to load configuration")?;

    // --- Validate Configuration ---
    if config.benchmark_mode && config.benchmark_csv_output.is_none() {
        log::warn!("Running in benchmark mode without --benchmark-csv-output specified. Results will not be saved to CSV.");
    }
    // Add other validations here...

    // Initialize logging with the configured log level
    logging::init_logger(&config);

    log::info!("Wave Forge App Starting");
    log::debug!("Loaded Config: {:?}", config);

    // --- Initialize Visualizer in a separate thread if configured ---
    let (viz_tx, main_viz_handle) = visualization::setup_visualization(&config);

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
    if let Some(viz_sender) = &viz_tx {
        log::info!("Sending initial grid state to visualization thread");
        if let Err(e) = viz_sender.send(VizMessage::UpdateGrid(Box::new(grid.clone()))) {
            log::error!("Failed to send initial grid state: {}", e);
        }
    }

    // Declare handles *before* the benchmark/run split
    let mut snapshot_handle: Option<thread::JoinHandle<()>> = None;

    // Call the appropriate execution function
    let run_result = if config.benchmark_mode {
        execution::run_benchmark_mode(
            &config,
            &viz_tx,
            &mut snapshot_handle,
            shutdown_signal.clone(),
        )
        .await
    } else {
        execution::run_standard_mode(
            &config,
            &tileset,
            &rules,
            &mut grid,
            &viz_tx,
            &mut snapshot_handle,
            shutdown_signal.clone(),
        )
        .await
    };

    // Handle the AppError after await
    run_result.map_err(|app_err| anyhow::Error::new(app_err))?;

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
