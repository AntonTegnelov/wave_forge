// wave-forge-app/src/main.rs

pub mod benchmark;
pub mod config;
pub mod visualization;

use anyhow::Result;
use clap::Parser;
use config::AppConfig;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use wfc_core::grid::PossibilityGrid;
use wfc_rules::loader::load_from_file;

// Make main async
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging (using env_logger)
    env_logger::init();

    // Parse command-line arguments
    let config = AppConfig::parse();

    log::info!("Wave Forge App Starting");
    log::debug!("Loaded Config: {:?}", config);

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

    if config.benchmark_mode {
        log::info!("Benchmark mode enabled.");

        // Ensure the GPU feature is enabled if comparing implementations
        #[cfg(not(feature = "gpu"))]
        {
            log::error!("Benchmark comparison requires the 'gpu' feature to be enabled.");
            return Err(anyhow::anyhow!("GPU feature not enabled for benchmark"));
        }

        #[cfg(feature = "gpu")]
        {
            // Run comparison benchmark
            log::info!("Running CPU vs GPU benchmark...");
            let initial_grid_for_bench = grid.clone(); // Clone initial state
            match benchmark::compare_implementations(&initial_grid_for_bench, &tileset, &rules)
                .await
            {
                Ok((cpu_result, gpu_result)) => {
                    benchmark::report_comparison(&cpu_result, &gpu_result);
                }
                Err(e) => {
                    log::error!("Benchmark comparison failed: {}", e);
                    return Err(e);
                }
            }
        }
    } else {
        log::info!("Running standard WFC...");

        let use_gpu = !config.cpu_only && cfg!(feature = "gpu");

        // --- Progress Reporting Setup ---
        let last_report_time = Arc::new(Mutex::new(Instant::now()));
        let report_interval = config.report_progress_interval;

        let progress_callback: Option<Box<dyn Fn(wfc_core::ProgressInfo) + Send + Sync>> =
            if let Some(interval) = report_interval {
                Some(Box::new(move |info: wfc_core::ProgressInfo| {
                    let mut last_time = last_report_time.lock().unwrap();
                    if last_time.elapsed() >= interval {
                        let percentage = if info.total_cells > 0 {
                            (info.collapsed_cells as f32 / info.total_cells as f32) * 100.0
                        } else {
                            100.0 // Grid is empty, consider it 100% done?
                        };
                        // Simple console log for progress
                        log::info!(
                            "Progress: Iteration {}, Collapsed {}/{} ({:.1}%)                    ",
                            info.iteration,
                            info.collapsed_cells,
                            info.total_cells,
                            percentage
                        );
                        *last_time = Instant::now(); // Reset timer
                    }
                }))
            } else {
                None
            };
        // --- End Progress Reporting Setup ---

        if use_gpu {
            #[cfg(feature = "gpu")]
            // This cfg block ensures GpuAccelerator code is only compiled with feature
            {
                log::info!("Initializing GPU Accelerator...");
                match wfc_gpu::accelerator::GpuAccelerator::new(&grid, &rules).await {
                    Ok(mut gpu_accelerator) => {
                        log::info!("Running WFC on GPU...");
                        // Assuming run signature is fixed to take refs
                        match wfc_core::runner::run(
                            &mut grid,
                            &tileset,
                            &rules,
                            &mut gpu_accelerator, // Pass the single instance mutably
                            &gpu_accelerator,     // Pass the single instance immutably
                            progress_callback.clone(), // Pass the callback (clone Arc)
                        ) {
                            Ok(_) => log::info!("GPU WFC completed successfully."),
                            Err(e) => {
                                log::error!("GPU WFC failed: {}", e);
                                return Err(anyhow::anyhow!(e));
                            }
                        }
                    }
                    Err(e) => {
                        log::error!(
                            "Failed to initialize GPU Accelerator: {}. Falling back to CPU.",
                            e
                        );
                        // Fallback to CPU if GPU initialization fails
                        run_cpu(&mut grid, &tileset, &rules, progress_callback)?;
                    }
                }
            }
            #[cfg(not(feature = "gpu"))]
            {
                log::error!("GPU mode selected but GPU feature not compiled. Using CPU.");
                run_cpu(&mut grid, &tileset, &rules, progress_callback)?;
            }
        } else {
            log::info!("Running WFC on CPU...");
            run_cpu(&mut grid, &tileset, &rules, progress_callback)?;
        }
    }

    log::info!("Wave Forge App Finished.");
    Ok(())
}

// Helper function to run WFC on CPU
fn run_cpu(
    grid: &mut PossibilityGrid,
    tileset: &wfc_core::TileSet,
    rules: &wfc_core::rules::AdjacencyRules,
    progress_callback: Option<Box<dyn Fn(wfc_core::ProgressInfo) + Send + Sync>>,
) -> Result<(), anyhow::Error> {
    let mut propagator = wfc_core::propagator::CpuConstraintPropagator::new();
    let entropy_calculator = wfc_core::entropy::CpuEntropyCalculator::new();

    // Assuming run signature is fixed to take refs
    match wfc_core::runner::run(
        grid,
        tileset,
        rules,
        &mut propagator,
        &entropy_calculator,
        progress_callback, // Pass the callback (already an Option<Box>)
    ) {
        Ok(_) => {
            log::info!("CPU WFC completed successfully.");
            Ok(())
        }
        Err(e) => {
            log::error!("CPU WFC failed: {}", e);
            Err(anyhow::anyhow!(e)) // Convert WfcError to anyhow::Error
        }
    }
}
