// wave-forge-app/src/main.rs

pub mod benchmark;
pub mod config;

use anyhow::Result;
use clap::Parser;
use config::AppConfig;
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
        // TODO: Select CPU/GPU components based on config.cpu_only and feature flags
        // TODO: Instantiate the correct propagator and entropy_calculator

        // Placeholder: Always use CPU for now
        let mut propagator = wfc_core::propagator::CpuConstraintPropagator::new();
        let entropy_calculator = wfc_core::entropy::CpuEntropyCalculator::new();

        // TODO: Set up progress reporting based on config.report_progress_interval
        let progress_callback = None; // Placeholder

        // Run WFC using the selected components
        // This call needs the runner signature to be fixed manually in wfc-core
        match wfc_core::runner::run(
            &mut grid,
            &tileset,
            &rules,
            &mut propagator,
            &entropy_calculator,
            progress_callback,
        ) {
            Ok(_) => {
                log::info!("WFC completed successfully.");
                // TODO: Implement output saving based on config.output_path
            }
            Err(e) => {
                log::error!("WFC failed: {}", e);
                // Return an error using anyhow
                return Err(anyhow::anyhow!(e));
            }
        }
    }

    log::info!("Wave Forge App Finished.");
    Ok(())
}
