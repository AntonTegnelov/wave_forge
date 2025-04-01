// wave-forge-app/src/main.rs

pub mod benchmark;
pub mod config;
pub mod output;
pub mod visualization;

use anyhow::Result;
use clap::Parser;
use config::AppConfig;
use config::VisualizationMode;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use visualization::{TerminalVisualizer, Visualizer};
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

    // --- Initialize Visualizer (if configured) ---
    #[allow(unused_variables, unused_mut)] // Allow unused for now
    let mut visualizer: Option<Box<dyn Visualizer + Send + Sync>> = match config.visualization_mode
    {
        VisualizationMode::None => None,
        VisualizationMode::Terminal => {
            log::info!("Terminal visualization enabled.");
            Some(Box::new(TerminalVisualizer {}))
        }
        VisualizationMode::Simple2D => {
            log::warn!("Simple2D visualization not yet implemented, using None.");
            // TODO: Instantiate Simple2DVisualizer when implemented
            None
        }
    };
    // TODO: Call initial display? visualizer.as_mut().map(|v| v.display_state(&initial_grid));
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
        let mut all_results: Vec<((usize, usize, usize), Result<BenchmarkTuple, anyhow::Error>)> =
            Vec::new();

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
                all_results.push(((width, height, depth), result.map_err(anyhow::Error::from)));
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

        let mut last_cpu_time: Option<Duration> = None;

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
                        "-"
                    );
                    last_cpu_time = Some(cpu_result.total_time); // Store for speedup calculation

                    // Print GPU result
                    let speedup_str = if let Some(cpu_time) = last_cpu_time {
                        if gpu_result.total_time > Duration::ZERO
                            && cpu_time > Duration::ZERO
                            && gpu_result.wfc_result.is_ok()
                            && cpu_result.wfc_result.is_ok()
                        {
                            format!(
                                "{:.2}x",
                                cpu_time.as_secs_f64() / gpu_result.total_time.as_secs_f64()
                            )
                        } else if gpu_result.wfc_result.is_err() {
                            "N/A (GPU Fail)".to_string()
                        } else {
                            "N/A (CPU Fail)".to_string()
                        }
                    } else {
                        "N/A".to_string()
                    };
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
                    last_cpu_time = None; // Reset for next size group
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
                    last_cpu_time = None; // Reset on error
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

        // --- Progress Reporting Setup ---
        let last_report_time = Arc::new(Mutex::new(Instant::now()));
        let report_interval = config.report_progress_interval;

        let progress_callback: Option<Box<dyn Fn(wfc_core::ProgressInfo) + Send + Sync>> =
            if let Some(interval) = report_interval {
                let last_report_time_clone = Arc::clone(&last_report_time); // Clone Arc here
                Some(Box::new(move |info: wfc_core::ProgressInfo| {
                    let mut last_time = last_report_time_clone.lock().unwrap(); // Use cloned Arc
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
            {
                log::info!("Initializing GPU Accelerator...");
                match wfc_gpu::accelerator::GpuAccelerator::new(&grid, &rules).await {
                    #[allow(unused_variables)] // Allow unused until run call is fixed
                    Ok(gpu_accelerator) => {
                        log::info!("Running WFC on GPU...");
                        // TODO: Ownership conflict! GpuAccelerator implements both traits,
                        //       but run() takes ownership, and GpuAccelerator is not Clone.
                        //       Requires refactoring GpuAccelerator or run() signature.
                        // match wfc_core::runner::run(
                        //     &mut grid,
                        //     &tileset,
                        //     &rules,
                        //     gpu_accelerator, // Passes ownership
                        //     gpu_accelerator, // Error: Use of moved value
                        //     progress_callback,
                        // ) { ... }
                        log::warn!("GPU run skipped due to ownership conflict.");
                        // Placeholder: Treat as error for now
                        return Err(anyhow::anyhow!("GPU run skipped due to ownership conflict"));
                    }
                    Err(e) => {
                        log::error!(
                            "Failed to initialize GPU Accelerator: {}. Falling back to CPU.",
                            e
                        );
                        // Fallback to CPU if GPU initialization fails
                        run_cpu(&mut grid, &tileset, &rules, progress_callback, &config)?;
                    }
                }
            }
            #[cfg(not(feature = "gpu"))]
            {
                log::error!("GPU mode selected but GPU feature not compiled. Using CPU.");
                run_cpu(&mut grid, &tileset, &rules, progress_callback, &config)?;
            }
        } else {
            log::info!("Running WFC on CPU...");
            run_cpu(&mut grid, &tileset, &rules, progress_callback, &config)?;
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
    config: &AppConfig,
) -> Result<(), anyhow::Error> {
    let propagator = wfc_core::propagator::CpuConstraintPropagator::new();
    let entropy_calculator = wfc_core::entropy::CpuEntropyCalculator::new();

    // Run with owned components (matches reverted signature)
    match wfc_core::runner::run(
        grid,
        tileset,
        rules,
        propagator,         // Pass ownership
        entropy_calculator, // Pass ownership
        progress_callback,
    ) {
        Ok(_) => {
            log::info!("CPU WFC completed successfully.");
            // Save the grid using the passed config
            if let Err(e) = output::save_grid_to_file(&grid, config.output_path.as_path()) {
                log::error!("Failed to save grid: {}", e);
                // Decide whether to return error or just log
                return Err(e);
            }
            Ok(())
        }
        Err(e) => {
            log::error!("CPU WFC failed: {}", e);
            Err(anyhow::anyhow!(e)) // Convert WfcError to anyhow::Error
        }
    }
}
