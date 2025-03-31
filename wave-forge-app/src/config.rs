use clap::{Parser, ValueEnum};
use std::path::PathBuf;
use std::time::Duration;

/// Represents the different visualization modes available.
#[derive(ValueEnum, Clone, Debug, Default)]
pub enum VisualizationMode {
    #[default] // Default to None
    None,
    Terminal, // Simple text-based output
    Simple2D, // Placeholder for a simple 2D slice view
}

/// Configuration for the Wave Forge application.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct AppConfig {
    /// Path to the RON rule file defining tiles and adjacencies.
    #[arg(short, long, value_name = "FILE")]
    pub rule_file: PathBuf,

    /// Width of the output grid.
    #[arg(long, default_value_t = 10)]
    pub width: usize,

    /// Height of the output grid.
    #[arg(long, default_value_t = 10)]
    pub height: usize,

    /// Depth of the output grid.
    #[arg(long, default_value_t = 10)]
    pub depth: usize,

    /// Optional seed for the random number generator.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Path to save the generated output grid.
    #[arg(short, long, value_name = "FILE", default_value = "output.txt")]
    pub output_path: PathBuf,

    /// Force using the CPU implementation even if GPU is available.
    #[arg(long, default_value_t = false)]
    pub cpu_only: bool,

    /// Run in benchmark mode, comparing CPU and GPU performance.
    #[arg(long, default_value_t = false)]
    pub benchmark_mode: bool,

    /// Report progress updates every specified interval (e.g., "1s", "500ms").
    #[arg(long, value_name = "DURATION", value_parser = humantime::parse_duration)]
    pub report_progress_interval: Option<Duration>,

    /// Choose the visualization mode.
    #[arg(long, value_enum, default_value_t = VisualizationMode::None)]
    pub visualization_mode: VisualizationMode,
    // Note: use_gpu can be inferred later based on gpu availability and cpu_only flag.
}
