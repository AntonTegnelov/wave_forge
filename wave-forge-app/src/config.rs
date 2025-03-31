use std::path::PathBuf;

/// Application configuration parameters derived from command-line arguments.
#[derive(Debug)] // Using Debug for now, might switch to clap::Parser later
pub struct AppConfig {
    /// Grid dimensions (width, height, depth).
    pub size: (usize, usize, usize),
    /// Path to the rule definition file (e.g., RON format).
    pub rule_file_path: PathBuf,
    /// Whether to use GPU acceleration.
    pub use_gpu: bool,
    /// Optional seed for the random number generator.
    pub seed: Option<u64>,
    /// Optional path to save the output grid.
    pub output_path: Option<PathBuf>,
}
