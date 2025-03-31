use clap::Parser;
use std::path::PathBuf;

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
    // Note: use_gpu can be inferred later based on gpu availability and cpu_only flag.
}
