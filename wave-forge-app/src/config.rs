use clap::Parser;
use std::path::PathBuf;

/// Parses a string like "10,10,5" into a tuple (usize, usize, usize).
fn parse_size(s: &str) -> Result<(usize, usize, usize), String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 3 {
        return Err("Size must be provided as width,height,depth (e.g., 10,10,5)".to_string());
    }
    let width = parts[0]
        .parse::<usize>()
        .map_err(|e| format!("Invalid width: {}", e))?;
    let height = parts[1]
        .parse::<usize>()
        .map_err(|e| format!("Invalid height: {}", e))?;
    let depth = parts[2]
        .parse::<usize>()
        .map_err(|e| format!("Invalid depth: {}", e))?;
    Ok((width, height, depth))
}

/// Multithreaded 3D GPU Terrain Generation using Wave Function Collapse
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct AppConfig {
    /// Grid dimensions (width,height,depth).
    #[arg(short, long, value_parser = parse_size, default_value = "10,10,10")]
    pub size: (usize, usize, usize),

    /// Path to the rule definition file (e.g., RON format).
    #[arg(short = 'r', long, value_name = "FILE")]
    pub rule_file_path: PathBuf,

    /// Use GPU acceleration if available.
    #[arg(long, default_value_t = false)]
    pub use_gpu: bool,

    /// Seed for the random number generator.
    #[arg(short, long)]
    pub seed: Option<u64>,

    /// Path to save the output grid.
    #[arg(short, long, value_name = "FILE")]
    pub output_path: Option<PathBuf>,
}

impl AppConfig {
    /// Parses command line arguments into an AppConfig instance.
    pub fn from_args() -> Self {
        Self::parse()
    }
}
