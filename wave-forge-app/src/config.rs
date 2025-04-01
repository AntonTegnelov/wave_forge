use clap::{Parser, ValueEnum};
use std::path::PathBuf;
use std::time::Duration;

/// Represents the different visualization modes available.
#[derive(ValueEnum, Clone, Debug, Default, PartialEq)]
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

    /// Optional: Path to save benchmark results as a CSV file.
    /// Only used if benchmark_mode is also enabled.
    #[arg(long, value_name = "CSV_FILE")]
    pub benchmark_csv_output: Option<PathBuf>,
    // Note: use_gpu can be inferred later based on gpu availability and cpu_only flag.
}

#[cfg(test)]
mod tests {
    use super::*; // Import items from parent module (config)
    use std::time::Duration;

    #[test]
    fn test_basic_args() {
        let args = vec![
            "wave-forge",
            "--rule-file",
            "rules.ron",
            "--width",
            "20",
            "--output-path",
            "out.txt",
        ];
        let config = AppConfig::try_parse_from(args).unwrap();
        assert_eq!(config.rule_file, PathBuf::from("rules.ron"));
        assert_eq!(config.width, 20);
        assert_eq!(config.height, 10); // Default
        assert_eq!(config.depth, 10); // Default
        assert_eq!(config.output_path, PathBuf::from("out.txt"));
        assert_eq!(config.cpu_only, false); // Default
        assert_eq!(config.benchmark_mode, false); // Default
        assert_eq!(config.report_progress_interval, None); // Default
        assert_eq!(config.visualization_mode, VisualizationMode::None); // Default
    }

    #[test]
    fn test_cpu_only_flag() {
        let args = vec!["wave-forge", "--rule-file", "r.ron", "--cpu-only"];
        let config = AppConfig::try_parse_from(args).unwrap();
        assert!(config.cpu_only);
    }

    #[test]
    fn test_benchmark_flag() {
        let args = vec!["wave-forge", "--rule-file", "r.ron", "--benchmark-mode"];
        let config = AppConfig::try_parse_from(args).unwrap();
        assert!(config.benchmark_mode);
    }

    #[test]
    fn test_progress_interval() {
        let args = vec![
            "wave-forge",
            "--rule-file",
            "r.ron",
            "--report-progress-interval",
            "2s",
        ];
        let config = AppConfig::try_parse_from(args).unwrap();
        assert_eq!(
            config.report_progress_interval,
            Some(Duration::from_secs(2))
        );
    }

    // TODO: Add test for visualization mode enum parsing
    #[test]
    fn test_visualization_mode() {
        let args = vec![
            "wave-forge",
            "--rule-file",
            "r.ron",
            "--visualization-mode",
            "terminal",
        ];
        let config = AppConfig::try_parse_from(args).unwrap();
        assert_eq!(config.visualization_mode, VisualizationMode::Terminal);

        let args_err = vec![
            "wave-forge",
            "--rule-file",
            "r.ron",
            "--visualization-mode",
            "invalid-mode",
        ];
        assert!(AppConfig::try_parse_from(args_err).is_err());
    }

    #[test]
    fn test_benchmark_csv_output_flag() {
        let args = vec![
            "wave-forge",
            "--rule-file",
            "r.ron",
            "--benchmark-mode", // CSV output only makes sense with benchmark mode
            "--benchmark-csv-output",
            "bench_results.csv",
        ];
        let config = AppConfig::try_parse_from(args).unwrap();
        assert!(config.benchmark_mode);
        assert_eq!(
            config.benchmark_csv_output,
            Some(PathBuf::from("bench_results.csv"))
        );
    }

    #[test]
    fn test_benchmark_csv_output_flag_no_benchmark_mode() {
        // Should parse ok, but the path might be ignored later if benchmark_mode is false
        let args = vec![
            "wave-forge",
            "--rule-file",
            "r.ron",
            "--benchmark-csv-output",
            "results.csv",
        ];
        let config = AppConfig::try_parse_from(args).unwrap();
        assert_eq!(
            config.benchmark_csv_output,
            Some(PathBuf::from("results.csv"))
        );
        assert!(!config.benchmark_mode); // Verify benchmark mode is off
    }
}
