//! Handles command-line argument parsing and application configuration.

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

/// Log level for progress reporting.
#[derive(ValueEnum, Clone, Debug, Default, PartialEq)]
pub enum ProgressLogLevel {
    /// Use trace level for progress reports (very verbose)
    Trace,
    /// Use debug level for progress reports (detailed)
    Debug,
    /// Use info level for progress reports (normal)
    #[default]
    Info,
    /// Use warn level for progress reports (less frequent)
    Warn,
}

/// Global log level for all application components.
#[derive(ValueEnum, Clone, Debug, PartialEq)]
pub enum GlobalLogLevel {
    /// Trace level - extremely verbose (all details)
    Trace,
    /// Debug level - detailed information for debugging
    Debug,
    /// Info level - general information about program execution
    Info,
    /// Warn level - potentially harmful situations
    Warn,
    /// Error level - error events that might still allow the application to continue
    Error,
}

impl Default for GlobalLogLevel {
    fn default() -> Self {
        GlobalLogLevel::Info
    }
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

    /// Run in benchmark mode (now GPU only).
    #[arg(long, default_value_t = false)]
    pub benchmark_mode: bool,

    /// Report progress updates every specified interval (e.g., "1s", "500ms").
    #[arg(long, value_name = "DURATION", value_parser = humantime::parse_duration)]
    pub report_progress_interval: Option<Duration>,

    /// Log level to use for progress reporting.
    #[arg(long, value_enum, default_value_t = ProgressLogLevel::Info)]
    pub progress_log_level: ProgressLogLevel,

    /// Global log level for the application (overrides RUST_LOG if provided).
    #[arg(long, value_enum, default_value_t = GlobalLogLevel::Info)]
    pub global_log_level: GlobalLogLevel,

    /// Choose the visualization mode.
    #[arg(long, value_enum, default_value_t = VisualizationMode::None)]
    pub visualization_mode: VisualizationMode,

    /// Key to toggle visualization on/off during runtime (single character).
    #[arg(long, default_value = "T")]
    pub visualization_toggle_key: char,

    /// Optional: Path to save benchmark results as a CSV file.
    /// Only used if benchmark_mode is also enabled.
    #[arg(long, value_name = "CSV_FILE")]
    pub benchmark_csv_output: Option<PathBuf>,
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
        assert_eq!(config.benchmark_mode, false); // Default
        assert_eq!(config.report_progress_interval, None); // Default
        assert_eq!(config.visualization_mode, VisualizationMode::None); // Default
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

    #[test]
    fn test_visualization_toggle_key() {
        // Test default value
        let args = vec!["wave-forge", "--rule-file", "r.ron"];
        let config = AppConfig::try_parse_from(args).unwrap();
        assert_eq!(config.visualization_toggle_key, 'T');

        // Test custom value
        let args = vec![
            "wave-forge",
            "--rule-file",
            "r.ron",
            "--visualization-toggle-key",
            "V",
        ];
        let config = AppConfig::try_parse_from(args).unwrap();
        assert_eq!(config.visualization_toggle_key, 'V');
    }

    #[test]
    fn test_progress_log_level() {
        // Test default value
        let args = vec!["wave-forge", "--rule-file", "r.ron"];
        let config = AppConfig::try_parse_from(args).unwrap();
        assert_eq!(config.progress_log_level, ProgressLogLevel::Info); // Default should be Info

        // Test each value
        for (level_str, expected_level) in &[
            ("trace", ProgressLogLevel::Trace),
            ("debug", ProgressLogLevel::Debug),
            ("info", ProgressLogLevel::Info),
            ("warn", ProgressLogLevel::Warn),
        ] {
            let args = vec![
                "wave-forge",
                "--rule-file",
                "r.ron",
                "--progress-log-level",
                level_str,
            ];
            let config = AppConfig::try_parse_from(args).unwrap();
            assert_eq!(config.progress_log_level, *expected_level);
        }

        // Test invalid value should error
        let args = vec![
            "wave-forge",
            "--rule-file",
            "r.ron",
            "--progress-log-level",
            "invalid-level",
        ];
        assert!(AppConfig::try_parse_from(args).is_err());
    }

    #[test]
    fn test_global_log_level() {
        // Test default value
        let args = vec!["wave-forge", "--rule-file", "r.ron"];
        let config = AppConfig::try_parse_from(args).unwrap();
        assert_eq!(config.global_log_level, GlobalLogLevel::Info); // Default should be Info

        // Test each value
        for (level_str, expected_level) in &[
            ("trace", GlobalLogLevel::Trace),
            ("debug", GlobalLogLevel::Debug),
            ("info", GlobalLogLevel::Info),
            ("warn", GlobalLogLevel::Warn),
            ("error", GlobalLogLevel::Error),
        ] {
            let args = vec![
                "wave-forge",
                "--rule-file",
                "r.ron",
                "--global-log-level",
                level_str,
            ];
            let config = AppConfig::try_parse_from(args).unwrap();
            assert_eq!(config.global_log_level, *expected_level);
        }

        // Test invalid value should error
        let args = vec![
            "wave-forge",
            "--rule-file",
            "r.ron",
            "--global-log-level",
            "invalid-level",
        ];
        assert!(AppConfig::try_parse_from(args).is_err());
    }
}
