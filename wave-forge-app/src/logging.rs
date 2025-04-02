//! Logging setup for the application.

use crate::config::{AppConfig, GlobalLogLevel, ProgressLogLevel};
use env_logger::{Builder, Env};
use log::LevelFilter;

/// Initializes the logger with the appropriate configuration based on the application settings.
///
/// This function configures the log levels for the different components of the application:
/// - The progress reports use the level specified in `config.progress_log_level`
/// - Other module levels can be controlled via the RUST_LOG environment variable
///
/// If RUST_LOG is set, it is respected and combined with the progress log level configuration.
/// If no RUST_LOG is provided, a default configuration is used.
///
/// # Arguments
///
/// * `config` - The application configuration containing the progress log level setting
pub fn init_logger(config: &AppConfig) {
    // Convert ProgressLogLevel to log::LevelFilter
    let progress_level = match config.progress_log_level {
        ProgressLogLevel::Trace => LevelFilter::Trace,
        ProgressLogLevel::Debug => LevelFilter::Debug,
        ProgressLogLevel::Info => LevelFilter::Info,
        ProgressLogLevel::Warn => LevelFilter::Warn,
    };

    // Convert GlobalLogLevel to log::LevelFilter
    let global_level = match config.global_log_level {
        GlobalLogLevel::Trace => LevelFilter::Trace,
        GlobalLogLevel::Debug => LevelFilter::Debug,
        GlobalLogLevel::Info => LevelFilter::Info,
        GlobalLogLevel::Warn => LevelFilter::Warn,
        GlobalLogLevel::Error => LevelFilter::Error,
    };

    // Start with the environment configuration, but allow for override
    let env = Env::default().filter_or("RUST_LOG", "info");

    // Create a new builder
    let mut builder = Builder::from_env(env);

    // Set the default global log level
    builder.filter_level(global_level);

    // Set the progress module's log level specifically (prioritize this over global level)
    builder.filter_module("wave_forge_app::progress", progress_level);

    // Initialize the logger
    builder.init();

    log::debug!(
        "Logger initialized with global log level: {:?}, progress log level: {:?}",
        config.global_log_level,
        config.progress_log_level
    );
}
