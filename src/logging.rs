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

/// Records every `tracing` span to a Chrome trace file for timeline viewers such as Perfetto.
///
/// Spans are recorded from creation to close (async style) rather than per enter/exit, so stages
/// that span `.await` points and parent spans that are never entered (the run and each
/// iteration) still appear with their real durations.
pub fn init_chrome_trace(path: &std::path::Path) -> tracing_chrome::FlushGuard {
    use tracing_subscriber::prelude::*;

    let (layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
        .file(path)
        .include_args(true)
        .trace_style(tracing_chrome::TraceStyle::Async)
        .build();
    // `set_global_default` instead of `.init()`: `.init()` also installs a `log` bridge, which
    // fails because env_logger already owns the global logger.
    match tracing::subscriber::set_global_default(tracing_subscriber::registry().with(layer)) {
        Ok(()) => log::info!("Writing trace timeline to {}", path.display()),
        Err(error) => {
            log::warn!("Trace timeline disabled, a tracing subscriber is already set: {error}")
        }
    }
    guard
}
