//! Wave Forge Application Library
//!
//! This crate contains the core logic, configuration, setup,
//! and utilities for the Wave Forge application.

pub mod benchmark;
pub mod config;
pub mod error;
pub mod logging;
pub mod output;
pub mod profiler;
pub mod progress;
pub mod setup;
pub mod visualization;

// Optionally re-export key types if needed elsewhere
pub use config::AppConfig;
pub use error::AppError;

// Make main module public for external access
pub mod main;

/// Calls the main function from main.rs to allow the root crate to invoke it.
/// Returns Result<()> which should be handled by the caller.
pub fn main() -> anyhow::Result<()> {
    // We need to block on the future since the main() function is async
    tokio::runtime::Runtime::new()
        .expect("Failed to create Tokio runtime")
        .block_on(main::main())
}
