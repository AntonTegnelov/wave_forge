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

// Include main.rs as a module
pub mod main;

// Optionally re-export key types if needed elsewhere
pub use config::AppConfig;
pub use error::AppError;

// Re-export the main function so it can be called from the root crate
pub use crate::main::main;
