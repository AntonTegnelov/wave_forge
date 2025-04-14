//! # Wave Forge Application (Binary)
//!
//! Main executable entry point.

// Re-export the library's main function
pub fn main() -> anyhow::Result<()> {
    // The tokio runtime is already set up inside the async main function
    tokio::runtime::Runtime::new()
        .expect("Failed to create Tokio runtime")
        .block_on(wave_forge_app::main::main())
}
