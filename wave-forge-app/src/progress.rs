// wave-forge-app/src/progress.rs

use anyhow::Result;
use wfc_core::ProgressInfo; // Assuming ProgressInfo is the data payload

/// Trait for reporting the progress of the WFC algorithm.
///
/// Implementors of this trait can display progress information in various ways
/// (e.g., console output, GUI updates).
pub trait ProgressReporter: Send + Sync {
    // Send + Sync needed if used across threads
    /// Called periodically or on significant events with updated progress information.
    ///
    /// # Arguments
    ///
    /// * `info` - The latest `ProgressInfo` snapshot from the WFC runner.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if reporting was successful.
    /// * `Err(anyhow::Error)` if an error occurred during reporting.
    fn report(&mut self, info: &ProgressInfo) -> Result<()>;

    /// Called when the WFC process completes successfully.
    fn finish(&mut self) -> Result<()>;

    /// Called when the WFC process fails with an error.
    fn fail(&mut self, error: &wfc_core::WfcError) -> Result<()>;
}

// TODO: Implement concrete reporters like ConsoleProgressReporter
