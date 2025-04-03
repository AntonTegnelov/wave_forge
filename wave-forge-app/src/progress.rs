// wave-forge-app/src/progress.rs

use anyhow::Result;
use log;
use std::time::{Duration, Instant};
use wfc_core::ProgressInfo;
use wfc_core::WfcError;

/// Configuration for progress reporter behavior.
pub struct ReporterConfig {
    /// Minimum time that must elapse between progress reports.
    pub report_interval: Duration,
}

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
    fn fail(&mut self, error: &WfcError) -> Result<()>;
}

/// A `ProgressReporter` that outputs status updates to the console.
pub struct ConsoleProgressReporter {
    start_time: Instant,
    last_report_time: Instant,
    report_interval: Duration,
}

impl ConsoleProgressReporter {
    /// Creates a new `ConsoleProgressReporter`.
    ///
    /// # Arguments
    ///
    /// * `report_interval` - The minimum time that must elapse between progress reports.
    pub fn new(report_interval: Duration) -> Self {
        let now = Instant::now();
        Self {
            start_time: now,
            last_report_time: now,
            report_interval,
        }
    }

    // Helper to format duration nicely (optional, could be simpler)
    fn format_duration(duration: Duration) -> String {
        let secs = duration.as_secs();
        let millis = duration.subsec_millis();
        format!("{}.{:03}s", secs, millis)
    }
}

impl ProgressReporter for ConsoleProgressReporter {
    fn report(&mut self, info: &ProgressInfo) -> Result<()> {
        let now = Instant::now();
        if now.duration_since(self.last_report_time) >= self.report_interval {
            let elapsed = self.start_time.elapsed();
            let percentage = if info.total_cells > 0 {
                (info.collapsed_cells as f32 / info.total_cells as f32) * 100.0
            } else {
                100.0
            };

            // Simplified ETA calculation or remove
            let eta_str = if info.collapsed_cells > 0 && percentage < 100.0 {
                let time_per_cell = elapsed.as_secs_f64() / info.collapsed_cells as f64;
                let remaining_cells = info.total_cells - info.collapsed_cells;
                let eta_secs = time_per_cell * remaining_cells as f64;
                format!(
                    " | ETA: {}",
                    Self::format_duration(Duration::from_secs_f64(eta_secs))
                )
            } else {
                " | ETA: N/A".to_string()
            };

            log::info!(
                "Progress: Iter: {} | Collapsed: {}/{} ({:.1}%) | Elapsed: {}{}",
                info.iterations,
                info.collapsed_cells,
                info.total_cells,
                percentage,
                Self::format_duration(elapsed),
                eta_str
            );

            self.last_report_time = now;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        let total_time = self.start_time.elapsed();
        log::info!(
            "WFC finished successfully. Total time: {}",
            Self::format_duration(total_time)
        );
        Ok(())
    }

    fn fail(&mut self, error: &WfcError) -> Result<()> {
        let total_time = self.start_time.elapsed();
        log::error!(
            "WFC failed: {}. Total time: {}",
            error,
            Self::format_duration(total_time)
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{thread, time::Duration};

    #[test]
    fn test_console_reporter_basic_flow() {
        let mut reporter = ConsoleProgressReporter::new(Duration::from_millis(50));
        let info1 = ProgressInfo {
            iterations: 10,
            collapsed_cells: 50,
            total_cells: 1000,
            elapsed_time: Duration::from_secs(1),
            grid_state: wfc_core::grid::PossibilityGrid::new(1, 1, 1, 1),
        };
        let res1 = reporter.report(&info1);
        assert!(res1.is_ok());

        // Second report immediately after should be throttled
        let info2 = ProgressInfo {
            iterations: 11,
            collapsed_cells: 55,
            total_cells: 1000,
            elapsed_time: Duration::from_secs(1),
            grid_state: wfc_core::grid::PossibilityGrid::new(1, 1, 1, 1),
        };
        let res2 = reporter.report(&info2);
        assert!(res2.is_ok());

        // Wait longer than interval
        thread::sleep(Duration::from_millis(60));

        let info3 = ProgressInfo {
            iterations: 20,
            collapsed_cells: 100,
            total_cells: 1000,
            elapsed_time: Duration::from_secs(2),
            grid_state: wfc_core::grid::PossibilityGrid::new(1, 1, 1, 1),
        };
        let res3 = reporter.report(&info3);
        assert!(res3.is_ok());
    }

    #[test]
    fn test_console_reporter_zero_cells() {
        let mut reporter = ConsoleProgressReporter::new(Duration::from_millis(50));
        let info = ProgressInfo {
            iterations: 0,
            collapsed_cells: 0,
            total_cells: 0,
            elapsed_time: Duration::from_secs(0),
            grid_state: wfc_core::grid::PossibilityGrid::new(1, 1, 1, 1),
        };
        let res = reporter.report(&info);
        assert!(res.is_ok());
    }
}
