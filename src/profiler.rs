//! Profiling utilities for identifying performance bottlenecks in the WFC algorithm.
//!
//! This module provides instrumentation for tracking execution time of different parts
//! of the WFC algorithm and identifying hotspots.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// A simple profiler for tracking execution time of different sections of code.
#[derive(Debug, Clone)]
pub struct Profiler {
    /// Name of the profiler instance (e.g. "GPU")
    name: String,
    /// Metrics collected by the profiler, including time spent in different code sections
    metrics: Arc<Mutex<HashMap<String, ProfileMetric>>>,
}

/// Metrics tracked for a specific section of code
#[derive(Debug, Clone)]
pub struct ProfileMetric {
    /// Number of times the section was executed
    pub calls: usize,
    /// Total time spent in the section
    pub total_time: Duration,
    /// Minimum time spent in a single call
    pub min_time: Duration,
    /// Maximum time spent in a single call
    pub max_time: Duration,
}

/// Tracks the start time of a profiled section to allow timing when it completes
pub struct ProfilerGuard<'a> {
    profiler: &'a Profiler,
    section: String,
    start_time: Instant,
}

impl Profiler {
    /// Creates a new profiler with the given name.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            metrics: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Starts timing a section of code and returns a guard that will record the time
    /// when it's dropped.
    pub fn profile<'a>(&'a self, section: &str) -> ProfilerGuard<'a> {
        ProfilerGuard {
            profiler: self,
            section: section.to_string(),
            start_time: Instant::now(),
        }
    }

    /// Gets a clone of all metrics collected by this profiler.
    pub fn get_metrics(&self) -> HashMap<String, ProfileMetric> {
        let metrics_guard = self.metrics.lock().expect("Metrics lock poisoned");
        metrics_guard.clone()
    }

    /// Gets the profiler name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Clears all metrics collected by this profiler.
    pub fn reset(&self) {
        let mut metrics_guard = self.metrics.lock().expect("Metrics lock poisoned");
        metrics_guard.clear();
    }
}

impl Drop for ProfilerGuard<'_> {
    fn drop(&mut self) {
        let elapsed = self.start_time.elapsed();
        let mut metrics_guard = self.profiler.metrics.lock().expect("Metrics lock poisoned");

        metrics_guard
            .entry(self.section.clone())
            .and_modify(|metric| {
                metric.calls += 1;
                metric.total_time += elapsed;
                metric.min_time = std::cmp::min(metric.min_time, elapsed);
                metric.max_time = std::cmp::max(metric.max_time, elapsed);
            })
            .or_insert_with(|| ProfileMetric {
                calls: 1,
                total_time: elapsed,
                min_time: elapsed,
                max_time: elapsed,
            });
    }
}

impl ProfileMetric {
    /// Calculates the average time per call.
    pub fn average_time(&self) -> Duration {
        if self.calls == 0 {
            Duration::from_secs(0)
        } else {
            self.total_time / self.calls as u32
        }
    }
}

/// Helper function to format a duration for display
pub fn format_duration(duration: Duration) -> String {
    if duration.as_secs() > 0 {
        format!("{:.2}s", duration.as_secs_f64())
    } else if duration.as_millis() > 0 {
        format!("{:.2}ms", duration.as_millis() as f64)
    } else if duration.as_micros() > 0 {
        format!("{:.2}µs", duration.as_micros() as f64)
    } else {
        format!("{}ns", duration.as_nanos())
    }
}

/// Print a formatted summary of profiler results
pub fn print_profiler_summary(profiler: &Profiler) {
    let metrics = profiler.get_metrics();
    if metrics.is_empty() {
        println!("No profiling data collected for {}", profiler.name());
        return;
    }

    // Sort sections by total time (descending)
    let mut sections: Vec<(&String, &ProfileMetric)> = metrics.iter().collect();
    sections.sort_by(|a, b| b.1.total_time.cmp(&a.1.total_time));

    println!("=== {} Profiling Results ===", profiler.name());
    println!(
        "{:<25} | {:<10} | {:<10} | {:<10} | {:<10}",
        "Section", "Calls", "Total", "Average", "Max"
    );
    println!("{:-<79}", "");

    for (section, metric) in sections {
        println!(
            "{:<25} | {:<10} | {:<10} | {:<10} | {:<10}",
            section,
            metric.calls,
            format_duration(metric.total_time),
            format_duration(metric.average_time()),
            format_duration(metric.max_time)
        );
    }
    println!();
}

/// A global profiler for collecting performance metrics across components.
pub fn global_profiler() -> &'static Profiler {
    static INSTANCE: once_cell::sync::Lazy<Profiler> =
        once_cell::sync::Lazy::new(|| Profiler::new("Global"));
    &INSTANCE
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_profiler_basic() {
        let profiler = Profiler::new("Test");

        // Profile a section
        {
            let _guard = profiler.profile("test_section");
            thread::sleep(Duration::from_millis(5));
        }

        let metrics = profiler.get_metrics();
        assert_eq!(metrics.len(), 1);
        assert!(metrics.contains_key("test_section"));

        let section_metric = &metrics["test_section"];
        assert_eq!(section_metric.calls, 1);
        assert!(section_metric.total_time >= Duration::from_millis(5));
    }

    #[test]
    fn test_profiler_multiple_sections() {
        let profiler = Profiler::new("Test");

        // Profile first section
        {
            let _guard = profiler.profile("section_a");
            thread::sleep(Duration::from_millis(5));
        }

        // Profile second section
        {
            let _guard = profiler.profile("section_b");
            thread::sleep(Duration::from_millis(10));
        }

        // Profile first section again
        {
            let _guard = profiler.profile("section_a");
            thread::sleep(Duration::from_millis(5));
        }

        let metrics = profiler.get_metrics();
        assert_eq!(metrics.len(), 2);

        let section_a = &metrics["section_a"];
        assert_eq!(section_a.calls, 2);
        assert!(section_a.total_time >= Duration::from_millis(10));

        let section_b = &metrics["section_b"];
        assert_eq!(section_b.calls, 1);
        assert!(section_b.total_time >= Duration::from_millis(10));
    }

    #[test]
    fn test_global_profiler() {
        let profiler = global_profiler();

        // Reset to start clean
        profiler.reset();

        // Profile a section
        {
            let _guard = profiler.profile("global_test");
            thread::sleep(Duration::from_millis(1));
        }

        let metrics = profiler.get_metrics();
        assert!(metrics.contains_key("global_test"));
        assert_eq!(metrics["global_test"].calls, 1);
    }

    #[test]
    fn test_profiler_end_to_end() {
        // Create a profiler for an end-to-end test
        let profiler = Profiler::new("EndToEnd");

        // Simulate a multi-stage algorithm
        {
            // Initialization phase
            let _init_guard = profiler.profile("initialization");
            thread::sleep(Duration::from_millis(10));

            // Processing phase with nested operations
            let _process_guard = profiler.profile("processing");

            for i in 0..3 {
                let _iteration_guard = profiler.profile("iteration");

                // Two sub-tasks per iteration
                {
                    let _sub1_guard = profiler.profile("sub_task_1");
                    thread::sleep(Duration::from_millis(5));
                }

                {
                    let _sub2_guard = profiler.profile("sub_task_2");
                    thread::sleep(Duration::from_millis(i * 2 + 3)); // Variable time
                }
            }

            // Finalization phase
            let _final_guard = profiler.profile("finalization");
            thread::sleep(Duration::from_millis(5));
        }

        // Verify metrics were collected for all phases
        let metrics = profiler.get_metrics();
        assert!(metrics.contains_key("initialization"));
        assert!(metrics.contains_key("processing"));
        assert!(metrics.contains_key("iteration"));
        assert!(metrics.contains_key("sub_task_1"));
        assert!(metrics.contains_key("sub_task_2"));
        assert!(metrics.contains_key("finalization"));

        // Verify call counts
        assert_eq!(metrics["initialization"].calls, 1);
        assert_eq!(metrics["processing"].calls, 1);
        assert_eq!(metrics["iteration"].calls, 3);
        assert_eq!(metrics["sub_task_1"].calls, 3);
        assert_eq!(metrics["sub_task_2"].calls, 3);
        assert_eq!(metrics["finalization"].calls, 1);

        // Print summary for manual verification
        print_profiler_summary(&profiler);
    }
}
