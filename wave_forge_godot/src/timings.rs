//! The most recent timings of something the node does, for `stats`.

use std::collections::VecDeque;

/// The last `capacity` durations of one kind, in milliseconds. A game runs for hours, so the node
/// keeps a window rather than every sample.
pub(crate) struct Timings {
    samples: VecDeque<f64>,
    capacity: usize,
}

impl Timings {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Records one duration, forgetting the oldest once the window is full.
    pub(crate) fn push(&mut self, ms: f64) {
        if self.samples.len() == self.capacity {
            self.samples.pop_front();
        }
        self.samples.push_back(ms);
    }

    /// The median, 99th percentile and maximum of the window, or `None` before the first sample.
    pub(crate) fn summary(&self) -> Option<[f64; 3]> {
        if self.samples.is_empty() {
            return None;
        }
        let mut sorted: Vec<f64> = self.samples.iter().copied().collect();
        sorted.sort_by(f64::total_cmp);
        let at = |q: f64| sorted[((sorted.len() - 1) as f64 * q).round() as usize];
        Some([at(0.5), at(0.99), at(1.0)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_summary_is_the_median_99th_percentile_and_maximum() {
        let mut timings = Timings::new(1000);

        for ms in 1..=101 {
            timings.push(f64::from(ms));
        }

        assert_eq!(timings.summary(), Some([51.0, 100.0, 101.0]));
    }

    #[test]
    fn a_full_window_forgets_its_oldest_samples() {
        let mut timings = Timings::new(3);

        for ms in [50.0, 1.0, 2.0, 3.0] {
            timings.push(ms);
        }

        assert_eq!(timings.summary(), Some([2.0, 3.0, 3.0]));
    }

    #[test]
    fn nothing_recorded_has_no_summary() {
        assert_eq!(Timings::new(3).summary(), None);
    }
}
