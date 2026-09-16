//! Shared helpers for the end-to-end and stress tests.

#![allow(dead_code)] // Each test binary uses a different subset.

use std::path::PathBuf;
use std::time::{Duration, Instant};
use wfc_core::BoundaryCondition;
use wfc_core::entropy::EntropyHeuristicType;
use wfc_core::grid::PossibilityGrid;
use wfc_devtools::fixtures::Fixture;
use wfc_gpu::gpu::accelerator::GpuAccelerator;
use std::sync::Arc;
use wfc_core::constraint::GlobalConstraint;
use wfc_rules::AdjacencyRules;

/// Directory for images produced by E2E tests, so a failing (or passing) run can be inspected.
/// Defaults to Cargo's per-target temporary directory; override with `WFC_ARTIFACT_DIR`.
pub fn artifact_dir() -> PathBuf {
    let dir = std::env::var_os("WFC_ARTIFACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("e2e-artifacts"));
    std::fs::create_dir_all(&dir).expect("create artifact directory");
    dir
}

/// Writes a Chrome/Perfetto trace of the solver's spans when `WFC_TRACE_CHROME` names a file, so a
/// stress run can be profiled without a separate binary (see docs/performance.md).
///
/// Keep the returned guard alive for the whole test: dropping it flushes the trace.
#[must_use]
pub fn trace_to_chrome() -> Option<tracing_chrome::FlushGuard> {
    use tracing_subscriber::layer::SubscriberExt;
    let path = std::env::var_os("WFC_TRACE_CHROME")?;
    let (layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
        .file(path)
        .include_args(true)
        .build();
    // `.init()` panics when a logger is already installed, so set the default explicitly.
    tracing::subscriber::set_global_default(tracing_subscriber::registry().with(layer)).ok()?;
    Some(guard)
}

/// A successful run and what it took.
pub struct Solved {
    pub grid: PossibilityGrid,
    /// Runs started, including the successful one.
    pub attempts: usize,
    /// Time spent in the successful run, excluding accelerator setup.
    pub solve_time: Duration,
    /// Time from the first attempt until success, including setup and failed attempts.
    pub total_time: Duration,
}

/// Solves `initial` on the GPU with a fixture's rules; see [`solve_rules`].
pub async fn solve(
    initial: &PossibilityGrid,
    fixture: &Fixture,
    boundary: BoundaryCondition,
    attempts: usize,
) -> PossibilityGrid {
    solve_rules(initial, &fixture.rules, None, None, boundary, attempts).await.grid
}

/// Solves `initial` on the GPU, starting over from scratch after a contradiction.
///
/// Retrying is a stopgap: the solver has neither backtracking nor seeded restarts yet
/// (docs/status.md A-9) and ignores seeds (A-6), so an unlucky run can only be repeated.
/// Any other error fails the test immediately.
pub async fn solve_rules(
    initial: &PossibilityGrid,
    rules: &AdjacencyRules,
    weights: Option<&[f32]>,
    constraint: Option<Arc<dyn GlobalConstraint>>,
    boundary: BoundaryCondition,
    attempts: usize,
) -> Solved {
    // Backtracking redoes collapses it undid, so a constrained run needs a far larger budget than
    // one iteration per cell.
    let cells = initial.width * initial.height * initial.depth;
    let mut max_iterations = (cells * if constraint.is_some() { 50 } else { 2 }) as u64;
    // WFC_SWEEP=1 makes a configuration that thrashes report quickly instead of retrying for hours.
    let sweeping = std::env::var("WFC_SWEEP").is_ok();
    let attempts = if sweeping { 1 } else { attempts };
    if sweeping {
        max_iterations = (cells * 4) as u64;
    }
    let started = Instant::now();
    let mut last_error = String::new();
    for attempt in 1..=attempts {
        let mut accelerator =
            GpuAccelerator::new(initial, rules, boundary, EntropyHeuristicType::Count, None)
                .await
                .expect("GPU accelerator initialises");
        if let Some(weights) = weights {
            accelerator.with_tile_weights(weights).expect("valid tile weights");
        }
        if let Some(constraint) = &constraint {
            accelerator.with_global_constraint(Arc::clone(constraint));
        }
        // Collapse several cells per propagation round when asked; see
        // GpuAccelerator::with_collapse_batch and docs/solver-fit.md.
        if let Some(batch) = std::env::var("WFC_COLLAPSE_BATCH").ok().and_then(|v| v.parse().ok()) {
            accelerator.with_collapse_batch(batch);
        }
        // A fixed seed makes a difference between configurations a real difference rather than luck.
        if let Some(seed) = std::env::var("WFC_SEED").ok().and_then(|v| v.parse().ok()) {
            accelerator.with_seed(seed);
        }
        let run_started = Instant::now();
        match accelerator
            .run_with_callback(initial, rules, max_iterations, |_| Ok(true), None)
            .await
        {
            Ok(grid) => {
                return Solved { grid, attempts: attempt, solve_time: run_started.elapsed(), total_time: started.elapsed() };
            }
            Err(error) if error.to_string().contains("Contradiction") => {
                eprintln!("attempt {attempt}/{attempts} hit a contradiction after {:?}: {error}", run_started.elapsed());
                last_error = error.to_string();
            }
            Err(error) => panic!("WFC run failed: {error}"),
        }
    }
    panic!("no solution after {attempts} attempts; last error: {last_error}");
}
