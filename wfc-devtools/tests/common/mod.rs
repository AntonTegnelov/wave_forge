//! Shared helpers for the end-to-end tests.

use std::path::PathBuf;
use wfc_core::BoundaryCondition;
use wfc_core::entropy::EntropyHeuristicType;
use wfc_core::grid::PossibilityGrid;
use wfc_devtools::fixtures::Fixture;
use wfc_gpu::gpu::accelerator::GpuAccelerator;

/// Directory for images produced by E2E tests, so a failing (or passing) run can be inspected.
/// Defaults to Cargo's per-target temporary directory; override with `WFC_ARTIFACT_DIR`.
pub fn artifact_dir() -> PathBuf {
    let dir = std::env::var_os("WFC_ARTIFACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("e2e-artifacts"));
    std::fs::create_dir_all(&dir).expect("create artifact directory");
    dir
}

/// Solves `initial` on the GPU, starting over from scratch after a contradiction.
///
/// Retrying is a stopgap: the solver has neither backtracking nor seeded restarts yet
/// (docs/status.md A-9) and ignores seeds (A-6), so an unlucky run can only be repeated.
/// Any other error fails the test immediately.
pub async fn solve(
    initial: &PossibilityGrid,
    fixture: &Fixture,
    boundary: BoundaryCondition,
    attempts: usize,
) -> PossibilityGrid {
    let max_iterations = (initial.width * initial.height * initial.depth * 2) as u64;
    let mut last_error = String::new();
    for attempt in 1..=attempts {
        let mut accelerator = GpuAccelerator::new(
            initial,
            &fixture.rules,
            boundary,
            EntropyHeuristicType::Count,
            None,
        )
        .await
        .expect("GPU accelerator initialises");
        match accelerator
            .run_with_callback(initial, &fixture.rules, max_iterations, |_| Ok(true), None)
            .await
        {
            Ok(grid) => return grid,
            Err(error) if error.to_string().contains("Contradiction") => {
                eprintln!("attempt {attempt}/{attempts} hit a contradiction: {error}");
                last_error = error.to_string();
            }
            Err(error) => panic!("WFC run failed: {error}"),
        }
    }
    panic!("no solution after {attempts} attempts; last error: {last_error}");
}
