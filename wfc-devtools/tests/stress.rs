//! Opt-in stress and profiling runs with large grids.
//!
//! Every test is `#[ignore]`d, so `cargo test` stays fast. Run them in release mode, one at a time
//! so they do not compete for the GPU:
//!
//! ```text
//! cargo test -p wfc-devtools --release --test stress -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Each run prints one `stress:` line with its size and timings, so results can be compared across
//! changes. See docs/testing.md.

mod common;

use wfc_core::BoundaryCondition;
use wfc_core::grid::PossibilityGrid;
use wfc_devtools::city;
use wfc_devtools::render;
use wfc_devtools::{TileGrid, adjacency_violations};
use wfc_rules::AdjacencyRules;

/// Restarts allowed per run; the solver has no backtracking yet (docs/status.md A-9).
const ATTEMPTS: usize = 20;

fn report(name: &str, grid: &PossibilityGrid, tiles: usize, solved: &common::Solved) {
    let cells = grid.width * grid.height * grid.depth;
    eprintln!(
        "stress: {name} {}x{}x{} cells={cells} tiles={tiles} attempts={} run_s={:.3} total_s={:.3} cells_per_s={:.0}",
        grid.width,
        grid.height,
        grid.depth,
        solved.attempts,
        solved.solve_time.as_secs_f64(),
        solved.total_time.as_secs_f64(),
        cells as f64 / solved.solve_time.as_secs_f64()
    );
}

async fn city_run(name: &str, width: usize, height: usize, depth: usize) {
    let _trace = common::trace_to_chrome();
    let city = city::city();
    let m = &city.modules;
    let mut initial = PossibilityGrid::new(width, height, depth, m.variants.len());
    city::constrain_city(&mut initial, &city);
    let solved = common::solve_rules(
        &initial,
        &m.rules,
        Some(&m.tileset.weights),
        None,
        BoundaryCondition::Finite,
        ATTEMPTS,
    )
    .await;
    report(name, &initial, m.variants.len(), &solved);

    let grid =
        TileGrid::from_possibilities(&solved.grid).expect("every cell collapsed to one tile");
    let violations = adjacency_violations(&grid, &m.rules, BoundaryCondition::Finite);
    assert!(
        violations.is_empty(),
        "{} adjacency violations, first: {:?}",
        violations.len(),
        violations.first()
    );
    eprintln!(
        "stress: {name} disconnected_walkable_cells={}",
        city::disconnected_walkable_cells(&grid, &city).len()
    );
    let path = common::artifact_dir().join(format!("stress_{name}.png"));
    render::render_voxel_isometric(&grid, &city.voxels, 2)
        .save(&path)
        .expect("write PNG");
    eprintln!("rendered {}", path.display());
}

/// Two tiles that may touch anything: no contradictions, so this measures raw per-collapse
/// overhead (entropy, selection, readback) with the cheapest possible propagation.
async fn permissive_run(name: &str, size: usize) {
    let _trace = common::trace_to_chrome();
    let num_tiles = 2;
    let tuples: Vec<(usize, usize, usize)> = (0..6)
        .flat_map(|axis| {
            (0..num_tiles).flat_map(move |a| (0..num_tiles).map(move |b| (axis, a, b)))
        })
        .collect();
    let rules = AdjacencyRules::from_allowed_tuples(num_tiles, 6, tuples);
    let initial = PossibilityGrid::new(size, size, size, num_tiles);
    let solved =
        common::solve_rules(&initial, &rules, None, None, BoundaryCondition::Finite, 1).await;
    report(name, &initial, num_tiles, &solved);
    assert_eq!(solved.grid.is_fully_collapsed(), Ok(true));
}

#[tokio::test]
#[ignore = "stress test; run with --ignored in release mode"]
async fn stress_city_medium_24x24x8() {
    city_run("city_medium", 24, 24, 8).await;
}

#[tokio::test]
#[ignore = "stress test; run with --ignored in release mode"]
async fn stress_city_large_48x48x10() {
    city_run("city_large", 48, 48, 10).await;
}

#[tokio::test]
#[ignore = "stress test; run with --ignored in release mode"]
async fn stress_city_huge_96x96x12() {
    city_run("city_huge", 96, 96, 12).await;
}

#[tokio::test]
#[ignore = "stress test; run with --ignored in release mode"]
async fn stress_permissive_24_cubed() {
    permissive_run("permissive_24", 24).await;
}
