//! Times the single-threaded CPU reference solver on the city, the yardstick every GPU solver is
//! printed against (docs/solver-redesign.md, "The CPU reference").
//!
//! ```text
//! cargo test -p wfc-devtools --release --test cpu_reference -- --ignored --nocapture
//! ```
//!
//! Each run prints one `cpu_reference:` line. Numbers describe one build on one machine; the full
//! grid scan makes selection quadratic in grid size, so large grids understate what a CPU can do.

use wfc_core::BoundaryCondition;
use wfc_devtools::city;
use wfc_devtools::reference::{self, ReferenceSolver};
use wfc_devtools::{TileGrid, adjacency_violations};

#[test]
#[ignore = "benchmark; run with --ignored in release mode"]
fn cpu_reference_city() {
    let city = city::city();
    let m = &city.modules;

    for (width, height, depth) in [(8, 8, 8), (12, 12, 6), (24, 24, 8), (48, 48, 10)] {
        let solver = ReferenceSolver::new(&m.rules, &m.tileset.weights, width, height, depth);
        let initial = reference::city_initial_cells(&city, width, height, depth);
        let cells = width * height * depth;
        for seed in 1..=8 {
            let outcome = solver.solve(initial.clone(), seed);
            if outcome.thrashed {
                eprintln!(
                    "cpu_reference: {width}x{height}x{depth} seed={seed} thrashed backtracks={}",
                    outcome.backtracks
                );
                continue;
            }

            let tiles = outcome
                .grid
                .iter()
                .map(|&cell| {
                    assert_eq!(
                        reference::count(cell),
                        1,
                        "every cell collapses to one tile"
                    );
                    reference::set_bits(cell).next().expect("one tile")
                })
                .collect();
            let grid = TileGrid::new(width, height, depth, tiles).expect("dimensions match");
            let violations = adjacency_violations(&grid, &m.rules, BoundaryCondition::Finite);
            assert!(
                violations.is_empty(),
                "{} adjacency violations",
                violations.len()
            );
            eprintln!(
                "cpu_reference: {width}x{height}x{depth} seed={seed} cells={cells} collapses={} backtracks={} run_s={:.4} cells_per_s={:.0}",
                outcome.collapses,
                outcome.backtracks,
                outcome.seconds,
                cells as f64 / outcome.seconds
            );
        }
    }
}
