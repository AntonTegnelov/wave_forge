//! The CPU reference solver on the city: the yardstick every GPU number is printed against.
//!
//! ```text
//! cargo test -p wfc-devtools --release --test cpu_reference -- --ignored --nocapture
//! ```
//!
//! Numbers describe one build on one machine; the full-grid scan makes selection quadratic in grid
//! size, so large grids understate what a CPU can do.

use std::sync::Arc;
use std::time::Instant;
use wfc_core::reference::ReferenceSolver;
use wfc_core::{
    ChunkCoord, ChunkShape, ChunkStore, Region, RegionStatus, Ruleset, WorldExtent, region_init,
};
use wfc_devtools::city::{self, city_prior};
use wfc_devtools::{BoundaryCondition, TileGrid, adjacency_violations};

/// One grid as a single chunk of a world that holds nothing else: what the CLI generates.
fn one_chunk_world(width: u32, height: u32, depth: u32) -> (ChunkStore, Region) {
    let shape = ChunkShape {
        x: width,
        y: height,
        z: depth,
    };
    let extent = WorldExtent::new(shape)
        .with_x(0..1)
        .with_y(0..1)
        .with_z(0..1);
    let region = Region::new(ChunkCoord::new(0, 0, 0), shape.region(extent.halo(1)));
    (ChunkStore::new(extent), region)
}

#[test]
#[ignore = "benchmark; run with --ignored in release mode"]
fn cpu_reference_city() {
    let city = city::city();
    let ruleset = Arc::new(Ruleset::from_modules(&city.modules).expect("the city compiles"));
    let solver = ReferenceSolver::new(Arc::clone(&ruleset));

    for (width, height, depth) in [(8, 8, 8), (12, 12, 6), (24, 24, 8), (48, 48, 10)] {
        let (store, region) = one_chunk_world(width, height, depth);
        let init = region_init(&store, &city_prior(&city, depth), &ruleset, &region, false);
        let cells = width * height * depth;
        for seed in 1..=8 {
            let started = Instant::now();
            let (domains, status, stats) = solver.solve_region(region.shape(), &init, 1, seed);
            let seconds = started.elapsed().as_secs_f64();
            if status != RegionStatus::Solved {
                eprintln!(
                    "cpu_reference: {width}x{height}x{depth} seed={seed} {status:?} backtracks={}",
                    stats.backtracks
                );
                continue;
            }

            let grid =
                TileGrid::from_domains(&domains, width as usize, height as usize, depth as usize)
                    .expect("every cell decided");
            let violations =
                adjacency_violations(&grid, &city.modules.rules, BoundaryCondition::Finite);
            assert!(
                violations.is_empty(),
                "{} adjacency violations",
                violations.len()
            );
            eprintln!(
                "cpu_reference: {width}x{height}x{depth} seed={seed} cells={cells} collapses={} backtracks={} run_s={seconds:.4} cells_per_s={:.0}",
                stats.collapses,
                stats.backtracks,
                f64::from(cells) / seconds
            );
        }
    }
}
