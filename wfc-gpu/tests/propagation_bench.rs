//! Opt-in measurements of one propagation pass, to attribute its cost between GPU work and the two
//! blocking readbacks each pass performs (docs/performance.md).
//!
//! ```text
//! cargo test -p wfc-gpu --release --test propagation_bench -- --ignored --nocapture
//! ```

use std::sync::Arc;
use std::time::Instant;
use wfc_core::BoundaryCondition;
use wfc_core::grid::PossibilityGrid;
use wfc_devtools::city;
use wfc_gpu::buffers::GpuBuffers;
use wfc_gpu::gpu::backend::{GpuBackend, WgpuBackend};
use wfc_gpu::gpu::sync::GpuSynchronizer;
use wfc_gpu::propagator::AsyncPropagationStrategy;
use wfc_gpu::propagator::direct_strategy::DirectPropagationStrategy;
use wfc_gpu::shader::pipeline::ComputePipelines;
use wfc_gpu::utils::error_recovery::GridCoord;

/// Builds the GPU side for the city rule set on a `size³`-ish grid.
async fn city_setup(
    width: usize,
    height: usize,
    depth: usize,
) -> (
    Arc<GpuBuffers>,
    GpuSynchronizer,
    Arc<ComputePipelines>,
    PossibilityGrid,
) {
    let city = city::city();
    let rules = &city.modules.rules;
    let num_tiles = city.modules.variants.len();
    let mut grid = PossibilityGrid::new(width, height, depth, num_tiles);
    city::constrain_city(&mut grid, &city);

    let backend = WgpuBackend::new();
    let device = backend.device();
    let queue = backend.queue();
    let pipelines = Arc::new(
        ComputePipelines::new(&device, num_tiles.div_ceil(32) as u32, &[]).expect("pipelines"),
    );
    let buffers = Arc::new(
        GpuBuffers::new(&device, &queue, &grid, rules, BoundaryCondition::Finite).expect("buffers"),
    );
    let sync = GpuSynchronizer::new(device.clone(), queue.clone(), buffers.clone());
    sync.upload_grid(&grid).expect("upload");
    (buffers, sync, pipelines, grid)
}

#[tokio::test]
#[ignore = "benchmark; run with --ignored in release mode"]
async fn propagation_pass_cost_readback_vs_gpu() {
    let (width, height, depth) = (24, 24, 8);
    let (buffers, sync, pipelines, mut grid) = city_setup(width, height, depth).await;
    let passes = 200u32;

    // Warm up first: a cold GPU runs at low clocks, which made an earlier cold measurement look
    // several times worse than the identical pattern measured warm.
    {
        let warm =
            DirectPropagationStrategy::benchmark_blind_passes(1000, pipelines.clone(), passes);
        for _ in 0..3 {
            warm.propagate(
                &mut grid,
                &[GridCoord { x: 0, y: 0, z: 0 }],
                &buffers,
                &sync,
            )
            .await
            .expect("warm-up passes");
        }
    }

    // Same dispatches, no readbacks: what the GPU work alone costs.
    let blind = DirectPropagationStrategy::benchmark_blind_passes(1000, pipelines.clone(), passes);
    let cells = vec![GridCoord { x: 0, y: 0, z: 0 }];
    let started = Instant::now();
    blind
        .propagate(&mut grid, &cells, &buffers, &sync)
        .await
        .expect("blind passes");
    let blind_time = started.elapsed();

    // Same dispatches again, but recorded into one command buffer and submitted once.
    let batched =
        DirectPropagationStrategy::benchmark_blind_batched(1000, pipelines.clone(), passes);
    let started = Instant::now();
    batched
        .propagate(&mut grid, &cells, &buffers, &sync)
        .await
        .expect("batched passes");
    let batched_time = started.elapsed();

    // The same dispatches again, but each pass covers every cell instead of one. If this costs the
    // same as a 1-cell pass, the per-dispatch fixed cost dominates and tiny worklists waste it.
    let every_cell: Vec<GridCoord> = (0..depth)
        .flat_map(|z| (0..height).flat_map(move |y| (0..width).map(move |x| GridCoord { x, y, z })))
        .collect();
    let sweep = DirectPropagationStrategy::benchmark_blind_passes(1000, pipelines.clone(), passes);
    let started = Instant::now();
    sweep
        .propagate(&mut grid, &every_cell, &buffers, &sync)
        .await
        .expect("sweep passes");
    let sweep_time = started.elapsed();

    // Is a pass's cost the dispatch, or the per-cell work? The shader unions allowed-neighbour masks
    // by testing every tile pair, so an uncollapsed cell costs ~num_tiles^2 checks while a collapsed
    // one costs ~num_tiles. Time the same full-grid pass on a grid where every cell is collapsed.
    let mut collapsed_grid = grid.clone();
    for z in 0..collapsed_grid.depth {
        for y in 0..collapsed_grid.height {
            for x in 0..collapsed_grid.width {
                let cell = collapsed_grid.get_mut(x, y, z).expect("cell in bounds");
                let first = cell.iter_ones().next().expect("cell has a possibility");
                cell.fill(false);
                cell.set(first, true);
            }
        }
    }
    sync.upload_grid(&collapsed_grid)
        .expect("upload collapsed grid");
    let collapsed =
        DirectPropagationStrategy::benchmark_blind_passes(1000, pipelines.clone(), passes);
    let started = Instant::now();
    collapsed
        .propagate(&mut collapsed_grid, &every_cell, &buffers, &sync)
        .await
        .expect("collapsed passes");
    let collapsed_time = started.elapsed();
    sync.upload_grid(&grid).expect("restore grid");

    // The normal path: every pass reads the contradiction flag and the worklist count.
    let normal = DirectPropagationStrategy::new(1000, pipelines);
    let started = Instant::now();
    let result = normal.propagate(&mut grid, &cells, &buffers, &sync).await;
    let normal_time = started.elapsed();

    eprintln!(
        "bench: {width}x{height}x{depth} city, {passes} passes over a 1-cell worklist\n  \
         one submit per pass, no readbacks: {:?} ({:.3} ms/pass)\n  \
         one submit for all passes:        {:?} ({:.3} ms/pass)\n  \
         full-grid worklist per pass:      {:?} ({:.3} ms/pass, {} cells)\n  \
         full grid, every cell collapsed:  {:?} ({:.3} ms/pass)\n  \
         one full propagate with readbacks: {:?} ({})",
        blind_time,
        blind_time.as_secs_f64() * 1000.0 / f64::from(passes),
        batched_time,
        batched_time.as_secs_f64() * 1000.0 / f64::from(passes),
        sweep_time,
        sweep_time.as_secs_f64() * 1000.0 / f64::from(passes),
        every_cell.len(),
        collapsed_time,
        collapsed_time.as_secs_f64() * 1000.0 / f64::from(passes),
        normal_time,
        result.map(|()| "ok").unwrap_or("contradiction"),
    );
}
