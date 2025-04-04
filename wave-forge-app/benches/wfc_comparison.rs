use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use wfc_core::{
    grid::PossibilityGrid, runner::run as wfc_run, BoundaryMode, ProgressInfo, WfcCheckpoint,
    WfcError,
};
use wfc_gpu::accelerator::GpuAccelerator;
use wfc_rules::{AdjacencyRules, TileSet, TileSetError, Transformation};

// Placeholder: Define setup function to load rules/tileset and create initial grid
fn setup_wfc_data(
    width: usize,
    height: usize,
    depth: usize,
) -> Result<(TileSet, AdjacencyRules, PossibilityGrid), TileSetError> {
    // Create a simple tileset with identity transformations
    let num_base_tiles = 4;
    let weights = vec![1.0; num_base_tiles];
    let allowed_transforms = vec![vec![Transformation::Identity]; num_base_tiles];
    let tileset = TileSet::new(weights, allowed_transforms)?;

    let num_transformed_tiles = tileset.num_transformed_tiles();
    let num_axes = 6; // 3D

    // Create uniform adjacency rules (all combinations allowed)
    let mut allowed_tuples = Vec::new();
    for axis in 0..num_axes {
        for ttid1 in 0..num_transformed_tiles {
            for ttid2 in 0..num_transformed_tiles {
                allowed_tuples.push((axis, ttid1, ttid2));
            }
        }
    }
    let rules =
        AdjacencyRules::from_allowed_tuples(num_transformed_tiles, num_axes, allowed_tuples);

    let grid = PossibilityGrid::new(width, height, depth, num_transformed_tiles);
    Ok((tileset, rules, grid))
}

// Benchmark function for GPU (requires async setup)
fn bench_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("WFC GPU");
    let shutdown_signal = Arc::new(AtomicBool::new(false)); // Create shutdown signal

    // GPU setup needs to happen outside the iteration loop, but accelerator setup depends on grid size
    // We will create the accelerator inside the loop for now.

    // Example sizes
    for size in [(8, 8, 1), (16, 16, 1)].iter() {
        let (width, height, depth) = *size;
        let (tileset, rules, initial_grid) = setup_wfc_data(width, height, depth).unwrap();
        let num_cells = (width * height * depth) as u64;
        let boundary_mode = BoundaryMode::Clamped; // Use clamped for benchmark

        // Setup GPU Accelerator (inside loop as it depends on grid)
        // Use pollster::block_on for the async new function
        let gpu_accelerator_result = pollster::block_on(GpuAccelerator::new(
            &initial_grid,
            &rules,
            boundary_mode, // Pass boundary mode
        ));

        let gpu_accelerator = match gpu_accelerator_result {
            Ok(acc) => acc,
            Err(e) => {
                eprintln!(
                    "Failed to create GPU accelerator for size {}x{}x{}: {}. Skipping GPU bench.",
                    width, height, depth, e
                );
                // If accelerator fails for one size, we might want to stop the whole group
                // or just skip this specific size. Skipping size for now.
                continue;
            }
        };

        group.throughput(Throughput::Elements(num_cells)); // Measure throughput by cells processed

        group.bench_with_input(
            BenchmarkId::new("GPU", format!("{}x{}x{}", width, height, depth)),
            size,
            |b, _| {
                // Clone necessary data for each iteration
                let mut grid_clone = initial_grid.clone();
                let rules_clone = rules.clone();
                let tileset_clone = tileset.clone();
                let accelerator_clone = gpu_accelerator.clone(); // Clone the accelerator Arc

                b.iter(|| {
                    // Clone grid for each iteration inside b.iter
                    let mut iter_grid_clone = grid_clone.clone();
                    // TODO: Implement EntropyCalculator and ConstraintPropagator for GpuAccelerator
                    // For now, passing accelerator_clone - this will fail compilation until traits are implemented
                    let result = wfc_run(
                        &mut iter_grid_clone,
                        &tileset_clone,
                        &rules_clone,
                        accelerator_clone.clone(), // Pass accelerator as propagator
                        accelerator_clone.clone(), // Pass accelerator as entropy calculator
                        boundary_mode,             // Use the mode defined for this size
                        None::<Box<dyn Fn(ProgressInfo) -> Result<(), WfcError> + Send + Sync>>, // No progress callback
                        shutdown_signal.clone(), // Pass shutdown signal
                        None::<WfcCheckpoint>,   // No initial checkpoint
                        None,                    // No checkpoint interval
                        None::<PathBuf>,         // No checkpoint path
                        None,                    // No max iterations
                    );
                    // Use black_box to prevent optimization
                    black_box(result);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_gpu);
criterion_main!(benches);
