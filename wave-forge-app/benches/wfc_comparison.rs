use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::path::PathBuf;
use std::sync::Arc;
use wfc_core::{
    accelerator::Accelerator, entropy::SimpleEntropyCalculator, grid::PossibilityGrid,
    propagator::SimpleConstraintPropagator, runner::WfcRunner,
};
use wfc_gpu::accelerator::GpuAccelerator;
use wfc_rules::{load_rule_file, load_tileset_file, AdjacencyRules, TileSet};

// Placeholder: Define setup function to load rules/tileset and create initial grid
fn setup_wfc_data(
    width: usize,
    height: usize,
    depth: usize,
) -> (TileSet, AdjacencyRules, PossibilityGrid) {
    // TODO: Replace with actual loading logic, maybe from a test resource
    // For now, create minimal dummy data
    let num_tiles = 4;
    let tileset = TileSet {
        weights: vec![1.0; num_tiles],
        // other fields if needed
    };
    let rules = AdjacencyRules::new_uniform(num_tiles, 6); // Example: 6 axes, all allowed
    let grid = PossibilityGrid::new(width, height, depth, num_tiles);
    (tileset, rules, grid)
}

// Benchmark function for CPU
fn bench_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("WFC CPU vs GPU");

    // Example sizes
    for size in [(8, 8, 1), (16, 16, 1)].iter() {
        let (width, height, depth) = *size;
        let (tileset, rules, initial_grid) = setup_wfc_data(width, height, depth);
        let num_cells = (width * height * depth) as u64;

        group.throughput(Throughput::Elements(num_cells)); // Measure throughput by cells processed

        group.bench_with_input(
            BenchmarkId::new("CPU", format!("{}x{}x{}", width, height, depth)),
            size,
            |b, _| {
                // Clone necessary data for each iteration
                let mut grid_clone = initial_grid.clone();
                let rules_clone = rules.clone();
                let tileset_clone = tileset.clone();

                b.iter(|| {
                    let mut runner = WfcRunner::new(
                        rules_clone.clone(),
                        tileset_clone.clone(),
                        Box::new(SimpleEntropyCalculator::new()),
                        Box::new(SimpleConstraintPropagator::new()),
                        None, // No progress callback
                        None, // Use default random seed
                    );
                    // Use black_box to prevent optimization
                    let result = runner.run(&mut grid_clone);
                    black_box(result);
                });
            },
        );
    }
    group.finish();
}

// Benchmark function for GPU (requires async setup)
fn bench_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("WFC CPU vs GPU");

    // GPU setup (needs to happen outside the iteration loop)
    let instance = wgpu::Instance::default();
    let adapter_result =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()));
    let adapter = match adapter_result {
        Some(a) => a,
        None => {
            eprintln!("Failed to find suitable GPU adapter for benchmarking.");
            return; // Skip GPU benchmarks if no adapter found
        }
    };
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Bench Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        },
        None,
    ))
    .expect("Failed to request device for benchmarking");

    let arc_device = Arc::new(device);
    let arc_queue = Arc::new(queue);

    // Example sizes
    for size in [(8, 8, 1), (16, 16, 1)].iter() {
        let (width, height, depth) = *size;
        let (tileset, rules, initial_grid) = setup_wfc_data(width, height, depth);
        let num_cells = (width * height * depth) as u64;

        // Setup GPU Accelerator (this might fail if resources aren't correct)
        let gpu_accelerator =
            match GpuAccelerator::new(arc_device.clone(), arc_queue.clone(), &initial_grid, &rules)
            {
                Ok(acc) => acc,
                Err(e) => {
                    eprintln!(
                        "Failed to create GPU accelerator for size {}x{}x{}: {}",
                        width, height, depth, e
                    );
                    continue; // Skip this size if setup fails
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
                // Accelerator should be clonable if its internal Arcs are set up correctly
                let accelerator_clone = gpu_accelerator.clone();

                b.iter(|| {
                    let mut runner = WfcRunner::new(
                        rules_clone.clone(),
                        tileset_clone.clone(),
                        Box::new(accelerator_clone.entropy_calculator()),
                        Box::new(accelerator_clone.constraint_propagator()),
                        None, // No progress callback
                        None, // Use default random seed
                    );
                    // Use black_box to prevent optimization
                    let result = runner.run(&mut grid_clone);
                    black_box(result);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_cpu, bench_gpu);
criterion_main!(benches);
