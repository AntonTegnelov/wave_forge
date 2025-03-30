use wfc_core::{entropy::EntropyCalculator, grid::PossibilityGrid, rules::AdjacencyRules};
use wfc_gpu::accelerator::GpuAccelerator; // Ensure accelerator is public or crate-visible

// Helper to initialize logging for tests
fn setup_logger() {
    // Use `try_init` to avoid panic if logger is already set
    let _ = env_logger::builder().is_test(true).try_init();
}

#[test]
fn test_gpu_calculate_entropy_basic_run() {
    setup_logger();

    // Basic setup
    let width = 4;
    let height = 4;
    let depth = 1;
    let num_tiles = 3;
    let grid = PossibilityGrid::new(width, height, depth, num_tiles);

    // Create simple rules: Allow all adjacencies in 6 directions (3 axes)
    let num_axes = 6;
    let num_rules = num_axes * num_tiles * num_tiles;
    let allowed_rules = vec![true; num_rules]; // All true
    let rules = AdjacencyRules::new(num_tiles, num_axes, allowed_rules);

    // Initialize GPU Accelerator
    let accelerator_result = pollster::block_on(GpuAccelerator::new(&grid, &rules));
    if let Err(e) = accelerator_result {
        // If GPU initialization fails (e.g., no compatible adapter), skip the test.
        // This is common in CI environments without a GPU.
        eprintln!(
            "Skipping GPU test: Failed to initialize GpuAccelerator: {}",
            e
        );
        return;
    }
    let accelerator = accelerator_result.unwrap();

    // Call calculate_entropy
    // The PossibilityGrid parameter is currently unused by the GPU implementation,
    // but we pass it to satisfy the trait method signature.
    let entropy_grid = accelerator.calculate_entropy(&grid);

    // Assertions
    assert_eq!(entropy_grid.width, width, "Entropy grid width mismatch");
    assert_eq!(entropy_grid.height, height, "Entropy grid height mismatch");
    assert_eq!(entropy_grid.depth, depth, "Entropy grid depth mismatch");
    assert_eq!(
        entropy_grid.data.len(),
        width * height * depth,
        "Entropy grid data length mismatch"
    );
    // We don't check the *values* yet, just that the process ran and returned a grid of the right size.
}
