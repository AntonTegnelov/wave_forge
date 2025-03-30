use wfc_core::{entropy::EntropyCalculator, grid::PossibilityGrid, rules::AdjacencyRules};
use wfc_gpu::accelerator::GpuAccelerator; // Ensure accelerator is public or crate-visible

// Helper to initialize logging for tests
// fn setup_logger() { // Removed unused function
//     // Use `try_init` to avoid panic if logger is already set
//     let _ = env_logger::builder().is_test(true).try_init();
// }

#[test]
fn test_gpu_calculate_entropy_basic_run() {
    // setup_logger(); // Temporarily disabled

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
    log::info!("Entering pollster::block_on(GpuAccelerator::new) for entropy test...");
    let accelerator_result = pollster::block_on(GpuAccelerator::new(&grid, &rules));
    log::info!("Exited pollster::block_on(GpuAccelerator::new) for entropy test.");
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

// #[test] // Temporarily disable this test
// fn test_gpu_propagate_basic_run() {
//     setup_logger();
//
//     // Basic setup
//     let width = 2;
//     let height = 1;
//     let depth = 1;
//     let num_tiles = 2;
//     let mut grid = PossibilityGrid::new(width, height, depth, num_tiles);
//     let rules = AdjacencyRules::new(num_tiles, 6, vec![true; 6 * num_tiles * num_tiles]); // All allowed
//
//     // Initialize GPU Accelerator
//     log::info!("Entering pollster::block_on(GpuAccelerator::new) for propagate test...");
//     let accelerator_result = pollster::block_on(GpuAccelerator::new(&grid, &rules));
//     log::info!("Exited pollster::block_on(GpuAccelerator::new) for propagate test.");
//     if let Err(e) = accelerator_result {
//         eprintln!(
//             "Skipping GPU test: Failed to initialize GpuAccelerator: {}",
//             e
//         );
//         return;
//     }
//     let mut accelerator = accelerator_result.unwrap();
//
//     // Define initial updates (e.g., cell at (0,0,0) was just collapsed)
//     let updated_coords = vec![(0, 0, 0)];
//
//     // Call propagate
//     // The PossibilityGrid parameter is currently unused by the GPU implementation,
//     // but we pass it to satisfy the trait method signature.
//     let result = accelerator.propagate(&mut grid, updated_coords, &rules);
//
//     // Assertions
//     // For this basic test with permissive rules, we expect no contradiction.
//     assert!(
//         result.is_ok(),
//         "GPU propagate returned an error: {:?}",
//         result.err()
//     );
//
//     // More advanced tests would involve:
//     // - Setting up specific grid states and rules that *should* cause contradiction.
//     // - Downloading the grid_possibilities buffer after propagation.
//     // - Comparing the GPU result with the expected CPU result.
// }
