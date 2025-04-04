use pollster::block_on;
use wfc_core::{grid::PossibilityGrid, BoundaryCondition};
use wfc_gpu::accelerator::GpuAccelerator;
use wfc_rules::AdjacencyRules;

/// Test that we can successfully create a WFC GPU accelerator with more than 128 tiles.
/// This test verifies that our changes to support larger tile sets are working.
#[test]
fn test_large_tileset() {
    // Create a rule set with 200 tiles (more than the previous limit of 128)
    let num_tiles = 200;
    let num_axes = 6; // Standard 3D grid (±X, ±Y, ±Z)

    // Create a collection of allowed tuples (tile1, tile2, axis)
    let mut allowed_tuples = Vec::new();

    // Add some basic rules to have a valid ruleset
    for i in 0..num_tiles {
        // Simple rule: tile i can be adjacent to tile i+1 in the positive x direction
        let neighbor = (i + 1) % num_tiles;
        allowed_tuples.push((i, neighbor, 0));
    }

    // Create rules from the tuples
    let rules = AdjacencyRules::from_allowed_tuples(num_tiles, num_axes, allowed_tuples);

    // Assert we actually have more than 128 tiles
    assert!(
        rules.num_tiles() > 128,
        "Test should use more than 128 tiles"
    );

    // Create a small grid for testing
    let grid = PossibilityGrid::new(16, 16, 1, num_tiles);

    // Try creating a GPU accelerator with the large tile set
    // This will fail if our changes to support larger tile sets aren't working
    let result = block_on(GpuAccelerator::new(
        &grid,
        &rules,
        BoundaryCondition::Finite,
    ));

    // Check that we can successfully create the accelerator
    assert!(
        result.is_ok(),
        "Failed to create GPU accelerator with large tile set: {:?}",
        result.err()
    );

    // Clean up resources
    let _gpu = result.unwrap();
    // GPU accelerator will be dropped here when it goes out of scope
}
