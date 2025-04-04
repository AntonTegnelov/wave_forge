#[cfg(test)]
mod tests {
    use crate::accelerator::GpuAccelerator;
    use crate::test_utils::create_test_device_queue;
    use wfc_core::{grid::PossibilityGrid, BoundaryCondition};
    use wfc_rules::{AdjacencyRules, TileSet};

    #[tokio::test]
    async fn test_progressive_results() {
        // Create minimal test setup
        let (device, queue) = create_test_device_queue();

        // Create a small grid for testing
        let width = 4;
        let height = 4;
        let depth = 1;
        let num_tiles = 2; // Simplest case with two tile types

        // Initialize grid with all possibilities
        let mut grid = PossibilityGrid::new(width, height, depth, num_tiles);

        // Create minimal ruleset
        let mut tileset = TileSet::default();
        tileset.weights = vec![1.0, 1.0]; // Equal weights for both tiles

        // Create simple rules (all tiles compatible with all neighbors)
        let mut rules = AdjacencyRules::new(num_tiles, 6); // 6 axes for 3D
        for i in 0..num_tiles {
            for j in 0..num_tiles {
                for axis in 0..6 {
                    rules.set_allowed(i, j, axis);
                }
            }
        }

        // Partially collapse the grid by manually setting some cells
        // Set (0,0,0) to only allow tile 0
        if let Some(cell) = grid.get_mut(0, 0, 0) {
            cell.fill(false);
            cell.set(0, true);
        }

        // Set (1,1,0) to only allow tile 1
        if let Some(cell) = grid.get_mut(1, 1, 0) {
            cell.fill(false);
            cell.set(1, true);
        }

        // Initialize GPU accelerator
        let accelerator = GpuAccelerator::new(&grid, &rules, BoundaryCondition::Finite, None)
            .await
            .unwrap();

        // Get intermediate result
        let result = accelerator.get_intermediate_result().await.unwrap();

        // Verify the result matches our expected partially collapsed grid
        assert_eq!(result.width, width);
        assert_eq!(result.height, height);
        assert_eq!(result.depth, depth);

        // Verify cell (0,0,0) is collapsed to tile 0
        if let Some(cell) = result.get(0, 0, 0) {
            assert_eq!(cell.count_ones(), 1);
            assert!(cell.get(0));
            assert!(!cell.get(1));
        } else {
            panic!("Cell (0,0,0) should exist");
        }

        // Verify cell (1,1,0) is collapsed to tile 1
        if let Some(cell) = result.get(1, 1, 0) {
            assert_eq!(cell.count_ones(), 1);
            assert!(!cell.get(0));
            assert!(cell.get(1));
        } else {
            panic!("Cell (1,1,0) should exist");
        }

        // All other cells should still have all possibilities
        if let Some(cell) = result.get(2, 2, 0) {
            assert_eq!(cell.count_ones(), 2);
            assert!(cell.get(0));
            assert!(cell.get(1));
        } else {
            panic!("Cell (2,2,0) should exist");
        }
    }
}
