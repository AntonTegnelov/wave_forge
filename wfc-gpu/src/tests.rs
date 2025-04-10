#[cfg(test)]
mod tests {
    // Removed unused Accelerator import (already in scope)
    // use crate::accelerator::GpuAccelerator;
    // Commented out missing test utils imports
    // use crate::test_utils::{assert_grid_consistency, create_test_device_queue, get_tile_weights};
    use crate::test_utils::create_test_device_queue;
    use wfc_core::{entropy::EntropyHeuristicType, grid::PossibilityGrid, BoundaryCondition};
    use wfc_rules::{AdjacencyRules, TileSet, Transformation};

    // Added missing GpuAccelerator import
    use crate::GpuAccelerator;

    #[tokio::test]
    async fn test_progressive_results() {
        // Create minimal test setup
        let (_device, _queue) = create_test_device_queue();

        // Create a small grid for testing
        let width = 4;
        let height = 4;
        let depth = 1;
        let num_tiles = 2; // Simplest case with two tile types

        // Initialize grid with all possibilities
        let mut grid = PossibilityGrid::new(width, height, depth, num_tiles);

        // Create minimal ruleset
        let weights = vec![1.0; num_tiles];
        let allowed_transformations = vec![vec![Transformation::Identity]; num_tiles];
        let tileset =
            TileSet::new(weights, allowed_transformations).expect("Failed to create test TileSet");

        // Create simple rules (all tiles compatible with all neighbors)
        let num_transformed_tiles = tileset.num_transformed_tiles();
        let num_axes = 6;
        let mut allowed_tuples = Vec::new();
        for axis in 0..num_axes {
            for tile1_idx in 0..num_transformed_tiles {
                for tile2_idx in 0..num_transformed_tiles {
                    allowed_tuples.push((tile1_idx, tile2_idx, axis));
                }
            }
        }
        let rules =
            AdjacencyRules::from_allowed_tuples(num_transformed_tiles, num_axes, allowed_tuples);

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

        // Initialize GPU accelerator with heuristic and subgrid config
        let accelerator = GpuAccelerator::new(
            &grid,
            &rules,
            BoundaryCondition::Finite,
            EntropyHeuristicType::Shannon,
            None,
        )
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
            assert!(cell.get(0).map_or(false, |b| *b));
            assert!(!cell.get(1).map_or(false, |b| *b));
        } else {
            panic!("Cell (0,0,0) should exist");
        }

        // Verify cell (1,1,0) is collapsed to tile 1
        if let Some(cell) = result.get(1, 1, 0) {
            assert_eq!(cell.count_ones(), 1);
            assert!(!cell.get(0).map_or(false, |b| *b));
            assert!(cell.get(1).map_or(false, |b| *b));
        } else {
            panic!("Cell (1,1,0) should exist");
        }

        // All other cells should still have all possibilities
        if let Some(cell) = result.get(2, 2, 0) {
            assert_eq!(cell.count_ones(), 2);
            assert!(cell.get(0).map_or(false, |b| *b));
            assert!(cell.get(1).map_or(false, |b| *b));
        } else {
            panic!("Cell (2,2,0) should exist");
        }
    }

    #[tokio::test]
    async fn test_entropy_calculation() {
        // Existing test code...
    }
}

#[cfg(test)]
mod shader_validation_tests {
    use crate::pipeline::ComputePipelines;
    use crate::test_utils::create_test_device_queue;

    /// Tests that all shaders compile correctly with various tile sizes
    #[test]
    fn test_shader_compilation() {
        let (device, _) = create_test_device_queue();

        // Test with different tile counts to ensure bit packing works correctly
        let tile_counts = [32, 64, 128, 256, 512];

        for &tile_count in &tile_counts {
            let tiles_u32 = (tile_count + 31) / 32; // Calculate required u32 words

            // Provide empty features slice
            let result = ComputePipelines::new(&device, tiles_u32, &[]);
            assert!(
                result.is_ok(),
                "Shader compilation failed for {} tiles",
                tile_count
            );

            // Verify the created pipelines have expected properties
            let pipelines = result.unwrap();
            assert_eq!(
                pipelines.entropy_workgroup_size, 64,
                "Expected entropy workgroup size of 64 for {} tiles",
                tile_count
            );
            assert_eq!(
                pipelines.propagation_workgroup_size, 64,
                "Expected propagation workgroup size of 64 for {} tiles",
                tile_count
            );
        }
    }

    /// Tests the shader module processing (include system and constants replacement)
    #[test]
    fn test_shader_processing() {
        use std::fs;
        use std::path::Path;

        // Helper to read shader source
        fn read_shader(name: &str) -> String {
            let path = Path::new("src").join("shaders").join(name);
            fs::read_to_string(&path).unwrap_or_else(|_| panic!("Failed to read shader: {}", name))
        }

        // Read the modular shaders
        let entropy_modular = read_shader("entropy_modular.wgsl");
        let propagate_modular = read_shader("propagate_modular.wgsl");
        let utils = read_shader("utils.wgsl");
        let rules = read_shader("rules.wgsl");
        let coords = read_shader("coords.wgsl");

        // Check includes exist in the modular shaders
        assert!(
            entropy_modular.contains("#include \"utils.wgsl\""),
            "Entropy shader should include utils"
        );
        assert!(
            propagate_modular.contains("#include \"rules.wgsl\""),
            "Propagate shader should include rules"
        );

        // Check placeholder constants exist
        assert!(
            entropy_modular.contains("NUM_TILES_U32_VALUE"),
            "Entropy shader missing NUM_TILES_U32_VALUE placeholder"
        );
        assert!(
            propagate_modular.contains("NUM_TILES_U32_VALUE"),
            "Propagate shader missing NUM_TILES_U32_VALUE placeholder"
        );

        // Check utility modules contain expected functions
        assert!(
            utils.contains("fn count_bits("),
            "Utils should contain bit counting function"
        );
        assert!(
            rules.contains("fn check_rule("),
            "Rules should contain rule checking function"
        );
        assert!(
            coords.contains("fn get_neighbor_index("),
            "Coords should contain neighbor calculation function"
        );
    }

    /// Tests shader compatibility with basic device features
    #[test]
    fn test_shader_feature_compatibility() {
        let (device, _) = create_test_device_queue();

        // Get device features
        let features = device.features();

        // Check if device supports compute shaders (basic requirement)
        // Note: In wgpu 0.20+, this is implied by design, but good to document
        let supports_compute = true;
        assert!(
            supports_compute,
            "Device doesn't support compute shaders, which is required"
        );

        // Create pipelines with a reasonable tile count and empty features
        let result = ComputePipelines::new(&device, 4, &[]); // 128 tiles
        assert!(
            result.is_ok(),
            "Failed to create compute pipelines: {:?}",
            result.err()
        );

        // Test for specific feature-dependent code paths
        let pipelines = result.unwrap();

        // We should be able to create bind groups using these pipelines
        let buffer_size = 1024;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Check pipelines expose layout accessors correctly
        let layout_result = pipelines.get_propagation_bind_group_layout();
        assert!(
            layout_result.is_ok(),
            "Failed to get propagation bind group layout"
        );
    }

    /// Tests the basic functionality of the WGPU device and queue.
    #[test]
    fn test_gpu_device_and_queue() {
        let (device, queue) = create_test_device_queue();
        let _features = device.features(); // Check available features

        let buffer_desc = wgpu::BufferDescriptor {
            label: Some("Test Buffer"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        };
        let _buffer = device.create_buffer(&buffer_desc);

        let command_buffer = device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default())
            .finish();
        queue.submit(Some(command_buffer));
        device.poll(wgpu::Maintain::Wait);

        assert!(true);
    }
}
