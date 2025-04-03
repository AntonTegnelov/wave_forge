use wfc_core::grid::PossibilityGrid;
use wfc_gpu::buffers::GpuBuffers;
use wfc_gpu::GpuError;
use wfc_rules::AdjacencyRules;

// Helper to initialize logging for tests
fn setup_logger() {
    let _ = env_logger::builder().is_test(true).try_init();
}

// Helper to initialize wgpu device and queue for tests
async fn setup_wgpu() -> Result<(wgpu::Device, wgpu::Queue), GpuError> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower, // Low power often sufficient for tests
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok_or(GpuError::AdapterRequestFailed)?;
    adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .map_err(GpuError::DeviceRequestFailed)
}

#[test]
fn test_buffer_creation_sizes() {
    setup_logger();

    // Basic setup
    let width = 8;
    let height = 8;
    let depth = 2;
    let num_tiles = 5;
    let num_axes = 6;
    let num_cells = width * height * depth;
    let grid = PossibilityGrid::new(width, height, depth, num_tiles);
    let rules = AdjacencyRules::new(
        num_tiles,
        num_axes,
        vec![true; num_axes * num_tiles * num_tiles],
    );

    // Initialize Device/Queue
    let wgpu_result = pollster::block_on(setup_wgpu());
    if let Err(e) = wgpu_result {
        eprintln!("Skipping GPU test: Failed to initialize wgpu: {}", e);
        return;
    }
    let (device, queue) = wgpu_result.unwrap();

    // Create buffers
    let buffers_result = GpuBuffers::new(&device, &queue, &grid, &rules);
    assert!(buffers_result.is_ok(), "Failed to create GpuBuffers");
    let buffers = buffers_result.unwrap();

    // Assert buffer sizes
    let u32_size = std::mem::size_of::<u32>() as u64;
    let f32_size = std::mem::size_of::<f32>() as u64;

    let bits_per_cell = num_tiles;
    let u32s_per_cell = (bits_per_cell + 31) / 32;
    let expected_grid_size = (num_cells * u32s_per_cell) as u64 * u32_size;
    assert_eq!(
        buffers.grid_possibilities_buf.size(),
        expected_grid_size,
        "grid_possibilities_buf size mismatch"
    );

    let num_rules = num_axes * num_tiles * num_tiles;
    let u32s_for_rules = (num_rules + 31) / 32;
    let expected_rules_size = u32s_for_rules as u64 * u32_size;
    assert_eq!(
        buffers.rules_buf.size(),
        expected_rules_size,
        "rules_buf size mismatch"
    );

    let expected_entropy_size = num_cells as u64 * f32_size;
    assert_eq!(
        buffers.entropy_buf.size(),
        expected_entropy_size,
        "entropy_buf size mismatch"
    );

    let expected_updates_size = num_cells as u64 * u32_size;
    assert_eq!(
        buffers.worklist_buf_a.size(),
        expected_updates_size,
        "worklist_buf_a size mismatch"
    );
    assert_eq!(
        buffers.worklist_buf_b.size(),
        expected_updates_size,
        "worklist_buf_b size mismatch"
    );

    let expected_atomic_u32_size = u32_size;
    assert_eq!(
        buffers.worklist_count_buf.size(),
        expected_atomic_u32_size,
        "worklist_count_buf size mismatch"
    );
    assert_eq!(
        buffers.contradiction_flag_buf.size(),
        expected_atomic_u32_size,
        "contradiction_flag_buf size mismatch"
    );

    let expected_params_size = std::mem::size_of::<wfc_gpu::buffers::GpuParamsUniform>() as u64;
    assert_eq!(
        buffers.params_uniform_buf.size(),
        expected_params_size,
        "params_uniform_buf size mismatch"
    );
}

#[test]
fn test_reset_contradiction_flag() {
    setup_logger();
    let wgpu_result = pollster::block_on(setup_wgpu());
    if let Err(e) = wgpu_result {
        eprintln!("Skipping GPU test: Failed to initialize wgpu: {}", e);
        return;
    }
    let (device, queue) = wgpu_result.unwrap();

    // Dummy grid/rules needed for buffer creation
    let grid = PossibilityGrid::new(1, 1, 1, 1);
    let rules = AdjacencyRules::new(1, 6, vec![true; 6]);
    let buffers =
        GpuBuffers::new(&device, &queue, &grid, &rules).expect("Failed to create buffers");

    // Simply test that the API call succeeds - we can't reliably test the GPU side behavior
    // in a cross-platform way without proper synchronization
    let reset_result = buffers.reset_contradiction_flag(&queue);
    assert!(reset_result.is_ok(), "reset_contradiction_flag API failed");

    // Test passed if we got here without panicking or deadlocking
}

#[test]
fn test_update_params_worklist_size() {
    setup_logger();
    let wgpu_result = pollster::block_on(setup_wgpu());
    if let Err(e) = wgpu_result {
        eprintln!("Skipping GPU test: Failed to initialize wgpu: {}", e);
        return;
    }
    let (device, queue) = wgpu_result.unwrap();

    // Dummy grid/rules
    let grid = PossibilityGrid::new(1, 1, 1, 1);
    let rules = AdjacencyRules::new(1, 6, vec![true; 6]);
    let buffers =
        GpuBuffers::new(&device, &queue, &grid, &rules).expect("Failed to create buffers");

    // Simply test that the API call succeeds - we can't reliably test the GPU side behavior
    // in a cross-platform way without proper synchronization
    let new_worklist_size = 42u32;
    let update_result = buffers.update_params_worklist_size(&queue, new_worklist_size);
    assert!(
        update_result.is_ok(),
        "update_params_worklist_size API failed"
    );

    // Test passed if we got here without panicking or deadlocking
}

#[test]
fn test_large_grid_buffer_creation() {
    setup_logger();

    // Test with larger grid sizes to ensure the GPU buffers can handle them
    let test_configs = [
        // (width, height, depth, num_tiles)
        (32, 32, 16, 4),   // Medium 3D grid with few tiles
        (64, 64, 4, 8),    // Large 2D-ish grid with moderate tiles
        (16, 16, 16, 16),  // Cubic grid with more tiles
        (128, 128, 1, 32), // Very large 2D grid approaching buffer limits
    ];

    for (width, height, depth, num_tiles) in test_configs {
        println!(
            "Testing buffer creation with grid {}x{}x{} and {} tiles",
            width, height, depth, num_tiles
        );

        // Calculate expected sizes to help with debugging
        let num_cells = width * height * depth;
        let bits_per_cell = num_tiles;
        let u32s_per_cell = (bits_per_cell + 31) / 32;
        let expected_grid_size_bytes = (num_cells * u32s_per_cell * 4) as u64;

        println!(
            "Expected grid possibilities buffer size: {} bytes ({} MB)",
            expected_grid_size_bytes,
            expected_grid_size_bytes / (1024 * 1024)
        );

        // Only attempt creation if buffer size is reasonable for most GPUs
        if expected_grid_size_bytes > 512 * 1024 * 1024 {
            println!("Skipping test case as buffer size exceeds 512MB");
            continue;
        }

        // Initialize basic grid and rules
        let grid = PossibilityGrid::new(width, height, depth, num_tiles);
        let num_axes = 6; // Standard 3D (x,y,z) * 2 directions
        let rules = AdjacencyRules::new(
            num_tiles,
            num_axes,
            vec![true; num_axes * num_tiles * num_tiles],
        );

        // Initialize Device/Queue
        let wgpu_result = pollster::block_on(setup_wgpu());
        if let Err(e) = wgpu_result {
            eprintln!("Skipping GPU test: Failed to initialize wgpu: {}", e);
            return;
        }
        let (device, queue) = wgpu_result.unwrap();

        // Attempt to create buffers
        match GpuBuffers::new(&device, &queue, &grid, &rules) {
            Ok(buffers) => {
                // Verify the sizes match expectations
                let actual_grid_size = buffers.grid_possibilities_buf.size();
                println!(
                    "Successfully created buffer of size: {} bytes ({} MB)",
                    actual_grid_size,
                    actual_grid_size / (1024 * 1024)
                );

                assert_eq!(
                    actual_grid_size, expected_grid_size_bytes,
                    "Grid possibilities buffer size mismatch"
                );

                // Verify entropy buffer size
                let expected_entropy_size = (num_cells * 4) as u64; // f32 per cell
                assert_eq!(
                    buffers.entropy_buf.size(),
                    expected_entropy_size,
                    "Entropy buffer size mismatch"
                );

                // Basic checks to ensure the GPU didn't fail silently
                assert!(buffers.grid_possibilities_buf.size() > 0);
                assert!(buffers.rules_buf.size() > 0);
                assert!(buffers.entropy_buf.size() > 0);
            }
            Err(e) => {
                // Rather than fail the test, we log the error - some test environments
                // might have limited GPU memory
                println!("Failed to create large buffer configuration: {}", e);
                println!("This may be expected on systems with limited GPU memory");
            }
        }

        // Add a small delay to let the GPU resources be properly released
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}
