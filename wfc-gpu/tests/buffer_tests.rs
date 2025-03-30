use wfc_core::{grid::PossibilityGrid, rules::AdjacencyRules};
use wfc_gpu::{
    buffers::{GpuBuffers, GpuParamsUniform},
    GpuError,
};

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
    let (device, _queue) = wgpu_result.unwrap();

    // Create buffers
    let buffers_result = GpuBuffers::new(&device, &grid, &rules);
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
        buffers.updates_buf.size(),
        expected_updates_size,
        "updates_buf size mismatch"
    );
    assert_eq!(
        buffers.output_worklist_buf.size(),
        expected_updates_size,
        "output_worklist_buf size mismatch"
    ); // Should match updates_buf

    let expected_atomic_u32_size = u32_size;
    assert_eq!(
        buffers.output_worklist_count_buf.size(),
        expected_atomic_u32_size,
        "output_worklist_count_buf size mismatch"
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
    let buffers = GpuBuffers::new(&device, &grid, &rules).expect("Failed to create buffers");

    // Write a non-zero value initially (e.g., 1)
    queue.write_buffer(
        &buffers.contradiction_flag_buf,
        0,
        bytemuck::bytes_of(&1u32),
    );
    // Need to wait for write to likely complete before reset/read
    device.poll(wgpu::Maintain::Wait);

    // Reset the flag
    let reset_result = buffers.reset_contradiction_flag(&queue);
    assert!(reset_result.is_ok(), "reset_contradiction_flag failed");
    device.poll(wgpu::Maintain::Wait);

    // Download and verify
    let flag_value = pollster::block_on(buffers.download_contradiction_flag(&device, &queue))
        .expect("Failed to download contradiction flag");
    assert_eq!(
        flag_value, false,
        "Contradiction flag was not reset to zero"
    );
}

#[test]
#[ignore] // FIXME: Test hangs, likely due to pollster/wgpu async interaction deadlock.
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
    let buffers = GpuBuffers::new(&device, &grid, &rules).expect("Failed to create buffers");

    // Download initial params to check default
    let initial_params = pollster::block_on(buffers.download_params(&device, &queue))
        .expect("Failed to download initial params");
    assert_eq!(
        initial_params.worklist_size, 0,
        "Initial worklist size not zero"
    );

    // Update the worklist size
    let new_worklist_size = 42u32;
    let update_result = buffers.update_params_worklist_size(&queue, new_worklist_size);
    assert!(update_result.is_ok(), "update_params_worklist_size failed");
    device.poll(wgpu::Maintain::Wait);

    // Download again and verify
    let updated_params = pollster::block_on(buffers.download_params(&device, &queue))
        .expect("Failed to download updated params");
    assert_eq!(
        updated_params.worklist_size, new_worklist_size,
        "Worklist size not updated correctly"
    );
}
