use std::sync::Arc;
use wfc_core::{grid::PossibilityGrid, BoundaryCondition};
use wfc_rules::AdjacencyRules;
use wgpu;

use crate::accelerator::GpuAccelerator;
use crate::buffers::GpuBuffers;
use crate::GpuError;

/// Synchronously initialize GPU for testing
/// This uses polling directly instead of block_on for compatibility with tokio
pub fn initialize_test_gpu() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::default();

    // Use polling approach instead of block_on to avoid deadlocks with tokio
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("Failed to get adapter");

    // Create a limits struct that increases the max storage buffers per shader stage
    let mut limits = wgpu::Limits::downlevel_defaults();
    limits.max_storage_buffers_per_shader_stage = 10;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Test Device"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
        },
        None,
    ))
    .expect("Failed to create device");

    (device, queue)
}

/// Create test device and queue wrapped in Arc for async sharing
pub fn create_test_device_queue() -> (Arc<wgpu::Device>, Arc<wgpu::Queue>) {
    let (device, queue) = initialize_test_gpu();
    (Arc::new(device), Arc::new(queue))
}

/// Test if a large tileset (over 128 tiles) can be initialized correctly
pub fn test_large_tileset_init(
    width: u32,
    height: u32,
    num_tiles: u32,
) -> Result<GpuAccelerator, GpuError> {
    // Create placeholder adjacency rules (empty)
    let num_axes = 6; // Standard 3D axes: +X, -X, +Y, -Y, +Z, -Z
    let rules = AdjacencyRules::from_allowed_tuples(
        num_tiles as usize,
        num_axes,
        Vec::<(usize, usize, usize)>::new(), // No allowed adjacencies specified
    );

    // Create a possibility grid
    let grid = PossibilityGrid::new(
        width as usize,
        height as usize,
        1, // depth of 1 for 2D
        num_tiles as usize,
    );

    // Initialize the accelerator (using pollster instead of block_on)
    let result = pollster::block_on(GpuAccelerator::new(
        &grid,
        &rules,
        BoundaryCondition::Periodic,
        wfc_core::entropy::EntropyHeuristicType::Shannon, // Added missing heuristic argument
        None,                                             // No subgrid config
    ));

    result
}

/// Creates GPU buffers for testing purposes.
///
/// # Arguments
///
/// * `device` - Arc reference to the WGPU device.
/// * `queue` - Arc reference to the WGPU queue.
///
/// # Returns
///
/// An Arc reference to the created `GpuBuffers`.
pub fn create_test_gpu_buffers(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
) -> Arc<GpuBuffers> {
    // Create a minimal grid and rules for buffer initialization
    let grid = PossibilityGrid::new(2, 2, 1, 3);
    let rules = AdjacencyRules::from_allowed_tuples(3, 6, vec![]);
    Arc::new(
        GpuBuffers::new(device, queue, &grid, &rules, BoundaryCondition::Finite)
            .expect("Failed to create test GPU buffers"),
    )
}
