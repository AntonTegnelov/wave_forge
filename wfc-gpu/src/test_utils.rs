use futures::executor::block_on;
use wfc_core::{grid::PossibilityGrid, BoundaryCondition};
use wfc_rules::AdjacencyRules;
use wgpu;

use crate::accelerator::GpuAccelerator;
use crate::GpuError;

// Initialize GPU for testing
pub fn initialize_test_gpu() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::default();

    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        .expect("Failed to get adapter");

    let (device, queue) = block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Test Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ))
    .expect("Failed to create device");

    (device, queue)
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

    // Initialize the accelerator (blocking on the async method)
    let result = block_on(GpuAccelerator::new(
        &grid,
        &rules,
        BoundaryCondition::Periodic,
    ));

    result
}
