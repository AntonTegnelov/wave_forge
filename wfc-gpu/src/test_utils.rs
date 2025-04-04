use pollster;
use wgpu;

/// Initializes a WGPU instance, adapter, device, and queue for testing purposes.
///
/// Panics if a suitable adapter or device cannot be found.
pub fn initialize_test_gpu() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("Failed to find suitable adapter");

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Test Device"),
            required_features: wgpu::Features::empty(), // Adjust features as needed by tests
            required_limits: wgpu::Limits::default(),
        },
        None, // Trace path
    ))
    .expect("Failed to request device");

    (device, queue)
}
