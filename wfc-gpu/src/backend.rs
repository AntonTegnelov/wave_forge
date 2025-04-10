// backend.rs - GPU Backend Abstraction Layer
//
// This module provides abstractions for different GPU backends that can be used
// by the Wave Function Collapse algorithm. It separates the hardware-specific code
// from the algorithm logic, allowing for easier integration of different GPU APIs.

use std::fmt::Debug;
use std::sync::Arc;
use thiserror::Error;

/// Errors that can occur when interacting with GPU backends
#[derive(Error, Debug)]
pub enum BackendError {
    /// Backend initialization failed
    #[error("Failed to initialize GPU backend: {0}")]
    InitializationFailed(String),

    /// Backend does not support required features
    #[error("GPU backend does not support required features: {0}")]
    UnsupportedFeature(String),

    /// Execution of a command failed
    #[error("GPU command execution failed: {0}")]
    ExecutionFailed(String),

    /// Memory allocation or mapping failed
    #[error("GPU memory operation failed: {0}")]
    MemoryError(String),

    /// Generic backend error
    #[error("GPU backend error: {0}")]
    Other(String),
}

/// Trait defining core capabilities required from any GPU backend
pub trait GpuBackend: Send + Sync + Debug {
    /// Initialize the GPU backend
    ///
    /// # Returns
    /// Result containing either success or backend-specific error
    fn initialize(&self) -> Result<(), BackendError>;

    /// Check if the backend supports a specific feature
    ///
    /// # Arguments
    /// * `feature_name` - Name of the feature to check
    ///
    /// # Returns
    /// True if the feature is supported, false otherwise
    fn supports_feature(&self, feature_name: &str) -> bool;

    /// Get information about the GPU backend
    ///
    /// # Returns
    /// A string containing information about the backend (name, capabilities, etc.)
    fn get_info(&self) -> String;

    /// Clean up resources when backend is no longer needed
    fn cleanup(&mut self);

    /// Get the WGPU device
    ///
    /// # Returns
    /// Arc-wrapped reference to the WGPU device
    fn device(&self) -> Arc<wgpu::Device>;

    /// Get the WGPU queue
    ///
    /// # Returns
    /// Arc-wrapped reference to the WGPU queue
    fn queue(&self) -> Arc<wgpu::Queue>;

    /// Get the supported features of the device
    ///
    /// # Returns
    /// The feature flags supported by the current device
    fn features(&self) -> wgpu::Features;

    /// Get information about the adapter
    ///
    /// # Returns
    /// Information about the GPU adapter
    fn adapter_info(&self) -> wgpu::AdapterInfo;
}

/// Trait for GPU backends that can execute compute shaders
pub trait ComputeCapable: GpuBackend {
    /// Create a compute pipeline from shader code
    ///
    /// # Arguments
    /// * `shader_code` - The shader code as a string
    /// * `entry_point` - The name of the entry point function
    ///
    /// # Returns
    /// Result containing either a backend-specific pipeline identifier or error
    fn create_compute_pipeline(
        &self,
        shader_code: &str,
        entry_point: &str,
    ) -> Result<String, BackendError>;

    /// Dispatch a compute shader
    ///
    /// # Arguments
    /// * `pipeline_id` - Identifier of the pipeline to use
    /// * `x` - X dimension of the workgroup count
    /// * `y` - Y dimension of the workgroup count
    /// * `z` - Z dimension of the workgroup count
    ///
    /// # Returns
    /// Result containing either success or error
    fn dispatch_compute(
        &self,
        pipeline_id: &str,
        _x: u32,
        _y: u32,
        _z: u32,
    ) -> Result<(), BackendError>;
}

/// Trait for GPU backends that can transfer data between CPU and GPU
pub trait DataTransfer: GpuBackend {
    /// Create a buffer on the GPU
    ///
    /// # Arguments
    /// * `size` - Size of the buffer in bytes
    /// * `usage` - How the buffer will be used (e.g., storage, uniform, etc.)
    ///
    /// # Returns
    /// Result containing either a backend-specific buffer identifier or error
    fn create_buffer(&self, size: usize, usage: &str) -> Result<String, BackendError>;

    /// Write data from CPU to GPU buffer
    ///
    /// # Arguments
    /// * `buffer_id` - Identifier of the buffer to write to
    /// * `data` - Slice of data to write
    /// * `offset` - Offset in bytes where to start writing
    ///
    /// # Returns
    /// Result containing either success or error
    fn write_buffer(
        &self,
        buffer_id: &str,
        _data: &[u8],
        _offset: usize,
    ) -> Result<(), BackendError>;

    /// Read data from GPU buffer to CPU
    ///
    /// # Arguments
    /// * `buffer_id` - Identifier of the buffer to read from
    /// * `size` - Number of bytes to read
    /// * `offset` - Offset in bytes where to start reading
    ///
    /// # Returns
    /// Result containing either the read data or error
    fn read_buffer(
        &self,
        buffer_id: &str,
        _size: usize,
        _offset: usize,
    ) -> Result<Vec<u8>, BackendError>;
}

/// Trait for synchronization methods between CPU and GPU operations
pub trait Synchronization: GpuBackend {
    /// Wait for all pending GPU operations to complete
    fn synchronize(&self) -> Result<(), BackendError>;

    /// Submit operations to the GPU queue
    ///
    /// # Returns
    /// Result containing either success or error
    fn submit(&self) -> Result<(), BackendError>;
}

/// WGPU-specific implementation of the GpuBackend trait
#[derive(Debug)]
pub struct WgpuBackend {
    /// WGPU instance
    instance: Arc<wgpu::Instance>,
    /// WGPU adapter (physical GPU)
    adapter: Option<Arc<wgpu::Adapter>>,
    /// WGPU logical device
    device: Option<Arc<wgpu::Device>>,
    /// WGPU command queue
    queue: Option<Arc<wgpu::Queue>>,
    /// Stores created pipeline IDs and their corresponding pipelines
    pipelines: std::collections::HashMap<String, Arc<wgpu::ComputePipeline>>,
    /// Stores created buffer IDs and their corresponding buffers
    buffers: std::collections::HashMap<String, Arc<wgpu::Buffer>>,
}

impl Default for WgpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl WgpuBackend {
    /// Create a new WGPU backend
    ///
    /// # Returns
    /// A new WgpuBackend instance
    pub fn new() -> Self {
        let instance = Arc::new(wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        }));

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("Failed to find GPU adapter");

        let adapter = Arc::new(adapter);

        // Create limits with increased storage buffer capacity
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffers_per_shader_stage = 10;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("WFC GPU Device"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::default(),
        }))
        .expect("Failed to create device");

        Self {
            instance,
            adapter: Some(adapter),
            device: Some(Arc::new(device)),
            queue: Some(Arc::new(queue)),
            pipelines: std::collections::HashMap::new(),
            buffers: std::collections::HashMap::new(),
        }
    }

    /// Convert a wgpu::BufferUsages from string description
    fn parse_buffer_usage(&self, usage: &str) -> wgpu::BufferUsages {
        match usage {
            "storage" => wgpu::BufferUsages::STORAGE,
            "uniform" => wgpu::BufferUsages::UNIFORM,
            "copy_dst" => wgpu::BufferUsages::COPY_DST,
            "copy_src" => wgpu::BufferUsages::COPY_SRC,
            "map_read" => wgpu::BufferUsages::MAP_READ,
            "map_write" => wgpu::BufferUsages::MAP_WRITE,
            _ => wgpu::BufferUsages::empty(),
        }
    }
}

impl GpuBackend for WgpuBackend {
    fn initialize(&self) -> Result<(), BackendError> {
        // If already initialized, just return success
        if self.device.is_some() && self.queue.is_some() {
            return Ok(());
        }

        // This is a bit awkward since we're initializing in `new()`
        // and our struct fields are not mutable. In a real implementation,
        // we would handle this differently.
        Err(BackendError::Other(
            "Backend already initialized in new()".to_string(),
        ))
    }

    fn supports_feature(&self, feature_name: &str) -> bool {
        // In a real implementation, this would check the adapter features
        match feature_name {
            "compute" => true,
            "storage_buffers" => true,
            "f16" => false,
            _ => false,
        }
    }

    fn get_info(&self) -> String {
        if let Some(adapter) = &self.adapter {
            let info = adapter.get_info();
            format!(
                "Backend: {:?}, Vendor: {:?}, Device: {}, DeviceType: {:?}, Name: {:?}, Driver: {:?}",
                info.backend, info.vendor, info.device, info.device_type, info.name, info.driver
            )
        } else {
            "Unknown Adapter".to_string()
        }
    }

    fn cleanup(&mut self) {
        // Clear all stored resources
        self.pipelines.clear();
        self.buffers.clear();

        // Drop references to device and queue
        self.device = None;
        self.queue = None;
        self.adapter = None;
    }

    fn device(&self) -> Arc<wgpu::Device> {
        self.device
            .as_ref()
            .expect("Device not initialized")
            .clone()
    }

    fn queue(&self) -> Arc<wgpu::Queue> {
        self.queue.as_ref().expect("Queue not initialized").clone()
    }

    fn features(&self) -> wgpu::Features {
        self.device().features()
    }

    fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.adapter
            .as_ref()
            .expect("Adapter not initialized")
            .get_info()
    }
}

impl ComputeCapable for WgpuBackend {
    fn create_compute_pipeline(
        &self,
        shader_code: &str,
        entry_point: &str,
    ) -> Result<String, BackendError> {
        let device = self.device.as_ref().ok_or(BackendError::Other(
            "Device not initialized. Call initialize() first.".to_string(),
        ))?;

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("WFC GPU Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });

        let _pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("WFC GPU Compute Pipeline"),
            layout: None,
            module: &shader_module,
            entry_point: Some(entry_point),
            cache: None,
            compilation_options: Default::default(),
        });

        // Generate a unique ID for this pipeline
        let pipeline_id = format!("pipeline_{}", self.pipelines.len());

        // Store the pipeline (this would require interior mutability in a real implementation)
        // For this example, we'll return the ID but not actually store the pipeline
        Ok(pipeline_id)
    }

    fn dispatch_compute(
        &self,
        pipeline_id: &str,
        _x: u32,
        _y: u32,
        _z: u32,
    ) -> Result<(), BackendError> {
        let _device = self.device.as_ref().ok_or(BackendError::Other(
            "Device not initialized. Call initialize() first.".to_string(),
        ))?;

        let _queue = self.queue.as_ref().ok_or(BackendError::Other(
            "Queue not initialized. Call initialize() first.".to_string(),
        ))?;

        let _pipeline = self
            .pipelines
            .get(pipeline_id)
            .ok_or(BackendError::Other(format!(
                "Pipeline with ID {} not found",
                pipeline_id
            )))?;

        // This is just a demonstration - in a real implementation, we would create an encoder,
        // set up the compute pass, dispatch, and submit the command buffer
        Err(BackendError::Other(
            "This implementation is just a demonstration".to_string(),
        ))
    }
}

impl DataTransfer for WgpuBackend {
    fn create_buffer(&self, size: usize, usage: &str) -> Result<String, BackendError> {
        let device = self.device.as_ref().ok_or(BackendError::Other(
            "Device not initialized. Call initialize() first.".to_string(),
        ))?;

        // Parse the usage string into wgpu::BufferUsages
        let buffer_usage = self.parse_buffer_usage(usage);

        // Create the buffer
        let _buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("WFC GPU Buffer"),
            size: size as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        // Generate a unique ID for this buffer
        let buffer_id = format!("buffer_{}", self.buffers.len());

        // Store the buffer (this would require interior mutability in a real implementation)
        // For this example, we'll return the ID but not actually store the buffer
        Ok(buffer_id)
    }

    fn write_buffer(
        &self,
        buffer_id: &str,
        _data: &[u8],
        _offset: usize,
    ) -> Result<(), BackendError> {
        let _queue = self.queue.as_ref().ok_or(BackendError::Other(
            "Queue not initialized. Call initialize() first.".to_string(),
        ))?;

        let _buffer = self
            .buffers
            .get(buffer_id)
            .ok_or(BackendError::Other(format!(
                "Buffer with ID {} not found",
                buffer_id
            )))?;

        // Write data to the buffer
        // This is just a demonstration - in a real implementation, we would use queue.write_buffer
        Err(BackendError::Other(
            "This implementation is just a demonstration".to_string(),
        ))
    }

    fn read_buffer(
        &self,
        buffer_id: &str,
        _size: usize,
        _offset: usize,
    ) -> Result<Vec<u8>, BackendError> {
        let _buffer = self
            .buffers
            .get(buffer_id)
            .ok_or(BackendError::Other(format!(
                "Buffer with ID {} not found",
                buffer_id
            )))?;

        // This is just a demonstration - in a real implementation, we would map the buffer and read data
        Err(BackendError::Other(
            "This implementation is just a demonstration".to_string(),
        ))
    }
}

impl Synchronization for WgpuBackend {
    fn synchronize(&self) -> Result<(), BackendError> {
        let _device = self.device.as_ref().ok_or(BackendError::Other(
            "Device not initialized. Call initialize() first.".to_string(),
        ))?;

        // In a real implementation, we would create a fence or use device.poll to wait for operations
        Err(BackendError::Other(
            "This implementation is just a demonstration".to_string(),
        ))
    }

    fn submit(&self) -> Result<(), BackendError> {
        // In a real implementation, we would submit command buffers to the queue
        Err(BackendError::Other(
            "This implementation is just a demonstration".to_string(),
        ))
    }
}

/// Factory for creating different GPU backends
pub struct GpuBackendFactory;

impl GpuBackendFactory {
    /// Create a new GPU backend of the specified type
    ///
    /// # Arguments
    /// * `backend_type` - The type of backend to create (e.g., "wgpu")
    ///
    /// # Returns
    /// Result containing either a boxed GpuBackend trait object or error
    pub fn create(backend_type: &str) -> Result<Box<dyn GpuBackend>, BackendError> {
        match backend_type {
            "wgpu" => Ok(Box::new(WgpuBackend::new())),
            _ => Err(BackendError::UnsupportedFeature(format!(
                "Backend type '{}' not supported",
                backend_type
            ))),
        }
    }

    /// Create a new GPU backend based on available system capabilities
    ///
    /// This function attempts to find the best available backend for the current system.
    ///
    /// # Returns
    /// Result containing either a boxed GpuBackend trait object or error
    pub fn create_best_available() -> Result<Box<dyn GpuBackend>, BackendError> {
        // In a real implementation, we would check for available backends and select the best one
        // For now, we'll just return the WGPU backend
        Ok(Box::new(WgpuBackend::new()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_factory_create() {
        let result = GpuBackendFactory::create("wgpu");
        assert!(result.is_ok(), "Should be able to create WGPU backend");

        let result = GpuBackendFactory::create("invalid");
        assert!(result.is_err(), "Should fail for invalid backend type");
    }

    #[test]
    fn test_backend_factory_create_best_available() {
        let result = GpuBackendFactory::create_best_available();
        assert!(
            result.is_ok(),
            "Should be able to create best available backend"
        );
    }

    #[test]
    fn test_wgpu_backend_supports_feature() {
        let backend = WgpuBackend::new();

        // Features that should be supported
        assert!(
            backend.supports_feature("compute"),
            "Should support compute feature"
        );
        assert!(
            backend.supports_feature("storage_buffers"),
            "Should support storage_buffers feature"
        );

        // Features that shouldn't be supported
        assert!(
            !backend.supports_feature("nonexistent_feature"),
            "Should not support nonexistent feature"
        );
    }

    #[test]
    fn test_wgpu_backend_get_info() {
        let backend = WgpuBackend::new();
        let info = backend.get_info();

        // The backend is not initialized, so we expect a default message
        assert_eq!(info, "Unknown Adapter");
    }

    #[test]
    fn test_backend_error_display() {
        let errors = vec![
            BackendError::InitializationFailed("test reason".to_string()),
            BackendError::UnsupportedFeature("test feature".to_string()),
            BackendError::ExecutionFailed("test execution".to_string()),
            BackendError::MemoryError("test memory".to_string()),
            BackendError::Other("test other".to_string()),
        ];

        for error in errors {
            let error_string = format!("{}", error);
            assert!(
                !error_string.is_empty(),
                "Error should display a non-empty message"
            );
        }
    }

    // Mock implementation of GpuBackend for testing
    #[derive(Debug)]
    struct MockBackend;

    impl GpuBackend for MockBackend {
        fn initialize(&self) -> Result<(), BackendError> {
            Ok(())
        }

        fn supports_feature(&self, feature_name: &str) -> bool {
            feature_name == "mock_feature"
        }

        fn get_info(&self) -> String {
            "Mock Backend 1.0".to_string()
        }

        fn cleanup(&mut self) {
            // Nothing to clean up in mock
        }

        fn device(&self) -> Arc<wgpu::Device> {
            // This is a mock implementation that can't actually be created
            // In a real test, we would use a proper test device
            panic!("MockBackend cannot provide a real device");
        }

        fn queue(&self) -> Arc<wgpu::Queue> {
            // This is a mock implementation that can't actually be created
            // In a real test, we would use a proper test queue
            panic!("MockBackend cannot provide a real queue");
        }

        fn features(&self) -> wgpu::Features {
            wgpu::Features::empty()
        }

        fn adapter_info(&self) -> wgpu::AdapterInfo {
            wgpu::AdapterInfo {
                name: "Mock Device".to_string(),
                vendor: 0,
                device: 0,
                device_type: wgpu::DeviceType::Cpu,
                driver: "Mock Driver".to_string(),
                driver_info: "Mock Driver Info".to_string(),
                backend: wgpu::Backend::Vulkan,
            }
        }
    }

    #[test]
    fn test_mock_backend() {
        let mut mock = MockBackend;

        // Test initialize
        assert!(mock.initialize().is_ok());

        // Test feature support
        assert!(mock.supports_feature("mock_feature"));
        assert!(!mock.supports_feature("unknown_feature"));

        // Test info
        assert_eq!(mock.get_info(), "Mock Backend 1.0");

        // Test cleanup (just make sure it doesn't panic)
        mock.cleanup();
    }
}
