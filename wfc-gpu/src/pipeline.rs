use crate::GpuError;
use std::collections::HashMap;
use std::sync::Arc;
use wgpu;

/// Manages the WGPU compute pipelines required for WFC acceleration.
///
/// This struct holds the compiled compute pipeline objects and their corresponding
/// bind group layouts for both the entropy calculation and constraint propagation shaders.
/// It is typically created once during the initialization of the `GpuAccelerator`.
#[derive(Clone, Debug)]
pub struct ComputePipelines {
    /// The compiled compute pipeline for the entropy calculation shader (`entropy.wgsl`).
    pub entropy_pipeline: Arc<wgpu::ComputePipeline>,
    /// The compiled compute pipeline for the constraint propagation shader (`propagate.wgsl`).
    pub propagation_pipeline: Arc<wgpu::ComputePipeline>,
    /// The layout describing the binding structure for the entropy pipeline's bind group.
    /// Required for creating bind groups compatible with `entropy_pipeline`.
    pub entropy_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    /// The layout describing the binding structure for the propagation pipeline's bind group.
    /// Required for creating bind groups compatible with `propagation_pipeline`.
    pub propagation_bind_group_layout: Arc<wgpu::BindGroupLayout>,
}

impl ComputePipelines {
    /// Creates new `ComputePipelines` by loading shaders and compiling them.
    ///
    /// This function:
    /// 1. Loads the WGSL source code for the entropy and propagation shaders.
    /// 2. Creates `wgpu::ShaderModule` objects from the source code.
    /// 3. Defines the `wgpu::BindGroupLayout` for each shader, specifying the types and bindings
    ///    of the GPU buffers they expect (e.g., storage buffers, uniform buffers).
    /// 4. Defines the `wgpu::PipelineLayout` using the bind group layouts.
    /// 5. Creates the `wgpu::ComputePipeline` objects using the shader modules and pipeline layouts.
    ///
    /// # Arguments
    ///
    /// * `device` - A reference to the WGPU `Device` used for creating pipeline resources.
    /// * `num_tiles_u32` - The number of u32 chunks needed per cell, used for specialization.
    ///
    /// # Returns
    ///
    /// * `Ok(Self)` containing the initialized `ComputePipelines`.
    /// * `Err(GpuError)` if shader loading, compilation, or pipeline creation fails.
    pub fn new(device: &wgpu::Device, num_tiles_u32: u32) -> Result<Self, GpuError> {
        // Load shader code
        let entropy_shader_code = include_str!("shaders/entropy.wgsl");
        let propagation_shader_code = include_str!("shaders/propagate.wgsl");

        // Create shader modules
        let entropy_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Entropy Shader"),
            source: wgpu::ShaderSource::Wgsl(entropy_shader_code.into()),
        });

        let propagation_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Propagation Shader"),
            source: wgpu::ShaderSource::Wgsl(propagation_shader_code.into()),
        });

        // --- Define Bind Group Layouts ---

        // Layout for entropy shader
        let entropy_bind_group_layout = Arc::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Entropy Bind Group Layout"),
                entries: &[
                    // grid_possibilities (read-only storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: Some(std::num::NonZeroU64::new(4).unwrap()),
                        },
                        count: None,
                    },
                    // entropy_output (write-only storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: Some(std::num::NonZeroU64::new(4).unwrap()),
                        },
                        count: None,
                    },
                    // params (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // min_entropy_info (read-write storage, atomic vec2<u32>)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3, // New binding for min entropy info
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false }, // Needs write access for atomic operations
                            has_dynamic_offset: false,
                            // Minimum size for vec2<u32>
                            min_binding_size: Some(std::num::NonZeroU64::new(8).unwrap()),
                        },
                        count: None,
                    },
                ],
            },
        ));

        // Layout for propagation shader
        let propagation_bind_group_layout = Arc::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Propagation Bind Group Layout"),
                entries: &[
                    // grid_possibilities (read-write storage, atomic)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: Some(std::num::NonZeroU64::new(4).unwrap()),
                        },
                        count: None,
                    },
                    // adjacency_rules (read-only storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: Some(std::num::NonZeroU64::new(4).unwrap()),
                        },
                        count: None,
                    },
                    // worklist (read-only storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: Some(std::num::NonZeroU64::new(4).unwrap()),
                        },
                        count: None,
                    },
                    // output_worklist (read-write storage, atomic)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: Some(std::num::NonZeroU64::new(4).unwrap()),
                        },
                        count: None,
                    },
                    // params (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // output_worklist_count (atomic u32)
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: Some(std::num::NonZeroU64::new(4).unwrap()),
                        },
                        count: None,
                    },
                    // contradiction_flag (atomic u32)
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: Some(std::num::NonZeroU64::new(4).unwrap()),
                        },
                        count: None,
                    },
                    // contradiction_location (atomic u32)
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: Some(std::num::NonZeroU64::new(4).unwrap()),
                        },
                        count: None,
                    },
                ],
            },
        ));

        // --- Define Pipeline Layouts ---
        let entropy_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Entropy Pipeline Layout"),
                bind_group_layouts: &[&entropy_bind_group_layout],
                push_constant_ranges: &[],
            });

        let propagation_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Propagation Pipeline Layout"),
                bind_group_layouts: &[&propagation_bind_group_layout],
                push_constant_ranges: &[],
            });

        // --- Create Compute Pipelines ---

        // Define the specialization constant value
        let specialization_constants = HashMap::from([
            // Value must be f64 according to wgpu docs
            ("NUM_TILES_U32".to_string(), num_tiles_u32 as f64),
        ]);

        let entropy_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Entropy Pipeline"),
                layout: Some(&entropy_pipeline_layout),
                module: &entropy_shader,
                entry_point: "main", // Assuming entry point is 'main'
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &specialization_constants,
                    ..Default::default()
                },
            },
        ));

        let propagation_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Propagation Pipeline"),
                layout: Some(&propagation_pipeline_layout),
                module: &propagation_shader,
                entry_point: "main", // Assuming entry point is 'main'
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &specialization_constants,
                    ..Default::default()
                },
            },
        ));

        Ok(Self {
            entropy_pipeline,
            propagation_pipeline,
            entropy_bind_group_layout,
            propagation_bind_group_layout,
        })
    }
}
