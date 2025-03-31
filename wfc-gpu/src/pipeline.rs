use crate::GpuError;
use wgpu;

// Placeholder struct for managing GPU pipelines
pub struct ComputePipelines {
    pub entropy_pipeline: wgpu::ComputePipeline,
    pub propagation_pipeline: wgpu::ComputePipeline,
    // Keep layouts if needed elsewhere (e.g., for creating bind groups dynamically)
    pub entropy_bind_group_layout: wgpu::BindGroupLayout,
    pub propagation_bind_group_layout: wgpu::BindGroupLayout,
}

impl ComputePipelines {
    pub fn new(device: &wgpu::Device) -> Result<Self, GpuError> {
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
        let entropy_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Entropy Bind Group Layout"),
                entries: &[
                    // grid_possibilities (read-only storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None, // TODO: Specify minimum size?
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
                            min_binding_size: None,
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
            });

        // Layout for propagation shader
        let propagation_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Propagation Bind Group Layout"),
                entries: &[
                    // grid_possibilities (read-write storage, atomic)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
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
                            min_binding_size: None,
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
                            min_binding_size: None,
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
                            min_binding_size: None,
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
            });

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
        let entropy_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Entropy Pipeline"),
            layout: Some(&entropy_pipeline_layout),
            module: &entropy_shader,
            entry_point: "main", // Assuming entry point is 'main'
            compilation_options: Default::default(),
        });

        let propagation_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Propagation Pipeline"),
                layout: Some(&propagation_pipeline_layout),
                module: &propagation_shader,
                entry_point: "main", // Assuming entry point is 'main'
                compilation_options: Default::default(),
            });

        Ok(Self {
            entropy_pipeline,
            propagation_pipeline,
            entropy_bind_group_layout,
            propagation_bind_group_layout,
        })
    }
}
