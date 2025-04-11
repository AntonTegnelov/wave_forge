use log;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use wgpu;
// Added imports for caching
use once_cell::sync::Lazy;
use seahash::SeaHasher;
use std::hash::{Hash, Hasher};
// Import ShaderManager and related types
use super::shaders::{ShaderManager, ShaderType};
use crate::GpuError;
use lazy_static::lazy_static;

// --- Cache Definitions ---

// Key for shader module cache: based on shader source code
#[derive(PartialEq, Eq, Hash, Clone)]
struct ShaderCacheKey {
    source_hash: u64,
}

// Key for pipeline cache: includes shader details and configuration
#[derive(PartialEq, Eq, Hash, Clone)]
struct PipelineCacheKey {
    source_hash: u64,
    entry_point: String,
}

// Static caches using Lazy and Mutex for thread-safe initialization and access
static SHADER_MODULE_CACHE: Lazy<Mutex<HashMap<ShaderCacheKey, Arc<wgpu::ShaderModule>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

// Use lazy_static macro correctly
lazy_static! {
    static ref COMPUTE_PIPELINE_CACHE: Mutex<HashMap<PipelineCacheKey, Arc<wgpu::ComputePipeline>>> =
        Mutex::new(HashMap::new());
}

// --- Helper Function to Hash Strings ---
fn hash_string(s: &str) -> u64 {
    let mut hasher = SeaHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

// --- TODO: Shader Source Loading and Compilation (Future Responsibility of ShaderCompiler) ---

// Placeholder function for loading shader components
// In the future, this will interact with ShaderRegistry and ShaderCompiler
// Remove this function as ShaderManager handles loading
/*
fn load_shader_source(shader_name: &str, features: &[&str]) -> Result<String, GpuError> {
    // Placeholder implementation: Read monolithic files for now until compiler exists
    // WARNING: This section needs to be replaced with the actual shader component loading and assembly logic.
    log::warn!(
        "Using placeholder shader loading for '{}'. Needs integration with ShaderCompiler.",
        shader_name
    );
    match shader_name {
        "entropy" => {
            // Decide based on features (e.g., atomics)
            let has_atomics = features.contains(&"atomics");
            if has_atomics {
                Ok(include_str!("shaders/entropy_modular.wgsl").to_string()) // Example: Use modular if atomics present
            } else {
                Ok(include_str!("shaders/entropy_fallback.wgsl").to_string())
            }
        }
        "propagate" => {
            let has_atomics = features.contains(&"atomics");
            if has_atomics {
                Ok(include_str!("shaders/propagate_modular.wgsl").to_string())
            } else {
                Ok(include_str!("shaders/propagate_fallback.wgsl").to_string())
            }
        }
        _ => Err(GpuError::ShaderError("Unknown shader name".to_string())),
    }
}
*/

// Placeholder for future shader assembly/compilation
// This would take component sources and assemble them based on features.
fn compile_shader(
    device: &wgpu::Device,
    shader_type: ShaderType, // Use ShaderType enum
    features: &[&str],
    shader_manager: &ShaderManager, // Pass ShaderManager instance
    _num_tiles_u32: u32,            // May be needed for specialization in future compiler
) -> Result<(Arc<wgpu::ShaderModule>, u64), GpuError> {
    // 1. Load/Assemble source using ShaderManager
    let source_code = shader_manager
        .load_shader_variant(shader_type, features)
        .map_err(|e| GpuError::ShaderError(e.to_string()))?;

    // TODO: Apply specialization constants (like NUM_TILES_U32_VALUE) using the compiler
    // let processed_source = future_shader_compiler.specialize(&source_code, num_tiles_u32);

    // 2. Check cache
    let source_hash = hash_string(&source_code);
    let shader_key = ShaderCacheKey {
        source_hash, // Use calculated hash
    };
    let mut cache = SHADER_MODULE_CACHE
        .lock()
        .map_err(|e| GpuError::MutexError(e.to_string()))?;

    if let Some(module) = cache.get(&shader_key) {
        log::debug!("Shader cache hit for {:?}", shader_type);
        return Ok((module.clone(), source_hash)); // Return cached module and hash
    }

    // 3. Create module if not in cache
    log::debug!(
        "Shader cache miss for {:?}. Creating new module.",
        shader_type
    );
    let shader_module = Arc::new(device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&format!("{:?}_Shader", shader_type)),
        source: wgpu::ShaderSource::Wgsl(source_code.into()),
    }));

    cache.insert(shader_key, shader_module.clone());
    Ok((shader_module, source_hash)) // Return new module and hash
}

/// Manages the WGPU compute pipelines required for WFC acceleration.
///
/// This struct holds the compiled compute pipeline objects and their corresponding
/// bind group layouts for both the entropy calculation and constraint propagation shaders.
/// It also stores the dynamically determined workgroup sizes for optimal dispatch.
/// It is typically created once during the initialization of the `GpuAccelerator`.
#[derive(Clone, Debug)]
pub struct ComputePipelines {
    /// The compiled compute pipeline for the entropy calculation shader (`entropy.wgsl`).
    pub entropy_pipeline: Arc<wgpu::ComputePipeline>,
    /// The compiled compute pipeline for the constraint propagation shader (`propagate.wgsl`).
    pub propagation_pipeline: Arc<wgpu::ComputePipeline>,
    /// The layout describing the binding structure for the entropy pipeline's bind group.
    /// Required for creating bind groups compatible with `entropy_pipeline`.
    pub entropy_bind_group_layout_0: Arc<wgpu::BindGroupLayout>,
    /// The layout describing the binding structure for the entropy pipeline's bind group.
    /// Required for creating bind groups compatible with `entropy_pipeline`.
    pub entropy_bind_group_layout_1: Arc<wgpu::BindGroupLayout>,
    /// The layout describing the binding structure for the propagation pipeline's bind group.
    /// Required for creating bind groups compatible with `propagation_pipeline`.
    pub propagation_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    /// Dynamically determined optimal workgroup size (X-dimension) for the entropy shader.
    pub entropy_workgroup_size: u32,
    /// Dynamically determined optimal workgroup size (X-dimension) for the propagation shader.
    pub propagation_workgroup_size: u32,
}

/// Groups the buffer resources needed for creating entropy bind groups.
#[derive(Debug)] // Added Debug derive
pub struct GpuComputeBindingResources<'a> {
    pub params_buf: &'a wgpu::Buffer, // Uniform buffer with entropy parameters
    pub grid_possibilities_buf: &'a wgpu::Buffer, // Storage buffer with grid possibility bitsets (read-only)
    pub entropy_data_buf: &'a wgpu::Buffer, // Storage buffer for calculated entropy values (write-only or read/write)
    pub min_entropy_info_buf: &'a wgpu::Buffer, // Storage buffer for atomic min entropy tracking (read/write)
}

/// Groups the buffer resources needed for creating propagation bind groups.
#[derive(Debug)] // Added Debug derive
pub struct PropagationBindingResources<'a> {
    pub grid_possibilities_buf: &'a wgpu::Buffer,
    pub adjacency_rules_buf: &'a wgpu::Buffer,
    pub rule_weights_buf: &'a wgpu::Buffer,
    pub worklist_bufs: &'a [wgpu::Buffer; 2],
    pub output_worklist_bufs: &'a [wgpu::Buffer; 2],
    pub params_uniform_buf: &'a wgpu::Buffer,
    pub worklist_count_bufs: &'a [wgpu::Buffer; 2],
    pub contradiction_flag_buf: &'a wgpu::Buffer,
    pub contradiction_location_buf: &'a wgpu::Buffer,
    pub pass_statistics_buf: &'a wgpu::Buffer,
}

impl ComputePipelines {
    /// Creates new `ComputePipelines` by loading shaders and compiling them.
    ///
    /// Uses caching to avoid recompiling shaders and pipelines if they have been created before
    /// with the same configuration.
    ///
    /// This function:
    /// 1. Queries device limits for optimal workgroup sizing.
    /// 2. Loads the WGSL source code for the entropy and propagation shaders.
    /// 3. Creates `wgpu::ShaderModule` objects from the source code.
    /// 4. Defines the `wgpu::BindGroupLayout` for each shader, specifying the types and bindings
    ///    of the GPU buffers they expect (e.g., storage buffers, uniform buffers).
    /// 5. Defines the `wgpu::PipelineLayout` using the bind group layouts.
    /// 6. Creates the `wgpu::ComputePipeline` objects using the shader modules, pipeline layouts,
    ///    and specialization constants (including the dynamically determined workgroup size).
    ///
    /// # Arguments
    ///
    /// * `device` - A reference to the WGPU `Device` used for creating pipeline resources.
    /// * `num_tiles_u32` - The number of u32 chunks needed per cell, used for specialization.
    /// * `features` - A slice of strings representing enabled GPU/shader features (e.g., ["atomics"]).
    ///
    /// # Returns
    ///
    /// * `Ok(Self)` containing the initialized `ComputePipelines`.
    /// * `Err(GpuError)` if shader loading, compilation, or pipeline creation fails.
    pub fn new(
        device: &wgpu::Device,
        num_tiles_u32: u32,
        features: &[&str],
    ) -> Result<Self, GpuError> {
        // Create ShaderManager instance
        let shader_manager =
            ShaderManager::new().map_err(|e| GpuError::ShaderError(e.to_string()))?;

        // Query device limits
        let limits = device.limits();
        let max_invocations = limits.max_compute_invocations_per_workgroup;
        log::debug!(
            "GPU max compute invocations per workgroup: {}",
            max_invocations
        );

        // Determine optimal workgroup size
        // Example: Aim for 64-256 invocations, typically square root for 2D
        let workgroup_size: u32 = if max_invocations >= 256 {
            16 // 16x16 = 256
        } else if max_invocations >= 64 {
            8 // 8x8 = 64
        } else {
            // Fallback for very low limits, adjust as needed
            (max_invocations as f64).sqrt() as u32
        };
        let entropy_workgroup_size = workgroup_size;
        let propagation_workgroup_size = workgroup_size; // Use same for now

        // --- Compile Shaders (using compile_shader helper) ---
        let (entropy_shader_module, entropy_hash) = compile_shader(
            device,
            ShaderType::Entropy,
            features,
            &shader_manager,
            num_tiles_u32,
        )?;
        let (propagation_shader_module, propagation_hash) = compile_shader(
            device,
            ShaderType::Propagation,
            features,
            &shader_manager,
            num_tiles_u32,
        )?;

        // --- Define Bind Group Layouts ---
        // Entropy Layout Group 0
        let entropy_bind_group_layout_0 = Arc::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Entropy Bind Group Layout 0"),
                entries: &[
                    // @group(0) @binding(0): grid_possibilities (Storage RO)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // @group(0) @binding(1): params (Uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None, // Size checked at buffer creation
                        },
                        count: None,
                    },
                ],
            },
        ));

        // Entropy Layout Group 1
        let entropy_bind_group_layout_1 = Arc::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Entropy Bind Group Layout 1"),
                entries: &[
                    // @group(1) @binding(0): entropy_grid (Storage RW)
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
                    // @group(1) @binding(1): min_entropy_info (Atomic Storage RW)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            // Needs to be Storage RW for atomics
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            // Size for atomic<u32> min_entropy_bits + atomic<u32> min_index
                            min_binding_size: Some(std::num::NonZeroU64::new(8).unwrap()),
                        },
                        count: None,
                    },
                ],
            },
        ));

        // Propagation Layout (unchanged for now, assuming previous structure)
        let propagation_bind_group_layout = Arc::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Propagation Bind Group Layout"),
                entries: &[
                    // Grid Possibilities (Read/Write)
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
                    // Adjacency Rules (Read Only)
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
                    // Rule Weights (Read Only) - Added
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
                    // Input Worklist (Read Only)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Output Worklist (Write Only)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Uniform Parameters (Read Only)
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Worklist Counter (Read/Write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Contradiction Flag (Read/Write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Contradiction Location (Read/Write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Pass Statistics (Read/Write) - Added
                    wgpu::BindGroupLayoutEntry {
                        binding: 9,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            },
        ));

        // --- Create Pipeline Layouts ---
        let entropy_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Entropy Pipeline Layout"),
                // IMPORTANT: List layouts for *all* bind groups used by the shader
                bind_group_layouts: &[&entropy_bind_group_layout_0, &entropy_bind_group_layout_1], // Assuming group 0 and 1 use the same layout structure for now
                push_constant_ranges: &[],
            });
        let propagation_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Propagation Pipeline Layout"),
                bind_group_layouts: &[&propagation_bind_group_layout],
                push_constant_ranges: &[],
            });

        // --- Create Compute Pipelines (check cache first) ---
        let entropy_pipeline = Self::get_or_create_compute_pipeline(
            device,
            &entropy_pipeline_layout,
            &entropy_shader_module,
            "main", // Entry point
            entropy_hash,
        )?;
        let propagation_pipeline = Self::get_or_create_compute_pipeline(
            device,
            &propagation_pipeline_layout,
            &propagation_shader_module,
            "main_propagate", // Entry point
            propagation_hash,
        )?;

        Ok(Self {
            entropy_pipeline,
            propagation_pipeline,
            entropy_bind_group_layout_0,
            entropy_bind_group_layout_1,
            propagation_bind_group_layout,
            entropy_workgroup_size,
            propagation_workgroup_size,
        })
    }

    // Helper function to get from cache or create compute pipeline
    fn get_or_create_compute_pipeline(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        module: &Arc<wgpu::ShaderModule>,
        entry_point: &str,
        source_hash: u64,
    ) -> Result<Arc<wgpu::ComputePipeline>, GpuError> {
        let key = PipelineCacheKey {
            source_hash,
            entry_point: entry_point.to_string(),
        };

        // Access the cache using the lazy_static macro
        let mut cache = COMPUTE_PIPELINE_CACHE
            .lock()
            .map_err(|e| GpuError::MutexError(e.to_string()))?;

        if let Some(pipeline) = cache.get(&key) {
            log::debug!(
                "Compute pipeline cache hit for entry point: {}",
                entry_point
            );
            return Ok(pipeline.clone());
        }

        log::debug!(
            "Compute pipeline cache miss for entry point: {}. Creating new pipeline.",
            entry_point
        );

        // Create compute pipeline
        let label = format!("Compute Pipeline: {}", entry_point);
        let desc = wgpu::ComputePipelineDescriptor {
            label: Some(&label),
            layout: Some(layout),
            module: &module,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        };

        let pipeline = device.create_compute_pipeline(&desc);
        let pipeline_arc = Arc::new(pipeline);

        cache.insert(key, pipeline_arc.clone());
        Ok(pipeline_arc)
    }

    pub fn create_propagation_bind_groups(
        &self,
        device: &wgpu::Device,
        resources: &PropagationBindingResources,
    ) -> [wgpu::BindGroup; 2] {
        let layout = self.get_propagation_bind_group_layout().unwrap(); // Assume layout exists

        let bind_group_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Propagation Bind Group 0"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.params_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.grid_possibilities_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.adjacency_rules_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.rule_weights_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: resources.worklist_bufs[0].as_entire_binding(),
                }, // Input worklist
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: resources.output_worklist_bufs[0].as_entire_binding(),
                }, // Output worklist
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: resources.worklist_count_bufs[0].as_entire_binding(),
                }, // Output count
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: resources.contradiction_flag_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: resources.contradiction_location_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: resources.pass_statistics_buf.as_entire_binding(),
                },
            ],
        });

        let bind_group_1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Propagation Bind Group 1"),
            layout, // Use the same layout
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.params_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.grid_possibilities_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.adjacency_rules_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.rule_weights_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: resources.worklist_bufs[1].as_entire_binding(),
                }, // Input worklist (ping-pong)
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: resources.output_worklist_bufs[1].as_entire_binding(),
                }, // Output worklist (ping-pong)
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: resources.worklist_count_bufs[1].as_entire_binding(),
                }, // Output count (ping-pong)
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: resources.contradiction_flag_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: resources.contradiction_location_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: resources.pass_statistics_buf.as_entire_binding(),
                }, // Statistics buffer is shared
            ],
        });

        [bind_group_0, bind_group_1]
    }

    pub fn get_propagation_bind_group_layout(&self) -> Result<&wgpu::BindGroupLayout, GpuError> {
        Ok(&self.propagation_bind_group_layout)
    }

    pub fn get_propagation_pipeline(
        &self,
        _supports_shader_i16: bool, // This param seems unused now?
    ) -> Result<&wgpu::ComputePipeline, GpuError> {
        // No longer selecting pipeline based on features here, selection happens at creation
        Ok(&self.propagation_pipeline)
    }
}
