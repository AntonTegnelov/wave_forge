use crate::GpuError;
use log;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use wgpu;
// Added imports for caching
use once_cell::sync::Lazy;
use seahash::SeaHasher;
use std::hash::{Hash, Hasher};

// --- Cache Definitions ---

// Key for shader module cache: based on shader source code
#[derive(PartialEq, Eq, Hash, Clone)]
struct ShaderCacheKey {
    source_hash: u64,
}

// Key for pipeline cache: includes shader details and configuration
#[derive(PartialEq, Eq, Hash, Clone)]
struct PipelineCacheKey {
    shader_key: ShaderCacheKey,
    entry_point: String,
}

// Static caches using Lazy and Mutex for thread-safe initialization and access
static SHADER_MODULE_CACHE: Lazy<Mutex<HashMap<ShaderCacheKey, Arc<wgpu::ShaderModule>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

static COMPUTE_PIPELINE_CACHE: Lazy<Mutex<HashMap<PipelineCacheKey, Arc<wgpu::ComputePipeline>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

// --- Helper Function to Hash Strings ---
fn hash_string(s: &str) -> u64 {
    let mut hasher = SeaHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

// --- TODO: Shader Source Loading and Compilation (Future Responsibility of ShaderCompiler) ---

// Placeholder function for loading shader components
// In the future, this will interact with ShaderRegistry and ShaderCompiler
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

// Placeholder for future shader assembly/compilation
// This would take component sources and assemble them based on features.
fn compile_shader(
    device: &wgpu::Device,
    shader_name: &str,
    features: &[&str],
    _num_tiles_u32: u32, // May be needed for specialization in future compiler
) -> Result<Arc<wgpu::ShaderModule>, GpuError> {
    // 1. Load/Assemble source using a future ShaderCompiler based on name and features
    let source_code = load_shader_source(shader_name, features)?;

    // TODO: Apply specialization constants (like NUM_TILES_U32_VALUE) using the compiler
    // let processed_source = future_shader_compiler.specialize(&source_code, num_tiles_u32);

    // 2. Check cache
    let shader_key = ShaderCacheKey {
        source_hash: hash_string(&source_code),
    };
    let mut cache = SHADER_MODULE_CACHE
        .lock()
        .map_err(|e| GpuError::MutexError(e.to_string()))?;

    if let Some(module) = cache.get(&shader_key) {
        log::debug!("Shader cache hit for {}", shader_name);
        return Ok(module.clone());
    }

    // 3. Create module if not in cache
    log::debug!(
        "Shader cache miss for {}. Creating new module.",
        shader_name
    );
    let shader_module = Arc::new(device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(shader_name),
        source: wgpu::ShaderSource::Wgsl(source_code.into()),
    }));

    cache.insert(shader_key, shader_module.clone());
    Ok(shader_module)
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
    pub entropy_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    /// The layout describing the binding structure for the propagation pipeline's bind group.
    /// Required for creating bind groups compatible with `propagation_pipeline`.
    pub propagation_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    /// Dynamically determined optimal workgroup size (X-dimension) for the entropy shader.
    pub entropy_workgroup_size: u32,
    /// Dynamically determined optimal workgroup size (X-dimension) for the propagation shader.
    pub propagation_workgroup_size: u32,
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
    ///
    /// # Returns
    ///
    /// * `Ok(Self)` containing the initialized `ComputePipelines`.
    /// * `Err(GpuError)` if shader loading, compilation, or pipeline creation fails.
    pub fn new(device: &wgpu::Device, num_tiles_u32: u32) -> Result<Self, GpuError> {
        // Query device limits
        let limits = device.limits();
        let max_invocations = limits.max_compute_invocations_per_workgroup;
        log::debug!(
            "GPU max compute invocations per workgroup: {}",
            max_invocations
        );

        // Determine workgroup size - NOTE: This is now informational only, not used for specialization
        let chosen_workgroup_size_x = max_invocations.min(64).max(64); // Assuming 64 in shaders
        log::info!("Assumed workgroup size X: {}", chosen_workgroup_size_x);

        // Check feature support (e.g., atomics)
        // Simplification: Check for features often associated with robust compute capabilities.
        // A more precise check might involve specific atomic feature flags if available and stable across wgpu versions.
        let supports_atomics = device
            .features()
            .contains(wgpu::Features::BUFFER_BINDING_ARRAY)
            || device
                .features()
                .contains(wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY);
        log::info!(
            "GPU supports potentially relevant features (atomics proxy): {}",
            supports_atomics
        );
        let features = if supports_atomics {
            vec!["atomics"]
        } else {
            vec![]
        };

        // --- Compile Shaders using the (placeholder) compile_shader function ---
        let entropy_shader = compile_shader(device, "entropy", &features, num_tiles_u32)?;
        let propagation_shader = compile_shader(device, "propagate", &features, num_tiles_u32)?;

        // --- Define Bind Group Layouts (Largely remains the same, but check bindings) ---

        // Layout for entropy shader (ensure bindings match component needs)
        // TODO: Bindings might need adjustment based on final compiled shader structure
        let entropy_bind_group_layout = Arc::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Entropy Bind Group Layout"),
                entries: &[
                    // @group(0) @binding(0) grid_possibilities (read-only storage)
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
                    // @group(0) @binding(1) entropy_output (write-only storage)
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
                    // @group(0) @binding(2) params (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None, // Use params struct size
                        },
                        count: None,
                    },
                    // @group(0) @binding(3) min_entropy_info (read-write storage, atomic vec2<u32>)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false }, // Atomics require write
                            has_dynamic_offset: false,
                            min_binding_size: Some(std::num::NonZeroU64::new(8).unwrap()), // vec2<u32>
                        },
                        count: None,
                    },
                ],
            },
        ));

        // Layout for propagation shader (ensure bindings match component needs)
        // TODO: Bindings might need adjustment based on final compiled shader structure
        let propagation_bind_group_layout = Arc::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Propagation Bind Group Layout"),
                entries: &[
                    // @group(0) @binding(0) params (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None, // Use params struct size
                        },
                        count: None,
                    },
                    // @group(0) @binding(1) grid_possibilities (read-write storage, atomic)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false }, // Atomics require write
                            has_dynamic_offset: false,
                            min_binding_size: Some(std::num::NonZeroU64::new(4).unwrap()),
                        },
                        count: None,
                    },
                    // @group(0) @binding(2) adjacency_rules (read-only storage)
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
                    // @group(0) @binding(3) rule_weights (read-only storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: Some(std::num::NonZeroU64::new(4).unwrap()), // Can be empty, but needs binding
                        },
                        count: None,
                    },
                    // @group(0) @binding(4) worklist (read-only storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: Some(std::num::NonZeroU64::new(4).unwrap()),
                        },
                        count: None,
                    },
                    // @group(0) @binding(5) output_worklist (read-write storage, atomic)
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false }, // Atomics require write
                            has_dynamic_offset: false,
                            min_binding_size: Some(std::num::NonZeroU64::new(4).unwrap()),
                        },
                        count: None,
                    },
                    // @group(0) @binding(6) output_worklist_count (read-write storage, atomic)
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false }, // Atomics require write
                            has_dynamic_offset: false,
                            min_binding_size: Some(std::num::NonZeroU64::new(4).unwrap()), // atomic<u32>
                        },
                        count: None,
                    },
                    // @group(0) @binding(7) contradiction_flag (read-write storage, atomic)
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false }, // Atomics require write
                            has_dynamic_offset: false,
                            min_binding_size: Some(std::num::NonZeroU64::new(4).unwrap()), // atomic<u32>
                        },
                        count: None,
                    },
                    // @group(0) @binding(8) contradiction_location (read-write storage, atomic)
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false }, // Atomics require write
                            has_dynamic_offset: false,
                            min_binding_size: Some(std::num::NonZeroU64::new(4).unwrap()), // atomic<u32>
                        },
                        count: None,
                    },
                    // @group(0) @binding(9) pass_statistics (read-write storage, atomic)
                    wgpu::BindGroupLayoutEntry {
                        binding: 9,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false }, // Atomics require write
                            has_dynamic_offset: false,
                            min_binding_size: Some(std::num::NonZeroU64::new(16).unwrap()), // array<atomic<u32>, 4> ?
                        },
                        count: None,
                    },
                ],
            },
        ));

        // --- Create Pipeline Layouts (Remains the same) ---
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

        // --- Create Compute Pipelines using cached/compiled modules ---
        // NOTE: Specialization constants are currently embedded in the placeholder WGSL loading.
        //       A real shader compiler would handle applying these.
        let entropy_entry_point = "main_entropy"; // Assuming this is the entry point in assembled shader
        let entropy_pipeline = {
            let pipeline_key = PipelineCacheKey {
                shader_key: ShaderCacheKey {
                    source_hash: hash_string(&load_shader_source("entropy", &features)?),
                }, // Re-hash potentially specialized source
                entry_point: entropy_entry_point.to_string(),
            };
            let mut cache = COMPUTE_PIPELINE_CACHE
                .lock()
                .map_err(|e| GpuError::MutexError(e.to_string()))?;
            cache
                .entry(pipeline_key)
                .or_insert_with(|| {
                    log::debug!("Creating new compute pipeline for entropy");
                    Arc::new(
                        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some("Entropy Compute Pipeline"),
                            layout: Some(&entropy_pipeline_layout),
                            module: &entropy_shader,
                            entry_point: entropy_entry_point,
                            // TODO: Pass specialization constants here via `constants` field
                            //       if the shader compiler doesn't handle embedding them.
                            compilation_options: wgpu::PipelineCompilationOptions::default(),
                        }),
                    )
                })
                .clone()
        };

        let propagation_entry_point = "main_propagate"; // Assuming this is the entry point
        let propagation_pipeline = {
            let pipeline_key = PipelineCacheKey {
                shader_key: ShaderCacheKey {
                    source_hash: hash_string(&load_shader_source("propagate", &features)?),
                }, // Re-hash potentially specialized source
                entry_point: propagation_entry_point.to_string(),
            };
            let mut cache = COMPUTE_PIPELINE_CACHE
                .lock()
                .map_err(|e| GpuError::MutexError(e.to_string()))?;
            cache
                .entry(pipeline_key)
                .or_insert_with(|| {
                    log::debug!("Creating new compute pipeline for propagation");
                    Arc::new(
                        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some("Propagation Compute Pipeline"),
                            layout: Some(&propagation_pipeline_layout),
                            module: &propagation_shader,
                            entry_point: propagation_entry_point,
                            // TODO: Pass specialization constants here
                            compilation_options: wgpu::PipelineCompilationOptions::default(),
                        }),
                    )
                })
                .clone()
        };

        Ok(Self {
            entropy_pipeline,
            propagation_pipeline,
            entropy_bind_group_layout,
            propagation_bind_group_layout,
            entropy_workgroup_size: chosen_workgroup_size_x, // Use assumed size
            propagation_workgroup_size: chosen_workgroup_size_x, // Use assumed size
        })
    }

    pub fn create_propagation_bind_groups(
        &self,
        device: &wgpu::Device,
        grid_possibilities_buf: &wgpu::Buffer,
        adjacency_rules_buf: &wgpu::Buffer,
        rule_weights_buf: &wgpu::Buffer,
        worklist_bufs: &[wgpu::Buffer; 2],
        output_worklist_bufs: &[wgpu::Buffer; 2],
        params_uniform_buf: &wgpu::Buffer,
        worklist_count_bufs: &[wgpu::Buffer; 2],
        contradiction_flag_buf: &wgpu::Buffer,
        contradiction_location_buf: &wgpu::Buffer,
        pass_statistics_buf: &wgpu::Buffer,
    ) -> [wgpu::BindGroup; 2] {
        let layout = self.get_propagation_bind_group_layout().unwrap(); // Assume layout exists

        let bind_group_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Propagation Bind Group 0"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grid_possibilities_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: adjacency_rules_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: rule_weights_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: worklist_bufs[0].as_entire_binding(),
                }, // Input worklist
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: output_worklist_bufs[0].as_entire_binding(),
                }, // Output worklist
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: worklist_count_bufs[0].as_entire_binding(),
                }, // Output count
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: contradiction_flag_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: contradiction_location_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: pass_statistics_buf.as_entire_binding(),
                },
            ],
        });

        let bind_group_1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Propagation Bind Group 1"),
            layout, // Use the same layout
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grid_possibilities_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: adjacency_rules_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: rule_weights_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: worklist_bufs[1].as_entire_binding(),
                }, // Input worklist (ping-pong)
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: output_worklist_bufs[1].as_entire_binding(),
                }, // Output worklist (ping-pong)
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: worklist_count_bufs[1].as_entire_binding(),
                }, // Output count (ping-pong)
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: contradiction_flag_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: contradiction_location_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: pass_statistics_buf.as_entire_binding(),
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
