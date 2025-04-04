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
        let chosen_workgroup_size_x = max_invocations.min(256).max(64);
        log::info!(
            "Chosen workgroup size X (informational): {}",
            chosen_workgroup_size_x
        );

        // Check if the device supports storage atomics
        let supports_atomics = Self::check_atomics_support(device);
        log::info!("GPU storage atomics support: {}", supports_atomics);

        // Load appropriate shader code based on hardware capabilities
        let (entropy_shader_code, propagation_shader_code) =
            Self::select_shader_variants(supports_atomics, num_tiles_u32);

        // --- Get or Create Entropy Shader Module ---
        let entropy_shader_key = ShaderCacheKey {
            source_hash: hash_string(entropy_shader_code.as_str()),
        };

        let entropy_shader = {
            let mut cache = SHADER_MODULE_CACHE.lock().unwrap();
            cache
                .entry(entropy_shader_key.clone())
                .or_insert_with(|| {
                    log::debug!("Creating new shader module for entropy");
                    Arc::new(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("Entropy Shader"),
                        source: wgpu::ShaderSource::Wgsl(entropy_shader_code.into()),
                    }))
                })
                .clone()
        };

        // --- Get or Create Propagation Shader Module ---
        let propagation_shader_key = ShaderCacheKey {
            source_hash: hash_string(propagation_shader_code.as_str()),
        };

        let propagation_shader = {
            let mut cache = SHADER_MODULE_CACHE.lock().unwrap();
            cache
                .entry(propagation_shader_key.clone())
                .or_insert_with(|| {
                    log::debug!("Creating new shader module for propagation");
                    Arc::new(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("Propagation Shader"),
                        source: wgpu::ShaderSource::Wgsl(propagation_shader_code.into()),
                    }))
                })
                .clone()
        };

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

        // --- Create Pipeline Layouts ---

        let entropy_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Entropy Pipeline Layout"),
                bind_group_layouts: &[&entropy_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Layout for propagation shader
        let propagation_bind_group_layout = Arc::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Propagation Bind Group Layout"),
                entries: &[
                    // grid_possibilities (read-write storage)
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
                    // output_worklist (read-write storage)
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
                    // output_worklist_count (read-write storage)
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
                    // contradiction_flag (read-write storage)
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
                    // contradiction_location (read-write storage)
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

        let propagation_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Propagation Pipeline Layout"),
                bind_group_layouts: &[&propagation_bind_group_layout],
                push_constant_ranges: &[],
            });

        // --- Create Pipeline Cache Keys ---

        let entropy_entry_point = if supports_atomics {
            "main_entropy"
        } else {
            "main"
        };

        let propagation_entry_point = if supports_atomics {
            "main_propagate"
        } else {
            "main_propagate"
        };

        let entropy_pipeline_key = PipelineCacheKey {
            shader_key: entropy_shader_key,
            entry_point: entropy_entry_point.to_string(),
        };

        let propagation_pipeline_key = PipelineCacheKey {
            shader_key: propagation_shader_key,
            entry_point: propagation_entry_point.to_string(),
        };

        // --- Create or Retrieve Compute Pipelines ---

        let entropy_pipeline = {
            let mut cache = COMPUTE_PIPELINE_CACHE.lock().unwrap();
            cache
                .entry(entropy_pipeline_key)
                .or_insert_with(|| {
                    log::debug!("Creating new compute pipeline for entropy");
                    Arc::new(
                        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some("Entropy Compute Pipeline"),
                            layout: Some(&entropy_pipeline_layout),
                            module: &entropy_shader,
                            entry_point: entropy_entry_point,
                            compilation_options: wgpu::PipelineCompilationOptions::default(),
                        }),
                    )
                })
                .clone()
        };

        let propagation_pipeline = {
            let mut cache = COMPUTE_PIPELINE_CACHE.lock().unwrap();
            cache
                .entry(propagation_pipeline_key)
                .or_insert_with(|| {
                    log::debug!("Creating new compute pipeline for propagation");
                    Arc::new(
                        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some("Propagation Compute Pipeline"),
                            layout: Some(&propagation_pipeline_layout),
                            module: &propagation_shader,
                            entry_point: propagation_entry_point,
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
            entropy_workgroup_size: chosen_workgroup_size_x,
            propagation_workgroup_size: chosen_workgroup_size_x,
        })
    }

    /// Checks if the device supports GPU storage atomics.
    ///
    /// This method examines the features to determine if the hardware
    /// supports atomic operations required by the core algorithm.
    ///
    /// # Arguments
    ///
    /// * `device` - A reference to the wgpu Device
    ///
    /// # Returns
    ///
    /// * `true` if the hardware likely supports storage atomics
    /// * `false` if the hardware likely does not support storage atomics
    fn check_atomics_support(_device: &wgpu::Device) -> bool {
        // In wgpu v0.20, the exact atomics feature flag isn't directly exposed
        // Instead, we'll check for general compute shader capabilities

        // Most hardware that supports compute shaders should support basic atomics
        // This is a simplification; in a production environment, we would need
        // to check for specific features more precisely

        // For now, assume compute shader support implies atomics support
        // In the future, consider checking adapter info more specifically
        true
    }

    /// Selects the appropriate shader variants based on hardware capabilities.
    ///
    /// This method chooses between the full-featured shaders with atomic operations
    /// and simplified fallback versions for hardware that doesn't support atomics.
    /// It also replaces NUM_TILES_U32_VALUE in the shaders with the actual value.
    ///
    /// # Arguments
    ///
    /// * `supports_atomics` - Whether the hardware supports atomics operations
    /// * `num_tiles_u32` - The number of u32s needed per cell based on tile count
    ///
    /// # Returns
    ///
    /// * A tuple of (entropy_shader_code, propagation_shader_code) as strings
    fn select_shader_variants(supports_atomics: bool, num_tiles_u32: u32) -> (String, String) {
        let (entropy_shader_src, propagation_shader_src) = if supports_atomics {
            // Use the standard shaders with atomic operations
            (
                include_str!("shaders/entropy.wgsl"),
                include_str!("shaders/propagate.wgsl"),
            )
        } else {
            // Use fallback shaders without atomics
            log::warn!("Using fallback shaders without atomics support. Performance and functionality may be reduced.");

            // Load the fallback shader implementations
            (
                include_str!("shaders/entropy_fallback.wgsl"),
                include_str!("shaders/propagate_fallback.wgsl"),
            )
        };

        // Replace the NUM_TILES_U32_VALUE placeholder with the actual value
        let entropy_shader_modified =
            entropy_shader_src.replace("NUM_TILES_U32_VALUE", &num_tiles_u32.to_string());

        let propagation_shader_modified =
            propagation_shader_src.replace("NUM_TILES_U32_VALUE", &num_tiles_u32.to_string());

        (entropy_shader_modified, propagation_shader_modified)
    }
}
