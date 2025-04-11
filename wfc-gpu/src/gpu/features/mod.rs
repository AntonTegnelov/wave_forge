//! Feature detection system for GPU capabilities.
//!
//! This module provides a structured way to detect and query GPU capabilities
//! that are relevant for shader compilation and algorithm selection.

use crate::GpuError;
use std::collections::HashSet;
use std::fmt::{Debug, Display};
use wgpu::{Adapter, Features as WgpuFeatures, Limits as WgpuLimits};

/// A GPU feature capability that can be used to guide shader compilation and algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuFeature {
    /// Support for compute shaders
    ComputeShaders,
    /// Support for atomic operations in shaders
    Atomics,
    /// Support for 64-bit atomic operations
    Atomics64Bit,
    /// Support for workgroup shared memory
    WorkgroupSharedMemory,
    /// Support for thread synchronization within workgroups
    WorkgroupBarriers,
    /// Support for subgroups (SIMD within workgroups)
    Subgroups,
    /// Support for 32-bit indices
    IndexUint32,
    /// Support for storage buffers (read/write)
    StorageBuffers,
    /// Support for storage textures (read/write)
    StorageTextures,
}

impl Display for GpuFeature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuFeature::ComputeShaders => write!(f, "compute_shader"),
            GpuFeature::Atomics => write!(f, "atomic_operations"),
            GpuFeature::Atomics64Bit => write!(f, "atomic_operations_64bit"),
            GpuFeature::WorkgroupSharedMemory => write!(f, "workgroup_shared_memory"),
            GpuFeature::WorkgroupBarriers => write!(f, "workgroup_barriers"),
            GpuFeature::Subgroups => write!(f, "subgroups"),
            GpuFeature::IndexUint32 => write!(f, "index_uint32"),
            GpuFeature::StorageBuffers => write!(f, "storage_buffers"),
            GpuFeature::StorageTextures => write!(f, "storage_textures"),
        }
    }
}

/// A utility struct representing the workgroup limits for a GPU.
#[derive(Debug, Clone, Copy)]
pub struct WorkgroupLimits {
    /// Maximum number of invocations in a workgroup (X * Y * Z).
    pub max_invocations: u32,
    /// Maximum size of a workgroup in the X dimension.
    pub max_size_x: u32,
    /// Maximum size of a workgroup in the Y dimension.
    pub max_size_y: u32,
    /// Maximum size of a workgroup in the Z dimension.
    pub max_size_z: u32,
    /// Maximum amount of shared memory per workgroup in bytes.
    pub max_shared_memory_size: u32,
}

/// A struct containing information about GPU capabilities.
#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    /// The features supported by the GPU.
    pub supported_features: HashSet<GpuFeature>,
    /// The workgroup limits of the GPU.
    pub workgroup_limits: WorkgroupLimits,
    /// The maximum buffer size supported by the GPU.
    pub max_buffer_size: u64,
    /// The maximum binding count supported by the GPU.
    pub max_bindings: u32,
    /// Whether the device is considered a high-performance device.
    pub is_high_performance: bool,
    /// A free-form name for the device type (integrated, discrete, etc.)
    pub device_type: String,
}

impl Default for GpuCapabilities {
    fn default() -> Self {
        Self {
            supported_features: HashSet::new(),
            workgroup_limits: WorkgroupLimits {
                max_invocations: 256,
                max_size_x: 256,
                max_size_y: 256,
                max_size_z: 64,
                max_shared_memory_size: 16384, // 16 KB
            },
            max_buffer_size: 1 << 30, // 1 GB
            max_bindings: 8,
            is_high_performance: false,
            device_type: "unknown".to_string(),
        }
    }
}

impl GpuCapabilities {
    /// Check if a specific feature is supported.
    pub fn supports(&self, feature: GpuFeature) -> bool {
        self.supported_features.contains(&feature)
    }

    /// Check if all specified features are supported.
    pub fn supports_all(&self, features: &[GpuFeature]) -> bool {
        features.iter().all(|f| self.supports(*f))
    }

    /// Check if any of the specified features are supported.
    pub fn supports_any(&self, features: &[GpuFeature]) -> bool {
        features.iter().any(|f| self.supports(*f))
    }

    /// Get a string set of supported feature names.
    pub fn feature_names(&self) -> HashSet<String> {
        self.supported_features
            .iter()
            .map(|f| f.to_string())
            .collect()
    }

    /// Create a GpuCapabilities instance from a wgpu Adapter.
    pub fn from_adapter(adapter: &Adapter) -> Result<Self, GpuError> {
        // Get adapter info
        let info = adapter.get_info();

        // Check if we have access to adapter features and limits
        let features = adapter.features();
        let limits = adapter.limits();

        // Map wgpu features to our GpuFeature enum
        let mut supported_features = HashSet::new();

        // Core features
        supported_features.insert(GpuFeature::ComputeShaders);
        supported_features.insert(GpuFeature::StorageBuffers);

        // Map specific wgpu features
        if features.contains(WgpuFeatures::SHADER_INT16) {
            // This is just an example - actual mapping would depend on our needs
        }

        // Atomics
        if features.contains(WgpuFeatures::SHADER_ATOMIC_INT32) {
            supported_features.insert(GpuFeature::Atomics);
        }

        // Workgroup features
        supported_features.insert(GpuFeature::WorkgroupSharedMemory);
        supported_features.insert(GpuFeature::WorkgroupBarriers);

        // Subgroups
        if features.contains(WgpuFeatures::SHADER_SUBGROUPS) {
            supported_features.insert(GpuFeature::Subgroups);
        }

        // 32-bit indices
        supported_features.insert(GpuFeature::IndexUint32);

        // Create workgroup limits
        let workgroup_limits = WorkgroupLimits {
            max_invocations: limits.max_compute_invocations_per_workgroup,
            max_size_x: limits.max_compute_workgroup_size_x,
            max_size_y: limits.max_compute_workgroup_size_y,
            max_size_z: limits.max_compute_workgroup_size_z,
            max_shared_memory_size: limits.max_compute_workgroup_storage_size,
        };

        Ok(Self {
            supported_features,
            workgroup_limits,
            max_buffer_size: limits.max_buffer_size as u64,
            max_bindings: limits.max_storage_buffers_per_shader_stage,
            is_high_performance: info.device_type.is_discrete(),
            device_type: format!("{:?}", info.device_type),
        })
    }
}

// Re-export submodules
pub mod atomics;
pub mod workgroups;

// Re-export key types
pub use atomics::AtomicsSupport;
pub use workgroups::WorkgroupSupport;
