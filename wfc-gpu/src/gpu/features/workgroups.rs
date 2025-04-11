//! Module for detecting and handling workgroup capabilities in GPU shaders.
//!
//! This module provides utilities for determining optimal workgroup sizes
//! and checking support for workgroup-related features like shared memory
//! and barriers.

use super::{GpuCapabilities, GpuFeature, WorkgroupLimits};

/// Struct for accessing workgroup-specific capabilities.
#[derive(Debug, Clone)]
pub struct WorkgroupSupport {
    limits: WorkgroupLimits,
    has_shared_memory: bool,
    has_barriers: bool,
    has_subgroups: bool,
}

impl WorkgroupSupport {
    /// Create a new WorkgroupSupport from the general GPU capabilities.
    pub fn new(capabilities: &GpuCapabilities) -> Self {
        let has_shared_memory = capabilities.supports(GpuFeature::WorkgroupSharedMemory);
        let has_barriers = capabilities.supports(GpuFeature::WorkgroupBarriers);
        let has_subgroups = capabilities.supports(GpuFeature::Subgroups);

        Self {
            limits: capabilities.workgroup_limits,
            has_shared_memory,
            has_barriers,
            has_subgroups,
        }
    }

    /// Get the workgroup limits.
    pub fn limits(&self) -> WorkgroupLimits {
        self.limits
    }

    /// Check if workgroup shared memory is supported.
    pub fn has_shared_memory(&self) -> bool {
        self.has_shared_memory
    }

    /// Check if workgroup barriers are supported.
    pub fn has_barriers(&self) -> bool {
        self.has_barriers
    }

    /// Check if subgroups are supported.
    pub fn has_subgroups(&self) -> bool {
        self.has_subgroups
    }

    /// Get the recommended workgroup size for the given operation type.
    pub fn recommended_workgroup_size(&self, operation_type: OperationType) -> (u32, u32, u32) {
        // Start with maximum workgroup size
        let base_size = match operation_type {
            OperationType::General => 64,
            OperationType::EntropyCalculation => 256,
            OperationType::Propagation => 64,
        };

        // Calculate size based on device limits
        let size_x = base_size.min(self.limits.max_size_x);
        let size_y = 1.min(self.limits.max_size_y);
        let size_z = 1.min(self.limits.max_size_z);

        // Ensure total invocations doesn't exceed limit
        if size_x * size_y * size_z > self.limits.max_invocations {
            // Simple fallback - just use 1D workgroup
            let max_size = self.limits.max_invocations.min(self.limits.max_size_x);
            return (max_size, 1, 1);
        }

        (size_x, size_y, size_z)
    }

    /// Get the maximum workgroup size supported by the device.
    pub fn max_workgroup_size(&self) -> u32 {
        self.limits.max_invocations
    }

    /// Get the maximum shared memory size per workgroup.
    pub fn max_shared_memory(&self) -> u32 {
        self.limits.max_shared_memory_size
    }

    /// Get appropriate shader defines for workgroup features.
    pub fn shader_defines(&self) -> Vec<String> {
        let mut defines = Vec::new();

        // Shared memory support
        if self.has_shared_memory {
            defines.push("ENABLE_WORKGROUP_SHARED_MEMORY=1".to_string());
        } else {
            defines.push("ENABLE_WORKGROUP_SHARED_MEMORY=0".to_string());
        }

        // Barrier support
        if self.has_barriers {
            defines.push("ENABLE_WORKGROUP_BARRIERS=1".to_string());
        } else {
            defines.push("ENABLE_WORKGROUP_BARRIERS=0".to_string());
        }

        // Subgroup support
        if self.has_subgroups {
            defines.push("ENABLE_SUBGROUPS=1".to_string());
        } else {
            defines.push("ENABLE_SUBGROUPS=0".to_string());
        }

        // Add limits
        defines.push(format!(
            "MAX_WORKGROUP_SIZE={}",
            self.limits.max_invocations
        ));
        defines.push(format!("MAX_WORKGROUP_SIZE_X={}", self.limits.max_size_x));
        defines.push(format!("MAX_WORKGROUP_SIZE_Y={}", self.limits.max_size_y));
        defines.push(format!("MAX_WORKGROUP_SIZE_Z={}", self.limits.max_size_z));
        defines.push(format!(
            "MAX_WORKGROUP_SHARED_MEMORY={}",
            self.limits.max_shared_memory_size
        ));

        defines
    }
}

/// The type of operation being performed, used to select appropriate workgroup sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationType {
    /// General compute operations.
    General,
    /// Entropy calculation operations.
    EntropyCalculation,
    /// Constraint propagation operations.
    Propagation,
}

/// Strategy to use when workgroup capabilities are limited.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkgroupFallbackStrategy {
    /// Use smaller workgroups with more dispatches.
    SmallerWorkgroups,
    /// Avoid using workgroup shared memory.
    AvoidSharedMemory,
    /// Use more global memory instead of shared memory.
    UseGlobalMemory,
    /// Process data in tiles to fit within workgroup limits.
    TiledProcessing,
}
