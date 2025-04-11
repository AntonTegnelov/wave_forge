//! Module for detecting and handling atomic operation support in GPU shaders.
//!
//! This module provides utilities for checking whether a GPU supports atomic operations,
//! which are crucial for concurrent workgroup processing in WGPU shaders.

use super::{GpuCapabilities, GpuFeature};

/// Enumeration of different levels of atomics support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicsLevel {
    /// No atomic operation support.
    None,
    /// Basic 32-bit atomic operations (add, exchange, compare-exchange).
    Basic,
    /// Extended 32-bit atomic operations (min, max, and, or, xor).
    Extended,
    /// Support for 64-bit atomic operations.
    Atomic64Bit,
    /// Support for atomic operations on storage textures.
    AtomicStorage,
}

/// Struct for accessing and checking atomics-specific capabilities.
#[derive(Debug, Clone)]
pub struct AtomicsSupport {
    level: AtomicsLevel,
    has_atomics: bool,
}

impl AtomicsSupport {
    /// Create a new AtomicsSupport from the general GPU capabilities.
    pub fn new(capabilities: &GpuCapabilities) -> Self {
        let has_atomics = capabilities.supports(GpuFeature::Atomics);
        let has_64bit = capabilities.supports(GpuFeature::Atomics64Bit);

        let level = if !has_atomics {
            AtomicsLevel::None
        } else if has_64bit {
            AtomicsLevel::Atomic64Bit
        } else {
            // We assume Extended support if basic atomics are supported
            // This is a simplification - in a real implementation, you might
            // want to have more detailed feature detection
            AtomicsLevel::Extended
        };

        Self { level, has_atomics }
    }

    /// Check if any atomic operations are supported.
    pub fn has_atomics(&self) -> bool {
        self.has_atomics
    }

    /// Get the atomics support level.
    pub fn level(&self) -> AtomicsLevel {
        self.level
    }

    /// Check if 64-bit atomics are supported.
    pub fn has_64bit_atomics(&self) -> bool {
        self.level == AtomicsLevel::Atomic64Bit
    }

    /// Get a fallback strategy when atomics aren't supported.
    pub fn fallback_strategy(&self) -> AtomicsFallbackStrategy {
        if self.has_atomics() {
            AtomicsFallbackStrategy::UseAtomics
        } else {
            AtomicsFallbackStrategy::SerialProcessing
        }
    }

    /// Get the set of WGSL shader defines to enable for atomic operations.
    pub fn shader_defines(&self) -> Vec<String> {
        let mut defines = Vec::new();

        if self.has_atomics() {
            defines.push("ENABLE_ATOMICS=1".to_string());

            match self.level {
                AtomicsLevel::None => {}
                AtomicsLevel::Basic => {
                    defines.push("ATOMICS_LEVEL=1".to_string());
                }
                AtomicsLevel::Extended => {
                    defines.push("ATOMICS_LEVEL=2".to_string());
                }
                AtomicsLevel::Atomic64Bit => {
                    defines.push("ATOMICS_LEVEL=3".to_string());
                    defines.push("ENABLE_64BIT_ATOMICS=1".to_string());
                }
                AtomicsLevel::AtomicStorage => {
                    defines.push("ATOMICS_LEVEL=4".to_string());
                    defines.push("ENABLE_STORAGE_ATOMICS=1".to_string());
                }
            }
        } else {
            defines.push("ENABLE_ATOMICS=0".to_string());
        }

        defines
    }
}

/// Strategy to use when atomics are not available.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicsFallbackStrategy {
    /// Use atomic operations where available.
    UseAtomics,
    /// Fall back to serial processing (slower but guaranteed to work).
    SerialProcessing,
    /// Use a hybrid approach with synchronized workgroups.
    WorkgroupSynchronization,
    /// Try to emulate atomics with other operations (may not be reliable).
    Emulation,
}
