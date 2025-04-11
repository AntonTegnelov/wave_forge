// Shader module for managing GPU shaders and pipeline creation
//
// This module provides functionality for:
// 1. Loading and compiling shader source code
// 2. Managing shader components and variants
// 3. Creating and caching GPU pipelines
// 4. Supporting feature detection and shader specialization

pub mod pipeline;
pub mod shader_compiler;
pub mod shader_registry;

// Re-export the main shader manager
mod shaders;
pub use shaders::*;

// Public re-export of ShaderType enum for use by clients
pub use shaders::ShaderType;
