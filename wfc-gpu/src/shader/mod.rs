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
pub mod shaders;

// Re-export key types for easier access
pub use self::shader_registry::ShaderComponent;
pub use self::shaders::ShaderError;
pub use self::shaders::ShaderManager;
pub use self::shaders::ShaderType;
