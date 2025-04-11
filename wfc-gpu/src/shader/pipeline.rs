use log;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use wgpu;
// Added imports for caching
use once_cell::sync::Lazy;
use seahash::SeaHasher;
use std::hash::{Hash, Hasher};
// Import ShaderManager and related types from new location
use crate::shader::shaders::{ShaderManager, ShaderType};
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
