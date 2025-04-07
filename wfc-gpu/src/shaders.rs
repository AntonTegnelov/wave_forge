//! Module intended for managing WGSL shader source code.
//!
//! Currently, shaders are loaded directly via `include_str!` in the `pipeline.rs` module.
//! This module could be used in the future to centralize shader loading, potentially
//! allowing for dynamic loading or compilation if needed.

// wfc-gpu/src/shaders.rs
// Placeholder for shader loading/management logic

// pub const ENTROPY_SHADER_SRC: &str = include_str!("entropy.wgsl");
// pub const PROPAGATE_SHADER_SRC: &str = include_str!("propagate.wgsl");

//! Manages runtime loading and access to pre-compiled WGSL shader variants.
#![allow(unused_variables, dead_code)] // Allow unused items during development

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error; // Added for registry

/// Represents the main types of compute shaders used in the WFC algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderType {
    Entropy,
    Propagation,
    // Add other types like Initialization, Collapse if needed
}

#[derive(Error, Debug)]
pub enum ShaderError {
    #[error("Shader variant not found at path: {0}")]
    VariantNotFound(String),
    #[error("I/O error loading shader variant from '{path}': {source}")]
    IoError {
        path: String,
        source: std::io::Error,
    },
    #[error("Shader management feature not yet implemented: {0}")]
    NotImplemented(String),
    #[error("Build script did not specify OUT_DIR")]
    OutDirNotSet,
}

/// Represents metadata about a single shader component.
#[derive(Debug, Clone)] // Added Clone
pub struct ShaderComponentInfo {
    pub name: String,
    pub path: PathBuf,
    pub dependencies: Vec<String>, // Names of other components it depends on
    pub features: Vec<String>,     // Features this component provides or requires
                                   // Add other fields like description, author, etc. if needed
}

/// Manages access to pre-compiled shader variants and shader component metadata.
#[derive(Debug)]
pub struct ShaderManager {
    variants_dir: PathBuf,
    /// Registry mapping component names to their metadata.
    component_registry: HashMap<String, ShaderComponentInfo>,
    // Optional: Cache loaded shaders
    // loaded_shaders: HashMap<(ShaderType, Vec<String>), String>,
}

impl ShaderManager {
    /// Creates a new ShaderManager.
    /// Loads component metadata and prepares to load shader variants from OUT_DIR.
    pub fn new() -> Result<Self, ShaderError> {
        // Get the output directory set by the build script
        let out_dir = std::env::var("OUT_DIR").map_err(|_| ShaderError::OutDirNotSet)?;
        if out_dir.is_empty() {
            return Err(ShaderError::OutDirNotSet);
        }
        let variants_dir = Path::new(&out_dir).join("shaders").join("variants");

        // --- Load Component Registry (Placeholder) ---
        // TODO: Load this from src/shaders/components/registry.json
        let mut component_registry = HashMap::new();

        // Example placeholder components based on planned structure
        let components_base_path = PathBuf::from("src/shaders/components"); // Relative to crate root

        component_registry.insert(
            "utils".to_string(),
            ShaderComponentInfo {
                name: "utils".to_string(),
                path: components_base_path.join("utils.wgsl"), // Assuming utils is also a component now
                dependencies: vec![],
                features: vec![],
            },
        );
        component_registry.insert(
            "coords".to_string(),
            ShaderComponentInfo {
                name: "coords".to_string(),
                path: components_base_path.join("coords.wgsl"), // Assuming coords is also a component
                dependencies: vec!["utils".to_string()],
                features: vec![],
            },
        );
        component_registry.insert(
            "entropy_calculation".to_string(),
            ShaderComponentInfo {
                name: "entropy_calculation".to_string(),
                path: components_base_path.join("entropy_calculation.wgsl"),
                dependencies: vec!["utils".to_string(), "coords".to_string()],
                features: vec![],
            },
        );
        component_registry.insert(
            "worklist_management".to_string(),
            ShaderComponentInfo {
                name: "worklist_management".to_string(),
                path: components_base_path.join("worklist_management.wgsl"),
                dependencies: vec!["utils".to_string()],
                features: vec!["atomics".to_string()], // Example feature requirement
            },
        );
        // Add more components here...

        println!(
            "[ShaderManager] Initialized. Expecting shader variants in: {:?}. Loaded {} components (placeholder).",
            variants_dir,
            component_registry.len()
        );
        Ok(Self {
            variants_dir,
            component_registry,
        })
    }

    /// Loads the source code for a specific pre-compiled shader variant.
    ///
    /// # Arguments
    /// * `shader_type` - The type of shader pipeline (e.g., Entropy, Propagation).
    /// * `features` - A slice of feature strings (e.g., ["atomics"]) that the variant was compiled with.
    ///              An empty slice indicates the fallback/base variant.
    ///
    /// # Returns
    /// The WGSL source code as a String, or a ShaderError.
    pub fn load_shader_variant(
        &self,
        shader_type: ShaderType,
        features: &[&str],
    ) -> Result<String, ShaderError> {
        let variant_filename = Self::get_variant_filename(shader_type, features);
        let variant_path = self.variants_dir.join(&variant_filename);

        println!(
            "[ShaderManager] Attempting to load shader variant: {:?}",
            variant_path
        );

        // TODO: Implement actual file loading once build.rs generates variants.
        // For now, return a placeholder error or empty string.
        match std::fs::read_to_string(&variant_path) {
            Ok(source) => {
                println!("[ShaderManager] Successfully loaded: {}", variant_filename);
                Ok(source)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(
                ShaderError::VariantNotFound(variant_path.display().to_string()),
            ),
            Err(e) => Err(ShaderError::IoError {
                path: variant_path.display().to_string(),
                source: e,
            }),
        }
    }

    /// Constructs the expected filename for a shader variant.
    /// Example: "Entropy_atomics.wgsl" or "Propagation.wgsl" (for fallback)
    fn get_variant_filename(shader_type: ShaderType, features: &[&str]) -> String {
        let base_name = match shader_type {
            ShaderType::Entropy => "Entropy",
            ShaderType::Propagation => "Propagation",
        };
        if features.is_empty() {
            format!("{}.wgsl", base_name)
        } else {
            // Sort features for consistent naming
            let mut sorted_features = features.to_vec();
            sorted_features.sort_unstable();
            format!("{}_{}.wgsl", base_name, sorted_features.join("_"))
        }
    }

    /// Retrieves metadata for a specific shader component.
    pub fn get_component_info(&self, name: &str) -> Option<&ShaderComponentInfo> {
        self.component_registry.get(name)
    }

    /// Lists the names of all registered shader components.
    pub fn list_components(&self) -> Vec<String> {
        self.component_registry.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*; // Import items from outer module

    #[test]
    fn test_get_variant_filename_no_features() {
        let filename = ShaderManager::get_variant_filename(ShaderType::Entropy, &[]);
        assert_eq!(filename, "Entropy.wgsl");

        let filename = ShaderManager::get_variant_filename(ShaderType::Propagation, &[]);
        assert_eq!(filename, "Propagation.wgsl");
    }

    #[test]
    fn test_get_variant_filename_single_feature() {
        let filename = ShaderManager::get_variant_filename(ShaderType::Entropy, &["atomics"]);
        assert_eq!(filename, "Entropy_atomics.wgsl");
    }

    #[test]
    fn test_get_variant_filename_multiple_features_sorted() {
        let features = vec!["feature_z", "feature_a", "atomics"];
        let filename = ShaderManager::get_variant_filename(ShaderType::Propagation, &features);
        // Features should be sorted alphabetically in the filename
        assert_eq!(filename, "Propagation_atomics_feature_a_feature_z.wgsl");
    }

    // TODO: Add tests for ShaderManager::new() and load_shader_variant()
    // These tests will likely require mocking the filesystem or setting up
    // OUT_DIR with dummy files, potentially using a dedicated test setup.
    // #[test]
    // fn test_shader_manager_new_ok() { ... }
    // #[test]
    // fn test_shader_manager_new_out_dir_err() { ... }
    // #[test]
    // fn test_load_shader_variant_found() { ... }
    // #[test]
    // fn test_load_shader_variant_not_found() { ... }
}
