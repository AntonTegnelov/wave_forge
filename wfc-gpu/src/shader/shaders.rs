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
use std::collections::HashSet;
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
#[derive(Debug, Clone)]
pub struct ShaderComponentInfo {
    pub name: String,
    pub path: PathBuf,
    pub dependencies: Vec<String>,
    pub features: Vec<String>,
    pub version: String,
    pub provided_features: Vec<String>,
    pub gpu_capabilities: Vec<String>,
}

/// Manages access to pre-compiled shader variants and shader component metadata.
#[derive(Debug)]
pub struct ShaderManager {
    variants_dir: PathBuf,
    /// Registry mapping component names to their metadata.
    component_registry: HashMap<String, ShaderComponentInfo>,
    /// Path to the component registry file
    registry_path: PathBuf,
    /// Cache loaded shaders for better performance
    loaded_shaders: HashMap<(ShaderType, Vec<String>), String>,
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

        // Load registry from JSON file
        let registry_path = PathBuf::from("src/shader/shaders/components/registry.json");
        let component_registry = Self::load_component_registry(&registry_path)?;

        println!(
            "[ShaderManager] Initialized. Found {} shader components. Expecting variants in: {:?}",
            component_registry.len(),
            variants_dir
        );

        Ok(Self {
            variants_dir,
            component_registry,
            registry_path,
            loaded_shaders: HashMap::new(),
        })
    }

    /// Loads the component registry from the specified JSON file.
    fn load_component_registry(
        registry_path: &Path,
    ) -> Result<HashMap<String, ShaderComponentInfo>, ShaderError> {
        let registry_content = match std::fs::read_to_string(registry_path) {
            Ok(content) => content,
            Err(e) => {
                println!(
                    "[ShaderManager] Warning: Could not read registry file: {}",
                    e
                );
                println!("[ShaderManager] Using placeholder registry data");
                return Ok(Self::create_placeholder_registry());
            }
        };

        #[derive(serde::Deserialize)]
        struct RegistryFile {
            components: HashMap<String, ComponentData>,
        }

        #[derive(serde::Deserialize)]
        struct ComponentData {
            path: String,
            #[serde(default)]
            version: String,
            #[serde(default)]
            dependencies: Vec<String>,
            #[serde(default)]
            required_features: Vec<String>,
            #[serde(default)]
            provided_features: Vec<String>,
            #[serde(default)]
            gpu_capabilities: Vec<String>,
        }

        // Parse the JSON
        let registry: RegistryFile = match serde_json::from_str(&registry_content) {
            Ok(reg) => reg,
            Err(e) => {
                println!("[ShaderManager] Error parsing registry JSON: {}", e);
                println!("[ShaderManager] Using placeholder registry data");
                return Ok(Self::create_placeholder_registry());
            }
        };

        // Convert to our internal format
        let mut component_registry = HashMap::new();
        for (name, data) in registry.components {
            component_registry.insert(
                name.clone(),
                ShaderComponentInfo {
                    name: name.clone(),
                    path: PathBuf::from(&data.path),
                    dependencies: data.dependencies,
                    features: data.required_features,
                    version: data.version,
                    provided_features: data.provided_features,
                    gpu_capabilities: data.gpu_capabilities,
                },
            );
        }

        Ok(component_registry)
    }

    /// Creates a placeholder registry for when the real registry can't be loaded
    fn create_placeholder_registry() -> HashMap<String, ShaderComponentInfo> {
        let mut registry = HashMap::new();
        let components_base_path = PathBuf::from("src/shader/shaders/components");

        registry.insert(
            "Utils".to_string(),
            ShaderComponentInfo {
                name: "Utils".to_string(),
                path: PathBuf::from("src/shaders/utils.wgsl"),
                dependencies: vec![],
                features: vec![],
                version: "1.0.0".to_string(),
                provided_features: vec!["bit_manipulation".to_string(), "math_utils".to_string()],
                gpu_capabilities: vec![],
            },
        );

        // Add other basic components
        registry.insert(
            "Coords".to_string(),
            ShaderComponentInfo {
                name: "Coords".to_string(),
                path: PathBuf::from("src/shaders/coords.wgsl"),
                dependencies: vec![],
                features: vec![],
                version: "1.0.0".to_string(),
                provided_features: vec!["coordinate_utils".to_string()],
                gpu_capabilities: vec![],
            },
        );

        // Add entropy components
        registry.insert(
            "EntropyCalculation".to_string(),
            ShaderComponentInfo {
                name: "EntropyCalculation".to_string(),
                path: components_base_path.join("entropy_calculation.wgsl"),
                dependencies: vec!["Utils".to_string(), "Coords".to_string()],
                features: vec![
                    "bit_manipulation".to_string(),
                    "coordinate_utils".to_string(),
                ],
                version: "1.0.0".to_string(),
                provided_features: vec!["entropy_calculation".to_string()],
                gpu_capabilities: vec![],
            },
        );

        // Add propagation components
        registry.insert(
            "DirectPropagation".to_string(),
            ShaderComponentInfo {
                name: "DirectPropagation".to_string(),
                path: components_base_path.join("propagation/direct.wgsl"),
                dependencies: vec!["Utils".to_string(), "Coords".to_string()],
                features: vec![
                    "bit_manipulation".to_string(),
                    "coordinate_utils".to_string(),
                ],
                version: "1.0.0".to_string(),
                provided_features: vec!["direct_propagation".to_string()],
                gpu_capabilities: vec!["compute_shader".to_string()],
            },
        );

        registry
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
        &mut self,
        shader_type: ShaderType,
        features: &[&str],
    ) -> Result<String, ShaderError> {
        // Sort features for consistency
        let mut sorted_features: Vec<String> = features.iter().map(|&s| s.to_string()).collect();
        sorted_features.sort();

        // Check cache first
        let cache_key = (shader_type, sorted_features.clone());
        if let Some(cached_shader) = self.loaded_shaders.get(&cache_key) {
            return Ok(cached_shader.clone());
        }

        let variant_filename = Self::get_variant_filename(shader_type, features);
        let variant_path = self.variants_dir.join(&variant_filename);

        println!(
            "[ShaderManager] Attempting to load shader variant: {:?}",
            variant_path
        );

        // First try to load from OUT_DIR (build-generated variants)
        match std::fs::read_to_string(&variant_path) {
            Ok(source) => {
                println!("[ShaderManager] Successfully loaded: {}", variant_filename);
                // Add to cache
                self.loaded_shaders.insert(cache_key, source.clone());
                return Ok(source);
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                println!("[ShaderManager] Variant not found in build dir, trying fallback...");
                // Continue to fallback mechanism
            }
            Err(e) => {
                println!("[ShaderManager] Error reading variant: {}", e);
                // Only return error if it's not a not-found error
                return Err(ShaderError::IoError {
                    path: variant_path.display().to_string(),
                    source: e,
                });
            }
        }

        // Fallback to monolithic shader files in source directory
        let fallback_filename = match shader_type {
            ShaderType::Entropy => "entropy.wgsl",
            ShaderType::Propagation => "propagate.wgsl",
        };

        // Construct path relative to crate root
        let fallback_path = PathBuf::from("src/shader/shaders").join(fallback_filename);
        println!(
            "[ShaderManager] Trying fallback shader: {:?}",
            fallback_path
        );

        // For include_str! we need constant paths, so use hardcoded fallbacks
        let shader_source = match shader_type {
            ShaderType::Entropy => {
                println!("[ShaderManager] Using hardcoded entropy shader");
                include_str!("shaders/entropy.wgsl").to_string()
            }
            ShaderType::Propagation => {
                println!("[ShaderManager] Using hardcoded propagation shader");
                include_str!("shaders/propagate.wgsl").to_string()
            }
        };

        println!("[ShaderManager] Successfully loaded fallback shader");
        // Add to cache
        self.loaded_shaders.insert(cache_key, shader_source.clone());
        Ok(shader_source)
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

    /// Returns a topologically sorted list of component names required to build the target component,
    /// considering the enabled features.
    pub fn resolve_component_dependencies(
        &self,
        target_component_name: &str,
        enabled_features: &HashSet<String>, // Changed from &[&str] to HashSet for easier lookup
    ) -> Result<Vec<String>, ShaderError> {
        let mut resolved_order = Vec::new();
        let mut visiting = std::collections::HashSet::new();
        let mut visited = std::collections::HashSet::new();

        self.visit_component(
            target_component_name,
            &mut resolved_order,
            &mut visiting,
            &mut visited,
            enabled_features,
        )?;

        Ok(resolved_order)
    }

    /// Recursive helper function for topological sort (DFS-based).
    fn visit_component<'a>(
        &'a self,
        component_name: &'a str,
        resolved_order: &mut Vec<String>,
        visiting: &mut std::collections::HashSet<&'a str>,
        visited: &mut std::collections::HashSet<&'a str>,
        enabled_features: &HashSet<String>,
    ) -> Result<(), ShaderError> {
        if visited.contains(component_name) {
            return Ok(());
        }
        if visiting.contains(component_name) {
            // ... (cycle detection) ...
            return Err(ShaderError::NotImplemented(format!(
                "Circular dependency detected involving component: {}",
                component_name
            )));
        }

        // Get component info
        let component_info = self.get_component_info_or_err(component_name)?;

        // Check if component's features are met
        if !self.component_features_met(component_info, enabled_features) {
            // If features not met, treat this component (and its subtree) as pruned/skipped
            // We don't add it to resolved_order and mark it visited to avoid re-processing.
            // We don't return an error, as this might be expected (e.g., alternative feature paths).
            visited.insert(component_name);
            return Ok(());
        }

        visiting.insert(component_name);

        // Recursively visit dependencies (which will also check features)
        for dep_name in &component_info.dependencies {
            self.visit_component(
                dep_name,
                resolved_order,
                visiting,
                visited,
                enabled_features,
            )?;
        }

        visiting.remove(component_name);
        visited.insert(component_name);
        // Only add the component if its features were met and it hasn't been added already
        // (The visited check at the start handles the 'already added' case)
        resolved_order.push(component_name.to_string());

        Ok(())
    }

    /// Helper to get component info or return a specific error.
    fn get_component_info_or_err(&self, name: &str) -> Result<&ShaderComponentInfo, ShaderError> {
        self.component_registry.get(name).ok_or_else(|| {
            ShaderError::NotImplemented(format!("Component '{}' not found in registry", name))
        })
    }

    /// Helper to check if a component's feature requirements are met.
    fn component_features_met(
        &self,
        component_info: &ShaderComponentInfo,
        enabled_features: &HashSet<String>,
    ) -> bool {
        // If the component requires features, check if all are enabled.
        !component_info
            .features
            .iter()
            .any(|req_feat| !enabled_features.contains(req_feat))
        // Equivalent to: component_info.features.iter().all(|req_feat| enabled_features.contains(req_feat))
    }

    /// Checks if all required GPU capabilities for a component are available
    pub fn check_gpu_capabilities(
        &self,
        component_name: &str,
        available_capabilities: &HashSet<String>,
    ) -> bool {
        if let Some(component) = self.get_component_info(component_name) {
            for capability in &component.gpu_capabilities {
                if !available_capabilities.contains(capability) {
                    return false;
                }
            }
            true
        } else {
            false
        }
    }

    /// Get the version of a component
    pub fn get_component_version(&self, component_name: &str) -> Option<&str> {
        self.get_component_info(component_name)
            .map(|info| info.version.as_str())
    }

    /// Checks if a component version meets a minimum requirement
    pub fn version_meets_requirement(
        &self,
        component_name: &str,
        min_version: &str,
    ) -> Result<bool, ShaderError> {
        let version = self.get_component_version(component_name).ok_or_else(|| {
            ShaderError::NotImplemented(format!("Component not found: {}", component_name))
        })?;

        // Simple version comparison (could use semver crate for more robust comparison)
        Ok(version >= min_version)
    }
}

#[cfg(test)]
mod tests {
    use super::*; // Import items from outer module
    use std::collections::HashSet;

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

    // Helper to create a test ShaderManager with specific components
    fn setup_test_registry() -> ShaderManager {
        let mut manager = ShaderManager::new().expect("Failed to create test ShaderManager");
        manager.component_registry.clear(); // Clear placeholder data

        let base_path = PathBuf::from("test/components");

        manager.component_registry.insert(
            "base".to_string(),
            ShaderComponentInfo {
                name: "base".to_string(),
                path: base_path.join("base.wgsl"),
                dependencies: vec![],
                features: vec![],
                version: "1.0.0".to_string(),
                provided_features: vec![],
                gpu_capabilities: vec![],
            },
        );
        manager.component_registry.insert(
            "util_a".to_string(),
            ShaderComponentInfo {
                name: "util_a".to_string(),
                path: base_path.join("util_a.wgsl"),
                dependencies: vec!["base".to_string()],
                features: vec![],
                version: "1.0.0".to_string(),
                provided_features: vec![],
                gpu_capabilities: vec![],
            },
        );
        manager.component_registry.insert(
            "util_b".to_string(),
            ShaderComponentInfo {
                name: "util_b".to_string(),
                path: base_path.join("util_b.wgsl"),
                dependencies: vec!["base".to_string()],
                features: vec![],
                version: "1.0.0".to_string(),
                provided_features: vec![],
                gpu_capabilities: vec![],
            },
        );
        manager.component_registry.insert(
            "complex".to_string(),
            ShaderComponentInfo {
                name: "complex".to_string(),
                path: base_path.join("complex.wgsl"),
                dependencies: vec!["util_a".to_string(), "util_b".to_string()],
                features: vec![],
                version: "1.0.0".to_string(),
                provided_features: vec![],
                gpu_capabilities: vec![],
            },
        );
        // For cycle detection
        manager.component_registry.insert(
            "cyclic_a".to_string(),
            ShaderComponentInfo {
                name: "cyclic_a".to_string(),
                path: base_path.join("cyclic_a.wgsl"),
                dependencies: vec!["cyclic_b".to_string()],
                features: vec![],
                version: "1.0.0".to_string(),
                provided_features: vec![],
                gpu_capabilities: vec![],
            },
        );
        manager.component_registry.insert(
            "cyclic_b".to_string(),
            ShaderComponentInfo {
                name: "cyclic_b".to_string(),
                path: base_path.join("cyclic_b.wgsl"),
                dependencies: vec!["cyclic_a".to_string()],
                features: vec![],
                version: "1.0.0".to_string(),
                provided_features: vec![],
                gpu_capabilities: vec![],
            },
        );

        manager
    }

    fn features(features: &[&str]) -> HashSet<String> {
        features.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn test_resolve_dependencies_simple() {
        let manager = setup_test_registry();
        let deps = manager
            .resolve_component_dependencies("util_a", &features(&[]))
            .unwrap();
        assert_eq!(deps, vec!["base", "util_a"]);
    }

    #[test]
    fn test_resolve_dependencies_complex() {
        let manager = setup_test_registry();
        let deps = manager
            .resolve_component_dependencies("complex", &features(&[]))
            .unwrap();
        assert_eq!(deps.len(), 4);
        assert_eq!(deps[0], "base");
        assert!(deps[1..3].contains(&"util_a".to_string()));
        assert!(deps[1..3].contains(&"util_b".to_string()));
        assert_eq!(deps[3], "complex");
    }

    #[test]
    fn test_resolve_dependencies_base() {
        let manager = setup_test_registry();
        let deps = manager
            .resolve_component_dependencies("base", &features(&[]))
            .unwrap();
        assert_eq!(deps, vec!["base"]);
    }

    #[test]
    fn test_resolve_dependencies_feature_met() {
        let manager = setup_test_registry();
        let deps = manager
            .resolve_component_dependencies("complex_with_feature", &features(&["feature_x"]))
            .unwrap();
        assert_eq!(deps.len(), 5);
        assert_eq!(deps[0], "base");
        assert_eq!(deps[1], "util_a");
        assert_eq!(deps[2], "feature_x_impl");
        assert_eq!(deps[3], "util_b");
        assert_eq!(deps[4], "complex_with_feature");
    }

    #[test]
    fn test_resolve_dependencies_feature_not_met() {
        let manager = setup_test_registry();
        let deps = manager
            .resolve_component_dependencies("complex_with_feature", &features(&[]))
            .unwrap();
        assert_eq!(deps.len(), 3);
        assert_eq!(deps[0], "base");
        assert_eq!(deps[1], "util_b");
        assert_eq!(deps[2], "complex_with_feature");
    }

    #[test]
    fn test_resolve_dependencies_feature_not_met_root() {
        let manager = setup_test_registry();
        let deps = manager
            .resolve_component_dependencies("feature_x_impl", &features(&[]))
            .unwrap();
        assert!(deps.is_empty());
    }

    #[test]
    fn test_resolve_dependencies_not_found() {
        let manager = setup_test_registry();
        let result = manager.resolve_component_dependencies("non_existent", &features(&[]));
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_dependencies_cyclic() {
        let manager = setup_test_registry();
        let result = manager.resolve_component_dependencies("cyclic_a", &features(&[]));
        assert!(result.is_err());
        match result {
            Err(ShaderError::NotImplemented(msg)) => {
                assert!(msg.contains("Circular dependency detected"));
            }
            _ => panic!("Expected cycle error"),
        }
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
