// wfc-gpu/build.rs

use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

mod tools {
    pub mod shader_optimizer;
    pub mod shader_validator;
}

use tools::shader_optimizer::{FeatureVariant, OptimizerConfig, ShaderOptimizer};
use tools::shader_validator::{ShaderValidator, ValidatorConfig};

// --- Duplicated Definitions (needed because build.rs runs before crate is compiled) ---

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum BuildShaderType {
    Entropy,
    Propagation,
}

/// Duplicated logic from `ShaderManager` to avoid build dependency
fn get_variant_filename(shader_type: BuildShaderType, features: &[&str]) -> String {
    let base_name = match shader_type {
        BuildShaderType::Entropy => "Entropy",
        BuildShaderType::Propagation => "Propagation",
    };
    if features.is_empty() {
        format!("{base_name}.wgsl")
    } else {
        let mut sorted_features = features.to_vec();
        sorted_features.sort_unstable();
        format!("{}_{}.wgsl", base_name, sorted_features.join("_"))
    }
}

// --- End Duplicated Definitions ---

// --- Registry Parsing Structs ---
#[derive(Deserialize, Debug)]
struct ShaderComponentInfo {
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

#[derive(Deserialize, Debug)]
struct ShaderRegistryData {
    components: HashMap<String, ShaderComponentInfo>,
}

// --- End Registry Parsing Structs ---

/// Simple version check to make sure component versions follow semver format (major.minor.patch)
fn validate_version(component_name: &str, version: &str) -> Result<(), String> {
    if version.is_empty() {
        return Ok(());
    }

    let parts: Vec<&str> = version.split('.').collect();
    if parts.len() != 3 {
        return Err(format!(
            "Invalid version format for '{component_name}': '{version}'. Expected 'major.minor.patch'"
        ));
    }

    for part in parts {
        if part.parse::<u32>().is_err() {
            return Err(format!(
                "Invalid version number in '{version}' for '{component_name}'. Each part must be a number"
            ));
        }
    }

    Ok(())
}

/// Validates dependencies between components in the registry
fn validate_dependencies(registry: &ShaderRegistryData) -> Result<(), String> {
    // Check that all dependencies exist
    for (component_name, info) in &registry.components {
        for dep_name in &info.dependencies {
            if !registry.components.contains_key(dep_name) {
                return Err(format!(
                    "Component '{component_name}' depends on '{dep_name}', but '{dep_name}' is not in the registry"
                ));
            }
        }

        // Validate version
        validate_version(component_name, &info.version)?;
    }

    // Check for circular dependencies
    for component_name in registry.components.keys() {
        let mut visited = HashSet::new();
        let mut path = Vec::new();
        check_circular_dependencies(registry, component_name, &mut visited, &mut path)?;
    }

    Ok(())
}

/// Helper function to check for circular dependencies using DFS
fn check_circular_dependencies(
    registry: &ShaderRegistryData,
    current: &str,
    visited: &mut HashSet<String>,
    path: &mut Vec<String>,
) -> Result<(), String> {
    if path.contains(&current.to_owned()) {
        // Found a cycle
        path.push(current.to_owned());
        let cycle_path = path.join(" -> ");
        return Err(format!("Circular dependency detected: {cycle_path}"));
    }

    if visited.contains(current) {
        // Already checked this path, no cycle
        return Ok(());
    }

    // Add to current path
    path.push(current.to_owned());

    // Get dependencies
    if let Some(info) = registry.components.get(current) {
        for dep in &info.dependencies {
            check_circular_dependencies(registry, dep, visited, path)?;
        }
    }

    // Remove from current path
    path.pop();

    // Mark as visited
    visited.insert(current.to_owned());

    Ok(())
}

/// Validate feature consistency - check that required features are provided somewhere
fn validate_features(registry: &ShaderRegistryData) -> Result<(), String> {
    // Collect all provided features
    let mut all_provided_features = HashSet::new();
    for info in registry.components.values() {
        for feature in &info.provided_features {
            all_provided_features.insert(feature.clone());
        }
    }

    // Check required features
    for (component_name, info) in &registry.components {
        for feature in &info.required_features {
            if !all_provided_features.contains(feature) {
                return Err(format!(
                    "Component '{component_name}' requires feature '{feature}', but this feature is not provided by any component"
                ));
            }
        }
    }

    Ok(())
}

/// Generate optimized shader variants
fn generate_shader_variants(registry_path: &Path, out_dir: &Path) -> Result<(), String> {
    println!("[Build Script] Generating shader variants...");

    // Create optimizer configuration
    let optimizer_config = OptimizerConfig {
        src_dir: PathBuf::from("src/shader/shaders"),
        out_dir: out_dir.join("shaders").join("variants"),
        aggressive: false,
        default_features: vec![],
        variants: vec![
            // Entropy variants
            FeatureVariant {
                name: "entropy_atomics".to_owned(),
                features: vec!["atomics".to_owned()],
                workgroup_size: Some((8, 8, 1)),
                defines: HashMap::new(),
            },
            FeatureVariant {
                name: "entropy_no_atomics".to_owned(),
                features: vec![],
                workgroup_size: Some((8, 8, 1)),
                defines: HashMap::new(),
            },
            // Propagation variants
            FeatureVariant {
                name: "propagation_atomics".to_owned(),
                features: vec!["atomics".to_owned()],
                workgroup_size: Some((8, 8, 1)),
                defines: HashMap::new(),
            },
            FeatureVariant {
                name: "propagation_no_atomics".to_owned(),
                features: vec![],
                workgroup_size: Some((8, 8, 1)),
                defines: HashMap::new(),
            },
        ],
        debug: true,
    };

    // Create optimizer
    let mut optimizer = ShaderOptimizer::new(optimizer_config);

    // Load registry
    match optimizer.load_registry(registry_path) {
        Ok(()) => println!("[Build Script] Successfully loaded shader registry"),
        Err(e) => return Err(format!("Failed to load shader registry: {e}")),
    }

    // Generate variants
    match optimizer.generate_variants() {
        Ok(()) => println!("[Build Script] Successfully generated shader variants"),
        Err(e) => return Err(format!("Failed to generate shader variants: {e}")),
    }

    Ok(())
}

/// Validate generated shader variants
fn validate_shader_variants(out_dir: &Path) -> Result<(), String> {
    println!("[Build Script] Validating shader variants...");

    // Create validator configuration
    let validator_config = ValidatorConfig {
        naga_validator_path: None, // Use internal validator
        tint_validator_path: None,
        shader_dir: out_dir.join("shaders").join("variants"),
        report_path: Some(out_dir.join("shader_validation_report.txt")),
        warnings: true,
    };

    // Create validator
    let mut validator = ShaderValidator::new(validator_config);

    // Validate shaders
    match validator.validate_directory() {
        Ok(true) => println!("[Build Script] All shader variants are valid"),
        Ok(false) => {
            println!("[Build Script] Some shader variants have validation errors");
            println!("[Build Script] See validation report for details");
        }
        Err(e) => return Err(format!("Failed to validate shader variants: {e}")),
    }

    Ok(())
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Update registry path to reflect new structure
    let registry_path_str = "src/shader/shaders/components/registry.json";
    println!("cargo:rerun-if-changed={registry_path_str}");

    // Get output directory
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR environment variable not set");
    let out_dir_path = Path::new(&out_dir);
    let variants_dir = out_dir_path.join("shaders").join("variants");

    // Ensure the target directory for compiled variants exists
    fs::create_dir_all(&variants_dir).expect("Failed to create variants output directory");

    // Parse the registry to get component paths
    let registry_content = match fs::read_to_string(registry_path_str) {
        Ok(content) => content,
        Err(e) => {
            println!("cargo:warning=Failed to read shader registry file: {e}");
            // Return early but don't fail the build
            return;
        }
    };

    let registry_data: ShaderRegistryData = match serde_json::from_str(&registry_content) {
        Ok(data) => data,
        Err(e) => {
            println!("cargo:warning=Failed to parse shader registry JSON: {e}");
            // Return early but don't fail the build
            return;
        }
    };

    // Validate dependencies and versions
    if let Err(e) = validate_dependencies(&registry_data) {
        println!("cargo:warning=Dependency validation error: {e}");
        // Don't fail the build, just warn
    }

    // Validate features
    if let Err(e) = validate_features(&registry_data) {
        println!("cargo:warning=Feature validation error: {e}");
        // Don't fail the build, just warn
    }

    // Add rerun triggers for all component files listed in the registry
    for component in registry_data.components.values() {
        // Ensure the path exists before adding the trigger
        let component_path = PathBuf::from(&component.path);
        if component_path.exists() {
            println!("cargo:rerun-if-changed={}", component.path);
        } else {
            println!(
                "cargo:warning=Shader component path not found: {}",
                component.path
            );
        }
    }

    println!("Running WFC-GPU build script...");

    // Generate optimized shader variants
    match generate_shader_variants(Path::new(registry_path_str), out_dir_path) {
        Ok(()) => {}
        Err(e) => println!("cargo:warning=Failed to generate shader variants: {e}"),
    }

    // Validate generated shader variants
    match validate_shader_variants(out_dir_path) {
        Ok(()) => {}
        Err(e) => println!("cargo:warning=Failed to validate shader variants: {e}"),
    }

    println!("WFC-GPU build script finished shader variant generation and validation.");
}
