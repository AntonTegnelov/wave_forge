// wfc-gpu/build.rs

use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

// --- Duplicated Definitions (needed because build.rs runs before crate is compiled) ---

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum BuildShaderType {
    Entropy,
    Propagation,
}

/// Duplicated logic from ShaderManager to avoid build dependency
fn get_variant_filename(shader_type: BuildShaderType, features: &[&str]) -> String {
    let base_name = match shader_type {
        BuildShaderType::Entropy => "Entropy",
        BuildShaderType::Propagation => "Propagation",
    };
    if features.is_empty() {
        format!("{}.wgsl", base_name)
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
            "Invalid version format for '{}': '{}'. Expected 'major.minor.patch'",
            component_name, version
        ));
    }

    for part in parts {
        if part.parse::<u32>().is_err() {
            return Err(format!(
                "Invalid version number in '{}' for '{}'. Each part must be a number",
                version, component_name
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
                    "Component '{}' depends on '{}', but '{}' is not in the registry",
                    component_name, dep_name, dep_name
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
    if path.contains(&current.to_string()) {
        // Found a cycle
        path.push(current.to_string());
        let cycle_path = path.join(" -> ");
        return Err(format!("Circular dependency detected: {}", cycle_path));
    }

    if visited.contains(current) {
        // Already checked this path, no cycle
        return Ok(());
    }

    // Add to current path
    path.push(current.to_string());

    // Get dependencies
    if let Some(info) = registry.components.get(current) {
        for dep in &info.dependencies {
            check_circular_dependencies(registry, dep, visited, path)?;
        }
    }

    // Remove from current path
    path.pop();

    // Mark as visited
    visited.insert(current.to_string());

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
                    "Component '{}' requires feature '{}', but this feature is not provided by any component",
                    component_name, feature
                ));
            }
        }
    }

    Ok(())
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Update registry path to reflect new structure
    let registry_path_str = "src/shader/shaders/components/registry.json";
    println!("cargo:rerun-if-changed={}", registry_path_str);

    // Parse the registry to get component paths
    let registry_content = match fs::read_to_string(registry_path_str) {
        Ok(content) => content,
        Err(e) => {
            println!("cargo:warning=Failed to read shader registry file: {}", e);
            // Return early but don't fail the build
            return;
        }
    };

    let registry_data: ShaderRegistryData = match serde_json::from_str(&registry_content) {
        Ok(data) => data,
        Err(e) => {
            println!("cargo:warning=Failed to parse shader registry JSON: {}", e);
            // Return early but don't fail the build
            return;
        }
    };

    // Validate dependencies and versions
    if let Err(e) = validate_dependencies(&registry_data) {
        println!("cargo:warning=Dependency validation error: {}", e);
        // Don't fail the build, just warn
    }

    // Validate features
    if let Err(e) = validate_features(&registry_data) {
        println!("cargo:warning=Feature validation error: {}", e);
        // Don't fail the build, just warn
    }

    // Add rerun triggers for all component files listed in the registry
    for (name, component) in registry_data.components.iter() {
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

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR environment variable not set");
    let variants_dir = Path::new(&out_dir).join("shaders").join("variants");

    // Ensure the target directory for compiled variants exists
    fs::create_dir_all(&variants_dir).expect("Failed to create variants output directory");

    // --- Actual Build-Time Shader Generation ---

    // 1. Initialize Registry
    // Use the crate's code directly by specifying the path relative to Cargo.toml
    // This requires adding the crate itself as a build-dependency in Cargo.toml
    // For now, we assume the necessary types are available directly.
    // We need to parse registry.json and use the compiler logic here.
    // NOTE: Actually *using* crate::shader_registry::ShaderRegistry here is complex
    // because build.rs runs before the crate is fully compiled.
    // A common pattern is to duplicate simplified loading/compiling logic
    // or use a separate helper crate.
    // For this step, we will SIMULATE the process using placeholder logic.

    println!(
        "[Build Script] Registry loaded with {} components",
        registry_data.components.len()
    );

    // 2. Initialize Compiler (Simulated)
    println!("[Build Script] Simulating Shader Compiler initialization...");
    // let compiler = wfc_gpu::shader_compiler::ShaderCompiler::new();

    // 3. Define Target Variants (Example: Entropy with/without atomics)
    // Use local BuildShaderType enum
    let targets = vec![
        (BuildShaderType::Entropy, vec!["atomics"]), // Request atomics version
        (BuildShaderType::Entropy, vec![]),          // Request fallback version
        (BuildShaderType::Propagation, vec!["atomics"]), // Request atomics version
        (BuildShaderType::Propagation, vec![]),      // Request fallback version
    ];
    println!("[Build Script] Defined target variants: {:?}", targets);

    // 4. Loop through targets, compile (simulated), and write to OUT_DIR
    for (shader_type, features) in targets {
        println!(
            "[Build Script] Processing {:?} with features: {:?}",
            shader_type, features
        );

        // TODO: Determine required specialization constants (e.g., NUM_TILES_U32_VALUE)
        // This might require reading config or passing build-time env vars.
        let specialization: std::collections::HashMap<String, u32> =
            std::collections::HashMap::new(); // Placeholder

        // Simulate compilation using the compiler (replace with actual call later)
        println!(
            "[Build Script Stub] Would call compiler.compile({:?}, {:?}, {:?})",
            shader_type, features, specialization
        );
        // let compiled_source = compiler.compile(shader_type, &features, &specialization)
        //     .expect(&format!("Failed to compile {:?} with {:?}", shader_type, features));

        // --- Placeholder Source Generation --- (Remove once compiler works)
        let placeholder_source = format!(
            "// Placeholder for {:?} with features {:?}\n// Specialization: {:?}\nfn main() {{}}",
            shader_type, features, specialization
        );
        let compiled_source = placeholder_source; // Use placeholder
                                                  // --- End Placeholder Source Generation ---

        // Determine output path
        // Use local get_variant_filename function
        let variant_filename = get_variant_filename(shader_type, &features);
        let out_path = variants_dir.join(&variant_filename);

        // Write the (placeholder) compiled shader
        let error_msg = format!("Failed to write compiled shader to {:?}", out_path);
        fs::write(&out_path, compiled_source).expect(&error_msg);
        println!(
            "[Build Script] Wrote placeholder shader variant to: {:?}",
            out_path
        );
    }

    println!("WFC-GPU build script finished generating placeholder shader variants.");
}
