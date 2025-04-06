// wfc-gpu/build.rs

use serde::Deserialize;
use std::collections::HashMap;
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

// Data structures to parse registry.json
#[derive(Deserialize, Debug)]
#[allow(dead_code)] // Allow unused fields for now (e.g., dependencies, features)
struct Registry {
    components: HashMap<String, ComponentInfo>,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)] // Allow unused fields for now
struct ComponentInfo {
    path: String,
    dependencies: Vec<String>,
    required_features: Vec<String>,
    #[allow(dead_code)] // Provided features might be used later
    provided_features: Vec<String>,
}

// --- End Duplicated Definitions ---

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // Rerun if the registry file changes
    let registry_path_str = "src/shaders/components/registry.json";
    println!("cargo:rerun-if-changed={}", registry_path_str);

    // Parse the registry
    let registry_path = PathBuf::from(registry_path_str);
    let registry_read_error_msg =
        format!("Failed to read registry file: {}", registry_path.display());
    let registry_content = fs::read_to_string(&registry_path).expect(&registry_read_error_msg);
    let registry_parse_error_msg = format!(
        "Failed to parse shader registry JSON: {}",
        registry_path.display()
    );
    let registry: Registry =
        serde_json::from_str(&registry_content).expect(&registry_parse_error_msg);

    // Add rerun triggers for all actual shader component files listed in registry.json
    for component_info in registry.components.values() {
        println!("cargo:rerun-if-changed={}", component_info.path);
    }

    // Comment out old hardcoded triggers
    //println!("cargo:rerun-if-changed=src/shaders/utils.wgsl");
    //println!("cargo:rerun-if-changed=src/shaders/coords.wgsl");
    //println!("cargo:rerun-if-changed=src/shaders/rules.wgsl");
    //println!("cargo:rerun-if-changed=src/shaders/components/entropy_calculation.wgsl");
    //println!("cargo:rerun-if-changed=src/shaders/components/worklist_management.wgsl");
    //println!("cargo:rerun-if-changed=src/shaders/components/contradiction_detection.wgsl");

    println!("Running WFC-GPU build script...");

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR environment variable not set");
    let variants_dir = Path::new(&out_dir).join("shaders").join("variants");

    // Ensure the target directory for compiled variants exists
    fs::create_dir_all(&variants_dir).expect("Failed to create variants output directory");

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

    println!("[Build Script] Simulating Shader Registry loading...");
    let _registry_path = PathBuf::from(registry_path_str);
    // Access the parsed registry data directly
    println!(
        "[Build Script] Registry loaded (parsed from JSON): {:?}",
        registry
    );
    // let registry = wfc_gpu::shader_registry::ShaderRegistry::new(&registry_path)
    //     .expect("Failed to load shader registry in build script");
    // println!("[Build Script] Registry loaded (simulated).");

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
