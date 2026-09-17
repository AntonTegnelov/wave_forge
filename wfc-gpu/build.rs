// wfc-gpu/build.rs

use std::env;
use std::fs;
use std::path::Path;

// Re-use types from main crate
// Note: In a real implementation, you might want to define these in a separate build-utils crate
#[derive(Debug, Clone, Copy)]
enum ShaderType {
    Entropy,
    Propagation,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=src/shader/shaders/");
    println!("cargo:rerun-if-changed=src/shader/shaders/components/");
    println!("cargo:rerun-if-changed=src/shader/shaders/components/registry.json");

    let out_dir = env::var("OUT_DIR")?;
    let variants_dir = Path::new(&out_dir).join("shaders").join("variants");
    fs::create_dir_all(&variants_dir)?;

    // Define feature combinations to pre-compile
    let feature_combinations = vec![
        vec![], // Base variant with no features
        vec!["atomics"],
        vec!["subgrid"],
        vec!["atomics", "subgrid"],
        // Add more combinations as needed
    ];

    // Pre-compile variants for each shader type and feature combination
    for shader_type in &[ShaderType::Entropy, ShaderType::Propagation] {
        for features in &feature_combinations {
            let variant_name = get_variant_filename(shader_type, features);
            let variant_path = variants_dir.join(&variant_name);

            println!("Pre-compiling shader variant: {}", variant_name);

            // TODO: Initialize ShaderCompiler and compile variant
            // For now, just copy the base shader as a placeholder
            let source = match shader_type {
                ShaderType::Entropy => include_str!("src/shader/shaders/entropy.wgsl"),
                ShaderType::Propagation => include_str!("src/shader/shaders/propagate.wgsl"),
            };

            // Write the compiled shader to the variants directory
            fs::write(&variant_path, source)?;
        }
    }

    Ok(())
}

fn get_variant_filename(shader_type: &ShaderType, features: &[&str]) -> String {
    let base_name = match shader_type {
        ShaderType::Entropy => "Entropy",
        ShaderType::Propagation => "Propagation",
    };
    if features.is_empty() {
        format!("{}.wgsl", base_name)
    } else {
        let mut sorted_features = features.to_vec();
        sorted_features.sort_unstable();
        format!("{}_{}.wgsl", base_name, sorted_features.join("_"))
    }
}
