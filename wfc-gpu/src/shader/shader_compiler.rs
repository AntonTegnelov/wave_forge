#![allow(dead_code, unused_variables)] // Allow unused items during development

//! Responsible for assembling WGSL shader components into complete, specialized shaders.
//!
//! This module takes shader component sources, handles includes, applies feature flags,
//! and replaces specialization constants to produce final WGSL code ready for compilation
//! by the WGPU backend.

use super::shaders::ShaderType;
// Import build-time components from shader_registry
use super::shader_registry::{self, ShaderComponent};
use crate::GpuError; // Or define a specific CompilationError
use std::collections::HashMap;
use std::collections::HashSet;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CompilationError {
    #[error("I/O error reading shader component '{0}': {1}")]
    IoError(String, std::io::Error),
    #[error("Failed to resolve #include directive for '{0}' in component '{1}")]
    IncludeResolutionError(String, String),
    #[error("Cyclic #include detected involving '{0}")]
    CyclicIncludeError(String),
    #[error("Unknown shader component referenced: {0:?}")]
    UnknownComponent(String),
    #[error("Missing required specialization constant: {0}")]
    MissingSpecialization(String),
    #[error("Shader assembly error: {0}")]
    AssemblyError(String),
}

/// Compiles shader components into a single WGSL shader string.
#[derive(Debug, Default)]
pub struct ShaderCompiler {
    // Potential future fields: cache for loaded components, etc.
}

impl ShaderCompiler {
    /// Creates a new ShaderCompiler.
    pub fn new() -> Self {
        Self::default()
    }

    /// Compiles a complete shader from its components based on type and features.
    ///
    /// TODO: Implement the actual compilation logic.
    pub fn compile(
        &self,
        shader_type: ShaderType,
        features: &[&str],                     // e.g., ["atomics"]
        specialization: &HashMap<String, u32>, // e.g., {"NUM_TILES_U32_VALUE": 4}
    ) -> Result<String, CompilationError> {
        println!(
            "Compiling {:?} with features: {:?}, specialization: {:?}",
            shader_type, features, specialization
        );

        // Use function from shader_registry module
        let required_components = shader_registry::get_required_components(shader_type);
        println!("Required components: {:?}", required_components);

        let mut assembled_source = String::new();
        let included_files: HashSet<String> = HashSet::new(); // Specify type for HashSet

        // 1. Add header/common definitions (like Params struct?)
        // TODO: Define where common structs/constants live if not in components.
        assembled_source.push_str("// Shader compiled by wave-forge ShaderCompiler\n");
        assembled_source.push_str(&format!(
            "// Type: {:?}, Features: {:?}\n\n",
            shader_type, features
        ));

        // Placeholder for common definitions that might be needed globally
        // assembled_source.push_str(self.get_common_definitions(specialization)?);

        // 2. Process components recursively (handling includes)
        for component in required_components {
            // TODO: Implement recursive loading that handles #include directives
            // let component_source = self.load_and_process_component(component, &mut included_files, features, specialization)?;
            // Use function from shader_registry module
            let component_path = shader_registry::get_component_path(component);
            assembled_source.push_str(&format!(
                "// --- Component: {:?} from {} ---\n",
                component, component_path
            ));
            // In a real implementation, read file content here:
            // let content = std::fs::read_to_string(component_path).map_err(|e| CompilationError::IoError(component_path.to_string(), e))?;
            let placeholder_content = format!("// Content for {:?} would go here.\n", component);
            assembled_source.push_str(&placeholder_content);
            assembled_source.push('\n');
        }

        // 3. Add main entry point wrapper?
        // TODO: Define how entry points (`main_entropy`, `main_propagate`) are included/defined.
        // Maybe the last component in the list defines it?
        assembled_source.push_str(&format!(
            "// TODO: Define entry point for {:?}\n",
            shader_type
        ));
        match shader_type {
            ShaderType::Entropy => assembled_source.push_str("@compute @workgroup_size(64) fn main_entropy() { /* ... call components ... */ }\n"),
            ShaderType::Propagation => assembled_source.push_str("@compute @workgroup_size(64) fn main_propagate() { /* ... call components ... */ }\n"),
        }

        // 4. Post-processing (e.g., replace specialization constants)
        // TODO: Implement specialization constant replacement.
        // assembled_source = self.apply_specialization(assembled_source, specialization)?;

        println!("--- Assembled Source (Placeholder) ---");
        println!("{}", assembled_source);
        println!("--- End Assembled Source ---");

        // For now, return the placeholder assembled string
        Ok(assembled_source)
    }

    // TODO: Helper function placeholders for future implementation
    fn load_and_process_component(
        &self,
        // Type is now correctly found in shader_registry
        component: ShaderComponent,
        included_files: &mut HashSet<String>,
        features: &[&str],
        specialization: &HashMap<String, u32>,
    ) -> Result<String, CompilationError> {
        // Use function from shader_registry module
        let path_str = shader_registry::get_component_path(component);
        if !included_files.insert(path_str.to_string()) {
            // Already included, skip (or handle differently if needed)
            return Ok(String::new());
        }

        let content = std::fs::read_to_string(path_str)
            .map_err(|e| CompilationError::IoError(path_str.to_string(), e))?;

        let mut processed_content = String::new();
        for line in content.lines() {
            if line.trim_start().starts_with("#include") {
                // Resolve include
                // ... check for cycles in included_files ...
                // ... recursively call load_and_process_component ...
                // ... append result ...
            } else if line.trim_start().starts_with("//#feature=") {
                // Handle feature flag
                // ... check if feature is in `features` list ...
            } else {
                processed_content.push_str(line);
                processed_content.push('\n');
            }
        }
        Ok(processed_content)
    }

    fn apply_specialization(
        &self,
        source: String,
        specialization: &HashMap<String, u32>,
    ) -> Result<String, CompilationError> {
        let mut specialized_source = source;
        for (key, value) in specialization {
            // Simple replacement, might need regex for more complex cases
            let placeholder = format!("const {}: u32 =", key);
            let replacement = format!("const {}: u32 = {}; // Specialized", key, value);
            // Replace lines like "const FOO_VALUE: u32 = ...;" or similar patterns
            // This needs careful implementation to avoid incorrect replacements.
            specialized_source = specialized_source.replace(&placeholder, &replacement);
            // Placeholder logic
        }
        Ok(specialized_source)
    }
}

// Map CompilationError to GpuError if needed, or handle separately
impl From<CompilationError> for GpuError {
    fn from(err: CompilationError) -> Self {
        GpuError::ShaderError(err.to_string())
    }
}
