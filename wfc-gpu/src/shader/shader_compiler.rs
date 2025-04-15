#![allow(dead_code, unused_variables)] // Allow unused items during development

//! Responsible for assembling WGSL shader components into complete, specialized shaders.
//!
//! This module takes shader component sources, handles includes, applies feature flags,
//! and replaces specialization constants to produce final WGSL code ready for compilation
//! by the WGPU backend.

use super::shaders::ShaderType;
// Import build-time components from shader_registry
use super::shader_registry::{self, ShaderComponent, ShaderRegistry};
use crate::gpu::features::{AtomicsSupport, GpuCapabilities, GpuFeature, WorkgroupSupport};
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
    #[error("Missing required feature: {0}")]
    MissingFeature(String),
    #[error("Registry error: {0}")]
    RegistryError(#[from] shader_registry::RegistryError),
}

/// Feature and capability set used to guide shader compilation.
#[derive(Debug, Clone)]
pub struct ShaderFeatures {
    /// GPU capabilities information.
    pub capabilities: GpuCapabilities,
    /// Atomics support information.
    pub atomics: AtomicsSupport,
    /// Workgroup support information.
    pub workgroups: WorkgroupSupport,
    /// Additional feature flags for conditional compilation.
    pub feature_flags: HashMap<String, bool>,
}

impl ShaderFeatures {
    /// Create a new ShaderFeatures instance from GPU capabilities.
    pub fn new(capabilities: &GpuCapabilities) -> Self {
        Self {
            capabilities: capabilities.clone(),
            atomics: AtomicsSupport::new(capabilities),
            workgroups: WorkgroupSupport::new(capabilities),
            feature_flags: HashMap::new(),
        }
    }

    /// Set a feature flag.
    pub fn with_feature(mut self, name: &str, enabled: bool) -> Self {
        self.feature_flags.insert(name.to_string(), enabled);
        self
    }

    /// Get all shader defines for the features.
    pub fn shader_defines(&self) -> Vec<String> {
        let mut defines = Vec::new();

        // Add atomics defines
        defines.extend(self.atomics.shader_defines());

        // Add workgroup defines
        defines.extend(self.workgroups.shader_defines());

        // Add custom feature flags
        for (name, enabled) in &self.feature_flags {
            defines.push(format!(
                "FEATURE_{}={}",
                name.to_uppercase(),
                if *enabled { "1" } else { "0" }
            ));
        }

        defines
    }

    /// Check if a specific GPU feature is supported.
    pub fn supports(&self, feature: GpuFeature) -> bool {
        self.capabilities.supports(feature)
    }

    /// Check if all specified features are supported.
    pub fn supports_all(&self, features: &[GpuFeature]) -> bool {
        self.capabilities.supports_all(features)
    }
}

/// Compiles shader components into a single WGSL shader string.
#[derive(Debug)]
pub struct ShaderCompiler {
    registry: ShaderRegistry,
    // Add cache for loaded components
    component_cache: HashMap<ShaderComponent, String>,
}

impl ShaderCompiler {
    /// Creates a new ShaderCompiler with the specified shader registry.
    pub fn new(registry: ShaderRegistry) -> Self {
        Self {
            registry,
            component_cache: HashMap::new(),
        }
    }

    /// Compiles a complete shader from its components based on type and features.
    pub fn compile(
        &mut self,
        shader_type: ShaderType,
        features: &ShaderFeatures,
        specialization: &HashMap<String, u32>, // e.g., {"NUM_TILES_U32_VALUE": 4}
    ) -> Result<String, CompilationError> {
        println!(
            "Compiling {:?} with features: {:?}, specialization: {:?}",
            shader_type, features.feature_flags, specialization
        );

        // Convert feature flags to string array for component selection
        let feature_flags: Vec<&str> = features
            .feature_flags
            .iter()
            .filter_map(|(name, enabled)| if *enabled { Some(name.as_str()) } else { None })
            .collect();

        // Get components required for this shader
        let required_components = self
            .registry
            .get_shader_variant_components(shader_type, &feature_flags)?;

        println!("Required components: {:?}", required_components);

        // Validate that the GPU supports all required features
        for component in &required_components {
            let info = self.registry.get_component_info(*component)?;

            // Check GPU capabilities for this component
            let capabilities_strs: HashSet<String> = features.capabilities.feature_names();

            for capability in info.required_capabilities() {
                if !capabilities_strs.contains(capability) {
                    return Err(CompilationError::MissingFeature(format!(
                        "Component {:?} requires GPU capability '{}' which is not available",
                        component, capability
                    )));
                }
            }
        }

        let mut assembled_source = String::new();
        let mut included_files = HashSet::new();

        // 1. Add header/common definitions
        assembled_source.push_str("// Shader compiled by wave-forge ShaderCompiler\n");
        assembled_source.push_str(&format!(
            "// Type: {:?}, Features: {:?}\n\n",
            shader_type, feature_flags
        ));

        // 2. Add feature defines
        assembled_source.push_str("// Feature defines\n");
        for define in features.shader_defines() {
            assembled_source.push_str(&format!("#define {}\n", define));
        }
        assembled_source.push('\n');

        // 3. Process components recursively (handling includes)
        for component in required_components {
            let component_source = self.load_and_process_component(
                component,
                &mut included_files,
                features,
                specialization,
            )?;

            assembled_source.push_str(&format!("// --- Component: {:?} ---\n", component));
            assembled_source.push_str(&component_source);
            assembled_source.push_str("\n\n");
        }

        // 4. Add main entry point
        assembled_source.push_str("// Entry point definition\n");

        // Determine optimal workgroup size based on GPU capabilities
        let workgroup_size = match shader_type {
            ShaderType::Entropy => features.workgroups.recommended_workgroup_size(
                crate::gpu::features::workgroups::OperationType::EntropyCalculation,
            ),
            ShaderType::Propagation => features.workgroups.recommended_workgroup_size(
                crate::gpu::features::workgroups::OperationType::Propagation,
            ),
        };

        match shader_type {
            ShaderType::Entropy => {
                assembled_source.push_str(&format!(
                    "@compute @workgroup_size({}, {}, {}) fn main_entropy() {{\n",
                    workgroup_size.0, workgroup_size.1, workgroup_size.2
                ));
                assembled_source.push_str("    // TODO: Call entropy calculation function\n");
                assembled_source.push_str("    calculate_entropy();\n");
                assembled_source.push_str("}\n");
            }
            ShaderType::Propagation => {
                assembled_source.push_str(&format!(
                    "@compute @workgroup_size({}, {}, {}) fn main_propagate() {{\n",
                    workgroup_size.0, workgroup_size.1, workgroup_size.2
                ));
                assembled_source.push_str("    // TODO: Call propagation function\n");
                assembled_source.push_str("    propagate_constraints();\n");
                assembled_source.push_str("}\n");
            }
        }

        // 5. Apply specialization constants
        assembled_source = self.apply_specialization(assembled_source, specialization)?;

        Ok(assembled_source)
    }

    fn load_and_process_component(
        &mut self,
        component: ShaderComponent,
        included_files: &mut HashSet<String>,
        features: &ShaderFeatures,
        _specialization: &HashMap<String, u32>,
    ) -> Result<String, CompilationError> {
        // Check component cache first
        if let Some(cached) = self.component_cache.get(&component) {
            return Ok(cached.clone());
        }

        // Get the component path
        let path_str = shader_registry::get_component_path(component);

        // Mark as included for cycle detection
        if !included_files.insert(path_str.to_string()) {
            // Already included, skip to avoid cycles
            return Ok(String::new());
        }

        // Load component content
        let content = std::fs::read_to_string(path_str)
            .map_err(|e| CompilationError::IoError(path_str.to_string(), e))?;

        let mut processed_content = String::new();

        // Process each line, handling includes and conditional compilation
        for line in content.lines() {
            let trimmed = line.trim_start();

            if trimmed.starts_with("#include") {
                // Process include directive
                let include_file = trimmed
                    .trim_start_matches("#include")
                    .trim()
                    .trim_matches('"');

                // Recursive include
                let included_component = self.resolve_include(include_file).map_err(|_| {
                    CompilationError::IncludeResolutionError(
                        include_file.to_string(),
                        path_str.to_string(),
                    )
                })?;

                let included_content = self.load_and_process_component(
                    included_component,
                    included_files,
                    features,
                    _specialization,
                )?;

                processed_content.push_str(&included_content);
                processed_content.push('\n');
            } else if trimmed.starts_with("#if") {
                // Handle conditional compilation
                let condition = self.evaluate_condition(trimmed, features)?;
                if !condition {
                    // Skip lines until matching #endif or #else
                    // This is a simplified implementation
                    continue;
                }

                // Keep the line for WGSL compiler
                processed_content.push_str(line);
                processed_content.push('\n');
            } else if trimmed.starts_with("#else") || trimmed.starts_with("#endif") {
                // Keep directive for WGSL compiler
                processed_content.push_str(line);
                processed_content.push('\n');
            } else {
                // Regular line, keep it
                processed_content.push_str(line);
                processed_content.push('\n');
            }
        }

        // Cache the processed component
        self.component_cache
            .insert(component, processed_content.clone());

        // Remove from included_files set to allow reuse in other contexts
        included_files.remove(path_str);

        Ok(processed_content)
    }

    fn resolve_include(&self, include_file: &str) -> Result<ShaderComponent, CompilationError> {
        // Convert include file name to component
        // This is a simplified implementation - in a real system, you might
        // have a more sophisticated mapping system
        match include_file {
            "utils.wgsl" => Ok(ShaderComponent::Utils),
            "coords.wgsl" => Ok(ShaderComponent::Coords),
            "rules.wgsl" => Ok(ShaderComponent::Rules),
            "entropy_calculation.wgsl" => Ok(ShaderComponent::EntropyCalculation),
            "worklist_management.wgsl" => Ok(ShaderComponent::WorklistManagement),
            "cell_collapse.wgsl" => Ok(ShaderComponent::CellCollapse),
            "contradiction_detection.wgsl" => Ok(ShaderComponent::ContradictionDetection),
            "shannon.wgsl" => Ok(ShaderComponent::ShannonEntropy),
            "count_based.wgsl" => Ok(ShaderComponent::CountBasedEntropy),
            "direct.wgsl" => Ok(ShaderComponent::DirectPropagation),
            "subgrid.wgsl" => Ok(ShaderComponent::SubgridPropagation),
            _ => Err(CompilationError::UnknownComponent(include_file.to_string())),
        }
    }

    fn evaluate_condition(
        &self,
        condition: &str,
        features: &ShaderFeatures,
    ) -> Result<bool, CompilationError> {
        // This is a simplified condition evaluator
        // In a real implementation, you would parse and evaluate more complex conditions

        let condition = condition.trim_start_matches("#if").trim();

        match condition {
            "ENABLE_ATOMICS" => Ok(features.atomics.has_atomics()),
            "ENABLE_SUBGROUPS" => Ok(features.workgroups.has_subgroups()),
            "ENABLE_WORKGROUP_SHARED_MEMORY" => Ok(features.workgroups.has_shared_memory()),
            _ => {
                // Check custom feature flags
                for (name, enabled) in &features.feature_flags {
                    if condition == format!("FEATURE_{}", name.to_uppercase()) {
                        return Ok(*enabled);
                    }
                }

                // Default to true if condition not recognized
                // Could also return an error instead
                Ok(true)
            }
        }
    }

    fn apply_specialization(
        &self,
        source: String,
        specialization: &HashMap<String, u32>,
    ) -> Result<String, CompilationError> {
        let mut specialized_source = source;

        for (key, value) in specialization {
            // Replace specialization constants
            // Format: const NAME: u32 = VALUE;
            let pattern = format!("const {}: u32 = [0-9]+", key);
            let replacement = format!("const {}: u32 = {}; // Specialized", key, value);

            // Use a regex for replacement
            // This is a placeholder for the real implementation
            specialized_source = specialized_source.replace(&pattern, &replacement);
        }

        Ok(specialized_source)
    }
}

// Map CompilationError to GpuError if needed, or handle separately
impl From<CompilationError> for GpuError {
    fn from(err: CompilationError) -> Self {
        GpuError::shader_error(
            err.to_string(),
            crate::utils::error::GpuErrorContext::default(),
        )
    }
}
