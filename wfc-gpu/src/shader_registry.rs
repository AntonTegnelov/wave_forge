#![allow(dead_code, unused_variables)] // Allow unused items during development

//! Manages metadata about shader components, features, and their relationships.
//!
//! This module will load information (potentially from a manifest file like registry.json)
//! about available shader components, their dependencies, and the features they require or provide.
//! It will be used by the ShaderCompiler to determine which components to assemble for a given shader variant.

use crate::shaders::ShaderType;
use crate::GpuError; // Or define a specific RegistryError
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use thiserror::Error;

/// Represents the individual WGSL source file components.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderComponent {
    // Core logic components
    EntropyCalculation,
    WorklistManagement,
    CellCollapse,
    ContradictionDetection,
    // Utility components
    Utils,
    Coords,
    Rules,
    // Feature-specific components (add as needed)
    // Atomics,
    // NoAtomics,
}

#[derive(Error, Debug)]
pub enum RegistryError {
    #[error("Component not found in registry: {0:?}")]
    ComponentNotFound(ShaderComponent),
    #[error("Feature conflict or missing dependency for component {0:?}")]
    DependencyError(String), // Use String for component name from JSON
    #[error("Failed to load or parse registry data from '{path}': {source}")]
    LoadError {
        path: String,
        source: Box<dyn std::error::Error + Send + Sync>,
    }, // Use Box<dyn Error>
    #[error("Invalid component name in registry: {0}")]
    InvalidComponentName(String),
    #[error("Cyclic dependency detected involving component: {0}")]
    CyclicDependency(String),
}

/// Placeholder for metadata about a single shader component.
#[derive(Debug, Clone, Deserialize)] // Add Deserialize
pub struct ComponentInfo {
    path: String,
    #[serde(default)] // Use default empty vec if missing
    dependencies: Vec<String>, // Load dependencies as Strings first
    #[serde(default)]
    required_features: HashSet<String>,
    #[serde(default)]
    provided_features: HashSet<String>,
}

// Struct to directly deserialize the registry.json file
#[derive(Debug, Deserialize)]
struct RegistryFile {
    components: HashMap<String, ComponentInfo>,
}

/// Manages the registry of available shader components and features.
#[derive(Debug, Default)]
pub struct ShaderRegistry {
    // Store ComponentInfo directly, mapped by ShaderComponent enum
    components: HashMap<ShaderComponent, ComponentInfo>,
    // Add map for name to enum conversion
    name_to_component: HashMap<String, ShaderComponent>,
}

impl ShaderRegistry {
    /// Creates a new ShaderRegistry by loading data from the specified manifest file.
    pub fn new(registry_path: &Path) -> Result<Self, RegistryError> {
        println!("Loading shader registry from: {:?}", registry_path);
        let path_str = registry_path.display().to_string();

        let file_content =
            fs::read_to_string(registry_path).map_err(|e| RegistryError::LoadError {
                path: path_str.clone(),
                source: Box::new(e),
            })?;

        let registry_file: RegistryFile =
            serde_json::from_str(&file_content).map_err(|e| RegistryError::LoadError {
                path: path_str,
                source: Box::new(e),
            })?;

        let mut components = HashMap::new();
        let mut name_to_component = HashMap::new();

        // Convert component names (Strings) to ShaderComponent enum variants
        for (name, info) in registry_file.components {
            let component_enum = match name.as_str() {
                "Utils" => ShaderComponent::Utils,
                "Coords" => ShaderComponent::Coords,
                "Rules" => ShaderComponent::Rules,
                "EntropyCalculation" => ShaderComponent::EntropyCalculation,
                "WorklistManagement" => ShaderComponent::WorklistManagement,
                "ContradictionDetection" => ShaderComponent::ContradictionDetection,
                "CellCollapse" => ShaderComponent::CellCollapse,
                // Add other components as they are defined
                _ => return Err(RegistryError::InvalidComponentName(name)),
            };
            components.insert(component_enum, info);
            name_to_component.insert(name, component_enum);
        }

        // TODO: Validate dependencies (check if all referenced components exist and detect cycles)
        // Self::validate_dependencies(&components, &name_to_component)?;

        Ok(Self {
            components,
            name_to_component,
        })
    }

    /// Gets information about a specific component.
    pub fn get_component_info(
        &self,
        component: ShaderComponent,
    ) -> Result<&ComponentInfo, RegistryError> {
        self.components
            .get(&component)
            .ok_or(RegistryError::ComponentNotFound(component))
    }

    /// Determines the list of components required for a specific shader variant.
    ///
    /// TODO: This needs a proper implementation using component dependencies and feature flags.
    pub fn get_shader_variant_components(
        &self,
        shader_type: ShaderType,
        features: &[&str],
    ) -> Result<Vec<ShaderComponent>, RegistryError> {
        println!(
            "Getting components for {:?} with features {:?} (using placeholder logic)",
            shader_type, features
        );
        // Placeholder: Use the static helper function for now.
        // This should be replaced with dependency resolution logic.
        let basic_components = get_required_components(shader_type);
        Ok(basic_components)
        // TODO: Implement dependency resolution logic here:
        // 1. Start with base components for shader_type (needs definition)
        // 2. Use a worklist and a set of included components
        // 3. While worklist is not empty:
        //    - Pop component C
        //    - If C is already included, continue
        //    - Check if C's required_features are met by `features`
        //    - Add C to included set
        //    - Get dependencies of C from self.components
        //    - Push dependencies onto worklist
        // 4. Return included set (potentially ordered)
        // Err(RegistryError::LoadError { path: "Not Implemented".to_string(), source: "Dependency resolution not implemented".into() })
    }

    // --- Helper for dependency validation (TODO) ---
    /*
    fn validate_dependencies(
        components: &HashMap<ShaderComponent, ComponentInfo>,
        name_to_component: &HashMap<String, ShaderComponent>,
    ) -> Result<(), RegistryError> {
        for (component_enum, info) in components {
            let mut visited = HashSet::new();
            let mut path = vec![*component_enum];
            Self::check_cyclic_deps(components, name_to_component, *component_enum, &mut visited, &mut path)?;
            for dep_name in &info.dependencies {
                if !name_to_component.contains_key(dep_name) {
                    return Err(RegistryError::DependencyError(format!(
                        "Component {:?} depends on unknown component '{}'",
                        component_enum, dep_name
                    )));
                }
            }
        }
        Ok(())
    }

    fn check_cyclic_deps(
        components: &HashMap<ShaderComponent, ComponentInfo>,
        name_to_component: &HashMap<String, ShaderComponent>,
        current: ShaderComponent,
        visited: &mut HashSet<ShaderComponent>,
        path: &mut Vec<ShaderComponent>,
    ) -> Result<(), RegistryError> {
        visited.insert(current);
        if let Some(info) = components.get(&current) {
            for dep_name in &info.dependencies {
                if let Some(&dep_enum) = name_to_component.get(dep_name) {
                    if path.contains(&dep_enum) {
                        // Cycle detected
                        path.push(dep_enum); // Add the repeated node to show the cycle
                        let cycle_str = path.iter().map(|c| format!("{:?}", c)).collect::<Vec<_>>().join(" -> ");
                        return Err(RegistryError::CyclicDependency(cycle_str));
                    }
                    if !visited.contains(&dep_enum) {
                        path.push(dep_enum);
                        Self::check_cyclic_deps(components, name_to_component, dep_enum, visited, path)?;
                        path.pop();
                    }
                }
                // Dependency name validity checked in validate_dependencies
            }
        }
        Ok(())
    }
    */
}

/// Placeholder function to get the source code path for a component.
/// TODO: Replace with actual file reading or embedding when ShaderCompiler is built.
///       This might better belong in the ShaderCompiler or be derived from registry data.
pub fn get_component_path(component: ShaderComponent) -> &'static str {
    match component {
        ShaderComponent::EntropyCalculation => "src/shaders/components/entropy_calculation.wgsl", // Updated path prefix
        ShaderComponent::WorklistManagement => "src/shaders/components/worklist_management.wgsl", // Updated path prefix
        ShaderComponent::CellCollapse => "src/shaders/components/cell_collapse.wgsl", // Updated path prefix
        ShaderComponent::ContradictionDetection => {
            "src/shaders/components/contradiction_detection.wgsl" // Updated path prefix
        }
        ShaderComponent::Utils => "src/shaders/utils.wgsl", // Updated path prefix
        ShaderComponent::Coords => "src/shaders/coords.wgsl", // Updated path prefix
        ShaderComponent::Rules => "src/shaders/rules.wgsl", // Updated path prefix
                                                             // ShaderComponent::Atomics => "src/shaders/features/atomics.wgsl", // Updated path prefix
                                                             // ShaderComponent::NoAtomics => "src/shaders/features/no_atomics.wgsl", // Updated path prefix
    }
}

/// Placeholder function to define the required components for a given shader type.
/// TODO: This logic will be more sophisticated, considering features, in the ShaderRegistry::get_shader_variant_components method.
pub fn get_required_components(shader_type: ShaderType) -> Vec<ShaderComponent> {
    match shader_type {
        ShaderType::Entropy => vec![
            ShaderComponent::Utils,  // Basic utilities (e.g., count_bits)
            ShaderComponent::Coords, // For position-based tie-breaking
            ShaderComponent::Rules,  // For possibility mask helpers
            ShaderComponent::EntropyCalculation,
            // TODO: Add feature components like Atomics/NoAtomics based on config
        ],
        ShaderType::Propagation => vec![
            ShaderComponent::Utils,
            ShaderComponent::Coords,
            ShaderComponent::Rules,
            ShaderComponent::WorklistManagement,
            ShaderComponent::ContradictionDetection,
            // ShaderComponent::CellCollapse, // Collapse might be separate shader or part of host logic
            // TODO: Add feature components
        ],
    }
}

// Map RegistryError to GpuError if needed
impl From<RegistryError> for GpuError {
    fn from(err: RegistryError) -> Self {
        // Choose an appropriate GpuError variant, e.g., ShaderError or a new one
        GpuError::ShaderError(format!("Shader Registry Error: {}", err))
    }
}
