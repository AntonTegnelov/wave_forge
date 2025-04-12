#![allow(dead_code, unused_variables)] // Allow unused items during development

//! Manages metadata about shader components, features, and their relationships.
//!
//! This module will load information (potentially from a manifest file like registry.json)
//! about available shader components, their dependencies, and the features they require or provide.
//! It will be used by the ShaderCompiler to determine which components to assemble for a given shader variant.

use super::shaders::ShaderType;
use crate::GpuError; // Or define a specific RegistryError
use semver::Version;
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
    // Entropy-specific components
    ShannonEntropy,
    CountBasedEntropy,
    // Propagation-specific components
    DirectPropagation,
    SubgridPropagation,
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
    #[error("Version parsing error for component {0}: {1}")]
    VersionParseError(String, String),
    #[error("GPU capability not available: {0}")]
    MissingCapability(String),
    #[error("Missing required feature for component {0}: {1}")]
    MissingFeature(String, String),
}

/// Metadata about a single shader component.
#[derive(Debug, Clone, Deserialize)]
pub struct ComponentInfo {
    path: String,
    #[serde(default)] // Use default empty string if missing
    version: String,
    #[serde(default)] // Use default empty vec if missing
    dependencies: Vec<String>, // Load dependencies as Strings first
    #[serde(default)]
    required_features: HashSet<String>,
    #[serde(default)]
    provided_features: HashSet<String>,
    #[serde(default)]
    gpu_capabilities: HashSet<String>,
}

impl ComponentInfo {
    /// Get the version as a semver::Version
    pub fn get_version(&self) -> Result<Version, RegistryError> {
        if self.version.is_empty() {
            // Default to 0.1.0 if no version specified
            return Ok(Version::new(0, 1, 0));
        }

        Version::parse(&self.version)
            .map_err(|e| RegistryError::VersionParseError(self.path.clone(), e.to_string()))
    }

    /// Check if this component requires a specific GPU capability
    pub fn requires_capability(&self, capability: &str) -> bool {
        self.gpu_capabilities.contains(capability)
    }

    /// Get all GPU capabilities required by this component
    pub fn required_capabilities(&self) -> &HashSet<String> {
        &self.gpu_capabilities
    }

    /// Get the path to the component source file
    pub fn path(&self) -> &str {
        &self.path
    }
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
                "ShannonEntropy" => ShaderComponent::ShannonEntropy,
                "CountBasedEntropy" => ShaderComponent::CountBasedEntropy,
                "DirectPropagation" => ShaderComponent::DirectPropagation,
                "SubgridPropagation" => ShaderComponent::SubgridPropagation,
                // Add other components as they are defined
                _ => return Err(RegistryError::InvalidComponentName(name)),
            };
            components.insert(component_enum, info);
            name_to_component.insert(name, component_enum);
        }

        // Validate dependencies (check if all referenced components exist and detect cycles)
        Self::validate_dependencies(&components, &name_to_component)?;

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
    pub fn get_shader_variant_components(
        &self,
        shader_type: ShaderType,
        features: &[&str],
    ) -> Result<Vec<ShaderComponent>, RegistryError> {
        println!(
            "Getting components for {:?} with features {:?}",
            shader_type, features
        );

        // Start with base components for shader type
        let base_components = match shader_type {
            ShaderType::Entropy => vec![ShaderComponent::EntropyCalculation],
            ShaderType::Propagation => {
                // Check if "subgrid" feature is requested, otherwise use direct propagation
                if features.contains(&"subgrid") {
                    vec![ShaderComponent::SubgridPropagation]
                } else {
                    vec![ShaderComponent::DirectPropagation]
                }
            }
        };

        // Convert features to a HashSet for easier lookup
        let feature_set: HashSet<&str> = features.iter().copied().collect();

        // Use a worklist algorithm to resolve all dependencies
        let mut worklist = base_components.clone();
        let mut included = HashSet::new();
        let mut provided_features = HashSet::new();

        while let Some(component) = worklist.pop() {
            // Skip if already processed
            if included.contains(&component) {
                continue;
            }

            // Get component info
            let info = self.get_component_info(component)?;

            // Check if all required features are available
            for req_feature in &info.required_features {
                if !provided_features.contains(req_feature)
                    && !feature_set.contains(req_feature.as_str())
                {
                    return Err(RegistryError::MissingFeature(
                        format!("{:?}", component),
                        req_feature.clone(),
                    ));
                }
            }

            // Add component's provided features
            provided_features.extend(info.provided_features.iter().cloned());

            // Mark as included
            included.insert(component);

            // Add dependencies to worklist
            for dep_name in &info.dependencies {
                if let Some(&dep_enum) = self.name_to_component.get(dep_name) {
                    if !included.contains(&dep_enum) {
                        worklist.push(dep_enum);
                    }
                } else {
                    return Err(RegistryError::DependencyError(dep_name.clone()));
                }
            }
        }

        // Convert the included set to a vector in dependency order
        // (components with no dependencies first)
        let mut result = Vec::new();
        let mut remaining = included.clone();

        // First add components with no dependencies
        for component in &included {
            let info = self.get_component_info(*component)?;
            if info.dependencies.is_empty() {
                result.push(*component);
                remaining.remove(component);
            }
        }

        // Then add components whose dependencies are already in the result
        while !remaining.is_empty() {
            let mut added_any = false;
            let components_to_check: Vec<_> = remaining.iter().copied().collect();

            for component in components_to_check {
                let info = self.get_component_info(component)?;
                let deps_satisfied = info.dependencies.iter().all(|dep_name| {
                    if let Some(&dep_enum) = self.name_to_component.get(dep_name) {
                        !remaining.contains(&dep_enum)
                    } else {
                        false
                    }
                });

                if deps_satisfied {
                    result.push(component);
                    remaining.remove(&component);
                    added_any = true;
                }
            }

            if !added_any && !remaining.is_empty() {
                // This should never happen if we validated dependencies correctly
                return Err(RegistryError::DependencyError(
                    "Unexpected dependency resolution failure".to_string(),
                ));
            }
        }

        Ok(result)
    }

    /// Validate that all component dependencies exist and there are no cycles
    fn validate_dependencies(
        components: &HashMap<ShaderComponent, ComponentInfo>,
        name_to_component: &HashMap<String, ShaderComponent>,
    ) -> Result<(), RegistryError> {
        for (component_enum, info) in components {
            let mut visited = HashSet::new();
            let mut path = vec![*component_enum];
            Self::check_cyclic_deps(
                components,
                name_to_component,
                *component_enum,
                &mut visited,
                &mut path,
            )?;
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

    /// Check for cyclic dependencies
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
                        let cycle_str = path
                            .iter()
                            .map(|c| format!("{:?}", c))
                            .collect::<Vec<_>>()
                            .join(" -> ");
                        return Err(RegistryError::CyclicDependency(cycle_str));
                    }
                    if !visited.contains(&dep_enum) {
                        path.push(dep_enum);
                        Self::check_cyclic_deps(
                            components,
                            name_to_component,
                            dep_enum,
                            visited,
                            path,
                        )?;
                        path.pop();
                    }
                }
                // Dependency name validity checked in validate_dependencies
            }
        }
        Ok(())
    }

    /// Check if the required GPU capabilities are available
    pub fn validate_capabilities(
        &self,
        components: &[ShaderComponent],
        available_capabilities: &HashSet<String>,
    ) -> Result<(), RegistryError> {
        for &component in components {
            let info = self.get_component_info(component)?;
            for capability in &info.gpu_capabilities {
                if !available_capabilities.contains(capability) {
                    return Err(RegistryError::MissingCapability(capability.clone()));
                }
            }
        }
        Ok(())
    }
}

/// Placeholder function to get the source code path for a component.
/// TODO: Replace with actual file reading or embedding when ShaderCompiler is built.
///       This might better belong in the ShaderCompiler or be derived from registry data.
pub fn get_component_path(component: ShaderComponent) -> &'static str {
    match component {
        ShaderComponent::EntropyCalculation => {
            "src/shader/shaders/components/entropy_calculation.wgsl"
        }
        ShaderComponent::WorklistManagement => {
            "src/shader/shaders/components/worklist_management.wgsl"
        }
        ShaderComponent::CellCollapse => "src/shader/shaders/components/cell_collapse.wgsl",
        ShaderComponent::ContradictionDetection => {
            "src/shader/shaders/components/contradiction_detection.wgsl"
        }
        ShaderComponent::Utils => "src/shader/shaders/utils.wgsl",
        ShaderComponent::Coords => "src/shader/shaders/coords.wgsl",
        ShaderComponent::Rules => "src/shader/shaders/rules.wgsl",
        ShaderComponent::ShannonEntropy => "src/shader/shaders/components/entropy/shannon.wgsl",
        ShaderComponent::CountBasedEntropy => {
            "src/shader/shaders/components/entropy/count_based.wgsl"
        }
        ShaderComponent::DirectPropagation => {
            "src/shader/shaders/components/propagation/direct.wgsl"
        }
        ShaderComponent::SubgridPropagation => {
            "src/shader/shaders/components/propagation/subgrid.wgsl"
        }
    }
}

/// Placeholder function to define the required components for a given shader type.
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
            ShaderComponent::Utils,  // Basic utilities
            ShaderComponent::Coords, // Coordinate calculations
            ShaderComponent::Rules,  // Adjacency rule evaluation
            ShaderComponent::WorklistManagement,
            ShaderComponent::ContradictionDetection,
            ShaderComponent::DirectPropagation, // Default is direct propagation
        ],
    }
}

impl From<RegistryError> for GpuError {
    fn from(err: RegistryError) -> Self {
        GpuError::shader_error(
            err.to_string(),
            crate::utils::error::GpuErrorContext::default(),
        )
    }
}
