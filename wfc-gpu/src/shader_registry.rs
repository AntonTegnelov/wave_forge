#![allow(dead_code, unused_variables)] // Allow unused items during development

//! Manages metadata about shader components, features, and their relationships.
//!
//! This module will load information (potentially from a manifest file like registry.json)
//! about available shader components, their dependencies, and the features they require or provide.
//! It will be used by the ShaderCompiler to determine which components to assemble for a given shader variant.

use crate::shaders::{ShaderComponent, ShaderType};
use crate::GpuError; // Or define a specific RegistryError
use std::collections::{HashMap, HashSet};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum RegistryError {
    #[error("Component not found in registry: {0:?}")]
    ComponentNotFound(ShaderComponent),
    #[error("Feature conflict or missing dependency for component {0:?}")]
    DependencyError(ShaderComponent),
    #[error("Failed to load or parse registry data: {0}")]
    LoadError(String),
}

/// Placeholder for metadata about a single shader component.
#[derive(Debug, Clone)]
pub struct ComponentInfo {
    path: String,
    dependencies: Vec<ShaderComponent>, // Other components this one #includes or depends on
    required_features: HashSet<String>, // Features that must be enabled to use this component
    provided_features: HashSet<String>, // Features this component provides (e.g., "atomics_impl")
}

/// Manages the registry of available shader components and features.
#[derive(Debug, Default)]
pub struct ShaderRegistry {
    components: HashMap<ShaderComponent, ComponentInfo>,
    // TODO: Add fields for feature definitions, etc.
}

impl ShaderRegistry {
    /// Creates a new, empty ShaderRegistry.
    pub fn new() -> Self {
        // TODO: Implement loading from a manifest file (e.g., registry.json) here.
        println!("Creating new ShaderRegistry (TODO: Implement loading)");
        Self::default()
    }

    /// Placeholder for registering a component (manual registration).
    pub fn register_component(&mut self, component: ShaderComponent, info: ComponentInfo) {
        println!("Registering component {:?}: {:?}", component, info);
        self.components.insert(component, info);
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
    ///       It should replace `shaders::get_required_components`.
    pub fn get_shader_variant_components(
        &self,
        shader_type: ShaderType,
        features: &[&str],
    ) -> Result<Vec<ShaderComponent>, RegistryError> {
        println!(
            "Getting components for {:?} with features {:?} (using placeholder logic)",
            shader_type, features
        );
        // Placeholder: Return the same components as the simple function in shaders.rs for now.
        let basic_components = crate::shaders::get_required_components(shader_type);

        // TODO: Implement logic based on self.components, dependencies, and features.
        // - Start with base components for shader_type.
        // - Recursively add dependencies.
        // - Filter components based on required/provided features.
        // - Check for conflicts.
        // - Return the final ordered list of components.

        Ok(basic_components)
    }
}

// Map RegistryError to GpuError if needed
impl From<RegistryError> for GpuError {
    fn from(err: RegistryError) -> Self {
        // Choose an appropriate GpuError variant, e.g., ShaderError or a new one
        GpuError::ShaderError(format!("Shader Registry Error: {}", err))
    }
}
