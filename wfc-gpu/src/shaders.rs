//! Module intended for managing WGSL shader source code.
//!
//! Currently, shaders are loaded directly via `include_str!` in the `pipeline.rs` module.
//! This module could be used in the future to centralize shader loading, potentially
//! allowing for dynamic loading or compilation if needed.

// wfc-gpu/src/shaders.rs
// Placeholder for shader loading/management logic

// pub const ENTROPY_SHADER_SRC: &str = include_str!("entropy.wgsl");
// pub const PROPAGATE_SHADER_SRC: &str = include_str!("propagate.wgsl");

#![allow(dead_code)] // Allow unused enum variants/structs for now

//! Manages WGSL shader source code components and definitions.
//!
//! This module will eventually interact with the ShaderCompiler and ShaderRegistry
//! to load, assemble, and specialize WGSL shaders based on features and requirements.

/// Represents the main types of compute shaders used in the WFC algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderType {
    Entropy,
    Propagation,
    // Add other types like Initialization, Collapse if needed
}

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

/// Placeholder function to get the source code path for a component.
/// TODO: Replace with actual file reading or embedding when ShaderCompiler is built.
pub fn get_component_path(component: ShaderComponent) -> &'static str {
    match component {
        ShaderComponent::EntropyCalculation => "shaders/components/entropy_calculation.wgsl",
        ShaderComponent::WorklistManagement => "shaders/components/worklist_management.wgsl",
        ShaderComponent::CellCollapse => "shaders/components/cell_collapse.wgsl",
        ShaderComponent::ContradictionDetection => {
            "shaders/components/contradiction_detection.wgsl"
        }
        ShaderComponent::Utils => "shaders/utils.wgsl",
        ShaderComponent::Coords => "shaders/coords.wgsl",
        ShaderComponent::Rules => "shaders/rules.wgsl",
        // ShaderComponent::Atomics => "shaders/features/atomics.wgsl",
        // ShaderComponent::NoAtomics => "shaders/features/no_atomics.wgsl",
    }
}

/// Placeholder function to define the required components for a given shader type.
/// TODO: This logic will be more sophisticated, considering features, in the ShaderRegistry.
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
