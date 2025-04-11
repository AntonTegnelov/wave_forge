#![allow(dead_code, unused_variables)] // Allow unused items during development

//! Responsible for assembling WGSL shader components into complete, specialized shaders.
//!
//! This module takes shader component sources, handles includes, applies feature flags,
//! and replaces specialization constants to produce final WGSL code ready for compilation
//! by the WGPU backend.

use crate::GpuError; // Or define a specific CompilationError
                     // Import from new location
use crate::shader::shader_registry::{self, ShaderComponent};
use crate::shader::shaders::ShaderType;
use std::collections::HashMap;
use std::collections::HashSet;
use thiserror::Error;

// Rest of the code remains unchanged
