//! Developer-only tooling for Wave Forge: invariant checks, reference rule sets and PNG
//! renderers used by end-to-end tests and for inspecting generated worlds.
//!
//! Nothing here is part of the shipped library (see docs/architecture/overview.md, "Testing
//! requirements"). The crate exists so that both humans and LLM-assisted development can *see* and
//! *mechanically verify* what the generator produced, which is the only practical way to debug a
//! parallel, GPU-driven solver.

pub mod city;
pub mod fixtures;
pub mod invariants;
pub mod models;
pub mod render;

pub use invariants::{BoundaryCondition, TileGrid, Violation, adjacency_violations};
