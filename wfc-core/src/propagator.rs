use crate::grid::PossibilityGrid;
use crate::rules::AdjacencyRules;
use std::default::Default;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PropagationError {
    #[error("Contradiction detected during propagation")]
    Contradiction,
    // Add other specific propagation errors later
}

#[must_use]
pub trait ConstraintPropagator {
    fn propagate(
        &mut self,
        grid: &mut PossibilityGrid,
        updated_coords: Vec<(usize, usize, usize)>,
        rules: &AdjacencyRules,
    ) -> Result<(), PropagationError>;
}

// Basic CPU implementation
#[derive(Debug, Clone)]
pub struct CpuConstraintPropagator;

impl CpuConstraintPropagator {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuConstraintPropagator {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstraintPropagator for CpuConstraintPropagator {
    fn propagate(
        &mut self,
        _grid: &mut PossibilityGrid,
        _updated_coords: Vec<(usize, usize, usize)>,
        _rules: &AdjacencyRules,
    ) -> Result<(), PropagationError> {
        // TODO: Implement CPU propagation logic using rules
        // TODO: Explore rayon for parallelism
        todo!()
    }
}
