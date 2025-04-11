use crate::{
    buffers::{GpuBuffers, GpuEntropyShaderParams},
    gpu::sync::GpuSynchronizer,
    GpuError,
};
use std::sync::Arc;
use wfc_core::entropy::EntropyError as CoreEntropyError;
use wfc_core::{
    entropy::EntropyHeuristicType,
    grid::{EntropyGrid, PossibilityGrid},
};

/// Strategy trait for calculating entropy and selecting cells in WFC algorithm.
/// This strategy pattern allows for different entropy calculation algorithms
/// to be swapped in without changing the core implementation.
pub trait EntropyStrategy: Send + Sync {
    /// Get the type of entropy heuristic this strategy implements
    fn heuristic_type(&self) -> EntropyHeuristicType;

    /// Configure shader parameters for this specific entropy strategy
    fn configure_shader_params(&self, params: &mut GpuEntropyShaderParams);

    /// Prepare to calculate entropy (any strategy-specific initialization)
    fn prepare(&self, synchronizer: &GpuSynchronizer) -> Result<(), CoreEntropyError>;

    /// Upload strategy-specific data to GPU if needed
    fn upload_data(&self, synchronizer: &GpuSynchronizer) -> Result<(), CoreEntropyError>;

    /// Any post-processing needed after shader execution
    fn post_process(&self, synchronizer: &GpuSynchronizer) -> Result<(), CoreEntropyError>;
}

/// Shannon entropy calculation strategy - uses information theory entropy
/// calculation for cell selection. This is the most "correct" entropy measure
/// but can be more computationally expensive.
pub struct ShannonEntropyStrategy {
    num_tiles: usize,
    u32s_per_cell: usize,
}

impl ShannonEntropyStrategy {
    pub fn new(num_tiles: usize, u32s_per_cell: usize) -> Self {
        Self {
            num_tiles,
            u32s_per_cell,
        }
    }
}

impl EntropyStrategy for ShannonEntropyStrategy {
    fn heuristic_type(&self) -> EntropyHeuristicType {
        EntropyHeuristicType::Shannon
    }

    fn configure_shader_params(&self, params: &mut GpuEntropyShaderParams) {
        params.heuristic_type = 0; // Shannon entropy type in shader
        params.num_tiles = self.num_tiles as u32;
        params.u32s_per_cell = self.u32s_per_cell as u32;
    }

    fn prepare(&self, synchronizer: &GpuSynchronizer) -> Result<(), CoreEntropyError> {
        // Shannon entropy doesn't need special preparation
        Ok(())
    }

    fn upload_data(&self, _synchronizer: &GpuSynchronizer) -> Result<(), CoreEntropyError> {
        // Shannon entropy doesn't need special data uploads
        Ok(())
    }

    fn post_process(&self, _synchronizer: &GpuSynchronizer) -> Result<(), CoreEntropyError> {
        // Shannon entropy doesn't need special post-processing
        Ok(())
    }
}

/// Count-based entropy strategy - uses the count of possible tiles
/// as the entropy measure. Simpler than Shannon but still effective.
pub struct CountEntropyStrategy {
    num_tiles: usize,
    u32s_per_cell: usize,
}

impl CountEntropyStrategy {
    pub fn new(num_tiles: usize, u32s_per_cell: usize) -> Self {
        Self {
            num_tiles,
            u32s_per_cell,
        }
    }
}

impl EntropyStrategy for CountEntropyStrategy {
    fn heuristic_type(&self) -> EntropyHeuristicType {
        EntropyHeuristicType::Count
    }

    fn configure_shader_params(&self, params: &mut GpuEntropyShaderParams) {
        params.heuristic_type = 1; // Count entropy type in shader
        params.num_tiles = self.num_tiles as u32;
        params.u32s_per_cell = self.u32s_per_cell as u32;
    }

    fn prepare(&self, synchronizer: &GpuSynchronizer) -> Result<(), CoreEntropyError> {
        // Count entropy doesn't need special preparation
        Ok(())
    }

    fn upload_data(&self, _synchronizer: &GpuSynchronizer) -> Result<(), CoreEntropyError> {
        // Count entropy doesn't need special data uploads
        Ok(())
    }

    fn post_process(&self, _synchronizer: &GpuSynchronizer) -> Result<(), CoreEntropyError> {
        // Count entropy doesn't need special post-processing
        Ok(())
    }
}

/// Simple count-based entropy strategy - uses just the raw count
/// of possible tiles without normalization.
pub struct CountSimpleEntropyStrategy {
    num_tiles: usize,
    u32s_per_cell: usize,
}

impl CountSimpleEntropyStrategy {
    pub fn new(num_tiles: usize, u32s_per_cell: usize) -> Self {
        Self {
            num_tiles,
            u32s_per_cell,
        }
    }
}

impl EntropyStrategy for CountSimpleEntropyStrategy {
    fn heuristic_type(&self) -> EntropyHeuristicType {
        EntropyHeuristicType::CountSimple
    }

    fn configure_shader_params(&self, params: &mut GpuEntropyShaderParams) {
        params.heuristic_type = 2; // Count simple entropy type in shader
        params.num_tiles = self.num_tiles as u32;
        params.u32s_per_cell = self.u32s_per_cell as u32;
    }

    fn prepare(&self, synchronizer: &GpuSynchronizer) -> Result<(), CoreEntropyError> {
        // Count simple entropy doesn't need special preparation
        Ok(())
    }

    fn upload_data(&self, _synchronizer: &GpuSynchronizer) -> Result<(), CoreEntropyError> {
        // Count simple entropy doesn't need special data uploads
        Ok(())
    }

    fn post_process(&self, _synchronizer: &GpuSynchronizer) -> Result<(), CoreEntropyError> {
        // Count simple entropy doesn't need special post-processing
        Ok(())
    }
}

/// Weighted count entropy strategy - takes into account tile weights
/// when calculating entropy.
pub struct WeightedCountEntropyStrategy {
    num_tiles: usize,
    u32s_per_cell: usize,
    // Would typically include tile weights here
}

impl WeightedCountEntropyStrategy {
    pub fn new(num_tiles: usize, u32s_per_cell: usize) -> Self {
        Self {
            num_tiles,
            u32s_per_cell,
        }
    }
}

impl EntropyStrategy for WeightedCountEntropyStrategy {
    fn heuristic_type(&self) -> EntropyHeuristicType {
        EntropyHeuristicType::WeightedCount
    }

    fn configure_shader_params(&self, params: &mut GpuEntropyShaderParams) {
        params.heuristic_type = 3; // Weighted count entropy type in shader
        params.num_tiles = self.num_tiles as u32;
        params.u32s_per_cell = self.u32s_per_cell as u32;
    }

    fn prepare(&self, synchronizer: &GpuSynchronizer) -> Result<(), CoreEntropyError> {
        // Weighted count entropy might need special preparation for weights
        Ok(())
    }

    fn upload_data(&self, _synchronizer: &GpuSynchronizer) -> Result<(), CoreEntropyError> {
        // For a full implementation, would upload weight data here
        Ok(())
    }

    fn post_process(&self, _synchronizer: &GpuSynchronizer) -> Result<(), CoreEntropyError> {
        // Weighted count entropy doesn't need special post-processing
        Ok(())
    }
}

/// Factory for creating entropy strategy instances based on heuristic type
pub struct EntropyStrategyFactory;

impl EntropyStrategyFactory {
    /// Create a new entropy strategy based on the specified heuristic type
    pub fn create_strategy(
        heuristic_type: EntropyHeuristicType,
        num_tiles: usize,
        u32s_per_cell: usize,
    ) -> Box<dyn EntropyStrategy> {
        match heuristic_type {
            EntropyHeuristicType::Shannon => {
                Box::new(ShannonEntropyStrategy::new(num_tiles, u32s_per_cell))
            }
            EntropyHeuristicType::Count => {
                Box::new(CountEntropyStrategy::new(num_tiles, u32s_per_cell))
            }
            EntropyHeuristicType::CountSimple => {
                Box::new(CountSimpleEntropyStrategy::new(num_tiles, u32s_per_cell))
            }
            EntropyHeuristicType::WeightedCount => {
                Box::new(WeightedCountEntropyStrategy::new(num_tiles, u32s_per_cell))
            }
        }
    }
}
