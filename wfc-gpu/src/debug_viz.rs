//! Debug visualization tools for the Wave Function Collapse algorithm's GPU execution.
//!
//! This module provides functionality to visualize:
//! - Propagation steps: How constraints are applied across the grid
//! - Entropy heatmaps: Visual representation of cell entropy distribution
//! - Contradictions: Where and why contradictions occur during execution

use crate::buffers::{DownloadRequest, GpuBuffers, GpuDownloadResults};
use crate::sync::GpuSynchronizer;
use crate::GpuError;
use pollster;
use std::sync::Arc;

/// Types of debug visualizations available
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisualizationType {
    /// Shows the entropy levels across the grid as a heatmap
    EntropyHeatmap,
    /// Shows propagation steps (which cells are being updated)
    PropagationSteps,
    /// Shows where contradictions occur and affected cells
    Contradictions,
}

/// Configuration options for the debug visualization
#[derive(Debug, Clone)]
pub struct DebugVisualizationConfig {
    /// The type of visualization to generate
    pub viz_type: VisualizationType,
    /// Whether to generate visualization data automatically during algorithm execution
    pub auto_generate: bool,
    /// Maximum number of snapshots to keep in memory
    pub max_snapshots: usize,
    /// Snapshot interval for automatic generation
    pub snapshot_interval: usize,
}

impl Default for DebugVisualizationConfig {
    fn default() -> Self {
        Self {
            viz_type: VisualizationType::EntropyHeatmap,
            auto_generate: false,
            max_snapshots: 100,
            snapshot_interval: 10,
        }
    }
}

/// Snapshot of a particular algorithm state for visualization
#[derive(Debug, Clone)]
pub struct DebugSnapshot {
    /// Timestamp or step number when this snapshot was taken
    pub step: usize,
    /// Entropy values for each cell (if available)
    pub entropy_data: Option<Vec<f32>>,
    /// Grid possibility state (if available)
    pub grid_possibilities: Option<Vec<u32>>,
    /// Recently updated cells (if available)
    pub updated_cells: Option<Vec<u32>>,
    /// Contradiction locations (if any)
    pub contradiction_locations: Option<Vec<u32>>,
    /// Dimensions of the grid (width, height, depth)
    pub dimensions: (usize, usize, usize),
    /// Number of tiles
    pub num_tiles: usize,
}

/// Manages debug visualization for WFC GPU execution
#[derive(Debug, Clone)]
pub struct DebugVisualizer {
    /// Configuration for the visualizer
    config: DebugVisualizationConfig,
    /// Collection of snapshots taken during algorithm execution
    snapshots: Vec<DebugSnapshot>,
    /// Current step counter
    current_step: usize,
    /// Whether the visualizer is enabled
    enabled: bool,
    synchronizer: Arc<GpuSynchronizer>,
}

impl Default for DebugVisualizer {
    fn default() -> Self {
        Self::new(
            DebugVisualizationConfig::default(),
            Arc::new(GpuSynchronizer::default()),
        )
    }
}

impl DebugVisualizer {
    /// Create a new debug visualizer with the given configuration
    pub fn new(config: DebugVisualizationConfig, synchronizer: Arc<GpuSynchronizer>) -> Self {
        Self {
            config,
            snapshots: Vec::new(),
            current_step: 0,
            enabled: true,
            synchronizer,
        }
    }

    /// Create a new debug visualizer with default configuration
    #[deprecated(since = "0.1.0", note = "Use Default::default() instead")]
    pub fn new_default() -> Self {
        Self::default()
    }

    /// Take a snapshot of the current GPU state
    pub async fn take_snapshot(&mut self, buffers: &GpuBuffers) -> Result<(), GpuError> {
        if !self.enabled {
            return Ok(());
        }

        // Download the data we need based on visualization type
        let _needs_entropy = self.config.viz_type == VisualizationType::EntropyHeatmap;
        let needs_grid = true; // Always useful
        let _needs_worklist = self.config.viz_type == VisualizationType::PropagationSteps;
        let needs_contradiction = self.config.viz_type == VisualizationType::Contradictions;

        let request = self.create_download_request(needs_grid, needs_contradiction);

        let result = self
            .synchronizer
            .buffers()
            .download_results(
                self.synchronizer.device().clone(),
                self.synchronizer.queue().clone(),
                request,
            )
            .await?;

        // Create and store the snapshot
        self.add_snapshot(
            buffers.grid_dims,
            buffers.num_tiles,
            result,
            buffers.worklist_buffers.current_worklist_size as u32,
        );

        Ok(())
    }

    /// Helper to create the DownloadRequest based on visualization needs
    fn create_download_request(
        &self,
        needs_grid: bool,
        needs_contradiction: bool,
    ) -> DownloadRequest {
        let _needs_entropy = self.config.viz_type == VisualizationType::EntropyHeatmap;
        DownloadRequest {
            download_entropy: false,
            download_min_entropy_info: self.config.viz_type == VisualizationType::PropagationSteps,
            download_grid_possibilities: needs_grid,
            download_contradiction_location: needs_contradiction,
        }
    }

    /// Helper to get buffer size (example for possibilities buffer)
    fn get_grid_buffer_size(&self, buffers: &GpuBuffers) -> u64 {
        buffers.grid_buffers.grid_possibilities_buf.size()
    }

    /// Add a snapshot from the downloaded results
    fn add_snapshot(
        &mut self,
        dimensions: (usize, usize, usize),
        num_tiles: usize,
        results: GpuDownloadResults,
        worklist_size: u32,
    ) {
        let snapshot = DebugSnapshot {
            step: self.current_step,
            entropy_data: results.entropy,
            grid_possibilities: results.grid_possibilities,
            updated_cells: if worklist_size > 0 {
                Some(vec![0; worklist_size as usize]) // Placeholder - actual data would need worklist download
            } else {
                None
            },
            contradiction_locations: results.contradiction_location.map(|loc| vec![loc]),
            dimensions,
            num_tiles,
        };

        self.snapshots.push(snapshot);

        // Limit the number of snapshots to conserve memory
        while self.snapshots.len() > self.config.max_snapshots {
            self.snapshots.remove(0);
        }

        self.current_step += 1;
    }

    /// Generate an entropy heatmap visualization for the grid
    pub fn generate_entropy_heatmap(&self, snapshot_index: Option<usize>) -> Option<Vec<f32>> {
        let index = snapshot_index.unwrap_or(self.snapshots.len().saturating_sub(1));

        if index >= self.snapshots.len() {
            return None;
        }

        // Simply return the entropy data for now - in a real implementation,
        // this would transform the raw entropy into a more usable visualization format
        self.snapshots[index].entropy_data.clone()
    }

    /// Generate a visualization of the propagation steps
    pub fn generate_propagation_viz(&self, snapshot_index: Option<usize>) -> Option<Vec<bool>> {
        let index = snapshot_index.unwrap_or(self.snapshots.len().saturating_sub(1));

        if index >= self.snapshots.len() || self.snapshots[index].updated_cells.is_none() {
            return None;
        }

        let snapshot = &self.snapshots[index];
        let updated_cells = snapshot.updated_cells.as_ref().unwrap();
        let total_cells = snapshot.dimensions.0 * snapshot.dimensions.1 * snapshot.dimensions.2;

        // Create a boolean array where true means the cell was updated
        let mut result = vec![false; total_cells];
        for &cell_idx in updated_cells {
            if (cell_idx as usize) < total_cells {
                result[cell_idx as usize] = true;
            }
        }

        Some(result)
    }

    /// Generate a visualization of contradictions
    pub fn generate_contradiction_viz(&self, snapshot_index: Option<usize>) -> Option<Vec<bool>> {
        let index = snapshot_index.unwrap_or(self.snapshots.len().saturating_sub(1));

        if index >= self.snapshots.len() || self.snapshots[index].contradiction_locations.is_none()
        {
            return None;
        }

        let snapshot = &self.snapshots[index];
        let contradictions = snapshot.contradiction_locations.as_ref().unwrap();
        let total_cells = snapshot.dimensions.0 * snapshot.dimensions.1 * snapshot.dimensions.2;

        // Create a boolean array where true means a contradiction was detected
        let mut result = vec![false; total_cells];
        for &cell_idx in contradictions {
            if (cell_idx as usize) < total_cells {
                result[cell_idx as usize] = true;
            }
        }

        Some(result)
    }

    /// Check if a snapshot should be taken based on the configured interval
    pub fn should_snapshot(&self, current_iteration: usize) -> bool {
        self.enabled
            && self.config.snapshot_interval > 0
            && current_iteration % self.config.snapshot_interval == 0
    }

    /// Get all stored snapshots
    pub fn get_snapshots(&self) -> &[DebugSnapshot] {
        &self.snapshots
    }

    /// Clear all stored snapshots
    pub fn clear_snapshots(&mut self) {
        self.snapshots.clear();
        self.current_step = 0;
    }

    /// Enable or disable the visualizer
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if the visualizer is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

/// Extension trait for GpuBuffers to add debug visualization capabilities
pub trait GpuBuffersDebugExt {
    /// Take a snapshot of the current GPU state for debugging/visualization
    fn take_debug_snapshot<'a>(
        &'a self,
        visualizer: &'a mut DebugVisualizer,
    ) -> Result<(), GpuError>;
}

impl GpuBuffersDebugExt for GpuBuffers {
    fn take_debug_snapshot<'a>(
        &'a self,
        visualizer: &'a mut DebugVisualizer,
    ) -> Result<(), GpuError> {
        // Use pollster to block on the async take_snapshot call
        // This avoids the Send requirement but makes the call blocking
        pollster::block_on(visualizer.take_snapshot(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffers::{DynamicBufferConfig, GpuBuffers, GpuDownloadResults, WorklistBuffers};
    use crate::sync::GpuSynchronizer;
    use crate::test_utils::{create_test_device_queue /* other utils if needed */};
    use std::sync::Arc;
    use wfc_core::{grid::PossibilityGrid, BoundaryCondition};
    use wfc_rules::AdjacencyRules;

    // Helper function to create a basic GpuBuffers for tests
    fn create_test_gpu_buffers(
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
    ) -> Arc<GpuBuffers> {
        let grid = PossibilityGrid::new(2, 2, 1, 3);
        let rules = AdjacencyRules::new_uniform(3, 6);
        Arc::new(GpuBuffers::new(device, queue, &grid, &rules, BoundaryCondition::Finite).unwrap())
    }

    #[test]
    fn test_debug_visualizer_creation() {
        // Initialize GPU resources for the test
        let (device, queue) = create_test_device_queue();
        let buffers = create_test_gpu_buffers(&device, &queue);
        let synchronizer = Arc::new(GpuSynchronizer::new(device, queue, buffers));

        // Create visualizer with the test synchronizer
        let visualizer = DebugVisualizer::new(
            DebugVisualizationConfig::default(),
            synchronizer, // Pass the initialized synchronizer
        );

        assert!(visualizer.config.enabled);
        assert_eq!(visualizer.snapshots.len(), 0);
        assert_eq!(visualizer.current_step, 0);
        assert!(visualizer.enabled);
    }

    #[test]
    fn test_debug_snapshot_management() {
        // Initialize GPU resources for the test
        let (device, queue) = create_test_device_queue();
        let buffers = create_test_gpu_buffers(&device, &queue);
        let synchronizer = Arc::new(GpuSynchronizer::new(device, queue, buffers));

        // Create visualizer with the test synchronizer
        let mut visualizer = DebugVisualizer::new(
            DebugVisualizationConfig::default(),
            synchronizer.clone(), // Clone Arc for visualizer
        );

        // Add mock snapshots
        for i in 0..5 {
            let mock_results = GpuDownloadResults {
                entropy: Some(vec![i as f32; 10]),
                min_entropy_info: None,
                contradiction_flag: None,
                contradiction_location: None,
                grid_possibilities: Some(vec![0; 10]),
            };
            // Pass the synchronizer needed by add_snapshot
            visualizer.add_snapshot((5, 2, 1), 4, mock_results, synchronizer.clone());
        }

        assert_eq!(visualizer.snapshots.len(), 5);
        assert_eq!(visualizer.current_step, 5);

        // Test snapshot limit
        visualizer.config.max_snapshots = 3;
        let another_mock_results = GpuDownloadResults {
            entropy: Some(vec![5.0; 10]),
            min_entropy_info: None,
            contradiction_flag: None,
            contradiction_location: None,
            grid_possibilities: Some(vec![0; 10]),
        };
        // Pass the synchronizer needed by add_snapshot
        visualizer.add_snapshot((5, 2, 1), 4, another_mock_results, synchronizer.clone());

        // After adding the 6th snapshot with a limit of 3, we should have snapshots 3, 4, 5
        assert_eq!(visualizer.snapshots.len(), 3);
        // First snapshot should now be the one with index 3 from the original sequence
        assert_eq!(visualizer.snapshots[0].step, 3);

        // Test clear
        visualizer.clear_snapshots();
        assert_eq!(visualizer.snapshots.len(), 0);
        assert_eq!(visualizer.current_step, 0);
    }
}
