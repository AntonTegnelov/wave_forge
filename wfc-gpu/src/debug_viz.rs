//! Debug visualization tools for the Wave Function Collapse algorithm's GPU execution.
//!
//! This module provides functionality to visualize:
//! - Propagation steps: How constraints are applied across the grid
//! - Entropy heatmaps: Visual representation of cell entropy distribution
//! - Contradictions: Where and why contradictions occur during execution

use crate::{
    backend::GpuBackend,
    buffers::{DownloadRequest, GpuBuffers, GpuDownloadResults},
    pipeline::ComputePipelines,
    shader_registry::ShaderRegistry,
    sync::GpuSynchronizer,
    GpuError,
};
use image::{ImageBuffer, Rgba, RgbaImage};
use log::trace;
use std::sync::Arc;
use std::time::{Duration, Instant};
use wfc_core::{
    grid::{GridDefinition, PossibilityGrid},
    TileId,
};
use wfc_rules::AdjacencyRules;

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
    last_update: Instant,
    update_interval: Duration,
    show_entropy: bool,
    show_min_entropy_index: bool,
    last_entropy_values: Option<Vec<f32>>,
    last_min_entropy_index: Option<Vec<u32>>,
    grid_dims: (usize, usize, usize),
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
            last_update: Instant::now(),
            update_interval: Duration::from_secs(1),
            show_entropy: false,
            show_min_entropy_index: false,
            last_entropy_values: None,
            last_min_entropy_index: None,
            grid_dims: (0, 0, 0),
        }
    }

    /// Create a new debug visualizer with default configuration
    #[deprecated(since = "0.1.0", note = "Use Default::default() instead")]
    pub fn new_default() -> Self {
        Self::default()
    }

    /// Take a snapshot of the current GPU state asynchronously
    /// Marked _visualizer as unused
    pub async fn take_snapshot(
        _visualizer: &mut DebugVisualizer,
        buffers: &GpuBuffers, // Pass buffers back in
    ) -> Result<(), GpuError> {
        if !_visualizer.enabled {
            return Ok(());
        }

        // Download the data we need based on visualization type
        let _needs_entropy = _visualizer.config.viz_type == VisualizationType::EntropyHeatmap;
        let needs_grid = true; // Always useful
        let _needs_worklist = _visualizer.config.viz_type == VisualizationType::PropagationSteps;
        let needs_contradiction = _visualizer.config.viz_type == VisualizationType::Contradictions;

        let request = _visualizer.create_download_request(needs_grid, needs_contradiction);

        let worklist_size = _visualizer
            .synchronizer
            .buffers()
            .worklist_buffers
            .current_worklist_size as u32;

        let result = _visualizer
            .synchronizer
            .buffers()
            .download_results(
                _visualizer.synchronizer.device().clone(),
                _visualizer.synchronizer.queue().clone(),
                request,
            )
            .await?;

        // Create and store the snapshot
        _visualizer.add_snapshot(buffers.grid_dims, buffers.num_tiles, result, worklist_size);

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
    pub fn add_snapshot(
        &mut self,
        dimensions: (usize, usize, usize),
        num_tiles: usize,
        results: GpuDownloadResults,
        _worklist_size: u32,
    ) {
        let snapshot = DebugSnapshot {
            step: self.current_step,
            entropy_data: results.entropy,
            grid_possibilities: results.grid_possibilities,
            updated_cells: if _worklist_size > 0 {
                Some(vec![0; _worklist_size as usize]) // Placeholder - actual data would need worklist download
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

    // Prefix unused parameter
    fn render_propagation_steps(_snapshot: &DebugSnapshot) -> Vec<u8> {
        // Placeholder: Render propagation steps (e.g., highlight updated cells)
        vec![0; 100] // Return dummy data
    }

    // Prefix unused parameter
    fn render_contradictions(_snapshot: &DebugSnapshot) -> Vec<u8> {
        // Placeholder: Render contradictions (e.g., highlight contradicted cells)
        vec![0; 100] // Return dummy data
    }

    // Prefix unused parameter
    fn get_snapshot_by_index(&self, _index: Option<usize>) -> Option<&DebugSnapshot> {
        // Placeholder: Get snapshot by index or latest
        self.snapshots.last()
    }

    /// Updates the visualizer with the current GPU state.
    ///
    /// This might involve downloading data from the GPU, which can be slow.
    pub async fn update(
        &mut self,
        backend: &dyn GpuBackend, // Use trait object
        // grid_state: &PossibilityGrid, // Pass PossibilityGrid if needed for CPU-side rendering
        gpu_buffers: &GpuBuffers, // Pass GPU buffers
    ) -> Result<(), GpuError> {
        let start = Instant::now();
        trace!("Updating debug visualizer...");

        if self.last_update.elapsed() < self.update_interval {
            return Ok(());
        }

        // Example: Download entropy data if configured
        if self.show_entropy {
            let request = DownloadRequest {
                download_entropy: true,
                ..Default::default()
            };
            match gpu_buffers.download_results(request).await? {
                results if results.entropy.is_some() => {
                    self.last_entropy_values = results.entropy;
                    trace!("Downloaded entropy data for visualization.");
                }
                _ => {
                    return Err(GpuError::InternalError(
                        "Entropy data not found in download results".to_string(),
                    ))
                }
            }
        }

        // Example: Download min entropy index if configured
        if self.show_min_entropy_index {
            let request = DownloadRequest {
                download_min_entropy_info: true,
                ..Default::default()
            };
            match gpu_buffers.download_results(request).await? {
                results if results.min_entropy_info.is_some() => {
                    self.last_min_entropy_index = results.min_entropy_info;
                    trace!("Downloaded min entropy index data for visualization.");
                }
                _ => {
                    return Err(GpuError::InternalError(
                        "Min entropy index data not found in download results".to_string(),
                    ))
                }
            }
        }

        // TODO: Add rendering logic based on downloaded data (e.g., create an image)
        // Example: Generate an image based on self.last_entropy_values
        // let image = self.generate_visualization_image(grid_state)?; // Pass grid if needed

        self.last_update = Instant::now();
        trace!("Debug visualizer updated in {:?}", start.elapsed());
        Ok(())
    }

    // Placeholder for a function to generate the actual visualization (e.g., an image)
    // This would use the downloaded data (last_entropy_values, etc.) and potentially the grid_state
    fn generate_visualization_image(
        &self, /* grid: &PossibilityGrid */
    ) -> Result<RgbaImage, GpuError> {
        // Implementation depends on what you want to visualize
        // Example: Create a simple image based on entropy
        let width = self.grid_dims.0;
        let height = self.grid_dims.1;
        let mut img = RgbaImage::new(width as u32, height as u32);

        if let Some(entropy_values) = &self.last_entropy_values {
            // Find min/max entropy for normalization
            let min_entropy = entropy_values
                .iter()
                .copied()
                .filter(|&e| e.is_finite())
                .fold(f32::INFINITY, f32::min);
            let max_entropy = entropy_values
                .iter()
                .copied()
                .filter(|&e| e.is_finite())
                .fold(f32::NEG_INFINITY, f32::max);
            let range = max_entropy - min_entropy;

            for y in 0..height {
                for x in 0..width {
                    let index = y * width + x;
                    if index < entropy_values.len() {
                        let entropy = entropy_values[index];
                        let normalized = if range > 1e-6 {
                            (entropy - min_entropy) / range
                        } else {
                            0.0
                        };
                        let gray = (normalized.max(0.0).min(1.0) * 255.0) as u8;
                        img.put_pixel(x as u32, y as u32, Rgba([gray, gray, gray, 255]));
                    } else {
                        img.put_pixel(x as u32, y as u32, Rgba([255, 0, 0, 255]));
                        // Error color
                    }
                }
            }
        }

        Ok(img)
    }
}

/// Trait extension for GpuBuffers specific to debug visualization
pub trait GpuBuffersDebugExt {
    /// Takes a debug snapshot using the visualizer.
    /// Marked _visualizer as unused
    fn take_debug_snapshot(&self, _visualizer: &mut DebugVisualizer) -> Result<(), GpuError>;
}

// Implementation remains commented out as take_snapshot is async
// impl GpuBuffersDebugExt for GpuBuffers {
//     fn take_debug_snapshot(&self, visualizer: &mut DebugVisualizer) -> Result<(), GpuError> {
//         pollster::block_on(visualizer.take_snapshot(self))
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{MockGpuBackend, MockGpuDevice, MockGpuQueue};
    use crate::sync::GpuSynchronizer;
    use std::sync::Arc;
    use wfc_core::{
        grid::{GridDefinition, PossibilityGrid},
        BoundaryCondition,
    };
    use wfc_rules::AdjacencyRules;

    // Mock setup helper
    async fn setup_test_environment() -> (
        Arc<MockGpuBackend>,
        Arc<GpuSynchronizer>,
        GridDefinition,
        Arc<GpuBuffers>,
    ) {
        let backend = Arc::new(MockGpuBackend::new());
        let device = Arc::new(MockGpuDevice::new()); // Mock device
        let queue = Arc::new(MockGpuQueue::new()); // Mock queue
        let grid_def = GridDefinition {
            dims: (10, 10, 1),
            num_tiles: 5,
        };

        // Create dummy grid/rules for buffer creation
        let dummy_grid = PossibilityGrid::new(
            grid_def.dims.0,
            grid_def.dims.1,
            grid_def.dims.2,
            grid_def.num_tiles,
        );
        let dummy_rules = AdjacencyRules::new(grid_def.num_tiles, 6);

        // Call GpuBuffers::new with mock device/queue references and dummy data
        let buffers = Arc::new(
            GpuBuffers::new(
                &device, // Pass reference
                &queue,  // Pass reference
                &dummy_grid,
                &dummy_rules,
                BoundaryCondition::Finite,
            )
            .expect("Failed to create mock GpuBuffers"),
        );

        // Call GpuSynchronizer::new with mock device/queue references and buffers Arc
        let sync = Arc::new(GpuSynchronizer::new(&device, &queue, buffers.clone())); // Pass references

        (backend, sync, grid_def, buffers)
    }

    #[tokio::test]
    async fn test_visualizer_creation() {
        let (_backend, sync, grid_def, _buffers) = setup_test_environment().await;
        let config = DebugVisualizationConfig::default();
        let visualizer = DebugVisualizer::new(config, sync);
        assert!(visualizer.enabled);
    }

    #[tokio::test]
    async fn test_visualizer_update_throttling() {
        let (backend, sync, grid_def, mock_buffers) = setup_test_environment().await;
        let mut config = DebugVisualizationConfig::default();
        config.update_interval = Duration::from_millis(100);

        let mut visualizer = DebugVisualizer::new(config, sync.clone());

        let result1 = visualizer.update(&*backend, &mock_buffers).await;
        assert!(result1.is_ok());
        let update_time1 = visualizer.last_update;

        let result2 = visualizer.update(&*backend, &mock_buffers).await;
        assert!(result2.is_ok());
        let update_time2 = visualizer.last_update;

        assert_eq!(update_time1, update_time2);

        tokio::time::sleep(Duration::from_millis(150)).await;
        let result3 = visualizer.update(&*backend, &mock_buffers).await;
        assert!(result3.is_ok());
        let update_time3 = visualizer.last_update;

        assert!(update_time3 > update_time2);
    }
}
