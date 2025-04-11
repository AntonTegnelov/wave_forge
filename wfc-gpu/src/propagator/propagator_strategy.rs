//! Propagation strategy implementations for the WFC algorithm.
//! This module provides different strategies for propagating constraints
//! after a cell collapse in the Wave Function Collapse algorithm.

use crate::{
    buffers::GpuBuffers,
    gpu::sync::GpuSynchronizer,
    utils::error_recovery::{GpuError, GridCoord},
};
use std::sync::Arc;
use wfc_core::{grid::PossibilityGrid, propagator::PropagationError};

/// Strategy trait for constraint propagation in WFC algorithm.
/// This defines the interface for different propagation strategies,
/// allowing the propagator to be adapted for different use cases.
pub trait PropagationStrategy: Send + Sync {
    /// Get the name of this propagation strategy
    fn name(&self) -> &str;

    /// Prepare for propagation by initializing any necessary buffers or state
    fn prepare(&self, synchronizer: &GpuSynchronizer) -> Result<(), PropagationError>;

    /// Propagate constraints from the specified cells
    fn propagate(
        &self,
        grid: &mut PossibilityGrid,
        updated_cells: &[GridCoord],
        buffers: &Arc<GpuBuffers>,
        synchronizer: &GpuSynchronizer,
    ) -> Result<(), PropagationError>;

    /// Clean up any resources used during propagation
    fn cleanup(&self, synchronizer: &GpuSynchronizer) -> Result<(), PropagationError>;
}

/// Direct propagation strategy - propagates constraints directly across
/// the entire grid without any partitioning or optimization.
pub struct DirectPropagationStrategy {
    name: String,
    max_iterations: u32,
}

impl DirectPropagationStrategy {
    /// Create a new direct propagation strategy
    pub fn new(max_iterations: u32) -> Self {
        Self {
            name: "Direct Propagation".to_string(),
            max_iterations,
        }
    }

    /// Create a new strategy with default settings
    pub fn default() -> Self {
        Self::new(1000)
    }
}

impl PropagationStrategy for DirectPropagationStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn prepare(&self, _synchronizer: &GpuSynchronizer) -> Result<(), PropagationError> {
        // Direct propagation doesn't need special preparation
        Ok(())
    }

    fn propagate(
        &self,
        _grid: &mut PossibilityGrid,
        updated_cells: &[GridCoord],
        buffers: &Arc<GpuBuffers>,
        synchronizer: &GpuSynchronizer,
    ) -> Result<(), PropagationError> {
        // Convert GridCoord to indices
        let updated_indices: Vec<u32> = updated_cells
            .iter()
            .map(|coord| {
                (coord.x + coord.y * _grid.width + coord.z * _grid.width * _grid.height) as u32
            })
            .collect();

        // Skip propagation if no cells were updated
        if updated_indices.is_empty() {
            return Ok(());
        }

        let device = &synchronizer.device();
        let queue = &synchronizer.queue();
        let label = "DirectPropagation";

        log::debug!(
            "Beginning direct constraint propagation with {} updated cells",
            updated_indices.len()
        );

        // Reset contradiction flag
        queue.write_buffer(
            &buffers.contradiction_flag_buf,
            0,
            bytemuck::cast_slice(&[0u32]),
        );

        // Initialize the worklist with the updated indices
        let max_worklist_size = buffers.worklist_buffers.max_worklist_size;

        // Ensure we don't exceed the buffer size
        if updated_indices.len() > max_worklist_size {
            return Err(PropagationError::InternalError(format!(
                "Worklist size ({}) exceeds maximum allowable size ({})",
                updated_indices.len(),
                max_worklist_size
            )));
        }

        // Initialize with current index being worklist A - using a local variable
        // since we don't have access to the atomic
        let mut current_worklist_idx = 0;

        // Write the initial worklist
        queue.write_buffer(
            &buffers.worklist_buffers.worklist_buf_a,
            0,
            bytemuck::cast_slice(&updated_indices),
        );
        queue.write_buffer(
            &buffers.worklist_buffers.worklist_count_buf,
            0,
            bytemuck::cast_slice(&[updated_indices.len() as u32]),
        );

        // Download buffers for checking propagation termination conditions
        let mut contradiction_flag = [0u32; 1];
        let mut worklist_count = [0u32; 1];

        let mut consecutive_low_impact_passes: u32 = 0;
        let mut pass_idx = 0;

        // Ping-pong between worklists until propagation completes
        loop {
            pass_idx += 1;

            // Create a bind group based on the current input/output worklists
            let bind_group =
                self.create_propagation_bind_group_for_pass(device, buffers, current_worklist_idx);

            // Start timestamp for this pass
            let timestamp_start = std::time::Instant::now();

            // Create a command encoder for this pass
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("PropagationPass-{}", pass_idx)),
            });

            {
                // Begin a compute pass for propagation
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("{}ComputePass-{}", label, pass_idx)),
                    timestamp_writes: None,
                });

                // Set pipeline and bind groups
                compute_pass.set_pipeline(&buffers.pipelines.propagation_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);

                // Dispatch based on worklist size
                queue.submit(Some(encoder.finish()));

                // Read worklist count before dispatch to determine workgroup count
                worklist_count = synchronizer
                    .download_buffer(
                        &buffers.worklist_buffers.worklist_count_buf,
                        0,
                        std::mem::size_of::<u32>(),
                    )
                    .map_err(|e| {
                        PropagationError::InternalError(format!(
                            "Failed to download worklist count: {}",
                            e
                        ))
                    })?;

                let current_worklist_count = worklist_count[0] as usize;

                // Start a new encoder and compute pass with the right workgroup size
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("PropagationDispatchPass-{}", pass_idx)),
                });

                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some(&format!("{}DispatchPass-{}", label, pass_idx)),
                            timestamp_writes: None,
                        });

                    compute_pass.set_pipeline(&buffers.pipelines.propagation_pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);

                    // Calculate workgroup count
                    let workgroup_size = 64; // Must match the shader
                    let workgroups =
                        (current_worklist_count as f32 / workgroup_size as f32).ceil() as u32;

                    if workgroups > 0 {
                        // Only dispatch if we have items in the worklist
                        compute_pass.dispatch_workgroups(workgroups, 1, 1);
                    }
                }

                // Submit the dispatch command buffer
                queue.submit(Some(encoder.finish()));
            }

            // Check for contradiction after each pass
            contradiction_flag = synchronizer
                .download_buffer(&buffers.contradiction_flag_buf, 0, 4)
                .map_err(|e| {
                    PropagationError::InternalError(format!(
                        "Failed to download contradiction flag: {}",
                        e
                    ))
                })?;

            if contradiction_flag[0] != 0 {
                return Err(PropagationError::Contradiction);
            }

            // Check the output worklist size to determine whether to continue
            let output_idx = 1 - current_worklist_idx;
            let output_worklist_count_buf = &buffers.worklist_buffers.worklist_count_buf;

            // Download the output worklist count
            worklist_count = synchronizer
                .download_buffer(output_worklist_count_buf, 0, 4)
                .map_err(|e| {
                    PropagationError::InternalError(format!(
                        "Failed to download worklist count: {}",
                        e
                    ))
                })?;

            let output_worklist_count = worklist_count[0] as u32;

            // Calculate pass duration
            let pass_duration = timestamp_start.elapsed();
            log::debug!(
                "Propagation pass {} completed in {:?} with {} affected cells",
                pass_idx,
                pass_duration,
                output_worklist_count
            );

            // Update the worklist index for the next pass
            current_worklist_idx = output_idx;

            // Early termination check
            if output_worklist_count <= self.max_iterations / 100 {
                // Use percentage of max iterations as threshold
                consecutive_low_impact_passes += 1;
                if consecutive_low_impact_passes >= 3 {
                    // Use fixed value for consecutive passes
                    log::debug!(
                        "Early termination after {} consecutive low-impact passes",
                        consecutive_low_impact_passes
                    );
                    break;
                }
            } else {
                consecutive_low_impact_passes = 0;
            }

            // Done if no more cells to propagate
            if output_worklist_count == 0 {
                break;
            }

            // Safety limit on propagation passes
            if pass_idx > self.max_iterations {
                return Err(PropagationError::InternalError(format!(
                    "Propagation exceeded {} passes, possible infinite loop",
                    self.max_iterations
                )));
            }
        }

        log::debug!("Direct propagation complete after {} passes", pass_idx);
        Ok(())
    }

    fn cleanup(&self, _synchronizer: &GpuSynchronizer) -> Result<(), PropagationError> {
        // Direct propagation doesn't need special cleanup
        Ok(())
    }
}

impl DirectPropagationStrategy {
    // Helper method to create a bind group for a propagation pass
    fn create_propagation_bind_group_for_pass(
        &self,
        device: &wgpu::Device,
        buffers: &GpuBuffers,
        current_worklist_idx: usize,
    ) -> wgpu::BindGroup {
        // Determine input and output binding resources based on current worklist index
        let (input_worklist, output_worklist) = if current_worklist_idx == 0 {
            (
                buffers.worklist_buffers.worklist_buf_a.as_entire_binding(),
                buffers.worklist_buffers.worklist_buf_b.as_entire_binding(),
            )
        } else {
            (
                buffers.worklist_buffers.worklist_buf_b.as_entire_binding(),
                buffers.worklist_buffers.worklist_buf_a.as_entire_binding(),
            )
        };

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PropagationBindGroup"),
            layout: &buffers.bind_group_layouts.propagation_bind_group_layout,
            entries: &[
                // Grid possibilities buffer
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers
                        .grid_buffers
                        .grid_possibilities_buf
                        .as_entire_binding(),
                },
                // Adjacency rules buffer
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.rule_buffers.adjacency_rules_buf.as_entire_binding(),
                },
                // Input worklist buffer
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: input_worklist,
                },
                // Output worklist buffer
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_worklist,
                },
                // Input worklist count buffer
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers
                        .worklist_buffers
                        .worklist_count_buf
                        .as_entire_binding(),
                },
                // Output worklist count buffer (same buffer as input, will be reset)
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers
                        .worklist_buffers
                        .worklist_count_buf
                        .as_entire_binding(),
                },
                // Contradiction flag buffer
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buffers.contradiction_flag_buf.as_entire_binding(),
                },
                // Grid params uniform buffer
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buffers.params_buf.as_entire_binding(),
                },
            ],
        })
    }
}

/// Subgrid propagation strategy - divides the grid into smaller subgrids
/// to optimize propagation for large grids.
pub struct SubgridPropagationStrategy {
    name: String,
    max_iterations: u32,
    subgrid_size: u32,
}

impl SubgridPropagationStrategy {
    /// Create a new subgrid propagation strategy
    pub fn new(max_iterations: u32, subgrid_size: u32) -> Self {
        Self {
            name: "Subgrid Propagation".to_string(),
            max_iterations,
            subgrid_size,
        }
    }

    /// Create a new strategy with default settings
    pub fn default() -> Self {
        Self::new(1000, 16)
    }
}

impl PropagationStrategy for SubgridPropagationStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn prepare(&self, _synchronizer: &GpuSynchronizer) -> Result<(), PropagationError> {
        // Subgrid propagation may need to initialize subgrid buffers
        // Placeholder - will be implemented when extracting logic from subgrid.rs
        Ok(())
    }

    fn propagate(
        &self,
        _grid: &mut PossibilityGrid,
        _updated_cells: &[GridCoord],
        _buffers: &Arc<GpuBuffers>,
        _synchronizer: &GpuSynchronizer,
    ) -> Result<(), PropagationError> {
        // Placeholder - will be implemented when extracting logic from propagator.rs and subgrid.rs
        Ok(())
    }

    fn cleanup(&self, _synchronizer: &GpuSynchronizer) -> Result<(), PropagationError> {
        // Subgrid propagation may need to clean up temporary buffers
        Ok(())
    }
}

/// Adaptive propagation strategy - automatically selects the best strategy
/// based on grid size and other factors.
pub struct AdaptivePropagationStrategy {
    name: String,
    direct_strategy: DirectPropagationStrategy,
    subgrid_strategy: SubgridPropagationStrategy,
    size_threshold: usize, // Grid size threshold for switching strategies
}

impl AdaptivePropagationStrategy {
    /// Create a new adaptive propagation strategy
    pub fn new(max_iterations: u32, subgrid_size: u32, size_threshold: usize) -> Self {
        Self {
            name: "Adaptive Propagation".to_string(),
            direct_strategy: DirectPropagationStrategy::new(max_iterations),
            subgrid_strategy: SubgridPropagationStrategy::new(max_iterations, subgrid_size),
            size_threshold,
        }
    }

    /// Create a new strategy with default settings
    pub fn default() -> Self {
        Self::new(1000, 16, 4096) // Use subgrid for grids larger than 64x64
    }

    /// Determine which strategy to use based on grid size
    fn select_strategy(&self, grid: &PossibilityGrid) -> &dyn PropagationStrategy {
        let total_cells = grid.width * grid.height * grid.depth;
        if total_cells > self.size_threshold {
            &self.subgrid_strategy as &dyn PropagationStrategy
        } else {
            &self.direct_strategy as &dyn PropagationStrategy
        }
    }
}

impl PropagationStrategy for AdaptivePropagationStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn prepare(&self, synchronizer: &GpuSynchronizer) -> Result<(), PropagationError> {
        // Prepare both strategies since we don't know which will be used
        self.direct_strategy.prepare(synchronizer)?;
        self.subgrid_strategy.prepare(synchronizer)?;
        Ok(())
    }

    fn propagate(
        &self,
        grid: &mut PossibilityGrid,
        updated_cells: &[GridCoord],
        buffers: &Arc<GpuBuffers>,
        synchronizer: &GpuSynchronizer,
    ) -> Result<(), PropagationError> {
        // Delegate to the appropriate strategy based on grid size
        let strategy = self.select_strategy(grid);
        strategy.propagate(grid, updated_cells, buffers, synchronizer)
    }

    fn cleanup(&self, synchronizer: &GpuSynchronizer) -> Result<(), PropagationError> {
        // Clean up both strategies
        self.direct_strategy.cleanup(synchronizer)?;
        self.subgrid_strategy.cleanup(synchronizer)?;
        Ok(())
    }
}

/// Factory for creating propagation strategy instances
pub struct PropagationStrategyFactory;

impl PropagationStrategyFactory {
    /// Create a direct propagation strategy with custom settings
    pub fn create_direct(max_iterations: u32) -> Box<dyn PropagationStrategy> {
        Box::new(DirectPropagationStrategy::new(max_iterations))
    }

    /// Create a subgrid propagation strategy with custom settings
    pub fn create_subgrid(max_iterations: u32, subgrid_size: u32) -> Box<dyn PropagationStrategy> {
        Box::new(SubgridPropagationStrategy::new(
            max_iterations,
            subgrid_size,
        ))
    }

    /// Create an adaptive propagation strategy with custom settings
    pub fn create_adaptive(
        max_iterations: u32,
        subgrid_size: u32,
        size_threshold: usize,
    ) -> Box<dyn PropagationStrategy> {
        Box::new(AdaptivePropagationStrategy::new(
            max_iterations,
            subgrid_size,
            size_threshold,
        ))
    }

    /// Create a strategy based on the grid size
    pub fn create_for_grid(grid: &PossibilityGrid) -> Box<dyn PropagationStrategy> {
        let total_cells = grid.width * grid.height * grid.depth;
        if total_cells > 4096 {
            Self::create_subgrid(1000, 16)
        } else {
            Self::create_direct(1000)
        }
    }
}
