use crate::utils::error_recovery::GpuError as OldGpuError;
use crate::{
    buffers::{GpuBuffers, GpuEntropyShaderParams},
    entropy::entropy_strategy::{
        EntropyStrategy as ImportedEntropyStrategy, EntropyStrategyFactory,
    },
    gpu::sync::GpuSynchronizer,
    shader::pipeline::ComputePipelines,
    utils::error::gpu_error::{GpuError as NewGpuError, GpuErrorContext},
};
use log::{debug, error, warn};
use pollster;
use std::{fmt::Debug, sync::Arc};
use wfc_core::{
    entropy::{EntropyCalculator, EntropyError as CoreEntropyError, EntropyHeuristicType},
    grid::{EntropyGrid, PossibilityGrid},
};

// Adapter to convert EntropyStrategy to GpuEntropyStrategy
struct EntropyStrategyAdapter(Box<dyn ImportedEntropyStrategy>);

impl Debug for EntropyStrategyAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EntropyStrategyAdapter")
    }
}

impl GpuEntropyStrategy for EntropyStrategyAdapter {
    fn heuristic_type(&self) -> EntropyHeuristicType {
        self.0.heuristic_type()
    }

    fn configure_shader_params(&self, params: &mut GpuEntropyShaderParams) {
        self.0.configure_shader_params(params)
    }

    fn prepare(&self, synchronizer: &GpuSynchronizer) -> Result<(), CoreEntropyError> {
        self.0.prepare(synchronizer)
    }

    fn calculate_entropy(
        &self,
        _buffers: &GpuBuffers,
        _pipelines: &ComputePipelines,
        _queue: &wgpu::Queue,
        _grid_dims: (usize, usize, usize),
    ) -> Result<(), CoreEntropyError> {
        // Default implementation as this isn't used in the imported strategy
        Ok(())
    }

    fn upload_data(&self, synchronizer: &GpuSynchronizer) -> Result<(), CoreEntropyError> {
        self.0.upload_data(synchronizer)
    }

    fn post_process(&self, synchronizer: &GpuSynchronizer) -> Result<(), CoreEntropyError> {
        self.0.post_process(synchronizer)
    }
}

/// GPU-accelerated entropy calculator for use in WFC algorithm.
///
/// This component computes the entropy of each cell in the grid and identifies
/// the cell with the minimum positive entropy, which is a key step in the WFC algorithm.
#[derive(Debug)]
pub struct GpuEntropyCalculator {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipelines: Arc<ComputePipelines>,
    buffers: Arc<GpuBuffers>,
    synchronizer: Arc<GpuSynchronizer>,
    grid_dims: (usize, usize, usize),
    strategy: Box<dyn GpuEntropyStrategy>,
}

// Manual clone implementation since Box<dyn GpuEntropyStrategy> is not Clone
impl Clone for GpuEntropyCalculator {
    fn clone(&self) -> Self {
        // Create a new strategy of the same type
        let base_strategy = EntropyStrategyFactory::create_strategy(
            self.strategy.heuristic_type(),
            self.buffers.num_tiles,
            self.buffers.grid_buffers.u32s_per_cell,
        );

        // Wrap with adapter
        let strategy = Box::new(EntropyStrategyAdapter(base_strategy));

        Self {
            device: self.device.clone(),
            queue: self.queue.clone(),
            pipelines: self.pipelines.clone(),
            buffers: self.buffers.clone(),
            synchronizer: self.synchronizer.clone(),
            grid_dims: self.grid_dims,
            strategy,
        }
    }
}

impl GpuEntropyCalculator {
    /// Creates a new GPU entropy calculator with the default Shannon entropy heuristic.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        pipelines: Arc<ComputePipelines>,
        buffers: Arc<GpuBuffers>,
        grid_dims: (usize, usize, usize),
    ) -> Self {
        let synchronizer = Arc::new(GpuSynchronizer::new(
            device.clone(),
            queue.clone(),
            buffers.clone(),
        ));

        // Use factory to create the default Shannon strategy
        let base_strategy = EntropyStrategyFactory::create_strategy(
            EntropyHeuristicType::default(),
            buffers.num_tiles,
            buffers.grid_buffers.u32s_per_cell,
        );

        // Wrap with adapter
        let strategy = Box::new(EntropyStrategyAdapter(base_strategy));

        Self {
            device,
            queue,
            pipelines,
            buffers,
            synchronizer,
            grid_dims,
            strategy,
        }
    }

    /// Creates a new GPU entropy calculator with a specific entropy heuristic.
    pub fn with_heuristic(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        pipelines: Arc<ComputePipelines>,
        buffers: Arc<GpuBuffers>,
        grid_dims: (usize, usize, usize),
        heuristic_type: EntropyHeuristicType,
    ) -> Self {
        let synchronizer = Arc::new(GpuSynchronizer::new(
            device.clone(),
            queue.clone(),
            buffers.clone(),
        ));

        // Use factory to create the specified strategy
        let base_strategy = EntropyStrategyFactory::create_strategy(
            heuristic_type,
            buffers.num_tiles,
            buffers.grid_buffers.u32s_per_cell,
        );

        // Wrap with adapter
        let strategy = Box::new(EntropyStrategyAdapter(base_strategy));

        Self {
            device,
            queue,
            pipelines,
            buffers,
            synchronizer,
            grid_dims,
            strategy,
        }
    }

    /// Sets a new entropy strategy
    pub fn with_strategy(&mut self, strategy: Box<dyn GpuEntropyStrategy>) -> &mut Self {
        self.strategy = strategy;
        self
    }

    /// Sets a new entropy strategy (same as with_strategy but with a more consistent naming)
    pub fn set_strategy_boxed(&mut self, strategy: Box<dyn GpuEntropyStrategy>) -> &mut Self {
        self.strategy = strategy;
        self
    }

    /// Asynchronous version of calculate_entropy that allows proper async/await patterns
    pub async fn calculate_entropy_async(
        &self,
        grid: &PossibilityGrid,
    ) -> Result<EntropyGrid, CoreEntropyError> {
        debug!("Entering calculate_entropy_async");
        let (width, height, depth) = (grid.width, grid.height, grid.depth);
        let num_cells = width * height * depth;

        if num_cells == 0 {
            return Ok(EntropyGrid::new(width, height, depth));
        }

        // Reset min entropy buffer
        self.synchronizer.reset_min_entropy_buffer()?;

        // Call strategy's prepare method
        self.strategy.prepare(&self.synchronizer)?;

        // Upload strategy data if needed (now available in the trait)
        self.strategy.upload_data(&self.synchronizer)?;

        // Convert grid possibilities to u32 arrays and upload them
        let u32s_per_cell = self.buffers.grid_buffers.u32s_per_cell;
        let mut packed_data = Vec::with_capacity(num_cells * u32s_per_cell);
        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    if let Some(cell) = grid.get(x, y, z) {
                        let mut cell_data = vec![0u32; u32s_per_cell];
                        for (i, bit) in cell.iter().enumerate() {
                            if *bit {
                                let u32_idx = i / 32;
                                let bit_idx = i % 32;
                                if u32_idx < cell_data.len() {
                                    cell_data[u32_idx] |= 1 << bit_idx;
                                }
                            }
                        }
                        packed_data.extend_from_slice(&cell_data);
                    } else {
                        // Handle case where grid.get might return None if dimensions are mismatched
                        packed_data.extend(vec![0u32; u32s_per_cell]);
                    }
                }
            }
        }

        // Upload the data to the GPU buffer
        self.queue.write_buffer(
            &self.buffers.grid_buffers.grid_possibilities_buf,
            0,
            bytemuck::cast_slice(&packed_data),
        );

        // --- Create and configure entropy parameters ---
        let mut entropy_shader_params = GpuEntropyShaderParams {
            grid_dims: [width as u32, height as u32, depth as u32],
            heuristic_type: 0, // Will be set by strategy
            num_tiles: 0,      // Will be set by strategy
            u32s_per_cell: 0,  // Will be set by strategy
            _padding1: 0,
            _padding2: 0,
        };

        // Let the strategy configure its specific parameters
        self.strategy
            .configure_shader_params(&mut entropy_shader_params);

        // --- Write entropy parameters to buffer ---
        self.synchronizer
            .upload_entropy_params(&entropy_shader_params)?;

        // --- Dispatch Compute Shader ---
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Entropy Compute Encoder"),
            });

        // Create main bind group (group 0)
        let grid_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Grid Possibilities Bind Group"),
            layout: &self.pipelines.entropy_bind_group_layout_0,
            entries: &[
                wgpu::BindGroupEntry {
                    // Grid Possibilities
                    binding: 0,
                    resource: self
                        .buffers
                        .grid_buffers
                        .grid_possibilities_buf
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    // Entropy Output
                    binding: 1,
                    resource: self.buffers.entropy_buffers.entropy_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    // Params (General)
                    binding: 2,
                    resource: self.buffers.params_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    // Min Entropy Output
                    binding: 3,
                    resource: self
                        .buffers
                        .entropy_buffers
                        .min_entropy_info_buf
                        .as_entire_binding(),
                },
            ],
        });

        // Create entropy parameters bind group (group 1)
        let entropy_params_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Entropy Parameters Bind Group"),
            layout: &self.pipelines.entropy_bind_group_layout_1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.buffers.entropy_params_buffer.as_entire_binding(),
            }],
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Entropy Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipelines.entropy_pipeline);
            compute_pass.set_bind_group(0, &grid_bind_group, &[]); // Main data buffers
            compute_pass.set_bind_group(1, &entropy_params_bind_group, &[]); // Entropy specific params

            // Calculate workgroups
            let workgroup_size = self.pipelines.entropy_workgroup_size;
            // Cast width and height to u32 for division
            let workgroup_x = (width as u32).div_ceil(workgroup_size);
            let workgroup_y = (height as u32).div_ceil(workgroup_size);
            let workgroup_z = depth as u32; // Depth likely doesn't need division by workgroup size

            compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, workgroup_z);
        } // End compute pass scope

        // Submit to Queue
        self.queue.submit(std::iter::once(encoder.finish()));

        // --- Download Results ---
        // Create encoder for copy
        let mut copy_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Copy Entropy Results Encoder"),
                });
        copy_encoder.copy_buffer_to_buffer(
            &self.buffers.entropy_buffers.entropy_buf,
            0,
            &self.buffers.entropy_buffers.staging_entropy_buf,
            0,
            self.buffers.entropy_buffers.entropy_buf.size(),
        );
        self.queue.submit(Some(copy_encoder.finish()));

        // Use the centralized download function, cloning the necessary Arcs
        let entropy_data = crate::buffers::download_buffer_data::<f32>(
            Some(self.device.clone()),
            Some(self.queue.clone()),
            &self.buffers.entropy_buffers.entropy_buf,
            &self.buffers.entropy_buffers.staging_entropy_buf,
            self.buffers.entropy_buffers.entropy_buf.size(),
            Some("Entropy Data Download".to_string()),
        )
        .await?;

        // Call post-process if needed (now available in the trait)
        self.strategy.post_process(&self.synchronizer)?;

        debug!("Exiting calculate_entropy_async");
        let mut entropy_grid =
            EntropyGrid::new(self.grid_dims.0, self.grid_dims.1, self.grid_dims.2);
        if entropy_grid.data.len() == entropy_data.len() {
            entropy_grid.data = entropy_data;
            Ok(entropy_grid)
        } else {
            error!(
                "Entropy data size mismatch: expected {}, got {}",
                entropy_grid.data.len(),
                entropy_data.len()
            );
            Err(CoreEntropyError::Other(
                "Downloaded entropy data size mismatch".into(),
            ))
        }
    }

    /// Asynchronously selects the cell with the lowest positive entropy.
    /// This reads the result buffer populated by the entropy compute shader.
    pub async fn select_lowest_entropy_cell_async(
        &self,
        _entropy_grid: &EntropyGrid, // Grid itself is not needed as data is on GPU
    ) -> Option<(usize, usize, usize)> {
        debug!("Entering select_lowest_entropy_cell_async");

        // --- Download Min Entropy Info ---
        // Create encoder for copy
        let mut copy_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Copy Min Entropy Info Encoder"),
                });
        copy_encoder.copy_buffer_to_buffer(
            &self.buffers.entropy_buffers.min_entropy_info_buf,
            0,
            &self.buffers.entropy_buffers.staging_min_entropy_info_buf,
            0,
            self.buffers.entropy_buffers.min_entropy_info_buf.size(),
        );
        self.queue.submit(Some(copy_encoder.finish()));

        // Use the centralized download function
        let min_info_data = crate::buffers::download_buffer_data::<u32>(
            Some(self.device.clone()),
            Some(self.queue.clone()),
            &self.buffers.entropy_buffers.min_entropy_info_buf,
            &self.buffers.entropy_buffers.staging_min_entropy_info_buf,
            self.buffers.entropy_buffers.min_entropy_info_buf.size(),
            Some("Min Entropy Info Download".to_string()),
        )
        .await;

        match min_info_data {
            Ok(data) => {
                if data.len() < 2 {
                    warn!(
                        "Downloaded min_entropy_info data has insufficient length ({})",
                        data.len()
                    );
                    return None;
                }
                let _min_entropy_bits = data[0];
                let min_index = data[1];

                // Check if a valid minimum was found (index != u32::MAX)
                if min_index != u32::MAX {
                    let (width, height, _depth) = self.grid_dims;
                    let idx = min_index as usize;
                    let z = idx / (width * height);
                    let y = (idx % (width * height)) / width;
                    let x = idx % width;
                    debug!("Selected lowest entropy cell: ({}, {}, {})", x, y, z);
                    Some((x, y, z))
                } else {
                    debug!("No cell with positive entropy found (or grid fully collapsed/contradiction).");
                    None // Grid might be fully collapsed or in a contradiction state
                }
            }
            Err(e) => {
                error!("Failed to download min entropy info: {}", e);
                None
            }
        }
    }

    /// Asynchronously selects the cell with the lowest positive entropy and returns
    /// both the coordinates and the entropy value in a single call.
    pub async fn select_lowest_entropy_cell_with_value_async(
        &self,
        _entropy_grid: &EntropyGrid, // Grid itself is not needed as data is on GPU
    ) -> Option<((usize, usize, usize), f32)> {
        debug!("Entering select_lowest_entropy_cell_with_value_async");

        // --- Download Min Entropy Info ---
        // Create encoder for copy
        let mut copy_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Copy Min Entropy Info Encoder"),
                });
        copy_encoder.copy_buffer_to_buffer(
            &self.buffers.entropy_buffers.min_entropy_info_buf,
            0,
            &self.buffers.entropy_buffers.staging_min_entropy_info_buf,
            0,
            self.buffers.entropy_buffers.min_entropy_info_buf.size(),
        );
        self.queue.submit(Some(copy_encoder.finish()));

        // Use the centralized download function
        let min_info_data = crate::buffers::download_buffer_data::<u32>(
            Some(self.device.clone()),
            Some(self.queue.clone()),
            &self.buffers.entropy_buffers.min_entropy_info_buf,
            &self.buffers.entropy_buffers.staging_min_entropy_info_buf,
            self.buffers.entropy_buffers.min_entropy_info_buf.size(),
            Some("Min Entropy Info Download".to_string()),
        )
        .await;

        match min_info_data {
            Ok(data) => {
                if data.len() < 2 {
                    warn!(
                        "Downloaded min_entropy_info data has insufficient length ({})",
                        data.len()
                    );
                    return None;
                }

                let min_entropy_bits = data[0];
                let min_index = data[1];

                // Check if a valid minimum was found (index != u32::MAX)
                if min_index != u32::MAX {
                    let (width, height, _depth) = self.grid_dims;
                    let idx = min_index as usize;
                    let z = idx / (width * height);
                    let y = (idx % (width * height)) / width;
                    let x = idx % width;

                    // Convert the bits back to float
                    let min_entropy = f32::from_bits(min_entropy_bits);

                    debug!(
                        "Selected lowest entropy cell: ({}, {}, {}) with entropy {}",
                        x, y, z, min_entropy
                    );
                    Some(((x, y, z), min_entropy))
                } else {
                    debug!("No cell with positive entropy found (or grid fully collapsed/contradiction).");
                    None // Grid might be fully collapsed or in a contradiction state
                }
            }
            Err(e) => {
                error!("Failed to download min entropy info: {}", e);
                None
            }
        }
    }

    #[allow(dead_code)]
    fn upload_data_from_strategy(&self) -> Result<(), CoreEntropyError> {
        // Implementation that would call a suitable method on self.strategy
        Ok(())
    }

    #[allow(dead_code)]
    fn post_process_from_strategy(&self) -> Result<(), CoreEntropyError> {
        // Implementation that would call a suitable method on self.strategy
        Ok(())
    }
}

impl EntropyCalculator for GpuEntropyCalculator {
    fn calculate_entropy(&self, grid: &PossibilityGrid) -> Result<EntropyGrid, CoreEntropyError> {
        // Use pollster instead of futures::executor::block_on
        pollster::block_on(self.calculate_entropy_async(grid))
    }

    fn select_lowest_entropy_cell(
        &self,
        entropy_grid: &EntropyGrid,
    ) -> Option<(usize, usize, usize)> {
        // Use pollster instead of futures::executor::block_on
        pollster::block_on(self.select_lowest_entropy_cell_async(entropy_grid))
    }

    fn set_entropy_heuristic(&mut self, heuristic_type: EntropyHeuristicType) -> bool {
        // Create a new strategy using the factory
        let base_strategy = EntropyStrategyFactory::create_strategy(
            heuristic_type,
            self.buffers.num_tiles,
            self.buffers.grid_buffers.u32s_per_cell,
        );

        // Wrap with adapter
        let strategy = Box::new(EntropyStrategyAdapter(base_strategy));

        // Replace the current strategy
        self.strategy = strategy;
        true // GPU implementation supports all heuristic types
    }

    fn get_entropy_heuristic(&self) -> EntropyHeuristicType {
        // Get the heuristic type from the strategy
        self.strategy.heuristic_type()
    }
}

impl From<OldGpuError> for CoreEntropyError {
    fn from(error: OldGpuError) -> Self {
        match error {
            OldGpuError::MemoryAllocation(msg) => {
                CoreEntropyError::Other(format!("GPU memory allocation failed: {}", msg))
            }
            OldGpuError::ComputationTimeout {
                grid_size,
                duration,
            } => CoreEntropyError::Other(format!(
                "GPU computation timed out after {:?} for grid size {:?}",
                duration, grid_size
            )),
            OldGpuError::KernelExecution(msg) => {
                CoreEntropyError::Other(format!("GPU kernel execution error: {}", msg))
            }
            OldGpuError::QueueSubmission(msg) => {
                CoreEntropyError::Other(format!("GPU queue submission error: {}", msg))
            }
            OldGpuError::DeviceLost(msg) => {
                CoreEntropyError::Other(format!("GPU device lost: {}", msg))
            }
            OldGpuError::InvalidState(msg) => {
                CoreEntropyError::Other(format!("GPU invalid state: {}", msg))
            }
            OldGpuError::BarrierSynchronization(msg) => {
                CoreEntropyError::Other(format!("GPU barrier sync error: {}", msg))
            }
            OldGpuError::BufferCopy(msg) => {
                CoreEntropyError::Other(format!("GPU buffer copy error: {}", msg))
            }
            OldGpuError::BufferMapping(msg) => {
                CoreEntropyError::Other(format!("GPU buffer mapping error: {}", msg))
            }
            OldGpuError::ContradictionDetected { context } => {
                CoreEntropyError::Other(format!("Contradiction detected: {}", context))
            }
            OldGpuError::Other(msg) => CoreEntropyError::Other(format!("GPU error: {}", msg)),
        }
    }
}

fn parse_coords_from_context(context: &GpuErrorContext) -> Option<(usize, usize, usize)> {
    // Check if we have grid coordinates directly
    if let Some(coord) = &context.grid_coord {
        return Some((coord.x, coord.y, coord.z));
    }

    // Fallback to parsing from details if available
    if let Some(details) = &context.details {
        let parts: Vec<&str> = details.split(&['(', ',', ')'][..]).collect();
        if parts.len() >= 4 {
            if let (Ok(x), Ok(y), Ok(z)) = (
                parts[1].trim().parse::<usize>(),
                parts[2].trim().parse::<usize>(),
                parts[3].trim().parse::<usize>(),
            ) {
                return Some((x, y, z));
            }
        }
    }
    None
}

impl From<NewGpuError> for CoreEntropyError {
    fn from(error: NewGpuError) -> Self {
        match error {
            NewGpuError::BufferMapFailed { msg, .. } => {
                CoreEntropyError::Other(format!("GPU buffer mapping failed: {}", msg))
            }
            NewGpuError::ShaderError { msg, .. } => {
                CoreEntropyError::Other(format!("GPU shader error: {}", msg))
            }
            NewGpuError::ResourceCreationFailed { msg, .. } => {
                CoreEntropyError::Other(format!("GPU resource creation failed: {}", msg))
            }
            NewGpuError::CommandExecutionError { msg, .. } => {
                CoreEntropyError::Other(format!("GPU command execution error: {}", msg))
            }
            NewGpuError::DeviceLost { msg, .. } => CoreEntropyError::Other(format!(
                "GPU device lost during entropy calculation: {}",
                msg
            )),
            NewGpuError::Timeout { msg, .. } => {
                CoreEntropyError::Other(format!("GPU entropy calculation timed out: {}", msg))
            }
            NewGpuError::ContradictionDetected { context } => {
                // Try to parse coordinates from context
                if let Some((x, y, z)) = parse_coords_from_context(&context) {
                    CoreEntropyError::GridAccessError(x, y, z)
                } else {
                    CoreEntropyError::Other(format!("Contradiction detected: {:?}", context))
                }
            }
            _ => CoreEntropyError::Other(format!("GPU error: {:?}", error)),
        }
    }
}

pub trait GpuEntropyCalculatorExt {
    fn select_lowest_entropy_cell_sync(
        &self,
        entropy_grid: &EntropyGrid,
    ) -> Option<(usize, usize, usize)>;
}

impl GpuEntropyCalculatorExt for GpuEntropyCalculator {
    fn select_lowest_entropy_cell_sync(
        &self,
        entropy_grid: &EntropyGrid,
    ) -> Option<(usize, usize, usize)> {
        self.select_lowest_entropy_cell(entropy_grid)
    }
}

/// Trait for GPU-specific entropy calculation strategies in the WFC algorithm.
pub trait GpuEntropyStrategy: Send + Sync + std::fmt::Debug {
    /// Get the heuristic type used by this strategy
    fn heuristic_type(&self) -> EntropyHeuristicType;

    /// Configure shader parameters for this strategy
    fn configure_shader_params(&self, params: &mut GpuEntropyShaderParams);

    /// Prepare the strategy for execution
    fn prepare(&self, synchronizer: &GpuSynchronizer) -> Result<(), CoreEntropyError>;

    /// Execute the entropy calculation on the GPU
    fn calculate_entropy(
        &self,
        buffers: &GpuBuffers,
        pipelines: &ComputePipelines,
        queue: &wgpu::Queue,
        grid_dims: (usize, usize, usize),
    ) -> Result<(), CoreEntropyError>;

    /// Upload strategy-specific data to GPU if needed
    fn upload_data(&self, _synchronizer: &GpuSynchronizer) -> Result<(), CoreEntropyError> {
        // Default implementation does nothing
        Ok(())
    }

    /// Any post-processing needed after shader execution
    fn post_process(&self, _synchronizer: &GpuSynchronizer) -> Result<(), CoreEntropyError> {
        // Default implementation does nothing
        Ok(())
    }
}
