use bitvec::prelude::*;
use std::sync::Arc;
use wgpu::util::DeviceExt;

use super::{DynamicBufferConfig, GpuBuffers}; // Adjust as needed
use crate::GpuError;
use wfc_core::grid::PossibilityGrid;

/// Stores the GPU buffers specifically related to the main WFC grid state (possibilities).
#[derive(Debug, Clone)]
pub struct GridBuffers {
    /// Buffer holding the current state of all cell possibilities - the primary WFC grid state.
    pub grid_possibilities_buf: Arc<wgpu::Buffer>,
    /// Staging buffer used during grid possibilities upload/download.
    pub staging_grid_possibilities_buf: Arc<wgpu::Buffer>,
    /// Number of u32 words used to represent the possibilities for a single cell.
    /// Needed for calculating buffer sizes.
    pub u32s_per_cell: usize,
    // Add other potentially relevant fields like grid_dims? Or pass them?
}

impl GridBuffers {
    /// Creates the grid-specific buffers.
    pub(crate) fn new(
        device: &wgpu::Device,
        initial_grid: &PossibilityGrid,
        _config: &DynamicBufferConfig, // Prefix unused variable
    ) -> Result<Self, GpuError> {
        let width = initial_grid.width;
        let height = initial_grid.height;
        let depth = initial_grid.depth;
        let _num_cells = width * height * depth; // Prefix with _ as it might be unused depending on config usage
        let num_tiles = initial_grid.num_tiles();
        let u32s_per_cell = (num_tiles + 31) / 32;

        let packed_possibilities = Self::pack_initial_grid(initial_grid)?;
        let grid_buffer_size = (packed_possibilities.len() * std::mem::size_of::<u32>()) as u64;

        // Main grid buffer
        let grid_possibilities_buf = Arc::new(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("WFC Grid Possibilities Buffer"),
                contents: bytemuck::cast_slice(&packed_possibilities),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            },
        ));

        // Staging grid buffer
        let staging_grid_possibilities_buf = GpuBuffers::create_buffer(
            device,
            grid_buffer_size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            Some("Staging Grid Possibilities Buffer"),
        );

        Ok(Self {
            grid_possibilities_buf,
            staging_grid_possibilities_buf,
            u32s_per_cell,
        })
    }

    /// Packs the initial PossibilityGrid into a Vec<u32> for GPU upload.
    fn pack_initial_grid(initial_grid: &PossibilityGrid) -> Result<Vec<u32>, GpuError> {
        let width = initial_grid.width;
        let height = initial_grid.height;
        let depth = initial_grid.depth;
        let _num_cells = width * height * depth; // Prefix with _ as it might be unused depending on config usage
        let num_tiles = initial_grid.num_tiles();
        let u32s_per_cell = (num_tiles + 31) / 32;
        let mut packed_possibilities = Vec::with_capacity(_num_cells * u32s_per_cell);

        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    if let Some(cell_possibilities) = initial_grid.get(x, y, z) {
                        if cell_possibilities.len() != num_tiles {
                            return Err(GpuError::BufferOperationError {
                                msg: format!(
                                    "Possibility grid cell ({}, {}, {}) has unexpected length: {} (expected {})",
                                    x, y, z, cell_possibilities.len(), num_tiles
                                ),
                                context: crate::utils::error::gpu_error::GpuErrorContext::default()
                            });
                        }
                        // Pack the BitSlice into u32s
                        let iter = cell_possibilities.chunks_exact(32);
                        let remainder = iter.remainder();
                        for chunk in iter {
                            packed_possibilities.push(chunk.load_le::<u32>());
                        }
                        if !remainder.is_empty() {
                            let mut last_u32 = 0u32;
                            for (i, bit) in remainder.iter().by_vals().enumerate() {
                                if bit {
                                    last_u32 |= 1 << i;
                                }
                            }
                            packed_possibilities.push(last_u32);
                        }
                    } else {
                        return Err(GpuError::BufferOperationError {
                            msg: format!(
                                "Failed to get possibility grid cell ({}, {}, {})",
                                x, y, z
                            ),
                            context: crate::utils::error::gpu_error::GpuErrorContext::default(),
                        });
                    }
                }
            }
        }

        if packed_possibilities.len() != _num_cells * u32s_per_cell {
            return Err(GpuError::BufferOperationError {
                msg: format!(
                    "Internal Error: Packed possibilities size mismatch. Expected {}, Got {}.",
                    _num_cells * u32s_per_cell,
                    packed_possibilities.len()
                ),
                context: crate::utils::error::gpu_error::GpuErrorContext::default(),
            });
        }
        Ok(packed_possibilities)
    }

    /// Ensure the grid possibilities buffer is sufficient for the given dimensions
    /// This needs GpuBuffers::is_buffer_sufficient and GpuBuffers::resize_buffer
    pub fn ensure_grid_possibilities_buffer(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        depth: u32,
        num_tiles: u32,
        config: &DynamicBufferConfig,
    ) -> Result<(), String> {
        let _num_cells = (width * height * depth) as usize;
        let u32s_per_cell = ((num_tiles + 31) / 32) as usize;
        let required_size = (u32s_per_cell * std::mem::size_of::<u32>()) as u64;

        // Resize main buffer if needed
        if !GpuBuffers::is_buffer_sufficient(&self.grid_possibilities_buf, required_size) {
            let new_buffer = GpuBuffers::resize_buffer(
                device,
                &self.grid_possibilities_buf,
                required_size,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                Some("Grid Possibilities Buffer"),
                config,
            );
            self.grid_possibilities_buf = new_buffer;
        }

        // Resize staging buffer if needed
        if !GpuBuffers::is_buffer_sufficient(&self.staging_grid_possibilities_buf, required_size) {
            let new_staging_buffer = GpuBuffers::resize_buffer(
                device,
                &self.staging_grid_possibilities_buf,
                required_size,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                Some("Staging Grid Possibilities Buffer"),
                config,
            );
            self.staging_grid_possibilities_buf = new_staging_buffer;
        }

        // Update u32s_per_cell if num_tiles changed
        self.u32s_per_cell = u32s_per_cell;

        Ok(())
    }
}
