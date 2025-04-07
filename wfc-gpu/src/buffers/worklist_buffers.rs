// Removed unused imports
// use bytemuck::{Pod, Zeroable};
// use log::{error, info, warn};

use std::sync::Arc;
use wgpu::util::DeviceExt;

use super::{DynamicBufferConfig, GpuBuffers};
use crate::GpuError; // Corrected path // GpuBuffers import might be removed/adjusted later

/// Stores the GPU buffers specifically related to the propagation worklist.
#[derive(Debug, Clone)]
pub struct WorklistBuffers {
    /// First buffer for storing the worklist of cells to update in propagation (double-buffered design).
    pub worklist_buf_a: Arc<wgpu::Buffer>,
    /// Second buffer for storing the worklist of cells to update in propagation (double-buffered design).
    pub worklist_buf_b: Arc<wgpu::Buffer>,
    /// Buffer for tracking the number of cells in the worklist.
    pub worklist_count_buf: Arc<wgpu::Buffer>,
    /// Staging buffer used during worklist count upload/download.
    pub staging_worklist_count_buf: Arc<wgpu::Buffer>,
    /// Current size of the worklist (number of elements).
    pub current_worklist_size: usize,
    /// Current active worklist buffer index (0 for worklist_buf_a, 1 for worklist_buf_b).
    pub current_worklist_idx: usize,
    // /// Tracks the last requested download status for the worklist size. (Removed, handled by DownloadRequest)
    // pub download_worklist_size: bool,
    // /// Stores the last downloaded worklist count. (Removed, handled by GpuDownloadResults)
    // pub worklist_count: Option<u32>,
}

impl WorklistBuffers {
    /// Creates the worklist-specific buffers.
    pub(crate) fn new(
        device: &wgpu::Device,
        num_cells: usize,
        config: &DynamicBufferConfig, // Added config for initial sizing?
    ) -> Result<Self, GpuError> {
        let worklist_element_size = std::mem::size_of::<u32>() as u64;
        let initial_worklist_capacity =
            (num_cells as u64).max(config.min_buffer_size / worklist_element_size);
        let worklist_buffer_size = initial_worklist_capacity * worklist_element_size;

        // Buffer A
        let worklist_buf_a = GpuBuffers::create_buffer(
            device,
            worklist_buffer_size,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            Some("Worklist Buffer A"),
        );

        // Buffer B
        let worklist_buf_b = GpuBuffers::create_buffer(
            device,
            worklist_buffer_size,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            Some("Worklist Buffer B"),
        );

        // Count Buffer
        let worklist_count_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Worklist Count Buffer"),
            contents: bytemuck::bytes_of(&0u32), // Initialize count to 0
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::INDIRECT,
        });

        // Staging Count Buffer
        let staging_worklist_count_buf = GpuBuffers::create_buffer(
            device,
            std::mem::size_of::<u32>() as u64,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            Some("Staging Worklist Count Buffer"),
        );

        Ok(Self {
            worklist_buf_a,
            worklist_buf_b,
            worklist_count_buf: Arc::new(worklist_count_buf), // Wrap in Arc
            staging_worklist_count_buf,
            current_worklist_size: 0,
            current_worklist_idx: 0,
        })
    }

    /// Ensure the worklist buffers are sufficient for the given dimensions
    /// This method needs access to `GpuBuffers::is_buffer_sufficient` and `GpuBuffers::resize_buffer`.
    /// We'll temporarily keep it here and adjust dependencies later.
    pub fn ensure_worklist_buffers(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        depth: u32,
        config: &DynamicBufferConfig,
    ) -> Result<(), String> {
        let num_cells = (width * height * depth) as usize;
        // Worklist stores cell indices (u32)
        let required_size = (num_cells * std::mem::size_of::<u32>()) as u64;

        if !GpuBuffers::is_buffer_sufficient(&self.worklist_buf_a, required_size) {
            let new_buffer_a = GpuBuffers::resize_buffer(
                device,
                &self.worklist_buf_a,
                required_size,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                Some("Worklist Buffer A"),
                config,
            );
            self.worklist_buf_a = new_buffer_a;
        }

        if !GpuBuffers::is_buffer_sufficient(&self.worklist_buf_b, required_size) {
            let new_buffer_b = GpuBuffers::resize_buffer(
                device,
                &self.worklist_buf_b,
                required_size,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                Some("Worklist Buffer B"),
                config,
            );
            self.worklist_buf_b = new_buffer_b;
        }

        // Ensure the worklist count buffer (single u32)
        let count_buf_required_size = std::mem::size_of::<u32>() as u64;
        if !GpuBuffers::is_buffer_sufficient(&self.worklist_count_buf, count_buf_required_size) {
            let new_count_buf = GpuBuffers::resize_buffer(
                device,
                &self.worklist_count_buf,
                count_buf_required_size, // Size is always 4 bytes for a single u32
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_SRC, // Added COPY_SRC for download
                Some("Worklist Count Buffer"),
                config,
            );
            self.worklist_count_buf = new_count_buf;
        }

        // Ensure the staging worklist count buffer (single u32)
        let staging_count_buf_required_size = std::mem::size_of::<u32>() as u64;
        if !GpuBuffers::is_buffer_sufficient(
            &self.staging_worklist_count_buf,
            staging_count_buf_required_size,
        ) {
            // Staging buffers have different usage flags
            let new_staging_count_buf = GpuBuffers::resize_buffer(
                device,
                &self.staging_worklist_count_buf,
                staging_count_buf_required_size,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                Some("Staging Worklist Count Buffer"),
                config,
            );
            self.staging_worklist_count_buf = new_staging_count_buf;
        }

        Ok(())
    }

    // TODO: Move relevant parts of upload/download logic here? Or handle via GpuSynchronizer?
}
