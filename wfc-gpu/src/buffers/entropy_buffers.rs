// wfc-gpu/src/buffers/entropy_buffers.rs

//! Module for GPU buffers specifically related to entropy calculations.

use crate::buffers::{DynamicBufferConfig, GpuBuffers}; // Need GpuBuffers for create_buffer and resize_buffer
use crate::GpuError;
use std::sync::Arc;
use wgpu::BufferUsages;

/// Holds GPU buffers used during entropy calculation and minimum entropy finding.
#[derive(Debug, Clone)]
pub struct EntropyBuffers {
    /// Stores entropy value for each cell (f32).
    pub entropy_buf: Arc<wgpu::Buffer>,
    /// Staging buffer for reading entropy values back to the CPU.
    pub staging_entropy_buf: Arc<wgpu::Buffer>,
    /// Stores minimum entropy found and its index ([f32_bits, u32_index]).
    pub min_entropy_info_buf: Arc<wgpu::Buffer>,
    /// Staging buffer for reading minimum entropy info back to the CPU.
    pub staging_min_entropy_info_buf: Arc<wgpu::Buffer>,
    // TODO: Consider moving entropy_params_buffer here too?
}

impl EntropyBuffers {
    /// Creates new entropy-related GPU buffers.
    pub fn new(
        device: &wgpu::Device,
        num_cells: usize,
        _config: &DynamicBufferConfig, // Config not used yet, but likely needed for resizing
    ) -> Result<Self, GpuError> {
        let entropy_buffer_size = (num_cells * std::mem::size_of::<f32>()) as u64;
        let min_entropy_info_buffer_size = (2 * std::mem::size_of::<u32>()) as u64; // [f32_bits, u32_index]

        let entropy_buf = GpuBuffers::create_buffer(
            device,
            entropy_buffer_size,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST, // Added COPY_DST
            Some("Entropy Buffer"),
        );
        let staging_entropy_buf = GpuBuffers::create_buffer(
            device,
            entropy_buffer_size,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            Some("Staging Entropy Buffer"),
        );
        let min_entropy_info_buf = GpuBuffers::create_buffer(
            device,
            min_entropy_info_buffer_size,
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            Some("Min Entropy Info Buffer"),
        );
        let staging_min_entropy_info_buf = GpuBuffers::create_buffer(
            device,
            min_entropy_info_buffer_size,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            Some("Staging Min Entropy Info"),
        );

        Ok(Self {
            entropy_buf,
            staging_entropy_buf,
            min_entropy_info_buf,
            staging_min_entropy_info_buf,
        })
    }

    /// Ensures the entropy buffers are large enough for the given grid dimensions.
    pub fn ensure_buffers(
        &mut self,
        device: &wgpu::Device,
        num_cells: usize,
        config: &DynamicBufferConfig,
    ) -> Result<(), String> {
        let required_entropy_size = (num_cells * std::mem::size_of::<f32>()) as u64;

        // Ensure entropy_buf
        if !GpuBuffers::is_buffer_sufficient(&self.entropy_buf, required_entropy_size) {
            let new_buffer = GpuBuffers::resize_buffer(
                device,
                &self.entropy_buf,
                required_entropy_size,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST, // Added COPY_DST
                Some("Entropy Buffer"),
                config,
            );
            self.entropy_buf = new_buffer;
        }

        // Ensure staging_entropy_buf
        if !GpuBuffers::is_buffer_sufficient(&self.staging_entropy_buf, required_entropy_size) {
            let new_staging_buffer = GpuBuffers::resize_buffer(
                device,
                &self.staging_entropy_buf,
                required_entropy_size,
                BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                Some("Staging Entropy Buffer"),
                config,
            );
            self.staging_entropy_buf = new_staging_buffer;
        }

        // Note: min_entropy_info_buf is fixed size (2 * u32), no resizing needed unless logic changes.

        Ok(())
    }
}

// TODO: Add tests specific to EntropyBuffers if needed
