// wfc-gpu/src/buffers/mod.rs

//! Module organizing different GPU buffer types used in WFC-GPU.

// Imports
use crate::{error_recovery::RecoverableGpuOp, GpuError};
use bytemuck::{Pod, Zeroable};
use futures::future::{try_join_all, FutureExt};
use log::{debug, error, info, trace, warn};
use std::{
    any::Any,
    collections::HashMap,
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};
use tokio::time::timeout;
use wfc_core::{grid::PossibilityGrid, BoundaryCondition};
use wfc_rules::AdjacencyRules;
use wgpu::{self, util::DeviceExt};

// Declare submodules
pub mod entropy_buffers;
pub mod grid_buffers;
pub mod worklist_buffers;
// pub mod rule_buffers; // Future

// Re-export key structs from submodules
pub use entropy_buffers::EntropyBuffers;
pub use grid_buffers::GridBuffers;
pub use worklist_buffers::WorklistBuffers;

// --- Struct Definitions --- //

/// Uniform buffer structure holding parameters accessible by GPU compute shaders.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
pub struct GpuParamsUniform {
    pub grid_width: u32,
    pub grid_height: u32,
    pub grid_depth: u32,
    pub num_tiles: u32,
    pub num_axes: u32,
    pub boundary_mode: u32,
    pub heuristic_type: u32,
    pub tie_breaking: u32,
    pub max_propagation_steps: u32,
    pub contradiction_check_frequency: u32,
    pub worklist_size: u32,
    pub grid_element_count: u32,
    pub _padding: u32,
}

/// Uniform buffer structure for entropy shader parameters.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuEntropyShaderParams {
    pub grid_dims: [u32; 3],
    pub heuristic_type: u32,
    pub num_tiles: u32,
    pub u32s_per_cell: u32,
    pub _padding1: u32,
    pub _padding2: u32,
}

/// DynamicBufferConfig contains settings for how buffers are resized
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicBufferConfig {
    pub growth_factor: f32,
    pub min_buffer_size: u64,
    pub max_buffer_size: u64,
    pub auto_resize: bool,
}

impl Default for DynamicBufferConfig {
    fn default() -> Self {
        Self {
            growth_factor: 1.5,
            min_buffer_size: 1024,
            max_buffer_size: 1024 * 1024 * 1024,
            auto_resize: true,
        }
    }
}

/// Stores all the GPU buffers needed for the WFC algorithm.
#[derive(Debug, Clone)]
pub struct GpuBuffers {
    pub grid_buffers: GridBuffers,
    pub rules_buf: Arc<wgpu::Buffer>,
    pub entropy_buffers: EntropyBuffers,
    pub contradiction_flag_buf: Arc<wgpu::Buffer>,
    pub staging_contradiction_flag_buf: Arc<wgpu::Buffer>,
    pub contradiction_location_buf: Arc<wgpu::Buffer>,
    pub staging_contradiction_location_buf: Arc<wgpu::Buffer>,
    pub params_uniform_buf: Arc<wgpu::Buffer>,
    pub adjacency_rules_buf: Arc<wgpu::Buffer>,
    pub rule_weights_buf: Arc<wgpu::Buffer>,
    pub pass_statistics_buf: Arc<wgpu::Buffer>,
    pub staging_pass_statistics_buf: Arc<wgpu::Buffer>,
    pub worklist_buffers: WorklistBuffers,
    pub num_cells: usize,
    pub original_grid_dims: Option<(usize, usize, usize)>,
    pub grid_dims: (usize, usize, usize),
    pub num_tiles: usize,
    pub num_axes: usize,
    pub boundary_mode: wfc_core::BoundaryCondition,
    pub entropy_params_buffer: Arc<wgpu::Buffer>,
    pub dynamic_buffer_config: Option<DynamicBufferConfig>,
}

/// Holds the results downloaded from the GPU after relevant compute passes.
#[derive(Debug, Clone, Default)]
pub struct GpuDownloadResults {
    pub entropy: Option<Vec<f32>>,
    pub min_entropy_info: Option<(f32, u32)>,
    pub contradiction_flag: Option<bool>,
    pub contradiction_location: Option<u32>,
    pub grid_possibilities: Option<Vec<u32>>,
}

/// Specifies which data to download from the GPU.
pub struct DownloadRequest {
    pub download_entropy: bool,
    pub download_min_entropy_info: bool,
    pub download_grid_possibilities: bool,
    pub download_contradiction_location: bool,
}

// --- Implementations --- //

impl GpuBuffers {
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        initial_grid: &PossibilityGrid,
        rules: &AdjacencyRules,
        boundary_mode: BoundaryCondition,
    ) -> Result<Self, GpuError> {
        info!(
            "Creating GPU buffers with boundary mode: {:?}...",
            boundary_mode
        );
        let width = initial_grid.width;
        let height = initial_grid.height;
        let depth = initial_grid.depth;
        let num_cells = width * height * depth;
        let num_tiles = rules.num_tiles();
        let num_axes = rules.num_axes();

        let default_dynamic_config = DynamicBufferConfig::default();
        let grid_buffers = GridBuffers::new(device, initial_grid, &default_dynamic_config)?;
        let worklist_buffers = WorklistBuffers::new(device, num_cells, &default_dynamic_config)?;
        let entropy_buffers = EntropyBuffers::new(device, num_cells, &default_dynamic_config)?;

        let mut weighted_rules = Vec::new();
        for ((axis, tile1, tile2), weight) in rules.get_weighted_rules_map() {
            if *weight < 1.0 {
                let rule_idx = axis * num_tiles * num_tiles + tile1 * num_tiles + tile2;
                weighted_rules.push(rule_idx as u32);
                weighted_rules.push(weight.to_bits());
            }
        }
        if weighted_rules.is_empty() {
            weighted_rules.push(0);
            weighted_rules.push(1.0f32.to_bits());
        }

        let params = GpuParamsUniform {
            grid_width: width as u32,
            grid_height: height as u32,
            grid_depth: depth as u32,
            num_tiles: num_tiles as u32,
            num_axes: num_axes as u32,
            boundary_mode: match boundary_mode {
                BoundaryCondition::Finite => 0,
                BoundaryCondition::Periodic => 1,
            },
            heuristic_type: 0,
            tie_breaking: 0,
            max_propagation_steps: 1000,
            contradiction_check_frequency: 10,
            worklist_size: 0,
            grid_element_count: (num_cells * grid_buffers.u32s_per_cell) as u32,
            _padding: 0,
        };

        let contradiction_buffer_size = std::mem::size_of::<u32>() as u64;
        let contradiction_location_buffer_size = std::mem::size_of::<u32>() as u64;

        let rules_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dummy Rules Buf"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));

        let contradiction_flag_buf = Self::create_buffer(
            device,
            contradiction_buffer_size,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            Some("Contradiction Flag"),
        );
        let staging_contradiction_flag_buf = Self::create_buffer(
            device,
            contradiction_buffer_size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            Some("Staging Contradiction Flag"),
        );
        let contradiction_location_buf = Self::create_buffer(
            device,
            contradiction_location_buffer_size,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            Some("Contradiction Location"),
        );
        let staging_contradiction_location_buf = Self::create_buffer(
            device,
            contradiction_location_buffer_size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            Some("Staging Contradiction Location"),
        );
        let params_uniform_buf = Arc::new(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Params Uniform"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            },
        ));
        let adjacency_rules_buf = Arc::new(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("WFC Adjacency Rules Buffer"),
                contents: bytemuck::cast_slice(&weighted_rules),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            },
        ));
        let rule_weights_buf = Self::create_buffer(
            device,
            16,
            wgpu::BufferUsages::STORAGE,
            Some("Dummy Rule Weights"),
        ); // Placeholder
        let pass_statistics_buf = Self::create_buffer(
            device,
            16,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            Some("Pass Statistics"),
        );
        let staging_pass_statistics_buf = Self::create_buffer(
            device,
            16,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            Some("Staging Pass Statistics"),
        );
        let entropy_params_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Entropy Params Uniform"),
            size: std::mem::size_of::<GpuEntropyShaderParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        info!("GPU buffers created successfully.");
        Ok(Self {
            grid_buffers,
            rules_buf,
            entropy_buffers,
            contradiction_flag_buf,
            staging_contradiction_flag_buf,
            contradiction_location_buf,
            staging_contradiction_location_buf,
            params_uniform_buf,
            adjacency_rules_buf,
            rule_weights_buf,
            pass_statistics_buf,
            staging_pass_statistics_buf,
            worklist_buffers,
            num_cells,
            original_grid_dims: None,
            grid_dims: (width, height, depth),
            num_tiles,
            num_axes,
            boundary_mode,
            entropy_params_buffer,
            dynamic_buffer_config: Some(default_dynamic_config),
        })
    }

    pub fn create_buffer(
        device: &wgpu::Device,
        size: u64,
        usage: wgpu::BufferUsages,
        label: Option<&str>,
    ) -> Arc<wgpu::Buffer> {
        let padded_size = size.max(4); // Ensure minimum size of 4 bytes for alignment
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label,
            size: padded_size,
            usage,
            mapped_at_creation: false,
        });
        Arc::new(buffer)
    }

    pub fn resize_buffer(
        device: &wgpu::Device,
        old_buffer: &Arc<wgpu::Buffer>,
        new_size: u64,
        usage: wgpu::BufferUsages,
        label: Option<&str>,
        config: &DynamicBufferConfig,
    ) -> Arc<wgpu::Buffer> {
        let old_size = old_buffer.size();
        let mut actual_new_size = (new_size as f64 * config.growth_factor as f64).ceil() as u64;
        actual_new_size = actual_new_size.max(config.min_buffer_size);
        actual_new_size = actual_new_size.min(config.max_buffer_size);
        log::debug!(
            "Resizing buffer '{}' from {} to {} bytes (required: {})",
            label.unwrap_or("Unnamed"),
            old_size,
            actual_new_size,
            new_size
        );
        Self::create_buffer(device, actual_new_size, usage, label)
    }

    pub fn is_buffer_sufficient(buffer: &Arc<wgpu::Buffer>, required_size: u64) -> bool {
        buffer.size() >= required_size
    }

    pub fn resize_for_grid(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        depth: u32,
        num_tiles: u32,
        config: &DynamicBufferConfig,
    ) -> Result<(), String> {
        self.grid_buffers
            .ensure_grid_possibilities_buffer(device, width, height, depth, num_tiles, config)?;
        let num_cells = (width * height * depth) as usize;
        self.entropy_buffers
            .ensure_buffers(device, num_cells, config)?;
        self.worklist_buffers
            .ensure_worklist_buffers(device, width, height, depth, config)?;
        self.num_cells = num_cells;
        self.grid_dims = (width as usize, height as usize, depth as usize);
        Ok(())
    }

    pub async fn download_results(
        &self,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        request: DownloadRequest,
    ) -> Result<GpuDownloadResults, GpuError> {
        let recoverable_op = RecoverableGpuOp::default();
        recoverable_op
            .try_with_recovery(|| async {
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Download Results Encoder"),
                });
                let mut futures: Vec<Pin<Box<dyn Future<Output = Result<(String, Box<dyn Any>), GpuError>> + Send>>> = Vec::new();
                let operation_timeout = Duration::from_secs(10); // Example timeout

                if request.download_entropy {
                    encoder.copy_buffer_to_buffer(
                        &self.entropy_buffers.entropy_buf,
                        0,
                        &self.entropy_buffers.staging_entropy_buf,
                        0,
                        self.entropy_buffers.entropy_buf.size(),
                    );
                    let buffer = self.entropy_buffers.staging_entropy_buf.clone();
                    let expected_size = self.num_cells;
                    futures.push(
                        map_and_process::<f32>(buffer, operation_timeout, Some(expected_size))
                            .map(|r| r.map(|d| ("entropy".to_string(), d as Box<dyn Any>)))
                            .boxed(),
                    );
                }
                if request.download_min_entropy_info {
                    encoder.copy_buffer_to_buffer(
                        &self.entropy_buffers.min_entropy_info_buf,
                        0,
                        &self.entropy_buffers.staging_min_entropy_info_buf,
                        0,
                        self.entropy_buffers.min_entropy_info_buf.size(),
                    );
                    let buffer = self.entropy_buffers.staging_min_entropy_info_buf.clone();
                    futures.push(
                         map_and_process::<u32>(buffer, operation_timeout, Some(2)) // Expect [f32_bits, u32_idx]
                             .map(|r| r.map(|d| ("min_entropy_info".to_string(), d as Box<dyn Any>)))
                             .boxed(),
                     );
                }
                if request.download_grid_possibilities {
                    encoder.copy_buffer_to_buffer(
                        &self.grid_buffers.grid_possibilities_buf,
                        0,
                        &self.grid_buffers.staging_grid_possibilities_buf,
                        0,
                        self.grid_buffers.grid_possibilities_buf.size(),
                    );
                    let buffer = self.grid_buffers.staging_grid_possibilities_buf.clone();
                    let expected_size = self.num_cells * self.grid_buffers.u32s_per_cell;
                    futures.push(
                        map_and_process::<u32>(buffer, operation_timeout, Some(expected_size))
                            .map(|r| r.map(|d| ("grid_possibilities".to_string(), d as Box<dyn Any>)))
                            .boxed(),
                    );
                }
                if request.download_contradiction_location {
                    encoder.copy_buffer_to_buffer(
                        &self.contradiction_flag_buf,
                        0,
                        &self.staging_contradiction_flag_buf,
                        0,
                        self.contradiction_flag_buf.size(),
                    );
                    encoder.copy_buffer_to_buffer(
                        &self.contradiction_location_buf,
                        0,
                        &self.staging_contradiction_location_buf,
                        0,
                        self.contradiction_location_buf.size(),
                    );
                    let flag_buffer = self.staging_contradiction_flag_buf.clone();
                    futures.push(
                        map_and_process::<u32>(flag_buffer, operation_timeout, Some(1))
                            .map(|r| r.map(|d| ("contradiction_flag".to_string(), d as Box<dyn Any>)))
                            .boxed(),
                    );
                    let loc_buffer = self.staging_contradiction_location_buf.clone();
                    futures.push(
                        map_and_process::<u32>(loc_buffer, operation_timeout, Some(1))
                            .map(|r| r.map(|d| ("contradiction_location".to_string(), d as Box<dyn Any>)))
                            .boxed(),
                    );
                }

                queue.submit(Some(encoder.finish()));
                self.map_downloaded_data(futures, operation_timeout).await
            })
            .await
    }

    async fn map_downloaded_data(
        &self,
        futures: Vec<
            Pin<Box<dyn Future<Output = Result<(String, Box<dyn Any>), GpuError>> + Send>>,
        >,
        _operation_timeout: Duration, // Timeout handled within map_and_process
    ) -> Result<GpuDownloadResults, GpuError> {
        let results = try_join_all(futures).await?;
        let mut download_data = GpuDownloadResults::default();
        let data_map: HashMap<String, Box<dyn Any>> = results.into_iter().collect();

        for (key, data_box) in data_map {
            match key.as_str() {
                "entropy" => {
                    if let Ok(b) = data_box.downcast::<Vec<f32>>() {
                        download_data.entropy = Some(*b);
                    } else {
                        warn!("Failed to downcast entropy data Box<dyn Any> to Box<Vec<f32>>");
                    }
                }
                "min_entropy_info" => {
                    if let Ok(b) = data_box.downcast::<Vec<u32>>() {
                        if b.len() >= 2 {
                            download_data.min_entropy_info = Some((f32::from_bits(b[0]), b[1]));
                        } else {
                            warn!("min_entropy_info data has len {}, expected >= 2", b.len());
                        }
                    } else {
                        warn!("Failed to downcast min_entropy_info data Box<dyn Any> to Box<Vec<u32>>");
                    }
                }
                "grid_possibilities" => {
                    if let Ok(d) = data_box.downcast::<Vec<u32>>() {
                        download_data.grid_possibilities = Some(*d);
                    } else {
                        warn!("Failed to downcast grid_possibilities data Box<dyn Any> to Box<Vec<u32>>");
                    }
                }
                "contradiction_flag" => {
                    if let Ok(b) = data_box.downcast::<Vec<u32>>() {
                        if !b.is_empty() {
                            download_data.contradiction_flag = Some(b[0] != 0);
                        } else {
                            warn!("contradiction_flag data is empty");
                        }
                    } else {
                        warn!("Failed to downcast contradiction_flag data Box<dyn Any> to Box<Vec<u32>>");
                    }
                }
                "contradiction_location" => {
                    if let Ok(b) = data_box.downcast::<Vec<u32>>() {
                        if !b.is_empty() {
                            download_data.contradiction_location = Some(b[0]);
                        } else {
                            warn!("contradiction_location data is empty");
                        }
                    } else {
                        warn!("Failed to downcast contradiction_location data Box<dyn Any> to Box<Vec<u32>>");
                    }
                }
                _ => warn!("Unknown key found during data mapping: {}", key),
            }
        }
        Ok(download_data)
    }
}

/// Helper function to map a buffer and process its contents.
pub(crate) async fn map_and_process<
    T: bytemuck::Pod + bytemuck::Zeroable + Default + Clone + Send + 'static,
>(
    buffer: Arc<wgpu::Buffer>,
    timeout: Duration,
    expected_elements: Option<usize>,
) -> Result<Box<dyn Any>, GpuError> {
    let buffer_slice = buffer.slice(..);
    let (tx, rx) = tokio::sync::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });

    // This poll is important to drive the map_async operation forward,
    // especially if the device is not being polled elsewhere frequently.
    // Using Maintain::Poll allows other futures to run.
    // Consider using Maintain::Wait if blocking is acceptable/necessary.
    // buffer.device().poll(wgpu::Maintain::Poll); // Device polling might be handled elsewhere

    match timeout_after(rx, timeout).await {
        Ok(Ok(Ok(()))) => {
            let mapped_range = buffer_slice.get_mapped_range();
            let data: Vec<T> = bytemuck::cast_slice(&mapped_range).to_vec();
            drop(mapped_range);
            buffer.unmap();

            if let Some(expected) = expected_elements {
                if data.len() < expected {
                    return Err(GpuError::BufferSizeMismatch(format!(
                        "Downloaded data size mismatch. Expected at least {} elements, Got {}.",
                        expected,
                        data.len()
                    )));
                }
                // Optionally truncate if exactly `expected` are needed
                // data.truncate(expected);
            }

            Ok(Box::new(data))
        }
        Ok(Ok(Err(e))) => Err(GpuError::BufferMapFailed(e)),
        Ok(Err(_elapsed)) => Err(GpuError::BufferOperationError(
            "Buffer mapping timed out".to_string(),
        )),
        Err(_recv_error) => Err(GpuError::InternalError(
            "Buffer mapping channel closed unexpectedly".to_string(),
        )),
    }
}

// Custom timeout helper to avoid external crate dependency if needed
async fn timeout_after<F: Future>(
    future: F,
    duration: Duration,
) -> Result<F::Output, tokio::time::error::Elapsed> {
    tokio::time::timeout(duration, future).await
}

// Test module remains in the respective files (grid_buffers.rs, worklist_buffers.rs)
// OR could be moved here if testing GpuBuffers itself.
