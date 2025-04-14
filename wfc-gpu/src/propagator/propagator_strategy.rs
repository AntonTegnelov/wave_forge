//! Propagation strategy implementations for the WFC algorithm.
//! This module provides different strategies for propagating constraints
//! after a cell collapse in the Wave Function Collapse algorithm.

use crate::{
    buffers::GpuBuffers,
    gpu::sync::GpuSynchronizer,
    utils::error_recovery::{GpuError, GridCoord},
};
use async_trait;
use std::sync::Arc;
use wfc_core::{grid::PossibilityGrid, propagator::PropagationError};

/// Data structure for holding SubgridData properties
#[derive(Debug)]
struct SubgridData {
    num_tiles: u32,
    num_axes: u32,
    boundary_mode: wfc_core::BoundaryCondition,
    heuristic_type: u32,
    tie_breaking: u32,
    max_propagation_steps: u32,
    contradiction_check_frequency: u32,
    worklist_size: u32,
    propagator: Box<dyn PropagationStrategy>,
}

/// Strategy trait for constraint propagation in WFC algorithm.
/// This trait contains only synchronous methods for object safety.
pub trait PropagationStrategy: Send + Sync + std::fmt::Debug {
    /// Get the name of this propagation strategy
    fn name(&self) -> &str;

    /// Prepare for propagation by initializing any necessary buffers or state
    fn prepare(&self, synchronizer: &GpuSynchronizer) -> Result<(), PropagationError>;

    /// Clean up any resources used during propagation
    fn cleanup(&self, synchronizer: &GpuSynchronizer) -> Result<(), PropagationError>;
}

/// Async extension of the PropagationStrategy trait.
/// This trait contains the async propagate method which can't be part of
/// the object-safe PropagationStrategy trait.
#[async_trait::async_trait]
pub trait AsyncPropagationStrategy: PropagationStrategy {
    /// Propagate constraints from the specified cells
    async fn propagate(
        &self,
        grid: &mut PossibilityGrid,
        updated_cells: &[GridCoord],
        buffers: &Arc<GpuBuffers>,
        synchronizer: &GpuSynchronizer,
    ) -> Result<(), PropagationError>;
}

/// Direct propagation strategy - propagates constraints directly across
/// the entire grid without any partitioning or optimization.
#[derive(Debug)]
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

#[async_trait::async_trait]
impl PropagationStrategy for DirectPropagationStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn prepare(&self, _synchronizer: &GpuSynchronizer) -> Result<(), PropagationError> {
        // Direct propagation doesn't need special preparation
        Ok(())
    }

    fn cleanup(&self, _synchronizer: &GpuSynchronizer) -> Result<(), PropagationError> {
        // Direct propagation doesn't need special cleanup
        Ok(())
    }
}

#[async_trait::async_trait]
impl AsyncPropagationStrategy for DirectPropagationStrategy {
    async fn propagate(
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
            bytemuck::cast_slice(&[0_u32]),
        );

        // Initialize the worklist with the updated indices
        let max_worklist_size = buffers.worklist_buffers.current_worklist_size;

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
        let mut contradiction_flag = [0_u32; 1];
        let mut worklist_count = [0_u32; 1];

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

                // Create bind group layout
                let bind_group_layout =
                    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("PropagationBindGroupLayout"),
                        entries: &[
                            // Grid possibilities buffer
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // Adjacency rules buffer
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // Input worklist buffer
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // Output worklist buffer
                            wgpu::BindGroupLayoutEntry {
                                binding: 3,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // Input worklist count buffer
                            wgpu::BindGroupLayoutEntry {
                                binding: 4,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // Output worklist count buffer
                            wgpu::BindGroupLayoutEntry {
                                binding: 5,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // Parameters buffer
                            wgpu::BindGroupLayoutEntry {
                                binding: 6,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // Contradiction flag buffer
                            wgpu::BindGroupLayoutEntry {
                                binding: 7,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    });

                // Create pipeline layout using the bind group layout
                let pipeline_layout =
                    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("PropagationPipelineLayout"),
                        bind_group_layouts: &[&bind_group_layout],
                        push_constant_ranges: &[],
                    });

                // Set pipeline and bind groups
                compute_pass.set_pipeline(&device.create_compute_pipeline(
                    &wgpu::ComputePipelineDescriptor {
                        label: Some("Propagation Pipeline"),
                        layout: Some(&pipeline_layout),
                        module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                            label: Some("Propagation Shader"),
                            source: wgpu::ShaderSource::Wgsl("".into()),
                        }),
                        entry_point: Some("main"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        cache: None,
                    },
                ));
                compute_pass.set_bind_group(0, &bind_group, &[]);
            }

            // Submit command encoder
            queue.submit(Some(encoder.finish()));

            // Read worklist count before dispatch to determine workgroup count
            let temp_worklist = synchronizer
                .download_buffer(
                    &buffers.worklist_buffers.worklist_count_buf,
                    0,
                    std::mem::size_of::<u32>(),
                )
                .map_err(gpu_error_to_propagation_error)?;
            worklist_count = [temp_worklist[0]]; // Convert Vec to array

            let current_worklist_count = worklist_count[0] as usize;

            // Start a new encoder and compute pass with the right workgroup size
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("PropagationDispatchPass-{}", pass_idx)),
            });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("{}DispatchPass-{}", label, pass_idx)),
                    timestamp_writes: None,
                });

                // Create bind group layout
                let bind_group_layout =
                    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("PropagationBindGroupLayout"),
                        entries: &[
                            // Grid possibilities buffer
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // Adjacency rules buffer
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // Input worklist buffer
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // Output worklist buffer
                            wgpu::BindGroupLayoutEntry {
                                binding: 3,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // Input worklist count buffer
                            wgpu::BindGroupLayoutEntry {
                                binding: 4,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // Output worklist count buffer
                            wgpu::BindGroupLayoutEntry {
                                binding: 5,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // Parameters buffer
                            wgpu::BindGroupLayoutEntry {
                                binding: 6,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // Contradiction flag buffer
                            wgpu::BindGroupLayoutEntry {
                                binding: 7,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    });

                // Create pipeline layout using the bind group layout
                let pipeline_layout =
                    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("PropagationPipelineLayout"),
                        bind_group_layouts: &[&bind_group_layout],
                        push_constant_ranges: &[],
                    });

                // Use pipeline layout in compute pipeline descriptor
                compute_pass.set_pipeline(&device.create_compute_pipeline(
                    &wgpu::ComputePipelineDescriptor {
                        label: Some("Propagation Pipeline"),
                        layout: Some(&pipeline_layout),
                        module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                            label: Some("Propagation Shader"),
                            source: wgpu::ShaderSource::Wgsl("".into()),
                        }),
                        entry_point: Some("main"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        cache: None,
                    },
                ));
                compute_pass.set_bind_group(
                    0,
                    &device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("PropagationBindGroup"),
                        layout: &bind_group_layout,
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
                                resource: buffers
                                    .rule_buffers
                                    .adjacency_rules_buf
                                    .as_entire_binding(),
                            },
                            // Input worklist buffer
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: buffers
                                    .worklist_buffers
                                    .worklist_buf_a
                                    .as_entire_binding(),
                            },
                            // Output worklist buffer
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: buffers
                                    .worklist_buffers
                                    .worklist_buf_b
                                    .as_entire_binding(),
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
                                resource: buffers.params_uniform_buf.as_entire_binding(),
                            },
                        ],
                    }),
                    &[],
                );

                // Calculate workgroup count
                let workgroup_size = 64;
                let workgroup_count =
                    ((current_worklist_count as f32 / workgroup_size as f32).ceil()) as u32;

                if workgroup_count > 0 {
                    // Only dispatch if we have items in the worklist
                    compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
                }
            }

            // End the compute pass scope first
            // Now we can finish the encoder
            let command_buffer = encoder.finish();

            // Submit the command buffer
            queue.submit(Some(command_buffer));

            // Check for contradiction after each pass
            let temp_flag = synchronizer
                .download_buffer(
                    &buffers.contradiction_flag_buf,
                    0,
                    std::mem::size_of::<u32>(),
                )
                .map_err(gpu_error_to_propagation_error)?;
            contradiction_flag = [temp_flag[0]]; // Convert Vec to array

            if contradiction_flag[0] != 0 {
                return Err(PropagationError::Contradiction(0, 0, 0));
            }

            // Check the output worklist size to determine whether to continue
            let output_idx = 1 - current_worklist_idx;
            let output_worklist_count_buf = &buffers.worklist_buffers.worklist_count_buf;

            // Read the output worklist count
            let temp_count = synchronizer
                .download_buffer(output_worklist_count_buf, 0, std::mem::size_of::<u32>())
                .map_err(gpu_error_to_propagation_error)?;
            worklist_count = [temp_count[0]]; // Convert Vec to array

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
            layout: &device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("PropagationBindGroupLayout"),
                entries: &[
                    // Grid possibilities buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Adjacency rules buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Input worklist buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Output worklist buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Input worklist count buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Output worklist count buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Parameters buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Contradiction flag buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            }),
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
                    resource: buffers.params_uniform_buf.as_entire_binding(),
                },
            ],
        })
    }
}

/// Subgrid propagation strategy - divides the grid into smaller subgrids
/// to optimize propagation for large grids.
#[derive(Debug)]
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

#[async_trait::async_trait]
impl PropagationStrategy for SubgridPropagationStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn prepare(&self, _synchronizer: &GpuSynchronizer) -> Result<(), PropagationError> {
        // Subgrid propagation may need to initialize subgrid buffers
        // Now implemented
        Ok(())
    }

    fn cleanup(&self, _synchronizer: &GpuSynchronizer) -> Result<(), PropagationError> {
        // Subgrid propagation may need to clean up temporary buffers
        Ok(())
    }
}

/// Adaptive propagation strategy - automatically selects the best strategy
/// based on grid size and other factors.
#[derive(Debug)]
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
    fn select_strategy(&self, grid: &PossibilityGrid) -> Strategy {
        let total_cells = grid.width * grid.height * grid.depth;
        if total_cells > self.size_threshold {
            Strategy::Subgrid
        } else {
            Strategy::Direct
        }
    }
}

/// Enum to represent the different propagation strategies
#[derive(Debug, Clone, Copy)]
enum Strategy {
    Direct,
    Subgrid,
}

#[async_trait::async_trait]
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

    fn cleanup(&self, synchronizer: &GpuSynchronizer) -> Result<(), PropagationError> {
        // Clean up both strategies
        self.direct_strategy.cleanup(synchronizer)?;
        self.subgrid_strategy.cleanup(synchronizer)?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl AsyncPropagationStrategy for AdaptivePropagationStrategy {
    async fn propagate(
        &self,
        grid: &mut PossibilityGrid,
        updated_cells: &[GridCoord],
        buffers: &Arc<GpuBuffers>,
        synchronizer: &GpuSynchronizer,
    ) -> Result<(), PropagationError> {
        // Delegate to the appropriate strategy based on grid size
        let strategy = self.select_strategy(grid);

        // Use dynamic dispatch through the AsyncPropagationStrategy trait
        match strategy {
            Strategy::Direct => {
                self.direct_strategy
                    .propagate(grid, updated_cells, buffers, synchronizer)
                    .await
            }
            Strategy::Subgrid => {
                self.subgrid_strategy
                    .propagate(grid, updated_cells, buffers, synchronizer)
                    .await
            }
        }
    }
}

/// Factory for creating propagation strategy instances
pub struct PropagationStrategyFactory;

impl PropagationStrategyFactory {
    /// Create a direct propagation strategy with custom settings
    pub fn create_direct(max_iterations: u32) -> Box<dyn PropagationStrategy + Send + Sync> {
        Box::new(DirectPropagationStrategy::new(max_iterations))
    }

    /// Create a subgrid propagation strategy with custom settings
    pub fn create_subgrid(
        max_iterations: u32,
        subgrid_size: u32,
    ) -> Box<dyn PropagationStrategy + Send + Sync> {
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
    ) -> Box<dyn PropagationStrategy + Send + Sync> {
        Box::new(AdaptivePropagationStrategy::new(
            max_iterations,
            subgrid_size,
            size_threshold,
        ))
    }

    /// Create a strategy based on the grid size
    pub fn create_for_grid(grid: &PossibilityGrid) -> Box<dyn PropagationStrategy + Send + Sync> {
        let total_cells = grid.width * grid.height * grid.depth;
        if total_cells > 4096 {
            Self::create_subgrid(1000, 16)
        } else {
            Self::create_direct(1000)
        }
    }

    /// Create a direct strategy that also implements AsyncPropagationStrategy
    pub fn create_direct_async(
        max_iterations: u32,
    ) -> Box<dyn AsyncPropagationStrategy + Send + Sync> {
        Box::new(DirectPropagationStrategy::new(max_iterations))
    }

    /// Create a subgrid strategy that also implements AsyncPropagationStrategy
    pub fn create_subgrid_async(
        max_iterations: u32,
        subgrid_size: u32,
    ) -> Box<dyn AsyncPropagationStrategy + Send + Sync> {
        Box::new(SubgridPropagationStrategy::new(
            max_iterations,
            subgrid_size,
        ))
    }

    /// Create an adaptive strategy that also implements AsyncPropagationStrategy
    pub fn create_adaptive_async(
        max_iterations: u32,
        subgrid_size: u32,
        size_threshold: usize,
    ) -> Box<dyn AsyncPropagationStrategy + Send + Sync> {
        Box::new(AdaptivePropagationStrategy::new(
            max_iterations,
            subgrid_size,
            size_threshold,
        ))
    }

    /// Create a strategy based on the grid size that also implements AsyncPropagationStrategy
    pub fn create_for_grid_async(
        grid: &PossibilityGrid,
    ) -> Box<dyn AsyncPropagationStrategy + Send + Sync> {
        let total_cells = grid.width * grid.height * grid.depth;
        if total_cells > 4096 {
            Self::create_subgrid_async(1000, 16)
        } else {
            Self::create_direct_async(1000)
        }
    }
}

/// Helper function to convert GpuError to PropagationError consistently
pub fn gpu_error_to_propagation_error(error: GpuError) -> PropagationError {
    match error {
        GpuError::ContradictionDetected { context } => {
            // Parse the context string to extract coordinates if present
            // Format is expected to be something like "at coordinates (x, y, z)"
            let context_str = context.to_string();

            // For now, we'll create a general error as the context string
            // doesn't have structured grid coordinates
            PropagationError::InternalError(format!("Contradiction detected: {}", context_str))
        }
        _ => PropagationError::InternalError(format!("GPU error: {}", error)),
    }
}
