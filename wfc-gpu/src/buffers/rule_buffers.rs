// wfc-gpu/src/buffers/rule_buffers.rs

//! Module for GPU buffers related to WFC rules and constraints.

use crate::buffers::DynamicBufferConfig;
use crate::buffers::GpuBuffers; // For GpuBuffers::create_buffer
use crate::error_recovery::GpuError;
use std::sync::Arc;
use wfc_rules::AdjacencyRules;
use wgpu::{util::DeviceExt, BufferUsages}; // Import DynamicBufferConfig

/// Holds GPU buffers containing WFC adjacency rules and related data.
#[derive(Debug, Clone)] // Added Clone as buffers are Arc
pub struct RuleBuffers {
    /// Dummy buffer, likely a placeholder for more complex rule representations.
    pub rules_buf: Arc<wgpu::Buffer>,
    /// Buffer storing adjacency rules, potentially weighted.
    pub adjacency_rules_buf: Arc<wgpu::Buffer>,
    /// Dummy buffer, placeholder for rule weights if not combined with adjacency_rules_buf.
    pub rule_weights_buf: Arc<wgpu::Buffer>,
}

impl RuleBuffers {
    /// Creates new rule-related GPU buffers.
    pub fn new(
        device: &wgpu::Device,
        rules: &AdjacencyRules,
        config: &DynamicBufferConfig,
    ) -> Result<Self, GpuError> {
        let num_tiles = rules.num_transformed_tiles();
        let _num_axes = rules.num_axes();

        // Prepare weighted rules data for the buffer
        let mut weighted_rules_data = Vec::new();
        for ((axis, tile1, tile2), weight) in rules.get_weighted_rules_map() {
            // Only include rules with non-default weights (assuming default is 1.0)
            // TODO: Clarify how default weights are handled. Are they omitted or included?
            if *weight < 1.0 || *weight > 1.0 {
                // Assuming we store deviations from 1.0
                // Calculate index based on axis, tile1, tile2
                // This assumes a specific packing order, document this!
                let rule_idx = axis * num_tiles * num_tiles + tile1 * num_tiles + tile2;
                weighted_rules_data.push(rule_idx as u32);
                weighted_rules_data.push(weight.to_bits()); // Store f32 weight as u32 bits
            }
        }

        // If no specific weights are found, add a dummy entry
        // This prevents creating a zero-sized buffer, which might be problematic.
        if weighted_rules_data.is_empty() {
            weighted_rules_data.push(0); // Dummy index
            weighted_rules_data.push(1.0f32.to_bits()); // Dummy weight (1.0)
        }

        // Create the buffers
        let rules_buf = GpuBuffers::create_buffer(
            device,
            16,                    // Placeholder size
            BufferUsages::STORAGE, // Adjust usage as needed
            Some("Dummy Rules Buf"),
        );

        let adjacency_rules_buf = Arc::new(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("WFC Adjacency Rules Buffer"),
                contents: bytemuck::cast_slice(&weighted_rules_data),
                // Usage: STORAGE for reading in shaders, COPY_DST if updated later
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            },
        ));

        let rule_weights_buf = GpuBuffers::create_buffer(
            device,
            16,                    // Placeholder size
            BufferUsages::STORAGE, // Adjust usage as needed
            Some("Dummy Rule Weights"),
        );

        Ok(Self {
            rules_buf,
            adjacency_rules_buf,
            rule_weights_buf,
        })
    }

    // TODO: Add methods for updating rules if needed (e.g., for dynamic rule changes)
    // pub fn update_rules(...) -> Result<(), GpuError> { ... }
}

// TODO: Add tests specific to RuleBuffers if needed
