// wfc-gpu/src/buffers/rule_buffers.rs

//! Module for GPU buffers related to WFC rules and constraints.

use crate::buffers::DynamicBufferConfig;
use crate::buffers::GpuBuffers; // For GpuBuffers::create_buffer
use crate::utils::error_recovery::GpuError;
use std::sync::Arc;
use wfc_rules::AdjacencyRules;
use wgpu::{util::DeviceExt, BufferUsages}; // Import DynamicBufferConfig

/// Holds GPU buffers containing WFC adjacency rules and related data.
#[derive(Debug, Clone)] // Added Clone as buffers are Arc
pub struct RuleBuffers {
    /// Buffer storing basic adjacency rules as a bit array
    pub rules_buf: Arc<wgpu::Buffer>,
    /// Buffer storing weighted rules data (indices and weights)
    pub rule_weights_buf: Arc<wgpu::Buffer>,
}

impl RuleBuffers {
    /// Helper function to pack adjacency rules into a bit array
    fn pack_adjacency_rules(rules: &AdjacencyRules) -> Vec<u32> {
        let num_tiles = rules.num_tiles();
        let num_axes = rules.num_axes();

        // Calculate total number of rules and required u32s
        let total_rules = num_axes * num_tiles * num_tiles;
        let num_u32s = (total_rules + 31) / 32; // Round up division

        // Initialize bit array
        let mut bit_array = vec![0u32; num_u32s];

        // Pack each allowed rule into the bit array
        for ((axis, tile1, tile2), _) in rules.get_allowed_rules_map() {
            let rule_idx = axis * num_tiles * num_tiles + tile1 * num_tiles + tile2;
            let u32_idx = rule_idx / 32;
            let bit_idx = rule_idx % 32;

            // Set the bit for this rule
            bit_array[u32_idx] |= 1u32 << bit_idx;
        }

        bit_array
    }

    /// Creates new rule-related GPU buffers.
    pub fn new(
        device: &wgpu::Device,
        rules: &AdjacencyRules,
        _config: &DynamicBufferConfig,
    ) -> Result<Self, GpuError> {
        let num_tiles = rules.num_tiles();

        // Pack basic adjacency rules into a bit array
        let adjacency_bits = Self::pack_adjacency_rules(rules);

        // Create the basic rules buffer
        let rules_buf = Arc::new(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("WFC Basic Rules Buffer"),
                contents: bytemuck::cast_slice(&adjacency_bits),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            }),
        );

        // Prepare weighted rules data
        let mut weighted_rules_data = Vec::new();
        for ((axis, tile1, tile2), weight) in rules.get_weighted_rules_map() {
            // Only include rules with non-default weights
            if *weight != 1.0 {
                let rule_idx = axis * num_tiles * num_tiles + tile1 * num_tiles + tile2;
                weighted_rules_data.push(rule_idx as u32);
                weighted_rules_data.push(weight.to_bits()); // Store f32 weight as u32 bits
            }
        }

        // If no specific weights are found, add a dummy entry
        if weighted_rules_data.is_empty() {
            weighted_rules_data.push(0); // Dummy index
            weighted_rules_data.push(1.0f32.to_bits()); // Dummy weight (1.0)
        }

        // Create the weighted rules buffer
        let rule_weights_buf = Arc::new(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("WFC Rule Weights Buffer"),
                contents: bytemuck::cast_slice(&weighted_rules_data),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            },
        ));

        Ok(Self {
            rules_buf,
            rule_weights_buf,
        })
    }

    // TODO: Add methods for updating rules if needed (e.g., for dynamic rule changes)
    // pub fn update_rules(...) -> Result<(), GpuError> { ... }
}

// TODO: Add tests specific to RuleBuffers if needed
