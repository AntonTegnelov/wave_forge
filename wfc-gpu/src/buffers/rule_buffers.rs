// wfc-gpu/src/buffers/rule_buffers.rs

//! Module for GPU buffers related to WFC rules and constraints.

use crate::buffers::DynamicBufferConfig;
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
    pub(crate) fn pack_adjacency_rules(rules: &AdjacencyRules) -> Vec<u32> {
        let num_tiles = rules.num_tiles();
        let num_axes = rules.num_axes();

        // Calculate total number of rules and required u32s
        let total_rules = num_axes * num_tiles * num_tiles;
        let num_u32s = (total_rules + 31) / 32; // Round up division

        // Initialize bit array
        let mut bit_array = vec![0u32; num_u32s];

        // Pack each allowed rule into the bit array
        for (axis, tile1, tile2) in rules.get_allowed_rules_map().keys() {
            let rule_idx = axis * num_tiles * num_tiles + tile1 * num_tiles + tile2;
            let u32_idx = rule_idx / 32;
            let bit_idx = rule_idx % 32;

            // Set the bit for this rule
            bit_array[u32_idx] |= 1u32 << bit_idx;
        }

        bit_array
    }

    /// Creates new rule-related GPU buffers.
    /// Packs non-default rule weights as (rule index, f32 bits) pairs, with a dummy entry if none exist.
    pub(crate) fn pack_rule_weights(rules: &AdjacencyRules) -> Vec<u32> {
        let num_tiles = rules.num_tiles();
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
        weighted_rules_data
    }

    pub fn new(
        device: &wgpu::Device,
        rules: &AdjacencyRules,
        _config: &DynamicBufferConfig,
    ) -> Result<Self, GpuError> {

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
        let weighted_rules_data = Self::pack_rule_weights(rules);
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

#[cfg(test)]
mod tests {
    use super::RuleBuffers;
    use wfc_rules::AdjacencyRules;

    #[test]
    fn packs_each_allowed_rule_at_its_axis_tile_tile_bit() {
        // propagate.wgsl reads bit `axis * n * n + tile1 * n + tile2`; host and shader must agree.
        let n = 3;
        let rules = AdjacencyRules::from_allowed_tuples(n, 6, vec![(0, 0, 0), (1, 2, 1), (5, 2, 2)]);
        let words = RuleBuffers::pack_adjacency_rules(&rules);
        assert_eq!(words.len(), (6 * n * n).div_ceil(32));
        let set_bits: Vec<usize> = (0..words.len() * 32)
            .filter(|bit| words[bit / 32] & (1 << (bit % 32)) != 0)
            .collect();
        assert_eq!(set_bits, vec![0, n * n + 2 * n + 1, 5 * n * n + 2 * n + 2]);
    }

    #[test]
    fn default_weights_pack_to_a_single_dummy_entry() {
        // The weights buffer must never be empty because zero-sized storage bindings are invalid.
        let rules = AdjacencyRules::from_allowed_tuples(2, 6, vec![(0, 0, 1)]);
        assert_eq!(RuleBuffers::pack_rule_weights(&rules), vec![0, 1.0f32.to_bits()]);
    }
}
