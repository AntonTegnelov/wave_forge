// wfc-gpu/src/buffers/rule_buffers.rs

//! Module for GPU buffers related to WFC rules and constraints.

use crate::buffers::DynamicBufferConfig;
use crate::utils::error_recovery::GpuError;
use std::sync::Arc;
use wfc_rules::AdjacencyRules;
use wgpu::{BufferUsages, util::DeviceExt}; // Import DynamicBufferConfig

/// Holds GPU buffers containing WFC adjacency rules and related data.
#[derive(Debug, Clone)] // Added Clone as buffers are Arc
pub struct RuleBuffers {
    /// Buffer storing basic adjacency rules as a bit array
    pub rules_buf: Arc<wgpu::Buffer>,
    /// Buffer storing weighted rules data (indices and weights)
    pub rule_weights_buf: Arc<wgpu::Buffer>,
}

impl RuleBuffers {
    /// Packs the adjacency table as one bitmask per `(axis, tile)`: the set of tiles allowed on the
    /// other side of that face.
    ///
    /// Each mask starts on a word boundary, `ceil(num_tiles / 32)` words long, so the propagation
    /// shader can union the allowed neighbours of a tile with a few word ORs. Packing the table as
    /// one flat bit array instead (bit `axis*T*T + t1*T + t2`) saves a little memory but forces the
    /// shader to test tiles one bit at a time: about `num_tiles` tests per possible tile per axis,
    /// which measured as the dominant cost of a propagation pass (docs/solver-redesign.md).
    pub(crate) fn pack_adjacency_rules(rules: &AdjacencyRules) -> Vec<u32> {
        let num_tiles = rules.num_tiles();
        let num_axes = rules.num_axes();
        let words_per_row = Self::rule_words_per_row(num_tiles);

        let mut bit_array = vec![0u32; num_axes * num_tiles * words_per_row];
        for (axis, tile1, tile2) in rules.get_allowed_rules_map().keys() {
            let row = (axis * num_tiles + tile1) * words_per_row;
            bit_array[row + tile2 / 32] |= 1u32 << (tile2 % 32);
        }
        bit_array
    }

    /// Words per `(axis, tile)` mask in [`Self::pack_adjacency_rules`]; matches the shader's
    /// `words_per_cell()`, since both are `ceil(num_tiles / 32)`.
    pub(crate) fn rule_words_per_row(num_tiles: usize) -> usize {
        num_tiles.div_ceil(32).max(1)
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
        // propagate.wgsl reads the mask for (axis, tile1) at word `(axis * n + tile1) * words_per_row`
        // and tests bit `tile2` within it; host and shader must agree.
        let n = 3;
        let rules =
            AdjacencyRules::from_allowed_tuples(n, 6, vec![(0, 0, 0), (1, 2, 1), (5, 2, 2)]);
        let words = RuleBuffers::pack_adjacency_rules(&rules);
        let per_row = RuleBuffers::rule_words_per_row(n);
        assert_eq!(words.len(), 6 * n * per_row);
        let set_bits: Vec<usize> = (0..words.len() * 32)
            .filter(|bit| words[bit / 32] & (1 << (bit % 32)) != 0)
            .collect();
        let bit_of = |axis: usize, tile1: usize, tile2: usize| {
            ((axis * n + tile1) * per_row + tile2 / 32) * 32 + tile2 % 32
        };
        assert_eq!(
            set_bits,
            vec![bit_of(0, 0, 0), bit_of(1, 2, 1), bit_of(5, 2, 2)]
        );
    }

    #[test]
    fn default_weights_pack_to_a_single_dummy_entry() {
        // The weights buffer must never be empty because zero-sized storage bindings are invalid.
        let rules = AdjacencyRules::from_allowed_tuples(2, 6, vec![(0, 0, 1)]);
        assert_eq!(
            RuleBuffers::pack_rule_weights(&rules),
            vec![0, 1.0f32.to_bits()]
        );
    }
}
