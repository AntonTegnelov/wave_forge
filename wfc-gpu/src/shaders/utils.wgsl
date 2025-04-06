// utils.wgsl - Generic utility functions for WGSL shaders
//
// This file contains common, generic utility functions that can be reused
// across different shader types, not specific to WFC logic.

// Count number of set bits (1s) in a u32 integer.
// Uses a parallel algorithm (sometimes called Hamming weight or population count).
fn count_ones(x: u32) -> u32 {
    var bits = x;
    bits = bits - ((bits >> 1u) & 0x55555555u);
    bits = (bits & 0x33333333u) + ((bits >> 2u) & 0x33333333u);
    bits = (bits + (bits >> 4u)) & 0x0F0F0F0Fu;
    bits = bits + (bits >> 8u);
    bits = bits + (bits >> 16u);
    return bits & 0x0000003Fu;
}

// Alias for count_ones for potential clarity in some contexts.
fn count_bits(x: u32) -> u32 {
    return count_ones(x);
} 