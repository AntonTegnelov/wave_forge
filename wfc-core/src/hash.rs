//! The random choices a solve makes, as a function of where and when it makes them.
//!
//! Nothing keeps generator state: a choice is a hash of `(seed, chunk, attempt, step)`. That is what
//! lets a GPU workgroup, a CPU thread and a second machine reach the same result, and lets an
//! evicted chunk regenerate identically. The GPU kernel implements these two functions in WGSL;
//! keep the two in step.

use crate::rules::TileMask;

/// Jarzynski and Olano's pcg3d: three 32-bit words in, three hashed words out.
#[must_use]
pub fn pcg3d(input: [u32; 3]) -> [u32; 3] {
    let mut v = input.map(|word| word.wrapping_mul(1_664_525).wrapping_add(1_013_904_223));
    v[0] = v[0].wrapping_add(v[1].wrapping_mul(v[2]));
    v[1] = v[1].wrapping_add(v[2].wrapping_mul(v[0]));
    v[2] = v[2].wrapping_add(v[0].wrapping_mul(v[1]));
    v = v.map(|word| word ^ (word >> 16));
    v[0] = v[0].wrapping_add(v[1].wrapping_mul(v[2]));
    v[1] = v[1].wrapping_add(v[2].wrapping_mul(v[0]));
    v[2] = v[2].wrapping_add(v[0].wrapping_mul(v[1]));
    v
}

/// The golden-ratio odd constant that mixes a chunk's identity into a seed.
pub const CHUNK_SALT: u32 = 0x9E37_79B9;

/// The hash for the choice a region makes at `step` of attempt `tries`.
#[must_use]
pub fn choice_hash(seed: u32, chunk_id: u32, tries: u32, step: u32) -> u32 {
    pcg3d([seed ^ chunk_id.wrapping_mul(CHUNK_SALT), tries, step])[0]
}

/// The tile `hash` picks out of `mask`, in proportion to `weights`.
///
/// Integer arithmetic only: a float sum may be contracted into a fused multiply-add by one driver
/// and not another, and the same seed would then pick differently on different hardware.
#[must_use]
pub fn choose_tile(weights: &[u32], mask: TileMask, hash: u32) -> Option<u32> {
    let total: u64 = mask
        .iter()
        .map(|tile| u64::from(weights[tile as usize]))
        .sum();
    if total == 0 {
        // Every remaining tile has weight zero, which a valid rule set does not produce; fall back
        // to the lowest so a solve cannot stall on it.
        return mask.iter().next();
    }
    let mut pick = u64::from(hash) % total;
    for tile in mask.iter() {
        let weight = u64::from(weights[tile as usize]);
        if pick < weight {
            return Some(tile);
        }
        pick -= weight;
    }
    unreachable!("the walk covers the whole weight of the mask")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashing_is_stable_and_separates_neighbouring_inputs() {
        assert_eq!(pcg3d([1, 2, 3]), pcg3d([1, 2, 3]));
        let a = choice_hash(7, 11, 0, 0);
        assert_ne!(
            a,
            choice_hash(7, 11, 0, 1),
            "a later step chooses differently"
        );
        assert_ne!(a, choice_hash(7, 11, 1, 0), "a retry chooses differently");
        assert_ne!(
            a,
            choice_hash(7, 12, 0, 0),
            "another chunk chooses differently"
        );
        assert_ne!(
            a,
            choice_hash(8, 11, 0, 0),
            "another world seed chooses differently"
        );
    }

    #[test]
    fn a_choice_falls_in_the_mask_and_follows_the_weights() {
        let weights = [1, 99, 0];
        let mask = TileMask::all(3);

        let picks: Vec<u32> = (0..100)
            .filter_map(|step| choose_tile(&weights, mask, choice_hash(1, 1, 0, step)))
            .collect();

        assert!(
            picks.iter().all(|tile| *tile < 2),
            "a zero weight is never picked"
        );
        let heavy = picks.iter().filter(|tile| **tile == 1).count();
        assert!(heavy > 90, "the heavy tile dominates: {heavy} of 100");
    }

    #[test]
    fn a_single_tile_mask_picks_it_whatever_the_hash() {
        assert_eq!(choose_tile(&[5, 5], TileMask::single(1), 12345), Some(1));
        assert_eq!(choose_tile(&[0, 0], TileMask::single(0), 7), Some(0));
        assert_eq!(choose_tile(&[1, 1], TileMask::EMPTY, 7), None);
    }
}
