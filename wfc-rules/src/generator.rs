use crate::types::{TileId, TileSet};
use log::debug;

/// Generates the full adjacency rule vector based on base rules and tile transformations.
///
/// Takes a set of base adjacency rules (typically parsed from a file) and expands them
/// by applying the allowed transformations defined in the `TileSet`.
///
/// The resulting `Vec<bool>` represents the flattened adjacency matrix for *all transformed states*,
/// suitable for creating an `AdjacencyRules` instance.
///
/// # Arguments
/// * `base_rules`: A slice of tuples `(TileId, TileId, usize)`, where each tuple represents
///   an allowed adjacency `tile1 -> tile2` along the specified `axis` in their base orientation.
/// * `tileset`: The `TileSet` containing definitions of allowed transformations for each base tile
///   and mappings between base/transformed states.
/// * `num_axes`: The total number of axes adjacency is defined for (e.g., 6 for 3D).
///
/// # Returns
/// A `Vec<bool>` representing the flattened adjacency matrix for all transformed tile states.
/// The indexing is `axis * num_transformed * num_transformed + ttid1 * num_transformed + ttid2`.
///
/// # Assumptions
/// - The current implementation assumes rotational symmetry (specifically around the Z-axis
///   for transformations like Rot90/180/270) primarily affects rules along axes perpendicular
///   to the rotation axis (e.g., X and Y axes for Z rotation).
/// - It assumes that for a base rule `(base1, base2, axis)` to hold for transformed states
///   `(base1, tform1)` and `(base2, tform2)` along the `transformed_axis = tform1.transform_axis(axis)`,
///   the required transformation `tform2` must be equal to `tform1` (`tform2 == tform1`).
///   This holds for simple rotational symmetry but might need refinement for more complex symmetries
///   or rules involving the axis of rotation itself (e.g., Z-axis rules with Z-axis rotation).
///
pub fn generate_transformed_rules(
    base_rules: &[(TileId, TileId, usize)],
    tileset: &TileSet,
    num_axes: usize,
) -> Vec<bool> {
    let num_transformed_tiles = tileset.num_transformed_tiles();
    let matrix_size = num_axes * num_transformed_tiles * num_transformed_tiles;
    let mut allowed = vec![false; matrix_size];
    debug!(
        "Generating transformed rules: num_base_tiles={}, num_transformed_tiles={}, num_axes={}, matrix_size={}",
        tileset.weights.len(),
        num_transformed_tiles,
        num_axes,
        matrix_size
    );

    for &(base1_id, base2_id, base_axis) in base_rules {
        // Get allowed transformations for the base tiles involved in the rule
        // Use .get(..) to handle potential out-of-bounds TileId from base_rules (though unlikely if validated)
        let transforms1 = tileset
            .allowed_transformations
            .get(base1_id.0)
            .expect("Base rule references TileId out of bounds in TileSet");
        let transforms2 = tileset
            .allowed_transformations
            .get(base2_id.0)
            .expect("Base rule references TileId out of bounds in TileSet");

        // Iterate through all allowed transformations for the first tile
        for &tform1 in transforms1 {
            // Calculate the axis relative to the transformed tile1
            let transformed_axis = tform1.transform_axis(base_axis);

            // Determine the required transformation for tile2 for the rule to hold
            // Assumption: For rotational symmetry, tform2 must match tform1.
            let required_tform2 = tform1;

            // Check if this required transformation is allowed for the second tile
            if transforms2.contains(&required_tform2) {
                // Get the unique IDs for the transformed tile states
                let ttid1 = tileset
                    .get_transformed_id(base1_id, tform1)
                    .expect("Failed to get transformed ID for allowed transform (tile1)");
                let ttid2 = tileset
                    .get_transformed_id(base2_id, required_tform2)
                    .expect("Failed to get transformed ID for allowed transform (tile2)");

                // Calculate the index in the flattened adjacency matrix
                let flat_index = transformed_axis * num_transformed_tiles * num_transformed_tiles
                    + ttid1 * num_transformed_tiles
                    + ttid2;

                if flat_index < allowed.len() {
                    if !allowed[flat_index] {
                        // Avoid redundant debug messages
                        debug!(
                            "Rule derived: ({:?}, {:?}) + ({:?}, {:?}) along axis {} (orig base axis {}) -> Set allowed[{}] = true",
                            base1_id, tform1, base2_id, required_tform2, transformed_axis, base_axis, flat_index
                        );
                    }
                    allowed[flat_index] = true;
                } else {
                    // This should not happen if indices are calculated correctly
                    panic!(
                        "Internal error: Calculated flat index {} out of bounds (size {})",
                        flat_index,
                        allowed.len()
                    );
                }
            }
        }
    }
    debug!(
        "Generated rule vector with {} true entries.",
        allowed.iter().filter(|&&x| x).count()
    );
    allowed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{TileId, TileSet, Transformation};

    // Helper to create a simple TileSet for testing
    fn create_test_tileset(weights: Vec<f32>, transforms: Vec<Vec<Transformation>>) -> TileSet {
        TileSet::new(weights, transforms).expect("Failed to create test tileset")
    }

    #[test]
    fn generator_identity_only() {
        // Tile 0: Identity
        // Tile 1: Identity
        let tileset = create_test_tileset(
            vec![1.0, 1.0],
            vec![
                vec![Transformation::Identity],
                vec![Transformation::Identity],
            ],
        );
        // Base Rule: T0 -> T1 along Axis 0 (+X)
        let base_rules = vec![(TileId(0), TileId(1), 0)];
        let num_axes = 6;

        let allowed = generate_transformed_rules(&base_rules, &tileset, num_axes);

        let num_transformed = tileset.num_transformed_tiles();
        assert_eq!(num_transformed, 2); // T0_Id, T1_Id

        let ttid0_id = tileset
            .get_transformed_id(TileId(0), Transformation::Identity)
            .unwrap();
        let ttid1_id = tileset
            .get_transformed_id(TileId(1), Transformation::Identity)
            .unwrap();

        // Expect only the base rule to be true for the Identity transforms
        let expected_index =
            0 * num_transformed * num_transformed + ttid0_id * num_transformed + ttid1_id;
        assert!(allowed[expected_index]);
        assert_eq!(allowed.iter().filter(|&&x| x).count(), 1); // Only 1 rule should be true
    }

    #[test]
    fn generator_simple_rotation() {
        // Tile 0 ("Corner"): Rot90 allowed
        // Tile 1 ("Straight"): Identity only
        let tileset = create_test_tileset(
            vec![1.0, 1.0],
            vec![
                vec![Transformation::Identity, Transformation::Rot90],
                vec![Transformation::Identity],
            ],
        );
        // Base Rule: T0(Id) -> T1(Id) along Axis 0 (+X)
        let base_rules = vec![(TileId(0), TileId(1), 0)];
        let num_axes = 6;

        let allowed = generate_transformed_rules(&base_rules, &tileset, num_axes);

        let num_transformed = tileset.num_transformed_tiles();
        assert_eq!(num_transformed, 3); // T0_Id, T0_R90, T1_Id

        let ttid0_id = tileset
            .get_transformed_id(TileId(0), Transformation::Identity)
            .unwrap();
        let ttid1_id = tileset
            .get_transformed_id(TileId(1), Transformation::Identity)
            .unwrap();

        // Expected rules:
        // 1. T0(Id) -> T1(Id) along Axis 0 (+X) (Base rule)
        let idx1 = 0 * num_transformed * num_transformed + ttid0_id * num_transformed + ttid1_id;
        assert!(allowed[idx1], "Base rule T0(Id)->T1(Id) Axis 0 not found");

        // 2. T0(R90) -> T1(R90) - But T1 doesn't allow R90, so this rule shouldn't generate.
        // Our generator requires T1 to also have R90, which it doesn't.

        // Let's re-run with T1 also allowing Rot90
        let tileset_rot = create_test_tileset(
            vec![1.0, 1.0],
            vec![
                vec![Transformation::Identity, Transformation::Rot90],
                vec![Transformation::Identity, Transformation::Rot90], // T1 also Rot90
            ],
        );
        let allowed_rot = generate_transformed_rules(&base_rules, &tileset_rot, num_axes);
        let num_transformed_rot = tileset_rot.num_transformed_tiles();
        assert_eq!(num_transformed_rot, 4); // T0_Id, T0_R90, T1_Id, T1_R90

        let ttid0_id_rot = tileset_rot
            .get_transformed_id(TileId(0), Transformation::Identity)
            .unwrap();
        let ttid0_r90_rot = tileset_rot
            .get_transformed_id(TileId(0), Transformation::Rot90)
            .unwrap();
        let ttid1_id_rot = tileset_rot
            .get_transformed_id(TileId(1), Transformation::Identity)
            .unwrap();
        let ttid1_r90_rot = tileset_rot
            .get_transformed_id(TileId(1), Transformation::Rot90)
            .unwrap();

        // Expected rules (Rotational):
        // 1. T0(Id) -> T1(Id) along Axis 0 (+X) (Base rule)
        let idx1_rot = 0 * num_transformed_rot * num_transformed_rot
            + ttid0_id_rot * num_transformed_rot
            + ttid1_id_rot;
        assert!(
            allowed_rot[idx1_rot],
            "Rot: Base rule T0(Id)->T1(Id) Axis 0 not found"
        );

        // 2. T0(R90) -> T1(R90) along Axis 2 (+Y) (Transformed rule: Tform1=R90, Axis=0 -> NewAxis=2)
        let idx2_rot = 2 * num_transformed_rot * num_transformed_rot
            + ttid0_r90_rot * num_transformed_rot
            + ttid1_r90_rot;
        assert!(
            allowed_rot[idx2_rot],
            "Rot: Transformed rule T0(R90)->T1(R90) Axis 2 not found"
        );

        assert_eq!(allowed_rot.iter().filter(|&&x| x).count(), 2); // Should have 2 true entries
    }

    #[test]
    fn generator_multiple_rules_rotations() {
        let tileset = create_test_tileset(
            vec![1.0, 1.0], // T0, T1
            vec![
                vec![
                    Transformation::Identity,
                    Transformation::Rot90,
                    Transformation::Rot180,
                    Transformation::Rot270,
                ],
                vec![
                    Transformation::Identity,
                    Transformation::Rot90,
                    Transformation::Rot180,
                    Transformation::Rot270,
                ],
            ],
        );
        // Base Rules:
        // T0(Id) -> T1(Id) along Axis 0 (+X)
        // T1(Id) -> T0(Id) along Axis 1 (-X)
        let base_rules = vec![(TileId(0), TileId(1), 0), (TileId(1), TileId(0), 1)];
        let num_axes = 6;

        let allowed = generate_transformed_rules(&base_rules, &tileset, num_axes);
        let num_transformed = tileset.num_transformed_tiles();
        assert_eq!(num_transformed, 8); // 2 tiles * 4 transforms

        // Each base rule should generate 4 transformed rules (one for each transform of tile1)
        // Total expected rules = 2 base rules * 4 transforms = 8
        assert_eq!(allowed.iter().filter(|&&x| x).count(), 8);

        // Spot check a few generated rules:
        // Base rule 1: T0(Id) -> T1(Id) Axis 0
        let ttid0_id = tileset
            .get_transformed_id(TileId(0), Transformation::Identity)
            .unwrap();
        let ttid1_id = tileset
            .get_transformed_id(TileId(1), Transformation::Identity)
            .unwrap();
        let idx_base1 =
            0 * num_transformed * num_transformed + ttid0_id * num_transformed + ttid1_id;
        assert!(allowed[idx_base1]);

        // Transformed from rule 1: T0(R270) -> T1(R270) Axis 3 (-Y) (Tform=R270, Axis=0 -> NewAxis=3)
        let ttid0_r270 = tileset
            .get_transformed_id(TileId(0), Transformation::Rot270)
            .unwrap();
        let ttid1_r270 = tileset
            .get_transformed_id(TileId(1), Transformation::Rot270)
            .unwrap();
        let idx_trans1 =
            3 * num_transformed * num_transformed + ttid0_r270 * num_transformed + ttid1_r270;
        assert!(allowed[idx_trans1]);

        // Base rule 2: T1(Id) -> T0(Id) Axis 1
        let idx_base2 =
            1 * num_transformed * num_transformed + ttid1_id * num_transformed + ttid0_id;
        assert!(allowed[idx_base2]);

        // Transformed from rule 2: T1(R180) -> T0(R180) Axis 0 (+X) (Tform=R180, Axis=1 -> NewAxis=0)
        let ttid0_r180 = tileset
            .get_transformed_id(TileId(0), Transformation::Rot180)
            .unwrap();
        let ttid1_r180 = tileset
            .get_transformed_id(TileId(1), Transformation::Rot180)
            .unwrap();
        let idx_trans2 =
            0 * num_transformed * num_transformed + ttid1_r180 * num_transformed + ttid0_r180;
        assert!(allowed[idx_trans2]);
    }
}
