use crate::types::{TileId, TileSet};
use log::debug;

/// Generates the full adjacency rule list based on base rules and tile transformations.
///
/// Takes a set of base adjacency rules and expands them by applying allowed transformations.
///
/// # Returns
/// A `Vec<(usize, usize, usize)>` where each tuple `(axis, ttid1, ttid2)` represents
/// an allowed adjacency between transformed tiles.
///
/// # Arguments
/// * `base_rules`: A slice of tuples `(TileId, TileId, usize)`, where each tuple represents
///   an allowed adjacency `tile1 -> tile2` along the specified `axis` in their base orientation.
/// * `tileset`: The `TileSet` containing definitions of allowed transformations for each base tile
///   and mappings between base/transformed states.
/// * `num_axes`: The total number of axes adjacency is defined for (e.g., 6 for 3D).
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
#[must_use] pub fn generate_transformed_rules(
    base_rules: &[(TileId, TileId, usize)],
    tileset: &TileSet,
    num_axes: usize,
) -> Vec<(usize, usize, usize)> {
    let num_transformed_tiles = tileset.num_transformed_tiles();
    let mut allowed_tuples_set = std::collections::HashSet::new();

    debug!(
        "Generating transformed rules: num_base_tiles={}, num_transformed_tiles={}, num_axes={}",
        tileset.weights.len(),
        num_transformed_tiles,
        num_axes
    );

    for &(base1_id, base2_id, base_axis) in base_rules {
        if base_axis >= num_axes {
            log::warn!(
                "Skipping base rule {base1_id:?} -> {base2_id:?} along invalid axis {base_axis}"
            );
            continue;
        }

        let transforms1 = tileset
            .allowed_transformations
            .get(base1_id.0)
            .expect("Base rule references TileId out of bounds in TileSet");
        let transforms2 = tileset
            .allowed_transformations
            .get(base2_id.0)
            .expect("Base rule references TileId out of bounds in TileSet");

        for &tform1 in transforms1 {
            let transformed_axis = tform1.transform_axis(base_axis);
            let required_tform2 = tform1;

            if transforms2.contains(&required_tform2) {
                let ttid1 = tileset
                    .get_transformed_id(base1_id, tform1)
                    .expect("Failed to get transformed ID (tile1)");
                let ttid2 = tileset
                    .get_transformed_id(base2_id, required_tform2)
                    .expect("Failed to get transformed ID (tile2)");

                if transformed_axis >= num_axes {
                    log::error!("Derived rule {ttid1:?} -> {ttid2:?} resulted in invalid transformed axis {transformed_axis} from base axis {base_axis} and transform {tform1:?}");
                    continue;
                }

                debug!(
                    "Rule derived: ({base1_id:?}, {tform1:?}) + ({base2_id:?}, {required_tform2:?}) along axis {transformed_axis} (orig base axis {base_axis}) -> Add tuple ({transformed_axis}, {ttid1}, {ttid2})"
                );
                allowed_tuples_set.insert((transformed_axis, ttid1, ttid2));
            }
        }
    }
    let allowed_tuples: Vec<(usize, usize, usize)> = allowed_tuples_set.into_iter().collect();
    debug!(
        "Generated {} unique allowed rule tuples.",
        allowed_tuples.len()
    );
    allowed_tuples
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

        let allowed_tuples = generate_transformed_rules(&base_rules, &tileset, num_axes);

        let num_transformed = tileset.num_transformed_tiles();
        assert_eq!(num_transformed, 2); // T0_Id, T1_Id

        let ttid0_id = tileset
            .get_transformed_id(TileId(0), Transformation::Identity)
            .unwrap();
        let ttid1_id = tileset
            .get_transformed_id(TileId(1), Transformation::Identity)
            .unwrap();

        assert_eq!(allowed_tuples.len(), 1);
        assert!(allowed_tuples.contains(&(0, ttid0_id, ttid1_id)));
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

        let allowed_tuples = generate_transformed_rules(&base_rules, &tileset, num_axes);

        let num_transformed = tileset.num_transformed_tiles();
        assert_eq!(num_transformed, 3); // T0_Id, T0_R90, T1_Id

        let ttid0_id = tileset
            .get_transformed_id(TileId(0), Transformation::Identity)
            .unwrap();
        let ttid1_id = tileset
            .get_transformed_id(TileId(1), Transformation::Identity)
            .unwrap();

        let idx1 = (0, ttid0_id, ttid1_id); // Axis 0, T0_Id -> T1_Id
        assert!(allowed_tuples.contains(&idx1), "Base rule tuple not found");
        assert_eq!(allowed_tuples.len(), 1);

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

        let idx1_rot = (0, ttid0_id_rot, ttid1_id_rot); // Axis 0, T0_Id -> T1_Id
        assert!(
            allowed_rot.contains(&idx1_rot),
            "Rot: Base rule tuple not found"
        );
        let idx2_rot = (2, ttid0_r90_rot, ttid1_r90_rot); // Axis 2, T0_R90 -> T1_R90
        assert!(
            allowed_rot.contains(&idx2_rot),
            "Rot: Transformed rule tuple not found"
        );
        assert_eq!(allowed_rot.len(), 2);
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

        let allowed_tuples = generate_transformed_rules(&base_rules, &tileset, num_axes);
        let num_transformed = tileset.num_transformed_tiles();
        assert_eq!(num_transformed, 8); // 2 tiles * 4 transforms

        assert_eq!(allowed_tuples.len(), 8);

        let ttid0_id = tileset
            .get_transformed_id(TileId(0), Transformation::Identity)
            .unwrap();
        let ttid1_id = tileset
            .get_transformed_id(TileId(1), Transformation::Identity)
            .unwrap();

        let idx_base1 = (0, ttid0_id, ttid1_id);
        assert!(allowed_tuples.contains(&idx_base1));

        let ttid0_r270 = tileset
            .get_transformed_id(TileId(0), Transformation::Rot270)
            .unwrap();
        let ttid1_r270 = tileset
            .get_transformed_id(TileId(1), Transformation::Rot270)
            .unwrap();
        let idx_trans1 = (3, ttid0_r270, ttid1_r270);
        assert!(allowed_tuples.contains(&idx_trans1));

        let idx_base2 = (1, ttid1_id, ttid0_id);
        assert!(allowed_tuples.contains(&idx_base2));

        let ttid0_r180 = tileset
            .get_transformed_id(TileId(0), Transformation::Rot180)
            .unwrap();
        let ttid1_r180 = tileset
            .get_transformed_id(TileId(1), Transformation::Rot180)
            .unwrap();
        let idx_trans2 = (0, ttid1_r180, ttid0_r180);
        assert!(allowed_tuples.contains(&idx_trans2));
    }

    #[test]
    fn generator_with_flips() {
        // Tile 0: Identity, FlipX
        // Tile 1: Identity, FlipX
        let tileset = create_test_tileset(
            vec![1.0, 1.0],
            vec![
                vec![Transformation::Identity, Transformation::FlipX],
                vec![Transformation::Identity, Transformation::FlipX],
            ],
        );
        let num_transformed = tileset.num_transformed_tiles();
        assert_eq!(num_transformed, 4); // T0_Id, T0_FX, T1_Id, T1_FX

        // Base Rule: T0(Id) -> T1(Id) along Axis 0 (+X)
        let base_rules = vec![(TileId(0), TileId(1), 0)];
        let num_axes = 6;

        let allowed_tuples = generate_transformed_rules(&base_rules, &tileset, num_axes);

        let ttid0_id = tileset
            .get_transformed_id(TileId(0), Transformation::Identity)
            .unwrap();
        let ttid0_fx = tileset
            .get_transformed_id(TileId(0), Transformation::FlipX)
            .unwrap();
        let ttid1_id = tileset
            .get_transformed_id(TileId(1), Transformation::Identity)
            .unwrap();
        let ttid1_fx = tileset
            .get_transformed_id(TileId(1), Transformation::FlipX)
            .unwrap();

        // Expected rules based on tform2 == tform1 simplification:
        // 1. T0_Id -> T1_Id along Axis 0 (+X) (from tform1 = Identity)
        //    transformed_axis = Identity.transform_axis(0) = 0
        let expected1 = (0, ttid0_id, ttid1_id);
        assert!(
            allowed_tuples.contains(&expected1),
            "Flip test missing rule 1"
        );

        // 2. T0_FX -> T1_FX along Axis 1 (-X) (from tform1 = FlipX)
        //    transformed_axis = FlipX.transform_axis(0) = 1
        let expected2 = (1, ttid0_fx, ttid1_fx);
        assert!(
            allowed_tuples.contains(&expected2),
            "Flip test missing rule 2"
        );

        assert_eq!(
            allowed_tuples.len(),
            2,
            "Flip test generated unexpected number of rules"
        );

        // --- Test with a rule along an axis unaffected by FlipX (e.g., Axis 2, +Y) ---
        // Base Rule: T0(Id) -> T1(Id) along Axis 2 (+Y)
        let base_rules_y = vec![(TileId(0), TileId(1), 2)];
        let allowed_tuples_y = generate_transformed_rules(&base_rules_y, &tileset, num_axes);

        // Expected rules:
        // 1. T0_Id -> T1_Id along Axis 2 (+Y) (tform1 = Identity, transformed_axis = Identity.transform_axis(2) = 2)
        let expected_y1 = (2, ttid0_id, ttid1_id);
        assert!(
            allowed_tuples_y.contains(&expected_y1),
            "Flip+Y test missing rule 1"
        );
        // 2. T0_FX -> T1_FX along Axis 2 (+Y) (tform1 = FlipX, transformed_axis = FlipX.transform_axis(2) = 2)
        let expected_y2 = (2, ttid0_fx, ttid1_fx);
        assert!(
            allowed_tuples_y.contains(&expected_y2),
            "Flip+Y test missing rule 2"
        );

        assert_eq!(
            allowed_tuples_y.len(),
            2,
            "Flip+Y test generated unexpected number of rules"
        );
    }
}
