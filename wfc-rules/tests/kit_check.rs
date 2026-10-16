//! A model drawn for a module, checked against the contracts of the module's faces: it stays in its
//! cell, a stair rising into the cell above; it has a walking surface at the height every walkable
//! side's neighbours walk at; and it keeps clear of a side that faces air, a roof's foot excepted.

use wfc_rules::formats::module_format::module_faces;
use wfc_rules::kit_check::{ModelProblem, check_model};

const SET: &str = r#"(
    faces: {
        "air": Side(connector: "air"),
        "ground": Side(connector: "ground", walkable: true),
        "road": Side(connector: "road", walkable: true),
        "blocked": Side(connector: "ground"),
        "wall": Side(connector: "building"),
        "open": Top(connector: "open"),
        "solid": Top(connector: "solid"),
        "bedrock": Top(connector: "bedrock"),
        "stair": Top(connector: "stair", rotation: Some(0)),
    },
    connect: [("building", "air"), ("building", "ground")],
    modules: [
        (name: "grass", sides: ["ground", "ground", "ground", "ground"], up: "open", down: "bedrock"),
        (name: "lane", sides: ["road", "road", "ground", "ground"], up: "open", down: "bedrock"),
        (name: "roof", sides: ["air", "air", "air", "air"], up: "open", down: "solid"),
        (name: "lamp", sides: ["air", "air", "air", "air"], up: "open", down: "open"),
        (name: "stairs", sides: ["blocked", "ground", "wall", "wall"], up: "stair", down: "bedrock"),
        (name: "upper_stairs", sides: ["air", "ground", "wall", "wall"], up: "stair", down: "open"),
    ],
)"#;

/// The triangles of a box from `low` to `high`.
fn cuboid(low: [f32; 3], high: [f32; 3]) -> Vec<[f32; 3]> {
    let corner = |i: usize| {
        [
            if i & 1 == 0 { low[0] } else { high[0] },
            if i & 2 == 0 { low[1] } else { high[1] },
            if i & 4 == 0 { low[2] } else { high[2] },
        ]
    };
    let quads = [
        [0, 1, 3, 2],
        [4, 6, 7, 5],
        [0, 4, 5, 1],
        [2, 3, 7, 6],
        [0, 2, 6, 4],
        [1, 5, 7, 3],
    ];
    quads
        .iter()
        .flat_map(|q| [q[0], q[1], q[2], q[0], q[2], q[3]])
        .map(corner)
        .collect()
}

/// The ground slab every ground tile stands on: the cell's lowest quarter.
fn slab() -> Vec<[f32; 3]> {
    cuboid([-0.5, -0.5, -0.5], [0.5, -0.25, 0.5])
}

fn problems(module: &str, triangles: &[[f32; 3]]) -> Vec<ModelProblem> {
    let faces = module_faces(SET).expect("the set parses");
    let faces = faces
        .iter()
        .find(|faces| faces.name == module)
        .expect("the module is in the set");
    check_model(faces, triangles)
}

#[test]
fn a_ground_tile_on_its_slab_meets_its_contracts() {
    let mut model = slab();
    model.extend(cuboid([-0.1, -0.25, -0.1], [0.1, 0.3, 0.1]));

    assert_eq!(problems("grass", &model), Vec::new());
}

#[test]
fn a_ground_tile_walked_on_at_the_wrong_height_is_refused_on_every_side() {
    let model = cuboid([-0.5, -0.5, -0.5], [0.5, -0.35, 0.5]);

    let found = problems("grass", &model);

    assert_eq!(found.len(), 4, "{found:?}");
    assert!(
        found
            .iter()
            .all(|problem| matches!(problem, ModelProblem::NoWalkingSurface { .. })),
        "{found:?}"
    );
}

#[test]
fn a_road_side_without_its_surface_is_refused() {
    let mut model = cuboid([-0.5, -0.5, -0.5], [0.5, -0.25, -0.3]);
    model.extend(cuboid([-0.5, -0.5, 0.3], [0.5, -0.25, 0.5]));

    let found = problems("lane", &model);

    let sides: Vec<&str> = found
        .iter()
        .filter_map(|problem| match problem {
            ModelProblem::NoWalkingSurface { side, .. } => Some(side.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(sides, ["+X", "-X"], "{found:?}");
}

#[test]
fn a_model_reaching_out_of_its_cell_is_refused() {
    let mut model = slab();
    model.extend(cuboid([-0.1, -0.25, -0.1], [0.1, 0.6, 0.1]));

    let found = problems("grass", &model);

    assert!(
        found
            .iter()
            .any(|problem| matches!(problem, ModelProblem::OutsideCell { .. })),
        "{found:?}"
    );
}

#[test]
fn a_model_at_another_scale_is_refused() {
    let model = cuboid([-1.0, -1.0, -1.0], [1.0, -0.5, 1.0]);

    let found = problems("grass", &model);

    assert!(
        found
            .iter()
            .any(|problem| matches!(problem, ModelProblem::OutsideCell { .. })),
        "{found:?}"
    );
}

#[test]
fn a_stair_may_rise_into_the_cell_above() {
    let mut model = slab();
    model.extend(cuboid([-0.5, -0.5, -0.5], [0.5, 0.5, -0.4]));
    model.extend(cuboid([-0.25, -0.25, -0.25], [0.5, 0.75, 0.25]));

    assert_eq!(problems("stairs", &model), Vec::new());
}

#[test]
fn a_side_that_is_not_walkable_needs_no_walking_surface() {
    // The slab stops where the flight's solid block begins, so the blocked +X side has no
    // surface at the walking height.
    let mut model = cuboid([-0.5, -0.5, -0.5], [0.2, -0.25, 0.5]);
    model.extend(cuboid([0.2, -0.5, -0.5], [0.5, 0.75, 0.5]));

    assert_eq!(problems("stairs", &model), Vec::new());
}

#[test]
fn a_stairs_flight_may_reach_its_plus_x_side_where_it_climbs_on() {
    let mut model = cuboid([-0.5, -0.5, -0.5], [0.0, -0.25, 0.5]);
    model.extend(cuboid([-0.25, -0.25, -0.25], [0.5, 0.75, 0.25]));

    assert_eq!(problems("upper_stairs", &model), Vec::new());
}

#[test]
fn something_touching_a_side_that_faces_air_is_refused() {
    let model = cuboid([-0.1, -0.5, -0.1], [0.5, 0.3, 0.1]);

    let found = problems("lamp", &model);

    assert!(
        matches!(found.as_slice(), [ModelProblem::TouchesAir { side, .. }] if side == "+X"),
        "{found:?}"
    );
}

#[test]
fn a_roof_may_cover_its_whole_footprint_at_its_foot() {
    let mut model = cuboid([-0.5, -0.5, -0.5], [0.5, -0.25, 0.5]);
    model.extend(cuboid([-0.3, -0.25, -0.3], [0.3, 0.3, 0.3]));

    assert_eq!(problems("roof", &model), Vec::new());
}

#[test]
fn a_roof_reaching_its_side_above_its_foot_is_refused() {
    let mut model = cuboid([-0.5, -0.5, -0.5], [0.5, -0.25, 0.5]);
    model.extend(cuboid([-0.3, -0.25, -0.3], [0.5, 0.3, 0.3]));

    let found = problems("roof", &model);

    assert!(
        matches!(found.as_slice(), [ModelProblem::TouchesAir { side, .. }] if side == "+X"),
        "{found:?}"
    );
}

#[test]
fn an_empty_model_is_refused() {
    assert_eq!(problems("lamp", &[]), vec![ModelProblem::Empty]);
}
