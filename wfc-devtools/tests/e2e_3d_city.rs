//! End to end: a tiny 3D city (ground, roads, buildings) generated on the GPU, checked for
//! structural invariants and rendered as a four-view sheet plus a ground-level map.

mod common;

use wfc_core::BoundaryCondition;
use wfc_core::grid::PossibilityGrid;
use wfc_devtools::fixtures::{self, city};
use wfc_devtools::render::{self, Style};
use wfc_devtools::{TileGrid, adjacency_violations};

#[tokio::test]
async fn small_city_is_structurally_sound_and_renders() {
    let fixture = fixtures::city_3d();
    let (width, height, depth) = (12, 12, 6);
    let mut initial = PossibilityGrid::new(width, height, depth, fixture.names.len());
    fixtures::constrain_city_grid(&mut initial);

    let solved = common::solve(&initial, &fixture, BoundaryCondition::Finite, 5).await;
    let grid = TileGrid::from_possibilities(&solved).expect("every cell collapsed to one tile");

    let style = Style { palette: fixture.palette, empty_tiles: fixture.empty_tiles, cell_px: 12 };
    let artifacts = common::artifact_dir();
    for (name, image) in [
        ("city_3d_four_view.png", render::render_four_view(&grid, &style)),
        ("city_3d_ground_layer.png", render::render_layer(&grid, 0, &style)),
    ] {
        let path = artifacts.join(name);
        image.save(&path).expect("write PNG");
        eprintln!("rendered {}", path.display());
    }

    let violations = adjacency_violations(&grid, &fixture.rules, BoundaryCondition::Finite);
    assert!(
        violations.is_empty(),
        "{} adjacency violations, first: {:?}",
        violations.len(),
        violations.first()
    );

    for y in 0..height {
        for x in 0..width {
            let column: Vec<usize> = (0..depth).map(|z| grid.get(x, y, z)).collect();
            let names: Vec<&str> = column.iter().map(|&t| fixture.names[t]).collect();
            assert!(
                column[0] == city::GROUND || column[0] == city::WALL || city::ROADS.contains(&column[0]),
                "bottom layer must be ground, road or a building at ({x}, {y}): {names:?}"
            );
            assert_eq!(column[depth - 1], city::AIR, "top layer must be air at ({x}, {y}): {names:?}");

            if column[0] == city::WALL {
                // A building: walls from the ground up to exactly one roof, then only air.
                let roof = column
                    .iter()
                    .position(|&t| t == city::ROOF)
                    .unwrap_or_else(|| panic!("building without a roof at ({x}, {y}): {names:?}"));
                assert!(column[..roof].iter().all(|&t| t == city::WALL), "({x}, {y}): {names:?}");
                assert!(column[roof + 1..].iter().all(|&t| t == city::AIR), "({x}, {y}): {names:?}");
            } else {
                // Ground or road: nothing may stand on it.
                assert!(column[1..].iter().all(|&t| t == city::AIR), "({x}, {y}): {names:?}");
            }
        }
    }

    assert!(grid.count(city::WALL) > 0, "expected at least one building");
    assert!(grid.count(city::GROUND) > 0, "expected some open ground");
}
