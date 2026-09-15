//! End to end: a small marian42-style city generated on the GPU, checked for structural
//! invariants and rendered as voxel models plus a street-level map.
//!
//! This is the realistic workload next to the toy fixtures: 52 rotated module variants from
//! connectors (several possibility words per cell), weights, and structure spanning many cells.

mod common;

use std::collections::BTreeMap;
use wfc_core::BoundaryCondition;
use wfc_core::grid::PossibilityGrid;
use wfc_devtools::city::{self, STREET_LEVEL};
use wfc_devtools::render::{self, Style};
use wfc_devtools::{TileGrid, adjacency_violations};

#[tokio::test]
async fn small_city_is_structurally_sound_and_renders() {
    let city = city::city();
    let m = &city.modules;
    let (width, height, depth) = (12, 12, 6);
    let mut initial = PossibilityGrid::new(width, height, depth, m.variants.len());
    city::constrain_city(&mut initial, &city);

    let solved = common::solve_rules(&initial, &m.rules, Some(&m.tileset.weights), BoundaryCondition::Finite, 10).await;
    eprintln!(
        "solved {width}x{height}x{depth} city with {} variants: attempt {}, run {:?}, total {:?}",
        m.variants.len(),
        solved.attempts,
        solved.solve_time,
        solved.total_time
    );
    let grid = TileGrid::from_possibilities(&solved.grid).expect("every cell collapsed to one tile");

    let artifacts = common::artifact_dir();
    let map_palette = city.map_palette();
    let street_style = Style { palette: &map_palette, empty_tiles: &[city.air], cell_px: 12 };
    for (name, image) in [
        ("city_isometric.png", render::render_voxel_isometric(&grid, &city.voxels, 3)),
        ("city_street_level.png", render::render_layer(&grid, 0, &street_style)),
    ] {
        let path = artifacts.join(name);
        image.save(&path).expect("write PNG");
        eprintln!("rendered {}", path.display());
    }

    let violations = adjacency_violations(&grid, &m.rules, BoundaryCondition::Finite);
    assert!(violations.is_empty(), "{} adjacency violations, first: {:?}", violations.len(), violations.first());

    // Whole-column structure. Adjacency alone implies these, so they catch propagation that let a
    // bad state through somewhere the local check above cannot see as a single broken pair.
    let street_level = m.variants_tagged(STREET_LEVEL);
    let buildings = m.variants_tagged("building");
    let roofs = m.variants_tagged("roof");
    let name = |tile: usize| m.prototype_of(tile).name.as_str();
    for y in 0..height {
        for x in 0..width {
            let column: Vec<usize> = (0..depth).map(|z| grid.get(x, y, z)).collect();
            let names: Vec<&str> = column.iter().map(|&t| name(t)).collect();
            assert!(street_level.contains(&column[0]), "({x}, {y}) does not start at street level: {names:?}");
            assert!(column[1..].iter().all(|t| !street_level.contains(t)), "({x}, {y}) street level above z=0: {names:?}");
            assert_eq!(column[depth - 1], city.air, "top layer must be air at ({x}, {y}): {names:?}");

            if buildings.contains(&column[0]) {
                // A building: floors from the street up to exactly one roof, and no building above it.
                let roof = column
                    .iter()
                    .position(|t| roofs.contains(t))
                    .unwrap_or_else(|| panic!("building without a roof at ({x}, {y}): {names:?}"));
                assert!(column[..roof].iter().all(|t| buildings.contains(t)), "({x}, {y}): {names:?}");
                assert!(column[roof + 1..].iter().all(|t| !buildings.contains(t) && !roofs.contains(t)), "({x}, {y}): {names:?}");
            } else {
                assert!(column.iter().all(|t| !buildings.contains(t) && !roofs.contains(t)), "floating building at ({x}, {y}): {names:?}");
            }
            if name(column[0]) == "stair" {
                assert_eq!(name(column[1]), "stair_landing", "({x}, {y}): {names:?}");
            }
        }
    }

    // Local rules guarantee no path ends at a wall or in mid-air, but not that the whole network
    // connects to the street (a global constraint is planned for #7), so this is reported, not
    // asserted.
    let unreachable = city::unreachable_walkable_cells(&grid, &city);
    eprintln!("walkable cells unreachable from the street: {} {unreachable:?}", unreachable.len());

    let mut histogram: BTreeMap<&str, usize> = BTreeMap::new();
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                *histogram.entry(name(grid.get(x, y, z))).or_default() += 1;
            }
        }
    }
    eprintln!("modules used: {histogram:?}");
    for required in ["building_base", "road_straight"] {
        assert!(histogram.contains_key(required), "expected at least one {required}: {histogram:?}");
    }
    assert!(roofs.iter().any(|&t| grid.count(t) > 0), "expected at least one roof: {histogram:?}");
}
