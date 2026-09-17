//! End to end: a 2D coastline generated through the library, checked against its rules and
//! rendered to a PNG (written to the artifact directory, see `common::artifact_dir`).

mod common;

use wave_forge::{Builder, ChunkCoord, ChunkShape, FocusPoint, Prior, Ruleset, WorldExtent};
use wfc_core::BoundaryCondition;
use wfc_devtools::render::{self, Style};
use wfc_devtools::{TileGrid, adjacency_violations, fixtures};

#[test]
fn coastline_2d_obeys_its_rules_and_renders() {
    let fixture = fixtures::coast_2d();
    let (width, height, cell_px) = (24u32, 16u32, 16u32);
    let ruleset =
        Ruleset::new(&fixture.rules, &fixture.tileset.weights).expect("the fixture compiles");
    let chunk = ChunkShape {
        x: width,
        y: height,
        z: 1,
    };
    let at = ChunkCoord::new(0, 0, 0);

    let mut world = Builder::new(ruleset, Prior::open(fixture.names.len() as u32))
        .seed(5)
        .extent(
            WorldExtent::new(chunk)
                .with_x(0..1)
                .with_y(0..1)
                .with_z(0..1),
        )
        .build()
        .expect("a compute device");
    world.request(&[FocusPoint::new(at, 0)]);
    let events = world.run_until_idle().expect("the solver runs");

    let solved = world
        .chunk(at)
        .unwrap_or_else(|| panic!("the coastline was not generated: {events:?}"));
    let grid = TileGrid::new(
        width as usize,
        height as usize,
        1,
        solved.tiles.iter().map(|&tile| tile as usize).collect(),
    )
    .expect("one tile per cell");

    let violations = adjacency_violations(&grid, &fixture.rules, BoundaryCondition::Finite);
    assert!(
        violations.is_empty(),
        "{} adjacency violations, first: {:?}",
        violations.len(),
        violations.first()
    );

    let style = Style {
        palette: fixture.palette,
        empty_tiles: fixture.empty_tiles,
        cell_px,
    };
    let image = render::render_layer(&grid, 0, &style);
    let path = common::artifact_dir().join("coast_2d.png");
    image.save(&path).expect("write PNG");
    eprintln!("rendered {}", path.display());

    assert_eq!(image.dimensions(), (width * cell_px, height * cell_px));
    // The rendered image must show the solved grid: cell (0, 0) is the bottom-left square.
    let bottom_left = image.get_pixel(cell_px / 2, (height - 1) * cell_px + cell_px / 2);
    assert_eq!(bottom_left.0, fixture.palette[grid.get(0, 0, 0)]);
}
