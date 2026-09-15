//! End to end: a 2D coastline generated on the GPU, checked against its rules and rendered to a
//! PNG (written to the artifact directory, see `common::artifact_dir`).

mod common;

use wfc_core::BoundaryCondition;
use wfc_core::grid::PossibilityGrid;
use wfc_devtools::render::{self, Style};
use wfc_devtools::{TileGrid, adjacency_violations, fixtures};

#[tokio::test]
async fn coastline_2d_obeys_its_rules_and_renders() {
    let fixture = fixtures::coast_2d();
    let (width, height, cell_px) = (24, 16, 16);
    let initial = PossibilityGrid::new(width, height, 1, fixture.names.len());

    let solved = common::solve(&initial, &fixture, BoundaryCondition::Finite, 5).await;
    let grid = TileGrid::from_possibilities(&solved).expect("every cell collapsed to one tile");

    let violations = adjacency_violations(&grid, &fixture.rules, BoundaryCondition::Finite);
    assert!(
        violations.is_empty(),
        "{} adjacency violations, first: {:?}",
        violations.len(),
        violations.first()
    );

    let style = Style { palette: fixture.palette, empty_tiles: fixture.empty_tiles, cell_px };
    let image = render::render_layer(&grid, 0, &style);
    let path = common::artifact_dir().join("coast_2d.png");
    image.save(&path).expect("write PNG");
    eprintln!("rendered {}", path.display());

    assert_eq!(image.dimensions(), (width as u32 * cell_px, height as u32 * cell_px));
    // The rendered image must show the solved grid: cell (0, 0) is the bottom-left square.
    let bottom_left = image.get_pixel(cell_px / 2, (height as u32 - 1) * cell_px + cell_px / 2);
    assert_eq!(bottom_left.0, fixture.palette[grid.get(0, 0, 0)]);
}
