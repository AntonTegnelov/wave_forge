//! The maximal preset's eight cultures (docs/product/user-stories.md, M1, #247): each a WFC module
//! set of at least 60 modules that solves a town standing on the street, roofed and open to the sky,
//! with its walkable cells in one network about as often as the city's, whose local rules for
//! walking they share. Local rules cannot forbid a network cut off as a whole, so the share of
//! walkable cells in the largest network is compared with the city's on the same town, not with one.

use std::collections::BTreeMap;
use wave_forge::towns::{Selector, town_prior};
use wave_forge::{Builder, ChunkCoord, ChunkShape, FocusPoint, Ruleset, WorldExtent};
use wfc_devtools::city;
use wfc_devtools::{BoundaryCondition, TileGrid, adjacency_violations};
use wfc_rules::loader::{RuleFile, parse_rule_file};
use wfc_rules::modules::CompiledModules;

const CULTURES: [&str; 8] = [
    "coastfolk",
    "steppe_riders",
    "woodlanders",
    "sand_dwellers",
    "highlanders",
    "marsh_folk",
    "jungle_folk",
    "frostfolk",
];
const SIDE: u32 = 8;
const DEPTH: u32 = 6;
/// Chunks along each side of a town.
const CHUNKS: u32 = 3;

/// A town of the module set at `path` (from the repository's root), solved on the GPU as a bounded
/// world of 3 by 3 chunks of 8 by 8 by 6 cells, the size a town's chunk takes, with its module set.
fn town(path: &str) -> (TileGrid, CompiledModules) {
    let text = std::fs::read_to_string(format!("{}/../{path}", env!("CARGO_MANIFEST_DIR")))
        .expect("the module set");
    let file = parse_rule_file(&text).expect("a valid module set");
    let RuleFile::Modules(modules) = &file else {
        panic!("{path} is a module set");
    };
    let prior = town_prior(
        &file,
        DEPTH,
        Some(&Selector::Tagged("street_level".to_owned())),
        Some(&Selector::Named("air".to_owned())),
    )
    .expect("street-level and air tiles");
    let chunk = ChunkShape {
        x: SIDE,
        y: SIDE,
        z: DEPTH,
    };
    let mut world = Builder::new(Ruleset::from_modules(modules).expect("it compiles"), prior)
        .seed(5)
        .extent(
            WorldExtent::new(chunk)
                .with_x(0..CHUNKS as i32)
                .with_y(0..CHUNKS as i32)
                .with_z(0..1),
        )
        .build()
        .expect("a compute device");
    world.request(&[FocusPoint::new(ChunkCoord::new(0, 0, 0), CHUNKS)]);
    let events = world.run_until_idle().expect("the solver runs");
    let (side, depth, chunk_side) = ((SIDE * CHUNKS) as usize, DEPTH as usize, SIDE as usize);
    let mut tiles = vec![0; side * side * depth];
    for cy in 0..CHUNKS as usize {
        for cx in 0..CHUNKS as usize {
            let solved = world
                .chunk(ChunkCoord::new(cx as i32, cy as i32, 0))
                .unwrap_or_else(|| panic!("{path}'s town was not solved: {events:?}"));
            for z in 0..depth {
                for y in 0..chunk_side {
                    for x in 0..chunk_side {
                        let (gx, gy) = (cx * chunk_side + x, cy * chunk_side + y);
                        tiles[(z * side + gy) * side + gx] =
                            usize::from(solved.tiles[(z * chunk_side + y) * chunk_side + x]);
                    }
                }
            }
        }
    }
    let grid = TileGrid::new(side, side, depth, tiles).expect("one tile per cell");
    (grid, modules.clone())
}

/// How many cells of a town are walkable, and the share of them in its largest walkable network.
fn walkable_share(grid: &TileGrid, m: &CompiledModules) -> (usize, f64) {
    let walkable = city::walkable_tiles(m);
    let cells = (0..grid.depth)
        .flat_map(|z| (0..grid.height).flat_map(move |y| (0..grid.width).map(move |x| (x, y, z))))
        .filter(|&(x, y, z)| walkable.contains(&grid.get(x, y, z)))
        .count();
    let cut_off = city::disconnected_walkable_cells(grid, m).len();
    (cells, 1.0 - cut_off as f64 / cells.max(1) as f64)
}

#[test]
fn every_culture_is_a_set_of_sixty_modules_that_solves_a_standing_walkable_town() {
    let (city_grid, city_modules) = town("examples/city.ron");
    let city = walkable_share(&city_grid, &city_modules);

    let mut report = BTreeMap::new();
    for culture in CULTURES {
        let (grid, m) = town(&format!("examples/continent/cultures/{culture}.ron"));
        let street = m.variants_tagged("street_level");
        let air = m.variants_of("air");
        let buildings = m.variants_tagged("building");
        let roofs = m.variants_tagged("roof");
        let name = |tile: usize| m.prototype_of(tile).name.as_str();

        assert!(
            m.prototypes.len() >= 60,
            "{culture}: {} modules",
            m.prototypes.len()
        );
        assert_eq!(
            adjacency_violations(&grid, &m.rules, BoundaryCondition::Finite),
            vec![],
            "{culture}"
        );
        for y in 0..grid.height {
            for x in 0..grid.width {
                let column: Vec<usize> = (0..grid.depth).map(|z| grid.get(x, y, z)).collect();
                let names: Vec<&str> = column.iter().map(|&t| name(t)).collect();
                assert!(
                    street.contains(&column[0]),
                    "{culture} ({x}, {y}): {names:?}"
                );
                assert!(
                    air.contains(&column[grid.depth - 1]),
                    "{culture} ({x}, {y}): {names:?}"
                );
                // A building rises from the street to one roof, and nothing is built above it.
                if let Some(top) = column.iter().rposition(|t| buildings.contains(t)) {
                    assert!(
                        roofs.contains(&column[top + 1]),
                        "{culture} ({x}, {y}): {names:?}"
                    );
                    assert!(
                        column[..=top].iter().all(|t| buildings.contains(t)),
                        "{culture} ({x}, {y}): {names:?}"
                    );
                }
            }
        }
        let (cells, share) = walkable_share(&grid, &m);
        let used = (0..m.variants.len())
            .filter(|&tile| grid.count(tile) > 0)
            .count();
        report.insert(culture, (m.prototypes.len(), cells, share, used));
    }

    eprintln!(
        "cultures: the city's walkable cells and share in the largest network {city:?}; each \
         culture's modules, walkable cells, share and tiles used: {report:?}"
    );
    for (culture, (_, cells, share, _)) in &report {
        assert!(
            *cells > 0 && *share >= city.1 - 0.15,
            "{culture}: {share:.2} of {cells} walkable cells joined, the city {:.2}",
            city.1
        );
    }
}
