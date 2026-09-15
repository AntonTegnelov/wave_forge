//! Small hand-written rule sets for end-to-end tests and demos.
//!
//! They are defined in Rust rather than rule files because the tests assert properties that
//! follow from the rules (for example "ground only on the bottom layer"); keeping rules and
//! tile constants side by side keeps those assertions honest.

use crate::render::Color;
use std::collections::BTreeSet;
use wfc_core::grid::PossibilityGrid;
use wfc_rules::{AdjacencyRules, TileSet, Transformation};

/// Axis indices used by [`AdjacencyRules`].
pub mod axis {
    pub const POS_X: usize = 0;
    pub const NEG_X: usize = 1;
    pub const POS_Y: usize = 2;
    pub const NEG_Y: usize = 3;
    pub const POS_Z: usize = 4;
    pub const NEG_Z: usize = 5;
    pub const HORIZONTAL: [usize; 4] = [POS_X, NEG_X, POS_Y, NEG_Y];
    pub const ALL: [usize; 6] = [POS_X, NEG_X, POS_Y, NEG_Y, POS_Z, NEG_Z];

    /// Axes are stored in pairs, so the opposite direction differs only in the lowest bit.
    pub const fn opposite(axis: usize) -> usize {
        axis ^ 1
    }
}

/// A rule set together with what is needed to render and check it.
pub struct Fixture {
    pub names: &'static [&'static str],
    pub palette: &'static [Color],
    /// Tiles that represent empty space in 3D renders.
    pub empty_tiles: &'static [usize],
    pub tileset: TileSet,
    pub rules: AdjacencyRules,
}

/// Collects adjacency rules, always adding both directions of a pair.
///
/// Adjacency is symmetric: "B may be east of A" means "A may be west of B". Declaring only one
/// side produces rule sets that look reasonable but silently make tiles impossible.
#[derive(Default)]
struct RuleBuilder {
    tuples: BTreeSet<(usize, usize, usize)>,
}

impl RuleBuilder {
    fn allow(&mut self, tile: usize, neighbor: usize, along: usize) -> &mut Self {
        self.tuples.insert((along, tile, neighbor));
        self.tuples.insert((axis::opposite(along), neighbor, tile));
        self
    }

    fn allow_each(&mut self, tile: usize, neighbors: &[usize], axes: &[usize]) -> &mut Self {
        for &neighbor in neighbors {
            for &along in axes {
                self.allow(tile, neighbor, along);
            }
        }
        self
    }

    fn build(&self, num_tiles: usize) -> AdjacencyRules {
        let tuples: Vec<(usize, usize, usize)> = self.tuples.iter().copied().collect();
        AdjacencyRules::from_allowed_tuples(num_tiles, axis::ALL.len(), tuples)
    }
}

fn uniform_tileset(num_tiles: usize) -> TileSet {
    TileSet::new(vec![1.0; num_tiles], vec![vec![Transformation::Identity]; num_tiles])
        .expect("uniform tile set is valid")
}

/// Tiles of [`coast_2d`].
pub mod coast {
    pub const WATER: usize = 0;
    pub const SAND: usize = 1;
    pub const GRASS: usize = 2;
    pub const FOREST: usize = 3;
}

/// A 2D coastline: water only touches sand, sand touches grass, grass touches forest. Any valid
/// result therefore shows smooth bands, so broken propagation is obvious in the rendered PNG.
///
/// Meant for grids with depth 1 and finite borders (see docs/status.md A-2 for why 2D is still a
/// one-layer 3D grid).
pub fn coast_2d() -> Fixture {
    use coast::*;
    let mut rules = RuleBuilder::default();
    for tile in [WATER, SAND, GRASS, FOREST] {
        rules.allow_each(tile, &[tile], &axis::ALL);
    }
    for (tile, neighbor) in [(WATER, SAND), (SAND, GRASS), (GRASS, FOREST)] {
        rules.allow_each(tile, &[neighbor], &axis::HORIZONTAL);
    }
    Fixture {
        names: &["water", "sand", "grass", "forest"],
        palette: &[[38, 90, 190], [222, 204, 130], [92, 168, 72], [36, 104, 48]],
        empty_tiles: &[],
        tileset: uniform_tileset(4),
        rules: rules.build(4),
    }
}

/// Tiles of [`city_3d`].
pub mod city {
    pub const AIR: usize = 0;
    pub const GROUND: usize = 1;
    pub const ROAD_X: usize = 2;
    pub const ROAD_Y: usize = 3;
    pub const CROSSING: usize = 4;
    pub const WALL: usize = 5;
    pub const ROOF: usize = 6;
    pub const ROADS: [usize; 3] = [ROAD_X, ROAD_Y, CROSSING];
}

/// A tiny 3D city with `+z` up, in the spirit of marian42's WFC city: ground and roads on the
/// bottom layer, buildings made of wall blocks capped by roofs, air above.
///
/// Structure comes from the rules alone. Nothing may sit on top of ground or roads except air,
/// walls may only stand on walls, and only walls may support roofs. That also means nothing can
/// be *below* ground, so ground can only exist on the bottom layer. Roads connect along their
/// direction, have ground on both sides, and meet at crossings.
///
/// Use with [`constrain_city_grid`], which pins what rules alone cannot express at the grid's
/// edges: nothing lies below the bottom layer, so the rules would allow air or a roof there.
pub fn city_3d() -> Fixture {
    use axis::*;
    use city::*;
    let mut rules = RuleBuilder::default();
    rules
        .allow_each(GROUND, &[GROUND, WALL], &HORIZONTAL)
        .allow_each(ROAD_X, &[ROAD_X, CROSSING], &[POS_X, NEG_X])
        .allow_each(ROAD_X, &[GROUND], &[POS_Y, NEG_Y])
        .allow_each(ROAD_Y, &[ROAD_Y, CROSSING], &[POS_Y, NEG_Y])
        .allow_each(ROAD_Y, &[GROUND], &[POS_X, NEG_X])
        .allow_each(CROSSING, &[CROSSING], &HORIZONTAL)
        .allow_each(WALL, &[WALL, AIR, ROOF], &HORIZONTAL)
        .allow_each(ROOF, &[ROOF, AIR], &HORIZONTAL)
        .allow_each(AIR, &[AIR], &HORIZONTAL);
    for tile in [GROUND, ROAD_X, ROAD_Y, CROSSING, ROOF, AIR] {
        rules.allow(tile, AIR, POS_Z);
    }
    rules.allow(WALL, WALL, POS_Z).allow(WALL, ROOF, POS_Z);

    Fixture {
        names: &["air", "ground", "road_x", "road_y", "crossing", "wall", "roof"],
        palette: &[
            [0, 0, 0],
            [96, 160, 72],
            [70, 70, 76],
            [70, 70, 76],
            [96, 96, 102],
            [214, 196, 158],
            [180, 70, 50],
        ],
        empty_tiles: &[AIR],
        tileset: uniform_tileset(7),
        rules: rules.build(7),
    }
}

/// Pins the city's boundary layers: the bottom layer is ground, road or the base of a building
/// (never air or a roof floating on nothing), and the top layer is air, so every building ends in
/// a roof inside the grid.
pub fn constrain_city_grid(grid: &mut PossibilityGrid) {
    assert!(grid.depth >= 3, "a city needs at least a floor, a roof and air above it");
    let top = grid.depth - 1;
    for y in 0..grid.height {
        for x in 0..grid.width {
            if let Some(cell) = grid.get_mut(x, y, 0) {
                cell.set(city::AIR, false);
                cell.set(city::ROOF, false);
            }
            if let Some(cell) = grid.get_mut(x, y, top) {
                cell.fill(false);
                cell.set(city::AIR, true);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_symmetric(fixture: &Fixture) {
        let n = fixture.tileset.num_transformed_tiles();
        for along in axis::ALL {
            for a in 0..n {
                for b in 0..n {
                    assert_eq!(
                        fixture.rules.check(a, b, along),
                        fixture.rules.check(b, a, axis::opposite(along)),
                        "{} -> {} along axis {along}",
                        fixture.names[a],
                        fixture.names[b]
                    );
                }
            }
        }
    }

    #[test]
    fn fixture_rules_are_symmetric() {
        assert_symmetric(&coast_2d());
        assert_symmetric(&city_3d());
    }

    #[test]
    fn water_never_touches_grass() {
        let coast = coast_2d();
        for along in axis::ALL {
            assert!(!coast.rules.check(coast::WATER, coast::GRASS, along));
        }
    }

    #[test]
    fn nothing_can_be_below_ground_so_ground_stays_on_the_bottom_layer() {
        let city = city_3d();
        for below in 0..city.names.len() {
            assert!(!city.rules.check(below, city::GROUND, axis::POS_Z), "{} supports ground", city.names[below]);
        }
    }

    #[test]
    fn only_walls_support_roofs() {
        let city = city_3d();
        let supports: Vec<usize> = (0..city.names.len()).filter(|&t| city.rules.check(t, city::ROOF, axis::POS_Z)).collect();
        assert_eq!(supports, vec![city::WALL]);
    }

    #[test]
    fn city_constraints_pin_bottom_and_top_layers() {
        let mut grid = PossibilityGrid::new(2, 2, 3, 7);
        constrain_city_grid(&mut grid);
        let bottom = grid.get(0, 0, 0).unwrap();
        assert!(!bottom[city::AIR] && !bottom[city::ROOF]);
        assert_eq!(grid.get(1, 1, 2).unwrap().iter_ones().collect::<Vec<_>>(), vec![city::AIR]);
        assert_eq!(grid.get(0, 1, 1).unwrap().count_ones(), 7);
    }
}
