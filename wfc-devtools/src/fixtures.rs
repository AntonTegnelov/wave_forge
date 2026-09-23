//! Small hand-written rule sets for end-to-end tests and demos. The realistic 3D city lives in
//! [`crate::city`].
//!
//! They are defined in Rust rather than rule files because the tests assert properties that
//! follow from the rules (for example "ground only on the bottom layer"); keeping rules and
//! tile constants side by side keeps those assertions honest.

use crate::render::Color;
use std::collections::BTreeSet;
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
    TileSet::new(
        vec![1.0; num_tiles],
        vec![vec![Transformation::Identity]; num_tiles],
    )
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
/// Meant for grids with depth 1 and finite borders (see docs/architecture/solver.md, "Topology",
/// for why 2D is a one-layer 3D grid).
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
    }

    #[test]
    fn water_never_touches_grass() {
        let coast = coast_2d();
        for along in axis::ALL {
            assert!(!coast.rules.check(coast::WATER, coast::GRASS, along));
        }
    }
}
