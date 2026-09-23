//! A tiny 3D city in the spirit of marian42's WFC city generator, used as a realistic E2E and
//! benchmark workload.
//!
//! The toy rule sets elsewhere in this crate check that the solver works at all. They are far too
//! easy to show what real 3D generation costs: a handful of tiles propagate almost instantly and
//! never contradict. This set is closer to what the project is for. It has dozens of rotated module
//! variants (more than 32, so possibilities span several words per cell), connectors instead of
//! hand-written adjacency, weights, and structure that reaches across many cells: roads with
//! corners and junctions, buildings several floors high with doors, balconies and roofs, walkways
//! bridging streets, and stairs up to them.
//!
//! Every module also has a small voxel model, so renders look like a very crude version of
//! marian42's city rather than coloured cubes.
//!
//! Coordinates are `+z` up, with one module per cell. See [`wfc_rules::modules`] for how connectors
//! become adjacency.

use crate::invariants::{BoundaryCondition, TileGrid};
use crate::render::{Color, VoxelModel};
use std::collections::VecDeque;
use wfc_core::{Prior, TileMask};
use wfc_rules::loader::{RuleFile, parse_rule_file};
use wfc_rules::modules::{
    CompiledModules, DOWN, Face, ModulePrototype, NEG_X, NEG_Y, NUM_AXES, POS_X, POS_Y, UP,
    opposite,
};

/// Voxels along each edge of a module's model.
pub const RESOLUTION: usize = 4;

/// Cell coordinates `(x, y, z)`.
pub type Cell = (usize, usize, usize);

/// Tag of modules that belong on the bottom layer.
pub const STREET_LEVEL: &str = "street_level";

/// The compiled city module set with a voxel model per variant.
pub struct City {
    /// Modules, variants and the derived rules and tile set.
    pub modules: CompiledModules,
    /// Voxel model per tile id.
    pub voxels: Vec<VoxelModel>,
    /// Tile id of empty air.
    pub air: usize,
}

mod palette {
    use crate::render::Color;
    pub const GRASS: Color = [96, 160, 72];
    pub const PLAZA: Color = [196, 186, 164];
    pub const PLAZA_JOINT: Color = [172, 162, 142];
    pub const SIDEWALK: Color = [150, 150, 150];
    pub const ASPHALT: Color = [64, 64, 70];
    pub const WALL: Color = [214, 196, 158];
    pub const WINDOW: Color = [90, 130, 170];
    pub const DOOR: Color = [110, 70, 40];
    pub const ROOF: Color = [180, 70, 50];
    pub const FLAT_ROOF: Color = [120, 120, 125];
    pub const DECK: Color = [160, 120, 80];
    pub const STEP: Color = [176, 176, 176];
    pub const STONE: Color = [196, 190, 176];
    pub const WATER: Color = [80, 140, 210];
    pub const POST: Color = [60, 60, 64];
    pub const LAMP: Color = [240, 220, 120];
}

/// The city's module set, as a rule file any surface can load.
pub const CITY_RON: &str = include_str!("../../examples/city.ron");

/// Compiles the city from [`CITY_RON`] and builds its voxel models.
///
/// Walkability follows marian42: street-level faces are walkable, and walkway, door, flat-roof and
/// stair faces are *paths* that must meet another walkable face, so a path never ends at a wall or
/// in mid-air. Buildings are solid; only arcades lead through them. Flat roofs are walkable and join
/// neighbouring roofs and walkways at the same height, and an edge with nothing to join gets a
/// railing instead. Height is climbed by stairs: from the street, from a roof, or along a building
/// facade as wall stairs whose flights chain storey by storey, as in his set. Walkways rest on
/// pillars or span between roofs and stairs. Rules are local, so they cannot forbid a
/// network cut off as a whole; [`disconnected_walkable_cells`] measures that.
///
/// # Panics
/// If `examples/city.ron` is not a valid module set, which its tests rule out.
pub fn city() -> City {
    let RuleFile::Modules(modules) =
        parse_rule_file(CITY_RON).expect("examples/city.ron is a valid rule file")
    else {
        panic!("examples/city.ron is a module set");
    };
    let voxels = modules
        .variants
        .iter()
        .map(|variant| {
            prototype_model(&modules, &modules.prototypes[variant.prototype])
                .rotated(variant.rotation)
        })
        .collect();
    let air = modules.variants_of("air")[0];
    City {
        modules,
        voxels,
        air,
    }
}

impl City {
    /// One colour per tile for flat maps: the most common colour of the model's topmost voxel
    /// layer, which is what you would see looking straight down at a lone module.
    pub fn map_palette(&self) -> Vec<Color> {
        self.voxels
            .iter()
            .map(|model| {
                let r = model.resolution;
                let top_layer = (0..r)
                    .rev()
                    .find(|&z| (0..r * r).any(|i| model.get(i % r, i / r, z).is_some()));
                let Some(z) = top_layer else {
                    return crate::render::EMPTY;
                };
                let mut counts: Vec<(Color, usize)> = Vec::new();
                for color in (0..r * r).filter_map(|i| model.get(i % r, i / r, z)) {
                    match counts.iter_mut().find(|(c, _)| *c == color) {
                        Some((_, n)) => *n += 1,
                        None => counts.push((color, 1)),
                    }
                }
                counts
                    .into_iter()
                    .max_by_key(|&(_, n)| n)
                    .map(|(c, _)| c)
                    .expect("layer has voxels")
            })
            .collect()
    }
}

/// What the rules leave open at a bounded world's edges, as a [`Prior`] the solver reads.
///
/// The bottom layer is street level (the rules already keep street level off every other layer),
/// the top layer is air, so every building gets its roof inside the world, and no path (walkway,
/// door, landing) points out of the world's sides, where nothing would continue it. Roads may leave
/// it. This mirrors marian42's boundary constraints.
///
/// `depth` is how many layers tall the world is. Layer masks pin the bottom to street level and the
/// top to air; face bans keep paths from pointing out of a bounded world's sides. A streamed world
/// is unbounded along x and y, so no ban applies there and roads simply continue into the next
/// chunk.
///
/// # Panics
/// If `depth` is below three: a city needs a street, a roof and air above it.
#[must_use]
pub fn city_prior(city: &City, depth: u32) -> Prior {
    assert!(
        depth >= 3,
        "a city needs at least a street, a roof and air above it"
    );
    let num_tiles = city.modules.variants.len() as u32;
    let mask_of = |tiles: &[usize]| {
        tiles.iter().fold(TileMask::EMPTY, |mask, &tile| {
            mask.union(TileMask::single(tile as u32))
        })
    };
    let street_level = mask_of(&city.modules.variants_tagged(STREET_LEVEL));
    let mut layers = vec![TileMask::all(num_tiles); depth as usize];
    layers[0] = street_level;
    layers[depth as usize - 1] = TileMask::single(city.air as u32);
    let paths_out = |axis: usize| {
        (0..num_tiles)
            .filter(|&tile| {
                matches!(city.modules.face(tile as usize, axis), Face::Horizontal(f) if f.enforce_walkable_neighbor)
            })
            .fold(TileMask::EMPTY, |mask, tile| mask.union(TileMask::single(tile)))
    };
    [POS_X, NEG_X, POS_Y, NEG_Y].iter().fold(
        Prior::open(num_tiles).with_layers(layers),
        |prior, &axis| prior.with_face_ban(axis, paths_out(axis)),
    )
}

/// Pairs of tiles someone can walk between, as `(axis, from, to)`: `to` in the neighbour along
/// `axis` of a cell holding `from`. Only pairs the adjacency rules allow are listed.
///
/// Walking crosses horizontal faces that are both walkable and climbs from a stair to the headroom
/// above it, whose path continues beside the top step. Buildings are solid: only arcades, whose
/// passage faces are walkable, lead through them, and a door is just a walkable spot on the street.
pub fn walk_links(m: &CompiledModules) -> Vec<(usize, usize, usize)> {
    let stairs = m.variants_tagged("stair");
    let name = |tile: usize| m.prototype_of(tile).name.as_str();
    let n = m.variants.len();
    let mut links = Vec::new();
    for from in 0..n {
        for to in 0..n {
            for axis in [POS_X, NEG_X, POS_Y, NEG_Y] {
                let walk = walkable_face(m, from, axis) && walkable_face(m, to, opposite(axis));
                if walk && m.rules.check(from, to, axis) {
                    links.push((axis, from, to));
                }
            }
            let climb = stairs.contains(&from) && name(to) == "stair_head";
            if climb && m.rules.check(from, to, UP) {
                links.push((UP, from, to));
                links.push((DOWN, to, from));
            }
        }
    }
    links
}

/// Tiles with a walkable face: everything that must belong to the one walkable network.
pub fn walkable_tiles(m: &CompiledModules) -> Vec<usize> {
    (0..m.variants.len())
        .filter(|&tile| {
            [POS_X, NEG_X, POS_Y, NEG_Y]
                .into_iter()
                .any(|axis| walkable_face(m, tile, axis))
        })
        .collect()
}

/// Walkable cells outside the largest walkable network, found by flood fill over [`walk_links`].
/// Empty means every walkable cell can reach every other.
pub fn disconnected_walkable_cells(grid: &TileGrid, city: &City) -> Vec<Cell> {
    let m = &city.modules;
    let n = m.variants.len();
    let mut linked = vec![false; NUM_AXES * n * n];
    for (axis, from, to) in walk_links(m) {
        linked[(axis * n + from) * n + to] = true;
    }
    let walkable = walkable_tiles(m);
    let traversable = |tile: usize| walkable.contains(&tile);
    let index = |(x, y, z): Cell| (z * grid.height + y) * grid.width + x;

    let mut component = vec![usize::MAX; grid.width * grid.height * grid.depth];
    let mut walkable_per_component = Vec::new();
    for z in 0..grid.depth {
        for y in 0..grid.height {
            for x in 0..grid.width {
                if !traversable(grid.get(x, y, z)) || component[index((x, y, z))] != usize::MAX {
                    continue;
                }
                let id = walkable_per_component.len();
                let mut count = 0;
                component[index((x, y, z))] = id;
                let mut queue = VecDeque::from([(x, y, z)]);
                while let Some(cell) = queue.pop_front() {
                    let tile = grid.get(cell.0, cell.1, cell.2);
                    count += usize::from(walkable.contains(&tile));
                    for axis in 0..NUM_AXES {
                        let Some(next) = grid.neighbor(cell, axis, BoundaryCondition::Finite)
                        else {
                            continue;
                        };
                        let other = grid.get(next.0, next.1, next.2);
                        if linked[(axis * n + tile) * n + other]
                            && component[index(next)] == usize::MAX
                        {
                            component[index(next)] = id;
                            queue.push_back(next);
                        }
                    }
                }
                walkable_per_component.push(count);
            }
        }
    }
    let largest = walkable_per_component
        .iter()
        .enumerate()
        .max_by_key(|&(_, &count)| count)
        .map(|(id, _)| id);

    let mut disconnected = Vec::new();
    for z in 0..grid.depth {
        for y in 0..grid.height {
            for x in 0..grid.width {
                if walkable.contains(&grid.get(x, y, z))
                    && Some(component[index((x, y, z))]) != largest
                {
                    disconnected.push((x, y, z));
                }
            }
        }
    }
    disconnected
}

/// How often a walkable face meets a walkable face across it, among the horizontal faces where at
/// least one side is walkable.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FaceMeetings {
    /// Adjacent pairs with a walkable face on at least one side.
    pub walkable: usize,
    /// Of those, the pairs where the walk continues: both facing faces are walkable.
    pub continued: usize,
}

impl FaceMeetings {
    /// The share of walkable faces the walk continues through, or `None` when there were none.
    #[must_use]
    pub fn share(self) -> Option<f64> {
        (self.walkable > 0).then(|| self.continued as f64 / self.walkable as f64)
    }
}

impl std::ops::Add for FaceMeetings {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            walkable: self.walkable + other.walkable,
            continued: self.continued + other.continued,
        }
    }
}

/// Walkable faces meeting inside chunks and across the seams between them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WalkContinuity {
    pub inside: FaceMeetings,
    pub across_seams: FaceMeetings,
}

/// Counts, for a grid laid out in chunks of `chunk` cells along x and y, how often a path that
/// reaches a face goes on through it, separately inside chunks and across their seams.
///
/// A chunk is solved against its neighbours' fixed borders, so a seam is where generation could cut
/// paths that the rules alone would have continued. Comparing the two shares shows whether it does.
pub fn walk_continuity(grid: &TileGrid, city: &City, chunk: (usize, usize)) -> WalkContinuity {
    let m = &city.modules;
    let mut continuity = WalkContinuity::default();
    for z in 0..grid.depth {
        for y in 0..grid.height {
            for x in 0..grid.width {
                let tile = grid.get(x, y, z);
                for (axis, seam) in [
                    (POS_X, (x + 1) % chunk.0 == 0),
                    (POS_Y, (y + 1) % chunk.1 == 0),
                ] {
                    let Some(next) = grid.neighbor((x, y, z), axis, BoundaryCondition::Finite)
                    else {
                        continue;
                    };
                    let other = grid.get(next.0, next.1, next.2);
                    let (here, there) = (
                        walkable_face(m, tile, axis),
                        walkable_face(m, other, opposite(axis)),
                    );
                    if !(here || there) {
                        continue;
                    }
                    let meetings = if seam {
                        &mut continuity.across_seams
                    } else {
                        &mut continuity.inside
                    };
                    meetings.walkable += 1;
                    meetings.continued += usize::from(here && there);
                }
            }
        }
    }
    continuity
}

/// Whether `tile`'s face along a horizontal `axis` is one someone can walk through.
fn walkable_face(m: &CompiledModules, tile: usize, axis: usize) -> bool {
    matches!(m.face(tile, axis), Face::Horizontal(f) if f.walkable)
}

/// The voxel model of an unrotated prototype. Geometry is deliberately crude: it only has to make
/// the structure readable in a render.
fn prototype_model(modules: &CompiledModules, prototype: &ModulePrototype) -> VoxelModel {
    use palette::*;
    let r = RESOLUTION;
    let mut m = VoxelModel::empty(r);
    let name = prototype.name.as_str();
    let fill = |m: &mut VoxelModel,
                xs: std::ops::Range<usize>,
                ys: std::ops::Range<usize>,
                zs: std::ops::Range<usize>,
                color: Color| {
        for z in zs {
            for y in ys.clone() {
                for x in xs.clone() {
                    m.set(x, y, z, Some(color));
                }
            }
        }
    };
    // A 2-voxel-wide strip from the centre to every face whose connector is `connector`.
    let strips = |m: &mut VoxelModel, connector: &str, color: Color| {
        let connector = modules
            .connector(connector)
            .unwrap_or_else(|| panic!("the city has a {connector} connector"));
        let mut any = false;
        for (axis, xs, ys) in [
            (POS_X, 2..r, 1..3),
            (NEG_X, 0..2, 1..3),
            (POS_Y, 1..3, 2..r),
            (NEG_Y, 1..3, 0..2),
        ] {
            if matches!(prototype.face(axis), Face::Horizontal(face) if face.connector == connector)
            {
                fill(m, xs, ys, 0..1, color);
                any = true;
            }
        }
        if any {
            fill(m, 1..3, 1..3, 0..1, color);
        }
    };
    let windows = |m: &mut VoxelModel| {
        for i in 1..3 {
            for z in 1..3 {
                for (x, y) in [(r - 1, i), (0, i), (i, r - 1), (i, 0)] {
                    m.set(x, y, z, Some(WINDOW));
                }
            }
        }
    };

    match name {
        "air" => {}
        "grass" => fill(&mut m, 0..r, 0..r, 0..1, GRASS),
        "plaza" | "plaza_fountain" | "plaza_lamp" | "pillar_base" => {
            for y in 0..r {
                for x in 0..r {
                    m.set(
                        x,
                        y,
                        0,
                        Some(if (x + y) % 2 == 0 { PLAZA } else { PLAZA_JOINT }),
                    );
                }
            }
            match name {
                "plaza_fountain" => fill(&mut m, 1..3, 1..3, 1..2, WATER),
                "plaza_lamp" => {
                    fill(&mut m, 1..2, 1..2, 1..3, POST);
                    m.set(1, 1, 3, Some(LAMP));
                }
                "pillar_base" => fill(&mut m, 1..3, 1..3, 1..r, STONE),
                _ => {}
            }
        }
        "pillar" => fill(&mut m, 1..3, 1..3, 0..r, STONE),
        "building_arcade" => {
            fill(&mut m, 0..r, 0..1, 0..r, WALL);
            fill(&mut m, 0..r, r - 1..r, 0..r, WALL);
            fill(&mut m, 0..r, 1..r - 1, r - 1..r, WALL);
            fill(&mut m, 0..r, 1..r - 1, 0..1, PLAZA);
        }
        "roof_tower" => {
            fill(&mut m, 0..r, 0..r, 0..1, ROOF);
            fill(&mut m, 1..3, 1..3, 1..3, WALL);
            fill(&mut m, 1..3, 1..3, 3..4, ROOF);
        }
        _ if name.starts_with("road") => {
            fill(&mut m, 0..r, 0..r, 0..1, SIDEWALK);
            strips(&mut m, "road", ASPHALT);
        }
        "building_base" | "building_door" => {
            fill(&mut m, 0..r, 0..r, 0..r, WALL);
            if name == "building_door" {
                fill(&mut m, r - 1..r, 1..3, 0..3, DOOR);
            }
        }
        "building_floor" => {
            fill(&mut m, 0..r, 0..r, 0..r, WALL);
            windows(&mut m);
        }
        "building_balcony" => {
            fill(&mut m, 0..r, 0..r, 0..r, WALL);
            windows(&mut m);
            // Carve the balcony out of the outer voxel column only after placing windows, or a window
            // would float in it.
            for z in 0..r {
                for y in 0..r {
                    m.set(r - 1, y, z, None);
                }
            }
            fill(&mut m, r - 2..r - 1, 1..3, 1..3, DOOR);
            // Balcony floor and railing in the outer voxel column.
            fill(&mut m, r - 1..r, 0..r, 0..1, FLAT_ROOF);
            fill(&mut m, r - 1..r, 0..r, 1..2, WALL);
        }
        "building_passage" => {
            fill(&mut m, 0..r, 0..r, 0..r, WALL);
            windows(&mut m);
            // Carve the tunnel after placing windows, so none float in it.
            for z in 0..r - 1 {
                for y in 1..r - 1 {
                    for x in 0..r {
                        m.set(x, y, z, None);
                    }
                }
            }
            fill(&mut m, 0..r, 1..r - 1, 0..1, FLAT_ROOF);
        }
        "roof_pyramid" => {
            fill(&mut m, 0..r, 0..r, 0..1, ROOF);
            fill(&mut m, 1..3, 1..3, 1..2, ROOF);
        }
        _ if name.starts_with("roof_flat") => {
            fill(&mut m, 0..r, 0..r, 0..1, FLAT_ROOF);
            // A railing along every edge that is not a walkable connection.
            for (axis, xs, ys) in [
                (POS_X, r - 1..r, 0..r),
                (NEG_X, 0..1, 0..r),
                (POS_Y, 0..r, r - 1..r),
                (NEG_Y, 0..r, 0..1),
            ] {
                if matches!(prototype.face(axis), Face::Horizontal(f) if !f.walkable) {
                    fill(&mut m, xs, ys, 1..2, STONE);
                }
            }
        }
        _ if name.starts_with("walkway") => strips(&mut m, "walkway", DECK),
        // Headroom above a stair: nothing to draw.
        "stair_head" => {}
        "stair" | "stair_roof" | "stair_wall" | "stair_wall_street" => {
            if name != "stair_wall" {
                let floor = if name == "stair_roof" {
                    FLAT_ROOF
                } else {
                    GRASS
                };
                fill(&mut m, 0..r, 0..r, 0..1, floor);
            }
            for x in 0..r {
                fill(&mut m, x..x + 1, 1..3, 0..x + 1, STEP);
            }
        }
        other => panic!("no voxel model for city module {other:?}"),
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use wfc_core::{ChunkShape, WorldExtent};
    use wfc_rules::modules::{DOWN, UP};

    #[test]
    fn city_needs_several_possibility_words_but_fits_the_gpu() {
        let city = city();
        let n = city.modules.variants.len();
        assert!(
            n > 32,
            "only {n} variants; the city should exercise multi-word cells"
        );
        assert!(n <= 256, "{n} variants exceed the GPU propagation limit");
        assert_eq!(city.voxels.len(), n);
    }

    #[test]
    fn only_street_level_undersides_are_unmatched() {
        let city = city();
        let street_level = city.modules.variants_tagged(STREET_LEVEL);
        let expected: Vec<(usize, usize)> = street_level.iter().map(|&tile| (tile, DOWN)).collect();
        assert_eq!(
            city.modules.unmatched_faces, expected,
            "{:?}",
            city.modules.names
        );
    }

    #[test]
    fn roofs_stand_only_on_buildings() {
        let city = city();
        let buildings = city.modules.variants_tagged("building");
        for roof in city.modules.variants_tagged("roof") {
            for below in 0..city.modules.variants.len() {
                if city.modules.rules.check(below, roof, UP) {
                    assert!(
                        buildings.contains(&below),
                        "{} under {}",
                        city.modules.names[below],
                        city.modules.names[roof]
                    );
                }
            }
        }
    }

    #[test]
    fn stairs_lead_up_to_headroom_facing_the_same_way() {
        let city = city();
        let m = &city.modules;
        for stair in m.variants_tagged("stair") {
            let above: Vec<usize> = (0..m.variants.len())
                .filter(|&t| m.rules.check(stair, t, UP))
                .collect();
            assert_eq!(above.len(), 1, "{} has {:?} above", m.names[stair], above);
            assert_eq!(m.prototype_of(above[0]).name, "stair_head");
            assert_eq!(m.variants[above[0]].rotation, m.variants[stair].rotation);
        }
    }

    #[test]
    fn doors_open_onto_street_level_ground() {
        let city = city();
        let m = &city.modules;
        for door in m.variants_of("building_door") {
            let facing = rotate_to_door(m, door);
            for other in 0..m.variants.len() {
                if m.rules.check(door, other, facing) {
                    assert!(
                        matches!(m.face(other, wfc_rules::modules::opposite(facing)), Face::Horizontal(f) if Some(f.connector) == m.connector("ground")),
                        "{} opens onto {}",
                        m.names[door],
                        m.names[other]
                    );
                }
            }
        }
    }

    fn rotate_to_door(m: &CompiledModules, tile: usize) -> usize {
        [POS_X, NEG_X, POS_Y, NEG_Y]
            .into_iter()
            .find(|&axis| matches!(m.face(tile, axis), Face::Horizontal(f) if Some(f.connector) == m.connector("door")))
            .expect("door variant has a door face")
    }

    /// The world the prior is read against: three chunks along x and y so the sides are borders,
    /// and one chunk tall.
    fn world(depth: u32) -> WorldExtent {
        WorldExtent::new(ChunkShape {
            x: 3,
            y: 3,
            z: depth,
        })
        .with_x(0..1)
        .with_y(0..1)
        .with_z(0..1)
    }

    #[test]
    fn the_prior_pins_street_level_below_and_air_on_top() {
        let city = city();
        let n = city.modules.variants.len() as u32;
        let extent = world(3);
        let prior = city_prior(&city, 3);

        let bottom = prior.domain([1, 1, 0], &extent);
        let street_level = city.modules.variants_tagged(STREET_LEVEL);
        assert!(
            bottom.iter().all(|t| street_level.contains(&(t as usize))),
            "only street level on the bottom layer"
        );
        assert!(bottom.contains(city.modules.variants_of("grass")[0] as u32));
        assert_eq!(
            prior.domain([1, 1, 2], &extent).iter().collect::<Vec<_>>(),
            vec![city.air as u32],
            "air on top, so every building roofs inside the world"
        );
        assert_eq!(
            prior.domain([1, 1, 1], &extent).count(),
            n,
            "inner cells stay open"
        );
    }

    #[test]
    fn the_prior_lets_no_path_point_out_of_a_bounded_world() {
        let city = city();
        let m = &city.modules;
        let extent = world(3);
        let prior = city_prior(&city, 3);
        let outward_path = |tile: u32, axis: usize| matches!(m.face(tile as usize, axis), Face::Horizontal(f) if f.enforce_walkable_neighbor);

        for (x, y, axis) in [(2, 1, POS_X), (0, 1, NEG_X), (1, 2, POS_Y), (1, 0, NEG_Y)] {
            let cell = prior.domain([x, y, 1], &extent);
            assert!(
                cell.iter().all(|t| !outward_path(t, axis)),
                "({x}, {y}) axis {axis}"
            );
        }
        let road_out = m
            .variants_of("road_straight")
            .into_iter()
            .any(|t| prior.domain([2, 1, 0], &extent).contains(t as u32));
        assert!(road_out, "roads may leave the world");
    }

    /// The variant of `name` whose face along `axis` has `connector`.
    fn facing(m: &CompiledModules, name: &str, axis: usize, connector: &str) -> usize {
        let id = m
            .connector(connector)
            .expect("the city names its connectors");
        m.variants_of(name)
            .into_iter()
            .find(|&t| matches!(m.face(t, axis), Face::Horizontal(f) if f.connector == id))
            .unwrap_or_else(|| panic!("no {name} with connector {connector} along axis {axis}"))
    }

    #[test]
    fn walkways_never_end_at_walls_or_in_the_air() {
        let city = city();
        let m = &city.modules;
        for walkway in m.variants_tagged("walkway") {
            for axis in [POS_X, NEG_X, POS_Y, NEG_Y] {
                let Face::Horizontal(face) = m.face(walkway, axis) else {
                    unreachable!()
                };
                if Some(face.connector) != m.connector("walkway") {
                    continue;
                }
                for other in 0..m.variants.len() {
                    if m.rules.check(walkway, other, axis) {
                        assert!(
                            matches!(m.face(other, opposite(axis)), Face::Horizontal(f) if f.walkable),
                            "{} path runs into {}",
                            m.names[walkway],
                            m.names[other]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn buildings_are_solid_so_only_stairs_reach_a_roof() {
        let city = city();
        let m = &city.modules;
        let grass = m.variants_of("grass")[0];
        let base = m.variants_of("building_base")[0];
        let air = city.air;
        let door = facing(m, "building_door", NEG_X, "door");
        let stair = m.variants_of("stair")[0]; // rotation 0: entrance at -x, climbing towards +x
        let head = facing(m, "stair_head", POS_X, "walkway");
        let roof = facing(m, "roof_flat_end", NEG_X, "walkway");

        // x: grass, grass, stair or door, building, building; one storey up, the roof sits on x = 3.
        let with_stair = vec![grass, grass, stair, base, base, air, air, head, roof, air];
        assert_eq!(
            disconnected_walkable_cells(&TileGrid::new(5, 1, 2, with_stair).unwrap(), &city),
            vec![]
        );

        let with_door = vec![grass, grass, door, base, base, air, air, air, roof, air];
        assert_eq!(
            disconnected_walkable_cells(&TileGrid::new(5, 1, 2, with_door).unwrap(), &city),
            vec![(3, 0, 1)],
            "a door does not lead up through a solid building"
        );
    }

    #[test]
    fn a_path_that_goes_on_counts_as_continued() {
        let city = city();
        let grass = city.modules.variants_of("grass")[0];
        let grid = TileGrid::new(2, 1, 1, vec![grass, grass]).unwrap();

        let continuity = walk_continuity(&grid, &city, (2, 1));

        let one_continued = FaceMeetings {
            walkable: 1,
            continued: 1,
        };
        assert_eq!(continuity.inside, one_continued);
        assert_eq!(continuity.across_seams, FaceMeetings::default());
    }

    #[test]
    fn a_path_that_runs_into_a_wall_counts_as_cut() {
        let city = city();
        let m = &city.modules;
        let grid = TileGrid::new(
            2,
            1,
            1,
            vec![m.variants_of("grass")[0], m.variants_of("building_base")[0]],
        )
        .unwrap();

        let continuity = walk_continuity(&grid, &city, (2, 1));

        assert_eq!(
            continuity.inside,
            FaceMeetings {
                walkable: 1,
                continued: 0
            }
        );
    }

    #[test]
    fn a_pair_that_straddles_a_chunk_boundary_is_a_seam() {
        let city = city();
        let m = &city.modules;
        let (grass, base) = (m.variants_of("grass")[0], m.variants_of("building_base")[0]);
        // A 2x2 grid of 1x1 chunks: every adjacent pair is a seam.
        let grid = TileGrid::new(2, 2, 1, vec![grass, grass, grass, base]).unwrap();

        let continuity = walk_continuity(&grid, &city, (1, 1));

        assert_eq!(continuity.inside, FaceMeetings::default());
        assert_eq!(
            continuity.across_seams,
            FaceMeetings {
                walkable: 4,
                continued: 2
            }
        );
    }

    #[test]
    fn windows_sit_in_walls() {
        let city = city();
        for (tile, model) in city.voxels.iter().enumerate() {
            let r = model.resolution;
            for z in 0..r {
                for y in 0..r {
                    for x in 0..r {
                        if model.get(x, y, z) != Some(palette::WINDOW) {
                            continue;
                        }
                        let on_surface = x == 0 || y == 0 || x == r - 1 || y == r - 1;
                        let framed = |z: usize| {
                            matches!(model.get(x, y, z), Some(palette::WALL | palette::WINDOW))
                        };
                        assert!(
                            on_surface && z > 0 && z + 1 < r && framed(z - 1) && framed(z + 1),
                            "floating window in {} at ({x}, {y}, {z})",
                            city.modules.names[tile]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn every_non_air_module_has_voxels() {
        let city = city();
        for (tile, model) in city.voxels.iter().enumerate() {
            let filled = model.voxels.iter().flatten().count();
            let headroom = city.modules.prototype_of(tile).name == "stair_head";
            assert_eq!(
                filled == 0,
                tile == city.air || headroom,
                "{}",
                city.modules.names[tile]
            );
        }
    }
}
