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

use crate::invariants::TileGrid;
use crate::render::{Color, VoxelModel};
use std::collections::VecDeque;
use wfc_core::BoundaryCondition;
use wfc_core::grid::PossibilityGrid;
use wfc_rules::modules::{
    CompiledModules, DOWN, Face, HorizontalFace as H, ModulePrototype, ModuleSet, NEG_X, NEG_Y, NUM_AXES, POS_X,
    POS_Y, UP, VerticalFace as V, opposite,
};

/// Connector ids. Horizontal and vertical connectors never meet, but get distinct ids for clarity.
pub mod connector {
    /// Open air above street level.
    pub const AIR: u32 = 0;
    /// Open ground at street level: grass, plazas, road sides, stairs.
    pub const GROUND: u32 = 1;
    /// A road continuing across the face.
    pub const ROAD: u32 = 2;
    /// A building's outer wall or its continuation into a neighbouring building cell.
    pub const BUILDING: u32 = 3;
    /// A building wall with a street door.
    pub const DOOR: u32 = 4;
    /// A building wall with a balcony, which must open onto air.
    pub const BALCONY: u32 = 5;
    /// A walkway deck continuing across the face.
    pub const WALKWAY: u32 = 6;
    /// Vertical: open space continues (ground or roof below, air above).
    pub const OPEN: u32 = 10;
    /// Vertical: the building continues.
    pub const SOLID: u32 = 11;
    /// Vertical: the underside of street level. Nothing fits it, so street-level modules can only
    /// exist on the bottom layer.
    pub const BEDROCK: u32 = 12;
    /// Vertical, oriented: the top of a stair, which must meet a landing facing the same way.
    pub const STAIR: u32 = 13;
}

/// Voxels along each edge of a module's model.
pub const RESOLUTION: usize = 4;

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
}

/// The city's module prototypes and connector pairs.
///
/// Walkability follows marian42: street-level faces are walkable, and walkway, door and landing
/// faces are *paths* that must meet another walkable face. A path can therefore never end at a
/// wall or in mid-air, and every walkway runs between walkway doors, roof terraces and stair
/// landings. Rules are local, so they cannot rule out an island such as a bridge between two
/// buildings that have no street door; [`unreachable_walkable_cells`] measures that.
pub fn module_set() -> ModuleSet {
    use connector::*;
    let street = |name: &str, faces: [H; 4], up: u32| {
        ModulePrototype::new(name, faces, V::invariant(up), V::invariant(BEDROCK)).tag(STREET_LEVEL)
    };
    let building = |name: &str, faces: [H; 4]| {
        ModulePrototype::new(name, faces, V::invariant(SOLID), V::invariant(SOLID)).tag("building")
    };
    let above_building = |name: &str, faces: [H; 4]| {
        ModulePrototype::new(name, faces, V::invariant(OPEN), V::invariant(SOLID)).tag("roof")
    };
    let floating = |name: &str, faces: [H; 4]| {
        ModulePrototype::new(name, faces, V::invariant(OPEN), V::invariant(OPEN)).tag("walkway")
    };
    // Faces, listed in `+x, -x, +y, -y` order below.
    let air = H::symmetric(AIR);
    let wall = H::symmetric(BUILDING);
    let ground = H::symmetric(GROUND).walkable();
    let road = H::symmetric(ROAD).walkable();
    let door = H::plain(DOOR).path();
    let balcony = H::plain(BALCONY);
    let walkway = H::symmetric(WALKWAY).path();

    // Weights apply to each rotated variant, so a prototype with four distinct rotations weighs four
    // times its number in total.
    ModuleSet::new()
        .connect(BUILDING, AIR)
        .connect(BUILDING, GROUND)
        .connect(DOOR, GROUND)
        .connect(BALCONY, AIR)
        // Open space.
        .with(ModulePrototype::new("air", [air; 4], V::invariant(OPEN), V::invariant(OPEN)).weight(4.0))
        .with(street("grass", [ground; 4], OPEN).weight(1.0))
        .with(street("plaza", [ground; 4], OPEN).weight(0.4))
        // Roads: ROAD faces must continue into another road, road sides are open ground.
        .with(street("road_straight", [road, road, ground, ground], OPEN).weight(4.0).tag("road"))
        .with(street("road_corner", [road, ground, road, ground], OPEN).weight(1.0).tag("road"))
        .with(street("road_t", [road, road, road, ground], OPEN).weight(0.8).tag("road"))
        .with(street("road_cross", [road; 4], OPEN).weight(0.6).tag("road"))
        .with(street("road_end", [road, ground, ground, ground], OPEN).weight(0.05).tag("road"))
        // Buildings: a street-level base, floors on top, and a roof above the last floor.
        .with(street("building_base", [wall; 4], SOLID).weight(1.0).tag("building"))
        .with(street("building_door", [door, wall, wall, wall], SOLID).weight(0.3).tag("building"))
        .with(building("building_floor", [wall; 4]).weight(1.0))
        .with(building("building_balcony", [balcony, wall, wall, wall]).weight(0.3))
        .with(building("building_walkway_door", [walkway, wall, wall, wall]).weight(0.1))
        .with(above_building("roof_pyramid", [air; 4]).weight(1.0))
        .with(above_building("roof_flat", [air; 4]).weight(0.5))
        .with(above_building("roof_terrace", [walkway, air, air, air]).weight(0.15).tag("walkway"))
        // Walkways bridge between walkway doors, terraces and landings; there are no dead ends.
        .with(floating("walkway_straight", [walkway, walkway, air, air]).weight(0.3))
        .with(floating("walkway_corner", [walkway, air, walkway, air]).weight(0.05))
        // A stair climbs towards +x from the street; the landing above it continues as a walkway.
        .with(
            ModulePrototype::new("stair", [ground; 4], V::oriented(STAIR, 0), V::invariant(BEDROCK))
                .weight(0.05)
                .tag(STREET_LEVEL)
                .tag("stair"),
        )
        .with(
            ModulePrototype::new("stair_landing", [walkway, air, air, air], V::invariant(OPEN), V::oriented(STAIR, 0))
                .tag("walkway"),
        )
}

/// Compiles the city and builds its voxel models.
pub fn city() -> City {
    let modules = module_set().compile().expect("city module set is valid");
    let voxels = modules
        .variants
        .iter()
        .map(|variant| prototype_model(&modules.prototypes[variant.prototype]).rotated(variant.rotation))
        .collect();
    let air = modules.variants_of("air")[0];
    City { modules, voxels, air }
}

impl City {
    /// One colour per tile for flat maps: the most common colour of the model's topmost voxel
    /// layer, which is what you would see looking straight down at a lone module.
    pub fn map_palette(&self) -> Vec<Color> {
        self.voxels
            .iter()
            .map(|model| {
                let r = model.resolution;
                let top_layer = (0..r).rev().find(|&z| (0..r * r).any(|i| model.get(i % r, i / r, z).is_some()));
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
                counts.into_iter().max_by_key(|&(_, n)| n).map(|(c, _)| c).expect("layer has voxels")
            })
            .collect()
    }
}

/// Pins what the rules leave open at the grid's edges: the bottom layer is street level (the
/// rules already keep street level off every other layer), the top layer is air, so every
/// building gets its roof inside the grid, and no path (walkway, door, landing) points out of the
/// grid's sides, where nothing would continue it. Roads may leave the grid. This mirrors marian42's
/// boundary constraints.
pub fn constrain_city(grid: &mut PossibilityGrid, city: &City) {
    assert!(grid.depth >= 3, "a city needs at least a street, a roof and air above it");
    let street_level = city.modules.variants_tagged(STREET_LEVEL);
    let num_tiles = city.modules.variants.len();
    let top = grid.depth - 1;
    let (width, height) = (grid.width, grid.height);
    let paths_out = |axis: usize| -> Vec<usize> {
        (0..num_tiles)
            .filter(|&tile| matches!(city.modules.face(tile, axis), Face::Horizontal(f) if f.enforce_walkable_neighbor))
            .collect()
    };
    for (axis, on_border) in [
        (POS_X, &(|x: usize, _: usize| x + 1 == width) as &dyn Fn(usize, usize) -> bool),
        (NEG_X, &|x: usize, _: usize| x == 0),
        (POS_Y, &|_: usize, y: usize| y + 1 == height),
        (NEG_Y, &|_: usize, y: usize| y == 0),
    ] {
        let banned = paths_out(axis);
        for z in 0..grid.depth {
            for y in 0..grid.height {
                for x in 0..grid.width {
                    if on_border(x, y) {
                        let cell = grid.get_mut(x, y, z).expect("cell in bounds");
                        for &tile in &banned {
                            cell.set(tile, false);
                        }
                    }
                }
            }
        }
    }
    for y in 0..grid.height {
        for x in 0..grid.width {
            if let Some(cell) = grid.get_mut(x, y, 0) {
                for tile in 0..num_tiles {
                    if !street_level.contains(&tile) {
                        cell.set(tile, false);
                    }
                }
            }
            if let Some(cell) = grid.get_mut(x, y, top) {
                cell.fill(false);
                cell.set(city.air, true);
            }
        }
    }
}

/// Cells with a walkable face that cannot be reached on foot from the street.
///
/// Walking moves between horizontal neighbours whose touching faces are both walkable, climbs a
/// stair to its landing, and passes through buildings: building cells connect to each other in
/// every direction and to a roof directly above. The street network (street-level cells with a
/// walkable face) is the start. Local rules already prevent dead ends; this finds what they
/// cannot: whole networks cut off from the street.
pub fn unreachable_walkable_cells(grid: &TileGrid, city: &City) -> Vec<(usize, usize, usize)> {
    let m = &city.modules;
    let buildings = m.variants_tagged("building");
    let roofs = m.variants_tagged("roof");
    let street_level = m.variants_tagged(STREET_LEVEL);
    let is_building = |tile: usize| buildings.contains(&tile);
    let walkable_face = |tile: usize, axis: usize| matches!(m.face(tile, axis), Face::Horizontal(f) if f.walkable);
    let has_walkable_face = |tile: usize| [POS_X, NEG_X, POS_Y, NEG_Y].into_iter().any(|axis| walkable_face(tile, axis));
    let index = |(x, y, z): (usize, usize, usize)| (z * grid.height + y) * grid.width + x;

    let mut reached = vec![false; grid.width * grid.height * grid.depth];
    let mut queue = VecDeque::new();
    for y in 0..grid.height {
        for x in 0..grid.width {
            let tile = grid.get(x, y, 0);
            if street_level.contains(&tile) && !is_building(tile) && has_walkable_face(tile) {
                reached[index((x, y, 0))] = true;
                queue.push_back((x, y, 0));
            }
        }
    }
    while let Some(cell) = queue.pop_front() {
        let tile = grid.get(cell.0, cell.1, cell.2);
        for axis in 0..NUM_AXES {
            let Some(next) = grid.neighbor(cell, axis, BoundaryCondition::Finite) else {
                continue;
            };
            let other = grid.get(next.0, next.1, next.2);
            let linked = if axis == UP || axis == DOWN {
                let (lower, upper) = if axis == UP { (tile, other) } else { (other, tile) };
                (is_building(lower) && (is_building(upper) || roofs.contains(&upper)))
                    || (m.prototype_of(lower).name == "stair" && m.prototype_of(upper).name == "stair_landing")
            } else {
                (walkable_face(tile, axis) && walkable_face(other, opposite(axis))) || (is_building(tile) && is_building(other))
            };
            if linked && !reached[index(next)] {
                reached[index(next)] = true;
                queue.push_back(next);
            }
        }
    }

    let mut unreachable = Vec::new();
    for z in 0..grid.depth {
        for y in 0..grid.height {
            for x in 0..grid.width {
                if has_walkable_face(grid.get(x, y, z)) && !reached[index((x, y, z))] {
                    unreachable.push((x, y, z));
                }
            }
        }
    }
    unreachable
}

/// The voxel model of an unrotated prototype. Geometry is deliberately crude: it only has to make
/// the structure readable in a render.
fn prototype_model(prototype: &ModulePrototype) -> VoxelModel {
    use palette::*;
    let r = RESOLUTION;
    let mut m = VoxelModel::empty(r);
    let name = prototype.name.as_str();
    let fill = |m: &mut VoxelModel, xs: std::ops::Range<usize>, ys: std::ops::Range<usize>, zs: std::ops::Range<usize>, color: Color| {
        for z in zs {
            for y in ys.clone() {
                for x in xs.clone() {
                    m.set(x, y, z, Some(color));
                }
            }
        }
    };
    // A 2-voxel-wide strip from the centre to every face whose connector is `connector`.
    let strips = |m: &mut VoxelModel, connector: u32, color: Color| {
        let mut any = false;
        for (axis, xs, ys) in [(POS_X, 2..r, 1..3), (NEG_X, 0..2, 1..3), (POS_Y, 1..3, 2..r), (NEG_Y, 1..3, 0..2)] {
            if matches!(prototype.face(axis), Face::Horizontal(face) if face.connector == connector) {
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
        "plaza" => {
            for y in 0..r {
                for x in 0..r {
                    m.set(x, y, 0, Some(if (x + y) % 2 == 0 { PLAZA } else { PLAZA_JOINT }));
                }
            }
        }
        _ if name.starts_with("road") => {
            fill(&mut m, 0..r, 0..r, 0..1, SIDEWALK);
            strips(&mut m, connector::ROAD, ASPHALT);
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
            fill(&mut m, 0..r - 1, 0..r, 0..r, WALL);
            windows(&mut m);
            fill(&mut m, r - 2..r - 1, 1..3, 1..3, DOOR);
            // Balcony floor and railing in the outer voxel column.
            fill(&mut m, r - 1..r, 0..r, 0..1, FLAT_ROOF);
            fill(&mut m, r - 1..r, 0..r, 1..2, WALL);
        }
        "building_walkway_door" => {
            fill(&mut m, 0..r, 0..r, 0..r, WALL);
            windows(&mut m);
            fill(&mut m, r - 1..r, 1..3, 0..3, DOOR);
        }
        "roof_pyramid" => {
            fill(&mut m, 0..r, 0..r, 0..1, ROOF);
            fill(&mut m, 1..3, 1..3, 1..2, ROOF);
        }
        "roof_flat" | "roof_terrace" => {
            fill(&mut m, 0..r, 0..r, 0..1, FLAT_ROOF);
            for i in 0..r {
                for (x, y) in [(0, i), (r - 1, i), (i, 0), (i, r - 1)] {
                    m.set(x, y, 1, Some(FLAT_ROOF));
                }
            }
            if name == "roof_terrace" {
                strips(&mut m, connector::WALKWAY, DECK);
                fill(&mut m, r - 1..r, 1..3, 1..2, DECK);
                for y in 1..3 {
                    m.set(r - 1, y, 1, None);
                }
            }
        }
        _ if name.starts_with("walkway") || name == "stair_landing" => strips(&mut m, connector::WALKWAY, DECK),
        "stair" => {
            fill(&mut m, 0..r, 0..r, 0..1, GRASS);
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
    use wfc_rules::modules::{DOWN, UP};

    #[test]
    fn city_needs_several_possibility_words_but_fits_the_gpu() {
        let city = city();
        let n = city.modules.variants.len();
        assert!(n > 32, "only {n} variants; the city should exercise multi-word cells");
        assert!(n <= 256, "{n} variants exceed the GPU propagation limit");
        assert_eq!(city.voxels.len(), n);
    }

    #[test]
    fn only_street_level_undersides_are_unmatched() {
        let city = city();
        let street_level = city.modules.variants_tagged(STREET_LEVEL);
        let expected: Vec<(usize, usize)> = street_level.iter().map(|&tile| (tile, DOWN)).collect();
        assert_eq!(city.modules.unmatched_faces, expected, "{:?}", city.modules.names);
    }

    #[test]
    fn roofs_stand_only_on_buildings() {
        let city = city();
        let buildings = city.modules.variants_tagged("building");
        for roof in city.modules.variants_tagged("roof") {
            for below in 0..city.modules.variants.len() {
                if city.modules.rules.check(below, roof, UP) {
                    assert!(buildings.contains(&below), "{} under {}", city.modules.names[below], city.modules.names[roof]);
                }
            }
        }
    }

    #[test]
    fn stairs_lead_to_a_landing_facing_the_same_way() {
        let city = city();
        let m = &city.modules;
        for stair in m.variants_of("stair") {
            let above: Vec<usize> = (0..m.variants.len()).filter(|&t| m.rules.check(stair, t, UP)).collect();
            assert_eq!(above.len(), 1, "{} has {:?} above", m.names[stair], above);
            assert_eq!(m.prototype_of(above[0]).name, "stair_landing");
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
                        matches!(m.face(other, wfc_rules::modules::opposite(facing)), Face::Horizontal(f) if f.connector == connector::GROUND),
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
            .find(|&axis| matches!(m.face(tile, axis), Face::Horizontal(f) if f.connector == connector::DOOR))
            .expect("door variant has a door face")
    }

    #[test]
    fn constraints_pin_street_level_below_and_air_on_top() {
        let city = city();
        let n = city.modules.variants.len();
        let mut grid = PossibilityGrid::new(2, 2, 3, n);
        constrain_city(&mut grid, &city);
        let bottom: Vec<usize> = grid.get(0, 0, 0).unwrap().iter_ones().collect();
        let street_level = city.modules.variants_tagged(STREET_LEVEL);
        assert!(bottom.iter().all(|t| street_level.contains(t)), "only street level on the bottom layer");
        assert!(bottom.contains(&city.modules.variants_of("grass")[0]));
        assert_eq!(grid.get(1, 1, 2).unwrap().iter_ones().collect::<Vec<_>>(), vec![city.air]);
        assert!(grid.get(0, 1, 1).unwrap().count_ones() < n, "border cells lose outward paths");
    }

    #[test]
    fn no_path_points_out_of_the_grid() {
        let city = city();
        let m = &city.modules;
        let n = m.variants.len();
        let mut grid = PossibilityGrid::new(3, 3, 3, n);
        constrain_city(&mut grid, &city);
        let outward_path = |tile: usize, axis: usize| matches!(m.face(tile, axis), Face::Horizontal(f) if f.enforce_walkable_neighbor);
        for (x, y, axis) in [(2, 1, POS_X), (0, 1, NEG_X), (1, 2, POS_Y), (1, 0, NEG_Y)] {
            let cell = grid.get(x, y, 1).unwrap();
            assert!(cell.iter_ones().all(|t| !outward_path(t, axis)), "({x}, {y}) axis {axis}");
        }
        let centre = grid.get(1, 1, 1).unwrap();
        assert_eq!(centre.count_ones(), n, "inner cells stay unconstrained");
        let road_out = m.variants_of("road_straight").into_iter().any(|t| grid.get(2, 1, 0).unwrap()[t]);
        assert!(road_out, "roads may leave the grid");
    }

    /// The variant of `name` whose face along `axis` has `connector`.
    fn facing(m: &CompiledModules, name: &str, axis: usize, connector: u32) -> usize {
        m.variants_of(name)
            .into_iter()
            .find(|&t| matches!(m.face(t, axis), Face::Horizontal(f) if f.connector == connector))
            .unwrap_or_else(|| panic!("no {name} with connector {connector} along axis {axis}"))
    }

    #[test]
    fn walkways_never_end_at_walls_or_in_the_air() {
        let city = city();
        let m = &city.modules;
        for walkway in m.variants_tagged("walkway") {
            for axis in [POS_X, NEG_X, POS_Y, NEG_Y] {
                let Face::Horizontal(face) = m.face(walkway, axis) else { unreachable!() };
                if face.connector != connector::WALKWAY {
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
    fn walkways_are_reachable_only_through_a_street_door() {
        let city = city();
        let m = &city.modules;
        let grass = m.variants_of("grass")[0];
        let base = m.variants_of("building_base")[0];
        let air = city.air;
        let door = facing(m, "building_door", NEG_X, connector::DOOR);
        let upstairs = facing(m, "building_walkway_door", POS_X, connector::WALKWAY);
        let bridge = facing(m, "walkway_straight", NEG_X, connector::WALKWAY);

        // x: grass | building | building, with a walkway door and a bridge one floor up.
        let with_door = TileGrid::new(3, 1, 2, vec![grass, door, base, air, upstairs, bridge]).unwrap();
        assert_eq!(unreachable_walkable_cells(&with_door, &city), vec![]);

        let without_door = TileGrid::new(3, 1, 2, vec![grass, base, base, air, upstairs, bridge]).unwrap();
        assert_eq!(unreachable_walkable_cells(&without_door, &city), vec![(1, 0, 1), (2, 0, 1)]);
    }

    #[test]
    fn every_non_air_module_has_voxels() {
        let city = city();
        for (tile, model) in city.voxels.iter().enumerate() {
            let filled = model.voxels.iter().flatten().count();
            assert_eq!(filled == 0, tile == city.air, "{}", city.modules.names[tile]);
        }
    }
}
