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
use wfc_core::constraint::ConnectivityConstraint;
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
    /// The side of a wall stair, fixed to a building facade.
    pub const WALL_SIDE: u32 = 7;
    /// Vertical: open space continues (ground or roof below, air above).
    pub const OPEN: u32 = 10;
    /// Vertical: the building continues.
    pub const SOLID: u32 = 11;
    /// Vertical: the underside of street level. Nothing fits it, so street-level modules can only
    /// exist on the bottom layer.
    pub const BEDROCK: u32 = 12;
    /// Vertical, oriented: the top of a stair, which must meet the headroom above it facing the same way.
    pub const STAIR: u32 = 13;
    /// Vertical: a pillar continues, ending under a walkway.
    pub const PILLAR: u32 = 14;
}

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

/// The city's module prototypes and connector pairs.
///
/// Walkability follows marian42: street-level faces are walkable, and walkway, door, flat-roof and
/// stair faces are *paths* that must meet another walkable face, so a path never ends at a wall or
/// in mid-air. Buildings are solid; only arcades lead through them. Flat roofs are walkable and join
/// neighbouring roofs and walkways at the same height, and an edge with nothing to join gets a
/// railing instead. Height is climbed by stairs: from the street, from a roof, or along a building
/// facade as wall stairs whose flights chain storey by storey, as in his set. Walkways rest on
/// pillars or span between roofs and stairs. Rules are local, so they cannot forbid a
/// network cut off as a whole; [`disconnected_walkable_cells`] measures that.
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
    // A street door needs open ground in front of it but leads into a solid building, so it is not
    // part of the walkable network itself.
    let door = H { enforce_walkable_neighbor: true, ..H::plain(DOOR) };
    let balcony = H::plain(BALCONY);
    // Walkway decks, flat roofs, walkway doors and stair tops all sit at floor level of their cell,
    // so they share one connector and join each other.
    let walkway = H::symmetric(WALKWAY).path();
    // A roof edge with a railing: it borders open air (or a taller building) and nobody crosses it.
    let railing = air;
    // A street-level face nobody walks through, such as the side of a stair.
    let blocked = H::symmetric(GROUND);

    // Weights apply to each rotated variant, so a prototype with four distinct rotations weighs four
    // times its number in total.
    ModuleSet::new()
        .connect(BUILDING, AIR)
        .connect(BUILDING, GROUND)
        .connect(DOOR, GROUND)
        .connect(BALCONY, AIR)
        .connect(WALL_SIDE, BUILDING)
        // Open space and squares.
        .with(ModulePrototype::new("air", [air; 4], V::invariant(OPEN), V::invariant(OPEN)).weight(4.0))
        .with(street("grass", [ground; 4], OPEN).weight(1.0))
        .with(street("plaza", [ground; 4], OPEN).weight(0.4))
        .with(street("plaza_fountain", [ground; 4], OPEN).weight(0.05))
        .with(street("plaza_lamp", [ground; 4], OPEN).weight(0.1))
        // Roads: ROAD faces must continue into another road, road sides are open ground.
        .with(street("road_straight", [road, road, ground, ground], OPEN).weight(4.0).tag("road"))
        .with(street("road_corner", [road, ground, road, ground], OPEN).weight(1.0).tag("road"))
        .with(street("road_t", [road, road, road, ground], OPEN).weight(0.8).tag("road"))
        .with(street("road_cross", [road; 4], OPEN).weight(0.6).tag("road"))
        .with(street("road_end", [road, ground, ground, ground], OPEN).weight(0.05).tag("road"))
        // Buildings: a street-level base, floors on top, and a roof above the last floor. Arcades
        // let the street pass through a building.
        .with(street("building_base", [wall; 4], SOLID).weight(1.0).tag("building"))
        .with(street("building_door", [door, wall, wall, wall], SOLID).weight(0.3).tag("building"))
        .with(street("building_arcade", [ground.path(), ground.path(), wall, wall], SOLID).weight(0.4).tag("building"))
        .with(building("building_floor", [wall; 4]).weight(1.0))
        .with(building("building_balcony", [balcony, wall, wall, wall]).weight(0.3))
        // A tunnel through an upper storey, joining walkways and flat roofs on either side.
        .with(building("building_passage", [walkway, walkway, wall, wall]).weight(0.3))
        // Roofs: pitched roofs close a building; flat roofs are walkable.
        .with(above_building("roof_pyramid", [air; 4]).weight(0.6))
        .with(above_building("roof_tower", [air; 4]).weight(0.1))
        .with(above_building("roof_flat", [walkway; 4]).weight(0.3))
        .with(above_building("roof_flat_edge", [railing, walkway, walkway, walkway]).weight(0.3))
        .with(above_building("roof_flat_corner", [railing, walkway, railing, walkway]).weight(0.3))
        .with(above_building("roof_flat_strip", [walkway, walkway, railing, railing]).weight(0.15))
        .with(above_building("roof_flat_end", [walkway, railing, railing, railing]).weight(0.15))
        // Walkways bridge between walkway doors, flat roofs and stair tops, on pillars or spanning
        // between buildings; there are no dead ends.
        .with(floating("walkway_straight", [walkway, walkway, air, air]).weight(0.15))
        .with(floating("walkway_corner", [walkway, air, walkway, air]).weight(0.05))
        .with(
            ModulePrototype::new("walkway_on_pillar", [walkway, walkway, air, air], V::invariant(OPEN), V::invariant(PILLAR))
                .weight(0.15)
                .tag("walkway"),
        )
        .with(street("pillar_base", [ground; 4], PILLAR).weight(0.1))
        .with(ModulePrototype::new("pillar", [air; 4], V::invariant(PILLAR), V::invariant(PILLAR)).weight(1.0).tag("pillar"))
        // A stair climbs one storey towards +x, from the street or from a roof. Its top step is level
        // with the floor of the next storey. The cell above the stair is headroom, and its +x face is
        // the path the stair leads to (a walkway, flat roof or walkway door beside the top step, never
        // on top of the steps). Only the entrance at -x is walkable where the stair stands.
        .with(
            ModulePrototype::new("stair", [blocked, ground.path(), blocked, blocked], V::oriented(STAIR, 0), V::invariant(BEDROCK))
                .weight(0.6)
                .tag(STREET_LEVEL)
                .tag("stair"),
        )
        .with(
            ModulePrototype::new("stair_roof", [railing, walkway, railing, railing], V::oriented(STAIR, 0), V::invariant(SOLID))
                .weight(0.3)
                .tag("roof")
                .tag("stair"),
        )
        // A wall stair starting at the street, so flights along a facade can be reached from the
        // pavement: the only way upper networks join the street besides free-standing stairs.
        .with(
            ModulePrototype::new(
                "stair_wall_street",
                [blocked, ground.path(), H::plain(WALL_SIDE), blocked],
                V::oriented(STAIR, 0),
                V::invariant(BEDROCK),
            )
            .weight(0.3)
            .tag(STREET_LEVEL)
            .tag("stair"),
        )
        // A wall stair hangs on a building facade (its +y side) in open air. Its head opens onto the
        // next flight's entrance, so flights climb a tall facade storey by storey.
        .with(
            ModulePrototype::new(
                "stair_wall",
                [air, walkway, H::plain(WALL_SIDE), railing],
                V::oriented(STAIR, 0),
                V::invariant(OPEN),
            )
            .weight(0.3)
            .tag("stair"),
        )
        .with(
            ModulePrototype::new("stair_head", [walkway, air, air, air], V::invariant(OPEN), V::oriented(STAIR, 0))
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

/// Pairs of tiles someone can walk between, as `(axis, from, to)`: `to` in the neighbour along
/// `axis` of a cell holding `from`. Only pairs the adjacency rules allow are listed.
///
/// Walking crosses horizontal faces that are both walkable and climbs from a stair to the headroom
/// above it, whose path continues beside the top step. Buildings are solid: only arcades, whose
/// passage faces are walkable, lead through them, and a door is just a walkable spot on the street.
pub fn walk_links(m: &CompiledModules) -> Vec<(usize, usize, usize)> {
    let stairs = m.variants_tagged("stair");
    let name = |tile: usize| m.prototype_of(tile).name.as_str();
    let walkable_face = |tile: usize, axis: usize| matches!(m.face(tile, axis), Face::Horizontal(f) if f.walkable);
    let n = m.variants.len();
    let mut links = Vec::new();
    for from in 0..n {
        for to in 0..n {
            for axis in [POS_X, NEG_X, POS_Y, NEG_Y] {
                let walk = walkable_face(from, axis) && walkable_face(to, opposite(axis));
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
            [POS_X, NEG_X, POS_Y, NEG_Y].into_iter().any(|axis| matches!(m.face(tile, axis), Face::Horizontal(f) if f.walkable))
        })
        .collect()
}

/// A constraint that forces the city into a single walkable network: every cell that can only hold
/// walkable tiles must stay connected to every other over [`walk_links`].
///
/// The module set alone gets close (see docs/constraints.md); this guarantees it, at the cost of
/// work between propagation steps and more restarts. It is not used by the default city test.
pub fn connectivity_constraint(city: &City) -> ConnectivityConstraint {
    let m = &city.modules;
    let walkable = walkable_tiles(m);
    ConnectivityConstraint::new(m.variants.len(), walkable.clone(), walkable, walk_links(m))
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
                        let Some(next) = grid.neighbor(cell, axis, BoundaryCondition::Finite) else {
                            continue;
                        };
                        let other = grid.get(next.0, next.1, next.2);
                        if linked[(axis * n + tile) * n + other] && component[index(next)] == usize::MAX {
                            component[index(next)] = id;
                            queue.push_back(next);
                        }
                    }
                }
                walkable_per_component.push(count);
            }
        }
    }
    let largest = walkable_per_component.iter().enumerate().max_by_key(|&(_, &count)| count).map(|(id, _)| id);

    let mut disconnected = Vec::new();
    for z in 0..grid.depth {
        for y in 0..grid.height {
            for x in 0..grid.width {
                if walkable.contains(&grid.get(x, y, z)) && Some(component[index((x, y, z))]) != largest {
                    disconnected.push((x, y, z));
                }
            }
        }
    }
    disconnected
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
        "plaza" | "plaza_fountain" | "plaza_lamp" | "pillar_base" => {
            for y in 0..r {
                for x in 0..r {
                    m.set(x, y, 0, Some(if (x + y) % 2 == 0 { PLAZA } else { PLAZA_JOINT }));
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
            for (axis, xs, ys) in [(POS_X, r - 1..r, 0..r), (NEG_X, 0..1, 0..r), (POS_Y, 0..r, r - 1..r), (NEG_Y, 0..r, 0..1)] {
                if matches!(prototype.face(axis), Face::Horizontal(f) if !f.walkable) {
                    fill(&mut m, xs, ys, 1..2, STONE);
                }
            }
        }
        _ if name.starts_with("walkway") => strips(&mut m, connector::WALKWAY, DECK),
        // Headroom above a stair: nothing to draw.
        "stair_head" => {}
        "stair" | "stair_roof" | "stair_wall" | "stair_wall_street" => {
            if name != "stair_wall" {
                let floor = if name == "stair_roof" { FLAT_ROOF } else { GRASS };
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
    fn stairs_lead_up_to_headroom_facing_the_same_way() {
        let city = city();
        let m = &city.modules;
        for stair in m.variants_tagged("stair") {
            let above: Vec<usize> = (0..m.variants.len()).filter(|&t| m.rules.check(stair, t, UP)).collect();
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
    fn buildings_are_solid_so_only_stairs_reach_a_roof() {
        let city = city();
        let m = &city.modules;
        let grass = m.variants_of("grass")[0];
        let base = m.variants_of("building_base")[0];
        let air = city.air;
        let door = facing(m, "building_door", NEG_X, connector::DOOR);
        let stair = m.variants_of("stair")[0]; // rotation 0: entrance at -x, climbing towards +x
        let head = facing(m, "stair_head", POS_X, connector::WALKWAY);
        let roof = facing(m, "roof_flat_end", NEG_X, connector::WALKWAY);

        // x: grass, grass, stair or door, building, building; one storey up, the roof sits on x = 3.
        let with_stair = vec![grass, grass, stair, base, base, air, air, head, roof, air];
        assert_eq!(disconnected_walkable_cells(&TileGrid::new(5, 1, 2, with_stair).unwrap(), &city), vec![]);

        let with_door = vec![grass, grass, door, base, base, air, air, air, roof, air];
        assert_eq!(
            disconnected_walkable_cells(&TileGrid::new(5, 1, 2, with_door).unwrap(), &city),
            vec![(3, 0, 1)],
            "a door does not lead up through a solid building"
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
                        let framed = |z: usize| matches!(model.get(x, y, z), Some(palette::WALL | palette::WINDOW));
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
            assert_eq!(filled == 0, tile == city.air || headroom, "{}", city.modules.names[tile]);
        }
    }
}
