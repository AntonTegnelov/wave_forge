//! Connector-based module sets: describe what each tile's faces look like and derive adjacency.
//!
//! Writing adjacency tuples by hand stops scaling once a tile set has dozens of pieces that can
//! each be rotated, as any structured 3D set (streets, buildings, stairs, roofs) quickly does.
//! Following marian42's city generator, every module prototype labels its six faces with a
//! *connector*; two modules may touch where their touching faces carry the same connector. Rotated
//! variants are generated automatically, identical rotations are removed, and the result compiles
//! to the plain [`TileSet`] and [`AdjacencyRules`] the solver works with.
//!
//! Coordinates follow the rest of the crate with `+z` up. Axis indices match [`AdjacencyRules`]:
//! `0 = +x`, `1 = -x`, `2 = +y`, `3 = -y`, `4 = +z` (up), `5 = -z` (down). Rotations are quarter
//! turns counter-clockwise around `+z`.

use crate::types::{AdjacencyRules, TileSet, TileSetError, Transformation};
use std::collections::{BTreeSet, HashMap};
use thiserror::Error;

/// Axis index of `+x`.
pub const POS_X: usize = 0;
/// Axis index of `-x`.
pub const NEG_X: usize = 1;
/// Axis index of `+y`.
pub const POS_Y: usize = 2;
/// Axis index of `-y`.
pub const NEG_Y: usize = 3;
/// Axis index of `+z` (up).
pub const UP: usize = 4;
/// Axis index of `-z` (down).
pub const DOWN: usize = 5;
/// Number of face directions.
pub const NUM_AXES: usize = 6;

/// Horizontal directions in counter-clockwise order around `+z`: rotating a module by one quarter
/// turn moves the face at each entry to the next entry.
const HORIZONTAL_CCW: [usize; 4] = [POS_X, POS_Y, NEG_X, NEG_Y];

/// The direction opposite `axis`. Axes are stored in pairs, so only the lowest bit differs.
#[must_use]
pub const fn opposite(axis: usize) -> usize {
    axis ^ 1
}

/// Where `axis` points after rotating by `quarter_turns` counter-clockwise around `+z`.
#[must_use]
pub fn rotate_axis(axis: usize, quarter_turns: u8) -> usize {
    match HORIZONTAL_CCW.iter().position(|&a| a == axis) {
        Some(index) => HORIZONTAL_CCW[(index + quarter_turns as usize) % 4],
        None => axis,
    }
}

/// How a horizontal face's profile relates to its mirror image.
///
/// Profiles are drawn looking at the face from outside the module. Two faces that touch see each
/// other mirrored, so an asymmetric profile only lines up with its mirror image: a `Plain` face fits
/// a `Flipped` face with the same connector, and never another `Plain` one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Symmetry {
    /// The profile is its own mirror image, so it fits other symmetric faces with the same connector.
    Symmetric,
    /// An asymmetric profile.
    Plain,
    /// The mirror image of the `Plain` profile with the same connector.
    Flipped,
}

/// A face pointing along `±x` or `±y`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HorizontalFace {
    /// Connector id; only faces with the same id can touch.
    pub connector: u32,
    /// Mirror symmetry of the profile.
    pub symmetry: Symmetry,
    /// Whether someone can walk across this face (marian42's `Walkable`).
    pub walkable: bool,
    /// Whether the face across must be walkable too (marian42's `EnforceWalkableNeighbor`), so a
    /// path through this face can never end at a wall or in mid-air.
    pub enforce_walkable_neighbor: bool,
}

impl HorizontalFace {
    /// A symmetric face with this connector.
    #[must_use]
    pub const fn symmetric(connector: u32) -> Self {
        Self {
            connector,
            symmetry: Symmetry::Symmetric,
            walkable: false,
            enforce_walkable_neighbor: false,
        }
    }

    /// An asymmetric face with this connector.
    #[must_use]
    pub const fn plain(connector: u32) -> Self {
        Self {
            connector,
            symmetry: Symmetry::Plain,
            walkable: false,
            enforce_walkable_neighbor: false,
        }
    }

    /// The mirror image of [`HorizontalFace::plain`] with this connector.
    #[must_use]
    pub const fn flipped(connector: u32) -> Self {
        Self {
            connector,
            symmetry: Symmetry::Flipped,
            walkable: false,
            enforce_walkable_neighbor: false,
        }
    }

    /// Marks the face walkable.
    #[must_use]
    pub const fn walkable(mut self) -> Self {
        self.walkable = true;
        self
    }

    /// Marks the face walkable and requires the face across to be walkable as well.
    #[must_use]
    pub const fn path(mut self) -> Self {
        self.walkable = true;
        self.enforce_walkable_neighbor = true;
        self
    }

    fn fits(self, other: Self) -> bool {
        self.connector == other.connector
            && matches!(
                (self.symmetry, other.symmetry),
                (Symmetry::Symmetric, Symmetry::Symmetric)
                    | (Symmetry::Plain, Symmetry::Flipped)
                    | (Symmetry::Flipped, Symmetry::Plain)
            )
    }
}

/// A face pointing up or down.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VerticalFace {
    /// Connector id; only faces with the same id can touch.
    pub connector: u32,
    /// `None` if the face looks the same under every rotation; otherwise its orientation in quarter
    /// turns, which must line up with the touching face after both modules are rotated.
    pub rotation: Option<u8>,
}

impl VerticalFace {
    /// A face that looks the same under every rotation.
    #[must_use]
    pub const fn invariant(connector: u32) -> Self {
        Self {
            connector,
            rotation: None,
        }
    }

    /// A face with an orientation (`quarter_turns` modulo 4).
    #[must_use]
    pub const fn oriented(connector: u32, quarter_turns: u8) -> Self {
        Self {
            connector,
            rotation: Some(quarter_turns % 4),
        }
    }
}

/// One of a module's six faces, as stored by [`ModulePrototype`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Face {
    /// A face along `±x` or `±y`.
    Horizontal(HorizontalFace),
    /// A face along `±z`.
    Vertical(VerticalFace),
}

/// A hand-authored module before rotation.
#[derive(Debug, Clone)]
pub struct ModulePrototype {
    /// Unique name; rotated variants are named `"{name} r{quarter_turns}"`.
    pub name: String,
    /// Relative selection weight of each variant.
    pub weight: f32,
    /// Faces along `+x`, `-x`, `+y`, `-y`.
    pub horizontal: [HorizontalFace; 4],
    /// Face along `+z`.
    pub up: VerticalFace,
    /// Face along `-z`.
    pub down: VerticalFace,
    /// Whether rotated variants are generated.
    pub rotatable: bool,
    /// Prototypes that must not touch this one across a face given in the prototype's own frame.
    pub excluded: Vec<(usize, String)>,
    /// Free-form labels for tests, renderers and tools (for example `"building"` or `"walkable"`).
    pub tags: BTreeSet<String>,
    /// What walkers in the module's cell stand on, for footsteps: a name the game maps to sounds or
    /// effects.
    pub surface: Option<String>,
    /// Whether the module's cell is inside, out of the weather and in a room's acoustics.
    pub indoor: bool,
    /// The sounds the module makes, in its own frame.
    pub sounds: Vec<ModuleSound>,
}

/// A sound a module makes, and where in its cell it plays.
#[derive(Debug, Clone, PartialEq)]
pub struct ModuleSound {
    /// Where in the cell, from 0 to 1 along x, y and z.
    pub at: [f32; 3],
    /// What plays there: a name the game maps to a sound.
    pub key: String,
}

impl ModulePrototype {
    /// A rotatable prototype with weight 1 and the given faces (`horizontal` is `+x, -x, +y, -y`).
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        horizontal: [HorizontalFace; 4],
        up: VerticalFace,
        down: VerticalFace,
    ) -> Self {
        Self {
            name: name.into(),
            weight: 1.0,
            horizontal,
            up,
            down,
            rotatable: true,
            excluded: Vec::new(),
            tags: BTreeSet::new(),
            surface: None,
            indoor: false,
            sounds: Vec::new(),
        }
    }

    /// Sets the selection weight.
    #[must_use]
    pub fn weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Disables rotated variants.
    #[must_use]
    pub fn fixed_orientation(mut self) -> Self {
        self.rotatable = false;
        self
    }

    /// Forbids `other` from touching this prototype across `axis` (in this prototype's frame).
    #[must_use]
    pub fn exclude(mut self, axis: usize, other: impl Into<String>) -> Self {
        self.excluded.push((axis, other.into()));
        self
    }

    /// Adds a tag.
    #[must_use]
    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.insert(tag.into());
        self
    }

    /// Sets what walkers in the cell stand on.
    #[must_use]
    pub fn surface(mut self, surface: impl Into<String>) -> Self {
        self.surface = Some(surface.into());
        self
    }

    /// Marks the cell as inside.
    #[must_use]
    pub fn indoor(mut self) -> Self {
        self.indoor = true;
        self
    }

    /// Adds a sound playing `at` a point of the cell, from 0 to 1 along each axis in the
    /// prototype's own frame.
    #[must_use]
    pub fn sound(mut self, at: [f32; 3], key: impl Into<String>) -> Self {
        self.sounds.push(ModuleSound {
            at,
            key: key.into(),
        });
        self
    }

    /// The face along `axis` in the prototype's own frame.
    #[must_use]
    pub fn face(&self, axis: usize) -> Face {
        match axis {
            POS_X => Face::Horizontal(self.horizontal[0]),
            NEG_X => Face::Horizontal(self.horizontal[1]),
            POS_Y => Face::Horizontal(self.horizontal[2]),
            NEG_Y => Face::Horizontal(self.horizontal[3]),
            UP => Face::Vertical(self.up),
            DOWN => Face::Vertical(self.down),
            _ => panic!("invalid axis {axis}"),
        }
    }
}

/// A prototype placed with a rotation; one tile variant of the compiled set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModuleVariant {
    /// Index into [`CompiledModules::prototypes`].
    pub prototype: usize,
    /// Quarter turns counter-clockwise around `+z`.
    pub rotation: u8,
}

/// Errors in a module set description.
#[derive(Debug, Error)]
pub enum ModuleError {
    /// Two prototypes share a name.
    #[error("duplicate module prototype name {0:?}")]
    DuplicateName(String),
    /// An exclusion refers to a prototype that does not exist.
    #[error("module {module:?} excludes unknown module {excluded:?}")]
    UnknownExclusion {
        /// The excluding prototype.
        module: String,
        /// The unknown name.
        excluded: String,
    },
    /// The set has no prototypes.
    #[error("module set is empty")]
    Empty,
    /// The resulting tile set is invalid (for example a non-positive weight).
    #[error(transparent)]
    TileSet(#[from] TileSetError),
}

/// A module set compiled into solver input.
#[derive(Debug, Clone)]
pub struct CompiledModules {
    /// The prototypes, in the order given.
    pub prototypes: Vec<ModulePrototype>,
    /// Tile variants; the tile id used by the solver is the index into this list.
    pub variants: Vec<ModuleVariant>,
    /// Variant names, parallel to `variants`.
    pub names: Vec<String>,
    /// One base tile per variant (identity transformation only), weighted by its prototype.
    pub tileset: TileSet,
    /// Allowed adjacencies between variants.
    pub rules: AdjacencyRules,
    /// Faces for which no variant in the set fits, as `(variant, axis)`. Usually an authoring
    /// mistake, but expected at grid boundaries (for example the underside of the ground).
    pub unmatched_faces: Vec<(usize, usize)>,
    /// Connector names by id, for a set whose connectors were named (every set read from a file);
    /// empty for one built in code with bare ids.
    pub connector_names: Vec<String>,
}

impl CompiledModules {
    /// The id of the connector named `name`, if the set named its connectors and has this one.
    #[must_use]
    pub fn connector(&self, name: &str) -> Option<u32> {
        self.connector_names
            .iter()
            .position(|known| known == name)
            .map(|index| u32::try_from(index).expect("fewer than 2^32 connectors"))
    }

    /// Tile ids of all variants of the prototype named `name`.
    #[must_use]
    pub fn variants_of(&self, name: &str) -> Vec<usize> {
        self.variants
            .iter()
            .enumerate()
            .filter(|(_, v)| self.prototypes[v.prototype].name == name)
            .map(|(tile, _)| tile)
            .collect()
    }

    /// Tile ids of all variants whose prototype carries `tag`.
    #[must_use]
    pub fn variants_tagged(&self, tag: &str) -> Vec<usize> {
        self.variants
            .iter()
            .enumerate()
            .filter(|(_, v)| self.prototypes[v.prototype].tags.contains(tag))
            .map(|(tile, _)| tile)
            .collect()
    }

    /// The prototype a tile id was generated from.
    #[must_use]
    pub fn prototype_of(&self, tile: usize) -> &ModulePrototype {
        &self.prototypes[self.variants[tile].prototype]
    }

    /// The face of tile `tile` along `axis`, after its rotation.
    #[must_use]
    pub fn face(&self, tile: usize, axis: usize) -> Face {
        variant_face(&self.prototypes, self.variants[tile], axis)
    }

    /// The sounds of tile `tile`, turned with it about the cell's vertical axis.
    #[must_use]
    pub fn sounds(&self, tile: usize) -> Vec<ModuleSound> {
        let turns = self.variants[tile].rotation;
        self.prototype_of(tile)
            .sounds
            .iter()
            .map(|sound| {
                let [mut x, mut y, z] = sound.at;
                // A quarter turn counter-clockwise about the cell's centre takes (x, y) to
                // (1 - y, x), as it takes the +x face to +y.
                for _ in 0..turns {
                    (x, y) = (1.0 - y, x);
                }
                ModuleSound {
                    at: [x, y, z],
                    key: sound.key.clone(),
                }
            })
            .collect()
    }
}

fn variant_face(prototypes: &[ModulePrototype], variant: ModuleVariant, axis: usize) -> Face {
    let prototype = &prototypes[variant.prototype];
    match prototype.face(rotate_axis(axis, (4 - variant.rotation) % 4)) {
        Face::Horizontal(face) => Face::Horizontal(face),
        Face::Vertical(face) => Face::Vertical(VerticalFace {
            connector: face.connector,
            rotation: face.rotation.map(|r| (r + variant.rotation) % 4),
        }),
    }
}

fn faces_fit(a: Face, b: Face, compatible: &BTreeSet<(u32, u32)>) -> bool {
    let declared_pair = match (a, b) {
        (Face::Horizontal(a), Face::Horizontal(b)) => (a.connector, b.connector),
        (Face::Vertical(a), Face::Vertical(b)) => (a.connector, b.connector),
        _ => return false,
    };
    if compatible.contains(&declared_pair) {
        return true;
    }
    match (a, b) {
        (Face::Horizontal(a), Face::Horizontal(b)) => a.fits(b),
        (Face::Vertical(a), Face::Vertical(b)) => {
            a.connector == b.connector
                && match (a.rotation, b.rotation) {
                    (None, None) => true,
                    (Some(ra), Some(rb)) => ra == rb,
                    _ => false,
                }
        }
        _ => false,
    }
}

/// A collection of prototypes to compile.
#[derive(Debug, Clone, Default)]
pub struct ModuleSet {
    prototypes: Vec<ModulePrototype>,
    /// Pairs of different connectors that may face each other, stored in both orders.
    compatible: BTreeSet<(u32, u32)>,
    /// Connector names by id, when the set has them.
    connector_names: Vec<String>,
}

impl ModuleSet {
    /// An empty set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Lets faces with two different connectors touch, whatever their symmetry or rotation.
    ///
    /// Modules here are centred on grid cells, so a face is shared by two cells that can hold
    /// different materials: a building's facade borders open air or ground. marian42's modules
    /// sit on grid corners instead and describe such boundaries with one connector; with
    /// cell-centred modules, declaring which connectors may meet keeps the set small instead of
    /// needing a "ground next to a building" variant of every ground module.
    ///
    /// # Panics
    ///
    /// Panics if `a == b`; equal connectors already fit according to their symmetry.
    #[must_use]
    pub fn connect(mut self, a: u32, b: u32) -> Self {
        assert_ne!(
            a, b,
            "connector {a} already fits itself; connect() pairs different connectors"
        );
        self.compatible.insert((a, b));
        self.compatible.insert((b, a));
        self
    }

    /// Names the connectors, by id, so tools can refer to them by name.
    #[must_use]
    pub fn with_connector_names(mut self, names: Vec<String>) -> Self {
        self.connector_names = names;
        self
    }

    /// Adds a prototype.
    #[must_use]
    pub fn with(mut self, prototype: ModulePrototype) -> Self {
        self.prototypes.push(prototype);
        self
    }

    /// Expands rotations, derives adjacency from connectors and exclusions, and builds solver input.
    ///
    /// # Errors
    ///
    /// Returns [`ModuleError`] for duplicate names, unknown exclusions, an empty set or an invalid
    /// tile set.
    pub fn compile(&self) -> Result<CompiledModules, ModuleError> {
        if self.prototypes.is_empty() {
            return Err(ModuleError::Empty);
        }
        let mut index_by_name = HashMap::new();
        for (index, prototype) in self.prototypes.iter().enumerate() {
            if index_by_name
                .insert(prototype.name.clone(), index)
                .is_some()
            {
                return Err(ModuleError::DuplicateName(prototype.name.clone()));
            }
        }
        for prototype in &self.prototypes {
            for (_, excluded) in &prototype.excluded {
                if !index_by_name.contains_key(excluded) {
                    return Err(ModuleError::UnknownExclusion {
                        module: prototype.name.clone(),
                        excluded: excluded.clone(),
                    });
                }
            }
        }

        let variants = self.expand_rotations();
        let names: Vec<String> = variants
            .iter()
            .map(|v| format!("{} r{}", self.prototypes[v.prototype].name, v.rotation))
            .collect();

        let mut tuples = Vec::new();
        for (a, &va) in variants.iter().enumerate() {
            for (b, &vb) in variants.iter().enumerate() {
                for axis in 0..NUM_AXES {
                    if self.may_touch(va, vb, axis, &index_by_name) {
                        tuples.push((axis, a, b));
                    }
                }
            }
        }

        let mut unmatched_faces = Vec::new();
        for a in 0..variants.len() {
            for axis in 0..NUM_AXES {
                if !tuples
                    .iter()
                    .any(|&(t_axis, t_a, _)| t_axis == axis && t_a == a)
                {
                    unmatched_faces.push((a, axis));
                }
            }
        }

        let weights: Vec<f32> = variants
            .iter()
            .map(|v| self.prototypes[v.prototype].weight)
            .collect();
        let tileset = TileSet::new(
            weights,
            vec![vec![Transformation::Identity]; variants.len()],
        )?;
        let rules = AdjacencyRules::from_allowed_tuples(variants.len(), NUM_AXES, tuples);

        Ok(CompiledModules {
            prototypes: self.prototypes.clone(),
            variants,
            names,
            tileset,
            rules,
            unmatched_faces,
            connector_names: self.connector_names.clone(),
        })
    }

    /// All distinct rotations of every prototype. A rotation is dropped when every face of it equals
    /// the corresponding face of a rotation already kept, so symmetric pieces do not get duplicate
    /// tiles (duplicates would silently multiply their effective weight and slow propagation).
    fn expand_rotations(&self) -> Vec<ModuleVariant> {
        let mut variants = Vec::new();
        for prototype in 0..self.prototypes.len() {
            let rotations = if self.prototypes[prototype].rotatable {
                4
            } else {
                1
            };
            let mut kept: Vec<ModuleVariant> = Vec::new();
            for rotation in 0..rotations {
                let candidate = ModuleVariant {
                    prototype,
                    rotation,
                };
                let duplicate = kept.iter().any(|&existing| {
                    (0..NUM_AXES).all(|axis| {
                        variant_face(&self.prototypes, existing, axis)
                            == variant_face(&self.prototypes, candidate, axis)
                    })
                });
                if !duplicate {
                    kept.push(candidate);
                }
            }
            variants.extend(kept);
        }
        variants
    }

    fn may_touch(
        &self,
        a: ModuleVariant,
        b: ModuleVariant,
        axis: usize,
        index_by_name: &HashMap<String, usize>,
    ) -> bool {
        if !faces_fit(
            variant_face(&self.prototypes, a, axis),
            variant_face(&self.prototypes, b, opposite(axis)),
            &self.compatible,
        ) {
            return false;
        }
        if let (Face::Horizontal(here), Face::Horizontal(there)) = (
            variant_face(&self.prototypes, a, axis),
            variant_face(&self.prototypes, b, opposite(axis)),
        ) {
            // Walkability is part of the rules, not a later check: a path may not run into a face
            // nobody can walk through.
            if (here.enforce_walkable_neighbor && !there.walkable)
                || (there.enforce_walkable_neighbor && !here.walkable)
            {
                return false;
            }
        }
        let excludes = |from: ModuleVariant, to: ModuleVariant, direction: usize| {
            let local_axis = rotate_axis(direction, (4 - from.rotation) % 4);
            self.prototypes[from.prototype]
                .excluded
                .iter()
                .any(|(excluded_axis, name)| {
                    *excluded_axis == local_axis && index_by_name[name] == to.prototype
                })
        };
        !excludes(a, b, axis) && !excludes(b, a, opposite(axis))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPEN: u32 = 0;
    const WALL: u32 = 1;
    const RAMP: u32 = 2;

    fn uniform(name: &str, connector: u32) -> ModulePrototype {
        ModulePrototype::new(
            name,
            [HorizontalFace::symmetric(connector); 4],
            VerticalFace::invariant(connector),
            VerticalFace::invariant(connector),
        )
    }

    #[test]
    fn rotations_rotate_horizontal_axes_counter_clockwise() {
        assert_eq!(rotate_axis(POS_X, 1), POS_Y);
        assert_eq!(rotate_axis(POS_Y, 1), NEG_X);
        assert_eq!(rotate_axis(NEG_Y, 1), POS_X);
        assert_eq!(rotate_axis(UP, 3), UP);
        for axis in 0..NUM_AXES {
            assert_eq!(rotate_axis(rotate_axis(axis, 1), 3), axis);
        }
    }

    #[test]
    fn symmetric_faces_fit_symmetric_faces_and_plain_fits_flipped() {
        assert!(HorizontalFace::symmetric(WALL).fits(HorizontalFace::symmetric(WALL)));
        assert!(HorizontalFace::plain(RAMP).fits(HorizontalFace::flipped(RAMP)));
        assert!(!HorizontalFace::plain(RAMP).fits(HorizontalFace::plain(RAMP)));
        assert!(!HorizontalFace::symmetric(RAMP).fits(HorizontalFace::plain(RAMP)));
        assert!(!HorizontalFace::symmetric(WALL).fits(HorizontalFace::symmetric(OPEN)));
    }

    #[test]
    fn fully_symmetric_prototypes_get_one_variant() {
        let compiled = ModuleSet::new()
            .with(uniform("air", OPEN))
            .compile()
            .unwrap();
        assert_eq!(compiled.names, vec!["air r0"]);
    }

    #[test]
    fn corners_get_four_variants_and_straights_two() {
        let corner = ModulePrototype::new(
            "corner",
            [
                HorizontalFace::symmetric(WALL),
                HorizontalFace::symmetric(OPEN),
                HorizontalFace::symmetric(WALL),
                HorizontalFace::symmetric(OPEN),
            ],
            VerticalFace::invariant(OPEN),
            VerticalFace::invariant(OPEN),
        );
        let straight = ModulePrototype::new(
            "straight",
            [
                HorizontalFace::symmetric(WALL),
                HorizontalFace::symmetric(WALL),
                HorizontalFace::symmetric(OPEN),
                HorizontalFace::symmetric(OPEN),
            ],
            VerticalFace::invariant(OPEN),
            VerticalFace::invariant(OPEN),
        );
        let compiled = ModuleSet::new()
            .with(corner)
            .with(straight)
            .compile()
            .unwrap();
        assert_eq!(compiled.variants_of("corner").len(), 4);
        assert_eq!(compiled.variants_of("straight").len(), 2);
    }

    #[test]
    fn a_rotated_variant_carries_its_faces_to_the_rotated_directions() {
        let marker = ModulePrototype::new(
            "marker",
            [
                HorizontalFace::symmetric(WALL),
                HorizontalFace::symmetric(OPEN),
                HorizontalFace::symmetric(OPEN),
                HorizontalFace::symmetric(OPEN),
            ],
            VerticalFace::invariant(OPEN),
            VerticalFace::invariant(OPEN),
        );
        let compiled = ModuleSet::new().with(marker).compile().unwrap();
        let r1 = compiled
            .names
            .iter()
            .position(|n| n == "marker r1")
            .unwrap();
        assert_eq!(
            compiled.face(r1, POS_Y),
            Face::Horizontal(HorizontalFace::symmetric(WALL))
        );
        assert_eq!(
            compiled.face(r1, POS_X),
            Face::Horizontal(HorizontalFace::symmetric(OPEN))
        );
    }

    #[test]
    fn walls_only_touch_walls_and_open_faces_only_open_faces() {
        let wall_east = ModulePrototype::new(
            "wall_east",
            [
                HorizontalFace::symmetric(WALL),
                HorizontalFace::symmetric(OPEN),
                HorizontalFace::symmetric(OPEN),
                HorizontalFace::symmetric(OPEN),
            ],
            VerticalFace::invariant(OPEN),
            VerticalFace::invariant(OPEN),
        )
        .fixed_orientation();
        let wall_west = ModulePrototype::new(
            "wall_west",
            [
                HorizontalFace::symmetric(OPEN),
                HorizontalFace::symmetric(WALL),
                HorizontalFace::symmetric(OPEN),
                HorizontalFace::symmetric(OPEN),
            ],
            VerticalFace::invariant(OPEN),
            VerticalFace::invariant(OPEN),
        )
        .fixed_orientation();
        let compiled = ModuleSet::new()
            .with(wall_east)
            .with(wall_west)
            .compile()
            .unwrap();
        let (east, west) = (
            compiled.variants_of("wall_east")[0],
            compiled.variants_of("wall_west")[0],
        );
        assert!(
            compiled.rules.check(east, west, POS_X),
            "east wall meets west wall across +x"
        );
        assert!(!compiled.rules.check(east, east, POS_X));
        assert!(
            compiled.rules.check(east, east, POS_Y),
            "open faces meet along y"
        );
    }

    #[test]
    fn oriented_vertical_faces_must_line_up_after_rotation() {
        let lower = ModulePrototype::new(
            "lower",
            [HorizontalFace::symmetric(OPEN); 4],
            VerticalFace::oriented(RAMP, 0),
            VerticalFace::invariant(OPEN),
        );
        let upper = ModulePrototype::new(
            "upper",
            [HorizontalFace::symmetric(OPEN); 4],
            VerticalFace::invariant(OPEN),
            VerticalFace::oriented(RAMP, 0),
        );
        let compiled = ModuleSet::new().with(lower).with(upper).compile().unwrap();
        for &l in &compiled.variants_of("lower") {
            for &u in &compiled.variants_of("upper") {
                let aligned = compiled.variants[l].rotation == compiled.variants[u].rotation;
                assert_eq!(
                    compiled.rules.check(l, u, UP),
                    aligned,
                    "{} on {}",
                    compiled.names[u],
                    compiled.names[l]
                );
            }
        }
    }

    #[test]
    fn exclusions_apply_in_both_directions_and_follow_rotation() {
        let a = uniform("a", OPEN).exclude(POS_X, "b");
        let b = uniform("b", OPEN);
        let compiled = ModuleSet::new().with(a).with(b).compile().unwrap();
        let (a, b) = (compiled.variants_of("a")[0], compiled.variants_of("b")[0]);
        assert!(!compiled.rules.check(a, b, POS_X));
        assert!(!compiled.rules.check(b, a, NEG_X));
        assert!(
            compiled.rules.check(a, b, POS_Y),
            "only the excluded face is affected"
        );
    }

    #[test]
    fn faces_without_a_partner_are_reported() {
        let ground = ModulePrototype::new(
            "ground",
            [HorizontalFace::symmetric(WALL); 4],
            VerticalFace::invariant(OPEN),
            VerticalFace::invariant(99),
        );
        let compiled = ModuleSet::new().with(ground).compile().unwrap();
        assert_eq!(compiled.unmatched_faces, vec![(0, UP), (0, DOWN)]);
    }

    #[test]
    fn connected_connectors_fit_each_other_but_not_third_parties() {
        let facade = uniform("facade", WALL);
        let air = uniform("air", OPEN);
        let ramp = uniform("ramp", RAMP);
        let compiled = ModuleSet::new()
            .connect(WALL, OPEN)
            .with(facade)
            .with(air)
            .with(ramp)
            .compile()
            .unwrap();
        let [facade, air, ramp] =
            ["facade", "air", "ramp"].map(|name| compiled.variants_of(name)[0]);
        for axis in 0..NUM_AXES {
            assert!(compiled.rules.check(facade, air, axis));
            assert!(compiled.rules.check(air, facade, axis));
            assert!(
                compiled.rules.check(facade, facade, axis),
                "equal connectors still fit"
            );
            assert!(!compiled.rules.check(facade, ramp, axis));
        }
    }

    #[test]
    fn paths_only_meet_walkable_faces() {
        let path = ModulePrototype::new(
            "path",
            [HorizontalFace::symmetric(OPEN).path(); 4],
            VerticalFace::invariant(OPEN),
            VerticalFace::invariant(OPEN),
        );
        let floor = ModulePrototype::new(
            "floor",
            [HorizontalFace::symmetric(OPEN).walkable(); 4],
            VerticalFace::invariant(OPEN),
            VerticalFace::invariant(OPEN),
        );
        let ledge = uniform("ledge", OPEN);
        let compiled = ModuleSet::new()
            .with(path)
            .with(floor)
            .with(ledge)
            .compile()
            .unwrap();
        let [path, floor, ledge] =
            ["path", "floor", "ledge"].map(|name| compiled.variants_of(name)[0]);
        assert!(compiled.rules.check(path, path, POS_X));
        assert!(
            compiled.rules.check(path, floor, POS_X),
            "walkable faces accept paths"
        );
        assert!(
            !compiled.rules.check(path, ledge, POS_X),
            "a path cannot run into a non-walkable face"
        );
        assert!(
            !compiled.rules.check(ledge, path, NEG_X),
            "in either direction"
        );
        assert!(
            compiled.rules.check(floor, ledge, POS_X),
            "walkable faces without enforcement are unconstrained"
        );
    }

    #[test]
    fn invalid_sets_are_rejected() {
        assert!(matches!(
            ModuleSet::new().compile(),
            Err(ModuleError::Empty)
        ));
        let duplicate = ModuleSet::new()
            .with(uniform("x", OPEN))
            .with(uniform("x", OPEN));
        assert!(matches!(
            duplicate.compile(),
            Err(ModuleError::DuplicateName(_))
        ));
        let unknown = ModuleSet::new().with(uniform("x", OPEN).exclude(UP, "missing"));
        assert!(matches!(
            unknown.compile(),
            Err(ModuleError::UnknownExclusion { .. })
        ));
        let bad_weight = ModuleSet::new().with(uniform("x", OPEN).weight(0.0));
        assert!(matches!(bad_weight.compile(), Err(ModuleError::TileSet(_))));
    }

    #[test]
    fn compiled_tileset_weights_follow_prototypes() {
        let compiled = ModuleSet::new()
            .with(uniform("light", OPEN).weight(0.5))
            .with(uniform("heavy", OPEN).weight(3.0))
            .compile()
            .unwrap();
        assert_eq!(compiled.tileset.weights, vec![0.5, 3.0]);
        assert_eq!(compiled.tileset.num_transformed_tiles(), 2);
    }
}
