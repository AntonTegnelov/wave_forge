//! A module set proposed from a kit's meshes: connectors from the shapes of the modules' faces.
//!
//! An artist's building kit has a mesh per module and no rule file. As marian42's city generator
//! does, the vertices of a module's mesh that lie on one of its cell's faces outline what that face
//! looks like, and two faces whose outlines match get the same connector, so modules whose meshes
//! meet without a seam may touch there ([`crate::modules`]). [`propose`] writes the proposal as a
//! module set file ([`crate::formats::module_format`]); [`connectors`] lists the connectors it found,
//! for the artist to confirm, rename and mark walkable, which geometry cannot say, and
//! [`propose_named`] writes it with their names.
//!
//! A side face's outline is drawn looking at it from outside its cell, so the faces of two modules
//! that touch see each other mirrored: an outline that is its own mirror image is `Symmetric`, and
//! an asymmetric one is `Plain` or `Flipped`, whichever of it and its mirror image comes first, so
//! a face fits the mirror of its own outline. A top or bottom face's outline is drawn from above;
//! one that looks the same under every quarter turn is invariant, and any other gets the quarter
//! turns that take the first of its four turns to it.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

/// One module of a kit: its name and its mesh's vertices in the lattice's frame, with `+z` up and
/// its cell spanning 0 to 1 along each axis.
#[derive(Clone, Copy, Debug)]
pub struct KitModule<'a> {
    pub name: &'a str,
    pub positions: &'a [[f32; 3]],
}

/// A connector a proposal found ([`connectors`]): the name the proposal gives it, `"side n"` or
/// `"top n"` numbered in the order found, whether it joins tops and bottoms rather than sides, and
/// the modules with a face of it, in the kit's order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProposedConnector {
    pub name: String,
    pub top: bool,
    pub modules: Vec<String>,
}

/// How an artist confirmed a proposal ([`propose_named`]): a name of their own for any of its
/// connectors, by the name the proposal gives it, and the side connectors walkers may cross, also
/// by the proposal's names.
#[derive(Clone, Debug, Default)]
pub struct Naming {
    pub names: BTreeMap<String, String>,
    pub walkable: BTreeSet<String>,
}

/// An outline: the points of a face, each coordinate in steps of the tolerance.
type Outline = BTreeSet<(i64, i64)>;

/// The module set `modules` propose, as a module set file: a face per distinct outline, named by
/// its connector, and every module built from its six faces, rotatable, with a weight of 1. A
/// vertex counts as lying on a face when it is within `tolerance` of the face's plane, and two
/// outlines match when their points agree to within the tolerance. A face no vertex lies on is
/// `"empty side"` or `"empty top"`.
///
/// # Panics
/// If `tolerance` is not positive or does not divide a cell into whole steps within a hundredth of
/// a step, which would make outlines round differently on opposite faces.
#[must_use]
pub fn propose(modules: &[KitModule<'_>], tolerance: f32) -> String {
    Proposal::new(modules, tolerance)
        .write(&Naming::default())
        .expect("the proposal's own names never clash")
}

/// The connectors the proposal of `modules` finds, as [`propose`] finds them, in the order it
/// numbers them: sides first, then tops. The empty faces' connector is not among them.
///
/// # Panics
/// As [`propose`] does.
#[must_use]
pub fn connectors(modules: &[KitModule<'_>], tolerance: f32) -> Vec<ProposedConnector> {
    let proposal = Proposal::new(modules, tolerance);
    proposal
        .connectors
        .iter()
        .enumerate()
        .map(|(number, (name, top))| ProposedConnector {
            name: name.clone(),
            top: *top,
            modules: proposal
                .modules
                .iter()
                .filter(|(_, faces)| faces.iter().any(|face| face.connector() == Some(number)))
                .map(|(module, _)| module.clone())
                .collect(),
        })
        .collect()
}

/// The module set `modules` propose, as [`propose`] writes it, with each connector `naming` names
/// under its new name, and walkable sides on the side connectors it lists as walkable.
///
/// # Errors
/// If `naming` names a connector the proposal does not find, gives an empty name, `"empty"` (the
/// empty faces' connector) or a name two connectors would share, lists a top connector as
/// walkable, or names connectors so that two faces would take one name; with what is wrong.
///
/// # Panics
/// As [`propose`] does.
pub fn propose_named(
    modules: &[KitModule<'_>],
    tolerance: f32,
    naming: &Naming,
) -> Result<String, String> {
    Proposal::new(modules, tolerance).write(naming)
}

/// How an asymmetric side face lies against the connector's first outline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Symmetry {
    Symmetric,
    Plain,
    Flipped,
}

/// A face of a proposed module: empty, or of a connector the proposal found, by its number, with
/// how it lies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Face {
    EmptySide,
    EmptyTop,
    Side {
        connector: usize,
        symmetry: Symmetry,
    },
    Top {
        connector: usize,
        rotation: Option<u8>,
    },
}

impl Face {
    const fn connector(self) -> Option<usize> {
        match self {
            Self::EmptySide | Self::EmptyTop => None,
            Self::Side { connector, .. } | Self::Top { connector, .. } => Some(connector),
        }
    }
}

/// What the meshes of a kit propose: the connectors found and each module's faces.
struct Proposal {
    /// Each connector's proposed name, and whether it joins tops, in the order found.
    connectors: Vec<(String, bool)>,
    /// Each module's name and its faces: `+x`, `-x`, `+y`, `-y`, up and down.
    modules: Vec<(String, [Face; 6])>,
}

impl Proposal {
    fn new(modules: &[KitModule<'_>], tolerance: f32) -> Self {
        let steps = 1.0 / tolerance;
        assert!(
            tolerance > 0.0 && (steps - steps.round()).abs() < 0.01,
            "a tolerance of {tolerance} divides a cell into whole steps"
        );
        let steps = steps.round() as i64;
        let mut found = Found::default();
        let modules = modules
            .iter()
            .map(|module| {
                let outline = |axis: usize, at: f32, point: fn([f32; 3]) -> [f32; 2]| -> Outline {
                    module
                        .positions
                        .iter()
                        .filter(|p| (p[axis] - at).abs() <= tolerance)
                        .map(|&p| {
                            let [u, v] = point(p);
                            (
                                (u / tolerance).round() as i64,
                                (v / tolerance).round() as i64,
                            )
                        })
                        .collect()
                };
                // Each side seen from outside its cell, left to right and upward.
                let faces = [
                    found.side(&outline(0, 1.0, |p| [p[1], p[2]]), steps),
                    found.side(&outline(0, 0.0, |p| [1.0 - p[1], p[2]]), steps),
                    found.side(&outline(1, 1.0, |p| [1.0 - p[0], p[2]]), steps),
                    found.side(&outline(1, 0.0, |p| [p[0], p[2]]), steps),
                    found.top(&outline(2, 1.0, |p| [p[0], p[1]]), steps),
                    found.top(&outline(2, 0.0, |p| [p[0], p[1]]), steps),
                ];
                (module.name.to_owned(), faces)
            })
            .collect::<Vec<_>>();
        // Sides first, then tops, each in the order found.
        let mut connectors: Vec<(usize, String, bool)> = found
            .sides
            .values()
            .map(|&number| (number, format!("side {number}"), false))
            .chain(
                found
                    .tops
                    .values()
                    .map(|&number| (number, format!("top {number}"), true)),
            )
            .collect();
        connectors.sort_by_key(|&(number, _, top)| (top, number));
        let sides = found.sides.len();
        let modules = modules
            .into_iter()
            .map(|(name, faces)| {
                // Tops are numbered after the sides in the one list.
                let faces = faces.map(|face| match face {
                    Face::Top {
                        connector,
                        rotation,
                    } => Face::Top {
                        connector: sides + connector,
                        rotation,
                    },
                    other => other,
                });
                (name, faces)
            })
            .collect();
        Self {
            connectors: connectors
                .into_iter()
                .map(|(_, name, top)| (name, top))
                .collect(),
            modules,
        }
    }

    /// The proposal as a module set file, with `naming`'s names and walkable sides.
    fn write(&self, naming: &Naming) -> Result<String, String> {
        let proposed: BTreeMap<&str, bool> = self
            .connectors
            .iter()
            .map(|(name, top)| (name.as_str(), *top))
            .collect();
        for (from, to) in &naming.names {
            if !proposed.contains_key(from.as_str()) {
                return Err(format!("the proposal has no connector {from:?}"));
            }
            if to.trim().is_empty() || to == "empty" {
                return Err(format!("{from:?} cannot be named {to:?}"));
            }
        }
        for name in &naming.walkable {
            match proposed.get(name.as_str()) {
                None => return Err(format!("the proposal has no connector {name:?}")),
                Some(true) => {
                    return Err(format!("{name:?} joins tops, which walkers do not cross"));
                }
                Some(false) => {}
            }
        }
        let shown: Vec<&str> = self
            .connectors
            .iter()
            .map(|(name, _)| naming.names.get(name).map_or(name.as_str(), String::as_str))
            .collect();
        let mut distinct = BTreeSet::new();
        for name in &shown {
            if !distinct.insert(*name) {
                return Err(format!("two connectors would be named {name:?}"));
            }
        }
        let mut side_faces: BTreeMap<String, String> = BTreeMap::new();
        let mut top_faces: BTreeMap<String, String> = BTreeMap::new();
        let mut name_of = |face: Face| -> Result<String, String> {
            let (faces, name, definition) = match face {
                Face::EmptySide => (
                    &mut side_faces,
                    "empty side".to_owned(),
                    "Side(connector: \"empty\")".to_owned(),
                ),
                Face::EmptyTop => (
                    &mut top_faces,
                    "empty top".to_owned(),
                    "Top(connector: \"empty\")".to_owned(),
                ),
                Face::Side {
                    connector,
                    symmetry,
                } => {
                    let shown = shown[connector];
                    let walkable = if naming.walkable.contains(&self.connectors[connector].0) {
                        ", walkable: true"
                    } else {
                        ""
                    };
                    let (name, symmetry) = match symmetry {
                        Symmetry::Symmetric => (shown.to_owned(), "Symmetric"),
                        Symmetry::Plain => (format!("{shown} plain"), "Plain"),
                        Symmetry::Flipped => (format!("{shown} flipped"), "Flipped"),
                    };
                    let definition =
                        format!("Side(connector: {shown:?}, symmetry: {symmetry}{walkable})");
                    (&mut side_faces, name, definition)
                }
                Face::Top {
                    connector,
                    rotation: None,
                } => {
                    let shown = shown[connector];
                    (
                        &mut top_faces,
                        shown.to_owned(),
                        format!("Top(connector: {shown:?})"),
                    )
                }
                Face::Top {
                    connector,
                    rotation: Some(rotation),
                } => {
                    let shown = shown[connector];
                    (
                        &mut top_faces,
                        format!("{shown} r{rotation}"),
                        format!("Top(connector: {shown:?}, rotation: Some({rotation}))"),
                    )
                }
            };
            match faces.get(&name) {
                Some(kept) if *kept != definition => {
                    Err(format!("two faces would be named {name:?}"))
                }
                _ => {
                    faces.insert(name.clone(), definition);
                    Ok(name)
                }
            }
        };
        let mut modules_text = String::new();
        for (module, faces) in &self.modules {
            let [px, nx, py, ny, up, down] = faces.map(&mut name_of);
            let (px, nx, py, ny, up, down) = (px?, nx?, py?, ny?, up?, down?);
            writeln!(
                modules_text,
                "        (name: {module:?}, sides: [{px:?}, {nx:?}, {py:?}, {ny:?}], up: {up:?}, down: {down:?}),",
            )
            .expect("writing to a string");
        }
        // Names clash only across sides and tops if a side and a top share one.
        if let Some(name) = side_faces.keys().find(|name| top_faces.contains_key(*name)) {
            return Err(format!("two faces would be named {name:?}"));
        }
        let mut text = String::from("(\n    faces: {\n");
        for (name, definition) in side_faces.iter().chain(&top_faces) {
            writeln!(text, "        {name:?}: {definition},").expect("writing to a string");
        }
        text.push_str("    },\n    modules: [\n");
        text.push_str(&modules_text);
        text.push_str("    ],\n)\n");
        Ok(text)
    }
}

/// The connectors found so far, one per outline up to mirroring or turning, sides and tops each
/// numbered in the order they were found.
#[derive(Default)]
struct Found {
    sides: BTreeMap<Vec<(i64, i64)>, usize>,
    tops: BTreeMap<Vec<(i64, i64)>, usize>,
}

impl Found {
    /// The number of the connector of `canonical` among `found`, the first outline of its kind.
    fn number(found: &mut BTreeMap<Vec<(i64, i64)>, usize>, canonical: &Outline) -> usize {
        let next = found.len();
        *found
            .entry(canonical.iter().copied().collect())
            .or_insert(next)
    }

    /// The side face with `outline`, seen from outside a cell of `steps` a side.
    fn side(&mut self, outline: &Outline, steps: i64) -> Face {
        if outline.is_empty() {
            return Face::EmptySide;
        }
        let mirror: Outline = outline.iter().map(|&(u, v)| (steps - u, v)).collect();
        let canonical = outline.clone().min(mirror.clone());
        let connector = Self::number(&mut self.sides, &canonical);
        let symmetry = if mirror == *outline {
            Symmetry::Symmetric
        } else if *outline == canonical {
            Symmetry::Plain
        } else {
            Symmetry::Flipped
        };
        Face::Side {
            connector,
            symmetry,
        }
    }

    /// The top or bottom face with `outline`, seen from above in a cell of `steps` a side; its
    /// connector is numbered among the tops.
    fn top(&mut self, outline: &Outline, steps: i64) -> Face {
        if outline.is_empty() {
            return Face::EmptyTop;
        }
        // A quarter turn counter-clockwise about +z, as a module's variants turn.
        let turn = |outline: &Outline| -> Outline {
            outline.iter().map(|&(x, y)| (steps - y, x)).collect()
        };
        let mut turns = vec![outline.clone()];
        for _ in 1..4 {
            let next = turn(&turns[turns.len() - 1]);
            turns.push(next);
        }
        let canonical = turns.iter().min().expect("four turns").clone();
        let connector = Self::number(&mut self.tops, &canonical);
        let rotation = if turns.iter().all(|turned| *turned == canonical) {
            None
        } else {
            // The quarter turns that take the canonical outline to this one.
            let mut from = canonical.clone();
            let mut rotation = 0;
            while from != *outline {
                from = turn(&from);
                rotation += 1;
            }
            Some(rotation)
        };
        Face::Top {
            connector,
            rotation,
        }
    }
}

// The tests read the proposal back as a rule file, which needs the `serde` feature.
#[cfg(all(test, feature = "serde"))]
mod tests {
    use super::*;
    use crate::loader::{RuleFile, parse_rule_file};

    /// The corners of the box from `low` to `high`, in the cell's frame.
    fn cuboid(low: [f32; 3], high: [f32; 3]) -> Vec<[f32; 3]> {
        (0..8)
            .map(|corner| {
                std::array::from_fn(|axis| {
                    if corner >> axis & 1 == 1 {
                        high[axis]
                    } else {
                        low[axis]
                    }
                })
            })
            .collect()
    }

    /// The module set a proposal writes, compiled.
    fn compiled(modules: &[KitModule<'_>]) -> crate::modules::CompiledModules {
        match parse_rule_file(&propose(modules, 0.125)).expect("a valid module set") {
            RuleFile::Modules(modules) => modules,
            RuleFile::Tiles { .. } => panic!("a module set"),
        }
    }

    #[test]
    fn the_connectors_found_are_listed_with_the_modules_that_have_them() {
        let block = cuboid([0.0; 3], [1.0; 3]);
        let slab = cuboid([0.0; 3], [1.0, 1.0, 0.5]);
        let kit = [
            KitModule {
                name: "block",
                positions: &block,
            },
            KitModule {
                name: "slab",
                positions: &slab,
            },
        ];

        let found = connectors(&kit, 0.125);

        let listed: Vec<(&str, bool, Vec<&str>)> = found
            .iter()
            .map(|c| {
                let modules = c.modules.iter().map(String::as_str).collect();
                (c.name.as_str(), c.top, modules)
            })
            .collect();
        assert_eq!(
            listed,
            vec![
                ("side 0", false, vec!["block"]),
                ("side 1", false, vec!["slab"]),
                ("top 0", true, vec!["block", "slab"]),
            ]
        );
    }

    #[test]
    fn a_named_proposal_fits_as_the_proposal_does_under_the_artists_names() {
        let block = cuboid([0.0; 3], [1.0; 3]);
        let slab = cuboid([0.0; 3], [1.0, 1.0, 0.5]);
        let kit = [
            KitModule {
                name: "block",
                positions: &block,
            },
            KitModule {
                name: "slab",
                positions: &slab,
            },
        ];
        let naming = Naming {
            names: BTreeMap::from([
                ("side 0".to_owned(), "wall".to_owned()),
                ("side 1".to_owned(), "low wall".to_owned()),
            ]),
            walkable: BTreeSet::from(["side 1".to_owned()]),
        };

        let named = propose_named(&kit, 0.125, &naming).expect("a valid naming");

        assert!(named.contains(r#""wall": Side(connector: "wall", symmetry: Symmetric),"#));
        assert!(named.contains(
            r#""low wall": Side(connector: "low wall", symmetry: Symmetric, walkable: true),"#
        ));
        assert!(
            !named.contains("side 0") && !named.contains("side 1"),
            "{named}"
        );
        let RuleFile::Modules(modules) = parse_rule_file(&named).expect("a valid module set")
        else {
            panic!("a module set");
        };
        let (block, slab) = (
            modules.variants_of("block")[0],
            modules.variants_of("slab")[0],
        );
        assert!(modules.rules.check(block, block, crate::modules::POS_X));
        assert!(!modules.rules.check(block, slab, crate::modules::POS_X));
    }

    #[test]
    fn a_naming_that_would_lose_or_merge_connectors_is_refused() {
        let block = cuboid([0.0; 3], [1.0; 3]);
        let slab = cuboid([0.0; 3], [1.0, 1.0, 0.5]);
        let kit = [
            KitModule {
                name: "block",
                positions: &block,
            },
            KitModule {
                name: "slab",
                positions: &slab,
            },
        ];
        let named = |names: &[(&str, &str)], walkable: &[&str]| {
            propose_named(
                &kit,
                0.125,
                &Naming {
                    names: names
                        .iter()
                        .map(|&(from, to)| (from.to_owned(), to.to_owned()))
                        .collect(),
                    walkable: walkable.iter().map(|&name| name.to_owned()).collect(),
                },
            )
        };

        let refusals = [
            named(&[("side 9", "wall")], &[]),
            named(&[("side 0", "")], &[]),
            named(&[("side 0", "empty")], &[]),
            named(&[("side 0", "wall"), ("side 1", "wall")], &[]),
            named(&[("side 0", "side 1")], &[]),
            named(&[], &["top 0"]),
            named(&[], &["side 9"]),
        ];

        for refusal in refusals {
            assert!(refusal.is_err(), "{refusal:?}");
        }
    }

    #[test]
    fn a_block_filling_its_cell_fits_itself_on_every_side() {
        let block = cuboid([0.0; 3], [1.0; 3]);

        let modules = compiled(&[KitModule {
            name: "block",
            positions: &block,
        }]);

        let block = modules.variants_of("block")[0];
        for axis in 0..6 {
            assert!(modules.rules.check(block, block, axis), "along axis {axis}");
        }
    }

    #[test]
    fn a_wall_fits_its_mirror_image_and_not_itself_across_its_asymmetric_side() {
        // A wall standing along the cell's -y side, a quarter of the cell thick, whose +x side is
        // its end: asymmetric seen from outside.
        let wall = cuboid([0.0, 0.0, 0.0], [1.0, 0.25, 1.0]);

        let modules = compiled(&[KitModule {
            name: "wall",
            positions: &wall,
        }]);

        let unturned = modules.variants_of("wall")[0];
        // Two walls in a row along x meet end to end without a seam.
        assert!(
            modules
                .rules
                .check(unturned, unturned, crate::modules::POS_X)
        );
        // A wall turned half a turn stands along the other side, so its end does not meet.
        let turned = modules.variants_of("wall")[2];
        assert!(!modules.rules.check(unturned, turned, crate::modules::POS_X));
    }

    #[test]
    fn an_empty_face_fits_only_an_empty_face() {
        let block = cuboid([0.0; 3], [1.0; 3]);

        let modules = compiled(&[
            KitModule {
                name: "block",
                positions: &block,
            },
            KitModule {
                name: "air",
                positions: &[],
            },
        ]);

        let (block, air) = (
            modules.variants_of("block")[0],
            modules.variants_of("air")[0],
        );
        assert!(modules.rules.check(air, air, crate::modules::POS_X));
        assert!(!modules.rules.check(block, air, crate::modules::POS_X));
    }

    #[test]
    fn an_oriented_top_fits_a_bottom_turned_the_same_way() {
        // A ramp's footprint: a bar along +x on the top, and the same bar on the bottom of the
        // module above.
        let low = [
            cuboid([0.0, 0.0, 0.0], [1.0, 1.0, 0.5]),
            cuboid([0.0, 0.0, 0.5], [1.0, 0.25, 1.0]),
        ]
        .concat();
        let high = cuboid([0.0, 0.0, 0.0], [1.0, 0.25, 0.5]);

        let modules = compiled(&[
            KitModule {
                name: "low",
                positions: &low,
            },
            KitModule {
                name: "high",
                positions: &high,
            },
        ]);

        let low = modules.variants_of("low");
        let high = modules.variants_of("high");
        assert_eq!((low.len(), high.len()), (4, 4));
        for (turn, &below) in low.iter().enumerate() {
            for (other, &above) in high.iter().enumerate() {
                assert_eq!(
                    modules.rules.check(below, above, crate::modules::UP),
                    turn == other,
                    "low turned {turn}, high turned {other}"
                );
            }
        }
    }
}
