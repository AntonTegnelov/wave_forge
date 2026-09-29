//! A module set proposed from a kit's meshes: connectors from the shapes of the modules' faces.
//!
//! An artist's building kit has a mesh per module and no rule file. As marian42's city generator
//! does, the vertices of a module's mesh that lie on one of its cell's faces outline what that face
//! looks like, and two faces whose outlines match get the same connector, so modules whose meshes
//! meet without a seam may touch there ([`crate::modules`]). [`propose`] writes the proposal as a
//! module set file ([`crate::formats::module_format`]) for the artist to confirm, rename and mark
//! walkable, which geometry cannot say.
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
    let steps = 1.0 / tolerance;
    assert!(
        tolerance > 0.0 && (steps - steps.round()).abs() < 0.01,
        "a tolerance of {tolerance} divides a cell into whole steps"
    );
    let steps = steps.round() as i64;
    let mut sides = Connectors::default();
    let mut tops = Connectors::default();
    let mut modules_text = String::new();
    for module in modules {
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
        let side_faces = [
            outline(0, 1.0, |p| [p[1], p[2]]),
            outline(0, 0.0, |p| [1.0 - p[1], p[2]]),
            outline(1, 1.0, |p| [1.0 - p[0], p[2]]),
            outline(1, 0.0, |p| [p[0], p[2]]),
        ]
        .map(|face| sides.side(&face, steps));
        let up = tops.top(&outline(2, 1.0, |p| [p[0], p[1]]), steps);
        let down = tops.top(&outline(2, 0.0, |p| [p[0], p[1]]), steps);
        writeln!(
            modules_text,
            "        (name: {:?}, sides: [{:?}, {:?}, {:?}, {:?}], up: {up:?}, down: {down:?}),",
            module.name, side_faces[0], side_faces[1], side_faces[2], side_faces[3],
        )
        .expect("writing to a string");
    }
    let mut text = String::from("(\n    faces: {\n");
    for (name, definition) in sides.faces.iter().chain(&tops.faces) {
        writeln!(text, "        {name:?}: {definition},").expect("writing to a string");
    }
    text.push_str("    },\n    modules: [\n");
    text.push_str(&modules_text);
    text.push_str("    ],\n)\n");
    text
}

/// The connectors found so far, one per outline up to mirroring or turning, numbered in the order
/// they were found, and the faces named from them.
#[derive(Default)]
struct Connectors {
    by_outline: BTreeMap<Vec<(i64, i64)>, usize>,
    /// Each face's name and its definition in the file.
    faces: BTreeMap<String, String>,
}

impl Connectors {
    /// The number of the connector of `canonical`, the first outline of its kind.
    fn number(&mut self, canonical: &Outline) -> usize {
        let next = self.by_outline.len();
        *self
            .by_outline
            .entry(canonical.iter().copied().collect())
            .or_insert(next)
    }

    /// The name of the side face with `outline`, seen from outside a cell of `steps` a side,
    /// defined once.
    fn side(&mut self, outline: &Outline, steps: i64) -> String {
        if outline.is_empty() {
            self.faces.insert(
                "empty side".to_owned(),
                "Side(connector: \"empty\")".to_owned(),
            );
            return "empty side".to_owned();
        }
        let mirror: Outline = outline.iter().map(|&(u, v)| (steps - u, v)).collect();
        let canonical = outline.clone().min(mirror.clone());
        let connector = format!("side {}", self.number(&canonical));
        let (name, symmetry) = if mirror == *outline {
            (connector.clone(), "Symmetric")
        } else if *outline == canonical {
            (format!("{connector} plain"), "Plain")
        } else {
            (format!("{connector} flipped"), "Flipped")
        };
        self.faces.insert(
            name.clone(),
            format!("Side(connector: {connector:?}, symmetry: {symmetry})"),
        );
        name
    }

    /// The name of the top or bottom face with `outline`, seen from above in a cell of `steps` a
    /// side, defined once.
    fn top(&mut self, outline: &Outline, steps: i64) -> String {
        if outline.is_empty() {
            self.faces.insert(
                "empty top".to_owned(),
                "Top(connector: \"empty\")".to_owned(),
            );
            return "empty top".to_owned();
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
        let connector = format!("top {}", self.number(&canonical));
        let (name, definition) = if turns.iter().all(|turned| *turned == canonical) {
            (connector.clone(), format!("Top(connector: {connector:?})"))
        } else {
            // The quarter turns that take the canonical outline to this one.
            let mut from = canonical.clone();
            let mut rotation = 0;
            while from != *outline {
                from = turn(&from);
                rotation += 1;
            }
            (
                format!("{connector} r{rotation}"),
                format!("Top(connector: {connector:?}, rotation: Some({rotation}))"),
            )
        };
        self.faces.insert(name.clone(), definition);
        name
    }
}

#[cfg(test)]
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
