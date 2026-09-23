//! Module sets written as RON: faces with connectors, and modules made of faces.
//!
//! This is the file form of [`crate::modules`]. A set names the faces it uses once, then builds each
//! module from four side faces (`+x, -x, +y, -y`), a top and a bottom, so a set of dozens of rotatable
//! pieces stays readable. Connectors are named; two faces touch when their connectors fit, as
//! [`crate::modules`] describes. Rotated variants, adjacency and weights are derived when the set is
//! compiled.
//!
//! ```ron
//! (
//!     faces: {
//!         "air": Side(connector: "air"),
//!         "road": Side(connector: "road", walkable: true),
//!         "open": Top(connector: "open"),
//!     },
//!     connect: [],
//!     modules: [
//!         (name: "air", sides: ["air", "air", "air", "air"], up: "open", down: "open"),
//!         (name: "road", sides: ["road", "road", "air", "air"], up: "open", down: "open", weight: 2.0),
//!     ],
//! )
//! ```

use crate::LoadError;
use crate::modules::{
    HorizontalFace, ModulePrototype, ModuleSet, NEG_X, NEG_Y, POS_X, POS_Y, Symmetry, UP,
    VerticalFace,
};
use serde::Deserialize;
use std::collections::BTreeMap;

/// A whole module set file.
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct ModuleFile {
    /// The faces modules are built from, by name.
    faces: BTreeMap<String, FaceDef>,
    /// Pairs of different connectors that may face each other.
    #[serde(default)]
    connect: Vec<(String, String)>,
    modules: Vec<ModuleDef>,
}

/// One named face.
#[derive(Deserialize, Debug)]
enum FaceDef {
    /// A face along `±x` or `±y`.
    Side {
        connector: String,
        #[serde(default = "symmetric")]
        symmetry: SymmetryDef,
        #[serde(default)]
        walkable: bool,
        #[serde(default)]
        enforce_walkable_neighbor: bool,
    },
    /// A face along `±z`.
    Top {
        connector: String,
        /// Quarter turns, for a face that has an orientation; none for one that looks the same
        /// under every rotation.
        #[serde(default)]
        rotation: Option<u8>,
    },
}

#[derive(Deserialize, Debug, Clone, Copy)]
enum SymmetryDef {
    Symmetric,
    Plain,
    Flipped,
}

const fn symmetric() -> SymmetryDef {
    SymmetryDef::Symmetric
}

/// One module prototype.
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct ModuleDef {
    name: String,
    /// Faces along `+x`, `-x`, `+y`, `-y`.
    sides: Vec<String>,
    up: String,
    down: String,
    #[serde(default = "one")]
    weight: f32,
    #[serde(default = "yes")]
    rotatable: bool,
    #[serde(default)]
    tags: Vec<String>,
    /// Modules that may not touch this one, as `(axis, module)` with the axis in this module's own
    /// frame, for example `("+x", "road_end")`.
    #[serde(default)]
    exclude: Vec<(String, String)>,
}

const fn one() -> f32 {
    1.0
}

const fn yes() -> bool {
    true
}

/// Parses a module set file.
///
/// # Errors
/// If the text is not valid RON of this form, or a module names a face, a connector pairing or an
/// axis that does not exist, or uses a top face as a side or the other way round.
pub fn parse_module_set(content: &str) -> Result<ModuleSet, LoadError> {
    let file: ModuleFile = ron::from_str(content)
        .map_err(|error| LoadError::ParseError(format!("module set: {error}")))?;

    // Connector ids are handed out as the faces are read, in the order of their names, so the same
    // file always gives the same ids.
    let mut connectors: Vec<String> = Vec::new();
    let mut connector_id = |name: &str| -> u32 {
        let index = connectors
            .iter()
            .position(|known| known == name)
            .unwrap_or_else(|| {
                connectors.push(name.to_owned());
                connectors.len() - 1
            });
        u32::try_from(index).expect("fewer than 2^32 connectors")
    };

    enum Built {
        Side(HorizontalFace),
        Top(VerticalFace),
    }
    let mut faces: BTreeMap<&str, Built> = BTreeMap::new();
    for (name, face) in &file.faces {
        let built = match face {
            FaceDef::Side {
                connector,
                symmetry,
                walkable,
                enforce_walkable_neighbor,
            } => Built::Side(HorizontalFace {
                connector: connector_id(connector),
                symmetry: match symmetry {
                    SymmetryDef::Symmetric => Symmetry::Symmetric,
                    SymmetryDef::Plain => Symmetry::Plain,
                    SymmetryDef::Flipped => Symmetry::Flipped,
                },
                walkable: *walkable,
                enforce_walkable_neighbor: *enforce_walkable_neighbor,
            }),
            FaceDef::Top {
                connector,
                rotation,
            } => Built::Top(match rotation {
                Some(turns) => VerticalFace::oriented(connector_id(connector), *turns),
                None => VerticalFace::invariant(connector_id(connector)),
            }),
        };
        faces.insert(name, built);
    }

    let known = |name: &str| {
        connectors
            .iter()
            .position(|known| known == name)
            .map(|index| u32::try_from(index).expect("fewer than 2^32 connectors"))
            .ok_or_else(|| {
                LoadError::InvalidData(format!(
                    "connect names connector {name:?}, which no face uses"
                ))
            })
    };
    let mut set = ModuleSet::new();
    for (a, b) in &file.connect {
        if a == b {
            return Err(LoadError::InvalidData(format!(
                "connect pairs connector {a:?} with itself; equal connectors already fit"
            )));
        }
        set = set.connect(known(a)?, known(b)?);
    }

    for module in &file.modules {
        let side = |face: &str| match faces.get(face) {
            Some(Built::Side(side)) => Ok(*side),
            Some(Built::Top(_)) => Err(LoadError::InvalidData(format!(
                "module {:?} uses top face {face:?} as a side",
                module.name
            ))),
            None => Err(LoadError::InvalidData(format!(
                "module {:?} uses face {face:?}, which the set does not define",
                module.name
            ))),
        };
        let top = |face: &str| match faces.get(face) {
            Some(Built::Top(top)) => Ok(*top),
            Some(Built::Side(_)) => Err(LoadError::InvalidData(format!(
                "module {:?} uses side face {face:?} as a top or bottom",
                module.name
            ))),
            None => Err(LoadError::InvalidData(format!(
                "module {:?} uses face {face:?}, which the set does not define",
                module.name
            ))),
        };
        if module.sides.len() != 4 {
            return Err(LoadError::InvalidData(format!(
                "module {:?} lists {} sides; it needs four, along +x, -x, +y and -y",
                module.name,
                module.sides.len()
            )));
        }
        let sides = [
            side(&module.sides[0])?,
            side(&module.sides[1])?,
            side(&module.sides[2])?,
            side(&module.sides[3])?,
        ];
        let mut prototype =
            ModulePrototype::new(&module.name, sides, top(&module.up)?, top(&module.down)?)
                .weight(module.weight);
        if !module.rotatable {
            prototype = prototype.fixed_orientation();
        }
        for tag in &module.tags {
            prototype = prototype.tag(tag);
        }
        for (axis, other) in &module.exclude {
            prototype = prototype.exclude(axis_index(axis, &module.name)?, other);
        }
        set = set.with(prototype);
    }
    Ok(set.with_connector_names(connectors))
}

fn axis_index(axis: &str, module: &str) -> Result<usize, LoadError> {
    match axis {
        "+x" => Ok(POS_X),
        "-x" => Ok(NEG_X),
        "+y" => Ok(POS_Y),
        "-y" => Ok(NEG_Y),
        "+z" => Ok(UP),
        "-z" => Ok(crate::modules::DOWN),
        _ => Err(LoadError::InvalidData(format!(
            "module {module:?} excludes along {axis:?}, which is not one of +x, -x, +y, -y, +z, -z"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROADS: &str = r#"(
        faces: {
            "air": Side(connector: "air"),
            "road": Side(connector: "road", walkable: true),
            "door": Side(connector: "door", symmetry: Plain, enforce_walkable_neighbor: true),
            "open": Top(connector: "open"),
            "stair": Top(connector: "stair", rotation: Some(0)),
        },
        connect: [("door", "road")],
        modules: [
            (name: "air", sides: ["air", "air", "air", "air"], up: "open", down: "open", weight: 4.0),
            (name: "road", sides: ["road", "road", "air", "air"], up: "open", down: "open", tags: ["road"]),
            (name: "stair", sides: ["air", "road", "air", "air"], up: "stair", down: "open",
             rotatable: false, exclude: [("+x", "road")]),
        ],
    )"#;

    #[test]
    fn a_module_file_compiles_to_rotated_variants() {
        let compiled = parse_module_set(ROADS)
            .expect("valid file")
            .compile()
            .expect("valid set");

        assert_eq!(compiled.variants_of("air").len(), 1, "air is symmetric");
        assert_eq!(
            compiled.variants_of("road").len(),
            2,
            "a straight road has two turns"
        );
        assert_eq!(
            compiled.variants_of("stair").len(),
            1,
            "a fixed stair is not rotated"
        );
        assert_eq!(
            compiled.variants_tagged("road"),
            compiled.variants_of("road")
        );
        assert_eq!(
            compiled.tileset.weights[compiled.variants_of("air")[0]],
            4.0
        );
    }

    #[test]
    fn connectors_keep_their_names() {
        let compiled = parse_module_set(ROADS)
            .expect("valid file")
            .compile()
            .expect("valid set");

        let road = compiled.connector("road").expect("named");
        let tile = compiled.variants_of("road")[0];

        assert!(matches!(
            compiled.face(tile, POS_X),
            crate::modules::Face::Horizontal(face) if face.connector == road
        ));
        assert_eq!(compiled.connector("nothing"), None);
    }

    #[test]
    fn an_undefined_face_names_the_module_and_the_face() {
        let text = ROADS.replace(r#"up: "stair""#, r#"up: "stairs""#);

        let error = parse_module_set(&text).expect_err("stairs is not defined");

        let message = error.to_string();
        assert!(
            message.contains("\"stair\"") && message.contains("\"stairs\""),
            "{message}"
        );
    }

    #[test]
    fn a_top_face_cannot_be_a_side() {
        let text = ROADS.replace(
            r#"sides: ["air", "road", "air", "air"]"#,
            r#"sides: ["air", "open", "air", "air"]"#,
        );

        let error = parse_module_set(&text).expect_err("open is a top face");

        assert!(
            error.to_string().contains("top face \"open\" as a side"),
            "{error}"
        );
    }

    #[test]
    fn a_pairing_of_a_connector_no_face_uses_is_an_error() {
        let text = ROADS.replace(r#"("door", "road")"#, r#"("door", "raod")"#);

        let error = parse_module_set(&text).expect_err("raod is a typo");

        assert!(
            error.to_string().contains("\"raod\", which no face uses"),
            "{error}"
        );
    }

    #[test]
    fn a_misspelt_field_is_an_error_rather_than_a_default() {
        let text = ROADS.replace("weight: 4.0", "wieght: 4.0");

        let error = parse_module_set(&text).expect_err("wieght is not a field");

        assert!(error.to_string().contains("wieght"), "{error}");
    }
}
