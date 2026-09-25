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
//!
//! A module can also say what its cell is like, for sound and walking: `surface: "cobblestone"`,
//! what walkers in the cell stand on; `indoor: true`, that the cell is inside; and
//! `sounds: [(at: (0.5, 0.5, 0.2), key: "fountain")]`, sounds playing at points of the cell, from 0
//! to 1 along x, y and z in the module's own frame, which turn with it. A surface or key may not be
//! empty, and a sound's point must be in its cell. `solid: true` says the module's model fills its
//! whole cell with opaque geometry, so the cell can hide what is behind it: say it only of a model
//! with no gap a view could pass through, or an occluder hides what should show.

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
    /// What walkers in the cell stand on, for footsteps.
    #[serde(default)]
    surface: Option<String>,
    /// Whether the cell is inside.
    #[serde(default)]
    indoor: bool,
    /// Whether the module's model fills its cell with opaque geometry.
    #[serde(default)]
    solid: bool,
    /// Sounds the module makes, in its own frame.
    #[serde(default)]
    sounds: Vec<SoundDef>,
}

/// A sound a module makes: `at` a point of its cell, from 0 to 1 along x, y and z.
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct SoundDef {
    at: (f32, f32, f32),
    key: String,
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
    // An optional field such as `surface: "stone"` reads without `Some(...)`, which still works.
    let file: ModuleFile = ron::Options::default()
        .with_default_extension(ron::extensions::Extensions::IMPLICIT_SOME)
        .from_str(content)
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
        if let Some(surface) = &module.surface {
            if surface.is_empty() {
                return Err(LoadError::InvalidData(format!(
                    "module {:?} has an empty surface",
                    module.name
                )));
            }
            prototype = prototype.surface(surface);
        }
        if module.indoor {
            prototype = prototype.indoor();
        }
        if module.solid {
            prototype = prototype.solid();
        }
        for sound in &module.sounds {
            let at = [sound.at.0, sound.at.1, sound.at.2];
            if !at.iter().all(|coordinate| (0.0..=1.0).contains(coordinate)) {
                return Err(LoadError::InvalidData(format!(
                    "module {:?} plays {:?} at {at:?}, outside its cell: each coordinate runs \
                     from 0 to 1",
                    module.name, sound.key
                )));
            }
            if sound.key.is_empty() {
                return Err(LoadError::InvalidData(format!(
                    "module {:?} has a sound without a key",
                    module.name
                )));
            }
            prototype = prototype.sound(at, &sound.key);
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

    const SOUNDING: &str = r#"(
        faces: {
            "air": Side(connector: "air"),
            "road": Side(connector: "road", walkable: true),
            "open": Top(connector: "open"),
        },
        modules: [
            (name: "air", sides: ["air", "air", "air", "air"], up: "open", down: "open"),
            (name: "kiosk", sides: ["road", "air", "air", "air"], up: "open", down: "open",
             surface: "cobblestone", indoor: true, solid: true,
             sounds: [(at: (0.9, 0.5, 0.2), key: "radio")]),
        ],
    )"#;

    #[test]
    fn a_modules_surface_indoor_and_sounds_are_read() {
        let compiled = parse_module_set(SOUNDING)
            .expect("valid file")
            .compile()
            .expect("valid set");

        let kiosk = compiled.prototype_of(compiled.variants_of("kiosk")[0]);
        let air = compiled.prototype_of(compiled.variants_of("air")[0]);
        assert_eq!(kiosk.surface.as_deref(), Some("cobblestone"));
        assert!(kiosk.indoor);
        assert!(kiosk.solid);
        assert!(!air.solid);
        assert_eq!(kiosk.sounds.len(), 1);
        assert_eq!(
            (air.surface.as_deref(), air.indoor, air.sounds.len()),
            (None, false, 0)
        );
    }

    #[test]
    fn a_sound_turns_with_its_module() {
        let compiled = parse_module_set(SOUNDING)
            .expect("valid file")
            .compile()
            .expect("valid set");
        let road = compiled.connector("road").expect("named");
        // The side each axis faces, as the point in the middle of it.
        let middles = [
            (POS_X, [1.0, 0.5]),
            (NEG_X, [0.0, 0.5]),
            (POS_Y, [0.5, 1.0]),
            (NEG_Y, [0.5, 0.0]),
        ];

        let tiles = compiled.variants_of("kiosk");
        assert_eq!(tiles.len(), 4);
        for tile in tiles {
            let [x, y, z] = compiled.sounds(tile)[0].at;
            let nearest = middles
                .iter()
                .min_by(|a, b| {
                    let distance = |m: [f32; 2]| (m[0] - x).powi(2) + (m[1] - y).powi(2);
                    distance(a.1).total_cmp(&distance(b.1))
                })
                .expect("four sides")
                .0;

            assert!(
                matches!(
                    compiled.face(tile, nearest),
                    crate::modules::Face::Horizontal(face) if face.connector == road
                ),
                "tile {tile}: the radio is by axis {nearest}, which is not the road"
            );
            assert_eq!(z, 0.2);
        }
    }

    #[test]
    fn a_sound_outside_its_cell_is_an_error() {
        let text = SOUNDING.replace("(0.9, 0.5, 0.2)", "(1.5, 0.5, 0.2)");

        let error = parse_module_set(&text).expect_err("outside the cell");

        let message = error.to_string();
        assert!(
            message.contains("\"kiosk\"") && message.contains("outside"),
            "{message}"
        );
    }

    #[test]
    fn an_empty_surface_or_sound_key_is_an_error() {
        let surface = SOUNDING.replace(r#"surface: "cobblestone""#, r#"surface: """#);
        let key = SOUNDING.replace(r#"key: "radio""#, r#"key: """#);

        assert!(parse_module_set(&surface).is_err());
        assert!(parse_module_set(&key).is_err());
    }
}
