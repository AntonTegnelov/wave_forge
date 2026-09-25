use crate::formats::{FormatParser, ron_format::RonFormatParser};
use crate::modules::CompiledModules;
use crate::{AdjacencyRules, LoadError, TileSet, formats};
use std::fs::File;
use std::io::Read as _;
use std::path::Path;

/// Loads a `TileSet` and corresponding `AdjacencyRules` from a rule definition file.
///
/// This function acts as the main entry point for loading WFC rules.
/// It attempts to determine the appropriate parser based on the file extension.
///
/// # Arguments
///
/// * `path` - A reference to the `Path` of the rule file to load.
///
/// # Returns
///
/// * `Ok((TileSet, AdjacencyRules))` containing the loaded tile information and adjacency constraints
///   if the file is successfully read and parsed.
/// * `Err(LoadError)` if any error occurs during file reading, parsing, or data validation.
///   The specific error type (`Io`, `ParseError`, `InvalidData`) provides more details.
pub fn load_from_file(path: &Path) -> Result<(TileSet, AdjacencyRules), LoadError> {
    // 1. Read the file content
    let mut file = File::open(path).map_err(LoadError::Io)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents).map_err(LoadError::Io)?;

    // 2. Determine the parser based on file extension
    let parser = get_parser_for_path(path)?;

    // 3. Parse the content with the selected parser
    parser.parse(&contents)
}

/// Loads a `TileSet` and corresponding `AdjacencyRules` from a string in RON format.
///
/// This function is maintained for backward compatibility with the old API.
///
/// # Arguments
///
/// * `content` - A string containing the RON-formatted rule definition.
///
/// # Returns
///
/// * `Ok((TileSet, AdjacencyRules))` if parsing succeeds.
/// * `Err(LoadError)` if parsing fails.
pub fn load_from_ron_string(content: &str) -> Result<(TileSet, AdjacencyRules), LoadError> {
    formats::ron_format::parse_ron_rules(content)
}

/// Returns an appropriate parser implementation based on the file extension.
///
/// # Arguments
///
/// * `path` - The file path from which to determine the format.
///
/// # Returns
///
/// * `Ok(Box<dyn FormatParser>)` - A parser instance appropriate for the file extension.
/// * `Err(LoadError)` - If no appropriate parser can be determined.
fn get_parser_for_path(path: &Path) -> Result<Box<dyn FormatParser>, LoadError> {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("ron") => Ok(Box::new(RonFormatParser::new())),
        // Add additional formats here as they are implemented
        // Some("json") => Ok(Box::new(JsonFormatParser::new())),
        // Some("yaml") | Some("yml") => Ok(Box::new(YamlFormatParser::new())),
        // Some("bin") => Ok(Box::new(BitcodeFormatParser::new())),
        Some(ext) => Err(LoadError::InvalidData(format!(
            "Unsupported file extension: .{ext} - supported formats are: .ron"
        ))),
        None => Err(LoadError::InvalidData(
            "File has no extension - unable to determine format".to_string(),
        )),
    }
}

/// A rule file of either form, compiled.
#[derive(Debug, Clone)]
pub enum RuleFile {
    /// Tiles with adjacency written out tuple by tuple.
    Tiles {
        /// The tiles and their weights.
        tileset: TileSet,
        /// Which tiles may touch along each axis.
        rules: AdjacencyRules,
        /// Each tile's name, by tile id.
        names: Vec<String>,
    },
    /// Modules described by the connectors on their faces, with rotations derived; see
    /// [`crate::modules`].
    Modules(CompiledModules),
}

impl RuleFile {
    /// The tiles, whatever the file's form.
    #[must_use]
    pub const fn tileset(&self) -> &TileSet {
        match self {
            Self::Tiles { tileset, .. } => tileset,
            Self::Modules(modules) => &modules.tileset,
        }
    }

    /// The adjacency rules, whatever the file's form.
    #[must_use]
    pub const fn rules(&self) -> &AdjacencyRules {
        match self {
            Self::Tiles { rules, .. } => rules,
            Self::Modules(modules) => &modules.rules,
        }
    }

    /// The module set, for a file of module form: what each tile id is (prototype, rotation, tags).
    #[must_use]
    pub const fn modules(&self) -> Option<&CompiledModules> {
        match self {
            Self::Tiles { .. } => None,
            Self::Modules(modules) => Some(modules),
        }
    }

    /// How many tiles there are; tile ids run from zero to one less.
    #[must_use]
    pub fn num_tiles(&self) -> usize {
        self.tileset().weights.len()
    }

    /// What tile `tile` is called: the tile's own name in a tile set, its prototype's name in a
    /// module set, where the four rotations of one prototype share a name. A game draws one model
    /// per name.
    ///
    /// # Panics
    /// If `tile` is not a tile of the set.
    #[must_use]
    pub fn name(&self, tile: usize) -> &str {
        match self {
            Self::Tiles { names, .. } => &names[tile],
            Self::Modules(modules) => &modules.prototype_of(tile).name,
        }
    }

    /// How far tile `tile` is turned from its prototype, in quarter turns counter-clockwise about
    /// +z. Always zero in a tile set, which has no rotations.
    ///
    /// # Panics
    /// If `tile` is not a tile of the set.
    #[must_use]
    pub fn rotation(&self, tile: usize) -> u8 {
        match self {
            Self::Tiles { names, .. } => {
                assert!(tile < names.len(), "tile {tile} of {}", names.len());
                0
            }
            Self::Modules(modules) => modules.variants[tile].rotation,
        }
    }

    /// The tiles whose name is `name`: one in a tile set, every rotation of the prototype in a
    /// module set.
    #[must_use]
    pub fn tiles_named(&self, name: &str) -> Vec<usize> {
        (0..self.num_tiles())
            .filter(|&tile| self.name(tile) == name)
            .collect()
    }

    /// What walkers on tile `tile` stand on, if its module says. A tile set says nothing.
    ///
    /// # Panics
    /// If `tile` is not a tile of the set.
    #[must_use]
    pub fn surface(&self, tile: usize) -> Option<&str> {
        match self {
            Self::Tiles { .. } => None,
            Self::Modules(modules) => modules.prototype_of(tile).surface.as_deref(),
        }
    }

    /// Whether tile `tile` is inside. A tile set's tiles are not.
    ///
    /// # Panics
    /// If `tile` is not a tile of the set.
    #[must_use]
    pub fn indoor(&self, tile: usize) -> bool {
        match self {
            Self::Tiles { .. } => false,
            Self::Modules(modules) => modules.prototype_of(tile).indoor,
        }
    }

    /// Whether tile `tile`'s model fills its cell with opaque geometry. A tile set's tiles do not
    /// say so.
    ///
    /// # Panics
    /// If `tile` is not a tile of the set.
    #[must_use]
    pub fn solid(&self, tile: usize) -> bool {
        match self {
            Self::Tiles { .. } => false,
            Self::Modules(modules) => modules.prototype_of(tile).solid,
        }
    }

    /// The sounds tile `tile` makes, turned with it. A tile set's tiles make none.
    ///
    /// # Panics
    /// If `tile` is not a tile of the set.
    #[must_use]
    pub fn sounds(&self, tile: usize) -> Vec<crate::modules::ModuleSound> {
        match self {
            Self::Tiles { .. } => Vec::new(),
            Self::Modules(modules) => modules.sounds(tile),
        }
    }

    /// The tiles whose prototype carries `tag`. A tile set has no tags, so none.
    #[must_use]
    pub fn tiles_tagged(&self, tag: &str) -> Vec<usize> {
        match self {
            Self::Tiles { .. } => Vec::new(),
            Self::Modules(modules) => modules.variants_tagged(tag),
        }
    }
}

/// Parses and compiles a rule file of either form. A file with a `modules` list is a module set
/// ([`crate::formats::module_format`]); one with `tiles` and `adjacency` lists is a tile set.
///
/// # Errors
/// If the text is neither form, or the rules it describes are invalid.
#[cfg(feature = "serde")]
pub fn parse_rule_file(content: &str) -> Result<RuleFile, LoadError> {
    /// Only asks whether the file has a `modules` list; everything else is read by the form's own
    /// parser, which reports errors in its own terms.
    #[derive(serde::Deserialize)]
    struct Form {
        #[serde(default)]
        modules: Option<serde::de::IgnoredAny>,
        #[serde(default)]
        tiles: Vec<Named>,
    }
    #[derive(serde::Deserialize)]
    struct Named {
        name: String,
    }
    // RON writes an option as `Some(...)`; implicit `Some` lets a plain list count as present.
    let form: Form = ron::Options::default()
        .with_default_extension(ron::extensions::Extensions::IMPLICIT_SOME)
        .from_str(content)
        .map_err(|error| LoadError::ParseError(format!("rule file: {error}")))?;
    if form.modules.is_some() {
        let modules = crate::formats::module_format::parse_module_set(content)?
            .compile()
            .map_err(|error| LoadError::InvalidData(error.to_string()))?;
        Ok(RuleFile::Modules(modules))
    } else {
        let (tileset, rules) = load_from_ron_string(content)?;
        Ok(RuleFile::Tiles {
            tileset,
            rules,
            names: form.tiles.into_iter().map(|tile| tile.name).collect(),
        })
    }
}

/// Reads and compiles a rule file of either form; see [`parse_rule_file`].
///
/// # Errors
/// If the file cannot be read, or as [`parse_rule_file`].
#[cfg(feature = "serde")]
pub fn load_rule_file(path: &Path) -> Result<RuleFile, LoadError> {
    let content = std::fs::read_to_string(path).map_err(LoadError::Io)?;
    parse_rule_file(&content)
}
