//! What hides what is behind it: boxes of a chunk's solid cells, which an engine's occlusion
//! culling uses to skip drawing what they cover.
//!
//! A module set says which modules fill their cell with opaque geometry
//! ([`crate::modules::ModulePrototype`]'s `solid`). An occluder must be conservative: one that
//! reaches past the geometry it stands for hides what should show, so only whole solid cells
//! become occluders, merged into few boxes per chunk.

use crate::cell_boxes::{CellBox, cell_boxes};
use crate::loader::RuleFile;
use crate::space::YUpSpace;
use wfc_core::Chunk;

/// Boxes that cover a solved `chunk`'s solid cells, each once, in the engine's world space.
#[must_use]
pub fn occluders(chunk: &Chunk, rules: &RuleFile, space: &YUpSpace) -> Vec<CellBox> {
    let solid: Vec<bool> = chunk
        .tiles
        .iter()
        .map(|&tile| rules.solid(usize::from(tile)))
        .collect();
    cell_boxes(chunk.coord, &solid, space)
}
