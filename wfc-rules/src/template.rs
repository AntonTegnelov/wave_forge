//! A module set learnt from a template, as WFC's overlapping model learns its patterns.
//!
//! A template is a small picture whose cells hold values, a colour each say. Every `n` by `n`
//! window of it is a pattern, and each distinct pattern becomes a module, weighted by how often the
//! template holds it and named `pattern <i>` in the order the windows are first met, row by row. Two patterns may lie side by side where they agree on the `n - 1` columns or
//! rows they share: a module's `+x` face carries a connector named for its last `n - 1` columns and
//! its `-x` face one named for its first, and the same along y, so the faces of two modules fit
//! exactly when their patterns agree ([`crate::modules`]). A cell solved with the set stands for its
//! pattern's lowest corner, which every module carries as a tag, `value <n>`; since neighbours
//! agree, every `n` by `n` window of a solved world is one of the template's patterns.
//!
//! Modules keep their orientation and stack freely along z, so the set describes a plane, one
//! layer deep.

use std::collections::BTreeMap;
use std::fmt::Write;

/// The most patterns a set may have: a cell's possibilities are a mask of 256 tiles.
pub const MAX_PATTERNS: usize = 256;

/// A template: `width` by `height` cells, row by row with x fastest, each holding a value.
#[derive(Clone, Copy, Debug)]
pub struct Template<'a> {
    pub width: usize,
    pub height: usize,
    pub cells: &'a [u32],
}

/// Why a template cannot be learnt.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TemplateError {
    #[error("{cells} cells do not fill a template of {width} by {height}")]
    Shape {
        width: usize,
        height: usize,
        cells: usize,
    },
    #[error("a pattern of {n} by {n} does not fit a template of {width} by {height}")]
    Size {
        n: usize,
        width: usize,
        height: usize,
    },
    #[error("the template holds {0} patterns, more than the {MAX_PATTERNS} a set may have")]
    TooMany(usize),
}

/// The module set learnt from `template`'s patterns of `n` by `n` cells, as a module set file
/// ([`crate::formats::module_format`]). With `wrap`, windows run across the template's edges onto
/// the opposite side, as for a template that tiles; without it, only whole windows count.
///
/// # Errors
/// [`TemplateError::Shape`] when the cells do not fill the template, [`TemplateError::Size`] for a
/// pattern of no size or one larger than the template, and [`TemplateError::TooMany`] for more
/// patterns than [`MAX_PATTERNS`].
pub fn learn(template: Template<'_>, n: usize, wrap: bool) -> Result<String, TemplateError> {
    let Template {
        width,
        height,
        cells,
    } = template;
    if cells.len() != width * height {
        return Err(TemplateError::Shape {
            width,
            height,
            cells: cells.len(),
        });
    }
    if n == 0 || n > width || n > height {
        return Err(TemplateError::Size { n, width, height });
    }
    let (across, up) = if wrap {
        (width, height)
    } else {
        (width - n + 1, height - n + 1)
    };
    // Each distinct window, row by row with x fastest, in the order first met, and how often the
    // template holds it.
    let mut patterns: Vec<Vec<u32>> = Vec::new();
    let mut counts: BTreeMap<Vec<u32>, usize> = BTreeMap::new();
    for y in 0..up {
        for x in 0..across {
            let window: Vec<u32> = (0..n * n)
                .map(|i| cells[(y + i / n) % height * width + (x + i % n) % width])
                .collect();
            let count = counts.entry(window.clone()).or_insert(0);
            if *count == 0 {
                patterns.push(window);
            }
            *count += 1;
        }
    }
    if patterns.len() > MAX_PATTERNS {
        return Err(TemplateError::TooMany(patterns.len()));
    }
    // The columns `skip..skip + n - 1` of a window, and its rows likewise: the strips two
    // neighbours share.
    let columns = |window: &[u32], skip: usize| -> Vec<u32> {
        (0..n)
            .flat_map(|y| (skip..skip + n - 1).map(move |x| (x, y)))
            .map(|(x, y)| window[y * n + x])
            .collect()
    };
    let rows = |window: &[u32], skip: usize| window[skip * n..(skip + n - 1) * n].to_vec();
    let mut strips: [BTreeMap<Vec<u32>, usize>; 2] = [BTreeMap::new(), BTreeMap::new()];
    let mut strip = |axis: usize, cells: Vec<u32>| {
        let known = strips[axis].len();
        format!(
            "{} {}",
            ["x", "y"][axis],
            *strips[axis].entry(cells).or_insert(known)
        )
    };
    let mut modules = String::new();
    for (index, window) in patterns.iter().enumerate() {
        let sides = [
            strip(0, columns(window, 1)),
            strip(0, columns(window, 0)),
            strip(1, rows(window, 1)),
            strip(1, rows(window, 0)),
        ];
        writeln!(
            modules,
            "        (name: \"pattern {index}\", sides: [{}], up: \"layer\", down: \"layer\", \
             weight: {}.0, rotatable: false, tags: [\"value {}\"]),",
            sides.map(|side| format!("\"{side}\"")).join(", "),
            counts[window],
            window[0]
        )
        .expect("writing to a string cannot fail");
    }
    let mut text = String::from("(\n    faces: {\n");
    for (axis, known) in strips.iter().enumerate() {
        for number in 0..known.len() {
            let name = format!("{} {number}", ["x", "y"][axis]);
            writeln!(text, "        \"{name}\": Side(connector: \"{name}\"),")
                .expect("writing to a string cannot fail");
        }
    }
    text.push_str("        \"layer\": Top(connector: \"layer\"),\n    },\n    modules: [\n");
    text.push_str(&modules);
    text.push_str("    ],\n)\n");
    Ok(text)
}
