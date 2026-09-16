//! Renders collapsed grids to PNG images for inspection.
//!
//! This is a deliberately small CPU rasteriser rather than a game engine or GPU renderer:
//! the images are produced inside tests and containers that have neither a display nor a GPU,
//! they must be pixel-for-pixel reproducible so runs can be compared, and they must be cheap
//! enough to write after every run. They let humans and LLM-assisted development *see* what the
//! generator did; the shipped library never renders anything.

use crate::invariants::TileGrid;
use image::{Rgb, RgbImage};

/// An sRGB colour.
pub type Color = [u8; 3];

/// Canvas background.
pub const BACKGROUND: Color = [30, 30, 34];
/// Columns or cells that contain only empty tiles.
pub const EMPTY: Color = [48, 48, 54];

/// How tiles are drawn.
#[derive(Debug, Clone, Copy)]
pub struct Style<'a> {
    /// Colour per tile index; tiles without an entry get [`default_color`].
    pub palette: &'a [Color],
    /// Tiles that represent empty space (for example air) and are not drawn as voxels.
    pub empty_tiles: &'a [usize],
    /// Edge length of one cell in pixels.
    pub cell_px: u32,
}

impl Style<'_> {
    fn color(&self, tile: usize) -> Color {
        self.palette.get(tile).copied().unwrap_or_else(|| default_color(tile))
    }

    fn is_empty(&self, tile: usize) -> bool {
        self.empty_tiles.contains(&tile)
    }

    fn cell(&self) -> u32 {
        self.cell_px.max(1)
    }
}

/// A distinct, stable colour for a tile index. Hues step by the golden ratio so neighbouring
/// indices never get similar colours, whatever the number of tiles.
pub fn default_color(tile: usize) -> Color {
    let hue = (tile as f32 * 0.618_034).fract();
    hsv_to_rgb(hue, 0.6, 0.9)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Color {
    let sector = (h * 6.0).floor();
    let f = h * 6.0 - sector;
    let (p, q, t) = (v * (1.0 - s), v * (1.0 - f * s), v * (1.0 - (1.0 - f) * s));
    let (r, g, b) = match (sector as i32).rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    [r, g, b].map(|c| (c * 255.0).round() as u8)
}

fn shade(color: Color, factor: f32) -> Color {
    color.map(|c| (f32::from(c) * factor).round().clamp(0.0, 255.0) as u8)
}

/// Brightness for something `distance` cells away from the viewer out of `extent`: nearer is
/// brighter, so orthographic views still convey depth.
fn depth_brightness(distance: usize, extent: usize) -> f32 {
    1.0 - 0.5 * distance as f32 / extent.saturating_sub(1).max(1) as f32
}

fn fill_rect(image: &mut RgbImage, x: u32, y: u32, width: u32, height: u32, color: Color) {
    for py in y..(y + height).min(image.height()) {
        for px in x..(x + width).min(image.width()) {
            image.put_pixel(px, py, Rgb(color));
        }
    }
}

/// Draws a cell with a one-pixel darker outline so individual cells stay distinguishable.
fn fill_cell(image: &mut RgbImage, x: u32, y: u32, size: u32, color: Color) {
    fill_rect(image, x, y, size, size, shade(color, 0.7));
    if size > 2 {
        fill_rect(image, x + 1, y + 1, size - 2, size - 2, color);
    }
}

/// One `z` layer seen from above: `+x` to the right, `+y` up.
pub fn render_layer(grid: &TileGrid, z: usize, style: &Style) -> RgbImage {
    assert!(z < grid.depth, "layer {z} out of range for depth {}", grid.depth);
    let c = style.cell();
    let mut image = RgbImage::from_pixel(grid.width as u32 * c, grid.height as u32 * c, Rgb(BACKGROUND));
    for y in 0..grid.height {
        for x in 0..grid.width {
            let tile = grid.get(x, y, z);
            let color = if style.is_empty(tile) { EMPTY } else { style.color(tile) };
            fill_cell(&mut image, x as u32 * c, (grid.height - 1 - y) as u32 * c, c, color);
        }
    }
    image
}

/// A 2×2 sheet of the whole grid with `+z` up, laid out as:
///
/// | top (looking down, `+x` right, `+y` up) | isometric (from `+x`, `+y`, `+z`)       |
/// |------------------------------------------|------------------------------------------|
/// | front (from `-y`, `+x` right)            | side (from `+x`, `+y` right)            |
///
/// Orthographic views show exact positions without perspective distortion; the isometric view
/// shows how they fit together. Nearer surfaces are drawn brighter.
pub fn render_four_view(grid: &TileGrid, style: &Style) -> RgbImage {
    let top = top_view(grid, style);
    let iso = isometric_view(grid, style);
    let front = front_view(grid, style);
    let side = side_view(grid, style);

    let pad = style.cell().max(4);
    let col0 = top.width().max(front.width());
    let col1 = iso.width().max(side.width());
    let row0 = top.height().max(iso.height());
    let row1 = front.height().max(side.height());
    let mut sheet = RgbImage::from_pixel(pad * 3 + col0 + col1, pad * 3 + row0 + row1, Rgb(BACKGROUND));
    for (view, x, y) in [
        (&top, pad, pad),
        (&iso, pad * 2 + col0, pad),
        (&front, pad, pad * 2 + row0),
        (&side, pad * 2 + col0, pad * 2 + row0),
    ] {
        image::imageops::replace(&mut sheet, view, i64::from(x), i64::from(y));
    }
    sheet
}

fn top_view(grid: &TileGrid, style: &Style) -> RgbImage {
    let c = style.cell();
    let mut image = RgbImage::from_pixel(grid.width as u32 * c, grid.height as u32 * c, Rgb(BACKGROUND));
    for y in 0..grid.height {
        for x in 0..grid.width {
            let hit = (0..grid.depth).rev().find(|&z| !style.is_empty(grid.get(x, y, z)));
            let color = hit.map_or(EMPTY, |z| {
                shade(style.color(grid.get(x, y, z)), depth_brightness(grid.depth - 1 - z, grid.depth))
            });
            fill_cell(&mut image, x as u32 * c, (grid.height - 1 - y) as u32 * c, c, color);
        }
    }
    image
}

fn front_view(grid: &TileGrid, style: &Style) -> RgbImage {
    let c = style.cell();
    let mut image = RgbImage::from_pixel(grid.width as u32 * c, grid.depth as u32 * c, Rgb(BACKGROUND));
    for z in 0..grid.depth {
        for x in 0..grid.width {
            let hit = (0..grid.height).find(|&y| !style.is_empty(grid.get(x, y, z)));
            let color = hit.map_or(EMPTY, |y| shade(style.color(grid.get(x, y, z)), depth_brightness(y, grid.height)));
            fill_cell(&mut image, x as u32 * c, (grid.depth - 1 - z) as u32 * c, c, color);
        }
    }
    image
}

fn side_view(grid: &TileGrid, style: &Style) -> RgbImage {
    let c = style.cell();
    let mut image = RgbImage::from_pixel(grid.height as u32 * c, grid.depth as u32 * c, Rgb(BACKGROUND));
    for z in 0..grid.depth {
        for y in 0..grid.height {
            let hit = (0..grid.width).rev().find(|&x| !style.is_empty(grid.get(x, y, z)));
            let color = hit.map_or(EMPTY, |x| {
                shade(style.color(grid.get(x, y, z)), depth_brightness(grid.width - 1 - x, grid.width))
            });
            fill_cell(&mut image, y as u32 * c, (grid.depth - 1 - z) as u32 * c, c, color);
        }
    }
    image
}

/// Voxel cubes projected with `screen = ((x - y) · a, (x + y) · a / 2 - z · a)`. Points along
/// (1, 1, 1) project onto each other, so the viewer looks down that diagonal and sees the +x, +y
/// and +z faces. Drawing cubes in increasing `x + y + z` therefore paints far to near.
fn isometric_view(grid: &TileGrid, style: &Style) -> RgbImage {
    let a = style.cell().max(2) as f32;
    let (w, h, d) = (grid.width as f32, grid.height as f32, grid.depth as f32);
    let width = ((w + h) * a).ceil() as u32 + 1;
    let height = ((w + h) * a / 2.0 + d * a).ceil() as u32 + 1;
    let mut image = RgbImage::from_pixel(width, height, Rgb(BACKGROUND));
    let project = |x: f32, y: f32, z: f32| ((x - y + h) * a, (x + y) * a / 2.0 + (d - z) * a);

    let mut voxels: Vec<(usize, usize, usize)> = (0..grid.depth)
        .flat_map(|z| (0..grid.height).flat_map(move |y| (0..grid.width).map(move |x| (x, y, z))))
        .filter(|&(x, y, z)| !style.is_empty(grid.get(x, y, z)))
        .collect();
    voxels.sort_by_key(|&(x, y, z)| (x + y + z, z));

    for (x, y, z) in voxels {
        let color = style.color(grid.get(x, y, z));
        let (x, y, z) = (x as f32, y as f32, z as f32);
        let top = [project(x, y, z + 1.0), project(x + 1.0, y, z + 1.0), project(x + 1.0, y + 1.0, z + 1.0), project(x, y + 1.0, z + 1.0)];
        let pos_x = [project(x + 1.0, y, z), project(x + 1.0, y + 1.0, z), project(x + 1.0, y + 1.0, z + 1.0), project(x + 1.0, y, z + 1.0)];
        let pos_y = [project(x, y + 1.0, z), project(x + 1.0, y + 1.0, z), project(x + 1.0, y + 1.0, z + 1.0), project(x, y + 1.0, z + 1.0)];
        fill_convex_quad(&mut image, top, color);
        fill_convex_quad(&mut image, pos_x, shade(color, 0.78));
        fill_convex_quad(&mut image, pos_y, shade(color, 0.6));
    }
    image
}

/// A tile drawn as `resolution³` coloured voxels with `+z` up, so structured tile sets (stairs,
/// roofs, walkways) are recognisable instead of flat colour blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoxelModel {
    /// Voxels along each edge of the cell.
    pub resolution: usize,
    /// `resolution³` voxels indexed by [`VoxelModel::index`]; `None` is empty.
    pub voxels: Vec<Option<Color>>,
}

impl VoxelModel {
    /// A model with no voxels.
    pub fn empty(resolution: usize) -> Self {
        assert!(resolution > 0, "voxel resolution must be positive");
        Self { resolution, voxels: vec![None; resolution.pow(3)] }
    }

    /// Index of voxel `(x, y, z)` in [`VoxelModel::voxels`].
    pub fn index(&self, x: usize, y: usize, z: usize) -> usize {
        let r = self.resolution;
        assert!(x < r && y < r && z < r, "voxel ({x}, {y}, {z}) outside resolution {r}");
        (z * r + y) * r + x
    }

    /// The voxel at `(x, y, z)`.
    pub fn get(&self, x: usize, y: usize, z: usize) -> Option<Color> {
        self.voxels[self.index(x, y, z)]
    }

    /// Sets or clears the voxel at `(x, y, z)`.
    pub fn set(&mut self, x: usize, y: usize, z: usize, color: Option<Color>) {
        let index = self.index(x, y, z);
        self.voxels[index] = color;
    }

    /// This model rotated by `quarter_turns` counter-clockwise around `+z` (seen from above), the
    /// rotation `wfc_rules::modules` applies to module variants.
    pub fn rotated(&self, quarter_turns: u8) -> Self {
        let r = self.resolution;
        let mut out = Self::empty(r);
        for z in 0..r {
            for y in 0..r {
                for x in 0..r {
                    // Undo the rotation to find where this voxel came from.
                    let (mut sx, mut sy) = (x, y);
                    for _ in 0..quarter_turns % 4 {
                        (sx, sy) = (sy, r - 1 - sx);
                    }
                    out.set(x, y, z, self.get(sx, sy, z));
                }
            }
        }
        out
    }
}

/// Isometric view (from `+x`, `+y`, `+z`) of a grid whose cells are drawn as their tile's
/// [`VoxelModel`]. `models[tile]` must exist for every tile in the grid and all models must share
/// one resolution. Voxels hidden behind filled neighbours are skipped, so cost follows the visible
/// surface rather than the grid volume.
pub fn render_voxel_isometric(grid: &TileGrid, models: &[VoxelModel], voxel_px: u32) -> RgbImage {
    let r = models.first().map_or(1, |m| m.resolution);
    assert!(models.iter().all(|m| m.resolution == r), "voxel models must share one resolution");
    let (w, h, d) = (grid.width * r, grid.height * r, grid.depth * r);
    let voxel_at = |x: usize, y: usize, z: usize| -> Option<Color> {
        if x >= w || y >= h || z >= d {
            return None;
        }
        let tile = grid.get(x / r, y / r, z / r);
        models.get(tile).unwrap_or_else(|| panic!("no voxel model for tile {tile}")).get(x % r, y % r, z % r)
    };

    let a = voxel_px.max(1) as f32;
    let width = ((w + h) as f32 * a).ceil() as u32 + 1;
    let height = ((w + h) as f32 * a / 2.0 + d as f32 * a).ceil() as u32 + 1;
    let mut image = RgbImage::from_pixel(width, height, Rgb(BACKGROUND));
    let project = |x: f32, y: f32, z: f32| ((x - y + h as f32) * a, (x + y) * a / 2.0 + (d as f32 - z) * a);

    let mut visible = Vec::new();
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                if let Some(color) = voxel_at(x, y, z) {
                    let open = [voxel_at(x + 1, y, z).is_none(), voxel_at(x, y + 1, z).is_none(), voxel_at(x, y, z + 1).is_none()];
                    if open.iter().any(|&o| o) {
                        visible.push((x, y, z, color, open));
                    }
                }
            }
        }
    }
    // Same painter's order as `isometric_view`: far to near along the view diagonal.
    visible.sort_by_key(|&(x, y, z, _, _)| (x + y + z, z));

    for (x, y, z, color, [open_x, open_y, open_z]) in visible {
        let (x, y, z) = (x as f32, y as f32, z as f32);
        if open_z {
            let top = [project(x, y, z + 1.0), project(x + 1.0, y, z + 1.0), project(x + 1.0, y + 1.0, z + 1.0), project(x, y + 1.0, z + 1.0)];
            fill_convex_quad(&mut image, top, color);
        }
        if open_x {
            let pos_x = [project(x + 1.0, y, z), project(x + 1.0, y + 1.0, z), project(x + 1.0, y + 1.0, z + 1.0), project(x + 1.0, y, z + 1.0)];
            fill_convex_quad(&mut image, pos_x, shade(color, 0.78));
        }
        if open_y {
            let pos_y = [project(x, y + 1.0, z), project(x + 1.0, y + 1.0, z), project(x + 1.0, y + 1.0, z + 1.0), project(x, y + 1.0, z + 1.0)];
            fill_convex_quad(&mut image, pos_y, shade(color, 0.6));
        }
    }
    image
}

/// Fills a convex quadrilateral by testing pixel centres against its edges.
fn fill_convex_quad(image: &mut RgbImage, points: [(f32, f32); 4], color: Color) {
    let signed_area: f32 = (0..4)
        .map(|i| {
            let (p, q) = (points[i], points[(i + 1) % 4]);
            p.0 * q.1 - q.0 * p.1
        })
        .sum();
    if signed_area.abs() < f32::EPSILON || image.width() == 0 || image.height() == 0 {
        return;
    }
    let orientation = signed_area.signum();
    let bound = |select: fn(&(f32, f32)) -> f32, limit: u32, round: fn(f32) -> f32, pick: fn(f32, f32) -> f32, start: f32| {
        round(points.iter().map(select).fold(start, pick)).clamp(0.0, (limit - 1) as f32) as u32
    };
    let (min_x, max_x) = (bound(|p| p.0, image.width(), f32::floor, f32::min, f32::INFINITY), bound(|p| p.0, image.width(), f32::ceil, f32::max, f32::NEG_INFINITY));
    let (min_y, max_y) = (bound(|p| p.1, image.height(), f32::floor, f32::min, f32::INFINITY), bound(|p| p.1, image.height(), f32::ceil, f32::max, f32::NEG_INFINITY));
    for py in min_y..=max_y {
        for px in min_x..=max_x {
            let (cx, cy) = (px as f32 + 0.5, py as f32 + 0.5);
            let inside = (0..4).all(|i| {
                let (p, q) = (points[i], points[(i + 1) % 4]);
                ((q.0 - p.0) * (cy - p.1) - (q.1 - p.1) * (cx - p.0)) * orientation >= 0.0
            });
            if inside {
                image.put_pixel(px, py, Rgb(color));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RED: Color = [200, 40, 40];
    const BLUE: Color = [40, 40, 200];

    fn style<'a>(palette: &'a [Color], empty_tiles: &'a [usize]) -> Style<'a> {
        Style { palette, empty_tiles, cell_px: 8 }
    }

    #[test]
    fn layer_puts_positive_y_at_the_top_of_the_image() {
        let grid = TileGrid::new(1, 2, 1, vec![0, 1]).unwrap();
        let image = render_layer(&grid, 0, &style(&[RED, BLUE], &[]));
        assert_eq!(image.dimensions(), (8, 16));
        assert_eq!(image.get_pixel(4, 4).0, BLUE, "y = 1 is drawn in the top cell");
        assert_eq!(image.get_pixel(4, 12).0, RED);
        assert_eq!(image.get_pixel(0, 0).0, shade(BLUE, 0.7), "cells have a darker outline");
    }

    #[test]
    fn top_view_shows_the_highest_non_empty_tile() {
        // Column (0, 0) holds red at z = 0 and blue at z = 1; empty tile 2 is above both.
        let grid = TileGrid::new(1, 1, 3, vec![0, 1, 2]).unwrap();
        let image = top_view(&grid, &style(&[RED, BLUE, [0, 0, 0]], &[2]));
        assert_eq!(image.get_pixel(4, 4).0, shade(BLUE, depth_brightness(1, 3)));
    }

    #[test]
    fn fully_empty_columns_are_marked_empty() {
        let grid = TileGrid::new(1, 1, 2, vec![3, 3]).unwrap();
        assert_eq!(top_view(&grid, &style(&[], &[3])).get_pixel(4, 4).0, EMPTY);
    }

    #[test]
    fn isometric_view_draws_the_top_face_at_full_brightness() {
        let grid = TileGrid::new(1, 1, 1, vec![0]).unwrap();
        let s = style(&[RED], &[]);
        let image = isometric_view(&grid, &s);
        // Centre of the top face: (0.5, 0.5, 1) -> ((0.5 - 0.5 + 1) * 8, (0.5 + 0.5) * 4 + 0 * 8).
        assert_eq!(image.get_pixel(8, 4).0, RED);
    }

    #[test]
    fn four_view_sheet_contains_all_views() {
        let grid = TileGrid::new(2, 3, 4, vec![0; 24]).unwrap();
        let s = style(&[RED], &[]);
        let sheet = render_four_view(&grid, &s);
        let (top, iso, front, side) = (top_view(&grid, &s), isometric_view(&grid, &s), front_view(&grid, &s), side_view(&grid, &s));
        let pad = 8;
        assert_eq!(
            sheet.dimensions(),
            (
                pad * 3 + top.width().max(front.width()) + iso.width().max(side.width()),
                pad * 3 + top.height().max(iso.height()) + front.height().max(side.height())
            )
        );
    }

    #[test]
    fn voxel_rotation_moves_the_positive_x_edge_to_positive_y() {
        let mut model = VoxelModel::empty(4);
        model.set(3, 1, 2, Some(RED));
        let rotated = model.rotated(1);
        assert_eq!(rotated.get(2, 3, 2), Some(RED));
        assert_eq!(rotated.voxels.iter().flatten().count(), 1);
        assert_eq!(model.rotated(4), model);
    }

    #[test]
    fn voxel_isometric_draws_each_voxel_top_face() {
        // One cell at resolution 2 holding a single voxel at its origin.
        let mut model = VoxelModel::empty(2);
        model.set(0, 0, 0, Some(BLUE));
        let grid = TileGrid::new(1, 1, 1, vec![0]).unwrap();
        let image = render_voxel_isometric(&grid, &[model], 8);
        // Top face centre (0.5, 0.5, 1) -> ((0.5 - 0.5 + 2) * 8, (0.5 + 0.5) * 4 + (2 - 1) * 8).
        assert_eq!(image.get_pixel(16, 12).0, BLUE);
    }

    #[test]
    fn default_colors_are_stable_and_distinct_for_neighbours() {
        assert_eq!(default_color(3), default_color(3));
        assert_ne!(default_color(0), default_color(1));
    }
}
