//! A chunk's ground: a mesh and a height-field collider built from a height field stage.
//!
//! A field holds one height per cell column. The ground puts a vertex on every column's centre and
//! adds the first column of the chunks beyond its +x and +y edges, so two neighbouring chunks share
//! the vertices along their common edge and meet without a seam. Normals come from the heights on
//! either side of a vertex, which at an edge are the neighbour's, so the shading is continuous across
//! chunks too. A chunk's ground therefore reads the fields of its eight neighbours as well as its
//! own, and [`ground`] builds it only once all nine have arrived; [`ground_readers`] says which
//! chunks' ground may have become buildable when one field arrives.
//!
//! Far chunks are drawn coarser: each of a mesh's [`GroundLevel`]s triangulates every `step`-th
//! vertex and says how far that surface strays from full detail, which is what an engine's level of
//! detail selection weighs. Neighbours drawn at different levels no longer share every edge vertex,
//! so each level hangs a skirt below the chunk's edges, deep enough to close the gap between any
//! two levels.
//!
//! [`ground_materials`] gives the category of every vertex of the same grid from a Rules stage, so
//! an engine can tell its ground shader which material each part of the ground is.
//!
//! Everything is in a Y-up engine's axes, as [`crate::YUpSpace`] maps them: the lattice's x is the
//! engine's x, its y the engine's z, and a field's height the engine's y.

use crate::stages::{Categories, Field};
use wfc_core::ChunkCoord;

/// One chunk's ground, ready for an engine: a triangle mesh and the same heights as a grid for a
/// height-field collider.
#[derive(Clone, Debug, PartialEq)]
pub struct GroundMesh {
    pub chunk: ChunkCoord,
    /// Vertices along the engine's x and z: one more than the chunk's columns each way.
    pub size: [u32; 2],
    /// Relative to the chunk's corner on the ground plane (the engine's x and z), with the height
    /// absolute. First the grid, row by row along z, x fastest: vertex `(i, j)` is at index
    /// `j * size[0] + i`, above the centre of column `(i, j)` of the chunk. Then the skirt: a copy
    /// of every vertex on the grid's edge, hung below it.
    pub positions: Vec<[f32; 3]>,
    /// Unit normals, one per vertex; a skirt vertex has its edge vertex's.
    pub normals: Vec<[f32; 3]>,
    /// The levels of detail, finest first: the first has a step of 1, full detail.
    pub levels: Vec<GroundLevel>,
    /// The height of every vertex of the grid in engine units, in the same order as `positions`:
    /// what a height-field collider takes.
    pub heights: Vec<f32>,
}

/// One level of detail of a chunk's ground.
#[derive(Clone, Debug, PartialEq)]
pub struct GroundLevel {
    /// The level takes every `step`-th vertex of the grid along x and z: 1, 2, 4 and so on, each
    /// dividing the chunk's columns both ways.
    pub step: u32,
    /// Triangles into [`GroundMesh::positions`]: first the surface, two per square of the level's
    /// vertices, counter-clockwise seen from above (from +y); then the skirt, two per step along
    /// each edge, counter-clockwise seen from outside the chunk.
    pub indices: Vec<u32>,
    /// How far, in engine units, the level's surface is at most above or below the grid's.
    pub error: f32,
}

/// The ground of `chunk` from a height field whose chunks `field` looks up, with cells `cell_size`
/// along the engine's x, y and z; a field's value is a height in cells.
///
/// Returns `None` until the field of `chunk` and of all eight chunks around it have arrived.
///
/// # Panics
/// If the fields do not all have the size of `chunk`'s field: a stage's chunks are one size.
#[must_use]
pub fn ground<'a>(
    chunk: ChunkCoord,
    field: impl Fn(ChunkCoord) -> Option<&'a Field>,
    cell_size: [f32; 3],
) -> Option<GroundMesh> {
    let own = field(chunk)?;
    let [sx, sy] = own.size;
    let mut around = [[None; 3]; 3];
    for (dy, row) in around.iter_mut().enumerate() {
        for (dx, slot) in row.iter_mut().enumerate() {
            let at = ChunkCoord::new(chunk.x + dx as i32 - 1, chunk.y + dy as i32 - 1, chunk.z);
            let neighbour = field(at)?;
            assert_eq!(
                neighbour.size, own.size,
                "the fields of one stage share a size"
            );
            *slot = Some(neighbour);
        }
    }
    // The height of a column relative to this chunk's, from -1 to one past the far edge.
    let height = |x: i64, y: i64| -> f32 {
        let (cx, cy) = (x.div_euclid(i64::from(sx)), y.div_euclid(i64::from(sy)));
        let neighbour = around[(cy + 1) as usize][(cx + 1) as usize]
            .expect("every neighbour was looked up above");
        neighbour.get(
            x.rem_euclid(i64::from(sx)) as u32,
            y.rem_euclid(i64::from(sy)) as u32,
        )
    };
    let [cell_x, cell_up, cell_z] = cell_size;
    let size = [sx + 1, sy + 1];
    let mut positions = Vec::with_capacity((size[0] * size[1]) as usize);
    let mut normals = Vec::with_capacity(positions.capacity());
    for j in 0..i64::from(size[1]) {
        for i in 0..i64::from(size[0]) {
            let up = height(i, j) * cell_up;
            positions.push([(i as f32 + 0.5) * cell_x, up, (j as f32 + 0.5) * cell_z]);
            let slope_x = (height(i + 1, j) - height(i - 1, j)) * cell_up / (2.0 * cell_x);
            let slope_z = (height(i, j + 1) - height(i, j - 1)) * cell_up / (2.0 * cell_z);
            let length = (slope_x * slope_x + 1.0 + slope_z * slope_z).sqrt();
            normals.push([-slope_x / length, 1.0 / length, -slope_z / length]);
        }
    }
    let heights: Vec<f32> = positions.iter().map(|position| position[1]).collect();
    let steps: Vec<u32> = (0..)
        .map(|power| 1_u32 << power)
        .take_while(|&step| {
            step <= sx.min(sy) && sx.is_multiple_of(step) && sy.is_multiple_of(step)
        })
        .collect();
    let errors: Vec<(f32, f32)> = steps
        .iter()
        .map(|&step| level_errors(&heights, size, step))
        .collect();
    let edge_error = errors.iter().map(|&(_, edge)| edge).fold(0.0, f32::max);
    // Every vertex of a coarser level is one of each finer level's, so along a shared edge two
    // neighbours at different levels part by at most the coarser level's error there. The margin
    // covers the cracks rasterisation leaves along an edge where one side has vertices the other
    // lacks, which open even where the heights agree.
    let depth = edge_error + SKIRT_MARGIN * cell_up;
    let edge = edge_vertices(size);
    let grid = positions.len() as u32;
    for &vertex in &edge {
        let [x, y, z] = positions[vertex as usize];
        positions.push([x, y - depth, z]);
        normals.push(normals[vertex as usize]);
    }
    let levels = steps
        .iter()
        .zip(errors)
        .map(|(&step, (error, _))| GroundLevel {
            step,
            indices: level_indices(size, step, &edge, grid),
            error,
        })
        .collect();
    Some(GroundMesh {
        chunk,
        size,
        positions,
        normals,
        levels,
        heights,
    })
}

/// How deep a skirt hangs beyond what the levels' edges need, in cells of height.
const SKIRT_MARGIN: f32 = 0.1;

/// The grid's vertices along its edge, once each, clockwise seen from above: along z = 0, up the
/// far x edge, back along the far z edge and down the x = 0 edge.
fn edge_vertices(size: [u32; 2]) -> Vec<u32> {
    let [w, h] = size;
    let along_near = 0..w - 1;
    let up_far = (0..h - 1).map(|j| j * w + w - 1);
    let back_far = (1..w).rev().map(|i| (h - 1) * w + i);
    let down_near = (1..h).rev().map(|j| j * w);
    along_near
        .chain(up_far)
        .chain(back_far)
        .chain(down_near)
        .collect()
}

/// A level's triangles: its surface over every `step`-th vertex of a grid of `size`, then its skirt
/// below the `edge` vertices, whose copies start at index `grid`.
fn level_indices(size: [u32; 2], step: u32, edge: &[u32], grid: u32) -> Vec<u32> {
    let [w, h] = size;
    let mut indices = Vec::new();
    for j in (0..h - 1).step_by(step as usize) {
        for i in (0..w - 1).step_by(step as usize) {
            let corner = j * w + i;
            let (right, below) = (corner + step, corner + step * w);
            indices.extend([corner, below, right, right, below, below + step]);
        }
    }
    // Seen from outside the chunk, a vertex, the next one along the edge and the vertex's skirt
    // copy go counter-clockwise, and so do the next one, its copy and the vertex's copy.
    let on_level =
        |vertex: u32| (vertex % w).is_multiple_of(step) && (vertex / w).is_multiple_of(step);
    let kept: Vec<usize> = (0..edge.len()).filter(|&at| on_level(edge[at])).collect();
    for (k, &at) in kept.iter().enumerate() {
        let next = kept[(k + 1) % kept.len()];
        let (a, b) = (edge[at], edge[next]);
        let (a_below, b_below) = (grid + at as u32, grid + next as u32);
        indices.extend([a, b, a_below, b, b_below, a_below]);
    }
    indices
}

/// How far a level of `step` strays from a grid of `size` with these `heights`: anywhere, and along
/// the grid's edges.
fn level_errors(heights: &[f32], size: [u32; 2], step: u32) -> (f32, f32) {
    let [w, h] = size;
    let at = |i: u32, j: u32| heights[(j * w + i) as usize];
    let (mut error, mut edge) = (0.0_f32, 0.0_f32);
    for j in 0..h {
        for i in 0..w {
            // The level's square holding the vertex; on the far edges, the square before it.
            let (i0, j0) = (
                (i / step).min((w - 1) / step - 1) * step,
                (j / step).min((h - 1) / step - 1) * step,
            );
            let (u, v) = ((i - i0) as f32 / step as f32, (j - j0) as f32 / step as f32);
            let (c00, c10, c01, c11) = (
                at(i0, j0),
                at(i0 + step, j0),
                at(i0, j0 + step),
                at(i0 + step, j0 + step),
            );
            // The square's triangles meet along the diagonal from (i0 + step, j0) to (i0, j0 + step).
            let level = if u + v <= 1.0 {
                c00 + u * (c10 - c00) + v * (c01 - c00)
            } else {
                c11 + (1.0 - u) * (c01 - c11) + (1.0 - v) * (c10 - c11)
            };
            let off = (at(i, j) - level).abs();
            error = error.max(off);
            if i == 0 || j == 0 || i == w - 1 || j == h - 1 {
                edge = edge.max(off);
            }
        }
    }
    (error, edge)
}

/// The category of every vertex of `chunk`'s ground, from a Rules stage at the height field's
/// scale whose chunks `categories` looks up: in the order of [`GroundMesh::positions`], so the
/// vertices along the +x and +y edges take the first columns of the chunks beyond them, as their
/// heights do.
///
/// Returns `None` until the categories of `chunk` and of the chunks beyond its +x edge, its +y edge
/// and its +x+y corner have arrived.
///
/// # Panics
/// If those chunks do not all have one size: a stage's chunks are one size.
#[must_use]
pub fn ground_materials<'a>(
    chunk: ChunkCoord,
    categories: impl Fn(ChunkCoord) -> Option<&'a Categories>,
) -> Option<Vec<u8>> {
    let own = categories(chunk)?;
    let [sx, sy] = own.size;
    let mut beyond = [[None; 2]; 2];
    for (dy, row) in beyond.iter_mut().enumerate() {
        for (dx, slot) in row.iter_mut().enumerate() {
            let at = ChunkCoord::new(chunk.x + dx as i32, chunk.y + dy as i32, chunk.z);
            let neighbour = categories(at)?;
            assert_eq!(
                neighbour.size, own.size,
                "the categories of one stage share a size"
            );
            *slot = Some(neighbour);
        }
    }
    let mut materials = Vec::with_capacity(((sx + 1) * (sy + 1)) as usize);
    for j in 0..=sy {
        for i in 0..=sx {
            let (dx, dy) = (usize::from(i == sx), usize::from(j == sy));
            let from = beyond[dy][dx].expect("every neighbour was looked up above");
            materials.push(from.get(i % sx, j % sy));
        }
    }
    Some(materials)
}

/// The chunks whose ground reads the field of `chunk`: itself and the eight around it. When that
/// field arrives, these are the chunks whose ground may have become buildable.
#[must_use]
pub fn ground_readers(chunk: ChunkCoord) -> [ChunkCoord; 9] {
    std::array::from_fn(|index| {
        let (dx, dy) = (index as i32 % 3 - 1, index as i32 / 3 - 1);
        ChunkCoord::new(chunk.x + dx, chunk.y + dy, chunk.z)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    const SIZE: [u32; 2] = [4, 3];
    const CELL: [f32; 3] = [2.0, 0.5, 3.0];

    /// A height field that is a function of the world column, as every pure field stage is.
    fn fields(height: impl Fn(i64, i64) -> f32) -> BTreeMap<ChunkCoord, Field> {
        fields_of(SIZE, height)
    }

    /// The same, with chunks of `size` columns.
    fn fields_of(size: [u32; 2], height: impl Fn(i64, i64) -> f32) -> BTreeMap<ChunkCoord, Field> {
        let mut out = BTreeMap::new();
        for cy in -2..=2 {
            for cx in -2..=2 {
                let chunk = ChunkCoord::new(cx, cy, 0);
                let values = (0..size[1])
                    .flat_map(|y| (0..size[0]).map(move |x| (x, y)))
                    .map(|(x, y)| {
                        height(
                            i64::from(cx) * i64::from(size[0]) + i64::from(x),
                            i64::from(cy) * i64::from(size[1]) + i64::from(y),
                        )
                    })
                    .collect();
                out.insert(
                    chunk,
                    Field {
                        chunk,
                        size,
                        values,
                    },
                );
            }
        }
        out
    }

    /// Chunks of 8 by 4 columns, which have levels of steps 1, 2 and 4.
    const LEVELLED: [u32; 2] = [8, 4];

    fn hilly(x: i64, y: i64) -> f32 {
        ((x * 7 + y * 13) % 5) as f32 + 0.25 * y as f32
    }

    fn build(fields: &BTreeMap<ChunkCoord, Field>, chunk: ChunkCoord) -> GroundMesh {
        ground(chunk, |at| fields.get(&at), CELL).expect("every neighbour is there")
    }

    #[test]
    fn neighbouring_chunks_share_the_vertices_and_normals_of_their_common_edge() {
        let fields = fields(hilly);
        let here = build(&fields, ChunkCoord::new(0, 0, 0));
        let east = build(&fields, ChunkCoord::new(1, 0, 0));
        let north = build(&fields, ChunkCoord::new(0, 1, 0));
        let [w, h] = here.size;
        let chunk_x = SIZE[0] as f32 * CELL[0];
        let chunk_z = SIZE[1] as f32 * CELL[2];

        for j in 0..h {
            let (mine, theirs) = ((j * w + w - 1) as usize, (j * w) as usize);
            let [x, y, z] = here.positions[mine];
            assert_eq!(
                [x - chunk_x, y, z],
                east.positions[theirs],
                "east edge, row {j}"
            );
            assert_eq!(
                here.normals[mine], east.normals[theirs],
                "east normal, row {j}"
            );
        }
        for i in 0..w {
            let (mine, theirs) = (((h - 1) * w + i) as usize, i as usize);
            let [x, y, z] = here.positions[mine];
            assert_eq!(
                [x, y, z - chunk_z],
                north.positions[theirs],
                "north edge, column {i}"
            );
            assert_eq!(
                here.normals[mine], north.normals[theirs],
                "north normal, column {i}"
            );
        }
    }

    /// A triangle's normal, unnormalised, on the side it is drawn from.
    fn facing(mesh: &GroundMesh, triangle: &[u32]) -> [f32; 3] {
        let [a, b, c] = [0, 1, 2].map(|k| mesh.positions[triangle[k] as usize]);
        let (u, v) = (
            [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
            [c[0] - a[0], c[1] - a[1], c[2] - a[2]],
        );
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    }

    #[test]
    fn every_surface_triangle_faces_up_and_every_skirt_triangle_faces_out() {
        let fields = fields_of(LEVELLED, hilly);
        let mesh = build(&fields, ChunkCoord::new(-1, 1, 0));
        let grid = (mesh.size[0] * mesh.size[1]) as usize;
        let centre = [
            LEVELLED[0] as f32 * CELL[0] / 2.0 + CELL[0] / 2.0,
            LEVELLED[1] as f32 * CELL[2] / 2.0 + CELL[2] / 2.0,
        ];

        for level in &mesh.levels {
            for triangle in level.indices.chunks(3) {
                let normal = facing(&mesh, triangle);
                if triangle.iter().all(|&vertex| (vertex as usize) < grid) {
                    assert!(
                        normal[1] > 0.0,
                        "step {}: {triangle:?} faces down",
                        level.step
                    );
                } else {
                    let [a, b, c] = [0, 1, 2].map(|k| mesh.positions[triangle[k] as usize]);
                    let middle = [(a[0] + b[0] + c[0]) / 3.0, (a[2] + b[2] + c[2]) / 3.0];
                    let out = [middle[0] - centre[0], middle[1] - centre[1]];
                    assert!(
                        normal[1].abs() < 1e-4,
                        "step {}: {triangle:?} is not upright",
                        level.step
                    );
                    assert!(
                        normal[0] * out[0] + normal[2] * out[1] > 0.0,
                        "step {}: {triangle:?} faces in",
                        level.step
                    );
                }
            }
        }
    }

    #[test]
    fn a_level_takes_every_step_th_vertex() {
        let fields = fields_of(LEVELLED, hilly);
        let mesh = build(&fields, ChunkCoord::new(0, 0, 0));
        let [w, _] = mesh.size;
        let grid = mesh.size[0] * mesh.size[1];

        let steps: Vec<u32> = mesh.levels.iter().map(|level| level.step).collect();
        assert_eq!(steps, [1, 2, 4]);
        for level in &mesh.levels {
            let surface: Vec<u32> = level
                .indices
                .chunks(3)
                .filter(|triangle| triangle.iter().all(|&vertex| vertex < grid))
                .flatten()
                .copied()
                .collect();
            let squares = (LEVELLED[0] / level.step) * (LEVELLED[1] / level.step);
            assert_eq!(surface.len() as u32, squares * 6, "step {}", level.step);
            for vertex in surface {
                assert_eq!((vertex % w) % level.step, 0, "step {}", level.step);
                assert_eq!((vertex / w) % level.step, 0, "step {}", level.step);
            }
        }
    }

    #[test]
    fn a_chunk_whose_columns_do_not_halve_has_only_full_detail() {
        let fields = fields(hilly);
        let mesh = build(&fields, ChunkCoord::new(0, 0, 0));

        let steps: Vec<u32> = mesh.levels.iter().map(|level| level.step).collect();
        assert_eq!(steps, [1]);
        assert_eq!(mesh.levels[0].error, 0.0);
    }

    #[test]
    fn a_levels_error_is_how_far_its_surface_strays_from_full_detail() {
        // Flat but for one column between the level of step 2's vertices, 4 cells high.
        let fields = fields_of(LEVELLED, |x, y| if (x, y) == (3, 1) { 4.0 } else { 0.0 });
        let mesh = build(&fields, ChunkCoord::new(0, 0, 0));

        let errors: Vec<f32> = mesh.levels.iter().map(|level| level.error).collect();
        assert_eq!(errors, [0.0, 4.0 * CELL[1], 4.0 * CELL[1]]);
    }

    #[test]
    fn a_plane_is_the_same_at_every_level() {
        let fields = fields_of(LEVELLED, |x, y| 0.5 * x as f32 - 2.0 * y as f32);
        let mesh = build(&fields, ChunkCoord::new(1, -1, 0));

        for level in &mesh.levels {
            assert!(level.error < 1e-4, "step {}: {}", level.step, level.error);
        }
    }

    /// The heights along a chunk's shared edge at a level, one per grid vertex: the level's vertices
    /// and straight lines between them, as its triangles' edges run.
    fn along_edge(heights: &[f32], step: u32) -> Vec<f32> {
        (0..heights.len())
            .map(|at| {
                let low = at / step as usize * step as usize;
                let high = (low + step as usize).min(heights.len() - 1);
                if low == high {
                    return heights[low];
                }
                let t = (at - low) as f32 / (high - low) as f32;
                heights[low] + t * (heights[high] - heights[low])
            })
            .collect()
    }

    /// How far below the grid a chunk's skirt hangs.
    fn skirt_depth(mesh: &GroundMesh) -> f32 {
        let grid = (mesh.size[0] * mesh.size[1]) as usize;
        mesh.positions[0][1] - mesh.positions[grid][1]
    }

    #[test]
    fn a_skirt_closes_the_gap_to_a_neighbour_at_any_level() {
        let fields = fields_of(LEVELLED, |x, y| ((x * 7 + y * 13) % 5) as f32 * 1.5);
        let here = build(&fields, ChunkCoord::new(0, 0, 0));
        let east = build(&fields, ChunkCoord::new(1, 0, 0));
        let [w, h] = here.size;
        let edge: Vec<f32> = (0..h)
            .map(|j| here.heights[(j * w + w - 1) as usize])
            .collect();

        for mine in &here.levels {
            for theirs in &east.levels {
                let (a, b) = (along_edge(&edge, mine.step), along_edge(&edge, theirs.step));
                for (row, (a, b)) in a.iter().zip(&b).enumerate() {
                    let depth = if a >= b {
                        skirt_depth(&here)
                    } else {
                        skirt_depth(&east)
                    };
                    assert!(
                        (a - b).abs() < depth,
                        "steps {} and {}, row {row}: a gap of {} under a skirt of {depth}",
                        mine.step,
                        theirs.step,
                        (a - b).abs()
                    );
                }
            }
        }
    }

    #[test]
    fn a_skirt_vertex_hangs_below_an_edge_vertex_with_its_normal() {
        let fields = fields_of(LEVELLED, hilly);
        let mesh = build(&fields, ChunkCoord::new(0, 0, 0));
        let [w, h] = mesh.size;
        let grid = (w * h) as usize;
        let depth = skirt_depth(&mesh);

        assert_eq!(mesh.positions.len() - grid, (2 * (w + h) - 4) as usize);
        for skirt in grid..mesh.positions.len() {
            let [x, y, z] = mesh.positions[skirt];
            let above = (0..grid)
                .find(|&vertex| {
                    let [ex, _, ez] = mesh.positions[vertex];
                    ex == x && ez == z
                })
                .expect("an edge vertex above");
            assert!((mesh.positions[above][1] - y - depth).abs() < 1e-5);
            assert_eq!(mesh.normals[skirt], mesh.normals[above]);
        }
    }

    #[test]
    fn a_vertex_stands_on_its_columns_height_at_the_columns_centre() {
        let fields = fields(hilly);
        let chunk = ChunkCoord::new(1, -1, 0);
        let mesh = build(&fields, chunk);

        let [i, j] = [2_u32, 1];
        let vertex = mesh.positions[(j * mesh.size[0] + i) as usize];
        let column = fields[&chunk].get(i, j);
        assert_eq!(vertex, [2.5 * CELL[0], column * CELL[1], 1.5 * CELL[2]]);
        assert_eq!(mesh.heights[(j * mesh.size[0] + i) as usize], vertex[1]);
    }

    #[test]
    fn a_plane_has_the_normal_of_its_slope() {
        // Rising one cell of height per cell along x: in engine units 0.5 up per 2 across.
        let fields = fields(|x, _| x as f32);
        let mesh = build(&fields, ChunkCoord::new(0, 0, 0));

        let expected = {
            let length = (0.25_f32 * 0.25 + 1.0).sqrt();
            [-0.25 / length, 1.0 / length, 0.0]
        };
        for normal in &mesh.normals {
            for (got, want) in normal.iter().zip(expected) {
                assert!((got - want).abs() < 1e-6, "{normal:?} against {expected:?}");
            }
        }
    }

    /// Categories that are a function of the world column, as every Rules stage's are.
    fn categories(category: impl Fn(i64, i64) -> u8) -> BTreeMap<ChunkCoord, Categories> {
        let mut out = BTreeMap::new();
        for cy in -2..=2 {
            for cx in -2..=2 {
                let chunk = ChunkCoord::new(cx, cy, 0);
                let values = (0..SIZE[1])
                    .flat_map(|y| (0..SIZE[0]).map(move |x| (x, y)))
                    .map(|(x, y)| {
                        category(
                            i64::from(cx) * i64::from(SIZE[0]) + i64::from(x),
                            i64::from(cy) * i64::from(SIZE[1]) + i64::from(y),
                        )
                    })
                    .collect();
                out.insert(
                    chunk,
                    Categories {
                        chunk,
                        size: SIZE,
                        values,
                    },
                );
            }
        }
        out
    }

    #[test]
    fn every_vertex_takes_the_category_of_the_column_it_stands_over() {
        let banded = |x: i64, y: i64| ((x.div_euclid(3) + 2 * y.div_euclid(2)).rem_euclid(5)) as u8;
        let all = categories(banded);
        let chunk = ChunkCoord::new(-1, 1, 0);

        let materials = ground_materials(chunk, |at| all.get(&at)).expect("all arrived");

        let size = [SIZE[0] + 1, SIZE[1] + 1];
        assert_eq!(materials.len(), (size[0] * size[1]) as usize);
        for j in 0..size[1] {
            for i in 0..size[0] {
                let column = (
                    i64::from(chunk.x) * i64::from(SIZE[0]) + i64::from(i),
                    i64::from(chunk.y) * i64::from(SIZE[1]) + i64::from(j),
                );
                assert_eq!(
                    materials[(j * size[0] + i) as usize],
                    banded(column.0, column.1),
                    "vertex ({i}, {j})"
                );
            }
        }
    }

    #[test]
    fn no_materials_until_the_chunks_beyond_the_far_edges_have_arrived() {
        let mut all = categories(|_, _| 1);
        let chunk = ChunkCoord::new(0, 0, 0);
        all.remove(&ChunkCoord::new(1, 1, 0));

        let materials = ground_materials(chunk, |at| all.get(&at));

        assert_eq!(materials, None);
    }

    #[test]
    fn nothing_is_built_until_every_neighbour_has_arrived() {
        let mut fields = fields(hilly);
        fields.remove(&ChunkCoord::new(1, 1, 0));

        assert!(ground(ChunkCoord::new(0, 0, 0), |at| fields.get(&at), CELL).is_none());
        assert!(ground(ChunkCoord::new(-1, -1, 0), |at| fields.get(&at), CELL).is_some());
    }

    #[test]
    fn the_readers_of_a_field_are_the_chunks_whose_ground_needs_it() {
        let fields = fields(hilly);
        let arrived = ChunkCoord::new(0, 0, 0);
        let readers = ground_readers(arrived);

        for cy in -1..=1 {
            for cx in -1..=1 {
                let chunk = ChunkCoord::new(cx, cy, 0);
                let without = ground(chunk, |at| (at != arrived).then(|| &fields[&at]), CELL);
                assert_eq!(
                    without.is_none(),
                    readers.contains(&chunk),
                    "chunk {chunk:?}"
                );
            }
        }
    }
}
