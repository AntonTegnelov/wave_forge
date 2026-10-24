//! The ground beyond the near ground, from a coarse height field (docs/reference/packs.md,
//! "Levels"): one mesh per chunk of the coarse stage, drawn where no near ground is.
//!
//! A coarse chunk covers `scale` by `scale` chunks of the WFC lattice. Its far ground has a vertex at
//! the corner of every one of them, the centre of that chunk's first column, where a near ground's
//! first vertex is ([`fn@crate::ground`]); and where a lattice chunk spans several coarse columns, a
//! vertex for each of them along its edges and inside, so the far ground follows the coarse field at
//! its own resolution. Every vertex has the height a fine stage reading the coarse field there would
//! get, linearly between the four coarse columns around it. So each lattice chunk is a block of far
//! squares with exactly a near ground's outline. The squares of the chunks
//! whose near ground is drawn are left out, so the two never overlap; and along every edge a far
//! square shares with a near ground, a wall joins the near ground's edge, at full detail, to the far
//! square's, so no view sees between them. A near ground drawn at a coarser level of detail hangs
//! its own skirt below that edge, which covers what the wall's top misses.
//!
//! Two neighbouring coarse chunks compute the same heights for the corners they share, so their far
//! grounds meet without a seam.

use crate::ground::GroundMesh;
use crate::stages::{Categories, Field};
use std::collections::BTreeSet;
use wfc_core::ChunkCoord;

/// One coarse chunk's far ground, in a Y-up engine's axes as [`crate::YUpSpace`] maps them.
#[derive(Clone, Debug, PartialEq)]
pub struct FarGround {
    /// The chunk of the coarse stage.
    pub chunk: ChunkCoord,
    /// Relative to the corner of the lattice chunk at the coarse chunk's lowest corner, on the
    /// ground plane (the engine's x and z), with the height absolute: first the grid, `scale * n + 1`
    /// vertices each way for `n` coarse columns to a lattice chunk ([`far_steps`]), row by row along z, x fastest; then the walls' vertices.
    pub positions: Vec<[f32; 3]>,
    /// Unit normals, one per vertex.
    pub normals: Vec<[f32; 3]>,
    /// Triangles into `positions`: the surface's, counter-clockwise seen from above, and each
    /// wall's, once facing either way.
    pub indices: Vec<u32>,
}

/// The far ground of the coarse `chunk`, from a coarse height field of `scale` whose chunks `field`
/// looks up, with cells `cell_size` along the engine's x, y and z; a field's value is a height in
/// cells. `near` gives the near ground drawn on a chunk of the WFC lattice, if there is one: that
/// chunk is left out, and walled off where it meets the far ground.
///
/// Returns `None` until the field of `chunk` and of all eight chunks around it have arrived, since
/// the heights along its edges read theirs.
///
/// # Panics
/// If the fields do not all have one size, if that size is not square, or if `scale` is zero.
#[must_use]
pub fn far_ground<'a>(
    chunk: ChunkCoord,
    scale: u32,
    field: impl Fn(ChunkCoord) -> Option<&'a Field>,
    cell_size: [f32; 3],
    near: impl Fn(ChunkCoord) -> Option<&'a GroundMesh>,
) -> Option<FarGround> {
    assert!(
        scale > 0,
        "a coarse stage's scale is a whole number of cells"
    );
    let own = field(chunk)?;
    // A lattice chunk's far squares are counted, spaced and read along x and used along z too.
    assert_eq!(
        own.size[0], own.size[1],
        "the far ground is built for square chunks"
    );
    let [columns_x, columns_y] = own.size.map(i64::from);
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
    // A coarse column's height, in columns relative to this chunk's first.
    let coarse = |x: i64, y: i64| -> f32 {
        let (cx, cy) = (x.div_euclid(columns_x), y.div_euclid(columns_y));
        around[(cy + 1) as usize][(cx + 1) as usize]
            .expect("every neighbour was looked up above")
            .get(
                x.rem_euclid(columns_x) as u32,
                y.rem_euclid(columns_y) as u32,
            )
    };
    // The height at the centre of a fine column, in columns of the WFC lattice relative to this
    // chunk's first, as a fine stage reads a coarser field: linearly between the four coarse
    // columns around it.
    let ratio = f64::from(scale);
    let height = |x: i64, y: i64| -> f32 {
        let place = |column: i64| (column as f64 + 0.5) / ratio - 0.5;
        let (u, v) = (place(x), place(y));
        let (i, j) = (u.floor() as i64, v.floor() as i64);
        let (s, t) = ((u - i as f64) as f32, (v - j as f64) as f32);
        let bottom = coarse(i, j) + (coarse(i + 1, j) - coarse(i, j)) * s;
        let top = coarse(i, j + 1) + (coarse(i + 1, j + 1) - coarse(i, j + 1)) * s;
        bottom + (top - bottom) * t
    };
    let [cell_x, cell_up, cell_z] = cell_size;
    // A lattice chunk has as many columns as a coarse one, and `steps` far squares along each side.
    let steps = far_steps(own.size[0], scale);
    let (span_x, span_z) = (columns_x as f32 * cell_x, columns_x as f32 * cell_z);
    let (step_x, step_z) = (span_x / steps as f32, span_z / steps as f32);
    let stride = columns_x / i64::from(steps);
    let side = scale * steps + 1;
    let first = ChunkCoord::new(chunk.x * scale as i32, chunk.y * scale as i32, chunk.z);
    // The height of the far ground's vertex `i`, `j`, which may lie a step outside the chunk for
    // a normal at its edge: the coarse neighbours cover that.
    let corner_height = |i: i64, j: i64| height(i * stride, j * stride) * cell_up;
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    for j in 0..i64::from(side) {
        for i in 0..i64::from(side) {
            positions.push([
                i as f32 * step_x + 0.5 * cell_x,
                corner_height(i, j),
                j as f32 * step_z + 0.5 * cell_z,
            ]);
            let slope_x = (corner_height(i + 1, j) - corner_height(i - 1, j)) / (2.0 * step_x);
            let slope_z = (corner_height(i, j + 1) - corner_height(i, j - 1)) / (2.0 * step_z);
            let length = (slope_x * slope_x + 1.0 + slope_z * slope_z).sqrt();
            normals.push([-slope_x / length, 1.0 / length, -slope_z / length]);
        }
    }
    // Where a far vertex stands on a near ground's vertex it takes that vertex's height and normal,
    // so the far ground meets the near ground's edge with no wall between them and the step between
    // the two spreads over the far squares beside it. Every far chunk does the same, so neighbours
    // still meet exactly.
    for j in 0..i64::from(side) {
        for i in 0..i64::from(side) {
            let (column, row) = (i * stride, j * stride);
            if let Some((height, normal)) = near_vertex(&near, first, own.size, column, row) {
                let vertex = (j * i64::from(side) + i) as usize;
                positions[vertex][1] = height;
                normals[vertex] = normal;
            }
        }
    }
    let lattice = |i: u32, j: u32| ChunkCoord::new(first.x + i as i32, first.y + j as i32, first.z);
    let vertex = |i: u32, j: u32| j * side + i;
    let mut indices = Vec::new();
    for j in 0..scale {
        for i in 0..scale {
            if near(lattice(i, j)).is_some() {
                continue;
            }
            let here = lattice(i, j);
            let beside_near = [(-1, 0), (0, 1), (1, 0), (0, -1)].iter().any(|&(dx, dy)| {
                near(ChunkCoord::new(here.x + dx, here.y + dy, first.z)).is_some()
            });
            let (gi, gj) = (i * steps, j * steps);
            if !beside_near {
                for b in 0..steps {
                    for a in 0..steps {
                        let corner = vertex(gi + a, gj + b);
                        let (right, below) = (corner + 1, corner + side);
                        indices.extend([corner, below, right, right, below, below + 1]);
                    }
                }
                continue;
            }
            let corner = vertex(gi, gj);
            let (right, below) = (vertex(gi + steps, gj), vertex(gi, gj + steps));
            let far_corner = vertex(gi + steps, gj + steps);
            // The block's edges in the order its surface winds, counter-clockwise seen from above:
            // each edge's neighbour across it, and the far vertices along it from its first corner up
            // to its last.
            let edges = [
                (
                    (-1, 0),
                    (0..steps).map(|k| vertex(gi, gj + k)).collect::<Vec<u32>>(),
                ),
                (
                    (0, 1),
                    (0..steps)
                        .map(|k| vertex(gi + k, gj + steps))
                        .collect::<Vec<u32>>(),
                ),
                (
                    (1, 0),
                    (0..steps)
                        .map(|k| vertex(gi + steps, gj + steps - k))
                        .collect::<Vec<u32>>(),
                ),
                (
                    (0, -1),
                    (0..steps)
                        .map(|k| vertex(gi + steps - k, gj))
                        .collect::<Vec<u32>>(),
                ),
            ];
            // The block's outline: along an edge shared with a near ground, the top of the curtain
            // under it, one vertex on each of the near edge's at its height; along any other, its far
            // vertices, which the far squares beside it share.
            let mut outline = Vec::new();
            for ((dx, dy), along) in edges {
                let across = ChunkCoord::new(here.x + dx, here.y + dy, first.z);
                let Some(ground) = near(across) else {
                    outline.extend(along);
                    continue;
                };
                let feet = wall(
                    &mut positions,
                    &mut normals,
                    &mut indices,
                    ground,
                    (dx, dy),
                    [
                        (across.x - first.x) as f32 * span_x,
                        (across.y - first.y) as f32 * span_z,
                    ],
                );
                outline.extend(&feet[..feet.len() - 1]);
            }
            // A fan from the block's centre through every vertex of its outline, so it shares its
            // edges' vertices with the walls and the far squares beside it, and no crack opens.
            let centre = positions.len() as u32;
            let corners = [corner, right, below, far_corner].map(|v| positions[v as usize]);
            let average = |axis: usize| corners.iter().map(|p| p[axis]).sum::<f32>() / 4.0;
            positions.push([average(0), average(1), average(2)]);
            normals.push(unit(
                [corner, right, below, far_corner]
                    .iter()
                    .map(|&v| normals[v as usize])
                    .fold([0.0; 3], |a, n| [a[0] + n[0], a[1] + n[1], a[2] + n[2]]),
            ));
            for k in 0..outline.len() {
                indices.extend([centre, outline[k], outline[(k + 1) % outline.len()]]);
            }
        }
    }
    Some(FarGround {
        chunk,
        positions,
        normals,
        indices,
    })
}

/// The coarse chunks at `scale` whose far ground reads the near ground of the lattice chunk `fine`,
/// which an engine builds again when that near ground arrives, changes or goes: those holding
/// `fine` or any of the eight lattice chunks around it, since a far ground leaves the near ground's
/// block out, hangs curtains on the blocks beside it and takes its heights at every far vertex on
/// its edges and corners.
///
/// # Panics
/// If `scale` is zero.
#[must_use]
pub fn far_chunks_reading(fine: ChunkCoord, scale: u32) -> BTreeSet<ChunkCoord> {
    assert!(
        scale > 0,
        "a coarse stage's scale is a whole number of cells"
    );
    let scale = scale as i32;
    (-1..=1)
        .flat_map(|dy| (-1..=1).map(move |dx| (dx, dy)))
        .map(|(dx, dy)| {
            ChunkCoord::new(
                (fine.x + dx).div_euclid(scale),
                (fine.y + dy).div_euclid(scale),
                fine.z,
            )
        })
        .collect()
}

/// How many far squares a far ground of a coarse stage at `scale`, whose chunks have `columns`
/// columns along each side, has along each side of a lattice chunk: one per coarse column the
/// lattice chunk spans, or one where it spans less than one or not a whole number of them.
#[must_use]
pub fn far_steps(columns: u32, scale: u32) -> u32 {
    if scale > 0 && columns >= scale && columns.is_multiple_of(scale) {
        columns / scale
    } else {
        1
    }
}

/// Hangs a curtain below a near ground's edge facing across `(dx, dy)`, which the far square meets,
/// and returns the curtain's tops along it, running the way the far square's edge does: a vertex on
/// each of the near edge's, at its height, which the far square's surface takes as its edge. The
/// curtain hangs as far as the near ground's skirt, which is as far as any of its levels of detail
/// strays from the full-detail edge, so it closes the gap to whichever level the near ground is
/// drawn at. The near ground's positions are moved by `offset` into the far ground's frame.
fn wall(
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
    ground: &GroundMesh,
    (dx, dy): (i32, i32),
    offset: [f32; 2],
) -> Vec<u32> {
    let [w, h] = ground.size;
    // The near ground's edge facing the far square, as grid vertices of it, running the way the
    // far edge does from `ends[0]` to `ends[1]`.
    let mut edge: Vec<u32> = match (dx, dy) {
        (1, 0) => (0..h).map(|k| k * w).collect(),
        (-1, 0) => (0..h).map(|k| k * w + w - 1).collect(),
        (0, 1) => (0..w).collect(),
        _ => (0..w).map(|k| (h - 1) * w + k).collect(),
    };
    if (dx, dy) == (1, 0) || (dx, dy) == (0, -1) {
        edge.reverse();
    }
    // The skirt's first vertex is the grid's first, hung below it.
    let depth = ground.positions[0][1] - ground.positions[(w * h) as usize][1];
    // Down each place along the edge: the curtain's top, at the near edge, and its bottom.
    let mut columns = Vec::with_capacity(edge.len());
    for &vertex in &edge {
        let [x, near, z] = ground.positions[vertex as usize];
        let normal = ground.normals[vertex as usize];
        let column = [near, near - depth].map(|height| {
            positions.push([x + offset[0], height, z + offset[1]]);
            normals.push(normal);
            positions.len() as u32 - 1
        });
        columns.push(column);
    }
    for pair in columns.windows(2) {
        let ([a_top, a_low], [b_top, b_low]) = (pair[0], pair[1]);
        indices.extend([a_top, b_top, a_low, b_top, b_low, a_low]);
        indices.extend([a_top, a_low, b_top, b_top, a_low, b_low]);
    }
    columns.iter().map(|&[top, _]| top).collect()
}

/// The height and normal of the near ground's vertex standing at `column`, `row`, counted in columns
/// of the WFC lattice from the corner of the lattice chunk `first`, if any near ground has one there:
/// a vertex on a lattice chunk's edge belongs to each chunk the edge bounds.
fn near_vertex<'a>(
    near: &impl Fn(ChunkCoord) -> Option<&'a GroundMesh>,
    first: ChunkCoord,
    [columns_x, columns_y]: [u32; 2],
    column: i64,
    row: i64,
) -> Option<(f32, [f32; 3])> {
    let (columns_x, columns_y) = (i64::from(columns_x), i64::from(columns_y));
    // The chunks whose squares hold the vertex: the one it lies in, and the one before it along an
    // axis where it lies on their shared edge.
    let chunks = |at: i64, columns: i64| {
        let chunk = at.div_euclid(columns);
        if at.rem_euclid(columns) == 0 {
            vec![chunk - 1, chunk]
        } else {
            vec![chunk]
        }
    };
    for cy in chunks(row, columns_y) {
        for cx in chunks(column, columns_x) {
            let at = ChunkCoord::new(first.x + cx as i32, first.y + cy as i32, first.z);
            let Some(ground) = near(at) else {
                continue;
            };
            let (i, j) = (column - cx * columns_x, row - cy * columns_y);
            let vertex = (j * i64::from(ground.size[0]) + i) as usize;
            return Some((ground.positions[vertex][1], ground.normals[vertex]));
        }
    }
    None
}

/// `v` scaled to unit length.
fn unit(v: [f32; 3]) -> [f32; 3] {
    let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    v.map(|axis| axis / length)
}

/// The category each vertex of `far` stands on: that of the column under it of `categories`, the
/// chunk of a Rules stage at the far ground's `scale`, with cells `cell_size` along the engine's x,
/// y and z. A vertex on the chunk's far edges, which it shares with the next chunk, takes the edge
/// column's; so does a wall's, from where it stands. An engine colours the far ground by these, as
/// it colours the near ground by a Rules stage's categories.
///
/// # Panics
/// If `categories` has no columns, or `scale` is zero.
#[must_use]
pub fn far_ground_categories(
    far: &FarGround,
    categories: &Categories,
    scale: u32,
    cell_size: [f32; 3],
) -> Vec<u8> {
    far_columns(far, categories.size, scale, cell_size)
        .map(|[x, y]| categories.get(x, y))
        .collect()
}

/// The value each vertex of `far` stands on: that of the column under it of `field`, the chunk of
/// a field stage at the far ground's `scale`, taken as [`far_ground_categories`] takes categories.
/// An engine shades the far ground by these, as it shades the near ground by a field's channels.
///
/// # Panics
/// If `field` has no columns, or `scale` is zero.
#[must_use]
pub fn far_ground_values(
    far: &FarGround,
    field: &Field,
    scale: u32,
    cell_size: [f32; 3],
) -> Vec<f32> {
    far_columns(far, field.size, scale, cell_size)
        .map(|[x, y]| field.get(x, y))
        .collect()
}

/// The column of a chunk of `size` columns at the far ground's `scale` under each vertex of `far`,
/// the chunk's far edges' and walls' clamped into it.
fn far_columns(
    far: &FarGround,
    size: [u32; 2],
    scale: u32,
    cell_size: [f32; 3],
) -> impl Iterator<Item = [u32; 2]> + '_ {
    let column = move |at: f32, cell: f32, columns: u32| {
        ((at / (cell * scale as f32)).floor().max(0.0) as u32).min(columns - 1)
    };
    far.positions.iter().map(move |&[x, _, z]| {
        [
            column(x, cell_size[0], size[0]),
            column(z, cell_size[2], size[1]),
        ]
    })
}
