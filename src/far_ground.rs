//! The ground beyond the near ground, from a coarse height field (docs/reference/packs.md,
//! "Levels"): one mesh per chunk of the coarse stage, drawn where no near ground is.
//!
//! A coarse chunk covers `scale` by `scale` chunks of the WFC lattice. Its far ground has a vertex at
//! the corner of every one of them: the centre of that chunk's first column, where a near ground's
//! first vertex is ([`fn@crate::ground`]), with the height a fine stage reading the coarse field there
//! would get, linearly between the four coarse columns around it. So each lattice chunk is one
//! square of the far ground, and it has exactly a near ground's outline. The squares of the chunks
//! whose near ground is drawn are left out, so the two never overlap; and along every edge a far
//! square shares with a near ground, a wall joins the near ground's edge, at full detail, to the far
//! square's, so no view sees between them. A near ground drawn at a coarser level of detail hangs
//! its own skirt below that edge, which covers what the wall's top misses.
//!
//! Two neighbouring coarse chunks compute the same heights for the corners they share, so their far
//! grounds meet without a seam.

use crate::ground::GroundMesh;
use crate::stages::Field;
use wfc_core::ChunkCoord;

/// One coarse chunk's far ground, in a Y-up engine's axes as [`crate::YUpSpace`] maps them.
#[derive(Clone, Debug, PartialEq)]
pub struct FarGround {
    /// The chunk of the coarse stage.
    pub chunk: ChunkCoord,
    /// Relative to the corner of the lattice chunk at the coarse chunk's lowest corner, on the
    /// ground plane (the engine's x and z), with the height absolute: first the grid, `scale + 1`
    /// vertices each way, row by row along z, x fastest; then the walls' vertices.
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
/// If the fields do not all have one size, or `scale` is zero.
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
    // A lattice chunk has as many columns as a coarse one.
    let (span_x, span_z) = (columns_x as f32 * cell_x, columns_x as f32 * cell_z);
    let side = scale + 1;
    let first = ChunkCoord::new(chunk.x * scale as i32, chunk.y * scale as i32, chunk.z);
    // The height of the far ground's vertex `i`, `j`, which may lie a step outside the chunk for
    // a normal at its edge: the coarse neighbours cover that.
    let corner_height = |i: i64, j: i64| height(i * columns_x, j * columns_y) * cell_up;
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    for j in 0..i64::from(side) {
        for i in 0..i64::from(side) {
            positions.push([
                i as f32 * span_x + 0.5 * cell_x,
                corner_height(i, j),
                j as f32 * span_z + 0.5 * cell_z,
            ]);
            let slope_x = (corner_height(i + 1, j) - corner_height(i - 1, j)) / (2.0 * span_x);
            let slope_z = (corner_height(i, j + 1) - corner_height(i, j - 1)) / (2.0 * span_z);
            let length = (slope_x * slope_x + 1.0 + slope_z * slope_z).sqrt();
            normals.push([-slope_x / length, 1.0 / length, -slope_z / length]);
        }
    }
    let lattice = |i: u32, j: u32| ChunkCoord::new(first.x + i as i32, first.y + j as i32, first.z);
    let mut indices = Vec::new();
    for j in 0..scale {
        for i in 0..scale {
            if near(lattice(i, j)).is_some() {
                continue;
            }
            let corner = j * side + i;
            let (right, below) = (corner + 1, corner + side);
            // The square's corners in the order its surface winds, counter-clockwise seen from
            // above, and each edge's neighbour across it.
            let edges = [
                ((-1_i32, 0_i32), corner, below),
                ((0, 1), below, below + 1),
                ((1, 0), below + 1, right),
                ((0, -1), right, corner),
            ];
            // The square's outline: along an edge shared with a near ground, the far foot of its
            // wall, one vertex under or over each of the near edge's; along any other, its corner.
            let mut outline = Vec::new();
            for ((dx, dy), from, to) in edges {
                let here = lattice(i, j);
                let across = ChunkCoord::new(here.x + dx, here.y + dy, first.z);
                let Some(ground) = near(across) else {
                    outline.push(from);
                    continue;
                };
                let feet = wall(
                    &mut positions,
                    &mut normals,
                    &mut indices,
                    ground,
                    (dx, dy),
                    [from, to],
                    [
                        (across.x - first.x) as f32 * span_x,
                        (across.y - first.y) as f32 * span_z,
                    ],
                );
                outline.extend(&feet[..feet.len() - 1]);
            }
            if outline.len() == 4 {
                indices.extend([corner, below, right, right, below, below + 1]);
                continue;
            }
            // A fan from the square's centre through every vertex of its outline, so it shares its
            // edges' vertices with the walls and no crack opens between them.
            let centre = positions.len() as u32;
            let corners = [corner, right, below, below + 1].map(|v| positions[v as usize]);
            let average = |axis: usize| corners.iter().map(|p| p[axis]).sum::<f32>() / 4.0;
            positions.push([average(0), average(1), average(2)]);
            normals.push(unit(
                [corner, right, below, below + 1]
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

/// Hangs a wall between a near ground's edge facing across `(dx, dy)` and the far square's edge
/// from the vertex `ends[0]` to `ends[1]`, and returns the wall's far feet in that order: a vertex
/// on the far edge under or over each of the near edge's. The wall reaches from the higher of the
/// two edges to below the lower by as much as the near ground's skirt hangs, which is as far as any
/// of its levels of detail strays from the full-detail edge, so it closes the gap to whichever
/// level the near ground is drawn at. The near ground's positions are moved by `offset` into the
/// far ground's frame.
fn wall(
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
    ground: &GroundMesh,
    (dx, dy): (i32, i32),
    ends: [u32; 2],
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
    let [start, end] = ends.map(|end| positions[end as usize]);
    let [start_normal, end_normal] = ends.map(|end| normals[end as usize]);
    let steps = (edge.len() - 1) as f32;
    // Down each place along the edge: the wall's top, its foot on the far edge, and its bottom.
    let mut columns = Vec::with_capacity(edge.len());
    for (k, &vertex) in edge.iter().enumerate() {
        let [x, near, z] = ground.positions[vertex as usize];
        let along = k as f32 / steps;
        let far = start[1] + (end[1] - start[1]) * along;
        let at = [x + offset[0], z + offset[1]];
        let normal = unit(
            [0, 1, 2]
                .map(|axis| start_normal[axis] + (end_normal[axis] - start_normal[axis]) * along),
        );
        let column = [near.max(far), far, near.min(far) - depth].map(|height| {
            positions.push([at[0], height, at[1]]);
            normals.push(normal);
            positions.len() as u32 - 1
        });
        columns.push(column);
    }
    for pair in columns.windows(2) {
        let [a, b] = [pair[0], pair[1]];
        for level in 0..2 {
            let (a_top, a_low, b_top, b_low) = (a[level], a[level + 1], b[level], b[level + 1]);
            indices.extend([a_top, b_top, a_low, b_top, b_low, a_low]);
            indices.extend([a_top, a_low, b_top, b_top, a_low, b_low]);
        }
    }
    columns.iter().map(|column| column[1]).collect()
}

/// `v` scaled to unit length.
fn unit(v: [f32; 3]) -> [f32; 3] {
    let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    v.map(|axis| axis / length)
}
