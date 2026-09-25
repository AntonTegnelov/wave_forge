//! What stands in for a chunk seen from far away: one mesh of coloured boxes, with levels of
//! detail, that an engine draws in one call where the chunk's modules would take many.
//!
//! The game gives each module a colour, the colour of its model seen from afar; a module without
//! one, such as air, is left out. The finest level is a box per cell of a coloured module. Each
//! coarser level groups the cells of the one before it two by two by two into one box, filled
//! where at least half its cells are, in their average colour. Only faces that no filled neighbour
//! of the same level covers are kept, so a level is its boxes' outer surface. Faces on the chunk's
//! edges are always kept: the neighbour beyond may be drawn at another level, or not at all.
//!
//! Everything is in a Y-up engine's axes, as [`crate::YUpSpace`] maps them, relative to the
//! chunk's lowest corner.

use crate::loader::RuleFile;
use crate::space::YUpSpace;
use wfc_core::{Chunk, ChunkCoord};

/// A chunk's far stand-in, ready for an engine.
#[derive(Clone, Debug, PartialEq)]
pub struct ProxyMesh {
    pub chunk: ChunkCoord,
    /// Relative to the chunk's lowest corner, four per face: every level's faces, finest first.
    pub positions: Vec<[f32; 3]>,
    /// Unit normals, one per vertex, out of its box.
    pub normals: Vec<[f32; 3]>,
    /// Linear RGBA, one per vertex: its box's colour.
    pub colours: Vec<[f32; 4]>,
    /// The levels of detail, finest first.
    pub levels: Vec<ProxyLevel>,
}

/// One level of detail of a chunk's stand-in.
#[derive(Clone, Debug, PartialEq)]
pub struct ProxyLevel {
    /// How many cells along each axis one box of the level stands for: 1, 2, 4 and so on.
    pub cells: u32,
    /// Triangles into [`ProxyMesh::positions`], counter-clockwise seen from outside their box.
    pub indices: Vec<u32>,
    /// How far, in engine units, the level's boxes may reach past the finest level's: a box of
    /// `cells` cells a side can stand for one filled cell in a corner of it.
    pub error: f32,
}

/// The six directions a box's faces look along, in the lattice's axes (z up).
const DIRECTIONS: [[i32; 3]; 6] = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
];

/// The far stand-in of a solved `chunk`, from `colour`, each module's colour by name.
#[must_use]
pub fn proxy_mesh(
    chunk: &Chunk,
    rules: &RuleFile,
    space: &YUpSpace,
    colour: impl Fn(&str) -> Option<[f32; 4]>,
) -> ProxyMesh {
    let shape = space.chunk_shape();
    let dims = [shape.x, shape.y, shape.z];
    // A colour per cell, or none for a cell whose module has none.
    let mut grid: Vec<Option<[f32; 4]>> = chunk
        .tiles
        .iter()
        .map(|&tile| colour(rules.name(usize::from(tile))))
        .collect();
    let [cell_x, cell_up, cell_z] = space.cell_size();
    let largest_cell = cell_x.max(cell_up).max(cell_z);
    let mut mesh = ProxyMesh {
        chunk: chunk.coord,
        positions: Vec::new(),
        normals: Vec::new(),
        colours: Vec::new(),
        levels: Vec::new(),
    };
    let mut cells = 1_u32;
    let mut size = dims;
    loop {
        let indices = level_faces(&grid, size, cells, [cell_x, cell_up, cell_z], &mut mesh);
        mesh.levels.push(ProxyLevel {
            cells,
            indices,
            error: (cells - 1) as f32 * largest_cell,
        });
        if size.iter().all(|&along| along == 1) {
            break;
        }
        (grid, size) = coarser(&grid, size);
        cells *= 2;
    }
    mesh
}

/// The grid of a level, from the one finer: groups of two by two by two, fewer at an odd edge,
/// filled where at least half their cells are, in the average colour of those.
fn coarser(grid: &[Option<[f32; 4]>], size: [u32; 3]) -> (Vec<Option<[f32; 4]>>, [u32; 3]) {
    let next = size.map(|along| along.div_ceil(2));
    let at = |[x, y, z]: [u32; 3], dims: [u32; 3]| (x + dims[0] * (y + dims[1] * z)) as usize;
    let mut out = Vec::with_capacity((next[0] * next[1] * next[2]) as usize);
    for z in 0..next[2] {
        for y in 0..next[1] {
            for x in 0..next[0] {
                let (mut total, mut filled, mut sum) = (0_u32, 0_u32, [0.0_f32; 4]);
                for dz in 0..2 {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let cell = [2 * x + dx, 2 * y + dy, 2 * z + dz];
                            if (0..3).any(|axis| cell[axis] >= size[axis]) {
                                continue;
                            }
                            total += 1;
                            if let Some(colour) = grid[at(cell, size)] {
                                filled += 1;
                                for channel in 0..4 {
                                    sum[channel] += colour[channel];
                                }
                            }
                        }
                    }
                }
                out.push((2 * filled >= total).then(|| sum.map(|channel| channel / filled as f32)));
            }
        }
    }
    (out, next)
}

/// Adds a level's outer faces to `mesh`: boxes of `cells` cells a side on a grid of `size`,
/// and returns their triangles.
fn level_faces(
    grid: &[Option<[f32; 4]>],
    size: [u32; 3],
    cells: u32,
    cell: [f32; 3],
    mesh: &mut ProxyMesh,
) -> Vec<u32> {
    let at = |[x, y, z]: [u32; 3]| (x + size[0] * (y + size[1] * z)) as usize;
    // A box's extent in the lattice's axes, in cells, clamped to the chunk at an odd edge.
    let chunk_cells = [size[0] * cells, size[1] * cells, size[2] * cells];
    let mut indices = Vec::new();
    for z in 0..size[2] {
        for y in 0..size[1] {
            for x in 0..size[0] {
                let Some(colour) = grid[at([x, y, z])] else {
                    continue;
                };
                let low = [x * cells, y * cells, z * cells];
                let high = [0, 1, 2].map(|axis| (low[axis] + cells).min(chunk_cells[axis]));
                for direction in DIRECTIONS {
                    let neighbour = [0, 1, 2]
                        .map(|axis| i64::from([x, y, z][axis]) + i64::from(direction[axis]));
                    let inside = (0..3).all(|axis| {
                        neighbour[axis] >= 0 && neighbour[axis] < i64::from(size[axis])
                    });
                    if inside && grid[at(neighbour.map(|n| n as u32))].is_some() {
                        continue;
                    }
                    push_face(mesh, &mut indices, low, high, direction, colour, cell);
                }
            }
        }
    }
    indices
}

/// Adds the face of the box from `low` to `high` (in cells, lattice axes) that looks along
/// `direction`, counter-clockwise seen from outside, in the engine's axes.
fn push_face(
    mesh: &mut ProxyMesh,
    indices: &mut Vec<u32>,
    low: [u32; 3],
    high: [u32; 3],
    direction: [i32; 3],
    colour: [f32; 4],
    cell: [f32; 3],
) {
    let axis = direction
        .iter()
        .position(|&d| d != 0)
        .expect("a direction has an axis");
    let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
    let plane = if direction[axis] > 0 {
        high[axis]
    } else {
        low[axis]
    };
    // Corners around the face; u then v is counter-clockwise seen from the positive side.
    let mut corners = [[0_u32; 3]; 4];
    for (corner, (a, b)) in corners.iter_mut().zip([
        (low[u], low[v]),
        (high[u], low[v]),
        (high[u], high[v]),
        (low[u], high[v]),
    ]) {
        corner[axis] = plane;
        corner[u] = a;
        corner[v] = b;
    }
    if direction[axis] < 0 {
        corners.reverse();
    }
    // The lattice's x, y and z are the engine's x, z and y.
    let engine = |[x, y, z]: [u32; 3]| [x as f32 * cell[0], z as f32 * cell[1], y as f32 * cell[2]];
    let normal = engine_direction(direction);
    let first = u32::try_from(mesh.positions.len()).expect("fewer than 2^32 vertices");
    for corner in corners {
        mesh.positions.push(engine(corner));
        mesh.normals.push(normal);
        mesh.colours.push(colour);
    }
    // Swapping two axes mirrors, so what winds counter-clockwise in the lattice winds clockwise
    // in the engine: the triangles go the other way round.
    indices.extend([first, first + 2, first + 1, first, first + 3, first + 2]);
}

fn engine_direction(direction: [i32; 3]) -> [f32; 3] {
    [
        direction[0] as f32,
        direction[2] as f32,
        direction[1] as f32,
    ]
}
