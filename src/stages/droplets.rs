//! The Droplets stage's simulation: hydraulic erosion by droplets of water run one after another over
//! a region's height field (Beyer, "Implementation of a method for hydraulic erosion", 2015). Each
//! droplet rolls downhill with some inertia, picks up sediment where it speeds down and the water can
//! carry more, and drops it where it slows or climbs, so many droplets wear valleys that join as
//! they descend and lay fans where they reach flat ground.
//!
//! The droplets run in a fixed order from hashed starts, so a region drains the same on every run.
//! A droplet that leaves the region stops, and the change fades out toward the region's edges, so
//! regions never read each other and the field meets its input along every region border.

use wfc_core::hash::pcg3d;

/// How a Droplets stage drains its region, as its pack declares it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Droplets {
    /// Droplets run per column of the region.
    pub per_column: f32,
    /// The most steps, a column each, a droplet takes before it stops.
    pub lifetime: u32,
    /// The radius in columns over which a droplet wears the ground away.
    pub radius: u32,
    /// The sediment, in cells of height, water can carry on a slope of one cell's fall per cell,
    /// at unit speed and water.
    pub capacity: f32,
    /// The share of the room left in its capacity a droplet picks up in a step, 0 to 1.
    pub erosion: f32,
    /// The share of the sediment beyond its capacity a droplet drops in a step, 0 to 1.
    pub deposition: f32,
    /// Columns from the region's edge over which the change fades in.
    pub fade: u32,
}

/// How much of its direction a droplet keeps each step, against turning straight downhill.
const INERTIA: f32 = 0.05;
/// The share of its water a droplet loses each step.
const EVAPORATION: f32 = 0.01;
/// How fast a droplet gains speed for each cell it falls.
const GRAVITY: f32 = 4.0;
/// The slope a droplet's capacity never falls below, so it still carries a little on flat ground.
const MIN_SLOPE: f32 = 0.01;

/// `heights`, `size[0]` columns by `size[1]` rows, x fastest, in cells, drained by droplets started
/// at the columns `start` hashes their index to, with columns `cell` cells wide: the field the
/// droplets leave, faded into `heights` over `droplets.fade` columns from the edges.
#[must_use]
pub fn drain(
    heights: &[f32],
    size: [usize; 2],
    cell: f32,
    droplets: &Droplets,
    start: impl Fn(u32) -> [u32; 3],
) -> Vec<f32> {
    let [width, depth] = size;
    assert_eq!(heights.len(), width * depth, "a height per column");
    let mut ground = heights.to_vec();
    let brush = brush(droplets.radius);
    let count = (droplets.per_column * (width * depth) as f32).round() as u32;
    for index in 0..count {
        let [x, y, _] = start(index);
        let at = [
            (x >> 8) as f32 / (1u32 << 24) as f32 * (width - 1) as f32,
            (y >> 8) as f32 / (1u32 << 24) as f32 * (depth - 1) as f32,
        ];
        run_droplet(&mut ground, size, cell, droplets, &brush, at);
    }
    fade_to(heights, &ground, size, droplets.fade)
}

/// The hashed starts of droplets in a region: three words for droplet `index`, from the world, the
/// stage's salt and the region.
#[must_use]
pub fn droplet_start(world: u32, salt: u32, region: (i32, i32), index: u32) -> [u32; 3] {
    pcg3d([
        world ^ salt,
        (region.0 as u32) ^ index.wrapping_mul(0x9E37_79B9),
        (region.1 as u32) ^ index.wrapping_mul(0x85EB_CA6B),
    ])
}

/// The offsets within `radius` columns of a droplet's column, each with its share of the wear:
/// more the nearer it is, all of them summing to 1.
fn brush(radius: u32) -> Vec<([isize; 2], f32)> {
    let reach = radius.max(1) as isize;
    let mut offsets = Vec::new();
    for dy in -reach..=reach {
        for dx in -reach..=reach {
            let weight = reach as f32 - ((dx * dx + dy * dy) as f32).sqrt();
            if weight > 0.0 {
                offsets.push(([dx, dy], weight));
            }
        }
    }
    let total: f32 = offsets.iter().map(|(_, weight)| weight).sum();
    offsets
        .into_iter()
        .map(|(offset, weight)| (offset, weight / total))
        .collect()
}

/// The ground's height at `at`, blended between the four columns around it, and its slope there
/// along x and y, in cells a column.
fn height_and_slope(ground: &[f32], width: usize, at: [f32; 2]) -> (f32, [f32; 2]) {
    let (x, y) = (at[0] as usize, at[1] as usize);
    let (u, v) = (at[0] - x as f32, at[1] - y as f32);
    let corner = |dx: usize, dy: usize| ground[(y + dy) * width + x + dx];
    let (nw, ne, sw, se) = (corner(0, 0), corner(1, 0), corner(0, 1), corner(1, 1));
    let slope = [
        (ne - nw) * (1.0 - v) + (se - sw) * v,
        (sw - nw) * (1.0 - u) + (se - ne) * u,
    ];
    let height = nw * (1.0 - u) * (1.0 - v) + ne * u * (1.0 - v) + sw * (1.0 - u) * v + se * u * v;
    (height, slope)
}

/// Whether `at` lies where its four surrounding columns are all inside the region.
fn inside(size: [usize; 2], at: [f32; 2]) -> bool {
    at[0] >= 0.0 && at[1] >= 0.0 && at[0] < (size[0] - 1) as f32 && at[1] < (size[1] - 1) as f32
}

/// Runs one droplet from `at` over `ground` until it stops: it leaves the region, comes to rest, or
/// its lifetime ends.
fn run_droplet(
    ground: &mut [f32],
    size: [usize; 2],
    cell: f32,
    droplets: &Droplets,
    brush: &[([isize; 2], f32)],
    start: [f32; 2],
) {
    let width = size[0];
    let mut at = start;
    let mut direction = [0.0f32; 2];
    let (mut speed, mut water, mut sediment) = (1.0f32, 1.0f32, 0.0f32);
    for _ in 0..droplets.lifetime {
        let column = [at[0] as usize, at[1] as usize];
        let offset = [at[0] - column[0] as f32, at[1] - column[1] as f32];
        let (height, slope) = height_and_slope(ground, width, at);
        direction = [
            direction[0] * INERTIA - slope[0] * (1.0 - INERTIA),
            direction[1] * INERTIA - slope[1] * (1.0 - INERTIA),
        ];
        let length = direction[0].hypot(direction[1]);
        if length == 0.0 {
            // On perfectly flat ground nothing says which way to run.
            break;
        }
        direction = [direction[0] / length, direction[1] / length];
        at = [at[0] + direction[0], at[1] + direction[1]];
        if !inside(size, at) {
            break;
        }
        let (next, _) = height_and_slope(ground, width, at);
        // Cells fallen this step: negative where the droplet climbs.
        let fall = height - next;
        // What the water can carry grows with the slope, a fall in cells per cell run, so a stage
        // drains alike at any scale.
        let room = (fall / cell).max(MIN_SLOPE) * speed * water * droplets.capacity;
        if sediment > room || fall < 0.0 {
            // Climbing, it fills the hollow behind it as far as it can; slowing, it drops the
            // sediment it can no longer carry.
            let dropped = if fall < 0.0 {
                sediment.min(-fall)
            } else {
                (sediment - room) * droplets.deposition
            };
            sediment -= dropped;
            for (dx, dy, weight) in [
                (0, 0, (1.0 - offset[0]) * (1.0 - offset[1])),
                (1, 0, offset[0] * (1.0 - offset[1])),
                (0, 1, (1.0 - offset[0]) * offset[1]),
                (1, 1, offset[0] * offset[1]),
            ] {
                ground[(column[1] + dy) * width + column[0] + dx] += dropped * weight;
            }
        } else {
            // Never more than the fall, and no column below where the droplet runs on to, so a
            // droplet does not dig a pit the next one deepens.
            let worn = ((room - sediment) * droplets.erosion).min(fall);
            for &([dx, dy], weight) in brush {
                let (x, y) = (column[0] as isize + dx, column[1] as isize + dy);
                if x < 0 || y < 0 || x >= width as isize || y >= size[1] as isize {
                    continue;
                }
                let index = y as usize * width + x as usize;
                let wear = (worn * weight).min((ground[index] - next).max(0.0));
                ground[index] -= wear;
                sediment += wear;
            }
        }
        speed = (speed * speed + fall * GRAVITY).max(0.0).sqrt();
        water *= 1.0 - EVAPORATION;
    }
}

/// `drained` faded into `input` over `fade` columns from the region's edges: the input itself on
/// the edge columns, the drained field from `fade` columns in.
fn fade_to(input: &[f32], drained: &[f32], size: [usize; 2], fade: u32) -> Vec<f32> {
    let [width, depth] = size;
    (0..depth)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .map(|(x, y)| {
            let edge = x.min(y).min(width - 1 - x).min(depth - 1 - y) as f32;
            let t = if fade == 0 {
                1.0
            } else {
                (edge / fade as f32).clamp(0.0, 1.0)
            };
            let weight = t * t * (3.0 - 2.0 * t);
            let index = y * width + x;
            input[index] + (drained[index] - input[index]) * weight
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const DROPLETS: Droplets = Droplets {
        per_column: 1.0,
        lifetime: 40,
        radius: 2,
        capacity: 4.0,
        erosion: 0.3,
        deposition: 0.3,
        fade: 8,
    };

    /// A slope falling half a cell a column toward +x for 40 columns, then flat, 64 by 64 columns,
    /// with a ripple so droplets find channels.
    fn slope_then_flat() -> Vec<f32> {
        (0..64)
            .flat_map(|y| (0..64).map(move |x| (x, y)))
            .map(|(x, y)| {
                let fall = (40.0 - x.min(40) as f32) * 0.5;
                fall + 0.3 * ((y as f32) * 0.7).sin() * ((x as f32) * 0.3).cos()
            })
            .collect()
    }

    fn drained(heights: &[f32]) -> Vec<f32> {
        drain(heights, [64, 64], 1.0, &DROPLETS, |index| {
            droplet_start(7, 11, (0, 0), index)
        })
    }

    #[test]
    fn a_region_drains_the_same_every_time() {
        let heights = slope_then_flat();

        let (first, second) = (drained(&heights), drained(&heights));

        assert_eq!(
            first
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            second
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn the_field_is_its_input_on_the_regions_edges() {
        let heights = slope_then_flat();

        let field = drained(&heights);

        for i in 0..64 {
            for index in [i, 63 * 64 + i, i * 64, i * 64 + 63] {
                assert_eq!(field[index], heights[index], "column {index}");
            }
        }
        assert!(
            field.iter().zip(&heights).any(|(a, b)| a != b),
            "nothing drained"
        );
    }

    #[test]
    fn droplets_wear_the_slope_down_and_lay_its_sediment_at_the_foot() {
        let heights = slope_then_flat();

        let field = drained(&heights);

        // Away from the faded edges: the slope's middle, and the flat just past its foot.
        let change = |xs: std::ops::Range<usize>| {
            let columns: Vec<usize> = (12..52)
                .flat_map(|y| xs.clone().map(move |x| y * 64 + x))
                .collect();
            columns.iter().map(|&i| field[i] - heights[i]).sum::<f32>() / columns.len() as f32
        };
        let (slope, foot) = (change(12..36), change(41..48));
        assert!(slope < 0.0, "the slope changed by {slope} on average");
        assert!(foot > 0.0, "the foot changed by {foot} on average");
    }
}
