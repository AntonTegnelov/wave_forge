//! The Transport stage's simulation: rivers that carry sediment down a region's height field, a pass
//! after pass, so channels settle into graded profiles and what they carry is laid down where they
//! slow, in fans, in lakes and in deltas at the sea.
//!
//! Every pass, a priority flood from the region's edges and the sea gives each column the column its
//! water runs to and the order to visit them in, every column after the one its water runs to. Each
//! column gathers the water of the columns above it; visiting the columns from the top down, the
//! water at each can carry sediment in proportion to the water it gathers raised to `area` and to its
//! slope (a transport-limited stream power law). Where it carries less than that it wears the ground
//! down, never below where it runs to; where it carries more it lays some down; in a hollow it fills
//! the hollow as far as it can. What reaches the sea is laid down there, up to the sea's level, and
//! what reaches the region's edge leaves it. Passes run in a fixed order, so a region comes out the
//! same on every run, and the change fades out toward the region's edges, so regions never read each
//! other and the field meets its input along every region border.

use super::droplets::fade_to;
use super::lakes::flood;

/// How a Transport stage works its region, as its pack declares it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transport {
    /// Passes of the water over the region.
    pub passes: u32,
    /// The sediment, in cells of height, water gathered from one cell of ground can carry on a
    /// slope of one cell's fall per cell.
    pub capacity: f32,
    /// The power the water a column gathers is raised to in what it can carry.
    pub area: f32,
    /// The share of the room left in what it can carry the water takes up at a column, 0 to 1.
    pub erosion: f32,
    /// The share of what it carries beyond that the water lays down at a column, 0 to 1.
    pub deposition: f32,
    /// Columns from the region's edge over which the change fades in.
    pub fade: u32,
}

/// How many columns under the sea what a river carries reaches past its mouth.
const DELTA_REACH: usize = 16;

/// The columns beside `at` in a region `width` by `depth` columns, along x then y.
fn neighbours(at: usize, width: usize, depth: usize) -> impl Iterator<Item = usize> {
    let (x, y) = (at % width, at / width);
    [
        (x > 0).then(|| at - 1),
        (x + 1 < width).then(|| at + 1),
        (y > 0).then(|| at - width),
        (y + 1 < depth).then(|| at + width),
    ]
    .into_iter()
    .flatten()
}

/// `heights`, `size[0]` columns by `size[1]` rows, x fastest, in cells, whose first column is
/// `origin` in the world, worked by rivers carrying sediment, with columns `cell` cells wide and
/// the sea at `sea`: the field they leave, faded into `heights` over `transport.fade` columns from
/// the edges.
#[must_use]
pub fn transport(
    heights: &[f32],
    size: [usize; 2],
    origin: [i64; 2],
    cell: f32,
    sea: f32,
    transport: &Transport,
) -> Vec<f32> {
    let [width, depth] = size;
    assert_eq!(heights.len(), width * depth, "a height per column");
    let mut ground = heights.to_vec();
    for _ in 0..transport.passes {
        let edge = |at: usize| {
            let (x, y) = (at % width, at / width);
            x == 0 || y == 0 || x == width - 1 || y == depth - 1
        };
        let flood = flood(&ground, size, origin, |at| edge(at) || ground[at] <= sea);
        // The cells of ground whose water each column gathers.
        let mut gathers = vec![cell * cell; ground.len()];
        for &at in flood.reached.iter().rev() {
            if let Some(next) = flood.toward[at] {
                gathers[next] += gathers[at];
            }
        }
        let mut carried = vec![0.0f32; ground.len()];
        for &at in flood.reached.iter().rev() {
            let load = carried[at];
            let Some(next) = flood.toward[at] else {
                // The sea keeps what reaches it up to its level, and a full column hands the rest on
                // to its deepest neighbour under the sea, so a delta builds out from the mouth; the
                // region's edge lets it go.
                let (mut at, mut load) = (at, load);
                for _ in 0..DELTA_REACH {
                    if ground[at] > sea {
                        break;
                    }
                    let laid = load.min(sea - ground[at]);
                    ground[at] += laid;
                    load -= laid;
                    let deepest = neighbours(at, width, depth)
                        .filter(|&next| ground[next] < sea)
                        .min_by(|&a, &b| ground[a].total_cmp(&ground[b]).then(a.cmp(&b)));
                    match deepest {
                        Some(next) if load > 0.0 => at = next,
                        _ => break,
                    }
                }
                continue;
            };
            let fall = ground[at] - ground[next];
            let load = if fall <= 0.0 {
                // In a hollow or on a flat the water stands still and drops what it carries, up to
                // the level it spills over.
                let laid = load.min(flood.filled[at] - ground[at]).max(0.0);
                ground[at] += laid;
                load - laid
            } else {
                let room = transport.capacity * gathers[at].powf(transport.area) * fall / cell;
                if load < room {
                    // Never below where it runs to, nor below the sea, the land's base level.
                    let floor = ground[next].max(sea.min(ground[at]));
                    let worn = ((room - load) * transport.erosion).min(ground[at] - floor);
                    ground[at] -= worn;
                    load + worn
                } else {
                    // Never above where it came from, so a river does not dam itself.
                    let laid = ((load - room) * transport.deposition).min(fall);
                    ground[at] += laid;
                    load - laid
                }
            };
            carried[next] += load;
        }
    }
    fade_to(heights, &ground, size, transport.fade)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRANSPORT: Transport = Transport {
        passes: 20,
        capacity: 0.01,
        area: 0.5,
        erosion: 0.5,
        deposition: 0.5,
        fade: 8,
    };

    /// A slope falling half a cell a column toward +x for 40 columns into a sea below 0, 64 by 64
    /// columns, with a ripple so the water finds channels.
    fn slope_into_the_sea() -> Vec<f32> {
        (0..64)
            .flat_map(|y| (0..64).map(move |x| (x, y)))
            .map(|(x, y)| {
                let fall = (40.0 - x as f32) * 0.5;
                fall.max(-3.0) + 0.3 * ((y as f32) * 0.7).sin() * ((x as f32) * 0.3).cos()
            })
            .collect()
    }

    fn worked(heights: &[f32]) -> Vec<f32> {
        transport(heights, [64, 64], [0, 0], 1.0, 0.0, &TRANSPORT)
    }

    #[test]
    fn a_region_is_worked_the_same_every_time() {
        let heights = slope_into_the_sea();

        let (first, second) = (worked(&heights), worked(&heights));

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
        let heights = slope_into_the_sea();

        let field = worked(&heights);

        for i in 0..64 {
            for index in [i, 63 * 64 + i, i * 64, i * 64 + 63] {
                assert_eq!(field[index], heights[index], "column {index}");
            }
        }
        assert!(
            field.iter().zip(&heights).any(|(a, b)| a != b),
            "nothing moved"
        );
    }

    #[test]
    fn rivers_wear_the_land_down_and_lay_it_in_the_sea_off_their_mouths() {
        let heights = slope_into_the_sea();

        let field = worked(&heights);

        // Away from the faded edges: the land's slope, and the sea just past the shore.
        let change = |xs: std::ops::Range<usize>| {
            let columns: Vec<usize> = (12..52)
                .flat_map(|y| xs.clone().map(move |x| y * 64 + x))
                .collect();
            columns.iter().map(|&i| field[i] - heights[i]).sum::<f32>() / columns.len() as f32
        };
        let (land, sea) = (change(12..36), change(41..50));
        assert!(land < 0.0, "the land changed by {land} on average");
        assert!(sea > 0.0, "the sea floor changed by {sea} on average");
        // The sea keeps what reaches it only up to its level.
        for (index, (&now, &was)) in field.iter().zip(&heights).enumerate() {
            assert!(
                was > 0.0 || now <= 0.0,
                "column {index} filled above the sea, to {now}"
            );
        }
    }
}
