//! The Talus stage's simulation: thermal weathering, steep ground shedding debris that settles below
//! it at the slope it can rest at (Musgrave, Kolb and Mace, "The synthesis and rendering of eroded
//! fractal terrains", 1989). Wherever a column stands above a neighbour by more than that slope, it
//! sheds part of the excess to it, pass after pass, so cliffs keep their faces while aprons of
//! debris build up at their feet and valley floors flatten.
//!
//! Every pass works out each column's shedding from the field as the pass found it and then applies
//! all of it at once, so the result does not depend on the order the columns are visited in. The
//! change fades in from the region's edges, as a Droplets stage's does, so regions never read each
//! other and the field meets its input along every region border.

use super::droplets::fade_to;

/// How a Talus stage weathers its region, as its pack declares it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Talus {
    /// The steepest slope debris rests at, in cells of height per cell along the ground.
    pub slope: f32,
    /// How many passes of shedding run.
    pub passes: u32,
    /// The share of a column's excess over its neighbours it sheds in a pass, 0 to 1.
    pub rate: f32,
    /// Columns from the region's edge over which the change fades in.
    pub fade: u32,
}

/// The eight neighbours of a column and how far each lies, in columns.
const NEIGHBOURS: [([isize; 2], f32); 8] = [
    ([-1, 0], 1.0),
    ([1, 0], 1.0),
    ([0, -1], 1.0),
    ([0, 1], 1.0),
    ([-1, -1], std::f32::consts::SQRT_2),
    ([1, -1], std::f32::consts::SQRT_2),
    ([-1, 1], std::f32::consts::SQRT_2),
    ([1, 1], std::f32::consts::SQRT_2),
];

/// `heights`, `size[0]` columns by `size[1]` rows, x fastest, in cells, weathered by `talus` with
/// columns `cell` cells wide: the field the debris leaves, faded into `heights` over `talus.fade`
/// columns from the edges.
#[must_use]
pub fn slump(heights: &[f32], size: [usize; 2], cell: f32, talus: &Talus) -> Vec<f32> {
    let [width, depth] = size;
    assert_eq!(heights.len(), width * depth, "a height per column");
    let mut ground = heights.to_vec();
    let mut change = vec![0.0f32; ground.len()];
    for _ in 0..talus.passes {
        change.fill(0.0);
        for y in 0..depth {
            for x in 0..width {
                let here = y * width + x;
                // How far each lower neighbour lies below the slope debris rests at.
                let mut excess = [0.0f32; 8];
                let (mut total, mut most) = (0.0f32, 0.0f32);
                for (k, &([dx, dy], distance)) in NEIGHBOURS.iter().enumerate() {
                    let (nx, ny) = (x as isize + dx, y as isize + dy);
                    if nx < 0 || ny < 0 || nx >= width as isize || ny >= depth as isize {
                        continue;
                    }
                    let drop = ground[here]
                        - ground[ny as usize * width + nx as usize]
                        - talus.slope * cell * distance;
                    if drop > 0.0 {
                        excess[k] = drop;
                        total += drop;
                        most = most.max(drop);
                    }
                }
                if total <= 0.0 {
                    continue;
                }
                // Half the steepest excess at most, so a column never sheds below its neighbour,
                // shared among the lower neighbours by how far each lies below.
                let shed = talus.rate * most * 0.5;
                change[here] -= shed;
                for (k, &([dx, dy], _)) in NEIGHBOURS.iter().enumerate() {
                    if excess[k] > 0.0 {
                        let there = (y as isize + dy) as usize * width + (x as isize + dx) as usize;
                        change[there] += shed * excess[k] / total;
                    }
                }
            }
        }
        for (height, delta) in ground.iter_mut().zip(&change) {
            *height += delta;
        }
    }
    fade_to(heights, &ground, size, talus.fade)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TALUS: Talus = Talus {
        slope: 0.5,
        passes: 200,
        rate: 0.5,
        fade: 0,
    };

    /// A cliff along x: a plateau of 20 cells over flat ground, the step between columns 15 and 16.
    fn cliff() -> (Vec<f32>, [usize; 2]) {
        let size = [32, 8];
        let heights = (0..size[1])
            .flat_map(|_| (0..size[0]).map(|x| if x < 16 { 20.0 } else { 0.0 }))
            .collect();
        (heights, size)
    }

    #[test]
    fn no_slope_steeper_than_debris_rests_at_remains() {
        let (heights, size) = cliff();
        // Enough passes for a cliff of 20 cells to come to rest.
        let settled = Talus {
            passes: 2000,
            ..TALUS
        };

        let slumped = slump(&heights, size, 1.0, &settled);

        let row = &slumped[size[0] * 4..size[0] * 5];
        let steepest = row
            .windows(2)
            .map(|pair| (pair[0] - pair[1]).abs())
            .fold(0.0, f32::max);
        assert!(steepest <= TALUS.slope * 1.05, "{steepest} in {row:?}");
    }

    #[test]
    fn debris_settles_at_the_cliffs_foot_and_nothing_is_lost() {
        let (heights, size) = cliff();

        let slumped = slump(&heights, size, 1.0, &TALUS);

        assert!(
            slumped[size[0] * 4 + 16] > 1.0,
            "{}",
            slumped[size[0] * 4 + 16]
        );
        let (before, after): (f32, f32) = (heights.iter().sum(), slumped.iter().sum());
        assert!(
            (before - after).abs() < 1e-2 * before,
            "{before} against {after}"
        );
    }

    #[test]
    fn ground_gentler_than_debris_rests_at_stays_as_it_is() {
        let size = [16, 16];
        let heights: Vec<f32> = (0..size[1])
            .flat_map(|y| (0..size[0]).map(move |x| (x + y) as f32 * 0.2))
            .collect();

        let slumped = slump(&heights, size, 1.0, &TALUS);

        assert_eq!(slumped, heights);
    }

    #[test]
    fn the_change_fades_to_nothing_at_the_regions_edge() {
        let (heights, size) = cliff();
        let fading = Talus { fade: 4, ..TALUS };

        let slumped = slump(&heights, size, 1.0, &fading);

        // The first and last rows are the region's edge.
        assert_eq!(slumped[..size[0]], heights[..size[0]]);
        assert_eq!(slumped[size[0] * 7..], heights[size[0] * 7..]);
    }
}
