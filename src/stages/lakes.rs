//! Lakes: the water that stands in a region's hollows, the region job a pack names with a Lakes
//! stage.
//!
//! A priority flood (Barnes, Lehman and Mulla, 2014) raises every column of a region to the lowest
//! height water there could drain away over: water runs off the region's edge columns and into the
//! sea, so the flood starts from those, and a column reached over higher ground is filled to that
//! height. A column filled above its ground and above the sea is under a lake; a lake is a
//! connected set of such columns, kept if it has at least `min_columns`. A lake never reaches its
//! region's edge, since the edge drains, so no region reads another's lakes and every region comes
//! out the same in any order.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

/// A column waiting in the flood: the height water stands at there, then where it is. Ordered by
/// height, then position, so the flood visits columns in the same order every time.
#[derive(Clone, Copy, PartialEq)]
struct Wet {
    level: f32,
    at: usize,
}

impl Eq for Wet {}

impl Ord for Wet {
    fn cmp(&self, other: &Self) -> Ordering {
        self.level
            .total_cmp(&other.level)
            .then(self.at.cmp(&other.at))
    }
}

impl PartialOrd for Wet {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// The water surface over a region of `size` columns (x fastest) whose ground is `heights`: a
/// lake's level over its columns, the ground everywhere else. Water drains off the region's edge
/// and into the sea at `sea`; a lake under `min_columns` columns is left dry.
///
/// # Panics
/// If `heights` does not hold `size[0] * size[1]` heights.
pub(crate) fn lake_surface(
    heights: &[f32],
    size: [usize; 2],
    sea: f32,
    min_columns: u32,
) -> Vec<f32> {
    let [w, h] = size;
    assert_eq!(heights.len(), w * h, "a height per column");
    let mut filled = heights.to_vec();
    let mut visited = vec![false; heights.len()];
    let mut queue = BinaryHeap::new();
    for y in 0..h {
        for x in 0..w {
            let at = y * w + x;
            let edge = x == 0 || y == 0 || x == w - 1 || y == h - 1;
            if edge || heights[at] <= sea {
                visited[at] = true;
                queue.push(Reverse(Wet {
                    level: heights[at],
                    at,
                }));
            }
        }
    }
    while let Some(Reverse(Wet { level, at })) = queue.pop() {
        let (x, y) = (at % w, at / w);
        for (dx, dy) in [(1_isize, 0_isize), (-1, 0), (0, 1), (0, -1)] {
            let (nx, ny) = (x as isize + dx, y as isize + dy);
            if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                continue;
            }
            let next = ny as usize * w + nx as usize;
            if visited[next] {
                continue;
            }
            visited[next] = true;
            filled[next] = heights[next].max(level);
            queue.push(Reverse(Wet {
                level: filled[next],
                at: next,
            }));
        }
    }

    // Lakes: connected columns filled above their ground and above the sea.
    let wet = |at: usize| filled[at] > heights[at] && filled[at] > sea;
    let mut surface = heights.to_vec();
    let mut seen = vec![false; heights.len()];
    for start in 0..heights.len() {
        if seen[start] || !wet(start) {
            continue;
        }
        let mut lake = vec![start];
        seen[start] = true;
        let mut next = 0;
        while next < lake.len() {
            let at = lake[next];
            next += 1;
            let (x, y) = (at % w, at / w);
            for (dx, dy) in [(1_isize, 0_isize), (-1, 0), (0, 1), (0, -1)] {
                let (nx, ny) = (x as isize + dx, y as isize + dy);
                if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                    continue;
                }
                let neighbour = ny as usize * w + nx as usize;
                if !seen[neighbour] && wet(neighbour) {
                    seen[neighbour] = true;
                    lake.push(neighbour);
                }
            }
        }
        if lake.len() >= min_columns as usize {
            for at in lake {
                surface[at] = filled[at];
            }
        }
    }
    surface
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A region of `size` columns whose ground is `height` at each column.
    fn ground(size: [usize; 2], height: impl Fn(usize, usize) -> f32) -> Vec<f32> {
        (0..size[1])
            .flat_map(|y| (0..size[0]).map(move |x| (x, y)))
            .map(|(x, y)| height(x, y))
            .collect()
    }

    /// A bowl in the middle of a 9 by 9 region: a rim of 5 around a floor of 1, with one low
    /// notch at 3 in the rim.
    fn bowl(x: usize, y: usize) -> f32 {
        let inside = (2..=6).contains(&x) && (2..=6).contains(&y);
        let rim = (1..=7).contains(&x) && (1..=7).contains(&y) && !inside;
        if (x, y) == (4, 1) {
            3.0
        } else if rim {
            5.0
        } else if inside {
            1.0
        } else {
            0.5
        }
    }

    #[test]
    fn a_hollow_fills_to_the_lowest_notch_in_its_rim() {
        let heights = ground([9, 9], bowl);

        let surface = lake_surface(&heights, [9, 9], -10.0, 1);

        for y in 0..9 {
            for x in 0..9 {
                let at = y * 9 + x;
                let inside = (2..=6).contains(&x) && (2..=6).contains(&y);
                let expected = if inside { 3.0 } else { heights[at] };
                assert_eq!(surface[at], expected, "({x}, {y})");
            }
        }
    }

    #[test]
    fn a_hollow_open_to_the_region_edge_drains() {
        // The same bowl with its notch cut down to the edge.
        let heights = ground(
            [9, 9],
            |x, y| if x == 4 && y <= 1 { 0.5 } else { bowl(x, y) },
        );

        let surface = lake_surface(&heights, [9, 9], -10.0, 1);

        assert_eq!(surface, heights);
    }

    #[test]
    fn a_hollow_below_the_sea_is_sea_not_lake() {
        let heights = ground([9, 9], bowl);

        let surface = lake_surface(&heights, [9, 9], 4.0, 1);

        assert_eq!(surface, heights);
    }

    #[test]
    fn a_lake_smaller_than_the_least_is_left_dry() {
        let heights = ground([9, 9], bowl);

        let small = lake_surface(&heights, [9, 9], -10.0, 26);
        let large = lake_surface(&heights, [9, 9], -10.0, 25);

        assert_eq!(small, heights);
        assert_ne!(large, heights);
    }

    #[test]
    fn a_lake_is_level_and_never_below_its_ground() {
        let heights = ground([24, 20], |x, y| {
            let (fx, fy) = (x as f32, y as f32);
            (fx * 0.9).sin() * 2.0 + (fy * 0.7).cos() * 2.0 + (fx * fy * 0.05).sin()
        });

        let surface = lake_surface(&heights, [24, 20], -10.0, 1);

        for at in 0..heights.len() {
            assert!(
                surface[at] >= heights[at],
                "column {at} is below its ground"
            );
            let (x, y) = (at % 24, at / 24);
            if surface[at] > heights[at] {
                assert!(
                    x > 0 && y > 0 && x < 23 && y < 19,
                    "a lake at the edge ({x}, {y})"
                );
                // A lake's neighbours are under the same water or stand at least as high.
                for (nx, ny) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] {
                    let next = ny * 24 + nx;
                    let neighbour_wet = surface[next] > heights[next];
                    assert!(
                        (neighbour_wet && surface[next] == surface[at])
                            || heights[next] >= surface[at],
                        "({x}, {y}) at {} beside ({nx}, {ny})",
                        surface[at]
                    );
                }
            }
        }
        assert!(surface != heights, "the rough ground holds some lake");
    }
}
