//! A Passages stage's digging (docs/reference/packs.md, "Passages"): passages through a region's
//! closed columns that join every open column to every other.

use std::collections::VecDeque;

/// The open columns of a region, row by row with x fastest, over `columns` columns along x and y,
/// with passages dug so that every one reaches every other: 1 at an open column and 0 at a closed
/// one. A column is open where `values` is above `above`, and so is each column of `crossings`.
///
/// Every open column is joined to the region's first open column by the route, edge to edge, that
/// passes through the fewest closed columns, and those columns are dug; of equally good routes the
/// search keeps the first it finds in a fixed order. So the passages depend on the open columns
/// alone, and a route runs through the caves it meets on the way rather than beside them.
pub(crate) fn passages(
    values: &[f32],
    [width, depth]: [usize; 2],
    above: f32,
    crossings: &[usize],
) -> Vec<f32> {
    let mut open: Vec<bool> = values.iter().map(|&value| value > above).collect();
    for &crossing in crossings {
        open[crossing] = true;
    }
    let Some(first) = open.iter().position(|&open| open) else {
        return vec![0.0; open.len()];
    };
    // The fewest closed columns on a route from `first` to each column, and the column each such
    // route comes from: a search that takes a step into an open column before any into a closed
    // one (a breadth-first search over steps that cost 0 or 1).
    let mut digs = vec![u32::MAX; open.len()];
    let mut from: Vec<Option<usize>> = vec![None; open.len()];
    digs[first] = 0;
    let mut queue = VecDeque::from([first]);
    while let Some(at) = queue.pop_front() {
        let (x, y) = (at % width, at / width);
        let neighbours = [
            (x > 0).then(|| at - 1),
            (x + 1 < width).then(|| at + 1),
            (y > 0).then(|| at - width),
            (y + 1 < depth).then(|| at + width),
        ];
        for next in neighbours.into_iter().flatten() {
            let cost = digs[at] + u32::from(!open[next]);
            if cost < digs[next] {
                digs[next] = cost;
                from[next] = Some(at);
                if open[next] {
                    queue.push_front(next);
                } else {
                    queue.push_back(next);
                }
            }
        }
    }
    let mut joined = vec![false; open.len()];
    joined[first] = true;
    for start in 0..open.len() {
        if !open[start] {
            continue;
        }
        let mut at = start;
        while !joined[at] {
            joined[at] = true;
            open[at] = true;
            at = from[at].expect("every column but the first is reached from another");
        }
    }
    open.into_iter()
        .map(|open| if open { 1.0 } else { 0.0 })
        .collect()
}
