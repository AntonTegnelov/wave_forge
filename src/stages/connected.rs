//! A Connected stage's labelling (docs/reference/packs.md, "Connected"): the connected areas of a
//! region's categories, each column given its area's id.

/// The id of the connected area each column of a region lies in, row by row with x fastest, from
/// `categories`, the region's category per column in the same order, over `columns` columns along x
/// and y. Two columns are joined when they share an edge and a category. Ids count up from 0 in the
/// order each area's first column comes, so they depend on the categories alone.
pub(crate) fn connected(categories: &[u8], [width, depth]: [usize; 2]) -> Vec<f32> {
    let mut ids: Vec<Option<u32>> = vec![None; categories.len()];
    let mut next = 0;
    let mut frontier = Vec::new();
    for start in 0..categories.len() {
        if ids[start].is_some() {
            continue;
        }
        ids[start] = Some(next);
        frontier.push(start);
        while let Some(at) = frontier.pop() {
            let (x, y) = (at % width, at / width);
            let neighbours = [
                (x > 0).then(|| at - 1),
                (x + 1 < width).then(|| at + 1),
                (y > 0).then(|| at - width),
                (y + 1 < depth).then(|| at + width),
            ];
            for neighbour in neighbours.into_iter().flatten() {
                if ids[neighbour].is_none() && categories[neighbour] == categories[at] {
                    ids[neighbour] = Some(next);
                    frontier.push(neighbour);
                }
            }
        }
        next += 1;
    }
    // An id is exact as a float up to 2^24, more areas than a region of 64 by 64 chunks of 64 by 64
    // columns can hold.
    ids.into_iter()
        .map(|id| id.expect("every column is labelled") as f32)
        .collect()
}
