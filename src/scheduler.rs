//! Which chunks to generate next, and in what order.
//!
//! Two rules shape every decision. Chunks that share a face cannot be solved in the same dispatch,
//! because each reads the other's cells, so a batch takes one parity of the chunk lattice. And a
//! chunk reads only its face neighbours, so it must not be solved before they are: that is what
//! makes its tiles a function of where it is rather than of when it was asked for.

use std::collections::BTreeSet;
use wfc_core::{ChunkCoord, ChunkStore, WorldExtent};

/// Where in the world generation is wanted, and how far around it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FocusPoint {
    /// The chunk at the centre, usually the one a player is in.
    pub chunk: ChunkCoord,
    /// How many chunks out from it to generate.
    pub radius: u32,
}

impl FocusPoint {
    /// A focus on one chunk and the chunks within `radius` of it.
    #[must_use]
    pub const fn new(chunk: ChunkCoord, radius: u32) -> Self {
        Self { chunk, radius }
    }

    /// How far `chunk` is from this focus, as a number of chunks.
    #[must_use]
    pub fn distance(&self, chunk: ChunkCoord) -> u32 {
        (chunk.x - self.chunk.x)
            .abs()
            .max((chunk.y - self.chunk.y).abs())
            .max((chunk.z - self.chunk.z).abs()) as u32
    }

    /// Whether this focus asks for `chunk`.
    #[must_use]
    pub fn covers(&self, chunk: ChunkCoord) -> bool {
        self.distance(chunk) <= self.radius
    }
}

/// The chunks these focus points ask for, plus the neighbours the asked-for chunks read.
///
/// A chunk of the second parity is solved against its face neighbours, so those have to be in the
/// set even when no focus covers them; otherwise the chunk would be solved with fewer fixed faces
/// when it sits at the edge of the view than when it sits inside it, and its tiles would depend on
/// where the player happened to be.
pub(crate) fn wanted(focus: &[FocusPoint], extent: &WorldExtent) -> BTreeSet<ChunkCoord> {
    let asked: BTreeSet<ChunkCoord> = focus
        .iter()
        .flat_map(|focus| {
            let radius = focus.radius as i32;
            let centre = focus.chunk;
            (-radius..=radius).flat_map(move |x| {
                (-radius..=radius).flat_map(move |y| {
                    (-radius..=radius)
                        .map(move |z| ChunkCoord::new(centre.x + x, centre.y + y, centre.z + z))
                })
            })
        })
        .filter(|chunk| extent.contains_chunk(*chunk))
        .collect();
    with_read_neighbours(&asked, extent)
}

/// `chunks` and the face neighbours their second-parity chunks are solved against, inside the
/// world: everything that has to be generated for `chunks` to be.
pub(crate) fn with_read_neighbours(
    chunks: &BTreeSet<ChunkCoord>,
    extent: &WorldExtent,
) -> BTreeSet<ChunkCoord> {
    let neighbours: BTreeSet<ChunkCoord> = chunks
        .iter()
        .filter(|chunk| chunk.parity() == 1)
        .flat_map(|chunk| chunk.face_neighbours())
        .filter(|chunk| extent.contains_chunk(*chunk))
        .collect();
    chunks.union(&neighbours).copied().collect()
}

/// Which of the lattice's eight classes a chunk's repair belongs to: its coordinates' parities.
/// Two chunks within one chunk of each other, diagonals included, differ along some axis by one
/// and so fall in different classes; repairs of one class are therefore at least two chunks
/// apart, and with a halo short of half a chunk they neither read nor write each other's cells.
pub(crate) const fn repair_class(chunk: ChunkCoord) -> u8 {
    ((chunk.x & 1) | ((chunk.y & 1) << 1) | ((chunk.z & 1) << 2)) as u8
}

/// The chunks a repair of `chunk` reads or rewrites, inside the world, apart from those that wait
/// for it: every chunk within one of it, diagonals included, for a chunk of the second parity; the
/// ones of its own parity for a chunk of the first, whose second-parity neighbours are solved only
/// after it.
///
/// A repair runs only once every one of these has had its first attempt, and once the failed ones
/// of lower classes among them are repaired. What it sees is then the same whatever order the world
/// was generated in, so its result is too. In a world more than one chunk tall, a first-parity
/// repair also reaches the corner chunks of the other parity, which this does not wait for: there,
/// a repair can still depend on the order.
pub(crate) fn repair_neighbourhood(chunk: ChunkCoord, extent: &WorldExtent) -> Vec<ChunkCoord> {
    (-1..=1)
        .flat_map(|x| (-1..=1).flat_map(move |y| (-1..=1).map(move |z| (x, y, z))))
        .filter(|&offset| offset != (0, 0, 0))
        .map(|(x, y, z)| ChunkCoord::new(chunk.x + x, chunk.y + y, chunk.z + z))
        .filter(|neighbour| extent.contains_chunk(*neighbour))
        .filter(|neighbour| chunk.parity() == 1 || neighbour.parity() == chunk.parity())
        .collect()
}

/// The most chunks one batch can hold for a single focus of `radius`: one parity of the chunks it
/// asks for and the neighbours they read, a box two chunks wider than its view where the world
/// allows. What a run dispatches never exceeds it, which is what kernels are compiled for.
pub(crate) fn largest_batch(radius: u32, extent: &WorldExtent) -> u32 {
    let across = 2 * radius + 3;
    let chunks: u32 = (0..3)
        .map(|axis| {
            extent
                .chunks_along(axis)
                .map_or(across, |range| across.min(range.len() as u32))
        })
        .product();
    chunks.div_ceil(2)
}

/// The wanted chunks still to generate, nearest focus first and stable on ties.
///
/// `skip` holds the chunks a dispatch would be wasted on: the ones given up on, and the ones
/// waiting for a repair. Dispatching either again would only fail the same way, because a region's
/// solve depends on nothing that changes in between.
pub(crate) fn missing(
    wanted: &BTreeSet<ChunkCoord>,
    store: &ChunkStore,
    skip: &BTreeSet<ChunkCoord>,
    focus: &[FocusPoint],
) -> Vec<ChunkCoord> {
    let mut missing: Vec<(u32, ChunkCoord)> = wanted
        .iter()
        .filter(|chunk| !store.contains(**chunk) && !skip.contains(*chunk))
        .map(|chunk| {
            let distance = focus
                .iter()
                .map(|focus| focus.distance(*chunk))
                .min()
                .unwrap_or(u32::MAX);
            (distance, *chunk)
        })
        .collect();
    missing.sort_unstable();
    missing.into_iter().map(|(_, chunk)| chunk).collect()
}

/// Whether `chunk` may be solved now.
///
/// A chunk of the first parity has no solved neighbours to wait for. One of the second parity waits
/// until every face neighbour inside the world is either solved or given up on, so it always sees
/// the same borders.
pub(crate) fn eligible(
    chunk: ChunkCoord,
    store: &ChunkStore,
    failed: &BTreeSet<ChunkCoord>,
    extent: &WorldExtent,
) -> bool {
    chunk.parity() == 0
        || chunk
            .face_neighbours()
            .iter()
            .filter(|neighbour| extent.contains_chunk(**neighbour))
            .all(|neighbour| store.contains(*neighbour) || failed.contains(neighbour))
}

/// The next batch to dispatch: up to `max` eligible chunks of one parity, nearest first.
pub(crate) fn next_batch(
    missing: &[ChunkCoord],
    store: &ChunkStore,
    failed: &BTreeSet<ChunkCoord>,
    extent: &WorldExtent,
    max: u32,
) -> Vec<ChunkCoord> {
    // The first parity goes first: the other one waits for it anyway.
    for parity in [0, 1] {
        let batch: Vec<ChunkCoord> = missing
            .iter()
            .copied()
            .filter(|chunk| chunk.parity() == parity)
            .filter(|chunk| eligible(*chunk, store, failed, extent))
            .take(max as usize)
            .collect();
        if !batch.is_empty() {
            return batch;
        }
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use wfc_core::{Chunk, ChunkShape};

    const SHAPE: ChunkShape = ChunkShape::cube(4);

    fn extent() -> WorldExtent {
        WorldExtent::new(SHAPE)
            .with_x(0..8)
            .with_y(0..8)
            .with_z(0..1)
    }

    fn store() -> ChunkStore {
        ChunkStore::new(extent())
    }

    fn solved(store: &mut ChunkStore, chunk: ChunkCoord) {
        store
            .insert(Chunk {
                coord: chunk,
                tiles: vec![0u16; SHAPE.cells() as usize].into_boxed_slice(),
                version: 1,
            })
            .expect("in the world");
    }

    #[test]
    fn chunks_within_one_of_each_other_have_different_repair_classes() {
        let centre = ChunkCoord::new(-3, 4, 0);

        let clashing: Vec<ChunkCoord> = (-1..=1)
            .flat_map(|x| (-1..=1).flat_map(move |y| (-1..=1).map(move |z| (x, y, z))))
            .filter(|&offset| offset != (0, 0, 0))
            .map(|(x, y, z)| ChunkCoord::new(centre.x + x, centre.y + y, centre.z + z))
            .filter(|chunk| repair_class(*chunk) == repair_class(centre))
            .collect();

        assert!(clashing.is_empty(), "{clashing:?}");
    }

    #[test]
    fn a_first_parity_repair_does_not_wait_for_the_chunks_that_wait_for_it() {
        let chunk = ChunkCoord::new(2, 2, 0);

        let seen = repair_neighbourhood(chunk, &extent());

        assert_eq!(seen.len(), 4, "the four diagonals: {seen:?}");
        assert!(seen.iter().all(|n| n.parity() == 0));
    }

    #[test]
    fn a_second_parity_repair_sees_every_chunk_around_it() {
        let chunk = ChunkCoord::new(2, 3, 0);

        let seen = repair_neighbourhood(chunk, &extent());

        assert_eq!(seen.len(), 8, "{seen:?}");
    }

    #[test]
    fn a_focus_asks_for_its_neighbourhood_inside_the_world() {
        let focus = [FocusPoint::new(ChunkCoord::new(0, 0, 0), 1)];

        let wanted = wanted(&focus, &extent());

        // A 3x3 square clipped to the world's corner (four chunks), plus the two neighbours the
        // second-parity chunks of that square read, and nothing outside the world.
        assert_eq!(wanted.len(), 6, "{wanted:?}");
        assert!(wanted.contains(&ChunkCoord::new(1, 1, 0)));
        assert!(
            wanted.contains(&ChunkCoord::new(2, 0, 0)),
            "read by (1, 0, 0)"
        );
        assert!(
            wanted.contains(&ChunkCoord::new(0, 2, 0)),
            "read by (0, 1, 0)"
        );
        assert!(!wanted.contains(&ChunkCoord::new(-1, 0, 0)));
        assert!(
            !wanted.contains(&ChunkCoord::new(2, 2, 0)),
            "nothing diagonal"
        );
    }

    #[test]
    fn a_wanted_chunk_brings_the_neighbours_it_reads() {
        // (1, 0, 0) is of the second parity, so it is solved against its four side neighbours.
        let focus = [FocusPoint::new(ChunkCoord::new(1, 0, 0), 0)];

        let wanted = wanted(&focus, &extent());

        assert!(wanted.contains(&ChunkCoord::new(1, 0, 0)));
        assert!(
            wanted.contains(&ChunkCoord::new(0, 0, 0)),
            "the neighbour it reads"
        );
        assert!(wanted.contains(&ChunkCoord::new(2, 0, 0)));
        assert!(
            !wanted.contains(&ChunkCoord::new(1, 1, 1)),
            "nothing diagonal"
        );
    }

    #[test]
    fn missing_chunks_come_nearest_first_and_skip_what_is_done() {
        let mut store = store();
        solved(&mut store, ChunkCoord::new(4, 4, 0));
        let focus = [FocusPoint::new(ChunkCoord::new(4, 4, 0), 2)];
        let failed = BTreeSet::from([ChunkCoord::new(5, 4, 0)]);

        let missing = missing(&wanted(&focus, &extent()), &store, &failed, &focus);

        assert!(
            !missing.contains(&ChunkCoord::new(4, 4, 0)),
            "already solved"
        );
        assert!(!missing.contains(&ChunkCoord::new(5, 4, 0)), "given up on");
        let distances: Vec<u32> = missing.iter().map(|c| focus[0].distance(*c)).collect();
        assert!(
            distances.windows(2).all(|pair| pair[0] <= pair[1]),
            "{distances:?}"
        );
    }

    #[test]
    fn a_batch_never_holds_two_chunks_that_share_a_face() {
        let store = store();
        let failed = BTreeSet::new();
        let focus = [FocusPoint::new(ChunkCoord::new(4, 4, 0), 3)];
        let missing = missing(&wanted(&focus, &extent()), &store, &failed, &focus);

        let batch = next_batch(&missing, &store, &failed, &extent(), 64);

        assert!(!batch.is_empty());
        for chunk in &batch {
            for neighbour in chunk.face_neighbours() {
                assert!(
                    !batch.contains(&neighbour),
                    "{chunk:?} and {neighbour:?} share a face"
                );
            }
        }
    }

    #[test]
    fn the_second_parity_waits_for_the_neighbours_it_reads() {
        let mut store = store();
        let failed = BTreeSet::new();
        let odd = ChunkCoord::new(1, 0, 0);

        assert!(
            !eligible(odd, &store, &failed, &extent()),
            "its neighbours are not solved"
        );
        for neighbour in odd.face_neighbours() {
            if extent().contains_chunk(neighbour) {
                solved(&mut store, neighbour);
            }
        }
        assert!(
            eligible(odd, &store, &failed, &extent()),
            "now it sees fixed borders"
        );
    }

    #[test]
    fn a_chunk_given_up_on_does_not_hold_up_its_neighbours() {
        let mut store = store();
        let odd = ChunkCoord::new(1, 0, 0);
        let mut failed = BTreeSet::new();
        for neighbour in odd.face_neighbours() {
            if extent().contains_chunk(neighbour) {
                solved(&mut store, neighbour);
            }
        }
        store.remove(ChunkCoord::new(0, 0, 0));
        failed.insert(ChunkCoord::new(0, 0, 0));

        assert!(eligible(odd, &store, &failed, &extent()));
    }

    #[test]
    fn no_parity_of_what_a_focus_asks_for_exceeds_the_largest_batch() {
        let worlds = [
            WorldExtent::new(ChunkShape::cube(4)).with_z(0..1),
            WorldExtent::new(ChunkShape::cube(4)),
            extent(),
        ];
        for world in &worlds {
            for radius in 0..5 {
                for centre in [ChunkCoord::new(0, 0, 0), ChunkCoord::new(1, 2, 0)] {
                    let asked = wanted(&[FocusPoint::new(centre, radius)], world);

                    let largest = largest_batch(radius, world);

                    for parity in 0..2 {
                        let batch = asked.iter().filter(|c| c.parity() == parity).count() as u32;
                        assert!(
                            batch <= largest,
                            "radius {radius} at {centre:?}: {batch} chunks of parity {parity}, \
                             bound {largest}"
                        );
                    }
                }
            }
        }
    }
}
