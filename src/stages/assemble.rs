//! Assemblies: pieces grown from connectors inside a site's footprint, as Minecraft's jigsaw
//! villages and Valheim's dungeons are (docs/reference/packs.md, Assemble).
//!
//! Growth reads nothing but the pieces, the footprint and a hash stream seeded from the site, so a
//! site's assembly is the same whichever chunk asks for it first.

use super::pack::{Facing, Piece};
use wfc_core::hash::pcg3d;

/// A piece placed in an assembly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Placed {
    /// The piece's index in the stage's list.
    pub(crate) piece: usize,
    /// Its lowest cell along the lattice's x and y, and its lowest level.
    pub(crate) min: [i64; 3],
    /// Its cells along x and y and its levels, as turned.
    pub(crate) size: [u32; 3],
    /// Quarter turns from +x toward +y.
    pub(crate) turn: u8,
}

impl Placed {
    fn overlaps(&self, other: &Self) -> bool {
        (0..3).all(|axis| {
            self.min[axis] < other.min[axis] + i64::from(other.size[axis])
                && other.min[axis] < self.min[axis] + i64::from(self.size[axis])
        })
    }
}

/// How an assembly grows: its pieces, the one it starts from, and its limits.
pub(crate) struct Growth<'a> {
    pub(crate) pieces: &'a [Piece],
    pub(crate) start: usize,
    pub(crate) max: u32,
    pub(crate) min: u32,
    pub(crate) tries: u32,
    pub(crate) rerolls: u32,
}

/// A door not yet joined: the cell it opens from, the way it faces, and its kind.
struct Open<'a> {
    at: [i64; 3],
    facing: Facing,
    kind: &'a str,
}

/// A draw from a hash stream: each call takes the next value.
struct Draws {
    seed: [u32; 2],
    count: u32,
}

impl Draws {
    fn next(&mut self) -> u32 {
        self.count += 1;
        pcg3d([self.seed[0], self.seed[1], self.count])[0]
    }

    /// A whole number below `bound`, which is above zero.
    fn below(&mut self, bound: u32) -> u32 {
        self.next() % bound
    }
}

impl Growth<'_> {
    /// The assembly grown inside the columns from `min` up to but not including `max`, from the
    /// hash stream `seed`: the first attempt with at least `min` growing pieces, or `None` if every
    /// reroll falls short.
    pub(crate) fn grow(&self, min: [i64; 2], max: [i64; 2], seed: [u32; 2]) -> Option<Vec<Placed>> {
        (0..=self.rerolls).find_map(|attempt| {
            let draws = Draws {
                seed: [seed[0] ^ attempt.wrapping_mul(0x9E37_79B9), seed[1]],
                count: 0,
            };
            let placed = self.attempt(min, max, draws);
            let grown = placed
                .iter()
                .filter(|placed| !self.pieces[placed.piece].end)
                .count();
            (grown >= self.min as usize).then_some(placed)
        })
    }

    fn attempt(&self, min: [i64; 2], max: [i64; 2], mut draws: Draws) -> Vec<Placed> {
        let inside = |placed: &Placed| {
            (0..2).all(|axis| {
                placed.min[axis] >= min[axis]
                    && placed.min[axis] + i64::from(placed.size[axis]) <= max[axis]
            })
        };
        let start = &self.pieces[self.start];
        let turn = draws.below(4) as u8;
        let size = turned_size(start, turn);
        let first = Placed {
            piece: self.start,
            min: [
                (min[0] + max[0] - i64::from(size[0])).div_euclid(2),
                (min[1] + max[1] - i64::from(size[1])).div_euclid(2),
                0,
            ],
            size,
            turn,
        };
        if !inside(&first) {
            return Vec::new();
        }
        let mut open: std::collections::VecDeque<Open<'_>> =
            doors_of(start, &first, None).collect();
        let mut placed = vec![first];
        let mut unjoined = Vec::new();
        while let Some(door) = open.pop_front() {
            if placed.len() >= self.max as usize {
                unjoined.push(door);
                continue;
            }
            let joined = (0..self.tries).find_map(|_| {
                let (piece, own) = self.draw(&door, &mut draws)?;
                let candidate = join(&self.pieces[piece], piece, own, &door);
                (inside(&candidate) && placed.iter().all(|other| !other.overlaps(&candidate)))
                    .then_some((candidate, own))
            });
            match joined {
                Some((candidate, own)) => {
                    open.extend(doors_of(
                        &self.pieces[candidate.piece],
                        &candidate,
                        Some(own),
                    ));
                    placed.push(candidate);
                }
                None => unjoined.push(door),
            }
        }
        for door in unjoined {
            let end = self.pieces.iter().enumerate().find_map(|(index, piece)| {
                if !piece.end {
                    return None;
                }
                piece.doors.iter().enumerate().find_map(|(own, candidate)| {
                    if candidate.kind != door.kind {
                        return None;
                    }
                    let candidate = join(piece, index, own, &door);
                    (inside(&candidate) && placed.iter().all(|other| !other.overlaps(&candidate)))
                        .then_some(candidate)
                })
            });
            placed.extend(end);
        }
        placed
    }

    /// A growing piece drawn by weight among those with a door of `door`'s kind, and one of those
    /// doors drawn evenly; `None` if no piece has one.
    fn draw(&self, door: &Open<'_>, draws: &mut Draws) -> Option<(usize, usize)> {
        let matching = |piece: &Piece| {
            piece
                .doors
                .iter()
                .enumerate()
                .filter(|(_, own)| own.kind == door.kind)
                .map(|(index, _)| index)
                .collect::<Vec<usize>>()
        };
        let total: u64 = self
            .pieces
            .iter()
            .filter(|piece| !piece.end && !matching(piece).is_empty())
            .map(|piece| u64::from(piece.weight))
            .sum();
        if total == 0 {
            return None;
        }
        let mut pick = u64::from(draws.next()) % total;
        for (index, piece) in self.pieces.iter().enumerate() {
            let doors = matching(piece);
            if piece.end || doors.is_empty() {
                continue;
            }
            if pick < u64::from(piece.weight) {
                let own = doors[draws.below(doors.len() as u32) as usize];
                return Some((index, own));
            }
            pick -= u64::from(piece.weight);
        }
        unreachable!("the pick is below the total weight")
    }
}

/// `piece`, the piece of index `index`, turned so its door `own` faces `door` from the next cell.
fn join(piece: &Piece, index: usize, own: usize, door: &Open<'_>) -> Placed {
    let target = opposite(door.facing);
    let from = &piece.doors[own];
    let turn = (0..4u8)
        .find(|&turn| turned_facing(from.facing, turn) == target)
        .expect("four quarter turns face every way");
    let size = turned_size(piece, turn);
    let (x, y) = turned_cell(piece, turn, (from.at.0, from.at.1));
    let (sx, sy) = step(door.facing);
    Placed {
        piece: index,
        min: [
            door.at[0] + sx - i64::from(x),
            door.at[1] + sy - i64::from(y),
            door.at[2] - i64::from(from.at.2),
        ],
        size,
        turn,
    }
}

/// The doors of `piece`, placed as `placed`, in the world, leaving out the door `joined` by.
fn doors_of<'a>(
    piece: &'a Piece,
    placed: &Placed,
    joined: Option<usize>,
) -> impl Iterator<Item = Open<'a>> {
    let (turn, min) = (placed.turn, placed.min);
    piece
        .doors
        .iter()
        .enumerate()
        .filter(move |&(index, _)| Some(index) != joined)
        .map(move |(_, door)| {
            let (x, y) = turned_cell(piece, turn, (door.at.0, door.at.1));
            Open {
                at: [
                    min[0] + i64::from(x),
                    min[1] + i64::from(y),
                    min[2] + i64::from(door.at.2),
                ],
                facing: turned_facing(door.facing, turn),
                kind: &door.kind,
            }
        })
}

fn turned_size(piece: &Piece, turn: u8) -> [u32; 3] {
    let (x, y, levels) = piece.size;
    if turn.is_multiple_of(2) {
        [x, y, levels]
    } else {
        [y, x, levels]
    }
}

/// A cell of `piece` after `turn` quarter turns from +x toward +y, within the turned box.
fn turned_cell(piece: &Piece, turn: u8, cell: (u32, u32)) -> (u32, u32) {
    let (mut width, mut depth) = (piece.size.0, piece.size.1);
    let mut cell = cell;
    for _ in 0..turn {
        cell = (depth - 1 - cell.1, cell.0);
        (width, depth) = (depth, width);
    }
    cell
}

fn turned_facing(facing: Facing, turn: u8) -> Facing {
    (0..turn).fold(facing, |facing, _| match facing {
        Facing::East => Facing::North,
        Facing::North => Facing::West,
        Facing::West => Facing::South,
        Facing::South => Facing::East,
    })
}

const fn opposite(facing: Facing) -> Facing {
    match facing {
        Facing::North => Facing::South,
        Facing::East => Facing::West,
        Facing::South => Facing::North,
        Facing::West => Facing::East,
    }
}

/// The cell next to a door, along the way it faces.
const fn step(facing: Facing) -> (i64, i64) {
    match facing {
        Facing::North => (0, 1),
        Facing::East => (1, 0),
        Facing::South => (0, -1),
        Facing::West => (-1, 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stages::pack::Door;

    fn piece(name: &str, size: (u32, u32, u32), doors: &[((u32, u32, u32), Facing)]) -> Piece {
        Piece {
            name: name.to_owned(),
            size,
            weight: 1,
            end: false,
            doors: doors
                .iter()
                .map(|&(at, facing)| Door {
                    at,
                    facing,
                    kind: "door".to_owned(),
                })
                .collect(),
        }
    }

    #[test]
    fn a_quarter_turn_takes_east_to_north_and_keeps_a_door_on_its_side() {
        let hall = piece("hall", (4, 2, 1), &[((3, 0, 0), Facing::East)]);

        let turned = (0..4u8)
            .map(|turn| {
                let facing = turned_facing(Facing::East, turn);
                let cell = turned_cell(&hall, turn, (3, 0));
                let [x, y, _] = turned_size(&hall, turn);
                (facing, cell, (x, y))
            })
            .collect::<Vec<_>>();

        assert_eq!(
            turned,
            vec![
                (Facing::East, (3, 0), (4, 2)),
                (Facing::North, (1, 3), (2, 4)),
                (Facing::West, (0, 1), (4, 2)),
                (Facing::South, (0, 0), (2, 4)),
            ]
        );
    }

    #[test]
    fn every_piece_joins_an_earlier_one_door_to_door_and_none_overlap() {
        let pieces = vec![
            piece(
                "hub",
                (3, 3, 1),
                &[((2, 1, 0), Facing::East), ((1, 2, 0), Facing::North)],
            ),
            piece(
                "hall",
                (1, 3, 1),
                &[((0, 0, 0), Facing::South), ((0, 2, 0), Facing::North)],
            ),
            piece(
                "bend",
                (2, 2, 1),
                &[((0, 0, 0), Facing::South), ((1, 1, 0), Facing::East)],
            ),
        ];
        let growth = Growth {
            pieces: &pieces,
            start: 0,
            max: 30,
            min: 10,
            tries: 20,
            rerolls: 4,
        };

        let placed = growth.grow([0, 0], [32, 32], [7, 9]).expect("an assembly");

        for (index, later) in placed.iter().enumerate().skip(1) {
            let joins = doors_of(&pieces[later.piece], later, None).any(|door| {
                let (sx, sy) = step(door.facing);
                let next = [door.at[0] + sx, door.at[1] + sy, door.at[2]];
                placed[..index].iter().any(|earlier| {
                    doors_of(&pieces[earlier.piece], earlier, None)
                        .any(|back| back.at == next && back.facing == opposite(door.facing))
                })
            });
            assert!(joins, "piece {index} {later:?} joins nothing before it");
            for other in &placed[..index] {
                assert!(!later.overlaps(other), "{later:?} and {other:?}");
            }
        }
        assert!(placed.len() >= 10);
    }

    #[test]
    fn a_joined_piece_opens_from_the_cell_next_to_the_door() {
        let room = piece("room", (3, 3, 1), &[((0, 1, 0), Facing::West)]);
        let door = Open {
            at: [10, 4, 2],
            facing: Facing::North,
            kind: "door",
        };

        let placed = join(&room, 0, 0, &door);
        let opened = doors_of(&room, &placed, None).next().expect("a door");

        assert_eq!(opened.facing, Facing::South);
        assert_eq!(opened.at, [10, 5, 2]);
    }
}
