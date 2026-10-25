//! Decals along curves: a road's pieces tile it from end to end across chunk borders, each one as
//! long as the road is wide there, lying along the road and standing on the ground at its middle.

use wave_forge::ChunkCoord;
use wave_forge::decals::{DecalPiece, curve_decals};
use wave_forge::stages::regions::{Curve, CurveId};

const SIZE: [u32; 2] = [16, 16];
const CELL: [f32; 3] = [2.0, 0.5, 2.0];

fn curve(points: Vec<[f32; 2]>, values: Vec<f32>) -> Curve {
    Curve {
        id: CurveId::Region {
            region: (0, 0),
            index: 0,
        },
        points,
        values,
        heights: Vec::new(),
    }
}

/// Ground whose height in cells is the column's x plus twice its y.
fn sloped(x: i64, y: i64) -> Option<f32> {
    Some(x as f32 + 2.0 * y as f32)
}

/// Every piece of `curves` in the chunks 0 to 3 each way.
fn all_pieces(curves: &[Curve]) -> Vec<(ChunkCoord, DecalPiece)> {
    (0..4)
        .flat_map(|y| (0..4).map(move |x| ChunkCoord::new(x, y, 0)))
        .flat_map(|chunk| {
            curve_decals(curves, chunk, SIZE, sloped, CELL)
                .expect("the height is everywhere")
                .into_iter()
                .map(move |piece| (chunk, piece))
        })
        .collect()
}

#[test]
fn a_straight_roads_pieces_tile_it_end_to_end_across_chunk_borders() {
    let road = curve(vec![[1.0, 20.0], [61.0, 20.0]], vec![1.5, 1.5]);

    let mut pieces = all_pieces(&[road]);

    pieces.sort_by(|a, b| a.1.at[0].total_cmp(&b.1.at[0]));
    let chunks: std::collections::BTreeSet<i32> = pieces.iter().map(|(c, _)| c.x).collect();
    assert_eq!(chunks.len(), 4, "the road crosses all four chunks");
    let first = &pieces[0].1;
    assert!(
        (first.at[0] - first.length / 2.0 - 1.0 * CELL[0]).abs() < 1e-3,
        "the first piece starts where the road does: {first:?}"
    );
    let last = &pieces[pieces.len() - 1].1;
    assert!(
        (last.at[0] + last.length / 2.0 - 61.0 * CELL[0]).abs() < 1e-3,
        "the last piece ends where the road does: {last:?}"
    );
    for pair in pieces.windows(2) {
        let (a, b) = (&pair[0].1, &pair[1].1);
        let gap = (b.at[0] - b.length / 2.0) - (a.at[0] + a.length / 2.0);
        assert!(
            gap.abs() < 1e-3,
            "pieces meet without a gap or an overlap: {a:?} then {b:?}"
        );
    }
}

#[test]
fn every_piece_belongs_to_the_one_chunk_its_middle_stands_in() {
    let road = curve(vec![[1.0, 20.0], [61.0, 20.0]], vec![1.5, 1.5]);

    let pieces = all_pieces(&[road]);

    for (chunk, piece) in &pieces {
        let column = [
            (piece.at[0] / CELL[0]).floor() as i32,
            (piece.at[2] / CELL[2]).floor() as i32,
        ];
        assert_eq!(
            [column[0].div_euclid(16), column[1].div_euclid(16)],
            [chunk.x, chunk.y],
            "{piece:?}"
        );
    }
}

#[test]
fn a_piece_is_as_long_as_the_road_is_wide_there() {
    let road = curve(vec![[1.0, 20.0], [61.0, 20.0]], vec![1.0, 3.0]);

    let mut pieces = all_pieces(&[road]);

    pieces.sort_by(|a, b| a.1.at[0].total_cmp(&b.1.at[0]));
    for (_, piece) in &pieces[..pieces.len() - 1] {
        let radius =
            1.0 + 2.0 * (piece.at[0] / CELL[0] - piece.length / CELL[0] / 2.0 - 1.0) / 60.0;
        assert!(
            (piece.length - 2.0 * radius * CELL[0]).abs() < 1e-3,
            "a piece is as long as the road's width at its start: {piece:?}"
        );
        assert!(
            piece.width >= piece.length - 1e-3,
            "and at least as wide as it is long, the road widening: {piece:?}"
        );
    }
}

#[test]
fn a_piece_lies_along_its_road() {
    let road = curve(vec![[4.0, 4.0], [40.0, 40.0]], vec![1.0, 1.0]);

    let pieces = all_pieces(&[road]);

    assert!(!pieces.is_empty());
    let diagonal = std::f32::consts::FRAC_1_SQRT_2;
    for (_, piece) in &pieces {
        assert!(
            (piece.along[0] - diagonal).abs() < 1e-4 && (piece.along[1] - diagonal).abs() < 1e-4,
            "{piece:?}"
        );
        let (x, z) = (piece.at[0] / CELL[0], piece.at[2] / CELL[2]);
        assert!((x - z).abs() < 1e-3, "its middle is on the road: {piece:?}");
    }
}

#[test]
fn a_piece_stands_on_the_ground_at_its_middle() {
    let road = curve(vec![[1.0, 20.5], [61.0, 20.5]], vec![1.0, 1.0]);

    let pieces = all_pieces(&[road]);

    assert!(!pieces.is_empty());
    for (_, piece) in &pieces {
        let column = [
            (piece.at[0] / CELL[0]).floor() as i64,
            (piece.at[2] / CELL[2]).floor() as i64,
        ];
        let ground = sloped(column[0], column[1]).expect("everywhere") * CELL[1];
        assert_eq!(piece.at[1], ground, "{piece:?}");
    }
}

#[test]
fn without_the_ground_under_a_piece_there_are_no_pieces_yet() {
    let road = curve(vec![[1.0, 20.0], [61.0, 20.0]], vec![1.0, 1.0]);

    let pieces = curve_decals(&[road], ChunkCoord::new(1, 1, 0), SIZE, |_, _| None, CELL);

    assert_eq!(pieces, None);
}
