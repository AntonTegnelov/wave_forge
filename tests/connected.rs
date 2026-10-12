//! A Connected stage: the connected areas of a Rules stage's categories, labelled per region. Two
//! neighbouring columns of a category share an id, columns of two categories never do, every id's
//! columns are one area, ids count up in the order each area's first column comes row by row, and
//! they are the same whatever order the chunks are generated in.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use wave_forge::stages::{Pack, Runtime};
use wave_forge::{ChunkCoord, FocusPoint};

const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "ground", kind: Field(Noise(frequency: 0.08, octaves: 2))),
        (name: "land", kind: Rules(rules: [
            (category: "high", when: [Greater(Input("ground"), Constant(0.5))]),
        ], otherwise: "low")),
        (name: "areas", kind: Connected(input: "land", region: 4)),
    ],
)"#;

const SIZE: [u32; 2] = [8, 8];
/// Columns along a region's side: 4 chunks of 8.
const REGION: i64 = 32;

fn area() -> Vec<ChunkCoord> {
    (-4..4)
        .flat_map(|y| (-4..4).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

/// Every column of the area: its category and its area's id, each chunk read once its request has
/// run, in the order of `requests`.
fn columns_asked(requests: &[Vec<ChunkCoord>]) -> BTreeMap<(i64, i64), (u8, f32)> {
    let mut runtime = Runtime::new(Arc::new(Pack::parse(PACK).expect("a valid pack")), 6, SIZE);
    let mut out = BTreeMap::new();
    for request in requests {
        let focus: Vec<FocusPoint> = request.iter().map(|&c| FocusPoint::new(c, 0)).collect();
        runtime
            .request(&focus, &["land", "areas"])
            .expect("the stages");
        runtime.run_until_idle().expect("the stages run");
        for &chunk in request {
            let land = runtime.categories("land", chunk).expect("land");
            let areas = runtime.field("areas", chunk).expect("areas");
            for y in 0..SIZE[1] {
                for x in 0..SIZE[0] {
                    let column = (
                        i64::from(chunk.x) * i64::from(SIZE[0]) + i64::from(x),
                        i64::from(chunk.y) * i64::from(SIZE[1]) + i64::from(y),
                    );
                    out.insert(column, (land.get(x, y), areas.get(x, y)));
                }
            }
        }
    }
    out
}

fn columns() -> BTreeMap<(i64, i64), (u8, f32)> {
    columns_asked(&[area()])
}

fn region_of((x, y): (i64, i64)) -> (i64, i64) {
    (x.div_euclid(REGION), y.div_euclid(REGION))
}

/// The columns of `region`, row by row.
fn region_columns((rx, ry): (i64, i64)) -> impl Iterator<Item = (i64, i64)> {
    (0..REGION).flat_map(move |y| (0..REGION).map(move |x| (rx * REGION + x, ry * REGION + y)))
}

fn regions() -> BTreeSet<(i64, i64)> {
    columns().keys().map(|&column| region_of(column)).collect()
}

#[test]
fn neighbours_of_one_category_share_an_id_and_two_categories_never_do() {
    let columns = columns();

    for (&(x, y), &(category, id)) in &columns {
        for next in [(x + 1, y), (x, y + 1)] {
            let Some(&(other, other_id)) = columns.get(&next) else {
                continue;
            };
            if region_of(next) != region_of((x, y)) {
                continue;
            }
            assert_eq!(
                category == other,
                id == other_id,
                "({x}, {y}) and {next:?}: categories {category} and {other}, ids {id} and {other_id}"
            );
        }
    }
}

#[test]
fn every_ids_columns_are_one_area() {
    let columns = columns();

    for region in regions() {
        let mut by_id: BTreeMap<u32, Vec<(i64, i64)>> = BTreeMap::new();
        for column in region_columns(region) {
            by_id
                .entry(columns[&column].1 as u32)
                .or_default()
                .push(column);
        }
        for (id, members) in by_id {
            let members: BTreeSet<(i64, i64)> = members.into_iter().collect();
            let start = *members.iter().next().expect("an id has columns");
            let mut reached = BTreeSet::from([start]);
            let mut frontier = vec![start];
            while let Some((x, y)) = frontier.pop() {
                for next in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] {
                    if members.contains(&next) && reached.insert(next) {
                        frontier.push(next);
                    }
                }
            }
            assert_eq!(reached, members, "region {region:?}, id {id}");
        }
    }
}

#[test]
fn ids_count_up_in_the_order_each_area_first_comes_row_by_row() {
    let columns = columns();

    for region in regions() {
        let mut next = 0.0;
        for column in region_columns(region) {
            let id = columns[&column].1;
            assert!(id <= next, "region {region:?}: id {id} before {next}");
            if id == next {
                next += 1.0;
            }
        }
        assert!(next > 2.0, "region {region:?} has only {next} areas");
    }
}

#[test]
fn the_ids_are_the_same_whatever_order_the_chunks_come_in() {
    let mut reversed = area();
    reversed.reverse();
    let singly: Vec<Vec<ChunkCoord>> = reversed.into_iter().map(|chunk| vec![chunk]).collect();

    assert_eq!(columns_asked(&singly), columns());
}

#[test]
fn loading_refuses_a_region_of_no_chunks_and_an_input_without_categories() {
    let none = PACK.replace("region: 4)", "region: 0)");
    let field = PACK.replace(r#"Connected(input: "land""#, r#"Connected(input: "ground""#);

    assert!(Pack::parse(&none).is_err(), "a region of 0 chunks");
    assert!(Pack::parse(&field).is_err(), "a field as the input");
}

#[test]
fn sampling_gives_the_ids_the_chunks_hold() {
    let runtime = Runtime::new(Arc::new(Pack::parse(PACK).expect("a valid pack")), 6, SIZE);
    let columns = columns();

    let sampled = runtime
        .atlas("areas", [-32, -32], [64, 64])
        .expect("a sampled stage");

    for (i, id) in sampled.into_iter().enumerate() {
        let column = (-32 + (i % 64) as i64, -32 + (i / 64) as i64);
        assert_eq!(id, columns[&column].1, "{column:?}");
    }
}
