//! A Passages stage: an open field with passages dug so that every open column reaches every other.
//! Within a region its open areas are joined, and each of its edges has a crossing its neighbour
//! across the edge digs to as well, so a whole area of regions is one open network. The input's open
//! columns stay open, a bounded world of one region digs nothing it does not need, and the field is
//! the same whatever order the chunks come in and when sampled.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use wave_forge::stages::{Pack, Runtime};
use wave_forge::{ChunkCoord, FocusPoint};

const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "caves", kind: Field(Noise(frequency: 0.15, octaves: 2))),
        (name: "open", kind: Passages(open: "caves", above: 0.6, region: 2)),
    ],
)"#;

const SIZE: [u32; 2] = [8, 8];
/// Columns along a region's side: 2 chunks of 8.
const REGION: i64 = 16;

type Column = (i64, i64);

/// 4 by 4 regions.
fn area() -> Vec<ChunkCoord> {
    (-4..4)
        .flat_map(|y| (-4..4).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

/// Every column of `chunks`: the input's value and the passages', each chunk read once its request
/// has run, in the order of `requests`.
fn columns_asked(pack: &str, requests: &[Vec<ChunkCoord>]) -> BTreeMap<Column, (f32, f32)> {
    let mut runtime = Runtime::new(Arc::new(Pack::parse(pack).expect("a valid pack")), 6, SIZE);
    let mut out = BTreeMap::new();
    for request in requests {
        let focus: Vec<FocusPoint> = request.iter().map(|&c| FocusPoint::new(c, 0)).collect();
        runtime
            .request(&focus, &["caves", "open"])
            .expect("the stages");
        runtime.run_until_idle().expect("the stages run");
        for &chunk in request {
            let caves = runtime.field("caves", chunk).expect("caves");
            let open = runtime.field("open", chunk).expect("open");
            for y in 0..SIZE[1] {
                for x in 0..SIZE[0] {
                    let column = (
                        i64::from(chunk.x) * i64::from(SIZE[0]) + i64::from(x),
                        i64::from(chunk.y) * i64::from(SIZE[1]) + i64::from(y),
                    );
                    out.insert(column, (caves.get(x, y), open.get(x, y)));
                }
            }
        }
    }
    out
}

fn columns() -> BTreeMap<Column, (f32, f32)> {
    columns_asked(PACK, &[area()])
}

/// The areas of `members`, columns joined edge to edge.
fn areas(members: &BTreeSet<Column>) -> Vec<BTreeSet<Column>> {
    let mut left = members.clone();
    let mut out = Vec::new();
    while let Some(&start) = left.iter().next() {
        left.remove(&start);
        let mut area = BTreeSet::from([start]);
        let mut frontier = vec![start];
        while let Some((x, y)) = frontier.pop() {
            for next in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] {
                if left.remove(&next) {
                    area.insert(next);
                    frontier.push(next);
                }
            }
        }
        out.push(area);
    }
    out
}

fn open(columns: &BTreeMap<Column, (f32, f32)>) -> BTreeSet<Column> {
    columns
        .iter()
        .filter(|(_, (_, open))| *open == 1.0)
        .map(|(&column, _)| column)
        .collect()
}

fn region_of((x, y): Column) -> Column {
    (x.div_euclid(REGION), y.div_euclid(REGION))
}

#[test]
fn every_open_column_of_an_area_of_regions_reaches_every_other() {
    let columns = columns();
    let open_before: BTreeSet<Column> = columns
        .iter()
        .filter(|(_, (caves, _))| *caves > 0.6)
        .map(|(&column, _)| column)
        .collect();

    let after = areas(&open(&columns));

    assert!(
        areas(&open_before).len() > 8,
        "the caves are {} areas in 16 regions",
        areas(&open_before).len()
    );
    assert_eq!(after.len(), 1, "{} areas", after.len());
}

#[test]
fn the_inputs_open_columns_stay_open_and_every_column_is_open_or_closed() {
    let columns = columns();

    for (column, &(caves, open)) in &columns {
        assert!(open == 0.0 || open == 1.0, "{column:?}: {open}");
        if caves > 0.6 {
            assert_eq!(open, 1.0, "{column:?}");
        }
    }
}

#[test]
fn a_bounded_world_of_one_region_whose_open_columns_are_one_area_digs_nothing() {
    let disc = PACK
        .replace(
            "Noise(frequency: 0.15, octaves: 2)",
            "Sub(Constant(5.0), Distance((8.0, 8.0)))",
        )
        .replace("above: 0.6", "above: 0.0")
        .replace(
            "version: 1,",
            "version: 1,\n    bound: Some(Rect(min: (0.0, 0.0), max: (15.0, 15.0))),",
        );
    let region: Vec<ChunkCoord> = (0..2)
        .flat_map(|y| (0..2).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect();

    let columns = columns_asked(&disc, &[region]);

    for (column, &(caves, open)) in &columns {
        assert_eq!(open, if caves > 0.0 { 1.0 } else { 0.0 }, "{column:?}");
    }
}

#[test]
fn without_a_bound_even_solid_rock_is_dug_through_to_every_crossing() {
    let solid = PACK.replace("Noise(frequency: 0.15, octaves: 2)", "Constant(0.0)");

    let columns = columns_asked(&solid, &[area()]);
    let dug = open(&columns);

    assert!(!dug.is_empty());
    assert_eq!(areas(&dug).len(), 1);
    for (rx, ry) in (-2..2).flat_map(|y| (-2..2).map(move |x| (x, y))) {
        let (min, max) = (
            (rx * REGION, ry * REGION),
            ((rx + 1) * REGION - 1, (ry + 1) * REGION - 1),
        );
        let on = |edge: &dyn Fn(&Column) -> bool| dug.iter().filter(|c| edge(c)).count();
        let within = |c: &Column| region_of(*c) == (rx, ry);
        assert!(on(&|c| within(c) && c.0 == min.0) > 0, "west of {rx}, {ry}");
        assert!(on(&|c| within(c) && c.0 == max.0) > 0, "east of {rx}, {ry}");
        assert!(
            on(&|c| within(c) && c.1 == min.1) > 0,
            "south of {rx}, {ry}"
        );
        assert!(
            on(&|c| within(c) && c.1 == max.1) > 0,
            "north of {rx}, {ry}"
        );
    }
}

#[test]
fn the_passages_are_the_same_whatever_order_the_chunks_come_in() {
    let mut reversed = area();
    reversed.reverse();
    let singly: Vec<Vec<ChunkCoord>> = reversed.into_iter().map(|chunk| vec![chunk]).collect();

    assert_eq!(columns_asked(PACK, &singly), columns());
}

#[test]
fn sampling_gives_what_the_chunks_hold() {
    let runtime = Runtime::new(Arc::new(Pack::parse(PACK).expect("a valid pack")), 6, SIZE);
    let columns = columns();

    let sampled = runtime
        .atlas("open", [-32, -32], [64, 64])
        .expect("a sampled stage");

    for (i, value) in sampled.into_iter().enumerate() {
        let column = (-32 + (i % 64) as i64, -32 + (i / 64) as i64);
        assert_eq!(value, columns[&column].1, "{column:?}");
    }
}

#[test]
fn loading_refuses_a_region_of_no_chunks_and_an_input_that_is_no_field() {
    let none = PACK.replace("region: 2)", "region: 0)");
    let categories = PACK.replace(
        "(name: \"open\"",
        "(name: \"rock\", kind: Rules(rules: [], otherwise: \"rock\")),\n        (name: \"open\"",
    )
    .replace("open: \"caves\"", "open: \"rock\"");

    assert!(Pack::parse(&none).is_err(), "a region of 0 chunks");
    assert!(Pack::parse(&categories).is_err(), "categories as the input");
}
