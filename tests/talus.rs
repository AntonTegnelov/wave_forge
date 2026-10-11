//! A Talus stage: a height field weathered per region, steep ground shedding debris onto the ground
//! below it, the same in any order, meeting its input along every region border, and its debris's
//! depth a Field stage away.

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::stages::{Pack, PackError, Runtime};
use wave_forge::{ChunkCoord, FocusPoint};

const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "ground", kind: Field(Mul(Noise(frequency: 0.05, octaves: 4), Constant(60.0)))),
        (name: "talus", kind: Talus(height: "ground", region: 4, slope: 0.6, fade: 8)),
        (name: "debris", kind: Field(Max(Sub(Input("talus"), Input("ground")), Constant(0.0)))),
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

fn pack(text: &str) -> Result<Pack, PackError> {
    Pack::parse(text)
}

/// Every column of the area: its ground, its weathered ground and its debris, each chunk read once
/// its request has run, in the order of `requests`.
fn columns_asked(requests: &[Vec<ChunkCoord>]) -> BTreeMap<(i64, i64), [f32; 3]> {
    let mut runtime = Runtime::new(Arc::new(pack(PACK).expect("a valid pack")), 6, SIZE);
    let mut out = BTreeMap::new();
    for request in requests {
        let focus: Vec<FocusPoint> = request.iter().map(|&c| FocusPoint::new(c, 0)).collect();
        runtime
            .request(&focus, &["talus", "ground", "debris"])
            .expect("the stages");
        runtime.run_until_idle().expect("the stages run");
        for &chunk in request {
            let field = |stage: &str| runtime.field(stage, chunk).expect(stage);
            let (ground, talus, debris) = (field("ground"), field("talus"), field("debris"));
            for y in 0..SIZE[1] {
                for x in 0..SIZE[0] {
                    let column = (
                        i64::from(chunk.x) * i64::from(SIZE[0]) + i64::from(x),
                        i64::from(chunk.y) * i64::from(SIZE[1]) + i64::from(y),
                    );
                    out.insert(
                        column,
                        [ground.get(x, y), talus.get(x, y), debris.get(x, y)],
                    );
                }
            }
        }
    }
    out
}

#[test]
fn the_weathered_ground_is_the_same_in_any_order() {
    let one_by_one: Vec<Vec<ChunkCoord>> = area().into_iter().map(|c| vec![c]).collect();
    let backwards: Vec<Vec<ChunkCoord>> = area().into_iter().rev().map(|c| vec![c]).collect();

    let at_once = columns_asked(&[area()]);

    assert_eq!(at_once, columns_asked(&one_by_one));
    assert_eq!(at_once, columns_asked(&backwards));
}

#[test]
fn the_weathered_ground_is_its_input_along_every_region_border() {
    let columns = columns_asked(&[area()]);

    let on_border = |c: i64| c.rem_euclid(REGION) == 0 || c.rem_euclid(REGION) == REGION - 1;
    let mut inside_changed = 0;
    for (&(x, y), &[ground, talus, _]) in &columns {
        if on_border(x) || on_border(y) {
            assert_eq!(talus, ground, "column ({x}, {y})");
        } else if talus != ground {
            inside_changed += 1;
        }
    }
    assert!(
        inside_changed > columns.len() / 4,
        "only {inside_changed} columns inside the regions were weathered"
    );
}

#[test]
fn debris_fills_the_hollows_and_the_crests_wear_down() {
    let columns = columns_asked(&[area()]);

    // Away from the borders' fade, a column below all its neighbours' mean receives debris, and
    // one above it sheds it.
    let deep = |x: i64| (8..REGION - 8).contains(&x.rem_euclid(REGION));
    let (mut hollows, mut crests) = (Vec::new(), Vec::new());
    for (&(x, y), &[ground, talus, _]) in &columns {
        let around: Vec<f32> = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            .iter()
            .filter_map(|(dx, dy)| columns.get(&(x + dx, y + dy)).map(|&[g, _, _]| g))
            .collect();
        if !deep(x) || !deep(y) || around.len() < 4 {
            continue;
        }
        let mean = around.iter().sum::<f32>() / 4.0;
        if ground < mean - 0.5 {
            hollows.push(talus - ground);
        } else if ground > mean + 0.5 {
            crests.push(talus - ground);
        }
    }
    let average = |changes: &[f32]| changes.iter().sum::<f32>() / changes.len() as f32;

    assert!(!hollows.is_empty() && !crests.is_empty());
    assert!(
        average(&hollows) > 0.0 && average(&crests) < 0.0,
        "hollows changed by {} on average, crests by {}",
        average(&hollows),
        average(&crests)
    );
}

#[test]
fn the_debris_field_is_how_deep_the_debris_lies() {
    let columns = columns_asked(&[area()]);

    let mut covered = 0;
    for (&(x, y), &[ground, talus, debris]) in &columns {
        assert_eq!(debris, (talus - ground).max(0.0), "column ({x}, {y})");
        covered += usize::from(debris > 0.1);
    }
    assert!(covered > 0, "no debris anywhere");
}

#[test]
fn talus_that_cannot_settle_is_refused() {
    for wrong in [
        "region: 0, slope: 0.6",
        "region: 4, slope: -1.0",
        "region: 4, slope: 0.6, passes: 0",
        "region: 4, slope: 0.6, rate: 1.5",
    ] {
        let text = PACK.replace("region: 4, slope: 0.6, fade: 8", wrong);

        let result = pack(&text);

        assert!(
            matches!(&result, Err(PackError::Invalid { stage, .. }) if stage == "talus"),
            "{wrong}: {result:?}"
        );
    }
}

#[test]
fn a_sample_of_the_weathered_ground_is_what_its_chunk_holds() {
    let columns = columns_asked(&[area()]);
    let sampler = Runtime::new(Arc::new(pack(PACK).expect("a valid pack")), 6, SIZE);

    for (&(x, y), &[_, talus, _]) in &columns {
        let sampled = sampler
            .sample("talus", [x as f32 + 0.5, y as f32 + 0.5])
            .expect("a sample");

        assert_eq!(sampled.to_bits(), talus.to_bits(), "column ({x}, {y})");
    }
}
