//! A Droplets stage: a height field worn by droplets of water per region, the same in any order,
//! meeting its input along every region border, and deeper where water gathers.

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::stages::{Pack, PackError, Runtime};
use wave_forge::{ChunkCoord, FocusPoint};

const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "ground", kind: Field(Mul(Noise(frequency: 0.03, octaves: 4), Constant(30.0)))),
        (name: "worn", kind: Droplets(height: "ground", region: 4, fade: 8)),
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

/// Every column of the area, its ground and its worn ground, each chunk read once its request has
/// run, in the order of `requests`.
fn columns_asked(requests: &[Vec<ChunkCoord>]) -> BTreeMap<(i64, i64), (f32, f32)> {
    let mut runtime = Runtime::new(Arc::new(pack(PACK).expect("a valid pack")), 6, SIZE);
    let mut out = BTreeMap::new();
    for request in requests {
        let focus: Vec<FocusPoint> = request.iter().map(|&c| FocusPoint::new(c, 0)).collect();
        runtime
            .request(&focus, &["worn", "ground"])
            .expect("the stages");
        runtime.run_until_idle().expect("the stages run");
        for &chunk in request {
            let ground = runtime.field("ground", chunk).expect("ground");
            let worn = runtime.field("worn", chunk).expect("worn");
            for y in 0..SIZE[1] {
                for x in 0..SIZE[0] {
                    let column = (
                        i64::from(chunk.x) * i64::from(SIZE[0]) + i64::from(x),
                        i64::from(chunk.y) * i64::from(SIZE[1]) + i64::from(y),
                    );
                    out.insert(column, (ground.get(x, y), worn.get(x, y)));
                }
            }
        }
    }
    out
}

#[test]
fn the_worn_ground_is_the_same_in_any_order() {
    let one_by_one: Vec<Vec<ChunkCoord>> = area().into_iter().map(|c| vec![c]).collect();
    let backwards: Vec<Vec<ChunkCoord>> = area().into_iter().rev().map(|c| vec![c]).collect();

    let at_once = columns_asked(&[area()]);

    assert_eq!(at_once, columns_asked(&one_by_one));
    assert_eq!(at_once, columns_asked(&backwards));
}

#[test]
fn the_worn_ground_is_its_input_along_every_region_border() {
    let columns = columns_asked(&[area()]);

    // The first and last column of every region, across and along.
    let on_border = |c: i64| c.rem_euclid(REGION) == 0 || c.rem_euclid(REGION) == REGION - 1;
    let mut inside_changed = 0;
    for (&(x, y), &(ground, worn)) in &columns {
        if on_border(x) || on_border(y) {
            assert_eq!(worn, ground, "column ({x}, {y})");
        } else if worn != ground {
            inside_changed += 1;
        }
    }
    assert!(
        inside_changed > columns.len() / 4,
        "only {inside_changed} columns inside the regions were worn"
    );
}

#[test]
fn droplets_cut_the_hollows_water_runs_down_and_leave_the_crests() {
    let columns = columns_asked(&[area()]);

    // Away from the borders' fade, a column lower than all its neighbours' mean gathers water.
    let deep = |x: i64| (8..REGION - 8).contains(&x.rem_euclid(REGION));
    let (mut hollows, mut crests) = (Vec::new(), Vec::new());
    for (&(x, y), &(ground, worn)) in &columns {
        let around: Vec<f32> = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            .iter()
            .filter_map(|(dx, dy)| columns.get(&(x + dx, y + dy)).map(|&(g, _)| g))
            .collect();
        if !deep(x) || !deep(y) || around.len() < 4 {
            continue;
        }
        let mean = around.iter().sum::<f32>() / 4.0;
        if ground < mean - 0.1 {
            hollows.push(worn - ground);
        } else if ground > mean + 0.1 {
            crests.push(worn - ground);
        }
    }
    let average = |changes: &[f32]| changes.iter().sum::<f32>() / changes.len() as f32;

    assert!(!hollows.is_empty() && !crests.is_empty());
    assert!(
        average(&hollows) < average(&crests),
        "hollows changed by {} on average, crests by {}",
        average(&hollows),
        average(&crests)
    );
}

#[test]
fn droplets_that_cannot_run_are_refused() {
    for wrong in [
        "region: 0",
        "region: 4, droplets: 0.0",
        "region: 4, lifetime: 0",
        "region: 4, radius: 0",
        "region: 4, capacity: 0.0",
        "region: 4, erosion: 1.5",
        "region: 4, deposition: -0.1",
    ] {
        let text = PACK.replace("region: 4, fade: 8", wrong);

        let result = pack(&text);

        assert!(
            matches!(&result, Err(PackError::Invalid { stage, .. }) if stage == "worn"),
            "{wrong}: {result:?}"
        );
    }
}
