//! Scatter stages: points that keep their distance across chunk seams, pass their own tests, and
//! come out the same whatever order chunks are asked for in.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use wave_forge::stages::{Pack, Point, Runtime};
use wave_forge::{ChunkCoord, FocusPoint};

const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "height", kind: Field(Mul(Noise(frequency: 0.04, octaves: 3), Constant(20.0)))),
        (name: "towns", kind: Sites(height: "height", region: 5, size: (1, 2), chance: 0.7)),
        (name: "level", kind: Flatten(height: "height", sites: "towns", blend: 4)),
        (name: "trees", kind: Scatter(kind: "tree", height: "level", spacing: 3, chance: 0.8,
            between: Some((4.0, 18.0)), max_slope: Some(1.5), avoid: Some(("towns", 3)), apart: 4)),
    ],
)"#;

const SIZE: [u32; 2] = [8, 8];
const APART: f32 = 4.0;
const MARGIN: f32 = 3.0;

fn area() -> Vec<ChunkCoord> {
    (0..12)
        .flat_map(|y| (0..12).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

/// The runtime after `order`, and every point it placed, by chunk.
fn scatter(order: &[Vec<ChunkCoord>]) -> (Runtime, BTreeMap<ChunkCoord, Vec<Point>>) {
    let pack = Arc::new(Pack::parse(PACK).expect("a valid pack"));
    let mut runtime = Runtime::new(pack, 9, SIZE);
    let mut points = BTreeMap::new();
    for request in order {
        let focus: Vec<FocusPoint> = request.iter().map(|&c| FocusPoint::new(c, 0)).collect();
        runtime.request(&focus, &["trees"]).expect("a stage");
        runtime.run_until_idle().expect("the stages run");
        for &chunk in request {
            points.insert(
                chunk,
                runtime.points("trees", chunk).expect("generated").to_vec(),
            );
        }
    }
    (runtime, points)
}

fn all(points: &BTreeMap<ChunkCoord, Vec<Point>>) -> Vec<&Point> {
    points.values().flatten().collect()
}

#[test]
fn points_come_out_the_same_in_any_order() {
    let (_, at_once) = scatter(&[area()]);
    let (_, one_by_one) = scatter(
        &area()
            .into_iter()
            .rev()
            .map(|c| vec![c])
            .collect::<Vec<_>>(),
    );

    assert!(all(&at_once).len() > 100, "{} points", all(&at_once).len());
    assert_eq!(at_once, one_by_one);
}

#[test]
fn no_two_points_are_closer_than_apart_across_seams_too() {
    let (_, points) = scatter(&[area()]);
    let points = all(&points);

    for (i, a) in points.iter().enumerate() {
        for b in &points[i + 1..] {
            let distance = ((a.position[0] - b.position[0]).powi(2)
                + (a.position[1] - b.position[1]).powi(2))
            .sqrt();
            assert!(distance >= APART, "{a:?} and {b:?} are {distance} apart");
        }
    }
}

#[test]
fn every_point_passes_its_tests_and_stands_on_the_ground() {
    let (runtime, points) = scatter(&[area()]);
    let level = |x: i64, y: i64| {
        let chunk = ChunkCoord::new(x.div_euclid(8) as i32, y.div_euclid(8) as i32, 0);
        runtime
            .field("level", chunk)
            .expect("generated")
            .get(x.rem_euclid(8) as u32, y.rem_euclid(8) as u32)
    };
    let sites: BTreeMap<wave_forge::stages::SiteId, wave_forge::stages::Site> = area()
        .into_iter()
        .flat_map(|chunk| runtime.sites("towns", chunk).expect("generated").to_vec())
        .map(|site| (site.id.clone(), site))
        .collect();

    for (&chunk, chunk_points) in &points {
        for point in chunk_points {
            let (x, y) = (
                point.position[0].floor() as i64,
                point.position[1].floor() as i64,
            );
            let edge = [0, 12 * 8 - 1];
            if edge.contains(&x) || edge.contains(&y) {
                continue;
            }
            assert_eq!(point.position[2], level(x, y), "{point:?}");
            assert!((4.0..=18.0).contains(&point.position[2]), "{point:?}");
            let slope = ((level(x + 1, y) - level(x - 1, y)).abs() / 2.0)
                .max((level(x, y + 1) - level(x, y - 1)).abs() / 2.0);
            assert!(slope <= 1.5, "{point:?} on a slope of {slope}");
            for site in sites.values() {
                assert!(
                    site.distance(x, y, SIZE) >= MARGIN,
                    "{point:?} near {site:?}"
                );
            }
            assert_eq!(point.id.chunk, chunk);
            assert_eq!(
                point.id.cell() as i64,
                (y - i64::from(chunk.y) * 8) * 8 + (x - i64::from(chunk.x) * 8)
            );
        }
    }
    let ids: BTreeSet<_> = all(&points).iter().map(|point| point.id).collect();
    assert_eq!(ids.len(), all(&points).len(), "every id is different");
}
