//! Locating sites: the nearest site of a Sites stage from any position, found without generating
//! the chunks between, the same site generation would place.

use std::sync::Arc;
use wave_forge::stages::{Pack, Runtime, Site, StageError};
use wave_forge::{ChunkCoord, FocusPoint};

const SIZE: [u32; 2] = [8, 8];

const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "height", kind: Field(Mul(Noise(frequency: 0.03, octaves: 2), Constant(12.0)))),
        (name: "outposts", kind: Sites(height: "height", region: 4, size: (1, 2), chance: 0.4)),
    ],
)"#;

fn runtime() -> Runtime {
    Runtime::new(Arc::new(Pack::parse(PACK).expect("a valid pack")), 21, SIZE)
}

/// How far `at` is from a site's footprint, in cells.
fn distance(site: &Site, at: [f32; 2]) -> f32 {
    let away =
        |low: i32, high: i32, x: f32| (low as f32 * 8.0 - x).max(x - high as f32 * 8.0).max(0.0);
    away(site.min.0, site.max.0, at[0]).hypot(away(site.min.1, site.max.1, at[1]))
}

#[test]
fn the_located_site_is_the_nearest_of_those_generation_places() {
    let area: Vec<ChunkCoord> = (-20..20)
        .flat_map(|y| (-20..20).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect();
    let mut generated = runtime();
    let focus: Vec<FocusPoint> = area.iter().map(|&c| FocusPoint::new(c, 0)).collect();
    generated.request(&focus, &["outposts"]).expect("a stage");
    generated.run_until_idle().expect("the stages run");
    let mut placed: Vec<Site> = area
        .iter()
        .flat_map(|&chunk| {
            generated
                .sites("outposts", chunk)
                .expect("generated")
                .to_vec()
        })
        .collect();
    placed.sort_by(|a, b| a.id.cmp(&b.id));
    placed.dedup_by(|a, b| a.id == b.id);
    let fresh = runtime();

    let points = [
        [0.0, 0.0],
        [37.5, -41.25],
        [-60.0, 34.0],
        [50.3, 50.7],
        [-12.0, -60.0],
    ];
    for at in points {
        let located = fresh.locate("outposts", at, 6).expect("a Sites stage");

        let nearest = placed
            .iter()
            .map(|site| distance(site, at))
            .fold(f32::INFINITY, f32::min);
        let located = located.expect("a site within six regions");
        assert!(
            placed.contains(&located),
            "{located:?} from {at:?} is not a site generation placed"
        );
        assert_eq!(distance(&located, at), nearest, "from {at:?}");
    }
}

#[test]
fn locating_generates_no_chunk() {
    let runtime = runtime();

    let located = runtime.locate("outposts", [500.0, -500.0], 8);

    assert!(matches!(located, Ok(Some(_))), "{located:?}");
    assert!(runtime.field("height", ChunkCoord::new(0, 0, 0)).is_none());
}

#[test]
fn only_a_sites_stage_can_be_located() {
    let runtime = runtime();

    let result = runtime.locate("height", [0.0, 0.0], 2);

    assert!(
        matches!(&result, Err(StageError::NotLocated(stage)) if stage == "height"),
        "{result:?}"
    );
}
