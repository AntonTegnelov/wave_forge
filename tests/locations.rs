//! A location table: sites of several kinds placed per region in order of priority, each kind
//! honouring its quota, its distance from its own kind and its conditions, with a log of what was
//! placed and why candidates were refused, the same in any order.

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::stages::{Pack, PackError, Runtime, Site, SiteId};
use wave_forge::{ChunkCoord, FocusPoint};

const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "ground", kind: Field(Mul(Noise(frequency: 0.03, octaves: 2), Constant(20.0)))),
        (name: "side", kind: Rules(rules: [(category: "west", when: [Less(X, Constant(0.0))])], otherwise: "east")),
        (name: "places", kind: Locations(height: "ground", region: 8, kinds: [
            (name: "altar", priority: 10, quota: 3, apart: 24.0, tries: 40),
            (name: "trader", priority: 5, quota: 1, size: 2, tries: 40,
                when: [Greater(Is("side", ["east"]), Constant(0.5))]),
            (name: "cairn", quota: 6, tries: 60, when: [Greater(Input("ground"), Constant(10.0))]),
        ])),
    ],
)"#;

const SIZE: [u32; 2] = [8, 8];
const REGION: i32 = 8;

fn area() -> Vec<ChunkCoord> {
    (-16..16)
        .flat_map(|y| (-16..16).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

/// Every site over the area once, asked for in the order of `requests`, and the runtime.
fn places(requests: &[Vec<ChunkCoord>]) -> (Runtime, Vec<Site>) {
    let mut runtime = Runtime::new(Arc::new(Pack::parse(PACK).expect("a valid pack")), 11, SIZE);
    let mut sites: BTreeMap<SiteId, Site> = BTreeMap::new();
    for request in requests {
        let focus: Vec<FocusPoint> = request.iter().map(|&c| FocusPoint::new(c, 0)).collect();
        runtime
            .request(&focus, &["places", "ground"])
            .expect("stages");
        runtime.run_until_idle().expect("the stages run");
        for &chunk in request {
            for site in runtime.sites("places", chunk).expect("generated") {
                sites.insert(site.id.clone(), site.clone());
            }
        }
    }
    (runtime, sites.into_values().collect())
}

fn kind(site: &Site) -> &str {
    site.kind.as_deref().expect("a location has a kind")
}

fn region(site: &Site) -> (i32, i32) {
    match site.id {
        SiteId::Location { region, .. } => region,
        _ => panic!("a location is named by its region: {site:?}"),
    }
}

/// The centre of a site's footprint, in cells.
fn centre(site: &Site) -> (f32, f32) {
    (
        (site.min.0 + site.max.0) as f32 * 4.0,
        (site.min.1 + site.max.1) as f32 * 4.0,
    )
}

#[test]
fn every_kind_keeps_to_its_quota_in_every_region_and_mostly_meets_it() {
    let (_, sites) = places(&[area()]);

    let mut counts: BTreeMap<((i32, i32), &str), u32> = BTreeMap::new();
    for site in &sites {
        *counts.entry((region(site), kind(site))).or_default() += 1;
    }

    for (&(region, kind), &count) in &counts {
        let quota = match kind {
            "altar" => 3,
            "trader" => 1,
            _ => 6,
        };
        assert!(count <= quota, "{count} {kind}s in region {region:?}");
    }
    let full = counts
        .iter()
        .filter(|&(&(_, k), &n)| k == "altar" && n == 3)
        .count();
    assert!(
        full >= 12,
        "only {full} of 16 regions have all their altars"
    );
}

#[test]
fn no_two_sites_come_within_a_chunk_even_across_regions() {
    let (_, sites) = places(&[area()]);

    for (i, a) in sites.iter().enumerate() {
        for b in &sites[i + 1..] {
            let touching = a.min.0 <= b.max.0
                && b.min.0 <= a.max.0
                && a.min.1 <= b.max.1
                && b.min.1 <= a.max.1;
            assert!(!touching, "{a:?} and {b:?}");
        }
        let inside = |at: i32, low: i32| at > low * REGION && at < (low + 1) * REGION;
        let (rx, ry) = region(a);
        assert!(
            inside(a.min.0, rx) && inside(a.min.1, ry),
            "{a:?} touches its region's edge"
        );
        assert!(
            a.max.0 < (rx + 1) * REGION && a.max.1 < (ry + 1) * REGION,
            "{a:?}"
        );
    }
}

#[test]
fn a_kind_keeps_its_distance_from_its_own_kind_and_meets_its_conditions() {
    let (runtime, sites) = places(&[area()]);

    let ground = |x: f32, y: f32| {
        let (x, y) = (x as i64, y as i64);
        let chunk = ChunkCoord::new(x.div_euclid(8) as i32, y.div_euclid(8) as i32, 0);
        runtime
            .field("ground", chunk)
            .expect("generated")
            .get(x.rem_euclid(8) as u32, y.rem_euclid(8) as u32)
    };
    for (i, a) in sites.iter().enumerate() {
        for b in &sites[i + 1..] {
            if kind(a) == "altar" && kind(b) == "altar" && region(a) == region(b) {
                let (p, q) = (centre(a), centre(b));
                assert!((p.0 - q.0).hypot(p.1 - q.1) >= 24.0, "{a:?} and {b:?}");
            }
        }
        let (x, y) = centre(a);
        match kind(a) {
            "trader" => assert!(x >= 0.0, "a trader in the west: {a:?}"),
            "cairn" => assert!(ground(x, y) > 10.0, "a cairn on low ground: {a:?}"),
            _ => {}
        }
    }
    assert!(sites.iter().any(|site| kind(site) == "trader"));
    assert!(sites.iter().any(|site| kind(site) == "cairn"));
}

#[test]
fn each_region_logs_what_every_kind_placed_and_refused_in_order_of_priority() {
    let (runtime, _) = places(&[area()]);

    let log = runtime
        .location_log("places", ChunkCoord::new(3, -5, 0))
        .expect("a placed region");

    assert_eq!(log.len(), 3);
    assert!(log[0].starts_with("altar: placed "), "{log:?}");
    assert!(log[1].starts_with("trader: placed "), "{log:?}");
    assert!(log[2].starts_with("cairn: placed "), "{log:?}");
    assert!(log.iter().all(|line| line.contains("refused")), "{log:?}");
}

#[test]
fn a_location_table_is_the_same_in_any_order() {
    let one_by_one: Vec<Vec<ChunkCoord>> = area().into_iter().map(|c| vec![c]).collect();
    let backwards: Vec<Vec<ChunkCoord>> = area().into_iter().rev().map(|c| vec![c]).collect();

    let (_, at_once) = places(&[area()]);

    assert_eq!(at_once, places(&one_by_one).1);
    assert_eq!(at_once, places(&backwards).1);
}

#[test]
fn a_table_that_cannot_be_placed_is_refused_by_stage() {
    let with = |kinds: &str| {
        Pack::parse(&format!(
            r#"(version: 1, stages: [
                (name: "ground", kind: Field(Constant(1.0))),
                (name: "places", kind: Locations(height: "ground", region: 4, kinds: [{kinds}])),
            ])"#
        ))
    };

    for kinds in [
        r#"(name: "a", quota: 0)"#,
        r#"(name: "a", quota: 1, size: 3)"#,
        r#"(name: "a", quota: 1, tries: 0)"#,
        r#"(name: "a", quota: 1, apart: -1.0)"#,
        r#"(name: "a", quota: 1), (name: "a", quota: 2)"#,
        r#"(name: "a", quota: 1, when: [Less(Input("nowhere"), Constant(1.0))])"#,
    ] {
        let result = with(kinds);
        assert!(
            matches!(&result, Err(PackError::Invalid { stage, .. } | PackError::UnknownInput { stage, .. }) if stage == "places"),
            "{kinds}: {result:?}"
        );
    }
}

#[test]
fn the_ring_worlds_shrines_stand_in_their_rings_within_their_quotas() {
    let text = std::fs::read_to_string(format!(
        "{}/examples/rings.world.ron",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("the ring world");
    let pack = Arc::new(Pack::parse(&text).expect("a valid pack"));
    let names = pack.kind("biome").expect("a stage").categories();
    let mut runtime = Runtime::new(Arc::clone(&pack), 7, SIZE);
    let chunks: Vec<ChunkCoord> = (-13..13)
        .flat_map(|y| (-13..13).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect();
    runtime
        .request_bound(&["shrines", "biome"])
        .expect("a bounded island");
    runtime.run_until_idle().expect("the stages run");
    let mut sites: BTreeMap<SiteId, Site> = BTreeMap::new();
    // Only the chunks that meet the island's bound are generated.
    for &chunk in &chunks {
        for site in runtime.sites("shrines", chunk).into_iter().flatten() {
            sites.insert(site.id.clone(), site.clone());
        }
    }

    let biome = |(x, y): (f32, f32)| {
        let (x, y) = (x as i64, y as i64);
        let chunk = ChunkCoord::new(x.div_euclid(8) as i32, y.div_euclid(8) as i32, 0);
        let category = runtime.categories("biome", chunk).expect("generated");
        names[usize::from(category.get(x.rem_euclid(8) as u32, y.rem_euclid(8) as u32))]
    };
    let mut counts: BTreeMap<((i32, i32), &str), u32> = BTreeMap::new();
    for site in sites.values() {
        let expected = match kind(site) {
            "woods_shrine" => "woods",
            "peak_shrine" => "peaks",
            _ => "grassland",
        };
        assert_eq!(biome(centre(site)), expected, "{site:?}");
        *counts.entry((region(site), kind(site))).or_default() += 1;
    }
    for (&(region, kind), &count) in &counts {
        let quota = if kind == "woods_shrine" { 2 } else { 1 };
        assert!(count <= quota, "{count} {kind}s in {region:?}");
    }
    for kind in ["woods_shrine", "peak_shrine", "meadow_shrine"] {
        assert!(
            counts.keys().any(|&(_, k)| k == kind),
            "no {kind} on the island: {counts:?}"
        );
    }
}
