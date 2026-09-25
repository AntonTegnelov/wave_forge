//! Place names: a location table's site is named by a translation key made from its kind, with
//! arguments that tell sites of a kind apart, never by a finished string.

use std::sync::Arc;
use wave_forge::stages::{Pack, PackError, Runtime, Site};
use wave_forge::{ChunkCoord, FocusPoint};

const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "ground", kind: Field(Constant(4.0))),
        (name: "places", kind: Locations(height: "ground", region: 4, kinds: [
            (name: "stone_circle", quota: 2, tries: 20),
        ])),
        (name: "towns", kind: Sites(height: "ground", region: 4, size: (1, 1), chance: 1.0)),
    ],
)"#;

fn sites(stage: &str) -> Vec<Site> {
    let mut runtime = Runtime::new(
        Arc::new(Pack::parse(PACK).expect("a valid pack")),
        5,
        [8, 8],
    );
    let area: Vec<ChunkCoord> = (0..8)
        .flat_map(|y| (0..8).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect();
    let focus: Vec<FocusPoint> = area
        .iter()
        .map(|&chunk| FocusPoint::new(chunk, 0))
        .collect();
    runtime.request(&focus, &[stage]).expect("stages");
    runtime.run_until_idle().expect("the stages run");
    area.iter()
        .flat_map(|&chunk| runtime.sites(stage, chunk).expect("generated").to_vec())
        .collect()
}

#[test]
fn a_location_is_named_by_its_kinds_key_with_its_region_and_index() {
    let sites = sites("places");

    assert!(!sites.is_empty());
    for site in sites {
        let name = site.name().expect("a location has a name");
        let wave_forge::stages::SiteId::Location { region, index } = site.id else {
            panic!("a location's id names its region: {site:?}");
        };
        assert_eq!(name.key, "wf-place-stone-circle");
        assert_eq!(
            name.args,
            [
                ("region_x", i64::from(region.0)),
                ("region_y", i64::from(region.1)),
                ("index", i64::from(index)),
            ]
        );
    }
}

#[test]
fn a_sites_stages_site_has_no_name_of_its_own() {
    let sites = sites("towns");

    assert!(!sites.is_empty());
    assert!(sites.iter().all(|site| site.name().is_none()));
}

#[test]
fn a_kind_whose_name_cannot_be_a_key_is_refused() {
    for name in ["Stone circle", "stone.circle", "1st", "circle!"] {
        let pack = PACK.replace(r#"(name: "stone_circle""#, &format!("(name: {name:?}"));

        let error = Pack::parse(&pack).expect_err(name);

        assert!(
            matches!(&error, PackError::Invalid { .. }) && error.to_string().contains("wf-place"),
            "{name}: {error}"
        );
    }
}
