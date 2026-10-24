//! A galaxy of sectors, systems and bodies (docs/product/user-stories.md, G6):
//! `tests/fixtures/galaxy.world.ron`. Each table reads only its parent and its budget, authored
//! systems move no generated id, a body's row drives its surface (relief, zones, rocks with their
//! spacing, and an authored outpost on flattened ground), and two runs and two travel orders give the
//! same systems and surfaces.

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::stages::{AUTHORED, Facts, Pack, Row, RowId, Runtime, Table};
use wave_forge::{ChunkCoord, FocusPoint};

const SEED: u64 = 17;
const SIZE: [u32; 2] = [8, 8];
const SURFACE: [&str; 3] = ["ground", "zones", "rocks"];

fn text() -> String {
    std::fs::read_to_string(format!(
        "{}/tests/fixtures/galaxy.world.ron",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("the galaxy's pack")
}

fn pack() -> Arc<Pack> {
    Arc::new(Pack::parse(&text()).expect("a valid pack"))
}

fn facts() -> Facts {
    Facts::new(pack(), SEED).expect("the galaxy's tables")
}

fn table<'f>(facts: &'f Facts, name: &str) -> &'f Table {
    facts.table(name).expect("a table")
}

fn value(table: &Table, row: &Row, column: &str) -> f32 {
    row.values[table.column(column).expect("a column")]
}

fn authored(id: &RowId) -> bool {
    id.0.iter().any(|&part| part >= AUTHORED)
}

/// The 4 by 4 chunks around the outpost.
fn area() -> Vec<ChunkCoord> {
    (0..4)
        .flat_map(|y| (0..4).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

/// A runtime focused on `body`'s row.
fn surface(body: &RowId) -> Runtime {
    let pack = pack();
    let mut runtime = Runtime::new(Arc::clone(&pack), SEED, SIZE);
    runtime
        .set_facts(Facts::new(pack, SEED).expect("the galaxy's tables"))
        .expect("facts of the pack and seed");
    runtime
        .focus("bodies", body.clone())
        .expect("a body of the table");
    runtime
}

/// Every chunk of the area's hash of the surface, the area generated as `requests` ask, one at a
/// time, each chunk hashed as soon as it is generated.
fn hashes(body: &RowId, requests: &[Vec<ChunkCoord>]) -> BTreeMap<ChunkCoord, u64> {
    let mut runtime = surface(body);
    let mut out = BTreeMap::new();
    for request in requests {
        let focus: Vec<FocusPoint> = request.iter().map(|&c| FocusPoint::new(c, 0)).collect();
        runtime.request(&focus, &SURFACE).expect("the surface");
        runtime.run_until_idle().expect("the surface generates");
        for &chunk in request {
            out.insert(
                chunk,
                runtime.chunk_hash(chunk, &SURFACE).expect("every stage"),
            );
        }
    }
    out
}

/// The body whose row has the most of `column` and the one with the least.
fn extremes(column: &str) -> (RowId, RowId) {
    let facts = facts();
    let bodies = table(&facts, "bodies");
    let by = |row: &&Row| value(bodies, row, column);
    let most = bodies
        .rows
        .iter()
        .max_by(|a, b| by(a).total_cmp(&by(b)))
        .expect("bodies");
    let least = bodies
        .rows
        .iter()
        .min_by(|a, b| by(a).total_cmp(&by(b)))
        .expect("bodies");
    (most.id.clone(), least.id.clone())
}

#[test]
fn each_level_reads_only_its_parent_and_shares_its_budget() {
    let facts = facts();
    let (sectors, systems, bodies) = (
        table(&facts, "sectors"),
        table(&facts, "systems"),
        table(&facts, "bodies"),
    );

    for sector in &sectors.rows {
        let shared: f32 = systems
            .rows
            .iter()
            .filter(|row| row.id.0[..1] == sector.id.0[..] && !authored(&row.id))
            .map(|row| value(systems, row, "mass"))
            .sum();
        assert_eq!(shared, value(sectors, sector, "mass"), "{:?}", sector.id);
    }
    for body in &bodies.rows {
        let system = systems
            .row(&RowId(body.id.0[..2].to_vec()))
            .expect("a body's system");
        assert!(value(bodies, body, "size") <= value(systems, system, "radius"));
    }
}

#[test]
fn authored_systems_move_no_generated_id() {
    let start = text()
        .find("            // A home system")
        .expect("authored systems");
    let end = text()[start..].find("])),").expect("their end") + start;
    let without = format!("{}            )),{}", &text()[..start], &text()[end + 4..]);
    let plain =
        Facts::new(Arc::new(Pack::parse(&without).expect("a valid pack")), SEED).expect("tables");
    let facts = facts();

    for name in ["systems", "bodies"] {
        let generated: Vec<&Row> = table(&facts, name)
            .rows
            .iter()
            .filter(|row| !authored(&row.id))
            .collect();
        let before: Vec<&Row> = table(&plain, name).rows.iter().collect();
        assert_eq!(generated, before, "{name}");
        assert!(
            table(&facts, name).rows.iter().any(|row| authored(&row.id)),
            "{name} has authored rows"
        );
    }
}

#[test]
fn a_bodys_row_drives_its_relief_zones_and_rocks() {
    let (rough, smooth) = extremes("roughness");
    let (hot, cold) = extremes("temperature");
    let (wet, dry) = extremes("water");
    let generate = |body: &RowId| {
        let mut runtime = surface(body);
        let focus: Vec<FocusPoint> = area().iter().map(|&c| FocusPoint::new(c, 0)).collect();
        runtime.request(&focus, &SURFACE).expect("the surface");
        runtime.run_until_idle().expect("the surface generates");
        runtime
    };
    let relief = |runtime: &Runtime| {
        let values: Vec<f32> = area()
            .iter()
            .flat_map(|&c| runtime.field("ground", c).expect("ground").values.clone())
            .collect();
        values.iter().copied().fold(f32::MIN, f32::max)
            - values.iter().copied().fold(f32::MAX, f32::min)
    };
    let zones = |runtime: &Runtime| -> Vec<u8> {
        area()
            .iter()
            .flat_map(|&c| {
                runtime
                    .categories("zones", c)
                    .expect("zones")
                    .values
                    .clone()
            })
            .collect()
    };
    let rocks = |runtime: &Runtime| -> Vec<[f32; 3]> {
        area()
            .iter()
            .flat_map(|&c| {
                runtime
                    .points("rocks", c)
                    .expect("rocks")
                    .iter()
                    .map(|p| p.position)
            })
            .collect()
    };
    let pack = pack();
    let names = pack.kind("zones").expect("zones").categories();
    let zone = |name: &str| names.iter().position(|n| *n == name).expect("a zone") as u8;

    assert!(relief(&generate(&rough)) > 2.0 * relief(&generate(&smooth)));
    assert!(zones(&generate(&cold)).iter().all(|&z| z == zone("ice")));
    assert!(zones(&generate(&hot)).iter().all(|&z| z == zone("dunes")));
    let (on_dry, on_wet) = (rocks(&generate(&dry)), rocks(&generate(&wet)));
    assert!(
        on_dry.len() > on_wet.len(),
        "{} rocks dry, {} wet",
        on_dry.len(),
        on_wet.len()
    );
    for (i, a) in on_dry.iter().enumerate() {
        for b in &on_dry[i + 1..] {
            assert!(
                libm::hypotf(a[0] - b[0], a[1] - b[1]) >= 3.0,
                "rocks at {a:?} and {b:?}"
            );
        }
    }
}

#[test]
fn the_authored_outpost_stands_on_flattened_ground() {
    let (rough, _) = extremes("roughness");
    let mut runtime = surface(&rough);
    let focus: Vec<FocusPoint> = area().iter().map(|&c| FocusPoint::new(c, 0)).collect();
    runtime
        .request(&focus, &["ground", "outposts"])
        .expect("the surface");
    runtime.run_until_idle().expect("the surface generates");

    let outpost = ChunkCoord::new(2, 2, 0);
    let site = &runtime
        .sites("outposts", outpost)
        .expect("the outpost's chunk")[0];
    let ground = runtime.field("ground", outpost).expect("ground");

    assert_eq!(
        site.id,
        wave_forge::stages::SiteId::Row(RowId(vec![AUTHORED]))
    );
    assert!(
        ground.values.iter().all(|&h| h == site.height),
        "{:?}",
        ground.values
    );
}

#[test]
fn two_runs_and_two_travel_orders_give_the_same_systems_and_surfaces() {
    let (body, _) = extremes("roughness");
    let one_by_one: Vec<Vec<ChunkCoord>> = area().into_iter().map(|c| vec![c]).collect();
    let mut backwards = one_by_one.clone();
    backwards.reverse();

    assert_eq!(table(&facts(), "bodies"), table(&facts(), "bodies"));
    let whole = hashes(&body, &[area()]);
    assert_eq!(hashes(&body, &one_by_one), whole);
    assert_eq!(hashes(&body, &backwards), whole);
}
