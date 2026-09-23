//! Sites from a table's rows: a history's villages, where the game put them and as large as it made
//! them, with towns whose rule set follows a column of their row.
//!
//! The town solver here records what it is asked and returns empty towns: which rule set a town
//! asks for is the contract under test, and the towns themselves are checked in `towns.rs`.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use wave_forge::stages::{
    Facts, GivenRow, Pack, PackError, RowId, Runtime, Site, SiteId, StageError, Value,
};
use wave_forge::towns::{Town, TownError, TownRequest, TownSolver};
use wave_forge::{ChunkCoord, ChunkShape, FocusPoint};

const PACK: &str = r#"(
    version: 1,
    tables: [(name: "villages", kind: Given(columns: [
        ("x", Number), ("y", Number), ("size", Number), ("fate", Names(["standing", "burned"])),
    ]))],
    stages: [
        (name: "height", kind: Field(Mul(Noise(frequency: 0.05, octaves: 2), Constant(10.0)))),
        (name: "villages", kind: TableSites(table: "villages", height: "height", at: ("x", "y"), size: "size", max_size: 3)),
        (name: "level", kind: Flatten(height: "height", sites: "villages", blend: 3)),
        (name: "buildings", kind: Solve(sites: "villages", rules: "blocks", by: Some(("fate", [("burned", "ruins")])))),
    ],
)"#;

const CHUNK: ChunkShape = ChunkShape { x: 4, y: 4, z: 3 };

/// The rule set and the size of every town asked for, in order.
type Asked = Vec<(String, (u32, u32))>;

/// A town solver that records what it is asked.
#[derive(Clone, Default)]
struct Recorder(Arc<Mutex<Asked>>);

impl TownSolver for Recorder {
    fn chunk_shape(&self) -> ChunkShape {
        CHUNK
    }

    fn solve(&mut self, request: &TownRequest<'_>) -> Result<Town, TownError> {
        self.0
            .lock()
            .expect("one test thread")
            .push((request.rules.to_owned(), request.size));
        let chunk: Arc<[u16]> = Arc::from(vec![0; (CHUNK.x * CHUNK.y * CHUNK.z) as usize]);
        Ok(Town {
            size: request.size,
            chunks: vec![chunk; (request.size.0 * request.size.1) as usize],
        })
    }
}

fn village(id: u64, at: (f32, f32), size: f32, fate: &str) -> GivenRow {
    GivenRow {
        id,
        values: BTreeMap::from([
            ("x".to_owned(), Value::Number(at.0)),
            ("y".to_owned(), Value::Number(at.1)),
            ("size".to_owned(), Value::Number(size)),
            ("fate".to_owned(), Value::Name(fate.to_owned())),
        ]),
    }
}

/// A runtime with the villages given, and what its town solver is asked.
fn runtime(villages: Vec<GivenRow>) -> Result<(Runtime, Recorder), StageError> {
    let pack = Arc::new(Pack::parse(PACK).expect("a valid pack"));
    let recorder = Recorder::default();
    let mut runtime = Runtime::new(Arc::clone(&pack), 3, [CHUNK.x, CHUNK.y])
        .with_towns(Box::new(recorder.clone()))
        .expect("matching chunks");
    let mut facts = Facts::new(pack, 3)?;
    facts.give("villages", villages)?;
    runtime.set_facts(facts)?;
    Ok((runtime, recorder))
}

/// Every site of the area from (-8, -8) to (8, 8) chunks, once each, after generating `stages`.
fn generate(runtime: &mut Runtime, stages: &[&str]) -> Vec<Site> {
    let focus: Vec<FocusPoint> = (-8..=8)
        .flat_map(|y| (-8..=8).map(move |x| FocusPoint::new(ChunkCoord::new(x, y, 0), 0)))
        .collect();
    runtime.request(&focus, stages).expect("stages");
    runtime.run_until_idle().expect("the stages run");
    let mut sites: BTreeMap<SiteId, Site> = BTreeMap::new();
    for focus in &focus {
        for site in runtime.sites("villages", focus.chunk).expect("generated") {
            sites.insert(site.id.clone(), site.clone());
        }
    }
    sites.into_values().collect()
}

#[test]
fn a_site_stands_around_its_rows_position_as_large_as_its_row_says() {
    let (mut runtime, _) = runtime(vec![
        village(1, (10.0, 10.0), 3.0, "standing"),
        village(2, (-20.0, 6.5), 1.0, "standing"),
    ])
    .expect("villages");

    let sites = generate(&mut runtime, &["level"]);

    let footprints: Vec<_> = sites
        .iter()
        .map(|site| (site.id.clone(), site.min, site.max))
        .collect();
    assert_eq!(
        footprints,
        vec![
            (SiteId::Row(RowId(vec![1])), (1, 1), (4, 4)),
            (SiteId::Row(RowId(vec![2])), (-5, 1), (-4, 2)),
        ]
    );
    for site in &sites {
        let level = runtime
            .field("level", ChunkCoord::new(site.min.0, site.min.1, 0))
            .expect("generated");
        assert!(
            level.values.iter().all(|&value| value == site.height),
            "{site:?}"
        );
    }
}

#[test]
fn rows_whose_sites_would_crowd_or_break_the_size_are_refused_by_row() {
    let refusals = [
        vec![
            village(1, (10.0, 10.0), 3.0, "standing"),
            village(2, (18.0, 10.0), 1.0, "standing"),
        ],
        vec![village(1, (10.0, 10.0), 4.0, "standing")],
        vec![village(1, (10.0, 10.0), 1.5, "standing")],
        vec![village(1, (f32::INFINITY, 10.0), 1.0, "standing")],
    ];

    for villages in refusals {
        let result = runtime(villages.clone()).map(|_| ());
        assert!(
            matches!(&result, Err(StageError::Table { table, .. }) if table == "villages"),
            "{villages:?}: {result:?}"
        );
    }
}

#[test]
fn a_town_takes_its_rule_set_from_its_rows_names() {
    let (mut runtime, recorder) = runtime(vec![
        village(1, (10.0, 10.0), 3.0, "standing"),
        village(2, (-20.0, 6.5), 1.0, "burned"),
    ])
    .expect("villages");

    generate(&mut runtime, &["buildings"]);

    let mut asked = recorder.0.lock().expect("one test thread").clone();
    asked.sort();
    assert_eq!(
        asked,
        vec![("blocks".to_owned(), (3, 3)), ("ruins".to_owned(), (1, 1))]
    );
}

#[test]
fn burning_a_village_solves_its_town_again_with_the_other_rule_set() {
    let pack = Arc::new(Pack::parse(PACK).expect("a valid pack"));
    let recorder = Recorder::default();
    let mut runtime = Runtime::new(Arc::clone(&pack), 3, [CHUNK.x, CHUNK.y])
        .with_towns(Box::new(recorder.clone()))
        .expect("matching chunks");
    let mut facts = Facts::new(pack, 3).expect("facts");
    facts
        .give("villages", vec![village(1, (10.0, 10.0), 3.0, "standing")])
        .expect("villages");
    runtime.set_facts(facts.clone()).expect("facts");
    generate(&mut runtime, &["buildings"]);

    facts
        .give("villages", vec![village(1, (10.0, 10.0), 3.0, "burned")])
        .expect("villages");
    let dropped = runtime.set_facts(facts).expect("facts");
    generate(&mut runtime, &["buildings"]);

    assert!(dropped.iter().any(|(stage, _)| stage == "buildings"));
    assert!(dropped.iter().all(|(stage, _)| stage != "height"));
    assert_eq!(
        *recorder.0.lock().expect("one test thread"),
        vec![("blocks".to_owned(), (3, 3)), ("ruins".to_owned(), (3, 3))]
    );
}

#[test]
fn a_solve_choosing_by_a_column_needs_a_tables_names() {
    let table = r#"tables: [(name: "villages", kind: Given(columns: [("x", Number), ("y", Number), ("size", Number), ("fate", Names(["standing", "burned"]))]))]"#;
    let stages = |solve: &str| {
        format!(
            r#"(version: 1, {table}, stages: [
                (name: "height", kind: Field(Constant(1.0))),
                (name: "hashed", kind: Sites(height: "height", region: 5, size: (1, 2), chance: 1.0)),
                (name: "villages", kind: TableSites(table: "villages", height: "height", at: ("x", "y"), size: "size", max_size: 3)),
                (name: "buildings", kind: {solve}),
            ])"#
        )
    };

    let refused = [
        stages(r#"Solve(sites: "hashed", rules: "blocks", by: Some(("fate", [])))"#),
        stages(
            r#"Solve(sites: "villages", rules: "blocks", by: Some(("fate", [("flooded", "ruins")])))"#,
        ),
        stages(r#"Solve(sites: "villages", rules: "blocks", by: Some(("size", [])))"#),
    ];

    for text in refused {
        let result = Pack::parse(&text);
        assert!(
            matches!(&result, Err(PackError::Invalid { stage, .. }) if stage == "buildings"),
            "{result:?}"
        );
    }
}
