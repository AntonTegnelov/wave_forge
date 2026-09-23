//! Tables of facts: rows the game gives, rows generated from a parent's, and stages reading the
//! row a runtime is focused on.
//!
//! The pack below has a generated hierarchy (sectors, each with systems that share its mass) and
//! a given table of villages with houses generated under each, and two stages: one reads the
//! focused system's column, the other reads nothing but noise.

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::stages::{
    Facts, GivenRow, Pack, PackError, RowId, Runtime, StageError, Table, Value,
};
use wave_forge::{ChunkCoord, FocusPoint};

const PACK: &str = r#"(
    version: 1,
    tables: [
        (name: "sectors", kind: Generated(count: Constant(24.0), columns: [
            ("mass", Floor(Random(0.0, 100000.0))),
        ])),
        (name: "systems", kind: Generated(parent: Some("sectors"), count: Floor(Random(1.0, 12.0)),
            columns: [
                ("mass", Share("mass")),
                ("order", Index),
                ("of", Count),
                ("radius", Random(4.0, 40.0)),
            ])),
        (name: "villages", kind: Given(columns: [
            ("population", Number),
            ("culture", Names(["river", "hill"])),
        ])),
        (name: "houses", kind: Generated(parent: Some("villages"),
            count: Floor(Mul(Parent("population"), Constant(0.1))),
            columns: [("floors", Floor(Random(1.0, 4.0)))])),
    ],
    stages: [
        (name: "terrain", kind: Field(Mul(Row("systems", "radius"), Noise(frequency: 0.1, octaves: 2)))),
        (name: "hills", kind: Field(Noise(frequency: 0.05, octaves: 3))),
        (name: "wealth", kind: Field(Add(Input("hills"), Row("villages", "population")))),
    ],
)"#;

fn pack() -> Arc<Pack> {
    Arc::new(Pack::parse(PACK).expect("a valid pack"))
}

fn village(id: u64, population: f32, culture: &str) -> GivenRow {
    GivenRow {
        id,
        values: BTreeMap::from([
            ("population".to_owned(), Value::Number(population)),
            ("culture".to_owned(), Value::Name(culture.to_owned())),
        ]),
    }
}

fn table<'f>(facts: &'f Facts, name: &str) -> &'f Table {
    facts.table(name).expect("a table")
}

#[test]
fn a_share_of_a_parent_budget_adds_up_to_it_exactly() {
    let facts = Facts::new(pack(), 11).expect("facts");
    let (sectors, systems) = (table(&facts, "sectors"), table(&facts, "systems"));

    let mass = systems.column("mass").expect("a column");
    for sector in &sectors.rows {
        let parts: Vec<f32> = systems
            .rows
            .iter()
            .filter(|row| row.id.0[..1] == sector.id.0[..])
            .map(|row| row.values[mass])
            .collect();
        assert!(!parts.is_empty());
        assert!(parts.iter().all(|part| part.fract() == 0.0 && *part >= 0.0));
        assert_eq!(
            parts.iter().sum::<f32>(),
            sector.values[0],
            "{:?}",
            sector.id
        );
    }
}

#[test]
fn a_generated_row_is_its_parents_id_and_its_place_among_its_siblings() {
    let facts = Facts::new(pack(), 11).expect("facts");
    let systems = table(&facts, "systems");
    let (order, of) = (
        systems.column("order").expect("a column"),
        systems.column("of").expect("a column"),
    );

    for row in &systems.rows {
        let siblings = systems
            .rows
            .iter()
            .filter(|other| other.id.0[0] == row.id.0[0])
            .count();
        assert_eq!(row.id.0.len(), 2);
        assert_eq!(row.id.0[1] as f32, row.values[order]);
        assert_eq!(siblings as f32, row.values[of]);
    }
}

#[test]
fn the_same_seed_gives_the_same_tables_and_another_seed_others() {
    let first = Facts::new(pack(), 11).expect("facts");
    let again = Facts::new(pack(), 11).expect("facts");
    let other = Facts::new(pack(), 12).expect("facts");

    assert_eq!(table(&first, "systems"), table(&again, "systems"));
    assert_ne!(table(&first, "systems"), table(&other, "systems"));
}

#[test]
fn adding_a_given_row_never_changes_another_rows_children() {
    let mut facts = Facts::new(pack(), 3).expect("facts");
    facts
        .give(
            "villages",
            vec![village(40, 30.0, "river"), village(7, 55.0, "hill")],
        )
        .expect("villages");
    let before = table(&facts, "houses").clone();

    facts
        .give(
            "villages",
            vec![
                village(40, 30.0, "river"),
                village(7, 55.0, "hill"),
                village(12, 90.0, "hill"),
            ],
        )
        .expect("villages");

    let after = table(&facts, "houses");
    let kept: Vec<_> = after
        .rows
        .iter()
        .filter(|row| row.id.0[0] != 12)
        .cloned()
        .collect();
    assert_eq!(kept, before.rows);
    assert_eq!(after.rows.len(), before.rows.len() + 9);
}

#[test]
fn a_given_row_is_checked_against_its_columns_and_nothing_changes_on_an_error() {
    let mut facts = Facts::new(pack(), 3).expect("facts");
    facts
        .give("villages", vec![village(1, 10.0, "river")])
        .expect("villages");
    let mut missing = village(2, 10.0, "hill");
    missing.values.remove("culture");

    let refusals = [
        facts
            .clone()
            .give("villages", vec![village(2, 10.0, "sea")]),
        facts.clone().give("villages", vec![missing]),
        facts.clone().give(
            "villages",
            vec![village(2, 1.0, "hill"), village(2, 3.0, "river")],
        ),
        facts
            .clone()
            .give("villages", vec![village(2, f32::NAN, "hill")]),
        facts.clone().give("systems", Vec::new()),
    ];
    let unknown = facts.give("towns", Vec::new());

    for refusal in refusals {
        assert!(
            matches!(&refusal, Err(StageError::Table { table, .. }) if table == "villages" || table == "systems"),
            "{refusal:?}"
        );
    }
    assert_eq!(unknown, Err(StageError::UnknownTable("towns".to_owned())));
    assert_eq!(table(&facts, "villages").rows.len(), 1);
}

/// Every value of `stage` in the chunks around the origin.
fn generated(runtime: &mut Runtime, stage: &str) -> Vec<f32> {
    runtime
        .request(&[FocusPoint::new(ChunkCoord::new(0, 0, 0), 1)], &[stage])
        .expect("a stage");
    runtime.run_until_idle().expect("the stages run");
    (-1..=1)
        .flat_map(|y| (-1..=1).map(move |x| ChunkCoord::new(x, y, 0)))
        .flat_map(|chunk| {
            runtime
                .field(stage, chunk)
                .expect("generated")
                .values
                .clone()
        })
        .collect()
}

#[test]
fn a_stage_reads_the_focused_rows_column_and_a_new_focus_regenerates_it() {
    let pack = pack();
    let facts = Facts::new(Arc::clone(&pack), 5).expect("facts");
    let first = RowId(vec![0, 0]);
    let second = RowId(vec![3, 0]);
    let fresh = |row: &RowId| {
        let mut runtime = Runtime::new(Arc::clone(&pack), 5, [8, 8]);
        runtime.set_facts(facts.clone()).expect("facts");
        runtime.focus("systems", row.clone()).expect("a row");
        generated(&mut runtime, "terrain")
    };
    let mut runtime = Runtime::new(Arc::clone(&pack), 5, [8, 8]);
    runtime.set_facts(facts.clone()).expect("facts");
    runtime.focus("systems", first.clone()).expect("a row");
    let on_first = generated(&mut runtime, "terrain");

    let dropped = runtime.focus("systems", second.clone()).expect("a row");
    let on_second = generated(&mut runtime, "terrain");

    assert_eq!(dropped.len(), 9);
    assert_eq!(on_first, fresh(&first));
    assert_eq!(on_second, fresh(&second));
    assert_ne!(on_first, on_second);
}

#[test]
fn new_facts_drop_only_the_stages_that_read_a_changed_table() {
    let pack = pack();
    let mut facts = Facts::new(Arc::clone(&pack), 5).expect("facts");
    facts
        .give("villages", vec![village(1, 20.0, "river")])
        .expect("villages");
    let mut runtime = Runtime::new(Arc::clone(&pack), 5, [8, 8]);
    runtime.set_facts(facts.clone()).expect("facts");
    runtime.focus("villages", RowId(vec![1])).expect("a row");
    let before = generated(&mut runtime, "wealth");

    facts
        .give("villages", vec![village(1, 70.0, "river")])
        .expect("villages");
    let dropped = runtime.set_facts(facts).expect("facts");
    let after = generated(&mut runtime, "wealth");

    assert!(
        dropped.iter().all(|(stage, _)| stage == "wealth"),
        "{dropped:?}"
    );
    assert_eq!(dropped.len(), 9);
    for (low, high) in before.iter().zip(&after) {
        assert!((high - low - 50.0).abs() < 1e-3, "{low} then {high}");
    }
}

#[test]
fn a_stage_reading_a_row_without_a_focus_names_the_table() {
    let pack = pack();
    let mut runtime = Runtime::new(Arc::clone(&pack), 5, [8, 8]);
    runtime
        .set_facts(Facts::new(Arc::clone(&pack), 5).expect("facts"))
        .expect("facts");
    runtime
        .request(
            &[FocusPoint::new(ChunkCoord::new(0, 0, 0), 0)],
            &["terrain"],
        )
        .expect("a stage");

    let result = runtime.run_until_idle();

    assert_eq!(result, Err(StageError::NoFocus("systems".to_owned())));
}

#[test]
fn facts_of_another_seed_or_a_row_that_is_not_there_are_refused() {
    let pack = pack();
    let mut runtime = Runtime::new(Arc::clone(&pack), 5, [8, 8]);

    let before = runtime.focus("systems", RowId(vec![0, 0]));
    let other = runtime.set_facts(Facts::new(Arc::clone(&pack), 6).expect("facts"));
    runtime
        .set_facts(Facts::new(Arc::clone(&pack), 5).expect("facts"))
        .expect("facts");
    let missing = runtime.focus("systems", RowId(vec![0, 999]));

    assert_eq!(before, Err(StageError::NoFacts));
    assert_eq!(other, Err(StageError::OtherFacts));
    assert!(
        matches!(missing, Err(StageError::Table { .. })),
        "{missing:?}"
    );
}

fn refusal(tables: &str, stages: &str) -> PackError {
    Pack::parse(&format!(
        "(version: 1, tables: [{tables}], stages: [{stages}])"
    ))
    .expect_err("a refused pack")
}

#[test]
fn an_expression_reading_what_its_place_cannot_is_refused_by_name() {
    let root = r#"(name: "stars", kind: Generated(count: Constant(3.0), columns: [("mass", Constant(10.0))]))"#;

    let cases = [
        (
            refusal(root, r#"(name: "a", kind: Field(Random(0.0, 1.0)))"#),
            "a",
        ),
        (
            refusal(root, r#"(name: "a", kind: Field(Row("stars", "age")))"#),
            "a",
        ),
        (
            refusal(
                r#"(name: "t", kind: Generated(count: Constant(1.0), columns: [("x", Noise(frequency: 0.1, octaves: 1))]))"#,
                "",
            ),
            "t",
        ),
        (
            refusal(
                &format!(
                    r#"{root}, (name: "t", kind: Generated(parent: Some("stars"), count: Share("mass"), columns: []))"#
                ),
                "",
            ),
            "t",
        ),
        (
            refusal(
                r#"(name: "t", kind: Generated(count: Constant(1.0), columns: [("x", Parent("mass"))]))"#,
                "",
            ),
            "t",
        ),
    ];

    for (error, name) in cases {
        let named = match &error {
            PackError::Invalid { stage, .. } => stage,
            PackError::InvalidTable { table, .. } => table,
            other => panic!("{other:?}"),
        };
        assert_eq!(named, name, "{error}");
    }
}

#[test]
fn tables_whose_parents_lead_back_to_them_are_refused() {
    let error = refusal(
        r#"(name: "a", kind: Generated(parent: Some("b"), count: Constant(1.0), columns: [])),
           (name: "b", kind: Generated(parent: Some("a"), count: Constant(1.0), columns: []))"#,
        "",
    );

    assert!(matches!(error, PackError::InvalidTable { .. }), "{error}");
}

#[test]
fn new_facts_that_leave_the_focused_row_as_it_was_drop_nothing_that_reads_it() {
    let pack = pack();
    let mut facts = Facts::new(Arc::clone(&pack), 5).expect("facts");
    facts
        .give("villages", vec![village(1, 20.0, "river")])
        .expect("villages");
    let mut runtime = Runtime::new(Arc::clone(&pack), 5, [8, 8]);
    runtime.set_facts(facts.clone()).expect("facts");
    runtime.focus("villages", RowId(vec![1])).expect("a row");
    generated(&mut runtime, "wealth");

    facts
        .give(
            "villages",
            vec![village(1, 20.0, "river"), village(2, 70.0, "hill")],
        )
        .expect("villages");
    let dropped = runtime.set_facts(facts).expect("facts");

    assert_eq!(dropped, Vec::new());
}
