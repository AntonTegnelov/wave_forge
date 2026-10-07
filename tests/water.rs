//! Water surfaces from a pack: a river down a valley, its bed carved 0.8 cells deep and its water
//! 0.3 cells below the banks, and a lake in a hollow, drawn as one water mesh per chunk. The river
//! is wet along its line, the water meets the banks with no edge in the open, and neighbouring
//! chunks' water meets exactly.

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::stages::{Facts, GivenRow, Pack, Runtime, Value};
use wave_forge::water::WET;
use wave_forge::{ChunkCoord, FocusPoint, WaterMesh, water_surface};

const SIZE: [u32; 2] = [8, 8];
const CELL: [f32; 3] = [2.0, 1.0, 2.0];

/// A valley falling along +x with a hollow in it, a river down its floor, and the water: the lakes
/// where they stand above the ground, and the river's water.
const PACK: &str = r#"(
    version: 1,
    // A sea far below the valley, so its lakes stand above it.
    water: Some((level: -100.0, lakes: Some("lakes"))),
    tables: [(name: "rivers", kind: Given(columns: [
        ("x0", Number), ("y0", Number), ("x1", Number), ("y1", Number), ("width", Number),
    ]))],
    stages: [
        (name: "terrain", kind: Field(Add(
            Add(Mul(X, Constant(-0.05)), Mul(Abs(Sub(Y, Constant(32.0))), Constant(0.3))),
            Mul(Smoothstep(10.0, 4.0, Distance((40.0, 20.0))), Constant(-3.0)),
        ))),
        (name: "lakes", kind: Lakes(height: "terrain", region: 8, min_columns: 4)),
        (name: "rivers", kind: TableCurves(table: "rivers", from: ("x0", "y0"), to: ("x1", "y1"), radius: "width")),
        (name: "ground", kind: Apply(height: "terrain", curves: "rivers", max_radius: 3, blend: 2, profile: Carve(0.8))),
        (name: "river_water", kind: Apply(height: "terrain", curves: "rivers", max_radius: 3, blend: 2, profile: Carve(0.3))),
        (name: "water", kind: Field(Max(
            Select(
                when: Greater(Input("lakes"), Add(Input("terrain"), Constant(0.05))),
                then: Input("lakes"),
                otherwise: Constant(-1000.0),
            ),
            Input("river_water"),
        ))),
    ],
)"#;

fn runtime() -> Runtime {
    let pack = Arc::new(Pack::parse(PACK).expect("a valid pack"));
    let mut facts = Facts::new(Arc::clone(&pack), 9).expect("facts");
    let river = GivenRow {
        id: 1,
        values: BTreeMap::from([
            ("x0".to_owned(), Value::Number(2.0)),
            ("y0".to_owned(), Value::Number(32.0)),
            ("x1".to_owned(), Value::Number(62.0)),
            ("y1".to_owned(), Value::Number(32.0)),
            ("width".to_owned(), Value::Number(1.5)),
        ]),
    };
    facts.give("rivers", vec![river]).expect("the river");
    let mut runtime = Runtime::new(pack, 9, SIZE);
    runtime.set_facts(facts).expect("facts");
    let focus: Vec<FocusPoint> = (0..8)
        .flat_map(|y| (0..8).map(move |x| FocusPoint::new(ChunkCoord::new(x, y, 0), 0)))
        .collect();
    runtime
        .request(&focus, &["water", "ground", "terrain"])
        .expect("the stages");
    runtime.run_until_idle().expect("the stages run");
    runtime
}

fn water(runtime: &Runtime, chunk: ChunkCoord) -> WaterMesh {
    water_surface(
        chunk,
        |at| runtime.field("water", at),
        |at| runtime.field("ground", at),
        CELL,
    )
    .expect("the fields around it arrived")
}

/// The value of field `stage` at the world column `(x, y)`.
fn value(runtime: &Runtime, stage: &str, x: i64, y: i64) -> f32 {
    let chunk = ChunkCoord::new(x.div_euclid(8) as i32, y.div_euclid(8) as i32, 0);
    runtime
        .field(stage, chunk)
        .expect("generated")
        .get(x.rem_euclid(8) as u32, y.rem_euclid(8) as u32)
}

#[test]
fn the_river_is_wet_along_its_line_below_its_banks() {
    let runtime = runtime();

    for x in 4..56 {
        let level = value(&runtime, "water", x, 32);
        let bed = value(&runtime, "ground", x, 32);
        let bank = value(&runtime, "terrain", x, 32);
        assert!(
            level - bed > WET,
            "column ({x}, 32): water {level}, bed {bed}"
        );
        assert!(
            level < bank,
            "column ({x}, 32): water {level} over its bank {bank}"
        );
    }
}

#[test]
fn the_lake_stands_level_in_its_hollow() {
    let runtime = runtime();
    let chunk = ChunkCoord::new(5, 2, 0);

    let mesh = water(&runtime, chunk);

    let wet: Vec<f32> = (0..9)
        .flat_map(|j| (0..9).map(move |i| (i, j)))
        .filter(|&(i, j)| {
            let (x, y) = (40 + i, 16 + j);
            value(&runtime, "water", x, y) - value(&runtime, "ground", x, y) > WET
        })
        .map(|(i, j)| mesh.positions[(j * 9 + i) as usize][1])
        .collect();
    assert!(
        wet.len() > 4,
        "only {} wet vertices in the hollow",
        wet.len()
    );
    assert!(
        wet.iter().all(|&h| (h - wet[0]).abs() < 1e-4),
        "the lake's surface is not level: {wet:?}"
    );
}

#[test]
fn a_wet_vertex_inside_a_chunk_has_water_on_every_side() {
    let runtime = runtime();

    for chunk in [ChunkCoord::new(3, 3, 0), ChunkCoord::new(5, 2, 0)] {
        let mesh = water(&runtime, chunk);
        let drawn: std::collections::BTreeSet<u32> =
            mesh.indices.chunks(6).map(|square| square[0]).collect();
        for j in 1..8 {
            for i in 1..8 {
                let (x, y) = (i64::from(chunk.x) * 8 + i, i64::from(chunk.y) * 8 + j);
                if value(&runtime, "water", x, y) - value(&runtime, "ground", x, y) <= WET {
                    continue;
                }
                // The four squares around the vertex, each named by its corner nearest the origin.
                for corner in [(i - 1, j - 1), (i, j - 1), (i - 1, j), (i, j)] {
                    let square = (corner.1 * 9 + corner.0) as u32;
                    assert!(
                        drawn.contains(&square),
                        "{chunk:?}: the square at {corner:?} beside a wet vertex is not drawn"
                    );
                }
            }
        }
    }
}

#[test]
fn neighbouring_chunks_water_meets_exactly() {
    let runtime = runtime();
    let (a, b) = (ChunkCoord::new(3, 4, 0), ChunkCoord::new(4, 4, 0));

    let (a, b) = (water(&runtime, a), water(&runtime, b));

    for j in 0..9 {
        let on_a = a.positions[(j * 9 + 8) as usize];
        let on_b = b.positions[(j * 9) as usize];
        assert_eq!(on_a[1], on_b[1], "row {j}");
        assert_eq!(on_a[0] - 8.0 * CELL[0], on_b[0], "row {j}");
    }
}
