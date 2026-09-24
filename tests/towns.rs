//! Solve stages: a town per site, solved whole, the same whatever order its chunks are asked for in,
//! on a thread of their own while other stages generate.
//!
//! These run on the CPU reference solver with a small module set of ground and air, so they need no
//! GPU; the city on a GPU is checked in `wfc-devtools`.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};
use wave_forge::loader::parse_rule_file;
use wave_forge::stages::{Pack, Runtime, StageError};
use wave_forge::towns::{Town, TownError, TownRequest, TownSolver, WfcTowns};
use wave_forge::{ChunkCoord, ChunkShape, FocusPoint};
use wfc_core::reference::ReferenceSolver;

const RULES: &str = r#"(
    faces: {
        "air": Side(connector: "air"),
        "ground": Side(connector: "ground", walkable: true),
        "open": Top(connector: "open"),
        "bedrock": Top(connector: "bedrock"),
    },
    modules: [
        (name: "air", sides: ["air", "air", "air", "air"], up: "open", down: "open"),
        (name: "ground", sides: ["ground", "ground", "ground", "ground"], up: "open", down: "bedrock", tags: ["street_level"]),
        (name: "plaza", sides: ["ground", "ground", "ground", "ground"], up: "open", down: "bedrock", tags: ["street_level"]),
    ],
)"#;

const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "height", kind: Field(Mul(Noise(frequency: 0.05, octaves: 2), Constant(10.0)))),
        (name: "towns", kind: Sites(height: "height", region: 5, size: (1, 2), chance: 1.0)),
        (name: "buildings", kind: Solve(sites: "towns", rules: "blocks", bottom: Some(Tagged("street_level")), top: Some(Named("air")))),
    ],
)"#;

const CHUNK: ChunkShape = ChunkShape { x: 4, y: 4, z: 3 };

fn runtime() -> Runtime {
    let file = parse_rule_file(RULES).expect("a module set");
    let towns = WfcTowns::new(CHUNK)
        .with_rules("blocks", file, |ruleset| Ok(ReferenceSolver::new(ruleset)))
        .expect("the rules compile");
    Runtime::new(
        Arc::new(Pack::parse(PACK).expect("a valid pack")),
        5,
        [CHUNK.x, CHUNK.y],
    )
    .with_towns(Box::new(towns))
    .expect("matching chunks")
}

fn area() -> Vec<ChunkCoord> {
    (0..10)
        .flat_map(|y| (0..10).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

/// Every chunk's town tiles over the area, or `None` outside the sites.
fn generate(order: &[Vec<ChunkCoord>]) -> BTreeMap<ChunkCoord, Option<Vec<u16>>> {
    let mut runtime = runtime();
    let mut out = BTreeMap::new();
    for request in order {
        let focus: Vec<FocusPoint> = request.iter().map(|&c| FocusPoint::new(c, 0)).collect();
        runtime.request(&focus, &["buildings"]).expect("a stage");
        runtime.run_until_idle().expect("the stages run");
        for &chunk in request {
            out.insert(
                chunk,
                runtime
                    .tiles("buildings", chunk)
                    .map(|town| town.tiles.to_vec()),
            );
        }
    }
    out
}

#[test]
fn a_town_stands_only_on_its_site_street_level_below_and_air_above() {
    let file = parse_rule_file(RULES).expect("a module set");
    let street: Vec<u16> = file
        .tiles_tagged("street_level")
        .iter()
        .map(|&t| t as u16)
        .collect();
    let air: Vec<u16> = file.tiles_named("air").iter().map(|&t| t as u16).collect();
    let layer = (CHUNK.x * CHUNK.y) as usize;

    let tiles = generate(&[area()]);

    let towns: Vec<&Vec<u16>> = tiles.values().flatten().collect();
    assert!(towns.len() >= 4, "only {} town chunks", towns.len());
    assert!(
        towns.len() < tiles.len(),
        "some chunks lie outside every site"
    );
    for chunk in towns {
        assert!(
            chunk[..layer].iter().all(|t| street.contains(t)),
            "{chunk:?}"
        );
        assert!(
            chunk[chunk.len() - layer..].iter().all(|t| air.contains(t)),
            "{chunk:?}"
        );
    }
}

#[test]
fn towns_come_out_the_same_in_any_order() {
    let all_at_once = generate(&[area()]);
    let one_by_one = generate(
        &area()
            .into_iter()
            .rev()
            .map(|c| vec![c])
            .collect::<Vec<_>>(),
    );

    assert_eq!(all_at_once, one_by_one);
}

#[test]
fn a_solve_stage_without_a_town_solver_says_so() {
    let mut runtime = Runtime::new(
        Arc::new(Pack::parse(PACK).expect("a valid pack")),
        5,
        [CHUNK.x, CHUNK.y],
    );
    runtime
        .request(
            &[FocusPoint::new(ChunkCoord::new(2, 2, 0), 3)],
            &["buildings"],
        )
        .expect("a stage");

    let result = runtime.run_until_idle();

    assert_eq!(
        result,
        Err(StageError::NoTownSolver("buildings".to_owned()))
    );
}

#[test]
fn a_rule_set_the_solver_was_not_given_is_named() {
    let file = parse_rule_file(RULES).expect("a module set");
    let towns = WfcTowns::new(CHUNK)
        .with_rules("other", file, |ruleset| Ok(ReferenceSolver::new(ruleset)))
        .expect("the rules compile");
    let mut runtime = Runtime::new(
        Arc::new(Pack::parse(PACK).expect("a valid pack")),
        5,
        [CHUNK.x, CHUNK.y],
    )
    .with_towns(Box::new(towns))
    .expect("matching chunks");
    runtime
        .request(
            &[FocusPoint::new(ChunkCoord::new(2, 2, 0), 3)],
            &["buildings"],
        )
        .expect("a stage");

    let result = runtime.run_until_idle();

    assert!(
        matches!(&result, Err(StageError::Town { message, .. }) if message.contains("blocks")),
        "{result:?}"
    );
}

/// A town solver that holds each town until the test lets it go.
struct Gated {
    gate: Receiver<()>,
}

impl TownSolver for Gated {
    fn chunk_shape(&self) -> ChunkShape {
        CHUNK
    }

    fn solve(&mut self, request: &TownRequest<'_>) -> Result<Town, TownError> {
        self.gate.recv().expect("the test lets the town go");
        let (w, h) = request.size;
        let cells = (CHUNK.x * CHUNK.y * CHUNK.z) as usize;
        Ok(Town {
            size: request.size,
            chunks: vec![Arc::from(vec![0u16; cells]); (w * h) as usize],
        })
    }
}

#[test]
fn every_other_stage_generates_while_a_town_is_being_solved() {
    let (open, gate) = channel::<()>();
    let mut runtime = Runtime::new(
        Arc::new(Pack::parse(PACK).expect("a valid pack")),
        5,
        [CHUNK.x, CHUNK.y],
    )
    .with_towns(Box::new(Gated { gate }))
    .expect("matching chunks");
    let focus: Vec<FocusPoint> = area().into_iter().map(|c| FocusPoint::new(c, 0)).collect();
    runtime
        .request(&focus, &["height", "buildings"])
        .expect("stages");

    let first = runtime.step(usize::MAX).expect("the stages run");
    let waiting = !runtime.is_idle();
    for _ in 0..area().len() {
        open.send(()).expect("the town thread");
    }
    let rest = runtime.run_until_idle().expect("the stages run");

    let in_a_town = |(stage, chunk): &(String, ChunkCoord)| {
        stage == "buildings" && runtime.tiles("buildings", *chunk).is_some()
    };
    assert!(first.iter().filter(|(stage, _)| stage == "height").count() >= area().len());
    assert!(
        !first.iter().any(in_a_town),
        "a town chunk came before its town"
    );
    assert!(waiting);
    assert!(rest.iter().any(in_a_town));
    assert!(runtime.is_idle());
}
