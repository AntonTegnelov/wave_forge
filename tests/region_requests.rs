//! A region stage after the focus has moved: once a region is computed its inputs are no longer
//! asked for, so a change that makes it stale has to ask for them again. Whatever the change, the
//! runtime regenerates without panicking and holds what a fresh runtime given the same change
//! holds.

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::stages::{Edit, Edits, Pack, Runtime};
use wave_forge::{ChunkCoord, FocusPoint};

const SIZE: [u32; 2] = [8, 8];
/// The islands preset, whose Talus stage `weathered` computes regions of 4 chunks from `eroded`.
const TARGET: &str = "height";

fn islands() -> Arc<Pack> {
    let text = std::fs::read_to_string(format!(
        "{}/examples/presets/islands.world.ron",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("the islands preset");
    Arc::new(Pack::parse(&text).expect("a valid pack"))
}

fn focus(x: i32) -> [FocusPoint; 1] {
    [FocusPoint::new(ChunkCoord::new(x, 0, 0), 1)]
}

/// A runtime that generated around chunk (0, 0) and then around (1, 0), so the regions it computed
/// first are held while the request no longer asks for all of their inputs.
fn moved() -> Runtime {
    let mut runtime = Runtime::new(islands(), 3, SIZE);
    runtime.request(&focus(0), &[TARGET]).expect("the stages");
    runtime.run_until_idle().expect("the stages run");
    runtime.request(&focus(1), &[TARGET]).expect("the stages");
    runtime.run_until_idle().expect("the stages run");
    runtime
}

/// The target's values in the chunks around (1, 0).
fn heights(runtime: &Runtime) -> Vec<Vec<f32>> {
    (-1..=1)
        .flat_map(|y| (0..=2).map(move |x| ChunkCoord::new(x, y, 0)))
        .map(|chunk| {
            runtime
                .field(TARGET, chunk)
                .expect("the target is generated")
                .values
                .clone()
        })
        .collect()
}

fn fresh(change: impl FnOnce(&mut Runtime)) -> Vec<Vec<f32>> {
    let mut runtime = Runtime::new(islands(), 3, SIZE);
    change(&mut runtime);
    runtime.request(&focus(1), &[TARGET]).expect("the stages");
    runtime.run_until_idle().expect("the stages run");
    heights(&runtime)
}

#[test]
fn a_parameter_changed_after_the_focus_moved_regenerates_as_a_fresh_runtime_would() {
    let values = BTreeMap::from([("land".to_owned(), 0.7_f32)]);
    let mut runtime = moved();

    runtime.set_params(&values).expect("a value in range");
    runtime.run_until_idle().expect("the stages run");

    let expected = fresh(|runtime| {
        runtime.set_params(&values).expect("a value in range");
    });
    assert!(heights(&runtime) == expected);
}

#[test]
fn an_input_raised_after_the_focus_moved_regenerates_as_a_fresh_runtime_would() {
    let mut edits = Edits::default();
    edits.push(Edit::Raise {
        stage: "land".to_owned(),
        column: (12, 3),
        by: 4.0,
    });
    let mut runtime = moved();

    runtime.set_edits(&edits).expect("edits");
    runtime.run_until_idle().expect("the stages run");

    let expected = fresh(|runtime| {
        runtime.set_edits(&edits).expect("edits");
    });
    assert!(heights(&runtime) == expected);
}
