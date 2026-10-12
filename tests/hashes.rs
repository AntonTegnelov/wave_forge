//! Product and chunk hashes (docs/product/user-stories.md, N9): two machines given the same seed and
//! pack hash every chunk alike whatever order they generate it in, so peers can compare hashes to
//! confirm they agree; another seed or an edit gives another hash, and a chunk has none until it
//! holds every stage asked about.

use std::sync::Arc;
use wave_forge::stages::{Edit, Edits, Pack, Runtime};
use wave_forge::{ChunkCoord, FocusPoint};

const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "ground", kind: Field(Mul(Noise(frequency: 0.05, octaves: 3), Constant(10.0)))),
        (name: "ring", kind: Rules(rules: [
            (category: "inner", when: [Less(Distance((3.0, 5.0)), Constant(20.0))]),
        ], otherwise: "rest")),
        (name: "trees", kind: Scatter(kind: "tree", height: "ground", spacing: 3, apart: 3)),
    ],
)"#;

const STAGES: [&str; 3] = ["ground", "ring", "trees"];

fn runtime(seed: u64) -> Runtime {
    Runtime::new(
        Arc::new(Pack::parse(PACK).expect("a valid pack")),
        seed,
        [8, 8],
    )
}

fn area() -> Vec<ChunkCoord> {
    (-2..=2)
        .flat_map(|y| (-2..=2).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

/// Each of `chunks`' hashes, generated one request at a time in that order, each hashed as soon as
/// it is generated, before the next request moves the focus away from it.
fn one_at_a_time(runtime: &mut Runtime, chunks: &[ChunkCoord]) -> Vec<(ChunkCoord, Option<u64>)> {
    chunks
        .iter()
        .map(|&chunk| {
            runtime
                .request(&[FocusPoint::new(chunk, 0)], &STAGES)
                .expect("known stages");
            runtime.run_until_idle().expect("the stages run");
            (chunk, runtime.chunk_hash(chunk, &STAGES))
        })
        .collect()
}

/// Every chunk of the area's hash, the whole area asked for at once.
fn hashes(runtime: &mut Runtime) -> Vec<Option<u64>> {
    let focus: Vec<FocusPoint> = area().into_iter().map(|c| FocusPoint::new(c, 0)).collect();
    runtime.request(&focus, &STAGES).expect("known stages");
    runtime.run_until_idle().expect("the stages run");
    area()
        .into_iter()
        .map(|chunk| runtime.chunk_hash(chunk, &STAGES))
        .collect()
}

#[test]
fn a_chunk_hashes_alike_whatever_order_it_is_generated_in() {
    let (mut forward, mut backward) = (runtime(9), runtime(9));
    let mut reversed = area();
    reversed.reverse();

    let a = one_at_a_time(&mut forward, &area());
    let mut b = one_at_a_time(&mut backward, &reversed);
    b.reverse();

    assert!(a.iter().all(|(_, hash)| hash.is_some()), "{a:?}");
    assert_eq!(a, b);
    assert_eq!(
        a.iter().map(|&(_, hash)| hash).collect::<Vec<_>>(),
        hashes(&mut runtime(9))
    );
}

#[test]
fn another_seed_hashes_every_chunk_otherwise() {
    let (nine, ten) = (hashes(&mut runtime(9)), hashes(&mut runtime(10)));

    for (chunk, (a, b)) in area().into_iter().zip(nine.into_iter().zip(ten)) {
        assert_ne!(a, b, "{chunk:?}");
    }
}

#[test]
fn a_chunk_has_no_hash_until_it_holds_every_stage_asked_about() {
    let mut runtime = runtime(9);
    let chunk = ChunkCoord::new(0, 0, 0);

    runtime
        .request(&[FocusPoint::new(chunk, 0)], &["ground"])
        .expect("a known stage");
    runtime.run_until_idle().expect("the stages run");

    assert!(runtime.chunk_hash(chunk, &["ground"]).is_some());
    assert!(runtime.chunk_hash(chunk, &STAGES).is_none());
    assert!(runtime.product_hash("trees", chunk).is_none());
}

#[test]
fn an_edit_changes_the_hash_of_what_it_changes_and_nothing_else() {
    let mut runtime = runtime(9);
    let before = hashes(&mut runtime);
    let mut edits = Edits::default();
    edits.push(Edit::Raise {
        stage: "ground".to_owned(),
        column: (5, 5),
        by: 7.0,
    });

    runtime.set_edits(&edits).expect("edits");
    let after = hashes(&mut runtime);

    let origin = ChunkCoord::new(0, 0, 0);
    for (chunk, (was, is)) in area().into_iter().zip(before.into_iter().zip(after)) {
        if chunk == origin {
            assert_ne!(was, is, "the raised chunk");
        } else {
            assert_eq!(was, is, "{chunk:?}");
        }
    }
}
