//! A Painted stage: the categories of a map a game paints and gives the runtime as pixels, each
//! column taking the category its pixel's colour stands for in the legend, the picture's x and y
//! along the world's, and `otherwise` beyond it. A colour the legend does not know, or pixels that do
//! not fill the picture, are refused when given; a stage whose painting was never given fails by
//! name; a new painting stales what read the old one; and sampling gives what the chunks hold.

use std::sync::Arc;
use wave_forge::stages::{Pack, PackError, Painting, Runtime, StageError};
use wave_forge::{ChunkCoord, FocusPoint};

const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "biome", kind: Painted(painting: "map", origin: (0.0, 0.0), cell: 8.0,
            legend: [((40, 120, 40), "forest"), ((30, 60, 200), "sea"), ((20, 50, 180), "sea")],
            otherwise: "void")),
    ],
)"#;

const SIZE: [u32; 2] = [8, 8];
const FOREST: [u8; 4] = [40, 120, 40, 255];
const SEA: [u8; 4] = [30, 60, 200, 255];
const DEEP: [u8; 4] = [20, 50, 180, 255];

/// Three pixels wide and two high: forest along the first row, sea along the second, deep sea at
/// its end.
fn painting() -> Painting {
    Painting {
        width: 3,
        height: 2,
        pixels: vec![FOREST, FOREST, FOREST, SEA, SEA, DEEP],
    }
}

fn runtime() -> Runtime {
    Runtime::new(Arc::new(Pack::parse(PACK).expect("a valid pack")), 1, SIZE)
}

fn categories(runtime: &mut Runtime, chunks: &[ChunkCoord]) -> Vec<Vec<u8>> {
    let focus: Vec<FocusPoint> = chunks.iter().map(|&c| FocusPoint::new(c, 0)).collect();
    runtime.request(&focus, &["biome"]).expect("a stage");
    runtime.run_until_idle().expect("the stage runs");
    chunks
        .iter()
        .map(|&chunk| {
            runtime
                .categories("biome", chunk)
                .expect("generated")
                .values
                .clone()
        })
        .collect()
}

fn name(runtime: &Runtime, chunk: ChunkCoord) -> Vec<String> {
    let pack = Pack::parse(PACK).expect("a valid pack");
    let names = pack.kind("biome").expect("a stage").categories();
    runtime
        .categories("biome", chunk)
        .expect("generated")
        .values
        .iter()
        .map(|&c| names[usize::from(c)].to_owned())
        .collect()
}

#[test]
fn a_column_takes_its_pixels_category_the_pictures_axes_along_the_worlds() {
    let mut runtime = runtime();
    runtime
        .give_painting("map", painting())
        .expect("a painting the legend reads");
    // A cell of 8 cells is a chunk of 8 columns: chunk (x, y) is pixel (x, y).
    let chunks: Vec<ChunkCoord> = [(0, 0), (2, 0), (0, 1), (2, 1), (3, 0), (0, 2)]
        .iter()
        .map(|&(x, y)| ChunkCoord::new(x, y, 0))
        .collect();

    categories(&mut runtime, &chunks);

    let first = |chunk: ChunkCoord| name(&runtime, chunk)[0].clone();
    assert_eq!(first(chunks[0]), "forest");
    assert_eq!(first(chunks[1]), "forest");
    assert_eq!(first(chunks[2]), "sea");
    assert_eq!(first(chunks[3]), "sea", "deep sea is sea too");
    assert_eq!(first(chunks[4]), "void", "beyond the picture's last column");
    assert_eq!(first(chunks[5]), "void", "beyond its last row");
    assert!(name(&runtime, chunks[0]).iter().all(|n| n == "forest"));
}

#[test]
fn a_colour_the_legend_does_not_know_or_pixels_that_do_not_fill_the_picture_are_refused() {
    let mut runtime = runtime();
    let mut unknown = painting();
    unknown.pixels[4] = [250, 0, 0, 255];
    let mut short = painting();
    short.pixels.pop();

    let refused = runtime
        .give_painting("map", unknown)
        .expect_err("an unknown colour");
    let short = runtime
        .give_painting("map", short)
        .expect_err("too few pixels");

    assert!(
        matches!(&refused, StageError::Painting { painting, message } if painting == "map" && message.contains("(1, 1)")),
        "{refused:?}"
    );
    assert!(matches!(short, StageError::Painting { .. }), "{short:?}");
}

#[test]
fn a_stage_whose_painting_was_never_given_fails_by_name() {
    let mut runtime = runtime();
    runtime
        .request(&[FocusPoint::new(ChunkCoord::new(0, 0, 0), 0)], &["biome"])
        .expect("a stage");

    let error = runtime.run_until_idle().expect_err("no painting");

    assert_eq!(error, StageError::NoPainting("map".to_owned()));
}

#[test]
fn a_new_painting_stales_what_read_the_old() {
    let mut runtime = runtime();
    runtime
        .give_painting("map", painting())
        .expect("a painting");
    let chunk = ChunkCoord::new(0, 0, 0);
    categories(&mut runtime, &[chunk]);
    let mut flooded = painting();
    flooded.pixels[0] = SEA;

    let dropped = runtime
        .give_painting("map", flooded)
        .expect("another painting");
    categories(&mut runtime, &[chunk]);

    assert!(
        dropped.contains(&("biome".to_owned(), chunk)),
        "{dropped:?}"
    );
    assert!(name(&runtime, chunk).iter().all(|n| n == "sea"));
}

#[test]
fn sampling_gives_what_the_chunks_hold() {
    let mut runtime = runtime();
    runtime
        .give_painting("map", painting())
        .expect("a painting");
    let chunks: Vec<ChunkCoord> = (-1..4)
        .flat_map(|y| (-1..4).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect();
    let held = categories(&mut runtime, &chunks);

    let sampled = runtime
        .atlas("biome", [-8, -8], [40, 40])
        .expect("a sampled stage");

    for (i, value) in sampled.into_iter().enumerate() {
        let (x, y) = (-8 + (i % 40) as i32, -8 + (i / 40) as i32);
        let chunk = (x.div_euclid(8) + 1 + (y.div_euclid(8) + 1) * 5) as usize;
        let column = (x.rem_euclid(8) + y.rem_euclid(8) * 8) as usize;
        assert_eq!(value, f32::from(held[chunk][column]), "({x}, {y})");
    }
}

#[test]
fn loading_refuses_a_cell_of_no_size_and_a_colour_given_twice() {
    let none = PACK.replace("cell: 8.0", "cell: 0.0");
    let twice = PACK.replace("((20, 50, 180), \"sea\")", "((40, 120, 40), \"sea\")");

    assert!(matches!(Pack::parse(&none), Err(PackError::Invalid { .. })));
    assert!(matches!(
        Pack::parse(&twice),
        Err(PackError::Invalid { .. })
    ));
}
