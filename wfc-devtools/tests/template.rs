//! A module set learnt from a template, as WFC's overlapping model learns its patterns: a module per
//! distinct window weighted by how often the template holds it, two modules side by side exactly
//! where their windows agree, and a solved plane every window of which is one of the template's.

use std::collections::BTreeSet;
use std::sync::Arc;
use wfc_core::reference::ReferenceSolver;
use wfc_core::{
    ChunkCoord, ChunkShape, ChunkStore, Prior, Region, RegionStatus, Ruleset, WorldExtent,
    region_init,
};
use wfc_devtools::TileGrid;
use wfc_rules::loader::{RuleFile, parse_rule_file};
use wfc_rules::modules::{CompiledModules, NEG_X, NEG_Y, POS_X, POS_Y};
use wfc_rules::template::{MAX_PATTERNS, Template, TemplateError, learn};

/// Rooms of floor (1) in walls (0), with a door (2) between two of them: a template that tiles.
const ROOMS: [u32; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, //
    0, 1, 1, 1, 0, 1, 1, 0, //
    0, 1, 1, 1, 2, 1, 1, 0, //
    0, 1, 1, 1, 0, 1, 1, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, //
    0, 1, 1, 0, 1, 1, 1, 0, //
    0, 1, 1, 0, 1, 1, 1, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, //
];

fn rooms() -> Template<'static> {
    Template {
        width: 8,
        height: 8,
        cells: &ROOMS,
    }
}

/// Every `n` by `n` window of `template`, row by row, in the order first met.
fn windows(template: Template<'_>, n: usize, wrap: bool) -> Vec<Vec<u32>> {
    let (across, up) = if wrap {
        (template.width, template.height)
    } else {
        (template.width - n + 1, template.height - n + 1)
    };
    let mut out: Vec<Vec<u32>> = Vec::new();
    for y in 0..up {
        for x in 0..across {
            let window: Vec<u32> = (0..n * n)
                .map(|i| {
                    let (wx, wy) = ((x + i % n) % template.width, (y + i / n) % template.height);
                    template.cells[wy * template.width + wx]
                })
                .collect();
            if !out.contains(&window) {
                out.push(window);
            }
        }
    }
    out
}

fn compiled(text: &str) -> CompiledModules {
    let RuleFile::Modules(modules) = parse_rule_file(text).expect("a valid module set") else {
        panic!("a module set");
    };
    modules
}

/// The value module `tile` stands for, from its tag.
fn value(modules: &CompiledModules, tile: usize) -> u32 {
    let tag = modules
        .prototype_of(tile)
        .tags
        .iter()
        .find_map(|tag| tag.strip_prefix("value "))
        .expect("every module carries its value");
    tag.parse().expect("a value is a number")
}

#[test]
fn each_distinct_window_is_a_module_weighted_by_how_often_the_template_holds_it() {
    let text = learn(rooms(), 3, true).expect("a template that fits");

    let modules = compiled(&text);
    let windows = windows(rooms(), 3, true);

    assert_eq!(modules.prototypes.len(), windows.len());
    let total: f32 = modules.prototypes.iter().map(|p| p.weight).sum();
    assert_eq!(total, 64.0, "every window of a tiling template counts once");
    for (index, window) in windows.iter().enumerate() {
        assert_eq!(value(&modules, index), window[0], "pattern {index}");
    }
}

#[test]
fn two_modules_lie_side_by_side_exactly_where_their_windows_agree() {
    let n = 3;
    let modules = compiled(&learn(rooms(), n, true).expect("a template that fits"));
    let windows = windows(rooms(), n, true);
    let at = |window: &[u32], x: usize, y: usize| window[y * n + x];

    for (a, first) in windows.iter().enumerate() {
        for (b, second) in windows.iter().enumerate() {
            let across = (0..n).all(|y| (1..n).all(|x| at(first, x, y) == at(second, x - 1, y)));
            let along = (1..n).all(|y| (0..n).all(|x| at(first, x, y) == at(second, x, y - 1)));

            assert_eq!(
                modules.rules.check(a, b, POS_X),
                across,
                "{a} then {b} along +x"
            );
            assert_eq!(
                modules.rules.check(b, a, NEG_X),
                across,
                "{b} then {a} along -x"
            );
            assert_eq!(
                modules.rules.check(a, b, POS_Y),
                along,
                "{a} then {b} along +y"
            );
            assert_eq!(
                modules.rules.check(b, a, NEG_Y),
                along,
                "{b} then {a} along -y"
            );
        }
    }
}

#[test]
fn every_window_of_a_solved_plane_is_one_of_the_templates() {
    let n = 3;
    let modules = compiled(&learn(rooms(), n, true).expect("a template that fits"));
    let known: BTreeSet<Vec<u32>> = windows(rooms(), n, true).into_iter().collect();
    let ruleset = Arc::new(Ruleset::from_modules(&modules).expect("the set compiles"));
    let solver = ReferenceSolver::new(Arc::clone(&ruleset));
    let shape = ChunkShape { x: 16, y: 16, z: 1 };
    let extent = WorldExtent::new(shape)
        .with_x(0..1)
        .with_y(0..1)
        .with_z(0..1);
    let region = Region::new(ChunkCoord::new(0, 0, 0), shape.region(extent.halo(1)));
    let store = ChunkStore::new(extent);
    let prior = Prior::open(modules.variants.len() as u32);
    let mut values_seen = BTreeSet::new();

    for seed in 1..=4 {
        let init = region_init(&store, &prior, &ruleset, &region, false);
        let (domains, status, _) = solver.solve_region(region.shape(), &init, 1, seed);

        assert_eq!(status, RegionStatus::Solved, "seed {seed}");
        let grid = TileGrid::from_domains(&domains, 16, 16, 1).expect("every cell decided");
        let cell = |x: usize, y: usize| value(&modules, grid.get(x, y, 0));
        for y in 0..=16 - n {
            for x in 0..=16 - n {
                let window: Vec<u32> = (0..n * n).map(|i| cell(x + i % n, y + i / n)).collect();
                assert!(
                    known.contains(&window),
                    "seed {seed}: {window:?} at ({x}, {y})"
                );
                values_seen.insert(window[0]);
            }
        }
    }
    assert_eq!(
        values_seen,
        BTreeSet::from([0, 1, 2]),
        "walls, floors and a door"
    );
}

#[test]
fn without_wrapping_only_whole_windows_count() {
    let modules = compiled(&learn(rooms(), 3, false).expect("a template that fits"));

    let total: f32 = modules.prototypes.iter().map(|p| p.weight).sum();

    assert_eq!(modules.prototypes.len(), windows(rooms(), 3, false).len());
    assert_eq!(total, 36.0, "6 by 6 whole windows");
}

#[test]
fn learning_refuses_cells_that_do_not_fill_the_template_a_pattern_that_does_not_fit_and_too_many_patterns()
 {
    let short = Template {
        cells: &ROOMS[..63],
        ..rooms()
    };
    // Cells scrambled by a hash, so nearly every window differs.
    let noise: Vec<u32> = (0..32 * 32_u32)
        .map(|i| {
            let h = i.wrapping_mul(0x9E37_79B9);
            ((h ^ (h >> 16)).wrapping_mul(0x85EB_CA6B) >> 13) % 4
        })
        .collect();
    let many = Template {
        width: 32,
        height: 32,
        cells: &noise,
    };

    assert!(matches!(
        learn(short, 3, true),
        Err(TemplateError::Shape { .. })
    ));
    assert!(matches!(
        learn(rooms(), 0, true),
        Err(TemplateError::Size { .. })
    ));
    assert!(matches!(
        learn(rooms(), 9, true),
        Err(TemplateError::Size { .. })
    ));
    assert!(
        matches!(learn(many, 3, true), Err(TemplateError::TooMany(count)) if count > MAX_PATTERNS)
    );
}
