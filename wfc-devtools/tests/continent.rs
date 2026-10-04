//! The maximal preset (docs/product/user-stories.md, M1, #247): a continent of 2 048 by 2 048
//! cells, 4 by 4 km at cells of 2 m, in chunks of 8 by 8 columns. Its biomes, coast and history are
//! checked by sampling, which generates no chunk; generating part of it to completion is a
//! measurement, run in release:
//!
//! ```text
//! cargo test -p wfc-devtools --release --test continent -- --ignored --nocapture
//! ```

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::stages::{Condition, Expr, Facts, Runtime, StageKind, Value};
use wave_forge::{ChunkCoord, FocusPoint};
use wfc_devtools::continent::{CULTURES, SEED, SIZE, history, history_json, pack, runtime, towns};

const SIDE: f32 = 2048.0;
const CENTRE: [f32; 2] = [1024.0, 1024.0];
const OCEANS: [&str; 2] = ["ocean", "deep_ocean"];

/// The biome at each of `points`, sampled without generating a chunk.
fn biomes(points: impl Iterator<Item = [f32; 2]>) -> Vec<String> {
    let pack = pack();
    let runtime = Runtime::new(Arc::clone(&pack), SEED, SIZE);
    let names = pack.kind("biome").expect("a biome stage").categories();
    points
        .map(|at| {
            let index = runtime.sample("biome", at).expect("a sampled biome") as usize;
            names[index].to_owned()
        })
        .collect()
}

/// `steps` by `steps` points evenly over the continent.
fn grid(steps: u32) -> impl Iterator<Item = [f32; 2]> {
    (0..steps).flat_map(move |j| {
        (0..steps).map(move |i| {
            [
                (i as f32 + 0.5) * SIDE / steps as f32,
                (j as f32 + 0.5) * SIDE / steps as f32,
            ]
        })
    })
}

/// The chunks from `low` up to but not including `high` along both axes.
fn square(low: i32, high: i32) -> Vec<ChunkCoord> {
    (low..high)
        .flat_map(|y| (low..high).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

#[test]
fn the_continent_chooses_among_forty_biomes_by_rules_and_forty_appear() {
    let pack = pack();

    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for biome in biomes(grid(128)) {
        *counts.entry(biome).or_default() += 1;
    }

    let declared = pack
        .kind("biome")
        .expect("a biome stage")
        .categories()
        .len();
    assert!(declared >= 40, "{declared} biomes declared");
    assert!(
        counts.len() >= 40,
        "{} biomes appear: {counts:?}",
        counts.len()
    );
}

#[test]
fn the_far_ground_s_biomes_are_the_near_ground_s_in_name_and_mostly_in_place() {
    let pack = pack();
    let runtime = Runtime::new(Arc::clone(&pack), SEED, SIZE);
    let near = pack.kind("biome").expect("a biome stage").categories();
    let far = pack
        .kind("far_biome")
        .expect("a far_biome stage")
        .categories();

    let points: Vec<[f32; 2]> = grid(96).collect();
    let agree = points
        .iter()
        .filter(|&&at| {
            runtime.sample("biome", at).expect("a sampled biome")
                == runtime
                    .sample("far_biome", at)
                    .expect("a sampled far biome")
        })
        .count();

    // The same categories in the same order, so the far ground takes the near ground's palette.
    assert_eq!(far, near);
    // Over the coarse height and temperature, the biome differs only near the bands' edges: 89.5
    // per cent of the points agree.
    let share = agree as f32 / points.len() as f32;
    assert!(
        share > 0.85,
        "the far ground's biomes agree with the near ground's at {share:.3} of the points"
    );
}

#[test]
fn the_coast_falls_into_the_ocean_before_the_bound_and_about_half_is_land() {
    let ring = (0..360).map(|degree| {
        let angle = (degree as f32).to_radians();
        [
            CENTRE[0] + 1000.0 * angle.cos(),
            CENTRE[1] + 1000.0 * angle.sin(),
        ]
    });

    let edge = biomes(ring);
    let all = biomes(grid(64));

    let dry = edge
        .iter()
        .filter(|biome| !OCEANS.contains(&biome.as_str()));
    assert_eq!(dry.count(), 0, "land 1 000 cells out: {edge:?}");
    let land = all
        .iter()
        .filter(|biome| !OCEANS.contains(&biome.as_str()) && biome.as_str() != "shallows")
        .count() as f32
        / all.len() as f32;
    assert!(
        (0.35..0.7).contains(&land),
        "{land} of the continent is land"
    );
}

#[test]
fn the_history_settles_every_culture_on_land_and_the_continent_takes_it() {
    let pack = pack();

    let rows = history(&pack);
    // The runtime refuses a history its table does not take, such as two settlements within a
    // chunk of each other.
    let _ = runtime();

    let mut cultures: BTreeMap<String, usize> = BTreeMap::new();
    for row in &rows {
        let Value::Name(culture) = &row.values["culture"] else {
            panic!("a culture is a name");
        };
        *cultures.entry(culture.clone()).or_default() += 1;
    }
    assert!(rows.len() >= 40, "{} settlements", rows.len());
    assert_eq!(cultures.len(), CULTURES.len(), "{cultures:?}");
}

#[test]
#[ignore = "a measurement: run in release with --ignored --nocapture"]
fn a_part_of_the_continent_generates_with_rivers_and_lakes() {
    let mut runtime = runtime();
    let part = square(120, 128);
    let focus: Vec<FocusPoint> = part.iter().map(|&c| FocusPoint::new(c, 0)).collect();

    let started = std::time::Instant::now();
    runtime
        .request(&focus, &["ground", "biome"])
        .expect("the stages");
    runtime.run_until_idle().expect("the stages run");
    let seconds = started.elapsed().as_secs_f64();

    // Rivers and lakes are region jobs over 64 chunks a side, so the region around the part holds
    // them wherever they are in it.
    let (mut rivers, mut lake_columns) = (0, 0);
    for chunk in square(64, 192) {
        rivers += runtime.curves("rivers", chunk).map_or(0, <[_]>::len);
        if let (Some(lakes), Some(terrain)) = (
            runtime.field("lakes", chunk),
            runtime.field("terrain", chunk),
        ) {
            lake_columns += lakes
                .values
                .iter()
                .zip(&terrain.values)
                .filter(|(water, ground)| *water > *ground)
                .count();
        }
    }
    eprintln!(
        "continent: 64 chunks of ground in {seconds:.1} s; {rivers} river pieces, \
         {lake_columns} lake columns"
    );
    for (stage, timing) in runtime.timings() {
        eprintln!(
            "continent: {stage}: {} chunks, {:.3} ms each",
            timing.products,
            timing.ms / timing.products.max(1) as f64
        );
    }
    for chunk in &part {
        assert!(runtime.field("ground", *chunk).is_some(), "{chunk:?}");
    }
    assert!(
        rivers > 0 && lake_columns > 0,
        "{rivers} rivers, {lake_columns} lake columns"
    );
}

#[test]
#[ignore = "a measurement: run in release with --ignored --nocapture"]
fn the_rock_under_cliffs_holds_caves_overhangs_and_ore() {
    let mut runtime = runtime();
    // Chunks on a plateau's rim, cliffs 15 to 50 cells up by sampling.
    let centre = ChunkCoord::new(95, 111, 0);
    let ores = ["coal", "iron", "copper", "gold"];
    let mut targets = vec!["ground", "rock"];
    targets.extend(ores);

    let started = std::time::Instant::now();
    runtime
        .request(&[FocusPoint::new(centre, 2)], &targets)
        .expect("the stages");
    runtime.run_until_idle().expect("the stages run");
    let seconds = started.elapsed().as_secs_f64();

    let around: Vec<ChunkCoord> = (-2..=2)
        .flat_map(|dy| (-2..=2).map(move |dx| ChunkCoord::new(centre.x + dx, centre.y + dy, 0)))
        .collect();
    let (mut caves, mut overhangs) = (0, 0);
    for &chunk in &around {
        let ground = runtime.field("ground", chunk).expect("ground");
        let rock = runtime.volume("rock", chunk).expect("rock");
        let [sx, sy, levels] = rock.size;
        for level in 0..levels {
            let z = (rock.bottom + level as i32) as f32 + 0.5;
            for column in 0..(sx * sy) as usize {
                let solid = rock.values[level as usize * (sx * sy) as usize + column] > 0.0;
                let surface = ground.values[column];
                caves += usize::from(!solid && z < surface - 8.0);
                overhangs += usize::from(solid && z > surface + 1.0);
            }
        }
    }
    let placed: Vec<usize> = ores
        .iter()
        .map(|ore| {
            around
                .iter()
                .map(|&chunk| runtime.points(ore, chunk).map_or(0, <[_]>::len))
                .sum()
        })
        .collect();
    let rock_ms = runtime
        .timings()
        .iter()
        .find(|(stage, _)| stage == "rock")
        .map_or(0.0, |(_, timing)| timing.ms / timing.products.max(1) as f64);
    eprintln!(
        "continent: 5x5 chunks of rock under cliffs in {seconds:.1} s: {caves} cave voxels, \
         {overhangs} overhanging voxels, ores {ores:?} {placed:?}; rock {rock_ms:.1} ms a chunk"
    );
    assert!(
        caves > 0 && overhangs > 0,
        "{caves} cave voxels, {overhangs} overhanging"
    );
    assert!(
        placed.iter().all(|&count| count > 0),
        "{ores:?}: {placed:?}"
    );
}

#[test]
#[ignore = "a measurement: run in release with --ignored --nocapture"]
fn a_region_places_its_locations_and_joins_them_by_roads() {
    let mut runtime = runtime();
    // The region of 64 chunks a side south-west of the centre, whole.
    let region = square(64, 128);
    let focus: Vec<FocusPoint> = region.iter().map(|&c| FocusPoint::new(c, 0)).collect();

    let started = std::time::Instant::now();
    runtime
        .request(&focus, &["places", "roads", "settlements", "settled"])
        .expect("the stages");
    runtime.run_until_idle().expect("the stages run");
    let seconds = started.elapsed().as_secs_f64();

    let mut places = BTreeMap::new();
    let mut roads = BTreeMap::new();
    let mut settlements = BTreeMap::new();
    for &chunk in &region {
        for site in runtime.sites("places", chunk).expect("placed") {
            places.insert(site.id.clone(), site.kind.clone());
        }
        for road in runtime.curves("roads", chunk).expect("joined") {
            roads.insert(road.id.clone(), road.points.len());
        }
        for site in runtime.sites("settlements", chunk).expect("settled") {
            settlements.insert(site.id.clone(), site.clone());
        }
    }
    // Every settlement of the history in the region stands on ground levelled to its height.
    for site in settlements.values() {
        let chunk = ChunkCoord::new(
            (site.min.0 + site.max.0) / 2,
            (site.min.1 + site.max.1) / 2,
            0,
        );
        let settled = runtime.field("settled", chunk).expect("levelled");
        assert!(
            settled
                .values
                .iter()
                .all(|&h| (h - site.height).abs() < 1e-3),
            "{:?} at {}",
            site.id,
            site.height
        );
    }
    let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
    for kind in places.values().flatten() {
        *kinds.entry(kind.to_string()).or_default() += 1;
    }
    eprintln!(
        "continent: a region of 64x64 chunks in {seconds:.1} s: {} places of {} kinds, {} roads, \
         {} settlements; {kinds:?}",
        places.len(),
        kinds.len(),
        roads.len(),
        settlements.len()
    );
    assert!(!settlements.is_empty(), "no settlement in the region");
    assert!(kinds.len() >= 15, "{kinds:?}");
    assert!(
        roads.len() + 1 >= places.len() / 2,
        "{} roads for {} places",
        roads.len(),
        places.len()
    );
}

#[test]
#[ignore = "a measurement: run in release with --ignored --nocapture"]
fn a_settlement_of_every_culture_gets_a_town_of_its_module_set() {
    let pack = pack();
    let rows = history(&pack);
    let mut runtime = runtime()
        .with_towns(Box::new(towns().expect("the cultures compile")))
        .expect("matching chunks");
    // The first settlement of each culture, by the history's order.
    let mut firsts: BTreeMap<String, [f32; 2]> = BTreeMap::new();
    for row in &rows {
        let (Value::Name(culture), Value::Number(x), Value::Number(y)) =
            (&row.values["culture"], &row.values["x"], &row.values["y"])
        else {
            panic!("a culture's name and a position");
        };
        firsts.entry(culture.clone()).or_insert([*x, *y]);
    }
    let focus: Vec<FocusPoint> = firsts
        .values()
        .map(|at| {
            let chunk = ChunkCoord::new(
                (at[0] / SIZE[0] as f32).floor() as i32,
                (at[1] / SIZE[1] as f32).floor() as i32,
                0,
            );
            FocusPoint::new(chunk, 0)
        })
        .collect();

    let started = std::time::Instant::now();
    runtime.request(&focus, &["towns"]).expect("the stages");
    runtime.run_until_idle().expect("the stages run");
    let seconds = started.elapsed().as_secs_f64();

    for (culture, point) in &firsts {
        let focus = FocusPoint::new(
            ChunkCoord::new(
                (point[0] / SIZE[0] as f32).floor() as i32,
                (point[1] / SIZE[1] as f32).floor() as i32,
                0,
            ),
            0,
        );
        let town = runtime
            .tiles("towns", focus.chunk)
            .unwrap_or_else(|| panic!("{culture}'s settlement has no town"));
        let site = runtime
            .sites("settlements", focus.chunk)
            .expect("settled")
            .iter()
            .find(|site| site.id == town.site)
            .expect("the town's site")
            .clone();
        assert_eq!(town.height, site.height, "{culture}");
    }
    eprintln!(
        "continent: a town of each of {} cultures in {seconds:.1} s",
        firsts.len()
    );
    assert_eq!(firsts.len(), CULTURES.len());
}

#[test]
fn the_continent_grows_two_hundred_pieces_in_its_dungeons_and_buildings() {
    let pack = pack();

    let pieces: usize = pack
        .stage_names()
        .filter_map(|name| match pack.kind(name) {
            Some(StageKind::Assemble { pieces, .. }) => Some(pieces.len()),
            _ => None,
        })
        .sum();

    assert!(pieces >= 200, "{pieces} pieces");
}

#[test]
#[ignore = "a measurement: run in release with --ignored --nocapture"]
fn every_larger_place_of_four_regions_grows_its_pieces() {
    let pack = pack();
    let assemblies: Vec<(String, Vec<String>, u32)> = pack
        .stage_names()
        .filter_map(|name| match pack.kind(name) {
            Some(StageKind::Assemble { kinds, min, .. }) => {
                Some((name.to_owned(), kinds.clone(), *min))
            }
            _ => None,
        })
        .collect();
    let mut runtime = runtime();
    let region = square(64, 192);
    let focus: Vec<FocusPoint> = region.iter().map(|&c| FocusPoint::new(c, 0)).collect();
    let mut targets = vec!["places"];
    targets.extend(assemblies.iter().map(|(name, ..)| name.as_str()));

    let started = std::time::Instant::now();
    runtime.request(&focus, &targets).expect("the stages");
    runtime.run_until_idle().expect("the stages run");
    let seconds = started.elapsed().as_secs_f64();

    let mut grown: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for (stage, kinds, min) in &assemblies {
        let mut places = BTreeMap::new();
        let mut pieces: BTreeMap<_, usize> = BTreeMap::new();
        for &chunk in &region {
            for site in runtime.sites("places", chunk).expect("placed") {
                if site
                    .kind
                    .as_deref()
                    .is_some_and(|kind| kinds.iter().any(|wanted| wanted == kind))
                {
                    places.insert(site.id.clone(), ());
                }
            }
            for stamp in runtime.stamps(stage, chunk).expect("grown") {
                *pieces.entry((stamp.site.clone(), stamp.id)).or_default() += 1;
            }
        }
        let mut per_place: BTreeMap<_, usize> = BTreeMap::new();
        for (site, _) in pieces.keys() {
            *per_place.entry(site.clone()).or_default() += 1;
        }
        for site in places.keys() {
            let count = per_place.get(site).copied().unwrap_or(0);
            assert!(count as u32 >= *min, "{stage} on {site:?}: {count} pieces");
        }
        grown.insert(stage.clone(), (places.len(), per_place.values().sum()));
    }
    eprintln!(
        "continent: four regions' assemblies in {seconds:.1} s, places and pieces by stage: {grown:?}"
    );
    assert!(grown.values().any(|(places, _)| *places > 0), "{grown:?}");
}

#[test]
fn the_pack_holds_what_m1_names() {
    let pack = pack();
    let kinds: Vec<&StageKind> = pack
        .stage_names()
        .map(|name| pack.kind(name).expect("a stage"))
        .collect();
    let count = |test: fn(&StageKind) -> bool| kinds.iter().filter(|kind| test(kind)).count();

    let locations: Vec<_> = kinds
        .iter()
        .filter_map(|kind| match kind {
            StageKind::Locations { kinds, .. } => Some(kinds),
            _ => None,
        })
        .flatten()
        .collect();
    let cultures = kinds.iter().find_map(|kind| match kind {
        StageKind::Solve {
            by: Some((column, cases)),
            ..
        } => Some((column.clone(), cases.len() + 1)),
        _ => None,
    });

    assert!(kinds.len() >= 100, "{} stages", kinds.len());
    assert!(locations.len() >= 30, "{} location kinds", locations.len());
    assert!(
        locations
            .iter()
            .all(|kind| kind.quota >= 1 && (kind.quota == 1 || kind.apart > 0.0)),
        "a location kind without a quota, or several without spacing"
    );
    assert_eq!(cultures, Some(("culture".to_owned(), CULTURES.len())));
    let facts = Facts::new(Arc::clone(&pack), SEED).expect("the tables");
    assert!(
        facts.table("settlements").is_some(),
        "no table of settlements"
    );
    assert_eq!(count(|kind| matches!(kind, StageKind::Rivers { .. })), 1);
    assert_eq!(count(|kind| matches!(kind, StageKind::Lakes { .. })), 1);
    assert_eq!(count(|kind| matches!(kind, StageKind::Network { .. })), 1);
    assert!(count(|kind| matches!(kind, StageKind::Volume { .. })) >= 1);
    assert!(count(|kind| matches!(kind, StageKind::Scatter { .. })) >= 40);
}

#[test]
fn every_vegetation_and_clutter_stage_grows_in_biomes_the_continent_has() {
    let pack = pack();
    let present: std::collections::BTreeSet<String> = biomes(grid(128)).into_iter().collect();

    for name in pack.stage_names() {
        let Some(StageKind::Scatter { when, .. }) = pack.kind(name) else {
            continue;
        };
        let wanted: Vec<&String> = when
            .iter()
            .filter_map(|condition| match condition {
                Condition::Greater(test, _) => match test.as_ref() {
                    Expr::Is(stage, names) if stage == "biome" => Some(names),
                    _ => None,
                },
                _ => None,
            })
            .flatten()
            .collect();

        assert!(!wanted.is_empty(), "{name} names no biome");
        assert!(
            wanted.iter().any(|biome| present.contains(*biome)),
            "none of {name}'s biomes {wanted:?} is on the continent"
        );
    }
}

#[test]
#[ignore = "a measurement: run in release with --ignored --nocapture"]
fn a_region_grows_its_vegetation_and_clutter() {
    let pack = pack();
    let scatters: Vec<String> = pack
        .stage_names()
        .filter(|name| matches!(pack.kind(name), Some(StageKind::Scatter { .. })))
        .map(ToOwned::to_owned)
        .collect();
    let mut runtime = runtime();
    let region = square(64, 128);
    let focus: Vec<FocusPoint> = region.iter().map(|&c| FocusPoint::new(c, 0)).collect();
    let mut targets: Vec<&str> = scatters.iter().map(String::as_str).collect();
    targets.push("biome");

    let started = std::time::Instant::now();
    runtime.request(&focus, &targets).expect("the stages");
    runtime.run_until_idle().expect("the stages run");
    let seconds = started.elapsed().as_secs_f64();

    let placed: BTreeMap<&str, usize> = scatters
        .iter()
        .map(|stage| {
            let count = region
                .iter()
                .map(|&chunk| runtime.points(stage, chunk).map_or(0, <[_]>::len))
                .sum();
            (stage.as_str(), count)
        })
        .collect();
    let empty: Vec<&&str> = placed
        .iter()
        .filter(|(_, count)| **count == 0)
        .map(|(stage, _)| stage)
        .collect();
    let slowest = runtime
        .timings()
        .into_iter()
        .filter(|(stage, _)| scatters.contains(stage))
        .map(|(stage, timing)| (timing.ms / timing.products.max(1) as f64, stage))
        .fold((0.0, String::new()), |a, b| if b.0 > a.0 { b } else { a });
    eprintln!(
        "continent: a region's {} vegetation and clutter stages in {seconds:.1} s, {} points; none from {empty:?}; the slowest {:.3} ms a chunk ({}); {placed:?}",
        scatters.len(),
        placed.values().sum::<usize>(),
        slowest.0,
        slowest.1
    );
    // One region holds some of the continent's biomes only: a stage whose biomes cover 500 of
    // its columns or more places something here. That every stage's biomes are on the continent
    // is checked by sampling.
    let names = pack.kind("biome").expect("a biome stage").categories();
    let mut columns: BTreeMap<&str, usize> = BTreeMap::new();
    for &chunk in &region {
        for &index in &runtime.categories("biome", chunk).expect("biomes").values {
            *columns.entry(names[usize::from(index)]).or_default() += 1;
        }
    }
    for stage in &scatters {
        let Some(StageKind::Scatter { when, .. }) = pack.kind(stage) else {
            unreachable!("listed as a Scatter stage");
        };
        let covered: usize = when
            .iter()
            .filter_map(|condition| match condition {
                Condition::Greater(test, _) => match test.as_ref() {
                    Expr::Is(of, names) if of == "biome" => Some(names),
                    _ => None,
                },
                _ => None,
            })
            .flatten()
            .map(|biome| columns.get(biome.as_str()).copied().unwrap_or(0))
            .sum();
        assert!(
            covered < 500 || placed[stage.as_str()] > 0,
            "{stage}: nothing placed on {covered} columns of its biomes"
        );
    }
}

/// A store that keeps only how many bytes each layer was given, to measure a whole run without
/// writing it.
#[derive(Default)]
struct Measure(BTreeMap<String, (usize, usize)>);

impl wave_forge::FrozenStore for Measure {
    fn keep(
        &mut self,
        layer: &str,
        _chunk: ChunkCoord,
        bytes: Vec<u8>,
    ) -> Result<(), wave_forge::StoreError> {
        let entry = self.0.entry(layer.to_owned()).or_default();
        entry.0 += 1;
        entry.1 += bytes.len();
        Ok(())
    }

    fn fetch(
        &mut self,
        _layer: &str,
        _chunk: ChunkCoord,
    ) -> Result<Option<Vec<u8>>, wave_forge::StoreError> {
        Ok(None)
    }
}

#[test]
#[ignore = "a measurement: run in release with --ignored --nocapture"]
fn the_whole_continent_runs_ahead_of_time() {
    let pack = pack();
    // What an engine draws and places: the ground and its materials, the rock, the water, the
    // towns, the places and every point and piece.
    let mut targets = vec![
        "ground",
        "biome",
        "rock",
        "lakes",
        "towns",
        "places",
        "settlements",
    ];
    targets.extend(pack.stage_names().filter(|name| {
        matches!(
            pack.kind(name),
            Some(StageKind::Scatter { .. } | StageKind::Embed { .. } | StageKind::Assemble { .. })
        )
    }));
    let mut runtime = runtime()
        .with_towns(Box::new(towns().expect("the cultures compile")))
        .expect("matching chunks");
    let mut store = Measure::default();

    let started = std::time::Instant::now();
    let end = runtime
        .run_world(&targets, &mut store, |progress| {
            if progress.done % 2048 == 0 {
                eprintln!(
                    "continent: {} of {} chunks after {:.0} s, {} held",
                    progress.done,
                    progress.total,
                    started.elapsed().as_secs_f64(),
                    progress.held
                );
            }
            std::ops::ControlFlow::Continue(())
        })
        .expect("the whole run");
    let seconds = started.elapsed().as_secs_f64();

    let (entries, bytes) = store
        .0
        .values()
        .fold((0, 0), |(entries, bytes), (n, b)| (entries + n, bytes + b));
    eprintln!(
        "continent: {} chunks of {} targets in {seconds:.0} s, {entries} entries, {:.2} GB",
        end.done,
        targets.len(),
        bytes as f64 / 1e9,
    );
    let mut costs: Vec<(f64, String)> = runtime
        .timings()
        .into_iter()
        .map(|(stage, timing)| (timing.ms / 1000.0, stage))
        .collect();
    costs.sort_by(|a, b| b.0.total_cmp(&a.0));
    eprintln!(
        "continent: the costliest stages in seconds {:?}",
        &costs[..8]
    );
    assert_eq!(end.done, end.total);
}

#[test]
fn the_history_engines_give_is_the_simulations() {
    let path = format!(
        "{}/../examples/continent/history.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let simulated = history_json(&history(&pack()));

    if std::env::var_os("WAVE_FORGE_BLESS").is_some() {
        std::fs::write(&path, &simulated).expect("write the history");
    }
    let committed = std::fs::read_to_string(&path)
        .expect("examples/continent/history.json; record it with WAVE_FORGE_BLESS=1");

    assert!(
        committed.replace("\r\n", "\n") == simulated,
        "examples/continent/history.json is not the simulation's; record it with WAVE_FORGE_BLESS=1"
    );
}
