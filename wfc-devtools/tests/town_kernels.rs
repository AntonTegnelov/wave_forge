//! What a town's first solve spends compiling kernels, against solving.
//!
//! A town is a bounded WFC world, and the GPU solver compiles a pipeline for each region shape it
//! first meets. This solves towns of every size a Sites stage of 2 to 3 chunks makes, one after
//! another on one solver, and checks that only the first compiles anything: the sizes differ in
//! how many regions a batch holds, never in the shapes. It also solves a first town twice on fresh
//! devices sharing a pipeline cache, cold and then warm. It prints the time each solve took and how
//! much of it was compiling, which docs/research/measurements.md records.

use std::time::Instant;
use wave_forge::ChunkShape;
use wave_forge::loader::RuleFile;
use wave_forge::towns::{
    Selector, TownRequest, TownSolver, WfcTowns, gpu_solver, gpu_solver_cached,
};
use wfc_devtools::city;

#[test]
fn only_the_first_town_compiles_kernels() {
    let city = RuleFile::Modules(city::city().modules);
    let mut towns = WfcTowns::new(ChunkShape::cube(8))
        .with_rules("city", city, gpu_solver)
        .expect("the city compiles");
    let bottom = Selector::Tagged("street_level".to_owned());
    let top = Selector::Named("air".to_owned());

    let mut compiled = Vec::new();
    for (index, size) in [(2, 2), (2, 3), (3, 2), (3, 3)].into_iter().enumerate() {
        let before = towns
            .solver("city")
            .expect("the city's solver")
            .compilation();
        let started = Instant::now();
        towns
            .solve(&TownRequest {
                rules: "city",
                seed: 17 + index as u64,
                size,
                bottom: Some(&bottom),
                top: Some(&top),
            })
            .expect("the town solves");
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        let after = towns
            .solver("city")
            .expect("the city's solver")
            .compilation();
        println!(
            "town {size:?}: {ms:.0} ms, of which {:.0} ms compiling {} pipelines",
            after.ms - before.ms,
            after.pipelines - before.pipelines
        );
        compiled.push(after.pipelines - before.pipelines);
    }

    assert!(compiled[0] > 0, "the first town compiles its kernels");
    assert_eq!(&compiled[1..], [0, 0, 0], "later towns reuse them");
}

#[test]
fn a_second_start_compiles_from_the_pipelines_the_first_cached() {
    let dir = std::env::temp_dir().join(format!("wave_forge_kernels_{}", std::process::id()));
    let bottom = Selector::Tagged("street_level".to_owned());
    let top = Selector::Named("air".to_owned());
    let first_town = |label: &str| {
        let city = RuleFile::Modules(city::city().modules);
        let mut towns = WfcTowns::new(ChunkShape::cube(8))
            .with_rules("city", city, |rules| gpu_solver_cached(rules, &dir))
            .expect("the city compiles");
        let started = Instant::now();
        towns
            .solve(&TownRequest {
                rules: "city",
                seed: 17,
                size: (3, 3),
                bottom: Some(&bottom),
                top: Some(&top),
            })
            .expect("the town solves");
        let solver = towns.solver("city").expect("the city's solver");
        println!(
            "{label} first town: {:.0} ms, of which {:.0} ms compiling {} pipelines; cached: {}",
            started.elapsed().as_secs_f64() * 1000.0,
            solver.compilation().ms,
            solver.compilation().pipelines,
            solver.backend().caches_pipelines()
        );
        solver.backend().caches_pipelines()
    };

    let cached = first_town("cold");
    let sizes: Vec<u64> = match std::fs::read_dir(&dir) {
        Ok(entries) => entries
            .flatten()
            .map(|entry| entry.metadata().expect("a cache file").len())
            .collect(),
        Err(error) if !cached && error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => panic!("the cache directory: {error}"),
    };
    first_town("warm");
    if cached {
        std::fs::remove_dir_all(&dir).expect("the test's cache directory");
    }

    if cached {
        assert_eq!(sizes.len(), 1, "one cache file for the adapter: {sizes:?}");
        assert!(sizes[0] > 0, "the cache file holds the compiled pipelines");
    } else {
        assert!(
            sizes.is_empty(),
            "no cache file where the device cannot cache: {sizes:?}"
        );
    }
}
