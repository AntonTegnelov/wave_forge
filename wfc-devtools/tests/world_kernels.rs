//! A world's GPU kernels kept across starts.
//!
//! A world built with `Builder::build_cached` keeps the pipelines it compiles in a directory, so a
//! game that starts a world at boot compiles them once per machine. This builds the city's world
//! twice on fresh devices sharing one directory, cold and then warm, warms each one's kernels for a
//! view of radius 1, and prints what each start spent compiling.

use std::time::Instant;
use wave_forge::{Builder, ChunkShape, Ruleset};
use wfc_devtools::city::{self, city_prior};

const CHUNK: ChunkShape = ChunkShape::cube(8);

#[test]
fn a_second_world_start_compiles_from_the_pipelines_the_first_cached() {
    let dir = std::env::temp_dir().join(format!("wave_forge_world_kernels_{}", std::process::id()));
    let start = |label: &str| {
        let city = city::city();
        let ruleset = Ruleset::from_modules(&city.modules).expect("the city compiles");
        let started = Instant::now();
        let mut world = Builder::new(ruleset, city_prior(&city, CHUNK.z))
            .seed(11)
            .halo(1)
            .build_cached(&dir)
            .expect("a compute device and a cache directory");
        let shapes = world.kernel_shapes(1);
        world
            .solver_mut()
            .warm(&shapes)
            .expect("the kernels compile");
        let solver = world.solver_mut();
        println!(
            "{label} start: {:.0} ms, of which {:.0} ms compiling {} pipelines; cached: {}",
            started.elapsed().as_secs_f64() * 1000.0,
            solver.compilation().ms,
            solver.compilation().pipelines,
            solver.backend().caches_pipelines()
        );
        solver.backend().caches_pipelines()
    };

    let cached = start("cold");
    let sizes: Vec<u64> = match std::fs::read_dir(&dir) {
        Ok(entries) => entries
            .flatten()
            .map(|entry| entry.metadata().expect("a cache file").len())
            .collect(),
        Err(error) if !cached && error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => panic!("the cache directory: {error}"),
    };
    start("warm");
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
