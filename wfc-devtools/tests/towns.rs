//! Towns of the city on a real device: a Solve stage places a bounded city on every site, each one
//! valid across its chunk seams and the same whatever order its chunks are asked for in.

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::stages::{Pack, Runtime, SiteId};
use wave_forge::towns::WfcTowns;
use wave_forge::{Chunk, ChunkCoord, ChunkShape, FocusPoint};
use wfc_devtools::city;
use wfc_devtools::invariants::{BoundaryCondition, TileGrid, adjacency_violations};

const CHUNK: ChunkShape = ChunkShape::cube(8);

const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "height", kind: Field(Mul(Noise(frequency: 0.01, octaves: 3), Constant(40.0)))),
        (name: "towns", kind: Sites(height: "height", region: 6, size: (2, 3), chance: 1.0)),
        (name: "city", kind: Solve(sites: "towns", rules: "city", bottom: Some(Tagged("street_level")), top: Some(Named("air")))),
    ],
)"#;

fn runtime() -> Runtime {
    let city = wave_forge::loader::RuleFile::Modules(city::city().modules);
    let towns = WfcTowns::new(CHUNK)
        .with_rules("city", city, wave_forge::towns::gpu_solver)
        .expect("the city compiles");
    Runtime::new(
        Arc::new(Pack::parse(PACK).expect("a valid pack")),
        3,
        [8, 8],
    )
    .with_towns(Box::new(towns))
    .expect("matching chunks")
}

fn area() -> Vec<ChunkCoord> {
    (0..12)
        .flat_map(|y| (0..12).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

/// Each town's chunks, by the site it stands on, as the runtime hands them out.
fn towns(order: &[Vec<ChunkCoord>]) -> BTreeMap<SiteId, BTreeMap<ChunkCoord, Vec<u16>>> {
    let mut runtime = runtime();
    let mut towns: BTreeMap<SiteId, BTreeMap<ChunkCoord, Vec<u16>>> = BTreeMap::new();
    for request in order {
        let focus: Vec<FocusPoint> = request.iter().map(|&c| FocusPoint::new(c, 0)).collect();
        runtime.request(&focus, &["city"]).expect("a stage");
        runtime.run_until_idle().expect("the stages run");
        for &chunk in request {
            if let Some(town) = runtime.tiles("city", chunk) {
                towns
                    .entry(town.site.clone())
                    .or_default()
                    .insert(chunk, town.tiles.to_vec());
            }
        }
    }
    towns
}

#[test]
fn every_town_is_valid_across_its_seams_and_the_same_in_any_order() {
    let city = city::city();

    let at_once = towns(&[area()]);
    let one_by_one = towns(
        &area()
            .into_iter()
            .rev()
            .map(|c| vec![c])
            .collect::<Vec<_>>(),
    );

    assert_eq!(at_once.len(), 4, "one town per region of the 2×2 regions");
    assert_eq!(at_once, one_by_one, "the same towns in any order");
    for (site, chunks) in &at_once {
        let SiteId::Region(x, y) = site else {
            panic!("a Sites stage's town is named by its region: {site:?}")
        };
        let chunks: Vec<Chunk> = chunks
            .iter()
            .map(|(&coord, tiles)| Chunk {
                coord,
                tiles: tiles.clone().into_boxed_slice(),
                version: 1,
            })
            .collect();
        let (grid, _) = TileGrid::from_chunks(CHUNK, &chunks, city.air).expect("chunks");
        let violations =
            adjacency_violations(&grid, &city.modules.rules, BoundaryCondition::Finite);
        eprintln!(
            "towns: region ({x}, {y}), {} chunks, {}x{}x{} cells, {} violations",
            chunks.len(),
            grid.width,
            grid.height,
            grid.depth,
            violations.len()
        );
        assert!(violations.is_empty(), "{site:?}: {violations:?}");
        let path =
            std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("town_{x}_{y}.png"));
        wfc_devtools::render::render_voxel_isometric(&grid, &city.voxels, 2)
            .save(&path)
            .expect("write PNG");
    }
}
