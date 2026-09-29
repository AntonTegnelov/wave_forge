//! N6's library half (docs/product/user-stories.md): a module set proposed from the meshes of a
//! building kit, here the city's stand-in voxel models, generates a city with no unplaced chunk.

use std::sync::Arc;
use wfc_core::reference::ReferenceSolver;
use wfc_core::{
    ChunkCoord, ChunkShape, ChunkStore, Prior, Region, RegionStatus, Ruleset, WorldExtent,
    region_init,
};
use wfc_devtools::city;
use wfc_devtools::models::mesh;
use wfc_devtools::{BoundaryCondition, TileGrid, adjacency_violations};
use wfc_rules::import::{KitModule, propose};
use wfc_rules::loader::{RuleFile, parse_rule_file};

#[test]
fn a_set_proposed_from_the_citys_meshes_generates_a_city_with_no_unplaced_chunk() {
    let city = city::city();
    // Each module's mesh, back in the lattice's frame from the engine's.
    let meshes: Vec<(&str, Vec<[f32; 3]>)> = city
        .prototype_models()
        .into_iter()
        .map(|(name, model)| {
            let positions = mesh(model)
                .positions
                .iter()
                .map(|&[x, y, z]| [x, z, y])
                .collect();
            (name, positions)
        })
        .collect();
    let kit: Vec<KitModule<'_>> = meshes
        .iter()
        .map(|(name, positions)| KitModule { name, positions })
        .collect();

    let proposed = propose(&kit, 1.0 / 16.0);

    let RuleFile::Modules(modules) = parse_rule_file(&proposed).expect("a valid module set") else {
        panic!("a module set");
    };
    assert_eq!(modules.prototypes.len(), kit.len());
    let ruleset = Arc::new(Ruleset::from_modules(&modules).expect("the proposal compiles"));
    let solver = ReferenceSolver::new(Arc::clone(&ruleset));
    let shape = ChunkShape { x: 8, y: 8, z: 6 };
    let extent = WorldExtent::new(shape)
        .with_x(0..1)
        .with_y(0..1)
        .with_z(0..1);
    let region = Region::new(ChunkCoord::new(0, 0, 0), shape.region(extent.halo(1)));
    let store = ChunkStore::new(extent);
    let prior = Prior::open(modules.variants.len() as u32);
    let mut used = std::collections::BTreeSet::new();
    for seed in 1..=4 {
        let init = region_init(&store, &prior, &ruleset, &region, false);

        let (domains, status, _) = solver.solve_region(region.shape(), &init, 1, seed);

        assert_eq!(status, RegionStatus::Solved, "seed {seed}");
        let grid = TileGrid::from_domains(&domains, 8, 8, 6).expect("every cell decided");
        assert!(adjacency_violations(&grid, &modules.rules, BoundaryCondition::Finite).is_empty());
        for z in 0..6 {
            for y in 0..8 {
                for x in 0..8 {
                    used.insert(modules.prototype_of(grid.get(x, y, z)).name.clone());
                }
            }
        }
    }
    // A city of many kinds of module, not a region of air.
    assert!(used.len() > kit.len() / 3, "only {used:?}");
}
