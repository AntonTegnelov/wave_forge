//! A frozen city: the chunks it evicts go to its store as they are and come back from it, not
//! generated again, so they are unchanged after walking away and back, and after the rule set
//! changes.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use wave_forge::{
    BlockSolver, Builder, ChunkCoord, ChunkShape, FocusPoint, FrozenStore, Ruleset, StoreError,
    WgpuBackend, WorldExtent, WorldGenerator,
};
use wfc_devtools::city::{self, city_prior};

const CHUNK: ChunkShape = ChunkShape::cube(8);
const WIDE: i32 = 12;
const DEEP: i32 = 3;

/// The bytes kept, by layer and chunk.
type Kept = HashMap<(String, ChunkCoord), Vec<u8>>;

/// A store in memory, shared by every world given a clone of it, as a directory on disk is.
#[derive(Clone, Default)]
struct SharedStore(Arc<Mutex<Kept>>);

impl FrozenStore for SharedStore {
    fn keep(&mut self, layer: &str, chunk: ChunkCoord, bytes: Vec<u8>) -> Result<(), StoreError> {
        self.0
            .lock()
            .expect("not poisoned")
            .insert((layer.to_owned(), chunk), bytes);
        Ok(())
    }

    fn fetch(&mut self, layer: &str, chunk: ChunkCoord) -> Result<Option<Vec<u8>>, StoreError> {
        Ok(self
            .0
            .lock()
            .expect("not poisoned")
            .get(&(layer.to_owned(), chunk))
            .cloned())
    }
}

/// The city on a device, with its tiles weighted as the city weighs them, or with every other
/// tile four times as likely when `changed`: another rule set whose tiles mean the same.
fn world(changed: bool) -> WorldGenerator<BlockSolver<WgpuBackend>> {
    let city = city::city();
    let weights: Vec<f32> = city
        .modules
        .tileset
        .weights
        .iter()
        .enumerate()
        .map(|(tile, &weight)| {
            if changed && tile % 2 == 1 {
                weight * 4.0
            } else {
                weight
            }
        })
        .collect();
    let ruleset = Ruleset::new(&city.modules.rules, &weights).expect("the city compiles");
    Builder::new(ruleset, city_prior(&city, CHUNK.z))
        .seed(8)
        .extent(
            WorldExtent::new(CHUNK)
                .with_x(0..WIDE)
                .with_y(0..DEEP)
                .with_z(0..1),
        )
        .halo(1)
        .build()
        .expect("a compute device")
}

const NEAR: FocusPoint = FocusPoint::new(ChunkCoord::new(1, 1, 0), 1);
const FAR: FocusPoint = FocusPoint::new(ChunkCoord::new(WIDE - 2, 1, 0), 1);

/// The tiles of the chunks `focus` asks for, after generating around it.
fn near(world: &mut WorldGenerator<BlockSolver<WgpuBackend>>) -> BTreeMap<ChunkCoord, Vec<u16>> {
    world.request(&[NEAR]);
    world.run_until_idle().expect("the solver runs");
    world
        .store()
        .iter()
        .filter(|chunk| NEAR.covers(chunk.coord))
        .map(|chunk| (chunk.coord, chunk.tiles.to_vec()))
        .collect()
}

/// Walks the world's focus to the far end of the city and evicts everything behind it.
fn walk_away(world: &mut WorldGenerator<BlockSolver<WgpuBackend>>) {
    world.request(&[FAR]);
    world.run_until_idle().expect("the solver runs");
    world.evict_outside(&[FAR], 0);
}

#[test]
fn a_frozen_city_walked_away_from_and_back_is_as_it_was() {
    let store = SharedStore::default();
    let mut city = world(false).with_store(Box::new(store.clone()));
    let first = near(&mut city);

    walk_away(&mut city);
    let back = near(&mut city);

    assert_eq!(first.len(), 9, "the chunks around the focus");
    assert!(
        first.keys().all(|chunk| store
            .0
            .lock()
            .expect("not poisoned")
            .contains_key(&("tiles".to_owned(), *chunk))),
        "the chunks walked away from are in the store"
    );
    assert_eq!(back, first);
    assert!(city.stats().restored >= 9, "{:?}", city.stats());
}

#[test]
fn a_frozen_city_comes_back_from_its_store_after_the_rule_set_changes() {
    let store = SharedStore::default();
    let mut before = world(false).with_store(Box::new(store.clone()));
    let frozen = near(&mut before);
    walk_away(&mut before);

    let mut after = world(true).with_store(Box::new(store));
    let restored = near(&mut after);
    let changed = near(&mut world(true));

    assert_ne!(changed, frozen, "the changed weights change the city");
    assert_eq!(restored, frozen);
    assert_eq!(after.stats().solved, 0, "{:?}", after.stats());
    assert!(after.stats().restored >= 9, "{:?}", after.stats());
}
