//! Region tags in a Bevy app: what walkers stand on and the rooms a generated chunk holds, placed
//! where the plugin's settings put its cells in Bevy's world. On the CPU reference solver, so no
//! device is needed.

mod common;

use common::{World, generated};
use wave_forge::ChunkCoord;

#[test]
fn each_cell_stands_on_its_modules_surface_and_rooms_are_in_one_interior() {
    let (app, tiles) = generated();
    let world = app.world().resource::<World>();
    let origin = ChunkCoord::new(0, 0, 0);
    let chunk = world.chunk(origin).expect("generated");

    let tags = world.region_tags(origin, &tiles).expect("generated");

    let mut rooms = 0;
    for (cell, &tile) in chunk.tiles.iter().enumerate() {
        let cell = u32::try_from(cell).expect("few cells");
        let centre = world.settings().cell_translation(origin, cell);
        let room = tiles.name(usize::from(tile)) == "room";
        rooms += usize::from(room);
        assert_eq!(
            world.surface_at(centre, &tiles),
            Some(if room { "wood" } else { "grass" }),
            "cell {cell}"
        );
        let holding = tags
            .interiors
            .iter()
            .filter(|interior| {
                (0..3).all(|axis| {
                    centre[axis] > interior.min[axis] && centre[axis] < interior.max[axis]
                })
            })
            .count();
        assert_eq!(holding, usize::from(room), "cell {cell}");
    }
    assert!(rooms > 0, "the chunk has no room, so nothing was checked");
    assert_eq!(tags.emitters.len(), rooms);
}
