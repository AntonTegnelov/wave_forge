//! What a chunk hands an engine: its tiles grouped into instances of the models to draw.
//!
//! A game draws one model per module name, at every cell holding one of that module's tiles,
//! turned by the tile's rotation and scaled to the cell. [`instance_sets`] does that arithmetic
//! once, in the library, so an integration only hands the result to its engine: Godot takes a set's
//! [`InstanceSet::transforms`] as a MultiMesh buffer in one call.

use crate::loader::RuleFile;
use crate::space::YUpSpace;
use std::collections::BTreeMap;
use wfc_core::Chunk;

/// Every placement of one module in one chunk.
#[derive(Clone, Debug, PartialEq)]
pub struct InstanceSet {
    /// The module's name, which names its model.
    pub name: String,
    /// Twelve floats per instance, the rows of a 3×4 transform in a Y-up engine's world space:
    /// `basis.x.x, basis.y.x, basis.z.x, origin.x`, then the same for y and z. That is Godot's
    /// MultiMesh buffer layout for 3D transforms without colours or custom data.
    pub transforms: Vec<f32>,
    /// Which cell each instance stands in, stable across runs: the chunk's id in the high 32 bits
    /// and the cell's index in the low ones. A game keys what it attaches to an instance by it.
    pub ids: Vec<u64>,
}

impl InstanceSet {
    /// How many instances the set holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether it holds none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// The placements of every module with something to draw in `chunk`, one set per module name in
/// name order. `drawn` says which names have a model; a module without one (air) gets no set.
///
/// Each instance's transform turns the module's unit model by its tile's rotation, scales it to the
/// cell, and puts it at the cell's centre.
#[must_use]
pub fn instance_sets(
    chunk: &Chunk,
    rules: &RuleFile,
    space: &YUpSpace,
    drawn: impl Fn(&str) -> bool,
) -> Vec<InstanceSet> {
    let scale = space.cell_size();
    let mut sets: BTreeMap<&str, InstanceSet> = BTreeMap::new();
    for (cell, &tile) in chunk.tiles.iter().enumerate() {
        let tile = usize::from(tile);
        let name = rules.name(tile);
        if !drawn(name) {
            continue;
        }
        let cell = u32::try_from(cell).expect("a chunk has fewer than 2^32 cells");
        let (sin, cos) = YUpSpace::yaw(rules.rotation(tile)).sin_cos();
        let origin = space.cell_center(chunk.coord, cell);
        // The basis turns about +Y, then scales each of the engine's axes to the cell.
        let rows = [
            [cos * scale[0], 0.0, sin * scale[0], origin[0]],
            [0.0, scale[1], 0.0, origin[1]],
            [-sin * scale[2], 0.0, cos * scale[2], origin[2]],
        ];
        let set = sets.entry(name).or_insert_with(|| InstanceSet {
            name: name.to_owned(),
            transforms: Vec::new(),
            ids: Vec::new(),
        });
        set.transforms.extend(rows.iter().flatten());
        set.ids
            .push((u64::from(chunk.coord.id()) << 32) | u64::from(cell));
    }
    sets.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use wfc_core::{ChunkCoord, ChunkShape};

    const ROADS: &str = r#"(
        faces: {
            "air": Side(connector: "air"),
            "road": Side(connector: "road", walkable: true),
            "open": Top(connector: "open"),
        },
        modules: [
            (name: "air", sides: ["air", "air", "air", "air"], up: "open", down: "open"),
            (name: "road", sides: ["road", "road", "air", "air"], up: "open", down: "open"),
        ],
    )"#;

    fn rules() -> RuleFile {
        crate::loader::parse_rule_file(ROADS).expect("a module set")
    }

    /// A 2x1x1 chunk at (3, 0, 0) with a road turned a quarter at cell 0 and air at cell 1.
    fn chunk(rules: &RuleFile) -> Chunk {
        let turned = rules
            .tiles_named("road")
            .into_iter()
            .find(|&tile| rules.rotation(tile) == 1)
            .expect("a turned road");
        let air = rules.tiles_named("air")[0];
        Chunk {
            coord: ChunkCoord::new(3, 0, 0),
            tiles: vec![turned as u16, air as u16].into_boxed_slice(),
            version: 1,
        }
    }

    fn space() -> YUpSpace {
        YUpSpace::new(ChunkShape { x: 2, y: 1, z: 1 }, [2.0, 3.0, 2.0])
    }

    #[test]
    fn a_chunk_gives_one_set_per_module_with_a_model() {
        let rules = rules();

        let sets = instance_sets(&chunk(&rules), &rules, &space(), |name| name != "air");

        assert_eq!(sets.len(), 1);
        assert_eq!(sets[0].name, "road");
        assert_eq!(sets[0].len(), 1);
        assert_eq!(sets[0].transforms.len(), 12);
    }

    #[test]
    fn an_instance_stands_at_its_cells_centre_scaled_and_turned() {
        let rules = rules();
        let space = space();
        let chunk = chunk(&rules);

        let set = &instance_sets(&chunk, &rules, &space, |name| name != "air")[0];

        let t = &set.transforms;
        // Transform the model's corner (0.5, 0, 0): turned a quarter it points along +z, scaled by
        // the cell's 2 along z, and placed at the cell's centre.
        let point = |x: f32, y: f32, z: f32| {
            [0, 4, 8].map(|row| t[row] * x + t[row + 1] * y + t[row + 2] * z + t[row + 3])
        };
        let centre = space.cell_center(chunk.coord, 0);
        let moved = point(0.5, 0.0, 0.0);
        assert!((moved[0] - centre[0]).abs() < 1e-5, "{moved:?}");
        assert!((moved[2] - (centre[2] + 1.0)).abs() < 1e-5, "{moved:?}");
        assert!(
            (point(0.0, 0.5, 0.0)[1] - (centre[1] + 1.5)).abs() < 1e-5,
            "up scales by 3"
        );
    }

    #[test]
    fn an_instance_id_names_its_chunk_and_cell() {
        let rules = rules();
        let chunk = chunk(&rules);

        let set = &instance_sets(&chunk, &rules, &space(), |_| true)
            .into_iter()
            .find(|set| set.name == "air")
            .expect("air drawn when asked for");

        assert_eq!(set.ids, vec![(u64::from(chunk.coord.id()) << 32) | 1]);
    }
}
