//! What a chunk hands an engine: its tiles grouped into instances of the models to draw.
//!
//! A game draws one model per module name, at every cell holding one of that module's tiles,
//! turned by the tile's rotation and scaled to the cell, and gives it a collider the same way,
//! unscaled. [`instance_sets`] does that arithmetic once, in the library, so an integration only
//! hands the result to its engine: Godot takes [`InstanceSet::transforms`] as a MultiMesh buffer
//! in one call, and the same placements unscaled as the shapes of a chunk's body.
//!
//! The same placements, with each module's collision shape as triangles, are what an engine bakes
//! a chunk's navigation mesh from: [`nav_source`] gathers them for one chunk and the edges of its
//! neighbours, so every engine bakes chunks that meet.

use crate::loader::RuleFile;
use crate::space::YUpSpace;
use std::collections::BTreeMap;
use wfc_core::{Chunk, ChunkCoord, WorldExtent};

/// Which placement an instance is, the same in every run, session and machine: the chunk it
/// belongs to and a local id within it. A game keys what it attaches to an instance, and saved
/// edits, by it.
///
/// The local id is positional, never an ordinal, so adding or removing other placements never
/// changes it: the stage that placed it in bits 48 to 62, a slot for several placements of one
/// stage in one cell in bits 32 to 47, and the cell's index within the chunk in the low 32 bits. It
/// stays below 2^63, so an engine's signed 64-bit integer holds it.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Deserialize, serde::Serialize,
)]
pub struct InstanceId {
    pub chunk: ChunkCoord,
    pub local: u64,
}

impl InstanceId {
    /// The stage that places a chunk's tiles.
    pub const TILES: u16 = 0;
    /// The largest stage number a local id holds.
    pub const MAX_STAGE: u16 = (1 << 15) - 1;

    /// The id of placement `slot` of `stage` in `cell` of `chunk`.
    ///
    /// # Panics
    /// If `stage` is above [`InstanceId::MAX_STAGE`]; stage numbers are assigned by the library.
    #[must_use]
    pub fn new(chunk: ChunkCoord, stage: u16, cell: u32, slot: u16) -> Self {
        assert!(
            stage <= Self::MAX_STAGE,
            "stage {stage} does not fit an instance id"
        );
        Self {
            chunk,
            local: (u64::from(stage) << 48) | (u64::from(slot) << 32) | u64::from(cell),
        }
    }

    /// The stage that placed it.
    #[must_use]
    pub const fn stage(self) -> u16 {
        (self.local >> 48) as u16
    }

    /// Which of its stage's placements in its cell it is.
    #[must_use]
    pub const fn slot(self) -> u16 {
        (self.local >> 32) as u16
    }

    /// The index of its cell within its chunk.
    #[must_use]
    pub const fn cell(self) -> u32 {
        self.local as u32
    }
}

/// Every placement of one module in one chunk.
#[derive(Clone, Debug, PartialEq)]
pub struct InstanceSet {
    /// The module's name, which names its model.
    pub name: String,
    /// Which placement each instance is.
    pub ids: Vec<InstanceId>,
    /// Each instance's turn from its module, in quarter turns about the lattice's +z.
    pub turns: Vec<u8>,
    /// Each instance's cell centre in a Y-up engine's world space.
    pub origins: Vec<[f32; 3]>,
}

impl InstanceSet {
    /// Twelve floats per instance, the rows of a 3×4 transform in a Y-up engine's world space:
    /// `basis.x.x, basis.y.x, basis.z.x, origin.x`, then the same for y and z. That is Godot's
    /// MultiMesh buffer layout for 3D transforms without colours or custom data.
    ///
    /// The basis turns about +Y by the instance's rotation, then scales the engine's axes by
    /// `scale`: the cell's size to draw a unit model, one for a collider shaped for the cell.
    #[must_use]
    pub fn transforms(&self, scale: [f32; 3]) -> Vec<f32> {
        self.turns
            .iter()
            .zip(&self.origins)
            .flat_map(|(&turns, origin)| {
                let (sin, cos) = YUpSpace::yaw(turns).sin_cos();
                [
                    cos * scale[0],
                    0.0,
                    sin * scale[0],
                    origin[0],
                    0.0,
                    scale[1],
                    0.0,
                    origin[1],
                    -sin * scale[2],
                    0.0,
                    cos * scale[2],
                    origin[2],
                ]
            })
            .collect()
    }

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

/// The placements of every module `wanted` names in `chunk`, one set per module name in name order:
/// a module without a model or a collider (air) gets no set.
#[must_use]
pub fn instance_sets(
    chunk: &Chunk,
    rules: &RuleFile,
    space: &YUpSpace,
    wanted: impl Fn(&str) -> bool,
) -> Vec<InstanceSet> {
    let mut sets: BTreeMap<&str, InstanceSet> = BTreeMap::new();
    for (cell, &tile) in chunk.tiles.iter().enumerate() {
        let tile = usize::from(tile);
        let name = rules.name(tile);
        if !wanted(name) {
            continue;
        }
        let cell = u32::try_from(cell).expect("a chunk has fewer than 2^32 cells");
        let set = sets.entry(name).or_insert_with(|| InstanceSet {
            name: name.to_owned(),
            ids: Vec::new(),
            turns: Vec::new(),
            origins: Vec::new(),
        });
        set.ids
            .push(InstanceId::new(chunk.coord, InstanceId::TILES, cell, 0));
        set.turns.push(rules.rotation(tile));
        set.origins.push(space.cell_center(chunk.coord, cell));
    }
    sets.into_values().collect()
}

/// The triangles an engine bakes one chunk's navigation mesh from, and where to bake.
#[derive(Clone, Debug, PartialEq)]
pub struct NavSource {
    /// Nine floats per triangle: three corners, each x, y and z in a Y-up engine's world space, in
    /// the winding the shape's faces were given in.
    pub triangles: Vec<f32>,
    /// The lowest corner and size of the box to bake within: the chunk grown by the border along
    /// the engine's x and z, and by a chunk's height below and above.
    pub bounds_origin: [f32; 3],
    pub bounds_size: [f32; 3],
    /// How far the bounds reach past the chunk along x and z. Baked with a border this wide, the
    /// mesh ends on the chunk's edges, shaped by the neighbours' geometry as if there were no seam.
    pub border: f32,
}

/// Why [`nav_source`] could not gather a chunk's source.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum NavSourceError {
    /// A neighbour inside the world is not generated yet; ask again once it is.
    #[error("chunk {0:?} is not generated yet")]
    Missing(ChunkCoord),
    /// The world is more than one chunk tall, or unbounded upwards. A chunk's source covers its
    /// column's neighbours only, so stacked chunks would bake floors that do not agree.
    #[error("navigation needs a world one chunk tall")]
    TallWorld,
    /// The border reaches past the neighbouring chunks, whose geometry is all the source holds.
    #[error("a border of {border} reaches past a neighbouring chunk {chunk} wide")]
    BorderTooWide { border: f32, chunk: f32 },
}

/// The navigation source of the chunk at `coord`: the collision shape of every module `faces`
/// gives triangles for, placed at each of its instances, in the chunk and in its neighbours as far
/// as `border` past the chunk's edges plus a cell.
///
/// `chunks` looks up generated chunks, `extent` says which ones the world holds, and `faces` gives
/// a module's collision shape as triangles around its cell's centre, unscaled, in the engine's
/// axes, or `None` for a module agents neither walk on nor bump into.
///
/// # Errors
/// [`NavSourceError::Missing`] until every neighbour the world holds is generated;
/// [`NavSourceError::TallWorld`] for a world more than one chunk tall;
/// [`NavSourceError::BorderTooWide`] when `border` is wider than a chunk.
pub fn nav_source<'a, 'f>(
    coord: ChunkCoord,
    chunks: impl Fn(ChunkCoord) -> Option<&'a Chunk>,
    extent: &WorldExtent,
    rules: &RuleFile,
    space: &YUpSpace,
    faces: impl Fn(&str) -> Option<&'f [[f32; 3]]>,
    border: f32,
) -> Result<NavSource, NavSourceError> {
    if extent
        .chunks_along(2)
        .is_none_or(|layers| layers.len() != 1)
    {
        return Err(NavSourceError::TallWorld);
    }
    let chunk_size = space.chunk_size();
    let narrowest = chunk_size[0].min(chunk_size[2]);
    if border > narrowest {
        return Err(NavSourceError::BorderTooWide {
            border,
            chunk: narrowest,
        });
    }
    let neighbours: Vec<ChunkCoord> = (-1..=1)
        .flat_map(|dx| (-1..=1).map(move |dy| ChunkCoord::new(coord.x + dx, coord.y + dy, coord.z)))
        .filter(|&neighbour| extent.contains_chunk(neighbour))
        .collect();
    let mut present = Vec::with_capacity(neighbours.len());
    for &neighbour in &neighbours {
        present.push(chunks(neighbour).ok_or(NavSourceError::Missing(neighbour))?);
    }

    let origin = space.chunk_origin(coord);
    let bounds_origin = [
        origin[0] - border,
        origin[1] - chunk_size[1],
        origin[2] - border,
    ];
    let bounds_size = [
        chunk_size[0] + 2.0 * border,
        3.0 * chunk_size[1],
        chunk_size[2] + 2.0 * border,
    ];
    // A shape reaches at most about a cell from its cell's centre, so an instance whose centre is
    // within a cell of the bounds may touch them.
    let cell = space.cell_size();
    let near = |at: &[f32; 3]| {
        (0..3).all(|axis| {
            at[axis] >= bounds_origin[axis] - cell[axis]
                && at[axis] <= bounds_origin[axis] + bounds_size[axis] + cell[axis]
        })
    };
    let mut triangles = Vec::new();
    for chunk in present {
        for set in instance_sets(chunk, rules, space, |name| faces(name).is_some()) {
            let corners = faces(&set.name).expect("only modules with faces have sets");
            let transforms = set.transforms([1.0; 3]);
            for (row, origin) in transforms.chunks(12).zip(&set.origins) {
                if !near(origin) {
                    continue;
                }
                for corner in corners {
                    for axis in 0..3 {
                        let r = &row[axis * 4..axis * 4 + 4];
                        triangles
                            .push(r[0] * corner[0] + r[1] * corner[1] + r[2] * corner[2] + r[3]);
                    }
                }
            }
        }
    }
    Ok(NavSource {
        triangles,
        bounds_origin,
        bounds_size,
        border,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use wfc_core::ChunkShape;

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
        assert_eq!(sets[0].transforms([1.0; 3]).len(), 12);
    }

    #[test]
    fn an_instance_stands_at_its_cells_centre_scaled_and_turned() {
        let rules = rules();
        let space = space();
        let chunk = chunk(&rules);

        let set = &instance_sets(&chunk, &rules, &space, |name| name != "air")[0];

        let t = &set.transforms(space.cell_size());
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
    fn at_a_scale_of_one_an_instance_is_only_turned_and_placed() {
        let rules = rules();
        let space = space();
        let chunk = chunk(&rules);

        let set = &instance_sets(&chunk, &rules, &space, |name| name != "air")[0];
        let t = set.transforms([1.0; 3]);

        // Each row of the basis is a unit vector: a collider keeps the size it was shaped with.
        for row in [0, 4, 8] {
            let length =
                (t[row] * t[row] + t[row + 1] * t[row + 1] + t[row + 2] * t[row + 2]).sqrt();
            assert!((length - 1.0).abs() < 1e-6, "row {row}: {length}");
        }
        assert_eq!([t[3], t[7], t[11]], space.cell_center(chunk.coord, 0));
    }

    /// A world three chunks along x of 4x1x1 cells one unit wide, every cell an unturned road.
    struct Strip {
        rules: RuleFile,
        space: YUpSpace,
        extent: WorldExtent,
        chunks: Vec<Chunk>,
    }

    impl Strip {
        fn new() -> Self {
            let rules = rules();
            let road = rules
                .tiles_named("road")
                .into_iter()
                .find(|&tile| rules.rotation(tile) == 0)
                .expect("an unturned road");
            let shape = ChunkShape { x: 4, y: 1, z: 1 };
            let chunks = (0..3)
                .map(|x| Chunk {
                    coord: ChunkCoord::new(x, 0, 0),
                    tiles: vec![road as u16; 4].into_boxed_slice(),
                    version: 1,
                })
                .collect();
            Self {
                rules,
                space: YUpSpace::new(shape, [1.0; 3]),
                extent: WorldExtent::new(shape)
                    .with_x(0..3)
                    .with_y(0..1)
                    .with_z(0..1),
                chunks,
            }
        }

        fn source(&self, x: i32, border: f32) -> Result<NavSource, NavSourceError> {
            nav_source(
                ChunkCoord::new(x, 0, 0),
                |coord| self.chunks.iter().find(|chunk| chunk.coord == coord),
                &self.extent,
                &self.rules,
                &self.space,
                |name| (name == "road").then_some(&ROAD_TOP[..]),
                border,
            )
        }
    }

    /// One triangle on top of a road's cell, from its centre along +x and +z.
    const ROAD_TOP: [[f32; 3]; 3] = [[0.0, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5]];

    #[test]
    fn a_nav_source_holds_the_chunk_and_its_neighbours_out_to_the_border_and_a_cell() {
        let strip = Strip::new();

        let source = strip.source(1, 0.25).expect("every chunk is there");

        // The middle chunk spans x 4 to 8; its bounds 3.75 to 8.25; a cell further, the centres
        // 3.5 and 8.5 of the neighbours' nearest cells are in, and 2.5 and 9.5 are out.
        let mut centres: Vec<f32> = source.triangles.chunks(9).map(|t| t[0]).collect();
        centres.sort_by(f32::total_cmp);
        assert_eq!(centres, vec![3.5, 4.5, 5.5, 6.5, 7.5, 8.5]);
        assert_eq!(
            &source.triangles[..9],
            &[3.5, 1.0, 0.5, 4.0, 1.0, 0.5, 3.5, 1.0, 1.0]
        );
    }

    #[test]
    fn the_bounds_are_the_chunk_grown_by_the_border_along_the_ground() {
        let strip = Strip::new();

        let source = strip.source(1, 0.25).expect("every chunk is there");

        assert_eq!(source.bounds_origin, [3.75, -1.0, -0.25]);
        assert_eq!(source.bounds_size, [4.5, 3.0, 1.5]);
        assert_eq!(source.border, 0.25);
    }

    #[test]
    fn a_neighbour_not_generated_yet_is_named() {
        let mut strip = Strip::new();
        strip.chunks.retain(|chunk| chunk.coord.x != 2);

        let result = strip.source(1, 0.25);

        assert_eq!(
            result,
            Err(NavSourceError::Missing(ChunkCoord::new(2, 0, 0)))
        );
    }

    #[test]
    fn a_chunk_on_the_worlds_edge_needs_no_neighbour_beyond_it() {
        let mut strip = Strip::new();
        strip.chunks.retain(|chunk| chunk.coord.x != 2);

        let source = strip
            .source(0, 0.25)
            .expect("chunk 1 is its only neighbour");

        assert_eq!(source.triangles.len(), 9 * 5);
    }

    #[test]
    fn a_world_more_than_one_chunk_tall_is_refused() {
        let mut strip = Strip::new();
        strip.extent = strip.extent.clone().with_z(0..2);

        assert_eq!(strip.source(1, 0.25), Err(NavSourceError::TallWorld));
    }

    #[test]
    fn a_border_wider_than_a_chunk_is_refused() {
        let strip = Strip::new();

        // The chunks are 4 along x but one cell, 1.0, along z.
        let result = strip.source(1, 1.5);

        assert_eq!(
            result,
            Err(NavSourceError::BorderTooWide {
                border: 1.5,
                chunk: 1.0
            })
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

        let id = set.ids[0];
        assert_eq!(set.ids.len(), 1);
        assert_eq!(id.chunk, chunk.coord);
        assert_eq!(
            (id.stage(), id.cell(), id.slot()),
            (InstanceId::TILES, 1, 0)
        );
    }

    #[test]
    fn an_instance_ids_parts_come_back_as_given_at_their_widest() {
        let chunk = ChunkCoord::new(-7, 3, 1);

        let id = InstanceId::new(chunk, InstanceId::MAX_STAGE, u32::MAX, u16::MAX);

        assert_eq!(
            (id.chunk, id.stage(), id.cell(), id.slot()),
            (chunk, InstanceId::MAX_STAGE, u32::MAX, u16::MAX)
        );
        assert!(
            i64::try_from(id.local).is_ok(),
            "fits a signed engine integer"
        );
    }

    #[test]
    fn instance_ids_differ_whenever_one_part_does() {
        let chunk = ChunkCoord::new(0, 0, 0);
        let base = InstanceId::new(chunk, 1, 2, 3);

        let others = [
            InstanceId::new(ChunkCoord::new(1, 0, 0), 1, 2, 3),
            InstanceId::new(chunk, 2, 2, 3),
            InstanceId::new(chunk, 1, 3, 3),
            InstanceId::new(chunk, 1, 2, 4),
        ];

        assert!(others.iter().all(|&other| other != base));
    }
}
