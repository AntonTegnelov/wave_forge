//! A world generated from a pack of stages (docs/reference/bevy.md), in Bevy.
//!
//! [`WaveForgeStagesPlugin`] runs the stages on a thread of their own through
//! [`wave_forge::stages::StageWorker`], asks for the chunks around every [`GenerationFocus`], and
//! sends a [`StageReady`] or [`StageDropped`] message for each product that arrives or goes. A
//! game reads the products from the [`WaveForgeStages`] resource. The runtime, and the town solver
//! with any GPU device it needs, are built on that thread, so a town solver gets a device of its
//! own there rather than Bevy's.
//!
//! A game binds a kind (a Scatter point's kind, or an Assemble piece's name) to what its entities
//! hold through [`StagePlacements`]: every point or piece of a bound kind gets an entity at its
//! transform with a [`Placed`] component, announced by [`InstanceSpawned`] and despawned with its
//! chunk.
//!
//! With [`WaveForgeStagesPlugin::with_ground`], the plugin also builds each chunk's ground from a
//! height field stage once the fields around it have arrived, and says so with [`GroundReady`]: a
//! [`GroundMesh`] a game turns into a [`Mesh`] with [`ground_mesh`] and hands its heights to the
//! height-field collider of its physics crate.

use crate::GenerationFocus;
use crate::levels::{LevelDetail, bounds_radius, level_ranges, visibility};
use bevy_app::{App, Plugin, Update};
use bevy_asset::RenderAssetUsages;
use bevy_camera::visibility::VisibilityRange;
use bevy_ecs::message::{Message, MessageReader, MessageWriter};
use bevy_ecs::prelude::{
    Commands, Component, Entity, IntoScheduleConfigs, Query, ResMut, Resource,
};
use bevy_ecs::system::{EntityCommands, SystemParam};
use bevy_math::{Mat3, Quat, Vec3};
use bevy_mesh::{Indices, Mesh, PrimitiveTopology};
use bevy_transform::components::{GlobalTransform, Transform};
use std::collections::{BTreeSet, HashMap};
use std::sync::Mutex;
use wave_forge::noise::NoiseConfig;
use wave_forge::stages::regions::Curve;
use wave_forge::stages::{
    Categories, Edits, Facts, Field, Point, RowId, Runtime, Save, Site, StageEvent, StageTiming,
    StageWorker, Stamp, TownChunk,
};
use wave_forge::{
    ChunkCoord, FarGround, FocusPoint, GroundMesh, InstanceId, far_ground, ground,
    ground_materials, ground_readers,
};

/// A stage's product for a chunk is ready to read from [`WaveForgeStages`].
#[derive(Message, Clone, Debug, PartialEq, Eq)]
pub struct StageReady {
    pub stage: String,
    pub chunk: ChunkCoord,
}

/// A stage's product for a chunk is no longer needed and was dropped.
#[derive(Message, Clone, Debug, PartialEq, Eq)]
pub struct StageDropped {
    pub stage: String,
    pub chunk: ChunkCoord,
}

/// The save [`WaveForgeStages::request_save`] asked for, to write to disk with [`Save::to_ron`].
#[derive(Message, Clone, Debug, PartialEq)]
pub struct StagesSaved(pub Save);

/// Generation stopped, and why.
#[derive(Message, Clone, Debug, PartialEq, Eq)]
pub struct StagesFailed(pub String);

/// What a game gives the entities of a kind, a Scatter point's kind or an Assemble piece's name:
/// a `SceneRoot` of a glTF scene, a mesh and a material, a collider, anything.
type Spawn = Box<dyn Fn(&mut EntityCommands) + Send + Sync>;

/// Kinds bound to what their entities hold, and the entities each stage's chunk has.
///
/// Insert it, and bind kinds with [`StagePlacements::bind`], before the stages it binds arrive:
/// a chunk is placed as it arrives.
#[derive(Resource, Default)]
pub struct StagePlacements {
    bound: HashMap<String, Spawn>,
    placed: HashMap<(String, ChunkCoord), Vec<Entity>>,
}

impl StagePlacements {
    /// Gives every entity of `kind` what `spawn` inserts, beside its [`Transform`] and [`Placed`].
    #[must_use]
    pub fn bind(
        mut self,
        kind: &str,
        spawn: impl Fn(&mut EntityCommands) + Send + Sync + 'static,
    ) -> Self {
        self.bound.insert(kind.to_owned(), Box::new(spawn));
        self
    }
}

/// An entity placed for a point or a piece: its stage and chunk, and its id, as the stage's
/// products give it.
#[derive(Component, Clone, Debug, PartialEq, Eq)]
pub struct Placed {
    pub stage: String,
    pub chunk: ChunkCoord,
    pub id: InstanceId,
}

/// An entity was placed for a point or a piece of a bound kind.
#[derive(Message, Clone, Debug, PartialEq, Eq)]
pub struct InstanceSpawned {
    pub entity: Entity,
    pub placed: Placed,
}

/// A chunk's ground is ready to read from [`WaveForgeStages::ground`].
#[derive(Message, Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroundReady(pub ChunkCoord);

/// A chunk's height field was dropped, and its ground with it.
#[derive(Message, Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroundDropped(pub ChunkCoord);

/// A chunk of the far ground's coarse stage has its far ground, new or built again, to read from
/// [`WaveForgeStages::far_ground`].
#[derive(Message, Clone, Copy, Debug, PartialEq, Eq)]
pub struct FarGroundReady(pub ChunkCoord);

/// A chunk of the far ground's coarse stage was dropped, and its far ground with it.
#[derive(Message, Clone, Copy, Debug, PartialEq, Eq)]
pub struct FarGroundDropped(pub ChunkCoord);

/// How chunks and cells sit in Bevy's world.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StagesSettings {
    /// Columns per chunk along the lattice's x and y, as the runtime was built with.
    pub chunk: [u32; 2],
    /// One cell's size along Bevy's x, y and z; a field's value is a height in cells.
    pub cell_size: Vec3,
}

/// The stages' products, as of the last frame.
#[derive(Resource)]
pub struct WaveForgeStages {
    worker: StageWorker,
    /// Each target and a radius of its own, if it has one.
    targets: Vec<(String, Option<u32>)>,
    settings: StagesSettings,
    asked: Vec<FocusPoint>,
    /// The field stage the ground is built from, if any.
    ground_stage: Option<String>,
    grounds: HashMap<ChunkCoord, GroundMesh>,
    /// The Rules or Area stage whose categories are the ground's materials, if any.
    ground_material_stage: Option<String>,
    /// Each built ground's category per vertex, with a material stage.
    ground_ids: HashMap<ChunkCoord, Vec<u8>>,
    /// The coarse field stage the far ground is built from and its scale, if any.
    far_ground_stage: Option<(String, u32)>,
    far_grounds: HashMap<ChunkCoord, FarGround>,
    /// Coarse chunks whose far ground may have to be built again.
    far_due: BTreeSet<ChunkCoord>,
}

impl WaveForgeStages {
    /// A field stage's values for a chunk, if it has arrived.
    #[must_use]
    pub fn field(&self, stage: &str, chunk: ChunkCoord) -> Option<&Field> {
        self.worker.field(stage, chunk)
    }

    /// A Rules stage's categories for a chunk, if they have arrived: indices into the categories
    /// the stage names ([`wave_forge::stages::StageKind::categories`]).
    #[must_use]
    pub fn categories(&self, stage: &str, chunk: ChunkCoord) -> Option<&Categories> {
        self.worker.categories(stage, chunk)
    }

    /// A Region stage's curves that pass through a chunk, if they have arrived, in the lattice's
    /// columns: the lattice's x and y are Bevy's x and z.
    #[must_use]
    pub fn curves(&self, stage: &str, chunk: ChunkCoord) -> Option<&[Curve]> {
        self.worker.curves(stage, chunk)
    }

    /// A Sites stage's sites overlapping a chunk, if they have arrived.
    #[must_use]
    pub fn sites(&self, stage: &str, chunk: ChunkCoord) -> Option<&[Site]> {
        self.worker.sites(stage, chunk)
    }

    /// A Solve stage's town in a chunk, if it lies in a site and has arrived.
    #[must_use]
    pub fn tiles(&self, stage: &str, chunk: ChunkCoord) -> Option<&TownChunk> {
        self.worker.tiles(stage, chunk)
    }

    /// A Scatter stage's points in a chunk, if they have arrived.
    #[must_use]
    pub fn points(&self, stage: &str, chunk: ChunkCoord) -> Option<&[Point]> {
        self.worker.points(stage, chunk)
    }

    /// An Assemble stage's pieces overlapping a chunk, if they have arrived.
    #[must_use]
    pub fn stamps(&self, stage: &str, chunk: ChunkCoord) -> Option<&[Stamp]> {
        self.worker.stamps(stage, chunk)
    }

    /// Where an assembled piece stands in Bevy's world: at the centre of its footprint on its
    /// floor, turned about +Y, where a scene of the piece authored at turn 0 with its footprint
    /// centred on its origin goes.
    #[must_use]
    pub fn stamp_transform(&self, stamp: &Stamp) -> Transform {
        let rows = stamp.y_up_basis();
        let column = |c: usize| Vec3::new(rows[0][c], rows[1][c], rows[2][c]);
        let [x, y, floor] = stamp.position;
        let cell = self.settings.cell_size;
        Transform {
            translation: Vec3::new(x * cell.x, floor * cell.y, y * cell.z),
            rotation: Quat::from_mat3(&Mat3::from_cols(column(0), column(1), column(2))),
            scale: Vec3::ONE,
        }
    }

    /// Where a point stands in Bevy's world: the lattice's x and y across Bevy's x and z, its
    /// height up Bevy's y.
    #[must_use]
    pub fn translation_of(&self, point: &Point) -> Vec3 {
        let [x, y, height] = point.position;
        let cell = self.settings.cell_size;
        Vec3::new(x * cell.x, height * cell.y, y * cell.z)
    }

    /// How a point stands in Bevy's world: where [`WaveForgeStages::translation_of`] puts it,
    /// turned, leant and scaled as its stage made it.
    #[must_use]
    pub fn transform_of(&self, point: &Point) -> Transform {
        let rows = point.y_up_basis();
        let column = |c: usize| Vec3::new(rows[0][c], rows[1][c], rows[2][c]) / point.scale;
        Transform {
            translation: self.translation_of(point),
            rotation: Quat::from_mat3(&Mat3::from_cols(column(0), column(1), column(2))),
            scale: Vec3::splat(point.scale),
        }
    }

    /// A chunk's ground, once its height field and the eight around it have arrived: a mesh in
    /// Bevy's axes relative to [`WaveForgeStages::chunk_corner`], and the same heights as a grid.
    #[must_use]
    pub fn ground(&self, chunk: ChunkCoord) -> Option<&GroundMesh> {
        self.grounds.get(&chunk)
    }

    /// A chunk of the coarse stage's far ground, if it is built: drawn at
    /// [`WaveForgeStages::far_ground_corner`] with [`far_ground_mesh`], and left out where a chunk
    /// has ground of its own ([`wave_forge::far_ground`]).
    #[must_use]
    pub fn far_ground(&self, chunk: ChunkCoord) -> Option<&FarGround> {
        self.far_grounds.get(&chunk)
    }

    /// Where the far ground of a chunk of the coarse stage goes in Bevy's world: the corner of the
    /// first chunk of the lattice it covers.
    ///
    /// # Panics
    /// If the plugin draws no far ground.
    #[must_use]
    pub fn far_ground_corner(&self, chunk: ChunkCoord) -> Vec3 {
        let (_, scale) = self
            .far_ground_stage
            .as_ref()
            .expect("the plugin draws a far ground");
        let scale = *scale as i32;
        self.chunk_corner(ChunkCoord::new(chunk.x * scale, chunk.y * scale, 0))
    }

    /// The height of the ground at a translation in Bevy's world, where its mesh at full detail
    /// stands above that point of the ground plane ([`wave_forge::ground_height`]): what a game
    /// stands a player or an object on. `None` until the fields of the ground's stage around it
    /// have arrived, or without a ground.
    #[must_use]
    pub fn ground_height(&self, translation: Vec3) -> Option<f32> {
        let stage = self.ground_stage.as_ref()?;
        wave_forge::ground_height(
            [translation.x, translation.z],
            self.settings.chunk,
            |at| self.worker.field(stage, at),
            self.settings.cell_size.to_array(),
        )
    }

    /// The category of every vertex of a chunk's ground, in the order of its positions, from the
    /// stage [`WaveForgeStagesPlugin::with_ground_materials`] names; `None` without one or before
    /// the chunk's ground is built.
    #[must_use]
    pub fn ground_materials(&self, chunk: ChunkCoord) -> Option<&[u8]> {
        self.ground_ids.get(&chunk).map(Vec::as_slice)
    }

    /// How chunks and cells sit in Bevy's world, as the plugin was given it.
    #[must_use]
    pub const fn settings(&self) -> StagesSettings {
        self.settings
    }

    /// Where a chunk's corner sits on Bevy's ground plane, which a chunk's ground is relative to.
    #[must_use]
    pub fn chunk_corner(&self, chunk: ChunkCoord) -> Vec3 {
        let (settings, cell) = (self.settings, self.settings.cell_size);
        Vec3::new(
            chunk.x as f32 * settings.chunk[0] as f32 * cell.x,
            0.0,
            chunk.y as f32 * settings.chunk[1] as f32 * cell.z,
        )
    }

    /// What each stage has cost on the stages' thread, as of the last frame, in the order the pack
    /// lists them.
    #[must_use]
    pub fn timings(&self) -> &[(String, StageTiming)] {
        self.worker.timings()
    }

    /// Why generation stopped, if it did.
    #[must_use]
    pub fn failure(&self) -> Option<&str> {
        self.worker.failure()
    }

    /// Gives the stages new tables of facts. A game keeps its own [`Facts`], gives it rows, and
    /// hands a copy here; what the change makes stale arrives as [`StageDropped`], then as
    /// [`StageReady`] again. An error stops generation and arrives as [`StagesFailed`].
    pub fn set_facts(&self, facts: Facts) {
        self.worker.set_facts(facts);
    }

    /// Gives the stages the player's edits, a log a game keeps and saves: what the change reaches
    /// arrives as [`StageDropped`], then as [`StageReady`] again with the edits applied. An error
    /// stops generation and arrives as [`StagesFailed`].
    pub fn set_edits(&self, edits: Edits) {
        self.worker.set_edits(edits);
    }

    /// Asks for a save of the world: the edits a game made, less those of ephemeral stages, and
    /// the frozen stages' chunks as first generated, stamped with the generator's version and the
    /// pack's digest. It arrives as [`StagesSaved`] in a later frame.
    pub fn request_save(&self) {
        self.worker.request_save();
    }

    /// Brings the world back from `save`: its edits, and the frozen stages' chunks as they were
    /// first generated, even under a changed pack. What it changes arrives as [`StageDropped`],
    /// then as [`StageReady`] again. An edit the pack refuses stops generation and arrives as
    /// [`StagesFailed`]. A game that calls [`set_edits`] afterwards hands it the save's edits with
    /// its own after them.
    ///
    /// [`set_edits`]: WaveForgeStages::set_edits
    pub fn load(&self, save: Save) {
        self.worker.load(save);
    }

    /// Focuses the stages on the row `id` of `table`, whose columns stages read through
    /// [`wave_forge::stages::Expr::Row`], with what that makes stale arriving as [`set_facts`]
    /// says.
    ///
    /// [`set_facts`]: WaveForgeStages::set_facts
    pub fn focus(&self, table: &str, id: RowId) {
        self.worker.focus(table, id);
    }
}

/// A chunk's ground as a Bevy mesh at full detail: positions, normals and the first level's
/// triangles, the surface facing up and the skirt out.
#[must_use]
pub fn ground_mesh(ground: &GroundMesh) -> Mesh {
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, ground.positions.clone())
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, ground.normals.clone())
    .with_inserted_indices(Indices::U32(ground.levels[0].indices.clone()))
}

/// A coarse chunk's far ground as a Bevy mesh, to spawn at
/// [`WaveForgeStages::far_ground_corner`]: the surface facing up and its walls both ways.
#[must_use]
pub fn far_ground_mesh(far: &FarGround) -> Mesh {
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, far.positions.clone())
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, far.normals.clone())
    .with_inserted_indices(Indices::U32(far.indices.clone()))
}

/// One level of detail of a chunk's ground, ready to spawn at the chunk's corner.
pub struct GroundLevelMesh {
    /// The level's [`wave_forge::GroundLevel::step`].
    pub step: u32,
    /// The full mesh's positions and normals with the level's triangles, skirt included.
    pub mesh: Mesh,
    /// The distances from the camera the level is drawn at, from the centre of the mesh's bounds.
    pub range: VisibilityRange,
}

/// A chunk's ground as one mesh per level of detail, each with the distances from the camera it is
/// drawn at: a coarser level once its error, seen from anywhere in the chunk, spans at most
/// `detail.pixels`. The skirts let neighbours at different levels meet without a gap. A level no
/// distance would draw, because a coarser one strays as little, is left out.
#[must_use]
pub fn ground_levels(ground: &GroundMesh, detail: LevelDetail) -> Vec<GroundLevelMesh> {
    let errors: Vec<f32> = ground.levels.iter().map(|level| level.error).collect();
    let ranges = level_ranges(
        &errors,
        bounds_radius(&ground.positions),
        detail.per_error(),
        0.0,
    );
    ranges
        .iter()
        .zip(&ground.levels)
        .filter(|(range, _)| range.start < range.end)
        .map(|(range, level)| {
            let mesh = Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::default(),
            )
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, ground.positions.clone())
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, ground.normals.clone())
            .with_inserted_indices(Indices::U32(level.indices.clone()));
            GroundLevelMesh {
                step: level.step,
                mesh,
                range: visibility(range),
            }
        })
        .collect()
}

type Build = Box<dyn FnOnce() -> Result<Runtime, String> + Send>;

/// Generates the `targets` stages of a pack around every [`GenerationFocus`].
pub struct WaveForgeStagesPlugin {
    build: Mutex<Option<Build>>,
    /// Each target and a radius of its own, if it has one.
    targets: Vec<(String, Option<u32>)>,
    settings: StagesSettings,
    ground_stage: Option<String>,
    ground_material_stage: Option<String>,
    far_ground_stage: Option<(String, u32)>,
}

impl WaveForgeStagesPlugin {
    /// A plugin whose runtime `build` makes, on the stages' own thread.
    #[must_use]
    pub fn new(
        targets: &[&str],
        settings: StagesSettings,
        build: impl FnOnce() -> Result<Runtime, String> + Send + 'static,
    ) -> Self {
        Self {
            build: Mutex::new(Some(Box::new(build))),
            targets: targets
                .iter()
                .map(|target| ((*target).to_owned(), None))
                .collect(),
            settings,
            ground_stage: None,
            ground_material_stage: None,
            far_ground_stage: None,
        }
    }

    /// Generates the target `stage` within `radius` chunks of every focus, not the focus's own
    /// radius: ground far out, locations nearer and clutter nearest, say.
    ///
    /// # Panics
    /// If `stage` is not one of the plugin's targets.
    #[must_use]
    pub fn with_radius(mut self, stage: &str, radius: u32) -> Self {
        let target = self
            .targets
            .iter_mut()
            .find(|(target, _)| target == stage)
            .unwrap_or_else(|| panic!("{stage:?} is not one of the plugin's targets"));
        target.1 = Some(radius);
        self
    }

    /// Builds each chunk's ground from the field stage `stage`, a height in cells per column,
    /// and announces it with [`GroundReady`]. The stage has to be generated, as a target or as
    /// what a target reads; a chunk's ground needs the fields around it, so it reaches one chunk
    /// less than the fields do.
    #[must_use]
    pub fn with_ground(mut self, stage: &str) -> Self {
        self.ground_stage = Some(stage.to_owned());
        self
    }

    /// Builds a far ground beyond the ground from the coarse field stage `stage` of `scale`
    /// (the pack's [`wave_forge::stages::Pack::scale`]), one mesh per chunk of it, and announces
    /// each with [`FarGroundReady`], again whenever ground comes or goes on or beside it. Give the
    /// stage a radius of its own with [`WaveForgeStagesPlugin::with_radius`], as far as the ground
    /// should reach; a coarse chunk's far ground needs the fields around it, so it reaches one
    /// coarse chunk less.
    #[must_use]
    pub fn with_far_ground(mut self, stage: &str, scale: u32) -> Self {
        self.far_ground_stage = Some((stage.to_owned(), scale));
        self
    }

    /// Gives the ground the categories of the Rules or Area stage `stage` as materials: a chunk's
    /// ground then also waits for them, and [`WaveForgeStages::ground_materials`] gives them per
    /// vertex, for [`crate::materials::ground_material`]. The stage has to be generated too, one
    /// chunk beyond the ground, since a chunk's ground reads the materials beyond its far edges.
    #[must_use]
    pub fn with_ground_materials(mut self, stage: &str) -> Self {
        self.ground_material_stage = Some(stage.to_owned());
        self
    }
}

/// The set both of the plugin's systems run in, so a game can order its own work around them.
#[derive(bevy_ecs::schedule::SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WaveForgeStagesSystems;

impl Plugin for WaveForgeStagesPlugin {
    fn build(&self, app: &mut App) {
        let build = self
            .build
            .lock()
            .expect("a plugin is built by one thread")
            .take()
            .expect("a plugin is built once");
        app.insert_resource(WaveForgeStages {
            worker: StageWorker::spawn(build),
            targets: self.targets.clone(),
            settings: self.settings,
            asked: Vec::new(),
            ground_stage: self.ground_stage.clone(),
            grounds: HashMap::new(),
            ground_material_stage: self.ground_material_stage.clone(),
            ground_ids: HashMap::new(),
            far_ground_stage: self.far_ground_stage.clone(),
            far_grounds: HashMap::new(),
            far_due: BTreeSet::new(),
        })
        .add_message::<StageReady>()
        .add_message::<StageDropped>()
        .add_message::<StagesFailed>()
        .add_message::<StagesSaved>()
        .add_message::<GroundReady>()
        .add_message::<InstanceSpawned>()
        .add_message::<GroundDropped>()
        .add_message::<FarGroundReady>()
        .add_message::<FarGroundDropped>()
        .register_type::<NoiseConfig>()
        .add_systems(
            Update,
            (follow_focus, drain, place)
                .chain()
                .in_set(WaveForgeStagesSystems),
        );
    }
}

/// Asks for the chunks around every focus when they change.
fn follow_focus(
    mut stages: ResMut<WaveForgeStages>,
    focus: Query<(&GlobalTransform, &GenerationFocus)>,
) {
    let settings = stages.settings;
    let wanted: Vec<FocusPoint> = focus
        .iter()
        .map(|(at, focus)| {
            let place = at.translation();
            let chunk = ChunkCoord::new(
                (place.x / (settings.chunk[0] as f32 * settings.cell_size.x)).floor() as i32,
                (place.z / (settings.chunk[1] as f32 * settings.cell_size.z)).floor() as i32,
                0,
            );
            FocusPoint::new(chunk, focus.radius)
        })
        .collect();
    if wanted == stages.asked {
        return;
    }
    let targets: Vec<(&str, Option<u32>)> = stages
        .targets
        .iter()
        .map(|(target, radius)| (target.as_str(), *radius))
        .collect();
    stages.worker.request_each(&wanted, &targets);
    stages.asked = wanted;
}

/// What [`drain`] says about the ground and the far ground.
#[derive(SystemParam)]
struct GroundWriters<'w> {
    ready: MessageWriter<'w, GroundReady>,
    dropped: MessageWriter<'w, GroundDropped>,
    far_ready: MessageWriter<'w, FarGroundReady>,
    far_dropped: MessageWriter<'w, FarGroundDropped>,
}

/// Takes what the stages' thread finished, as messages, and builds the ground it completed.
fn drain(
    mut stages: ResMut<WaveForgeStages>,
    mut ready: MessageWriter<StageReady>,
    mut dropped: MessageWriter<StageDropped>,
    mut failed: MessageWriter<StagesFailed>,
    mut saved: MessageWriter<StagesSaved>,
    mut grounds: GroundWriters,
) {
    let had_failed = stages.worker.failure().is_some();
    let ground_stage = stages.ground_stage.clone();
    let material_stage = stages.ground_material_stage.clone();
    let far_stage = stages.far_ground_stage.clone();
    let mut arrived = Vec::new();
    // Chunks whose ground came or went, which changes the far ground over and beside them.
    let mut near_changed = Vec::new();
    for event in stages.worker.drain() {
        match event {
            StageEvent::Generated { stage, chunk } => {
                if far_stage.as_ref().is_some_and(|(far, _)| *far == stage) {
                    for (dx, dy) in (-1..=1).flat_map(|dy| (-1..=1).map(move |dx| (dx, dy))) {
                        stages
                            .far_due
                            .insert(ChunkCoord::new(chunk.x + dx, chunk.y + dy, 0));
                    }
                }
                // A chunk's ground reads the materials of itself and of the chunks beyond its far
                // edges, which are among the chunks whose ground reads a field of this chunk.
                if ground_stage.as_ref() == Some(&stage) || material_stage.as_ref() == Some(&stage)
                {
                    arrived.push(chunk);
                }
                ready.write(StageReady { stage, chunk });
            }
            StageEvent::Dropped { stage, chunk } => {
                if ground_stage.as_ref() == Some(&stage) && stages.grounds.remove(&chunk).is_some()
                {
                    stages.ground_ids.remove(&chunk);
                    near_changed.push(chunk);
                    grounds.dropped.write(GroundDropped(chunk));
                }
                if far_stage.as_ref().is_some_and(|(far, _)| *far == stage) {
                    stages.far_due.remove(&chunk);
                    if stages.far_grounds.remove(&chunk).is_some() {
                        grounds.far_dropped.write(FarGroundDropped(chunk));
                    }
                }
                dropped.write(StageDropped { stage, chunk });
            }
            StageEvent::Saved => {
                let save = stages
                    .worker
                    .take_save()
                    .expect("a Saved event carries a save");
                saved.write(StagesSaved(save));
            }
        }
    }
    if let Some(stage) = &ground_stage {
        let cell = stages.settings.cell_size.to_array();
        for chunk in arrived.into_iter().flat_map(ground_readers) {
            if stages.grounds.contains_key(&chunk) {
                continue;
            }
            let Some(mesh) = ground(chunk, |at| stages.worker.field(stage, at), cell) else {
                continue;
            };
            if let Some(materials) = &material_stage {
                let Some(ids) =
                    ground_materials(chunk, |at| stages.worker.categories(materials, at))
                else {
                    continue;
                };
                stages.ground_ids.insert(chunk, ids);
            }
            stages.grounds.insert(chunk, mesh);
            near_changed.push(chunk);
            grounds.ready.write(GroundReady(chunk));
        }
    }
    if let Some((stage, scale)) = &far_stage {
        let scale = *scale as i32;
        for fine in near_changed {
            for (dx, dy) in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)] {
                stages.far_due.insert(ChunkCoord::new(
                    (fine.x + dx).div_euclid(scale),
                    (fine.y + dy).div_euclid(scale),
                    0,
                ));
            }
        }
        let cell = stages.settings.cell_size.to_array();
        let due: Vec<ChunkCoord> = std::mem::take(&mut stages.far_due).into_iter().collect();
        for chunk in due {
            // Waits for a field around it, whose arrival makes it due again.
            let Some(far) = far_ground(
                chunk,
                scale as u32,
                |at| stages.worker.field(stage, at),
                cell,
                |fine| stages.grounds.get(&fine),
            ) else {
                continue;
            };
            stages.far_grounds.insert(chunk, far);
            grounds.far_ready.write(FarGroundReady(chunk));
        }
    }
    if let (false, Some(reason)) = (had_failed, stages.worker.failure()) {
        failed.write(StagesFailed(reason.to_owned()));
    }
}

/// Spawns an entity for every point or piece of a bound kind in each chunk that arrived, and
/// despawns those of each chunk dropped. A piece overlapping several chunks is placed once, by the
/// chunk holding its footprint's centre.
fn place(
    mut commands: Commands,
    stages: ResMut<WaveForgeStages>,
    placements: Option<ResMut<StagePlacements>>,
    mut ready: MessageReader<StageReady>,
    mut dropped: MessageReader<StageDropped>,
    mut spawned: MessageWriter<InstanceSpawned>,
) {
    let Some(mut placements) = placements else {
        ready.clear();
        dropped.clear();
        return;
    };
    for StageDropped { stage, chunk } in dropped.read() {
        for entity in placements
            .placed
            .remove(&(stage.clone(), *chunk))
            .into_iter()
            .flatten()
        {
            commands.entity(entity).despawn();
        }
    }
    for StageReady { stage, chunk } in ready.read() {
        let items: Vec<(&str, Transform, InstanceId)> =
            if let Some(points) = stages.points(stage, *chunk) {
                points
                    .iter()
                    .map(|point| (&*point.kind, stages.transform_of(point), point.id))
                    .collect()
            } else if let Some(stamps) = stages.stamps(stage, *chunk) {
                stamps
                    .iter()
                    .filter(|stamp| stamp.id.chunk == *chunk)
                    .map(|stamp| (&*stamp.piece, stages.stamp_transform(stamp), stamp.id))
                    .collect()
            } else {
                continue;
            };
        let mut entities = Vec::new();
        for (kind, transform, id) in items {
            let Some(spawn) = placements.bound.get(kind) else {
                continue;
            };
            let placed = Placed {
                stage: stage.clone(),
                chunk: *chunk,
                id,
            };
            let mut entity = commands.spawn((transform, placed.clone()));
            spawn(&mut entity);
            let entity = entity.id();
            spawned.write(InstanceSpawned { entity, placed });
            entities.push(entity);
        }
        if let Some(old) = placements.placed.insert((stage.clone(), *chunk), entities) {
            for entity in old {
                commands.entity(entity).despawn();
            }
        }
    }
}
