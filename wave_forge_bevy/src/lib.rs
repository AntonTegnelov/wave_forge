//! Wave Forge as a Bevy plugin: a world generated in chunks around whatever the game marks as its
//! focus, on the GPU device Bevy already owns.
//!
//! ```no_run
//! use bevy_app::prelude::*;
//! use bevy_ecs::prelude::*;
//! use wave_forge_bevy::{GenerationFocus, WaveForgePlugin, WaveForgeSettings};
//! # use wave_forge::{Prior, Ruleset};
//! # fn plugin(ruleset: Ruleset, prior: Prior) -> WaveForgePlugin {
//! WaveForgePlugin::new(ruleset, prior, WaveForgeSettings::default())
//! # }
//! ```
//!
//! A game adds the plugin after its render plugin, marks an entity (usually the camera or the
//! player) with [`GenerationFocus`], and reads [`ChunkUpdated`] messages to build meshes from
//! [`WaveForgeWorld::chunk`]. One batch of chunks is started per frame and collected without ever
//! blocking, so generation costs a dispatch and a readback rather than a stall.
//!
//! # Sharing Bevy's device
//!
//! [`WaveForgePlugin::new`] takes Bevy's `RenderDevice` and `RenderQueue`, so generation runs on the
//! same device as rendering: no second adapter, no second allocator. That only compiles when the
//! plugin and Bevy agree on a wgpu version, which is why the plugin tracks the Bevy release that
//! uses the same one. [`WaveForgePlugin::on_own_device`] is the way out when they do not, and
//! [`WaveForgeSolverPlugin`] takes a solver of your own.
//!
//! # Coordinates
//!
//! Bevy is Y-up and the generator is Z-up. Bevy's `xz` ground plane is therefore the lattice's `xy`,
//! and Bevy's `y` is the lattice's `z`; [`WaveForgeSettings::cell_size`] says how large one cell is
//! along each of Bevy's axes.

use bevy_app::{App, Plugin, Update};
use bevy_ecs::error::Result;
use bevy_ecs::message::{Message, MessageWriter};
use bevy_ecs::prelude::{Component, IntoScheduleConfigs, Query, ResMut, Resource};
use bevy_math::{Quat, Vec3};
use bevy_render::renderer::{RenderDevice, RenderQueue};
use bevy_transform::components::GlobalTransform;
use std::sync::Mutex;
use wave_forge::loader::RuleFile;
use wave_forge::{
    BlockSolver, Builder, Chunk, ChunkCoord, ChunkEvent, ChunkShape, ChunkStore, FocusPoint,
    GeneratorStats, ModelError, Prior, RegionStatus, RepairPolicy, Ruleset, Solver, WgpuBackend,
    WorldExtent, WorldGenerator, YUpSpace,
};

/// An entity that generation follows, usually the player or the camera.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct GenerationFocus {
    /// How many chunks out from this entity to keep generated.
    pub radius: u32,
}

impl GenerationFocus {
    /// A focus that keeps `radius` chunks around it generated.
    #[must_use]
    pub const fn new(radius: u32) -> Self {
        Self { radius }
    }
}

/// A chunk's tiles are new or have changed, so anything built from them is stale.
///
/// A repair reports every chunk it rewrote, not only the one it was repairing, so a game that
/// builds meshes from chunks rebuilds each chunk it sees here.
#[derive(Message, Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkUpdated(pub ChunkCoord);

/// A chunk could not be generated. Asking again would fail the same way.
#[derive(Message, Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkFailed {
    pub chunk: ChunkCoord,
    pub status: RegionStatus,
}

/// A chunk has left every focus and was dropped, so anything built from it can go too.
#[derive(Message, Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkEvicted(pub ChunkCoord);

/// How the plugin generates, and how its lattice sits in Bevy's world space.
#[derive(Clone, Debug, PartialEq)]
pub struct WaveForgeSettings {
    /// Every choice in the world derives from this.
    pub seed: u64,
    /// How far the world reaches, in chunks. Its shape is the chunk shape.
    pub extent: WorldExtent,
    /// Cells solved around a chunk and then discarded, so a chunk's borders are completable.
    pub halo: u32,
    /// What to do about a chunk that will not solve.
    pub repair: RepairPolicy,
    /// The size of one cell in Bevy's world units, along Bevy's own axes.
    pub cell_size: Vec3,
    /// Chunks further than a focus's radius plus this many are dropped. `None` keeps everything.
    ///
    /// Keep it at one or more. The set of chunks a focus asks for reaches one chunk beyond its
    /// radius, because a chunk of the second parity is solved against its face neighbours, so a
    /// margin of zero drops chunks that are about to be asked for again.
    pub evict_margin: Option<u32>,
}

impl Default for WaveForgeSettings {
    fn default() -> Self {
        let chunk = ChunkShape::cube(8);
        Self {
            seed: 0,
            extent: WorldExtent::new(chunk).with_z(0..1),
            halo: 1,
            repair: RepairPolicy::default(),
            cell_size: Vec3::ONE,
            evict_margin: None,
        }
    }
}

impl WaveForgeSettings {
    /// The chunk shape, which the extent carries.
    #[must_use]
    pub fn chunk(&self) -> ChunkShape {
        self.extent.shape()
    }

    /// The size of one chunk in Bevy's world units.
    #[must_use]
    pub fn chunk_size(&self) -> Vec3 {
        Vec3::from_array(self.space().chunk_size())
    }

    /// The chunk a point in Bevy's world space falls in.
    #[must_use]
    pub fn chunk_at(&self, translation: Vec3) -> ChunkCoord {
        self.space().chunk_at(translation.to_array())
    }

    /// Where a chunk's lowest corner sits in Bevy's world space.
    #[must_use]
    pub fn translation_of(&self, chunk: ChunkCoord) -> Vec3 {
        Vec3::from_array(self.space().chunk_origin(chunk))
    }

    /// The centre of one cell of a chunk in Bevy's world space. `cell` is an index into the
    /// chunk's tiles.
    ///
    /// # Panics
    /// If `cell` is not a cell of the chunk.
    #[must_use]
    pub fn cell_translation(&self, chunk: ChunkCoord, cell: u32) -> Vec3 {
        Vec3::from_array(self.space().cell_center(chunk, cell))
    }

    /// The lattice in Bevy's Y-up world space.
    fn space(&self) -> YUpSpace {
        YUpSpace::new(self.chunk(), self.cell_size.to_array())
    }
}

/// What each tile of the rule set is, for a game placing one model per tile: its name, its
/// rotation, and the tiles a name or a tag picks out.
///
/// [`WaveForgePlugin::from_rules`] inserts it. It dereferences to the [`RuleFile`] it was built
/// from.
#[derive(Resource, Clone, Debug)]
pub struct WaveForgeTiles(pub RuleFile);

impl WaveForgeTiles {
    /// The rotation to place a tile's model with: its turn from its module, about Bevy's up axis.
    ///
    /// # Panics
    /// If `tile` is not a tile of the rule set.
    #[must_use]
    pub fn rotation_of(&self, tile: usize) -> Quat {
        Quat::from_rotation_y(YUpSpace::yaw(self.0.rotation(tile)))
    }
}

impl std::ops::Deref for WaveForgeTiles {
    type Target = RuleFile;

    fn deref(&self) -> &RuleFile {
        &self.0
    }
}

/// The generated world, as a Bevy resource.
///
/// A game reads chunks from here when a [`ChunkUpdated`] message tells it one has changed.
#[derive(Resource)]
pub struct WaveForgeWorld<S: Solver + Send + Sync + 'static> {
    generator: WorldGenerator<S>,
    settings: WaveForgeSettings,
    /// What was asked for last, so an unchanged focus does not rebuild the request every frame.
    asked: Vec<FocusPoint>,
}

impl<S: Solver + Send + Sync + 'static> WaveForgeWorld<S> {
    /// A generated chunk's tiles, or `None` if it has not been generated.
    #[must_use]
    pub fn chunk(&self, coord: ChunkCoord) -> Option<&Chunk> {
        self.generator.chunk(coord)
    }

    /// Every chunk generated so far.
    #[must_use]
    pub const fn store(&self) -> &ChunkStore {
        self.generator.store()
    }

    /// What generation has cost.
    #[must_use]
    pub const fn stats(&self) -> &GeneratorStats {
        self.generator.stats()
    }

    /// Whether there is nothing in flight and nothing left to start, which is what a loading
    /// screen waits for.
    #[must_use]
    pub fn is_idle(&self) -> bool {
        self.generator.is_idle()
    }

    /// How many chunks the current focus still wants, including the batch in flight.
    #[must_use]
    pub fn pending_chunks(&self) -> usize {
        self.generator.pending_chunks()
    }

    /// How the lattice sits in Bevy's world space, and how the world is configured.
    #[must_use]
    pub const fn settings(&self) -> &WaveForgeSettings {
        &self.settings
    }

    /// The generator, for a game that drives generation itself rather than through the focus
    /// systems: put an entity's [`GenerationFocus`] aside and call `request` and `tick` here.
    pub const fn generator_mut(&mut self) -> &mut WorldGenerator<S> {
        &mut self.generator
    }
}

/// Generates a world in chunks around the entities marked with [`GenerationFocus`], on Bevy's own
/// GPU device.
///
/// Add it after the plugin that creates Bevy's renderer (`DefaultPlugins`, or `RenderPlugin` on its
/// own), because the shared device only exists once that has run.
pub struct WaveForgePlugin {
    ruleset: Ruleset,
    prior: Prior,
    settings: WaveForgeSettings,
    own_device: bool,
    /// The focus radius to compile kernels for while the plugin is built.
    warm: Option<u32>,
    /// The rule file the rule set came from, inserted as [`WaveForgeTiles`].
    tiles: Option<RuleFile>,
}

impl WaveForgePlugin {
    /// A plugin that generates on the device Bevy renders with.
    #[must_use]
    pub fn new(ruleset: Ruleset, prior: Prior, settings: WaveForgeSettings) -> Self {
        Self {
            ruleset,
            prior,
            settings,
            own_device: false,
            warm: None,
            tiles: None,
        }
    }

    /// A plugin for the rule set in `file`, which also inserts [`WaveForgeTiles`] so systems can
    /// ask what each tile is.
    ///
    /// # Errors
    /// If the file's weights do not make a rule set.
    pub fn from_rules(
        file: RuleFile,
        prior: Prior,
        settings: WaveForgeSettings,
    ) -> std::result::Result<Self, ModelError> {
        let ruleset = Ruleset::new(file.rules(), &file.tileset().weights)?;
        Ok(Self {
            tiles: Some(file),
            ..Self::new(ruleset, prior, settings)
        })
    }

    /// Compiles the kernels a run will need while the plugin is built, rather than at the first
    /// dispatch that needs one.
    ///
    /// A kernel is specialised per region shape and per batch size rounded up to a power of two,
    /// and compiling one takes seconds on some drivers, which a game would feel as a freeze in the
    /// middle of play. `radius` is the largest [`GenerationFocus::radius`] the game uses; every
    /// batch such a focus can dispatch, repairs included, is compiled. Without it the first
    /// dispatch of each shape pays for its kernel instead.
    #[must_use]
    pub const fn warm(mut self, radius: u32) -> Self {
        self.warm = Some(radius);
        self
    }

    /// A plugin that asks for a device of its own, for a game whose Bevy build uses a different
    /// wgpu version than this crate, or one that wants generation isolated from rendering.
    #[must_use]
    pub fn on_own_device(ruleset: Ruleset, prior: Prior, settings: WaveForgeSettings) -> Self {
        Self {
            own_device: true,
            ..Self::new(ruleset, prior, settings)
        }
    }
}

impl Plugin for WaveForgePlugin {
    fn build(&self, app: &mut App) {
        register::<BlockSolver<WgpuBackend>>(app);
        if let Some(file) = &self.tiles {
            app.insert_resource(WaveForgeTiles(file.clone()));
        }
    }

    /// Builds the generator once every other plugin has built, which is when Bevy's device exists.
    fn finish(&self, app: &mut App) {
        let builder = builder(&self.ruleset, &self.prior, &self.settings);
        let mut generator = if self.own_device {
            builder.build()
        } else {
            let (device, queue) = bevy_device(app);
            builder.build_on(device, queue)
        }
        .expect("a device for Wave Forge, and a rule set that fits it");
        if let Some(radius) = self.warm {
            let shapes = generator.kernel_shapes(radius);
            generator
                .solver_mut()
                .warm(&shapes)
                .expect("the kernels compile");
        }
        insert(app, generator, self.settings.clone());
    }
}

/// Generates a world on a solver the game built: another engine's compute backend, or the CPU
/// reference in a test.
pub struct WaveForgeSolverPlugin<S: Solver + Send + Sync + 'static> {
    ruleset: Ruleset,
    prior: Prior,
    settings: WaveForgeSettings,
    solver: Mutex<Option<S>>,
}

impl<S: Solver + Send + Sync + 'static> WaveForgeSolverPlugin<S> {
    /// A plugin over `solver`.
    #[must_use]
    pub fn new(ruleset: Ruleset, prior: Prior, settings: WaveForgeSettings, solver: S) -> Self {
        Self {
            ruleset,
            prior,
            settings,
            solver: Mutex::new(Some(solver)),
        }
    }
}

impl<S: Solver + Send + Sync + 'static> Plugin for WaveForgeSolverPlugin<S> {
    fn build(&self, app: &mut App) {
        register::<S>(app);
    }

    fn finish(&self, app: &mut App) {
        let solver = self
            .solver
            .lock()
            .expect("a plugin is not built from two threads")
            .take()
            .expect("a plugin is built once");
        let generator = builder(&self.ruleset, &self.prior, &self.settings).build_with(solver);
        insert(app, generator, self.settings.clone());
    }
}

fn builder(ruleset: &Ruleset, prior: &Prior, settings: &WaveForgeSettings) -> Builder {
    Builder::new(ruleset.clone(), prior.clone())
        .seed(settings.seed)
        .extent(settings.extent.clone())
        .halo(settings.halo)
        .repair(settings.repair)
}

/// Bevy's own device and queue, which generation shares with rendering.
fn bevy_device(app: &App) -> (wgpu::Device, wgpu::Queue) {
    let world = app.world();
    let device = world
        .get_resource::<RenderDevice>()
        .expect(
            "no RenderDevice: add WaveForgePlugin after the render plugin, or use \
             WaveForgePlugin::on_own_device",
        )
        .wgpu_device()
        .clone();
    let queue: wgpu::Queue = (**world
        .get_resource::<RenderQueue>()
        .expect("a RenderQueue comes with the RenderDevice"))
    .clone();
    (device, queue)
}

/// The messages and systems, which do not depend on where the solver came from.
fn register<S: Solver + Send + Sync + 'static>(app: &mut App) {
    app.add_message::<ChunkUpdated>()
        .add_message::<ChunkFailed>()
        .add_message::<ChunkEvicted>()
        .add_systems(
            Update,
            (follow_focus::<S>, generate::<S>)
                .chain()
                .in_set(WaveForgeSystems),
        );
}

fn insert<S: Solver + Send + Sync + 'static>(
    app: &mut App,
    generator: WorldGenerator<S>,
    settings: WaveForgeSettings,
) {
    app.insert_resource(WaveForgeWorld {
        generator,
        settings,
        asked: Vec::new(),
    });
}

/// The set both of the plugin's systems run in, so a game can order its own work around them.
#[derive(bevy_ecs::schedule::SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WaveForgeSystems;

/// Asks for the chunks around every focus, and drops what has left them.
fn follow_focus<S: Solver + Send + Sync + 'static>(
    world: Option<ResMut<WaveForgeWorld<S>>>,
    focus: Query<(&GlobalTransform, &GenerationFocus)>,
    mut evicted: MessageWriter<ChunkEvicted>,
) {
    let Some(mut world) = world else {
        return;
    };
    let wanted: Vec<FocusPoint> = focus
        .iter()
        .map(|(at, focus)| FocusPoint::new(world.settings.chunk_at(at.translation()), focus.radius))
        .collect();
    if wanted == world.asked {
        return;
    }
    world.generator.request(&wanted);
    if let Some(margin) = world.settings.evict_margin {
        let dropped = world.generator.evict_outside(&wanted, margin);
        evicted.write_batch(dropped.into_iter().map(|chunk| ChunkEvicted(chunk.coord)));
    }
    world.asked = wanted;
}

/// Starts one batch and collects whatever has finished, without blocking.
///
/// # Errors
/// If the solver refused a batch or failed. A region that simply did not solve is a
/// [`ChunkFailed`] message, not an error.
fn generate<S: Solver + Send + Sync + 'static>(
    world: Option<ResMut<WaveForgeWorld<S>>>,
    mut updated: MessageWriter<ChunkUpdated>,
    mut failed: MessageWriter<ChunkFailed>,
) -> Result {
    let Some(mut world) = world else {
        return Ok(());
    };
    world.generator.tick()?;
    for event in world.generator.poll()? {
        match event {
            ChunkEvent::Updated(chunk) => {
                updated.write(ChunkUpdated(chunk));
            }
            ChunkEvent::Failed { chunk, status } => {
                failed.write(ChunkFailed { chunk, status });
            }
            // The generator hands evicted chunks back to `follow_focus` rather than reporting them.
            ChunkEvent::Evicted(_) => {}
        }
    }
    Ok(())
}
