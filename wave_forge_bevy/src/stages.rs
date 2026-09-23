//! A world generated from a pack of stages (docs/reference/bevy.md), in Bevy.
//!
//! [`WaveForgeStagesPlugin`] runs the stages on a thread of their own through
//! [`wave_forge::stages::StageWorker`], asks for the chunks around every [`GenerationFocus`], and
//! sends a [`StageReady`] or [`StageDropped`] message for each product that arrives or goes. A
//! game reads the products from the [`WaveForgeStages`] resource. The runtime, and the town solver
//! with any GPU device it needs, are built on that thread, so a town solver gets a device of its
//! own there rather than Bevy's.

use crate::GenerationFocus;
use bevy_app::{App, Plugin, Update};
use bevy_ecs::message::{Message, MessageWriter};
use bevy_ecs::prelude::{IntoScheduleConfigs, Query, ResMut, Resource};
use bevy_math::Vec3;
use bevy_transform::components::GlobalTransform;
use std::sync::Mutex;
use wave_forge::stages::{Field, Point, Runtime, Site, StageEvent, StageWorker, TownChunk};
use wave_forge::{ChunkCoord, FocusPoint};

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

/// Generation stopped, and why.
#[derive(Message, Clone, Debug, PartialEq, Eq)]
pub struct StagesFailed(pub String);

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
    targets: Vec<String>,
    settings: StagesSettings,
    asked: Vec<FocusPoint>,
}

impl WaveForgeStages {
    /// A field stage's values for a chunk, if it has arrived.
    #[must_use]
    pub fn field(&self, stage: &str, chunk: ChunkCoord) -> Option<&Field> {
        self.worker.field(stage, chunk)
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

    /// Where a point stands in Bevy's world: the lattice's x and y across Bevy's x and z, its
    /// height up Bevy's y.
    #[must_use]
    pub fn translation_of(&self, point: &Point) -> Vec3 {
        let [x, y, height] = point.position;
        let cell = self.settings.cell_size;
        Vec3::new(x * cell.x, height * cell.y, y * cell.z)
    }

    /// Why generation stopped, if it did.
    #[must_use]
    pub fn failure(&self) -> Option<&str> {
        self.worker.failure()
    }
}

type Build = Box<dyn FnOnce() -> Result<Runtime, String> + Send>;

/// Generates the `targets` stages of a pack around every [`GenerationFocus`].
pub struct WaveForgeStagesPlugin {
    build: Mutex<Option<Build>>,
    targets: Vec<String>,
    settings: StagesSettings,
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
            targets: targets.iter().map(|target| (*target).to_owned()).collect(),
            settings,
        }
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
        })
        .add_message::<StageReady>()
        .add_message::<StageDropped>()
        .add_message::<StagesFailed>()
        .add_systems(
            Update,
            (follow_focus, drain).chain().in_set(WaveForgeStagesSystems),
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
    let targets: Vec<String> = stages.targets.clone();
    let names: Vec<&str> = targets.iter().map(String::as_str).collect();
    stages.worker.request(&wanted, &names);
    stages.asked = wanted;
}

/// Takes what the stages' thread finished, as messages.
fn drain(
    mut stages: ResMut<WaveForgeStages>,
    mut ready: MessageWriter<StageReady>,
    mut dropped: MessageWriter<StageDropped>,
    mut failed: MessageWriter<StagesFailed>,
) {
    let had_failed = stages.worker.failure().is_some();
    for event in stages.worker.drain() {
        match event {
            StageEvent::Generated { stage, chunk } => {
                ready.write(StageReady { stage, chunk });
            }
            StageEvent::Dropped { stage, chunk } => {
                dropped.write(StageDropped { stage, chunk });
            }
        }
    }
    if let (false, Some(reason)) = (had_failed, stages.worker.failure()) {
        failed.write(StagesFailed(reason.to_owned()));
    }
}
