//! A world generated from a pack of stages (docs/reference/godot.md), for Godot.
//!
//! The node is a thin surface over [`wave_forge::stages::StageWorker`]: it loads a pack and the
//! rule sets its Solve stages name, runs the stages on a thread of their own around the position
//! the game hands it, and gives each product to the game as Godot data: a field's values, a town
//! chunk's tiles and height, points as MultiMesh buffers per kind.
//!
//! It also builds the ground a player walks on: a mesh per chunk from a height field stage, drawn
//! through the `RenderingServer`, and near the player one static body per chunk holding the ground
//! as a height map and the town's modules as the shapes the game assigned them, built through the
//! `PhysicsServer3D` with every shape added before the body joins the space.

use crate::gi::Gi;
use crate::grass::Grass;
use crate::lods::{add_levelled_surface, levelled_mesh};
use crate::occlusion::Occluders;
use crate::placements::{Item, Placements};
use crate::stage_navigation::StageNavigation;
use crate::timings::Timings;
use crate::{BODIES_PER_FRAME, RECENT_FRAMES, from_vector, local_id, to_vector};
use godot::builtin::math::ApproxEq;
use godot::classes::base_material_3d::{Flags, ShadingMode};
use godot::classes::image::Format as ImageFormat;
use godot::classes::physics_server_3d::BodyMode;
use godot::classes::rendering_server::ArrayType;
use godot::classes::rendering_server::MultimeshTransformFormat;
use godot::classes::{
    ArrayMesh, BoxMesh, CollisionShape3D, ConcavePolygonShape3D, Engine, FastNoiseLite, FileAccess,
    HeightMapShape3D, INode, Image, ImageTexture, Material, MeshInstance3D, MeshLibrary,
    NavigationMesh, NavigationServer3D, Node, Node3D, PhysicsServer3D, RenderingServer,
    ResourceSaver, ShaderMaterial, Shape3D, StandardMaterial3D, StaticBody3D,
};
use godot::global::Error;
use godot::obj::EngineEnum;
use godot::prelude::*;
use godot::register::info::{PropertyHint, PropertyHintInfo, PropertyInfo};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::Arc;
use wave_forge::DirectoryStore;
use wave_forge::loader::{RuleFile, parse_rule_file};
use wave_forge::noise::{
    CellularDistanceFunction, CellularReturnType, DomainWarpFractalType, DomainWarpType,
    FractalType, NoiseConfig, NoiseType,
};
use wave_forge::stages::brushes::{Brush, Canvas, stroke};
use wave_forge::stages::regions::CurveId;
use wave_forge::stages::{
    Column, Edit, Edits, Facts, GivenRow, Judgement, MAX_CATEGORIES, Pack, ParamDef, Point,
    PointId, Rejection, RowId, RunProgress, Runtime, Save, Site, SiteId, StageError, StageEvent,
    StageKind, StageTiming, StageWorker, TableKind, Value,
};
use wave_forge::towns::WfcTowns;
use wave_forge::{
    Chunk, ChunkCoord, ChunkShape, FocusPoint, GroundMesh, InstanceSet, SurfaceWorker, VolumeMesh,
    YUpSpace, far_ground, ground, ground_height, ground_materials, ground_readers, volume_height,
};

/// Generates a world from a pack of stages around a position the game keeps handing it.
///
/// Set `pack_file`, the rule files its Solve stages name and the stages to generate, call
/// [`WaveForgeStages::start`], then [`WaveForgeStages::follow`] as the player moves, and read each
/// chunk the `stage_ready` signal names.
#[derive(GodotClass)]
#[class(tool, base = Node)]
pub struct WaveForgeStages {
    base: Base<Node>,

    /// The pack of stages to generate: a `*.world.ron` file.
    #[export_group(name = "Pack")]
    #[export(file = "*.ron")]
    pack_file: GString,
    /// The pack as a stack of stages the inspector edits, a `WaveForgeStack` of the plugin, which
    /// the node generates instead of `pack_file` when set.
    #[export]
    stack: Option<Gd<Resource>>,
    /// The rule sets Solve stages name, as name to rule file path.
    #[export]
    rules_files: VarDictionary,
    /// The stages to generate around the followed position; what they read comes with them.
    #[export]
    targets: PackedStringArray,
    /// Noises that replace the ones the pack names, as name to `FastNoiseLite` resource: a Field
    /// stage reading `FastNoise(name)` then gives exactly what the resource's `get_noise_2d` gives
    /// at each column's centre in cells.
    #[export]
    noises: Dictionary<StringName, Option<Gd<FastNoiseLite>>>,
    /// Values of the pack's parameters, as name to number, which `start` gives the stages; a
    /// parameter left out keeps its default ([packs.md](packs.md#parameters)). The inspector shows
    /// each as a slider over its range, `params/<name>`, which the scene saves and which changes
    /// the running stages as it moves; `update_params` does the same from code.
    #[var]
    params: VarDictionary,
    /// The pack file whose parameters the inspector lists, and them, read when it asks.
    listed_params: Option<(GString, BTreeMap<String, ParamDef>)>,
    /// Whether to start as soon as the node enters the scene tree, when the game runs.
    #[export]
    start_on_ready: bool,
    /// Whether to start in the editor too, as a preview a brush paints on; the editor plugin
    /// follows the editor's camera with it.
    #[export]
    preview_in_editor: bool,
    /// The edits of the world as text, `edits_log`'s: what brushes painted in the editor, which
    /// the scene saves and `start` applies. Setting it replaces the edits, as `set_edits_log` does.
    #[export(multiline)]
    #[var(get = edits_log, set = set_edits_text)]
    edits_text: PhantomVar<GString>,
    /// Starts the node, or starts it again with its settings as they are now: in the editor, the
    /// preview. The same as `start`.
    #[export_tool_button(fn = Self::regenerate, name = "Start or regenerate", icon = "Reload")]
    regenerate_button: PhantomVar<Callable>,
    /// Takes a new `seed` at random, and starts again if the node is running.
    #[export_tool_button(fn = Self::reroll_seed, name = "Reroll seed", icon = "RandomNumberGenerator")]
    reroll_button: PhantomVar<Callable>,
    /// Bakes the chunks within `view_radius` of the followed one, as `bake` does, into a scene saved
    /// at `bake_path`.
    #[export_tool_button(fn = Self::bake_view, name = "Bake the view", icon = "PackedScene")]
    bake_button: PhantomVar<Callable>,
    /// Where "Bake the view" saves the scene it bakes.
    #[export(file = "*.tscn")]
    bake_path: GString,

    /// Every choice in the world derives from this.
    #[export_group(name = "World")]
    #[export]
    seed: i64,
    /// The cells of one chunk: columns along the lattice's x and y, and the height of a town's
    /// chunks along z.
    #[export]
    chunk_cells: Vector3i,
    /// How large one cell is in Godot's world units, along Godot's axes.
    #[export]
    cell_size: Vector3,

    /// How many chunks to keep generated around the followed position.
    #[export_group(name = "Streaming")]
    #[export]
    view_radius: i32,
    /// Whether to follow the viewport's current camera each frame while the game runs, so a world
    /// generates around the player with no code. A script that calls `follow` takes over, turning
    /// it off; in the editor the plugin follows the editor's camera instead.
    #[export]
    follow_camera: bool,
    /// A radius of its own for some targets, as stage name to chunks, the others keeping
    /// `view_radius`: ground far out, locations nearer and clutter nearest, say.
    #[export]
    target_radii: Dictionary<StringName, i32>,

    /// The field stage the ground is built from, a height in cells per column; empty for no
    /// ground. It has to be generated, as a target or as what a target reads. A chunk's ground
    /// needs the fields of the chunks around it, so it reaches one chunk less than the view.
    #[export_group(name = "Ground")]
    #[export]
    ground_stage: GString,
    /// The material the ground is drawn with; none draws it with Godot's default.
    #[export]
    ground_material: Option<Gd<Material>>,
    /// A Rules, Area or Nearest stage at the ground's scale whose categories are the ground's
    /// materials; empty for none. Each chunk's ground then gets its own copy of `ground_material`,
    /// which has to
    /// be a `ShaderMaterial` taking `wave_forge_materials`, `wave_forge_cell` and
    /// `wave_forge_palette`, or of the reference ground shader when `ground_material` is empty.
    #[export]
    ground_material_stage: GString,
    /// A colour per category of `ground_material_stage`, by index; categories past its end take
    /// colours of their own from their index.
    #[export]
    ground_palette: PackedColorArray,
    /// The material the pack's sea is drawn with: a plane at its water level under the followed
    /// chunk, as wide as the view. Empty, or a pack without water, draws no sea.
    #[export]
    sea_material: Option<Gd<Material>>,
    /// A coarse field stage the far ground is drawn from beyond the near ground, a height in cells
    /// per column like `ground_stage`'s; empty for none. Give it a radius of its own in
    /// `target_radii`, as far as the ground should reach; a coarse chunk's far ground needs the
    /// fields around it, so it reaches one coarse chunk less. It is drawn with `ground_material`.
    #[export]
    far_ground_stage: GString,

    /// A Volume or Carve stage at scale 1 whose surface is drawn and, within `collider_radius`,
    /// collided with, for overhangs and caves; empty for none. It has to be generated, as a target or as
    /// what a target reads. A chunk's surface needs the volumes of the chunks around it, so it
    /// reaches one chunk less than the view.
    #[export_group(name = "Volume")]
    #[export]
    volume_stage: GString,
    /// The material the volume's surface is drawn with; none draws it with Godot's default, or,
    /// for a stage with materials, with each vertex in its material's colour.
    #[export]
    volume_material: Option<Gd<Material>>,
    /// How long a frame may spend drawing volume surfaces meshed on the surface thread, in
    /// milliseconds; one is drawn a frame whatever it costs.
    #[export]
    volume_budget_ms: f64,
    /// A colour per material of `volume_stage`, by index, which each vertex of its surface carries
    /// as its colour; materials past its end take colours of their own from their index.
    #[export]
    volume_palette: PackedColorArray,
    /// A Volume, Carve or Aquifer stage at scale 1 whose surface is drawn as fluid, never collided
    /// with: water and lava in the caves of `volume_stage`, say; empty for none. Its surfaces are
    /// meshed and drawn as the volume's are, within the same `volume_budget_ms`.
    #[export]
    fluid_stage: GString,
    /// The material the fluid's surface is drawn with; none draws it with the reference fluid
    /// shader, each vertex in its material's colour from `fluid_palette`, its alpha the fluid's
    /// opacity, glowing as `fluid_glow` says, seen from both sides.
    #[export]
    fluid_material: Option<Gd<Material>>,
    /// A colour per material of `fluid_stage`, by index, as `volume_palette` is for the volume.
    #[export]
    fluid_palette: PackedColorArray,
    /// How brightly each material of `fluid_stage` glows, by index, as a multiple of its colour:
    /// lava's glow, say. Each vertex carries its material's in its first UV; materials past its
    /// end do not glow.
    #[export]
    fluid_glow: PackedFloat32Array,

    /// A field stage whose value per column, from 0 to 1, is how much of it grass covers; empty for
    /// no grass. Grass stands on the ground, so it needs `ground_stage`.
    #[export_group(name = "Grass")]
    #[export]
    grass_stage: GString,
    /// Blades per column where the cover is 1.
    #[export]
    grass_per_cell: i32,
    /// How many chunks around the followed position get grass.
    #[export]
    grass_radius: i32,
    /// The material grass is drawn with: a `ShaderMaterial` taking the reference grass shader's
    /// parameters, or empty for the reference grass shader itself.
    #[export]
    grass_material: Option<Gd<Material>>,

    /// How many chunks around the followed position get colliders: the ground, and every town's
    /// modules that have a shape (`set_collision_shape`). Below zero, none.
    #[export_group(name = "Physics")]
    #[export]
    collider_radius: i32,

    /// How many chunks around the followed position get a navigation region, baked on the
    /// navigation server's threads from what their colliders hold: the ground, the volume's
    /// surface, and every town's modules that have a shape. A chunk is baked once it and its
    /// neighbours inside the world have all of that, so navigation reaches a chunk less far than
    /// the ground. Below zero, none.
    #[export_group(name = "Navigation")]
    #[export]
    navigation_radius: i32,
    /// The settings chunks are baked with: agent size, climb, slope, partitioning. Its cell size
    /// and height are replaced by the navigation map's, which they must match to merge.
    #[export]
    navigation_template: Option<Gd<NavigationMesh>>,

    /// How many chunks around the followed position get occluders of their towns' solid cells, as
    /// the rule sets' `solid` modules say, for Godot's occlusion culling. Below zero, none.
    #[export_group(name = "Occlusion")]
    #[export]
    occluder_radius: i32,

    /// Scenes placed where Scatter and Assemble stages put things, as a kind (a point's kind or a
    /// piece's name) to a `PackedScene` or a path to one, loaded on Godot's loader threads; give a
    /// scene holding another extension's Rust resource as a `PackedScene`, since such a resource
    /// aborts the process when loaded on a loader thread. A
    /// scene whose root is a lone `MeshInstance3D` without a script is drawn as one MultiMesh per
    /// chunk; any other is instantiated as nodes under this one, and `instance_spawned` names each.
    #[export_group(name = "Scenes")]
    #[export]
    scenes: VarDictionary,
    /// How long each frame may spend placing scenes, in milliseconds; chunks still due wait for
    /// the next frames, nearest the followed position first. A chunk is placed whole, so a frame
    /// can go over by one chunk's placing.
    #[export]
    placement_budget_ms: f64,
    /// Chunks around the followed position within which a scene placed as nodes is so; farther
    /// out it is drawn as a MultiMesh of its first mesh, or not at all without one, and a chunk
    /// crossing the radius is placed again. Below zero, such scenes are always nodes.
    #[export]
    promotion_radius: i32,

    /// Where compiled GPU kernels are kept across runs, so a town's first solve does not compile
    /// them every time the game starts; `user://` paths are resolved. Empty keeps none.
    #[export_group(name = "Advanced")]
    #[export]
    kernel_cache: GString,

    /// A directory `run_world` wrote the pack's whole world to, which the node plays instead of
    /// generating: its targets' chunks come from there, and nothing is generated. Empty
    /// generates as usual.
    #[export(dir)]
    play_directory: GString,

    /// Where the chunks of frozen stages that the request no longer needs are kept, one file each,
    /// so they leave memory and come back unchanged; `user://` paths are resolved. The game keeps
    /// the directory with its saves, since a save then holds only the frozen chunks in memory.
    /// Empty keeps every frozen chunk in memory, and in the save.
    #[export]
    frozen_directory: GString,

    /// A Scatter stage whose candidates are drawn as small boxes over the ground, each coloured by
    /// what became of it: kept, or the modifier that rejected it ([`WaveForgeStages::candidate_legend`]),
    /// to see why a rule places what it places. Empty draws none.
    #[export_group(name = "Debug")]
    #[export]
    candidates_stage: GString,

    /// What builds the runtime the node generates with, from its settings when it started.
    builder: Option<Builder>,
    /// The world run under way, if one is.
    world_run: Option<WorldRunning>,
    /// When the editor's configuration warnings were last looked at.
    refresh: crate::warnings::Refresh,
    /// The candidates drawn, by chunk: the MultiMesh, its instance, and each verdict's count.
    candidates: HashMap<ChunkCoord, (Rid, Rid, BTreeMap<String, i64>)>,
    /// The box every candidate is drawn with, kept while any is drawn.
    candidate_box: Option<Gd<BoxMesh>>,

    pack: Option<Arc<Pack>>,
    /// The pack's tables of facts for the seed, as the game last gave them; the sampler and the
    /// stages' thread each hold a copy.
    facts: Option<Facts>,
    /// The player's edits, as the game last gave or made them; the sampler and the stages' thread
    /// each hold them too.
    edits: Edits,
    /// A runtime that only samples, on Godot's thread: it never generates a chunk.
    sampler: Option<Runtime>,
    /// The rule sets Solve stages name, kept here too, to say what a town's tiles are.
    rules: BTreeMap<String, RuleFile>,
    worker: Option<crate::ending::Ending<StageWorker>>,
    followed: Option<ChunkCoord>,
    /// The sea's plane, once there is a followed chunk to draw it under.
    sea: Option<crate::sea::Sea>,
    process_ms: Timings,
    /// Each town module's collision shape, by module name, for every Solve stage.
    collision_shapes: HashMap<String, Gd<Shape3D>>,
    /// The modules whose shape in `collision_shapes` is their bound mesh's, not one a script set;
    /// none until the bound scenes have loaded and the shapes are made.
    mesh_shapes: Option<BTreeSet<String>>,
    /// Chunks whose ground may be buildable: a field around them arrived since they were last
    /// looked at.
    ground_due: std::collections::BTreeSet<ChunkCoord>,
    /// The chunks whose ground is built: its mesh, and its `RenderingServer` mesh and instance.
    grounds: HashMap<ChunkCoord, (GroundMesh, Rid, Rid)>,
    /// The revision each built ground was built as, so a chunk's body tells a ground built again
    /// from the one it holds.
    ground_revisions: HashMap<ChunkCoord, u64>,
    /// How many grounds have been built, which numbers each one's revision.
    ground_builds: u64,
    /// Chunks of `far_ground_stage` whose far ground may have to be built again: a field around
    /// them arrived, or near ground came or went on or beside them.
    far_due: std::collections::BTreeSet<ChunkCoord>,
    /// The chunks of `far_ground_stage` whose far ground is drawn: its `RenderingServer` mesh and
    /// instance.
    far_grounds: HashMap<ChunkCoord, (Rid, Rid)>,
    /// The surfaces of `volume_stage`, while there is one.
    rock: Option<VolumeLayer>,
    /// The surfaces of `fluid_stage`, while there is one.
    fluid: Option<VolumeLayer>,
    /// How many surfaces have been built, which numbers each one's revision, so a chunk's body
    /// tells a surface built again from the one it holds.
    volume_revisions: u64,
    /// The milliseconds drawing every surface so far has taken on Godot's thread; meshing them
    /// takes place on the surface thread.
    volume_ms: f64,
    /// The material a volume with materials is drawn with when `volume_material` is empty: its
    /// vertices' colours as albedo.
    vertex_colours: Option<Gd<StandardMaterial3D>>,
    /// The material the fluid is drawn with when `fluid_material` is empty: the reference fluid
    /// shader.
    fluid_colours: Option<Gd<ShaderMaterial>>,
    /// Each chunk's copy of the ground material, holding its material ids, while its ground is
    /// built; none without `ground_material_stage`.
    chunk_materials: HashMap<ChunkCoord, Gd<ShaderMaterial>>,
    /// The grass: its shared blades and each chunk's instance; none without `grass_stage`.
    grass: Option<Grass>,
    /// `ground_palette` as the texture the ground material reads, and the material every chunk's
    /// is copied from, made when the node starts.
    palette: Option<(Gd<ImageTexture>, Gd<ShaderMaterial>)>,
    /// What the node's slowest frame since the start spent Godot's thread on.
    slowest_frame: FrameCost,
    /// How long the node's last frame took on Godot's thread, in milliseconds.
    last_frame_ms: f64,
    /// Signals not yet emitted, in the order their events arrived.
    pending: VecDeque<StageEvent>,
    /// Each chunk's static body and the shapes of its ground and its volume's surface, which the
    /// body does not own, and what it holds, to tell when it has to be built again.
    bodies: HashMap<ChunkCoord, (Rid, Vec<Rid>, BodyContents)>,
    /// Chunks within `collider_radius` still waiting for a body after the last frame.
    bodies_pending: usize,
    /// Each chunk's navigation region, and what the colliders of the chunk and its neighbours
    /// held when it was baked.
    navigation: StageNavigation<Vec<BodyContents>>,
    /// The triangles of each module's collision shape, counter-clockwise seen from outside as the
    /// navigation bake reads them.
    shape_faces: HashMap<String, Vec<[f32; 3]>>,
    /// The occluders of the town chunks within `occluder_radius`, and the chunks whose towns
    /// arrived since they were built.
    occluders: Occluders,
    occluders_due: std::collections::BTreeSet<ChunkCoord>,
    /// The scenes bound to kinds, and what each stage's chunk placed.
    placements: Placements,
}

/// A chunk's volume surface as the node holds it.
struct Surface {
    mesh: VolumeMesh,
    /// Its `RenderingServer` mesh and instance, which a surface without triangles has none of.
    drawn: Option<(Rid, Rid)>,
    /// The revision it was built as, so a body tells a surface built again from the one it holds.
    revision: u64,
}

/// A volume stage drawn as surfaces meshed on a thread of their own: the rock of `volume_stage`,
/// or the fluid of `fluid_stage`.
struct VolumeLayer {
    stage: String,
    /// Chunks whose surface may be buildable: a volume around them arrived since they were last
    /// looked at.
    due: std::collections::BTreeSet<ChunkCoord>,
    /// The chunks whose surface is built.
    built: HashMap<ChunkCoord, Surface>,
    /// The thread its surfaces are meshed on.
    surfaces: SurfaceWorker,
    /// Surfaces meshed and waiting to be drawn, a frame's budget at a time.
    meshed: Vec<VolumeMesh>,
}

impl VolumeLayer {
    fn new(stage: String) -> Self {
        Self {
            stage,
            due: std::collections::BTreeSet::new(),
            built: HashMap::new(),
            surfaces: SurfaceWorker::spawn(),
            meshed: Vec::new(),
        }
    }

    /// How many chunks wait for their surface: due, meshing, or meshed and not drawn yet.
    fn pending(&self) -> usize {
        self.due.len() + self.surfaces.building() + self.meshed.len()
    }

    /// Frees every surface it drew and forgets what waits.
    fn clear(&mut self) {
        let mut rendering = RenderingServer::singleton();
        self.due.clear();
        self.meshed.clear();
        for (_, surface) in self.built.drain() {
            if let Some((mesh, instance)) = surface.drawn {
                rendering.free_rid(instance);
                rendering.free_rid(mesh);
            }
        }
    }
}

/// The most `stage_ready` and `stage_dropped` signals one frame emits. A wide request can bring
/// thousands of products at once, and emitting each costs a microsecond or two before any handler
/// a game connected runs, so the rest wait for the next frames.
const SIGNALS_PER_FRAME: usize = 256;

/// The most chunks one frame gives ground. A wide request makes dozens buildable at once, and
/// building each takes tenths of a millisecond on Godot's thread, so the rest wait for the next
/// frames, nearest the followed position first.
const GROUNDS_PER_FRAME: usize = 8;
/// How long one frame may spend giving chunks ground, in milliseconds, after its first. Uploading
/// a ground's surface and its material costs a millisecond and more on some drivers
/// (measurements.md E55).
const GROUNDS_BUDGET_MS: f64 = 2.0;
/// How long one frame may spend building bodies, in milliseconds, after its first.
const BODIES_BUDGET_MS: f64 = 2.0;

/// How many coarse chunks get their far ground built per frame at most.
const FAR_GROUNDS_PER_FRAME: usize = 4;

/// The most chunks one frame gives grass: each uploads two small textures.
const GRASS_PER_FRAME: usize = 4;

/// What one frame of `process` cost Godot's thread, and what it spent it on.
#[derive(Clone, Copy, Debug, Default)]
struct FrameCost {
    ms: f64,
    /// Events drained, and the milliseconds spent emitting them as signals, the handlers a game
    /// connected to them included.
    events: usize,
    signals_ms: f64,
    /// Chunks whose ground or volume surface was built, and the milliseconds that took.
    grounds: usize,
    grounds_ms: f64,
    /// Chunks whose body was built, and the milliseconds that took.
    bodies: usize,
    bodies_ms: f64,
    /// The milliseconds spent keeping navigation regions: finished bakes put in place, and a bake
    /// prepared.
    navigation_ms: f64,
    /// Nodes placed from bound scenes, and the milliseconds placing took, MultiMeshes included.
    placed: usize,
    placements_ms: f64,
}

/// A Godot `FastNoiseLite` resource as the library's noise configuration, every property read as
/// the resource holds it.
fn noise_config(noise: &Gd<FastNoiseLite>) -> Result<NoiseConfig, String> {
    fn pick<T: Copy>(options: &[T], ord: i32, what: &str) -> Result<T, String> {
        usize::try_from(ord)
            .ok()
            .and_then(|at| options.get(at).copied())
            .ok_or_else(|| format!("{what} {ord} is not one Wave Forge knows"))
    }
    let offset = noise.get_offset();
    Ok(NoiseConfig {
        noise_type: pick(
            &[
                NoiseType::Simplex,
                NoiseType::SimplexSmooth,
                NoiseType::Cellular,
                NoiseType::Perlin,
                NoiseType::ValueCubic,
                NoiseType::Value,
            ],
            noise.get_noise_type().ord(),
            "noise type",
        )?,
        seed: noise.get_seed(),
        frequency: noise.get_frequency(),
        offset: [offset.x, offset.y, offset.z],
        fractal_type: pick(
            &[
                FractalType::None,
                FractalType::Fbm,
                FractalType::Ridged,
                FractalType::PingPong,
            ],
            noise.get_fractal_type().ord(),
            "fractal type",
        )?,
        fractal_octaves: noise.get_fractal_octaves(),
        fractal_lacunarity: noise.get_fractal_lacunarity(),
        fractal_gain: noise.get_fractal_gain(),
        fractal_weighted_strength: noise.get_fractal_weighted_strength(),
        fractal_ping_pong_strength: noise.get_fractal_ping_pong_strength(),
        cellular_distance_function: pick(
            &[
                CellularDistanceFunction::Euclidean,
                CellularDistanceFunction::EuclideanSquared,
                CellularDistanceFunction::Manhattan,
                CellularDistanceFunction::Hybrid,
            ],
            noise.get_cellular_distance_function().ord(),
            "cellular distance function",
        )?,
        cellular_return_type: pick(
            &[
                CellularReturnType::CellValue,
                CellularReturnType::Distance,
                CellularReturnType::Distance2,
                CellularReturnType::Distance2Add,
                CellularReturnType::Distance2Sub,
                CellularReturnType::Distance2Mul,
                CellularReturnType::Distance2Div,
            ],
            noise.get_cellular_return_type().ord(),
            "cellular return type",
        )?,
        cellular_jitter: noise.get_cellular_jitter(),
        domain_warp_enabled: noise.is_domain_warp_enabled(),
        domain_warp_type: pick(
            &[
                DomainWarpType::Simplex,
                DomainWarpType::SimplexReduced,
                DomainWarpType::BasicGrid,
            ],
            noise.get_domain_warp_type().ord(),
            "domain warp type",
        )?,
        domain_warp_amplitude: noise.get_domain_warp_amplitude(),
        domain_warp_frequency: noise.get_domain_warp_frequency(),
        domain_warp_fractal_type: pick(
            &[
                DomainWarpFractalType::None,
                DomainWarpFractalType::Progressive,
                DomainWarpFractalType::Independent,
            ],
            noise.get_domain_warp_fractal_type().ord(),
            "domain warp fractal type",
        )?,
        domain_warp_fractal_octaves: noise.get_domain_warp_fractal_octaves(),
        domain_warp_fractal_lacunarity: noise.get_domain_warp_fractal_lacunarity(),
        domain_warp_fractal_gain: noise.get_domain_warp_fractal_gain(),
    })
}

/// Puts what names a site in `out`: its `region` for a Sites stage's, its `row` for a TableSites
/// stage's, its `region` and `index` for a Locations stage's.
fn name_site(out: &mut VarDictionary, site: &SiteId) {
    match site {
        SiteId::Region(x, y) => {
            out.set(&"region".to_variant(), &Vector2i::new(*x, *y).to_variant())
        }
        SiteId::Location { region, index } => {
            out.set(
                &"region".to_variant(),
                &Vector2i::new(region.0, region.1).to_variant(),
            );
            out.set(&"index".to_variant(), &i64::from(*index).to_variant());
        }
        SiteId::Row(row) => {
            let row: PackedInt64Array = row.0.iter().map(|&part| part as i64).collect();
            out.set(&"row".to_variant(), &row.to_variant());
        }
    }
}

/// A row a game gives from GDScript, checked into the library's form: an `id`, a whole number from
/// 0, and every other key a column with a number or a name.
/// A site as `sites` and `locate` give it to GDScript.
fn site_dictionary(site: &Site) -> VarDictionary {
    let mut out = VarDictionary::new();
    name_site(&mut out, &site.id);
    if let Some(kind) = &site.kind {
        out.set(&"kind".to_variant(), &GString::from(&**kind).to_variant());
    }
    if let Some(name) = site.name() {
        let mut args = VarDictionary::new();
        for (arg, value) in &name.args {
            args.set(&arg.to_variant(), &value.to_variant());
        }
        out.set(
            &"name_key".to_variant(),
            &GString::from(name.key.as_str()).to_variant(),
        );
        out.set(&"name_args".to_variant(), &args.to_variant());
    }
    out.set(
        &"min".to_variant(),
        &Vector2i::new(site.min.0, site.min.1).to_variant(),
    );
    out.set(
        &"max".to_variant(),
        &Vector2i::new(site.max.0, site.max.1).to_variant(),
    );
    out.set(&"height".to_variant(), &site.height.to_variant());
    out
}

fn given_row(row: &VarDictionary) -> Result<GivenRow, String> {
    let id = row
        .get("id")
        .ok_or_else(|| format!("a row without an id: {row:?}"))?;
    // JSON reads every number back as a float, so a history saved as JSON has whole float ids.
    let whole = match id.get_type() {
        VariantType::INT => Some(id.to::<i64>()),
        VariantType::FLOAT => Some(id.to::<f64>())
            .filter(|id| id.fract() == 0.0)
            .map(|id| id as i64),
        _ => None,
    };
    let id = whole
        .and_then(|id| u64::try_from(id).ok())
        .ok_or_else(|| format!("a row id that is not a whole number from 0: {id:?}"))?;
    let mut values = BTreeMap::new();
    for (key, value) in row.iter_shared() {
        let key = key.to_string();
        if key == "id" {
            continue;
        }
        let value = match value.get_type() {
            VariantType::INT => Value::Number(value.to::<i64>() as f32),
            VariantType::FLOAT => Value::Number(value.to::<f64>() as f32),
            VariantType::STRING | VariantType::STRING_NAME => Value::Name(value.to_string()),
            other => {
                return Err(format!(
                    "row {id} gives {key:?} as a {other:?}; a column takes a number or a name"
                ));
            }
        };
        values.insert(key, value);
    }
    Ok(GivenRow { id, values })
}

fn elapsed_ms(since: std::time::Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1000.0
}

/// How long a frame may already have spent on Godot's thread, in milliseconds, for a navigation
/// bake to be started in it: starting one costs up to a millisecond, and P1 bounds the node's own
/// time at 2 ms a frame.
const NAVIGATION_START_MS: f64 = 1.0;

/// The names of the pack's Solve stages, in the pack's order.
fn solve_stages(pack: &Pack) -> Vec<String> {
    pack.stage_names()
        .filter(|name| matches!(pack.kind(name), Some(StageKind::Solve { .. })))
        .map(ToOwned::to_owned)
        .collect()
}

/// What a chunk's body was built from: whether it has the ground, and which Solve stages' towns.
#[derive(Clone, Debug, PartialEq, Eq)]
struct BodyContents {
    /// The revision of the chunk's ground, if it has one.
    ground: Option<u64>,
    /// The revision of the chunk's volume surface, if it has triangles.
    volume: Option<u64>,
    towns: Vec<String>,
}

#[godot_api]
impl INode for WaveForgeStages {
    fn init(base: Base<Node>) -> Self {
        Self {
            base,
            pack_file: GString::new(),
            stack: None,
            regenerate_button: PhantomVar::default(),
            reroll_button: PhantomVar::default(),
            bake_button: PhantomVar::default(),
            bake_path: GString::from("res://wave_forge_bake.tscn"),
            rules_files: VarDictionary::new(),
            noises: Dictionary::new(),
            target_radii: Dictionary::new(),
            targets: PackedStringArray::new(),
            params: VarDictionary::new(),
            listed_params: None,
            start_on_ready: false,
            preview_in_editor: false,
            edits_text: PhantomVar::default(),
            seed: 0,
            chunk_cells: Vector3i::new(8, 8, 8),
            cell_size: Vector3::ONE,
            view_radius: 2,
            follow_camera: true,
            pack: None,
            facts: None,
            edits: Edits::default(),
            sampler: None,
            rules: BTreeMap::new(),
            worker: None,
            followed: None,
            sea: None,
            sea_material: None,
            process_ms: Timings::new(RECENT_FRAMES),
            ground_stage: GString::new(),
            far_ground_stage: GString::new(),
            kernel_cache: GString::from("user://wave_forge/kernels"),
            frozen_directory: GString::new(),
            play_directory: GString::new(),
            builder: None,
            world_run: None,
            refresh: crate::warnings::Refresh::default(),
            candidates_stage: GString::new(),
            candidates: HashMap::new(),
            candidate_box: None,
            ground_material: None,
            ground_material_stage: GString::new(),
            grass_stage: GString::new(),
            grass_per_cell: 8,
            grass_radius: 1,
            grass_material: None,
            grass: None,
            ground_palette: PackedColorArray::new(),
            chunk_materials: HashMap::new(),
            palette: None,
            collider_radius: 1,
            collision_shapes: HashMap::new(),
            mesh_shapes: None,
            grounds: HashMap::new(),
            ground_revisions: HashMap::new(),
            ground_builds: 0,
            far_due: std::collections::BTreeSet::new(),
            far_grounds: HashMap::new(),
            volume_stage: GString::new(),
            volume_material: None,
            volume_palette: PackedColorArray::new(),
            volume_budget_ms: 2.0,
            vertex_colours: None,
            fluid_stage: GString::new(),
            fluid_material: None,
            fluid_palette: PackedColorArray::new(),
            fluid_glow: PackedFloat32Array::new(),
            fluid_colours: None,
            rock: None,
            fluid: None,
            volume_revisions: 0,
            volume_ms: 0.0,
            last_frame_ms: 0.0,
            ground_due: std::collections::BTreeSet::new(),
            bodies: HashMap::new(),
            bodies_pending: 0,
            navigation_radius: -1,
            navigation_template: None,
            navigation: StageNavigation::new(),
            shape_faces: HashMap::new(),
            occluder_radius: -1,
            occluders: Occluders::default(),
            occluders_due: std::collections::BTreeSet::new(),
            scenes: VarDictionary::new(),
            placement_budget_ms: 2.0,
            promotion_radius: -1,
            placements: Placements::default(),
            slowest_frame: FrameCost::default(),
            pending: VecDeque::new(),
        }
    }

    /// A slider per parameter of the pack, `params/<name>`, over its range.
    fn on_get_property_list(&mut self) -> Vec<PropertyInfo> {
        self.pack_param_defs()
            .iter()
            .map(|(name, param)| {
                PropertyInfo::new_export::<f32>(&format!("{PARAMS}{name}")).with_hint_info(
                    PropertyHintInfo {
                        hint: PropertyHint::RANGE,
                        hint_string: GString::from(
                            format!("{},{},0.01", param.range.0, param.range.1).as_str(),
                        ),
                    },
                )
            })
            .collect()
    }

    fn on_get(&self, property: StringName) -> Option<Variant> {
        let name = property.to_string().strip_prefix(PARAMS)?.to_owned();
        let (_, defs) = self.listed_params.as_ref()?;
        let default = defs.get(&name)?.default;
        Some(
            self.params
                .get(name.as_str())
                .unwrap_or_else(|| default.to_variant()),
        )
    }

    fn on_set(&mut self, property: StringName, value: Variant) -> bool {
        let Some(name) = property
            .to_string()
            .strip_prefix(PARAMS)
            .map(ToOwned::to_owned)
        else {
            return false;
        };
        self.params.set(name.as_str(), &value);
        if self.worker.is_some() {
            let mut change = VarDictionary::new();
            change.set(name.as_str(), &value);
            self.update_params(change);
        }
        true
    }

    fn on_property_get_revert(&self, property: StringName) -> Option<Variant> {
        let name = property.to_string().strip_prefix(PARAMS)?.to_owned();
        let (_, defs) = self.listed_params.as_ref()?;
        Some(defs.get(&name)?.default.to_variant())
    }

    /// Frees the ground's meshes and the bodies, which belong to the rendering and physics
    /// servers rather than to the node.
    fn exit_tree(&mut self) {
        self.clear_ground_and_bodies();
    }

    fn get_configuration_warnings(&self) -> PackedStringArray {
        self.configuration_warnings()
    }

    fn ready(&mut self) {
        let starts = if Engine::singleton().is_editor_hint() {
            self.preview_in_editor
        } else {
            self.start_on_ready
        };
        if starts {
            self.start();
        }
    }

    /// Hands the frame whatever the stages' thread finished, as signals.
    fn process(&mut self, _delta: f64) {
        if self.refresh.due() {
            let warnings = self.warnings();
            if self.refresh.changed(warnings) {
                self.base_mut().update_configuration_warnings();
            }
        }
        self.update_world_run();
        if self.follow_camera && !Engine::singleton().is_editor_hint() {
            let camera = self
                .base()
                .get_viewport()
                .and_then(|viewport| viewport.get_camera_3d());
            if let Some(camera) = camera {
                self.follow_position(camera.get_global_position());
            }
        }
        let processing = std::time::Instant::now();
        let Some(worker) = &mut self.worker else {
            return;
        };
        let events = worker.drain();
        let save = if events.contains(&StageEvent::Saved) {
            worker.take_save()
        } else {
            None
        };
        if let Some(reason) = worker.failure().map(ToOwned::to_owned) {
            self.worker = None;
            godot_error!("wave forge: stages stopped: {reason}");
            self.signals()
                .generation_failed()
                .emit(&GString::from(&reason));
            return;
        }
        // The chunks whose towns arrived this frame, which get their occluders built again.
        let towns: Vec<ChunkCoord> = events
            .iter()
            .filter_map(|event| match event {
                StageEvent::Generated { stage, chunk }
                    if matches!(
                        self.pack.as_ref().and_then(|pack| pack.kind(stage)),
                        Some(StageKind::Solve { .. })
                    ) =>
                {
                    Some(*chunk)
                }
                _ => None,
            })
            .collect();
        let ground_stage = self.ground_stage.to_string();
        let (mut arrived, mut gone) = (Vec::new(), Vec::new());
        if self.placements.any() {
            let placing = |stage: &str| {
                matches!(
                    self.pack.as_ref().and_then(|pack| pack.kind(stage)),
                    Some(
                        StageKind::Scatter { .. }
                            | StageKind::Embed { .. }
                            | StageKind::Deposit { .. }
                            | StageKind::Spawn { .. }
                            | StageKind::Assemble { .. }
                            | StageKind::Cave { .. }
                            | StageKind::Solve { .. }
                    )
                )
            };
            for event in &events {
                match event {
                    StageEvent::Generated { stage, chunk } if placing(stage) => {
                        self.placements.arrived(stage, *chunk);
                    }
                    StageEvent::Dropped { stage, chunk } if placing(stage) => {
                        self.placements.dropped(stage, *chunk);
                    }
                    StageEvent::Generated { .. }
                    | StageEvent::Dropped { .. }
                    | StageEvent::Saved
                    | StageEvent::Judged { .. } => {}
                }
            }
        }
        let material_stage = self.ground_material_stage.to_string();
        for event in &events {
            match event {
                // A chunk's ground reads the materials of itself and of the chunks beyond its
                // far edges, which are among the chunks whose ground reads a field of this chunk.
                StageEvent::Generated { stage, chunk }
                    if *stage == ground_stage || *stage == material_stage =>
                {
                    arrived.push(*chunk);
                }
                StageEvent::Dropped { stage, chunk }
                    if *stage == ground_stage || *stage == material_stage =>
                {
                    gone.push(*chunk);
                }
                StageEvent::Generated { .. }
                | StageEvent::Dropped { .. }
                | StageEvent::Saved
                | StageEvent::Judged { .. } => {}
            }
        }
        // What arrived and went of each volume stage drawn: the rock's, then the fluid's.
        let mut layers: [(Vec<ChunkCoord>, Vec<ChunkCoord>); 2] = Default::default();
        let stages = [self.volume_stage.to_string(), self.fluid_stage.to_string()];
        for event in &events {
            for (layer, stage_drawn) in stages.iter().enumerate() {
                match event {
                    StageEvent::Generated { stage, chunk } if stage == stage_drawn => {
                        layers[layer].0.push(*chunk);
                    }
                    StageEvent::Dropped { stage, chunk } if stage == stage_drawn => {
                        layers[layer].1.push(*chunk);
                    }
                    StageEvent::Generated { .. }
                    | StageEvent::Dropped { .. }
                    | StageEvent::Saved
                    | StageEvent::Judged { .. } => {}
                }
            }
        }
        let far_stage = self.far_ground_stage.to_string();
        let (mut far_arrived, mut far_gone) = (Vec::new(), Vec::new());
        for event in &events {
            match event {
                StageEvent::Generated { stage, chunk } if *stage == far_stage => {
                    far_arrived.push(*chunk);
                }
                StageEvent::Dropped { stage, chunk } if *stage == far_stage => {
                    far_gone.push(*chunk);
                }
                StageEvent::Generated { .. }
                | StageEvent::Dropped { .. }
                | StageEvent::Saved
                | StageEvent::Judged { .. } => {}
            }
        }
        if let Some(save) = save {
            self.signals()
                .saved()
                .emit(&GString::from(save.to_ron().as_str()));
        }
        self.update_candidates(&events);
        // A save is signalled as it arrives, and a report is drawn as it arrives.
        self.pending.extend(
            events
                .into_iter()
                .filter(|event| !matches!(event, StageEvent::Saved | StageEvent::Judged { .. })),
        );
        let emitted = self.pending.len().min(SIGNALS_PER_FRAME);
        let mut frame = FrameCost {
            events: emitted,
            ..FrameCost::default()
        };
        let signalling = std::time::Instant::now();
        for event in self.pending.drain(..emitted).collect::<Vec<_>>() {
            match event {
                StageEvent::Generated { stage, chunk } => self
                    .signals()
                    .stage_ready()
                    .emit(&GString::from(&stage), to_vector(chunk)),
                StageEvent::Dropped { stage, chunk } => self
                    .signals()
                    .stage_dropped()
                    .emit(&GString::from(&stage), to_vector(chunk)),
                StageEvent::Saved | StageEvent::Judged { .. } => {
                    unreachable!("a save and a report are handled as they arrive")
                }
            }
        }
        frame.signals_ms = elapsed_ms(signalling);
        let grounding = std::time::Instant::now();
        let near_before: std::collections::BTreeSet<ChunkCoord> =
            self.grounds.keys().copied().collect();
        frame.grounds = self.update_ground(&arrived, &gone);
        let near_changed: Vec<ChunkCoord> = self
            .grounds
            .keys()
            .copied()
            .filter(|chunk| !near_before.contains(chunk))
            .chain(
                near_before
                    .iter()
                    .copied()
                    .filter(|chunk| !self.grounds.contains_key(chunk)),
            )
            .collect();
        self.update_far_ground(&far_arrived, &far_gone, &near_changed);
        let drawing = std::time::Instant::now();
        frame.grounds +=
            self.update_layer(false, &layers[0].0, &layers[0].1, self.volume_budget_ms);
        let left = self.volume_budget_ms - elapsed_ms(drawing);
        frame.grounds += self.update_layer(true, &layers[1].0, &layers[1].1, left);
        frame.grounds_ms = elapsed_ms(grounding);
        let building = std::time::Instant::now();
        frame.bodies = self.update_colliders();
        frame.bodies_ms = elapsed_ms(building);
        let navigating = std::time::Instant::now();
        // A bake is started only on a frame that has not already spent much of Godot's time, so
        // it does not land on one that built grounds or bodies.
        let may_start = elapsed_ms(processing) < NAVIGATION_START_MS;
        for chunk in self.update_navigation(may_start) {
            self.signals().navigation_ready().emit(to_vector(chunk));
        }
        frame.navigation_ms = elapsed_ms(navigating);
        self.update_occluders(&towns);
        self.update_grass();
        let placing = std::time::Instant::now();
        self.give_mesh_shapes();
        frame.placed = self.update_placements();
        frame.placements_ms = elapsed_ms(placing);
        frame.ms = elapsed_ms(processing);
        self.process_ms.push(frame.ms);
        self.last_frame_ms = frame.ms;
        if frame.ms > self.slowest_frame.ms {
            self.slowest_frame = frame;
        }
    }
}

#[godot_api]
impl WaveForgeStages {
    /// What the editor shows as the node's configuration warnings, each a sentence: settings that
    /// would leave its world dark, without bodies, occluders or sound, or that name no fitting
    /// stage (docs/reference/godot.md, "Editor"). Empty when there is nothing to warn of.
    #[func]
    fn configuration_warnings(&self) -> PackedStringArray {
        self.warnings().iter().map(GString::from).collect()
    }

    /// A stage's product for a chunk is ready to read. At most 256 of these and `stage_dropped`
    /// together are emitted per frame, in the order the products arrived, so after a wide request
    /// some come a few frames later; by then a product can have been dropped again, and its
    /// `stage_dropped` follows.
    #[signal]
    fn stage_ready(stage: GString, chunk: Vector3i);

    /// A stage's product for a chunk is no longer needed and was dropped.
    #[signal]
    fn stage_dropped(stage: GString, chunk: Vector3i);

    /// A node of a bound scene was placed under this node, at the point or piece of `chunk` whose
    /// id is `id`, as `point_sets` and `stamps` give ids. It is freed when its chunk is dropped.
    #[signal]
    fn instance_spawned(node: Gd<Node3D>, chunk: Vector3i, id: i64);

    /// Generation stopped, and why.
    #[signal]
    fn generation_failed(reason: GString);

    /// The save `request_save` asked for, as text `load_save` takes: the player's edits, less
    /// those of ephemeral stages, and every chunk of a frozen stage, with the Wave Forge version
    /// and the pack's digest.
    #[signal]
    fn saved(save: GString);

    /// A world run got one chunk further: `done` of `total`. `stages` is what each stage has
    /// generated so far and what that cost, as `stats()` gives it.
    #[signal]
    fn world_run_progress(done: i64, total: i64, stages: VarDictionary);

    /// A world run ended, finished if `done` is `total`, stopped or failed if not; a failure is
    /// reported as an error too.
    #[signal]
    fn world_run_finished(done: i64, total: i64);

    /// A chunk's navigation mesh is baked and the navigation map has taken it in, so a path asked
    /// for now finds it; agents can path across it and into its neighbours that have theirs.
    #[signal]
    fn navigation_ready(chunk: Vector3i);

    /// Loads the pack, `stack`'s or `pack_file`'s, and the rule sets its Solve stages name, and
    /// starts the stages' thread. Returns whether it could start; why not is reported as an
    /// error. A town solver builds its device on the stages' thread, so a failure there arrives as
    /// `generation_failed`.
    #[func]
    fn start(&mut self) -> bool {
        let text = match self.pack_source() {
            Ok(text) => text,
            Err(error) => {
                godot_error!("wave forge: {error}");
                return false;
            }
        };
        let pack = match Pack::parse(&text) {
            Ok(pack) => Arc::new(pack),
            Err(error) => {
                godot_error!("wave forge: {}: {error}", self.pack_name());
                return false;
            }
        };
        let mut rules: BTreeMap<String, RuleFile> = BTreeMap::new();
        for (name, path) in self.rules_files.iter_shared() {
            let path = path.to::<GString>();
            match parse_rule_file(&FileAccess::get_file_as_string(&path).to_string()) {
                Ok(file) => {
                    rules.insert(name.to::<GString>().to_string(), file);
                }
                Err(error) => {
                    godot_error!("wave forge: {path}: {error}");
                    return false;
                }
            }
        }
        if let Some(problem) = self.stage_setting_problems(&pack).first() {
            godot_error!("wave forge: {problem}");
            return false;
        }
        if !self.ground_material_stage.is_empty() {
            match self.materials_template() {
                Ok(template) => self.palette = Some(template),
                Err(error) => {
                    godot_error!("wave forge: {error}");
                    return false;
                }
            }
        } else {
            self.palette = None;
        }
        self.rock = None;
        self.fluid = None;
        if !self.fluid_stage.is_empty() {
            let stage = self.fluid_stage.to_string();
            self.fluid = Some(VolumeLayer::new(stage));
            if self.fluid_colours.is_none() {
                let mut material = ShaderMaterial::new_gd();
                material.set_shader(&crate::shaders::reference(
                    "fluid.gdshader",
                    crate::shaders::FLUID,
                ));
                self.fluid_colours = Some(material);
            }
        }
        if !self.volume_stage.is_empty() {
            let stage = self.volume_stage.to_string();
            self.rock = Some(VolumeLayer::new(stage));
            if self.vertex_colours.is_none() {
                let mut colours = StandardMaterial3D::new_gd();
                colours.set_flag(Flags::ALBEDO_FROM_VERTEX_COLOR, true);
                // `volume_palette` holds colours as the inspector picks them, in sRGB.
                colours.set_flag(Flags::SRGB_VERTEX_COLOR, true);
                self.vertex_colours = Some(colours);
            }
        }
        if self.grass_stage.is_empty() {
            self.grass = None;
        } else {
            if pack.kind(&self.grass_stage.to_string()).is_none() || self.ground_stage.is_empty() {
                godot_error!(
                    "wave forge: grass_stage {} is no stage of the pack, or there is no ground_stage                      for grass to stand on",
                    self.grass_stage
                );
                return false;
            }
            let columns = [
                self.chunk_cells.x.max(1) as u32,
                self.chunk_cells.y.max(1) as u32,
            ];
            match Grass::new(
                self.grass_material.as_ref(),
                self.grass_per_cell.max(1) as u32,
                columns,
            ) {
                Ok(grass) => self.grass = Some(grass),
                Err(error) => {
                    godot_error!("wave forge: {error}");
                    return false;
                }
            }
        }
        let solves = pack
            .stage_names()
            .any(|name| matches!(pack.kind(name), Some(StageKind::Solve { .. })));
        let cells = self.chunk_cells;
        let shape = ChunkShape {
            x: cells.x.max(1) as u32,
            y: cells.y.max(1) as u32,
            z: cells.z.max(1) as u32,
        };
        let seed = self.seed as u64;
        let cache = (!self.kernel_cache.is_empty())
            .then(|| std::path::PathBuf::from(crate::paths::directory_path(&self.kernel_cache)));
        let frozen = (!self.frozen_directory.is_empty()).then(|| {
            std::path::PathBuf::from(crate::paths::directory_path(&self.frozen_directory))
        });
        let mut noises = Vec::new();
        for (name, noise) in self.noises.iter_shared() {
            let name = name.to_string();
            let Some(noise) = noise else {
                godot_error!("wave forge: noise {name:?} has no FastNoiseLite");
                return false;
            };
            match noise_config(&noise) {
                Ok(config) => noises.push((name, config)),
                Err(message) => {
                    godot_error!("wave forge: noise {name:?}: {message}");
                    return false;
                }
            }
        }
        let with_noises = move |mut runtime: Runtime| -> Result<Runtime, String> {
            for (name, config) in &noises {
                runtime = runtime
                    .with_noise(name, *config)
                    .map_err(|error| error.to_string())?;
            }
            Ok(runtime)
        };
        let facts = match Facts::new(Arc::clone(&pack), seed) {
            Ok(facts) => facts,
            Err(error) => {
                godot_error!("wave forge: {}: {error}", self.pack_name());
                return false;
            }
        };
        let mut sampler =
            match with_noises.clone()(Runtime::new(Arc::clone(&pack), seed, [shape.x, shape.y])) {
                Ok(sampler) => sampler,
                Err(error) => {
                    godot_error!("wave forge: {}: {error}", self.pack_name());
                    return false;
                }
            };
        sampler
            .set_facts(facts.clone())
            .expect("facts made for the sampler's pack and seed");
        let Some(values) = param_values(&self.params) else {
            godot_error!(
                "wave forge: params holds a name or value that is not a string and a number"
            );
            return false;
        };
        if let Err(error) = sampler.set_params(&values) {
            godot_error!("wave forge: {error}");
            return false;
        }
        // The edits `edits_text` gave, painted in the editor say.
        if let Err(error) = sampler.set_edits(&self.edits) {
            godot_error!("wave forge: {error}");
            return false;
        }
        let thread_params = values.clone();
        let thread_edits = self.edits.clone();
        let for_thread = Arc::clone(&pack);
        let thread_facts = facts.clone();
        self.rules = rules.clone();
        self.sampler = Some(sampler);
        self.facts = Some(facts);
        self.pack = Some(pack);
        self.followed = None;
        self.pending.clear();
        self.ground_due.clear();
        self.clear_ground_and_bodies();
        self.placements = match Placements::new(&self.scenes) {
            Ok(placements) => placements,
            Err(error) => {
                godot_error!("wave forge: {error}");
                return false;
            }
        };
        for module in self.mesh_shapes.take().into_iter().flatten() {
            self.collision_shapes.remove(&module);
        }
        // The towns' device is built on the stages' thread, which is where it is used.
        // The runtime the stages' thread generates with, and a world run too.
        let build: Builder = Arc::new(move || {
            let mut runtime = with_noises.clone()(Runtime::new(
                Arc::clone(&for_thread),
                seed,
                [shape.x, shape.y],
            ))?;
            if let Some(directory) = &frozen {
                runtime = runtime.with_store(Box::new(DirectoryStore::new(directory.clone())));
            }
            runtime
                .set_facts(thread_facts.clone())
                .map_err(|error| error.to_string())?;
            runtime
                .set_params(&thread_params)
                .map_err(|error| error.to_string())?;
            runtime
                .set_edits(&thread_edits)
                .map_err(|error| error.to_string())?;
            if !solves {
                return Ok(runtime);
            }
            let mut towns = WfcTowns::new(shape);
            for (name, file) in &rules {
                towns = match &cache {
                    Some(dir) => towns.with_rules(name, file.clone(), |rules| {
                        wave_forge::towns::gpu_solver_cached(rules, dir)
                    }),
                    None => towns.with_rules(name, file.clone(), wave_forge::towns::gpu_solver),
                }
                .map_err(|error| error.to_string())?;
            }
            runtime
                .with_towns(Box::new(towns))
                .map_err(|error| error.to_string())
        });
        self.builder = Some(Arc::clone(&build));
        self.worker = Some(crate::ending::Ending::new(
            if self.play_directory.is_empty() {
                StageWorker::spawn(move || build())
            } else {
                let directory = crate::paths::directory_path(&self.play_directory);
                StageWorker::play(
                    Arc::clone(self.pack.as_ref().expect("set above")),
                    [shape.x, shape.y],
                    Box::new(DirectoryStore::new(directory)),
                )
            },
        ));
        true
    }

    /// Generates around `position`, in Godot's world space. Call it as the player moves; it asks
    /// for new chunks only when the position enters another chunk. Called while the game runs, it
    /// turns `follow_camera` off: the script follows from then on.
    #[func]
    fn follow(&mut self, position: Vector3) {
        if !Engine::singleton().is_editor_hint() {
            self.follow_camera = false;
        }
        self.follow_position(position);
    }

    /// What `follow` does, without turning `follow_camera` off.
    fn follow_position(&mut self, position: Vector3) {
        let Some(worker) = &self.worker else {
            return;
        };
        let size = [
            self.chunk_cells.x.max(1) as f32 * self.cell_size.x,
            self.chunk_cells.y.max(1) as f32 * self.cell_size.z,
        ];
        let chunk = ChunkCoord::new(
            (position.x / size[0]).floor() as i32,
            (position.z / size[1]).floor() as i32,
            0,
        );
        if self.followed == Some(chunk) {
            return;
        }
        self.followed = Some(chunk);
        let targets: Vec<(String, Option<u32>)> = self
            .targets
            .as_slice()
            .iter()
            .map(|target| {
                let radius = self
                    .target_radii
                    .get(&StringName::from(target))
                    .map(|radius| radius.max(0) as u32);
                (target.to_string(), radius)
            })
            .collect();
        let targets: Vec<(&str, Option<u32>)> = targets
            .iter()
            .map(|(target, radius)| (target.as_str(), *radius))
            .collect();
        worker.request_each(
            &[FocusPoint::new(chunk, self.view_radius.max(0) as u32)],
            &targets,
        );
        self.place_sea(chunk, size);
    }

    /// A field stage's values for a chunk, column by column with the lattice's x fastest; empty if
    /// it is not a field stage or the chunk has not arrived.
    #[func]
    fn field_values(&self, stage: GString, chunk: Vector3i) -> PackedFloat32Array {
        self.worker
            .as_ref()
            .and_then(|worker| worker.field(&stage.to_string(), from_vector(chunk)))
            .map_or_else(PackedFloat32Array::new, |field| {
                PackedFloat32Array::from(field.values.as_slice())
            })
    }

    /// A Rules stage's categories for a chunk, column by column with the lattice's x fastest, as
    /// indices into `category_names(stage)`; empty if it is not a Rules stage or the chunk has not
    /// arrived.
    #[func]
    fn categories(&self, stage: GString, chunk: Vector3i) -> PackedByteArray {
        self.worker
            .as_ref()
            .and_then(|worker| worker.categories(&stage.to_string(), from_vector(chunk)))
            .map_or_else(PackedByteArray::new, |categories| {
                PackedByteArray::from(categories.values.as_slice())
            })
    }

    /// A Region or TableCurves stage's curves that pass through a chunk: what names each one, its
    /// `region` (Vector2i) and `index` for a region job's, its `row` (PackedInt64Array) for a
    /// table's, its `points` in Godot's world space on the ground plane (y is 0), and its
    /// `values`. Empty if it is neither or the chunk has not arrived.
    #[func]
    fn curves(&self, stage: GString, chunk: Vector3i) -> Array<VarDictionary> {
        let Some(curves) = self
            .worker
            .as_ref()
            .and_then(|worker| worker.curves(&stage.to_string(), from_vector(chunk)))
        else {
            return Array::new();
        };
        let cell = self.cell_size;
        curves
            .iter()
            .map(|curve| {
                let points: PackedVector3Array = curve
                    .points
                    .iter()
                    .map(|&[x, y]| Vector3::new(x * cell.x, 0.0, y * cell.z))
                    .collect();
                let mut out = VarDictionary::new();
                match &curve.id {
                    CurveId::Region { region, index } => {
                        out.set(
                            &"region".to_variant(),
                            &Vector2i::new(region.0, region.1).to_variant(),
                        );
                        out.set(&"index".to_variant(), &i64::from(*index).to_variant());
                    }
                    CurveId::Row(row) => {
                        let row: PackedInt64Array = row.0.iter().map(|&part| part as i64).collect();
                        out.set(&"row".to_variant(), &row.to_variant());
                    }
                }
                out.set(&"points".to_variant(), &points.to_variant());
                out.set(
                    &"values".to_variant(),
                    &PackedFloat32Array::from(curve.values.as_slice()).to_variant(),
                );
                out
            })
            .collect()
    }

    /// The height of the ground at a position in Godot's world space, where its mesh at full detail
    /// stands above that point of the ground plane ([`ground_height`]): what a game stands a
    /// player or an object on. Without a `ground_stage`, the highest point of `volume_stage`'s
    /// surface there ([`volume_height`]), the top of the ground over a cave. NaN until the fields
    /// of `ground_stage` around it have arrived, or the surfaces within a cell of it are built, and
    /// without either stage.
    #[func]
    fn ground_height(&self, position: Vector3) -> f32 {
        let Some(worker) = &self.worker else {
            return f32::NAN;
        };
        let columns = [
            self.chunk_cells.x.max(1) as u32,
            self.chunk_cells.y.max(1) as u32,
        ];
        if self.ground_stage.is_empty() {
            let Some(rock) = &self.rock else {
                return f32::NAN;
            };
            return volume_height(
                [position.x, position.z],
                columns,
                |at| rock.built.get(&at).map(|built| &built.mesh),
                self.cell_size.to_array(),
            )
            .unwrap_or(f32::NAN);
        }
        let stage = self.ground_stage.to_string();
        ground_height(
            [position.x, position.z],
            columns,
            |at| worker.field(&stage, at),
            self.cell_size.to_array(),
        )
        .unwrap_or(f32::NAN)
    }

    /// A stage's value at a position in Godot's world space, on the ground plane, computed on the
    /// spot without generating chunks: a field's value, or a category's index. Only field, rules
    /// and blur stages that read no others can be sampled; another stage is reported as an error
    /// and gives NaN.
    #[func]
    fn sample(&self, stage: GString, position: Vector3) -> f32 {
        let Some(sampler) = &self.sampler else {
            godot_error!("wave forge: sample before start");
            return f32::NAN;
        };
        let at = [position.x / self.cell_size.x, position.z / self.cell_size.z];
        match sampler.sample(&stage.to_string(), at) {
            Ok(value) => value,
            Err(error) => {
                godot_error!("wave forge: {error}");
                f32::NAN
            }
        }
    }

    /// A world map: a stage's values over `size` of its own columns from `min`, row by row with x
    /// fastest, one value per column of a coarse stage. Computed on Godot's thread without
    /// generating chunks, for a game to read before play, a history's say. Empty, with an error
    /// reported, for a stage that cannot be sampled.
    #[func]
    fn atlas(&self, stage: GString, min: Vector2i, size: Vector2i) -> PackedFloat32Array {
        let Some(sampler) = &self.sampler else {
            godot_error!("wave forge: atlas before start");
            return PackedFloat32Array::new();
        };
        let area = [size.x.max(0) as u32, size.y.max(0) as u32];
        match sampler.atlas(
            &stage.to_string(),
            [i64::from(min.x), i64::from(min.y)],
            area,
        ) {
            Ok(values) => PackedFloat32Array::from(values.as_slice()),
            Err(error) => {
                godot_error!("wave forge: {error}");
                PackedFloat32Array::new()
            }
        }
    }

    /// Replaces the rows of a given table of facts, a history's villages say: an Array, typed or
    /// not, of Dictionaries, each with an `id`, a whole number from 0 that the game chooses, and a
    /// value for every column of the table, a number or, for a column of names, one of its names. Every generated table below
    /// it is computed again, and the stages that read any of them are generated again, with
    /// `stage_dropped` and `stage_ready` for their chunks. Returns whether the rows were taken; why
    /// not is reported as an error, and nothing changes.
    #[func]
    fn give_table(&mut self, table: GString, rows: AnyArray) -> bool {
        let (Some(facts), Some(sampler), Some(worker)) =
            (&self.facts, &mut self.sampler, &self.worker)
        else {
            godot_error!("wave forge: give_table before start");
            return false;
        };
        let mut given = Vec::with_capacity(rows.len());
        for row in rows.iter_shared() {
            let Ok(row) = row.try_to::<VarDictionary>() else {
                godot_error!("wave forge: table {table:?}: a row that is not a Dictionary: {row}");
                return false;
            };
            match given_row(&row) {
                Ok(row) => given.push(row),
                Err(message) => {
                    godot_error!("wave forge: table {table:?}: {message}");
                    return false;
                }
            }
        }
        let mut next = facts.clone();
        if let Err(error) = next
            .give(&table.to_string(), given)
            .and_then(|()| sampler.set_facts(next.clone()).map(|_| ()))
        {
            godot_error!("wave forge: {error}");
            return false;
        }
        worker.set_facts(next.clone());
        self.facts = Some(next);
        true
    }

    /// Takes away a point a Scatter stage placed, a felled tree say: the point with the id
    /// `point_sets` gave it, in the chunk it arrived in. The chunk is generated again without it,
    /// and stays so through eviction; `edits_log` saves it. Returns whether the chunk holds such a
    /// point; if not, that is reported as an error and nothing changes.
    #[func]
    fn remove_point(&mut self, stage: GString, chunk: Vector3i, id: i64) -> bool {
        let Some(point) = self.worker.as_ref().and_then(|worker| {
            worker
                .points(&stage.to_string(), from_vector(chunk))?
                .iter()
                .find(|point| local_id(point.id.local) == id)
                .cloned()
        }) else {
            godot_error!("wave forge: {stage} holds no point {id} in chunk {chunk}");
            return false;
        };
        self.edit(Edit::Remove {
            point: PointId::from(point.id),
            at: [point.position[0], point.position[1]],
        })
    }

    /// Paints a stroke of `brush` along `path`, points in Godot's world space, as edits of the
    /// world ([packs.md](packs.md#edits)): what they reach is generated again, and `edits_log`
    /// saves them. `brush` is a Dictionary: its `brush` is `"raise"`, `"smooth"`, `"dig"`, `"fill"`
    /// or `"remove"`; `stage` names the field or volume it paints, or `stages` the point stages a
    /// remove takes points of; `radius` is in cells; and `strength`, for raise and smooth, is how
    /// many cells a raise lifts the path by, negative to lower, or from 0 to 1 how far a smooth
    /// pulls. An editor undoes a stroke by giving back the `edits_log` it had before. Returns
    /// whether the stroke painted; if not, why is reported as an error and nothing changes.
    #[func]
    fn paint(&mut self, brush: VarDictionary, path: PackedVector3Array) -> bool {
        let Some(brush) = brush_of(&brush) else {
            godot_error!("wave forge: a brush of {brush}");
            return false;
        };
        let (Some(sampler), Some(worker)) = (&self.sampler, &self.worker) else {
            godot_error!("wave forge: paint before start");
            return false;
        };
        let cells: Vec<[f32; 3]> = path
            .as_slice()
            .iter()
            .map(|&at| self.cells_at(at))
            .collect();
        let edits = match stroke(&NodeCanvas { sampler, worker }, &brush, &cells) {
            Ok(edits) => edits,
            Err(error) => {
                godot_error!("wave forge: {error}");
                return false;
            }
        };
        let mut next = self.edits.clone();
        next.log.extend(edits);
        self.set_edits(next)
    }

    /// Raises a field stage's value, the ground's height in cells say, by `by` at the column under
    /// `position` in Godot's world space; a negative `by` digs. Readers of the field within their
    /// reach are generated again, and the raise stays through eviction; `edits_log` saves it.
    /// Returns whether the stage is a field; if not, that is reported as an error.
    #[func]
    fn raise(&mut self, stage: GString, position: Vector3, by: f32) -> bool {
        let Some(scale) = self
            .pack
            .as_ref()
            .and_then(|pack| pack.scale(&stage.to_string()))
        else {
            godot_error!("wave forge: no stage is named {stage}");
            return false;
        };
        let cell = [
            self.cell_size.x * scale as f32,
            self.cell_size.z * scale as f32,
        ];
        self.edit(Edit::Raise {
            stage: stage.to_string(),
            column: (
                (position.x / cell[0]).floor() as i64,
                (position.z / cell[1]).floor() as i64,
            ),
            by,
        })
    }

    /// Digs a ball of `radius` cells out of a Volume or Carve stage around `position` in Godot's
    /// world space ([packs.md](packs.md#edits)): the chunks it reaches are generated again with it,
    /// and it stays through eviction; `edits_log` saves it. Returns whether the stage is a volume;
    /// if not, that is reported as an error.
    #[func]
    fn dig(&mut self, stage: GString, position: Vector3, radius: f32) -> bool {
        let at = self.cells_at(position);
        self.edit(Edit::Dig {
            stage: stage.to_string(),
            at,
            radius,
        })
    }

    /// Fills a ball of `radius` cells into a Volume or Carve stage around `position`, as `dig`
    /// digs one.
    #[func]
    fn fill(&mut self, stage: GString, position: Vector3, radius: f32) -> bool {
        let at = self.cells_at(position);
        self.edit(Edit::Fill {
            stage: stage.to_string(),
            at,
            radius,
        })
    }

    /// Asks the stages' thread for a save of the world, which arrives as the `saved` signal.
    #[func]
    fn request_save(&self) {
        match &self.worker {
            Some(worker) => worker.request_save(),
            None => godot_error!("wave forge: request_save before start"),
        }
    }

    /// Brings the world back from a save `saved` gave: the edits, and every chunk of a frozen
    /// stage as it was first generated, even if the pack has changed since. Returns whether the
    /// text is a save the pack takes; if not, that is reported as an error and nothing changes.
    #[func]
    fn load_save(&mut self, text: GString) -> bool {
        let (Some(sampler), Some(worker)) = (&mut self.sampler, &self.worker) else {
            godot_error!("wave forge: load_save before start");
            return false;
        };
        let save = match Save::from_ron(&text.to_string()) {
            Ok(save) => save,
            Err(error) => {
                godot_error!("wave forge: {error}");
                return false;
            }
        };
        if let Err(error) = sampler.load(&save) {
            godot_error!("wave forge: {error}");
            return false;
        }
        self.edits = save.edits.clone();
        worker.load(save);
        true
    }

    /// The player's edits as text, for a save: a world is its pack, seed, facts and edits.
    #[func]
    fn edits_log(&self) -> GString {
        GString::from(self.edits.to_ron().as_str())
    }

    /// Sets `edits_text`: before `start`, the edits `start` applies; after, as `set_edits_log`.
    #[func]
    fn set_edits_text(&mut self, text: GString) {
        match Edits::from_ron(&text.to_string()) {
            Ok(edits) if self.worker.is_none() => self.edits = edits,
            Ok(edits) => {
                self.set_edits(edits);
            }
            Err(error) => godot_error!("wave forge: edits_text: {error}"),
        }
    }

    /// Replaces the player's edits with a log `edits_log` gave, from a save. Returns whether the
    /// text is such a log for this pack; if not, that is reported as an error and nothing
    /// changes.
    #[func]
    fn set_edits_log(&mut self, text: GString) -> bool {
        match Edits::from_ron(&text.to_string()) {
            Ok(edits) => self.set_edits(edits),
            Err(error) => {
                godot_error!("wave forge: {error}");
                false
            }
        }
    }

    /// Focuses the stages on one row of a table, a planet's say, whose columns stages read through
    /// `Row`; `id` is the row's id as `table_rows` gives it. The stages that read the table are
    /// generated again. Returns whether the table has the row; if not, that is reported as an
    /// error and nothing changes.
    #[func]
    fn focus_row(&mut self, table: GString, id: PackedInt64Array) -> bool {
        let (Some(sampler), Some(worker)) = (&mut self.sampler, &self.worker) else {
            godot_error!("wave forge: focus_row before start");
            return false;
        };
        let Ok(parts) = id
            .as_slice()
            .iter()
            .map(|&part| u64::try_from(part))
            .collect::<Result<Vec<u64>, _>>()
        else {
            godot_error!("wave forge: a row id of negative numbers, {id:?}");
            return false;
        };
        let id = RowId(parts);
        if let Err(error) = sampler.focus(&table.to_string(), id.clone()) {
            godot_error!("wave forge: {error}");
            return false;
        }
        worker.focus(&table.to_string(), id);
        true
    }

    /// The pack's parameters as `stack` or `pack_file` declares them, read again when the stack's
    /// pack or the file named changes; none if they hold no pack that loads.
    fn pack_param_defs(&mut self) -> BTreeMap<String, ParamDef> {
        // A file is read again when another is named, and a stack's text each time, since the
        // inspector edits it in place.
        let (source, text) = match &self.stack {
            Some(_) => {
                let text = self.pack_source().unwrap_or_default();
                (GString::from(text.as_str()), Some(text))
            }
            None => (self.pack_file.clone(), None),
        };
        let fresh = !matches!(&self.listed_params, Some((listed, _)) if *listed == source);
        if fresh {
            let text = text.unwrap_or_else(|| FileAccess::get_file_as_string(&source).to_string());
            let defs = Pack::parse(&text)
                .map(|pack| pack.params().clone())
                .unwrap_or_default();
            self.listed_params = Some((source, defs));
        }
        self.listed_params
            .as_ref()
            .map(|(_, defs)| defs.clone())
            .unwrap_or_default()
    }

    /// Sets the pack's parameters named in `values`, name to number, while the stages run: what
    /// reads a changed one is generated again, and nothing else. Returns whether every name is a
    /// parameter of the pack and every value in its range; if not, that is reported as an error
    /// and nothing changes.
    #[func]
    fn update_params(&mut self, values: VarDictionary) -> bool {
        let (Some(sampler), Some(worker)) = (&mut self.sampler, &self.worker) else {
            godot_error!("wave forge: update_params before start");
            return false;
        };
        let Some(parsed) = param_values(&values) else {
            godot_error!("wave forge: update_params takes names and numbers, not {values}");
            return false;
        };
        if let Err(error) = sampler.set_params(&parsed) {
            godot_error!("wave forge: {error}");
            return false;
        }
        worker.set_params(parsed);
        for (name, value) in values.iter_shared() {
            self.params.set(&name, &value);
        }
        true
    }

    /// The pack's parameters, in the order of their names: each a Dictionary with its `name`, its
    /// `default`, its range as `min` and `max`, and its `value` now. Empty before `start`.
    #[func]
    fn pack_params(&self) -> Array<VarDictionary> {
        let (Some(pack), Some(sampler)) = (&self.pack, &self.sampler) else {
            return Array::new();
        };
        pack.params()
            .iter()
            .map(|(name, param)| {
                let mut out = VarDictionary::new();
                out.set(&"name".to_variant(), &name.to_variant());
                out.set(&"default".to_variant(), &param.default.to_variant());
                out.set(&"min".to_variant(), &param.range.0.to_variant());
                out.set(&"max".to_variant(), &param.range.1.to_variant());
                out.set(&"value".to_variant(), &sampler.params()[name].to_variant());
                out
            })
            .collect()
    }

    /// A table's rows, in the order of their ids: each a Dictionary with its `id`, a
    /// PackedInt64Array (the game's id for a given row; the parent row's id and the row's index
    /// among its siblings for a generated one), and a value for every column, a float or, for a
    /// column of names, the name. Empty, with an error reported, for a table the pack does not name.
    #[func]
    fn table_rows(&self, table: GString) -> Array<VarDictionary> {
        let name = table.to_string();
        let (Some(pack), Some(facts)) = (&self.pack, &self.facts) else {
            godot_error!("wave forge: table_rows before start");
            return Array::new();
        };
        let (Some(kind), Some(rows)) = (pack.table(&name), facts.table(&name)) else {
            godot_error!("wave forge: no table is named {name:?}");
            return Array::new();
        };
        let names: Vec<Option<&[String]>> = match kind {
            TableKind::Given { columns } => columns
                .iter()
                .map(|(_, column)| match column {
                    Column::Number => None,
                    Column::Names(names) => Some(names.as_slice()),
                })
                .collect(),
            TableKind::Generated { columns, .. } => vec![None; columns.len()],
        };
        rows.rows
            .iter()
            .map(|row| {
                let mut out = VarDictionary::new();
                let id: PackedInt64Array = row.id.0.iter().map(|&part| part as i64).collect();
                out.set(&"id".to_variant(), &id.to_variant());
                for ((column, value), names) in rows.columns.iter().zip(&row.values).zip(&names) {
                    let value = match names {
                        Some(names) => GString::from(names[*value as usize].as_str()).to_variant(),
                        None => value.to_variant(),
                    };
                    out.set(&column.to_variant(), &value);
                }
                out
            })
            .collect()
    }

    /// The pack `text` holds as plain data, the shape GDScript edits it in: a Dictionary of its
    /// `version`, `stages`, `params`, `noises` and the rest, each stage a Dictionary of its `name`,
    /// `scale`, `persist` and `kind`, and a kind a Dictionary of one key, the kind's name, holding
    /// its fields as a pack file writes them. Whole numbers are ints. Empty, with the library's
    /// reason as an error, if `text` is no valid pack.
    #[func]
    fn pack_dictionary(text: GString) -> VarDictionary {
        crate::pack_data::pack_data(&text.to_string()).unwrap_or_else(|error| {
            godot_error!("wave forge: not a pack: {error}");
            VarDictionary::new()
        })
    }

    /// Every translation key a site of the pack `text` can be named by, sorted: `wf-place-<kind>`
    /// for each kind of its location tables, which a game translates in the `wave_forge` context
    /// (docs/reference/godot.md, "Place names in translation templates"). Empty, with the library's
    /// reason as an error, if `text` is no valid pack.
    #[func]
    fn pack_name_keys(text: GString) -> PackedStringArray {
        match Pack::parse(&text.to_string()) {
            Ok(pack) => pack.name_keys().iter().map(GString::from).collect(),
            Err(error) => {
                godot_error!("wave forge: not a pack: {error}");
                PackedStringArray::new()
            }
        }
    }

    /// The text of the pack `data` holds, in the shape `pack_dictionary` gives, written as a pack
    /// file is: what a stack of stages edited in the editor saves. A whole number may be an int or
    /// a float. Empty, with the reason as an error, where `data` holds something no pack holds,
    /// or is no valid pack.
    #[func]
    fn pack_text(data: VarDictionary) -> GString {
        crate::pack_data::pack_text(&data).map_or_else(
            |error| {
                godot_error!("wave forge: not a pack: {error}");
                GString::new()
            },
            |text| GString::from(text.as_str()),
        )
    }

    /// The categories a Rules stage names, in the order of their indices; empty for another stage.
    #[func]
    fn category_names(&self, stage: GString) -> PackedStringArray {
        self.pack
            .as_ref()
            .and_then(|pack| pack.kind(&stage.to_string()))
            .map_or_else(PackedStringArray::new, |kind| {
                kind.categories().into_iter().map(GString::from).collect()
            })
    }

    /// A Solve stage's town in a chunk: its site's `region` (Vector2i) or `row` (PackedInt64Array)
    /// as `sites` gives them, the site's levelled `height`, in
    /// cells, and the chunk's `tiles` as the rule set's tile indices, x fastest, then y, then z.
    /// Empty outside every site or before the chunk arrives.
    #[func]
    fn town(&self, stage: GString, chunk: Vector3i) -> VarDictionary {
        let mut out = VarDictionary::new();
        let Some(town) = self
            .worker
            .as_ref()
            .and_then(|worker| worker.tiles(&stage.to_string(), from_vector(chunk)))
        else {
            return out;
        };
        let tiles: PackedInt32Array = town.tiles.iter().map(|&tile| i32::from(tile)).collect();
        name_site(&mut out, &town.site);
        out.set(&"height".to_variant(), &town.height.to_variant());
        out.set(
            &"rules".to_variant(),
            &GString::from(town.rules.as_ref()).to_variant(),
        );
        out.set(&"tiles".to_variant(), &tiles.to_variant());
        out
    }

    /// A town chunk's placements of the modules named in `names` (every module if it is empty),
    /// ready to draw, as [`crate::WaveForgeWorld`]'s `instance_sets` gives a city's: one dictionary
    /// per module with its `name`, its `transforms` as a MultiMesh buffer (each module's unit model
    /// turned by its tile, scaled to the cell, at the cell's centre, raised to the site's height),
    /// and each instance's `ids`. Empty outside every site or before the chunk arrives.
    #[func]
    fn town_instance_sets(
        &self,
        stage: GString,
        chunk: Vector3i,
        names: PackedStringArray,
    ) -> Array<VarDictionary> {
        let wanted: Vec<String> = names.as_slice().iter().map(ToString::to_string).collect();
        let drawn = |name: &str| wanted.is_empty() || wanted.iter().any(|w| w == name);
        let Some((sets, lift)) = self.town_sets(&stage.to_string(), from_vector(chunk), drawn)
        else {
            return Array::new();
        };
        sets.into_iter()
            .map(|set| {
                let mut transforms = set.transforms(self.cell_size.to_array());
                for row in transforms.chunks_mut(12) {
                    row[7] += lift;
                }
                let ids: PackedInt64Array = set.ids.iter().map(|id| local_id(id.local)).collect();
                let mut out = VarDictionary::new();
                out.set(&"name".to_variant(), &GString::from(&set.name).to_variant());
                out.set(
                    &"transforms".to_variant(),
                    &PackedFloat32Array::from(transforms.as_slice()).to_variant(),
                );
                out.set(&"ids".to_variant(), &ids.to_variant());
                out
            })
            .collect()
    }

    /// Gives every module a collider of its item's shapes in `library`, the kit a module set was
    /// imported from, as `set_collision_shape` gives one: an item's only shape if it has one at no
    /// offset, or else all of them as one concave shape, where a `GridMap` would place them in the
    /// item's cell. A module whose item has no shapes keeps what it has.
    #[func]
    fn set_collision_shapes(&mut self, library: Gd<MeshLibrary>) {
        for (module, shape) in crate::kit::item_shapes(&library) {
            self.set_collision_shape(GString::from(module.as_str()), Some(shape));
        }
    }

    /// Gives every cell of a town's `module` a collider of `shape` in the chunks within
    /// `collider_radius`, turned by the tile's rotation and centred on the cell, for every Solve
    /// stage. The shape is sized for one cell in Godot's world units. Null takes it away.
    #[func]
    fn set_collision_shape(&mut self, module: GString, shape: Option<Gd<Shape3D>>) {
        match shape {
            Some(shape) => self.collision_shapes.insert(module.to_string(), shape),
            None => self.collision_shapes.remove(&module.to_string()),
        };
        self.free_bodies();
        // Navigation is baked again with the new shapes over the next frames.
        self.shape_faces.clear();
        self.navigation.clear();
    }

    /// The chunks within `occluder_radius` that have occluders, of their towns' solid cells.
    #[func]
    fn occluder_chunks(&self) -> Array<Vector3i> {
        self.occluders.chunks().map(to_vector).collect()
    }

    /// The boxes that occlude for a chunk: an `AABB` per box of its towns' solid cells in Godot's
    /// world, raised to each town's site, the boxes together covering each solid cell once. Empty
    /// for a chunk without a town.
    #[func]
    fn town_occluders(&self, chunk: Vector3i) -> Array<Aabb> {
        self.town_occluder_boxes(from_vector(chunk))
            .iter()
            .map(|cell_box| {
                let min = Vector3::from_array(cell_box.min);
                Aabb::new(min, Vector3::from_array(cell_box.max) - min)
            })
            .collect()
    }

    /// The chunks whose navigation mesh is in the map. A chunk being baked again keeps its last
    /// mesh until the new one is in.
    #[func]
    fn navigation_chunks(&self) -> Array<Vector3i> {
        self.navigation.chunks().map(to_vector).collect()
    }

    /// The names of the modules in the rule set `rules` (as `rules_files` names it) that carry
    /// `tag`, each once: what a game assigns shapes and scenes by. Empty for a tile set, which has
    /// no tags, or a rule set the node has not loaded.
    #[func]
    fn modules_tagged(&self, rules: GString, tag: GString) -> PackedStringArray {
        let Some(file) = self.rules.get(&rules.to_string()) else {
            return PackedStringArray::new();
        };
        let mut names: Vec<&str> = file
            .tiles_tagged(&tag.to_string())
            .into_iter()
            .map(|tile| file.name(tile))
            .collect();
        names.sort_unstable();
        names.dedup();
        names.into_iter().map(GString::from).collect()
    }

    /// The material a chunk's ground is drawn with, when `ground_material_stage` gives it one of
    /// its own; null otherwise or before the chunk's ground is built.
    #[func]
    fn ground_material_of(&self, chunk: Vector3i) -> Option<Gd<ShaderMaterial>> {
        self.chunk_materials.get(&from_vector(chunk)).cloned()
    }

    /// The reference ground shader's code, to copy into a shader of a game's own.
    #[func]
    fn ground_shader_code(&self) -> GString {
        GString::from(crate::shaders::GROUND)
    }

    /// The chunks that have grass.
    #[func]
    fn grass_chunks(&self) -> Array<Vector3i> {
        self.grass
            .iter()
            .flat_map(Grass::chunks)
            .map(to_vector)
            .collect()
    }

    /// A chunk's copy of the grass material, holding its cover and ground heights; null without
    /// grass there.
    #[func]
    fn grass_material_of(&self, chunk: Vector3i) -> Option<Gd<ShaderMaterial>> {
        self.grass.as_ref()?.material_of(from_vector(chunk))
    }

    /// The reference vegetation shader's code: give it to the material of a plant's mesh bound in
    /// `scenes`, and the plant bends in the global wind by the phase and stiffness its MultiMesh
    /// carries per instance.
    #[func]
    fn vegetation_shader_code(&self) -> GString {
        GString::from(crate::shaders::VEGETATION)
    }

    /// The reference fluid shader's code, to copy into a shader of a game's own.
    #[func]
    fn fluid_shader_code(&self) -> GString {
        GString::from(crate::shaders::FLUID)
    }

    /// The reference grass shader's code, to copy into a shader of a game's own.
    #[func]
    fn grass_shader_code(&self) -> GString {
        GString::from(crate::shaders::GRASS)
    }

    /// The chunks whose ground is built.
    #[func]
    fn ground_chunks(&self) -> Array<Vector3i> {
        self.grounds.keys().map(|&chunk| to_vector(chunk)).collect()
    }

    /// The `RenderingServer` mesh a chunk's ground is drawn with; an invalid RID if it is not.
    #[func]
    fn ground_mesh_of(&self, chunk: Vector3i) -> Rid {
        self.grounds
            .get(&from_vector(chunk))
            .map_or(Rid::Invalid, |&(_, mesh, _)| mesh)
    }

    /// The chunks of `far_ground_stage` whose far ground is drawn.
    #[func]
    fn far_ground_chunks(&self) -> Array<Vector3i> {
        self.far_grounds
            .keys()
            .map(|&chunk| to_vector(chunk))
            .collect()
    }

    /// The chunks of `volume_stage` whose surface is built, those without triangles included.
    #[func]
    fn volume_chunks(&self) -> Array<Vector3i> {
        layer_chunks(self.rock.as_ref())
    }

    /// A chunk's volume surface, relative to the chunk's corner on the ground plane: `positions`
    /// and `normals` (PackedVector3Array), `indices` (PackedInt32Array, three per triangle,
    /// clockwise seen from the empty side as Godot's front faces are) and `materials`
    /// (PackedByteArray, each vertex's material, empty for a stage without materials). Empty if the
    /// surface is not built.
    #[func]
    fn volume_surface(&self, chunk: Vector3i) -> VarDictionary {
        layer_surface(self.rock.as_ref(), chunk)
    }

    /// The `RenderingServer` mesh a chunk's volume surface is drawn with; an invalid RID if it is
    /// not drawn.
    #[func]
    fn volume_mesh_of(&self, chunk: Vector3i) -> Rid {
        layer_mesh(self.rock.as_ref(), chunk)
    }

    /// The chunks of `fluid_stage` whose surface is built, those without triangles included.
    #[func]
    fn fluid_chunks(&self) -> Array<Vector3i> {
        layer_chunks(self.fluid.as_ref())
    }

    /// A chunk's fluid surface, as `volume_surface` gives the volume's.
    #[func]
    fn fluid_surface(&self, chunk: Vector3i) -> VarDictionary {
        layer_surface(self.fluid.as_ref(), chunk)
    }

    /// The `RenderingServer` mesh a chunk's fluid surface is drawn with; an invalid RID if it is
    /// not drawn.
    #[func]
    fn fluid_mesh_of(&self, chunk: Vector3i) -> Rid {
        layer_mesh(self.fluid.as_ref(), chunk)
    }

    /// A scene of plain nodes holding what the node draws over the chunks from `from` to `to`,
    /// both included, for a game to save and open without the extension. Under a node per chunk:
    /// its ground as a `MeshInstance3D` and a `StaticBody3D` holding its `HeightMapShape3D`; its
    /// volume surface and a body holding its `ConcavePolygonShape3D`; its fluid's surface; and
    /// every bound scene its stages placed, a kind drawn as a MultiMesh as a
    /// `MultiMeshInstance3D`, any other as an instance of its scene. Each keeps its material.
    /// Grass and the far ground are left out. Null, with an error, if a chunk's ground or surfaces
    /// are not built yet, or a scene is still loading.
    #[func]
    fn bake(&self, from: Vector3i, to: Vector3i) -> Option<Gd<PackedScene>> {
        match self.baked(from_vector(from), from_vector(to)) {
            Ok(scene) => Some(scene),
            Err(error) => {
                godot_error!("wave forge: cannot bake: {error}");
                None
            }
        }
    }

    /// Turns what a designer changed in a scene [`Self::bake`] gave, `baked` as instanced, into
    /// edits of the world: a point placed as a node that was moved or turned is moved there, and
    /// one that was deleted is removed, as `remove_point` removes one. Pieces, the ground, surfaces
    /// and MultiMeshes are the generator's, and changes to them are not carried. The world is
    /// generated again with the edits, and a bake after it holds them; `bake_keeping` carries over
    /// the nodes the designer added as well. Returns whether the edits were taken; if not, that is
    /// reported as an error.
    #[func]
    fn keep_bake_edits(&mut self, baked: Gd<Node3D>) -> bool {
        let cell = self.cell_size;
        let mut edits = self.edits.clone();
        for holder in baked.get_children().iter_shared() {
            if !holder.has_meta(BAKED_CHUNK) {
                continue;
            }
            // The points still there, by the stage and chunk their id names and their id.
            let mut present: HashMap<(String, Vector3i, i64), Gd<Node3D>> = HashMap::new();
            for child in holder.get_children().iter_shared() {
                if let (true, Ok(node)) = (
                    child.has_meta(BAKED_POINT),
                    child.clone().try_cast::<Node3D>(),
                ) {
                    let mark: VarDictionary = child.get_meta(BAKED_POINT).to();
                    present.insert(point_key(&mark), node);
                }
            }
            let listed: VarArray = holder.get_meta(BAKED_POINTS).to();
            for mark in listed.iter_shared() {
                let mark: VarDictionary = mark.to();
                let (_, chunk, id) = point_key(&mark);
                let point = PointId {
                    chunk: (chunk.x, chunk.y),
                    local: u64::try_from(id).expect("an id `bake` gave"),
                };
                let at: Vector2 = mark.at("at").to();
                let placed: Transform3D = mark.at("transform").to();
                match present.get(&point_key(&mark)) {
                    None => edits.push(Edit::Remove {
                        point,
                        at: [at.x, at.y],
                    }),
                    Some(node) if !node.get_transform().approx_eq(&placed) => {
                        let now = node.get_transform();
                        let x = now.basis.col_a();
                        edits.push(Edit::Move {
                            point,
                            from: [at.x, at.y],
                            to: [
                                now.origin.x / cell.x,
                                now.origin.z / cell.z,
                                now.origin.y / cell.y,
                            ],
                            // A point's basis turns it about Godot's +Y, taking +X toward -Z.
                            turn: ((-x.z).atan2(x.x) / std::f32::consts::TAU).rem_euclid(1.0),
                        });
                    }
                    Some(_) => {}
                }
            }
        }
        self.set_edits(edits)
    }

    /// A bake of the chunks from `from` to `to`, as [`Self::bake`] gives, that keeps the nodes a
    /// designer added under the chunks of `old`, an earlier bake as instanced: every node there
    /// that `bake` did not make is copied under the same chunk. With `keep_bake_edits` first, a
    /// linked bake is regenerated with the designer's edits kept. Null, with an error, as `bake`.
    #[func]
    fn bake_keeping(
        &self,
        from: Vector3i,
        to: Vector3i,
        old: Gd<Node3D>,
    ) -> Option<Gd<PackedScene>> {
        let scene = self.bake(from, to)?;
        let root = scene.instantiate_as::<Node3D>();
        for holder in old.get_children().iter_shared() {
            if !holder.has_meta(BAKED_CHUNK) {
                continue;
            }
            let Some(mut target) =
                root.get_node_or_null(&NodePath::from(holder.get_name().to_string().as_str()))
            else {
                continue;
            };
            for child in holder.get_children().iter_shared() {
                if child.has_meta(BAKED_PART) || child.has_meta(BAKED_POINT) {
                    continue;
                }
                target.add_child(&child.duplicate_node());
            }
        }
        own(&root.clone().upcast(), &root.clone().upcast());
        let mut kept = PackedScene::new_gd();
        let packed = kept.pack(&root);
        root.free();
        if packed != Error::OK {
            godot_error!("wave forge: cannot bake: packing the scene failed: {packed:?}");
            return None;
        }
        Some(kept)
    }

    /// The chunks whose candidates of `candidates_stage` are drawn.
    #[func]
    fn candidate_chunks(&self) -> Array<Vector3i> {
        self.candidates
            .keys()
            .map(|&chunk| to_vector(chunk))
            .collect()
    }

    /// What the drawn candidates of `candidates_stage` came to, over every chunk drawn: each
    /// verdict's `name` (`"kept"`, or the modifier that rejected it, as `"chance"`, `"height"`,
    /// `"slope"`, `"condition 0"`, `"water"`, `"sites"`, `"blocked"` or `"spacing"`), the `colour`
    /// its candidates are drawn in, and their `count`.
    #[func]
    fn candidate_legend(&self) -> Array<VarDictionary> {
        let mut totals: BTreeMap<String, i64> = BTreeMap::new();
        for (_, _, counts) in self.candidates.values() {
            for (name, count) in counts {
                *totals.entry(name.clone()).or_default() += count;
            }
        }
        totals
            .into_iter()
            .map(|(name, count)| {
                let mut out = VarDictionary::new();
                out.set(
                    &"colour".to_variant(),
                    &verdict_colour_of(&name).to_variant(),
                );
                out.set(&"name".to_variant(), &name.to_variant());
                out.set(&"count".to_variant(), &count.to_variant());
                out
            })
            .collect()
    }

    /// The drawn candidate of `candidates_stage` nearest `position`, in Godot's world space, no
    /// farther than `radius` along the ground, for a viewer to say why it went as it did: its
    /// `verdict` and `colour` as [`WaveForgeStages::candidate_legend`] names them, its `position` on
    /// the ground, and what the stage's modifiers read there
    /// ([packs.md](packs.md#scatter)): the ground's `height` in cells, its `slope` and the
    /// `water_depth` for a stage with those modifiers, and `conditions`, each condition of `when`
    /// in order with the `value` it tests and whether it `holds`. Empty when none lies that near.
    #[func]
    fn candidate_near(&self, position: Vector3, radius: f32) -> VarDictionary {
        let stage = self.candidates_stage.to_string();
        let mut out = VarDictionary::new();
        let Some(worker) = &self.worker else {
            return out;
        };
        let cell = self.cell_size;
        let mut nearest: Option<(f32, &Judgement)> = None;
        for chunk in self.candidates.keys() {
            for judged in worker.scatter_report(&stage, *chunk).unwrap_or_default() {
                let distance =
                    (judged.at[0] * cell.x - position.x).hypot(judged.at[1] * cell.z - position.z);
                if distance <= radius && nearest.is_none_or(|(best, _)| distance < best) {
                    nearest = Some((distance, judged));
                }
            }
        }
        let Some((_, judged)) = nearest else {
            return out;
        };
        let readings = &judged.readings;
        out.set(
            &"verdict".to_variant(),
            &verdict_name(judged.verdict).to_variant(),
        );
        out.set(
            &"colour".to_variant(),
            &verdict_colour(judged.verdict).to_variant(),
        );
        out.set(
            &"position".to_variant(),
            &Vector3::new(
                judged.at[0] * cell.x,
                (readings.height + 0.5) * cell.y,
                judged.at[1] * cell.z,
            )
            .to_variant(),
        );
        out.set(&"height".to_variant(), &readings.height.to_variant());
        if let Some(slope) = readings.slope {
            out.set(&"slope".to_variant(), &slope.to_variant());
        }
        if let Some(depth) = readings.water_depth {
            out.set(&"water_depth".to_variant(), &depth.to_variant());
        }
        let conditions: Array<VarDictionary> = readings
            .conditions
            .iter()
            .map(|&(value, holds)| {
                let mut condition = VarDictionary::new();
                condition.set(&"value".to_variant(), &value.to_variant());
                condition.set(&"holds".to_variant(), &holds.to_variant());
                condition
            })
            .collect();
        out.set(&"conditions".to_variant(), &conditions.to_variant());
        out
    }

    /// The drawn candidate of `candidates_stage` under a ray from `from` along `along`, both in
    /// Godot's world space, as [`WaveForgeStages::candidate_near`] gives it within a cell of where
    /// the ray first meets the ground the candidates stand on, the field the Scatter stage reads
    /// as its height: what the editor dock shows under the mouse. Empty when the ray meets that
    /// ground nowhere generated within 2 000 units, or no candidate lies near where it does.
    #[func]
    fn candidate_under(&self, from: Vector3, along: Vector3) -> VarDictionary {
        let (Some(worker), Some(pack)) = (&self.worker, &self.pack) else {
            return VarDictionary::new();
        };
        let Some(StageKind::Scatter { height, .. }) = pack.kind(&self.candidates_stage.to_string())
        else {
            return VarDictionary::new();
        };
        let columns = [
            self.chunk_cells.x.max(1) as u32,
            self.chunk_cells.y.max(1) as u32,
        ];
        let cell = self.cell_size;
        let along = along.normalized();
        let step = 0.5 * cell.x.min(cell.z);
        let mut travelled = 0.0;
        while travelled < 2000.0 {
            let at = from + along * travelled;
            let ground = ground_height(
                [at.x, at.z],
                columns,
                |chunk| worker.field(height, chunk),
                cell.to_array(),
            );
            if ground.is_some_and(|ground| at.y <= ground) {
                return self.candidate_near(at, cell.x);
            }
            travelled += step;
        }
        VarDictionary::new()
    }

    /// Generates the whole world of the pack's bound ahead of time, on a thread of its own: the
    /// node's `targets` over every chunk, each chunk's products kept under `directory` the moment
    /// it is done, which `play_directory` then plays ([packs.md](packs.md#a-whole-world-ahead-of-time)).
    /// Progress arrives as `world_run_progress`, the end as `world_run_finished`;
    /// `cancel_world_run` stops it, and running it again resumes from what `directory` holds. The
    /// node has to have started, since the run generates as the node does. Returns whether it
    /// started; not if the node has not, or a run is under way.
    #[func]
    fn run_world(&mut self, directory: GString) -> bool {
        let Some(build) = self.builder.clone() else {
            godot_error!("wave forge: run_world before start");
            return false;
        };
        if self.world_run.is_some() {
            godot_error!("wave forge: a world run is under way");
            return false;
        }
        let directory = crate::paths::directory_path(&directory);
        let targets: Vec<String> = self
            .targets
            .as_slice()
            .iter()
            .map(ToString::to_string)
            .collect();
        // The world as the node has it now: the tables given, the edits made and the parameters
        // set since it started, which the builder, made at the start, does not hold.
        let facts = self.facts.clone().expect("set when the node started");
        let edits = self.edits.clone();
        let params = self
            .sampler
            .as_ref()
            .expect("set when the node started")
            .params()
            .clone();
        let (sent, progress) = std::sync::mpsc::channel();
        let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop = Arc::clone(&cancel);
        let thread = std::thread::Builder::new()
            .name("wave forge world run".to_owned())
            .spawn(move || {
                let run = build().and_then(|mut runtime| {
                    runtime
                        .set_facts(facts)
                        .and_then(|_| runtime.set_params(&params))
                        .and_then(|_| runtime.set_edits(&edits))
                        .map_err(|error| error.to_string())?;
                    let targets: Vec<&str> = targets.iter().map(String::as_str).collect();
                    let mut store = DirectoryStore::new(directory);
                    runtime
                        .run_world(&targets, &mut store, |state| {
                            let _ = sent.send(RunEvent::Progress(state));
                            if stop.load(std::sync::atomic::Ordering::Relaxed) {
                                std::ops::ControlFlow::Break(())
                            } else {
                                std::ops::ControlFlow::Continue(())
                            }
                        })
                        .map_err(|error| error.to_string())
                });
                let _ = sent.send(RunEvent::Ended(run));
            })
            .expect("a thread");
        crate::ending::join_before_exit(thread);
        self.world_run = Some(WorldRunning { progress, cancel });
        true
    }

    /// Signals what the world run under way reported since the last frame, and forgets it once
    /// it has ended.
    fn update_world_run(&mut self) {
        let Some(run) = &self.world_run else {
            return;
        };
        let events = received(&run.progress);
        for event in events {
            match event {
                RunEvent::Progress(state) => {
                    self.signals().world_run_progress().emit(
                        state.done as i64,
                        state.total as i64,
                        &stage_costs(&state.stages),
                    );
                }
                RunEvent::Ended(result) => {
                    self.world_run = None;
                    let state = result.unwrap_or_else(|reason| {
                        godot_error!("wave forge: the world run failed: {reason}");
                        RunProgress {
                            done: 0,
                            total: 0,
                            skipped: 0,
                            held: 0,
                            stages: Vec::new(),
                        }
                    });
                    self.signals()
                        .world_run_finished()
                        .emit(state.done as i64, state.total as i64);
                }
            }
        }
    }

    /// Stops the world run under way after the chunk it is on; `run_world` again resumes it.
    #[func]
    fn cancel_world_run(&mut self) {
        if let Some(run) = &self.world_run {
            run.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// The chunks that have a static body: their ground, their volume's surface, and their towns'
    /// modules.
    #[func]
    fn collider_chunks(&self) -> Array<Vector3i> {
        self.bodies.keys().map(|&chunk| to_vector(chunk)).collect()
    }

    /// A Sites, TableSites or Locations stage's sites that overlap a chunk: what names each one,
    /// its `region` (Vector2i), the `row` (PackedInt64Array) of its table it stands for, or its
    /// `region` and `index` in a location table along with its `kind`, and then its name as a
    /// translation key `name_key` (`wf-place-<kind>`) with the arguments `name_args` a translation
    /// may use (`region_x`, `region_y`, `index`), for `tr(name_key, "wave_forge")` and `format`;
    /// the chunks it covers from `min` up to but not including `max` (Vector2i, along the
    /// lattice's x and y); and its levelled `height` in cells. Empty if there are none or the
    /// chunk has not arrived.
    #[func]
    fn sites(&self, stage: GString, chunk: Vector3i) -> Array<VarDictionary> {
        let Some(sites) = self
            .worker
            .as_ref()
            .and_then(|worker| worker.sites(&stage.to_string(), from_vector(chunk)))
        else {
            return Array::new();
        };
        sites.iter().map(site_dictionary).collect()
    }

    /// The site of a Sites stage nearest a position in Godot's world space, on the ground plane,
    /// found on Godot's thread without generating chunks ([packs.md](packs.md#sites)), as `sites`
    /// gives a site; its regions up to `within` regions from the position's are searched. Empty if
    /// none of them holds a site, or, with an error reported, for a stage that is no Sites stage.
    #[func]
    fn locate(&self, stage: GString, position: Vector3, within: i32) -> VarDictionary {
        let Some(sampler) = &self.sampler else {
            godot_error!("wave forge: locate before start");
            return VarDictionary::new();
        };
        let at = [position.x / self.cell_size.x, position.z / self.cell_size.z];
        match sampler.locate(&stage.to_string(), at, within.max(0) as u32) {
            Ok(site) => site
                .as_ref()
                .map_or_else(VarDictionary::new, site_dictionary),
            Err(error) => {
                godot_error!("wave forge: {error}");
                VarDictionary::new()
            }
        }
    }

    /// An Assemble stage's pieces overlapping a chunk, one dictionary each: its `piece` name, what
    /// names its site (as `sites` gives it), its `id` among the chunk's instances, its `transform`
    /// in Godot's world space (at the centre of its footprint on its floor, turned about +Y, where
    /// a scene of the piece authored at turn 0 with its footprint centred on its origin goes), and
    /// the cells it covers from `min` up to but not including `max`.
    #[func]
    fn stamps(&self, stage: GString, chunk: Vector3i) -> Array<VarDictionary> {
        let Some(stamps) = self
            .worker
            .as_ref()
            .and_then(|worker| worker.stamps(&stage.to_string(), from_vector(chunk)))
        else {
            return Array::new();
        };
        let cell = self.cell_size;
        stamps
            .iter()
            .map(|stamp| {
                let [row_x, row_y, row_z] = stamp.y_up_basis().map(Vector3::from_array);
                let [x, y, floor] = stamp.position;
                let transform = Transform3D::new(
                    Basis::from_rows(row_x, row_y, row_z),
                    Vector3::new(x * cell.x, floor * cell.y, y * cell.z),
                );
                let columns = |at: [i64; 2]| {
                    Vector2i::new(
                        i32::try_from(at[0]).expect("a cell in Godot's range"),
                        i32::try_from(at[1]).expect("a cell in Godot's range"),
                    )
                };
                let mut out = VarDictionary::new();
                out.set(
                    &"piece".to_variant(),
                    &GString::from(&*stamp.piece).to_variant(),
                );
                name_site(&mut out, &stamp.site);
                out.set(&"id".to_variant(), &local_id(stamp.id.local).to_variant());
                out.set(&"transform".to_variant(), &transform.to_variant());
                out.set(&"min".to_variant(), &columns(stamp.min).to_variant());
                out.set(&"max".to_variant(), &columns(stamp.max).to_variant());
                out
            })
            .collect()
    }

    /// A Scatter stage's points in a chunk, one dictionary per kind: its `kind`, its `transforms`
    /// as a MultiMesh buffer of twelve floats per point in Godot's world space (turned about +Y,
    /// unscaled, standing on the field), and each point's `ids` within the chunk.
    #[func]
    fn point_sets(&self, stage: GString, chunk: Vector3i) -> Array<VarDictionary> {
        let Some(points) = self
            .worker
            .as_ref()
            .and_then(|worker| worker.points(&stage.to_string(), from_vector(chunk)))
        else {
            return Array::new();
        };
        let cell = self.cell_size;
        let mut kinds: BTreeMap<&str, (Vec<f32>, Vec<i64>)> = BTreeMap::new();
        for point in points {
            let (transforms, ids) = kinds.entry(&point.kind).or_default();
            let [row_x, row_y, row_z] = point.y_up_basis();
            let [x, y, height] = point.position;
            transforms.extend(row_x);
            transforms.push(x * cell.x);
            transforms.extend(row_y);
            transforms.push(height * cell.y);
            transforms.extend(row_z);
            transforms.push(y * cell.z);
            ids.push(local_id(point.id.local));
        }
        kinds
            .into_iter()
            .map(|(kind, (transforms, ids))| {
                let mut out = VarDictionary::new();
                out.set(&"kind".to_variant(), &GString::from(kind).to_variant());
                out.set(
                    &"transforms".to_variant(),
                    &PackedFloat32Array::from(transforms.as_slice()).to_variant(),
                );
                out.set(
                    &"ids".to_variant(),
                    &PackedInt64Array::from(ids.as_slice()).to_variant(),
                );
                out
            })
            .collect()
    }

    /// The stage names of the loaded pack.
    #[func]
    fn stage_names(&self) -> PackedStringArray {
        self.pack
            .as_ref()
            .map_or_else(PackedStringArray::new, |pack| {
                pack.stage_names().map(GString::from).collect()
            })
    }

    /// The loaded pack's water: its `level`, the sea's height in cells of height, below which a
    /// game draws water. Empty if the pack declares none.
    #[func]
    fn water(&self) -> VarDictionary {
        let mut out = VarDictionary::new();
        if let Some(water) = self.pack.as_ref().and_then(|pack| pack.water()) {
            out.set("level", water.level);
        }
        out
    }

    /// What the node has cost Godot's thread: `process_ms_median`, `_p99` and `_max` over recent
    /// frames, once there are some; and what its slowest frame since the start spent the time on:
    /// `slowest_frame_ms` in all, `slowest_frame_events` signals emitted in
    /// `slowest_frame_signals_ms` (the handlers connected to them included),
    /// `slowest_frame_grounds` chunks given ground in `slowest_frame_grounds_ms`, and
    /// `slowest_frame_bodies` chunks given a body in `slowest_frame_bodies_ms`, and
    /// `slowest_frame_navigation_ms` keeping navigation regions. And `stages`: what
    /// each stage has cost on the stages' thread, by name, as `products`, `ms` in all and
    /// `slowest_ms` for one product. And `pending_signals`, `pending_grounds` and
    /// `pending_colliders`: the signals, grounds and bodies waiting for a later frame. And
    /// `navigation_baked`, how many navigation bakes have gone into their regions. And
    /// `sea_drawn`, whether the sea's plane is drawn.
    #[func]
    fn stats(&self) -> VarDictionary {
        let mut out = VarDictionary::new();
        out.set(&"sea_drawn".to_variant(), &self.sea.is_some().to_variant());
        let stages = stage_costs(self.worker.iter().flat_map(|worker| worker.timings()));
        out.set(&"stages".to_variant(), &stages.to_variant());
        out.set(
            &"pending_signals".to_variant(),
            &(self.pending.len() as i64).to_variant(),
        );
        out.set(
            &"pending_grounds".to_variant(),
            &(self.ground_due.len() as i64).to_variant(),
        );
        let pending = [&self.rock, &self.fluid]
            .into_iter()
            .flatten()
            .map(VolumeLayer::pending)
            .sum::<usize>();
        out.set(
            &"pending_volumes".to_variant(),
            &(pending as i64).to_variant(),
        );
        out.set(
            &"last_frame_ms".to_variant(),
            &self.last_frame_ms.to_variant(),
        );
        out.set(
            &"volume_surfaces".to_variant(),
            &(self.volume_revisions as i64).to_variant(),
        );
        out.set(
            &"volume_surfaces_ms".to_variant(),
            &self.volume_ms.to_variant(),
        );
        out.set(
            &"pending_far_grounds".to_variant(),
            &(self.far_due.len() as i64).to_variant(),
        );
        out.set(
            &"pending_colliders".to_variant(),
            &(self.bodies_pending as i64).to_variant(),
        );
        out.set(
            &"navigation_baked".to_variant(),
            &self.navigation.baked.to_variant(),
        );
        out.set(
            &"pending_placements".to_variant(),
            &(self.placements.pending() as i64).to_variant(),
        );
        let (nodes, instances) = self.placements.counts();
        out.set(&"placed_nodes".to_variant(), &(nodes as i64).to_variant());
        out.set(
            &"placed_instances".to_variant(),
            &(instances as i64).to_variant(),
        );
        out.set(
            &"pooled_nodes".to_variant(),
            &(self.placements.pooled() as i64).to_variant(),
        );
        let slowest = self.slowest_frame;
        for (key, value) in [
            ("slowest_frame_ms", slowest.ms),
            ("slowest_frame_signals_ms", slowest.signals_ms),
            ("slowest_frame_grounds_ms", slowest.grounds_ms),
            ("slowest_frame_bodies_ms", slowest.bodies_ms),
            ("slowest_frame_navigation_ms", slowest.navigation_ms),
            ("slowest_frame_placements_ms", slowest.placements_ms),
        ] {
            out.set(&key.to_variant(), &value.to_variant());
        }
        for (key, count) in [
            ("slowest_frame_events", slowest.events),
            ("slowest_frame_grounds", slowest.grounds),
            ("slowest_frame_bodies", slowest.bodies),
            ("slowest_frame_placed", slowest.placed),
        ] {
            out.set(&key.to_variant(), &(count as i64).to_variant());
        }
        if let Some([median, p99, max]) = self.process_ms.summary() {
            out.set(&"process_ms_median".to_variant(), &median.to_variant());
            out.set(&"process_ms_p99".to_variant(), &p99.to_variant());
            out.set(&"process_ms_max".to_variant(), &max.to_variant());
        }
        out
    }
}

impl WaveForgeStages {
    /// Adds `edit` to the player's edits and hands them on.
    fn edit(&mut self, edit: Edit) -> bool {
        let mut next = self.edits.clone();
        next.push(edit);
        self.set_edits(next)
    }

    /// Hands `edits` to the sampler and the stages' thread, if the pack takes them.
    fn set_edits(&mut self, edits: Edits) -> bool {
        let (Some(sampler), Some(worker)) = (&mut self.sampler, &self.worker) else {
            godot_error!("wave forge: an edit before start");
            return false;
        };
        if let Err(error) = sampler.set_edits(&edits) {
            godot_error!("wave forge: {error}");
            return false;
        }
        worker.set_edits(edits.clone());
        self.edits = edits;
        true
    }

    fn chunk_shape(&self) -> ChunkShape {
        let cells = self.chunk_cells;
        ChunkShape {
            x: cells.x.max(1) as u32,
            y: cells.y.max(1) as u32,
            z: cells.z.max(1) as u32,
        }
    }

    /// A town chunk's placements of the modules `wanted` names, unscaled, and how far to raise
    /// them: the site's height in Godot's units. `None` for a stage that is not a Solve stage, or
    /// a chunk outside every site or not yet arrived.
    fn town_sets(
        &self,
        stage: &str,
        chunk: ChunkCoord,
        wanted: impl Fn(&str) -> bool,
    ) -> Option<(Vec<InstanceSet>, f32)> {
        town_sets(
            self.worker.as_ref()?,
            &self.rules,
            self.chunk_shape(),
            self.cell_size,
            stage,
            chunk,
            wanted,
        )
    }

    /// A position in Godot's world space in cells, along the lattice's x and y and up.
    fn cells_at(&self, position: Vector3) -> [f32; 3] {
        [
            position.x / self.cell_size.x,
            position.z / self.cell_size.z,
            position.y / self.cell_size.y,
        ]
    }

    /// Where a chunk's corner sits on Godot's ground plane.
    fn chunk_corner(&self, chunk: ChunkCoord) -> Vector3 {
        let shape = self.chunk_shape();
        Vector3::new(
            chunk.x as f32 * shape.x as f32 * self.cell_size.x,
            0.0,
            chunk.y as f32 * shape.y as f32 * self.cell_size.z,
        )
    }

    /// Builds the ground of up to [`GROUNDS_PER_FRAME`] chunks a newly arrived field may have
    /// completed, within [`GROUNDS_BUDGET_MS`] after the first, nearest the followed position
    /// first, and frees the ground of chunks whose own field was dropped.
    ///
    /// Returns how many chunks got ground.
    fn update_ground(&mut self, arrived: &[ChunkCoord], gone: &[ChunkCoord]) -> usize {
        let mut rendering = RenderingServer::singleton();
        // A chunk's ground reads the fields and materials of the chunks around it, so it goes with
        // any of them, its grass with it, and is built again when they have all arrived again.
        for reader in gone.iter().copied().flat_map(ground_readers) {
            if let Some((_, mesh, instance)) = self.grounds.remove(&reader) {
                rendering.free_rid(instance);
                rendering.free_rid(mesh);
            }
            self.ground_revisions.remove(&reader);
            self.chunk_materials.remove(&reader);
            if let Some(grass) = &mut self.grass {
                grass.remove(reader);
            }
        }
        let Some(worker) = &self.worker else {
            return 0;
        };
        let Some(scenario) = self
            .base()
            .get_viewport()
            .and_then(|viewport| viewport.find_world_3d())
            .map(|world| world.get_scenario())
        else {
            return 0;
        };
        let stage = self.ground_stage.to_string();
        let material_stage = self.ground_material_stage.to_string();
        let cell = self.cell_size.to_array();
        // Drops first: a raise drops a chunk and generates it again in one frame, and it is due.
        for chunk in gone {
            self.ground_due.remove(chunk);
        }
        self.ground_due
            .extend(arrived.iter().copied().flat_map(ground_readers));
        let focus = self.followed.unwrap_or(ChunkCoord::new(0, 0, 0));
        let mut due: Vec<ChunkCoord> = self.ground_due.iter().copied().collect();
        due.sort_by_key(|chunk| {
            (
                (chunk.x - focus.x).abs().max((chunk.y - focus.y).abs()),
                *chunk,
            )
        });
        let building = std::time::Instant::now();
        let mut built = Vec::new();
        for chunk in due {
            if built.len() == GROUNDS_PER_FRAME {
                break;
            }
            // Looked at now: built, already built, or waiting for a field around it, whose arrival
            // makes it due again.
            self.ground_due.remove(&chunk);
            if self.grounds.contains_key(&chunk) {
                continue;
            }
            let Some(mesh) = ground(chunk, |at| worker.field(&stage, at), cell) else {
                continue;
            };
            if self.palette.is_none() {
                built.push((chunk, mesh, None));
                continue;
            }
            if let Some(ids) = ground_materials(chunk, |at| worker.categories(&material_stage, at))
            {
                built.push((chunk, mesh, Some(ids)));
            }
        }
        let mut count = 0;
        for (chunk, mesh, ids) in built {
            if count > 0 && elapsed_ms(building) >= GROUNDS_BUDGET_MS {
                self.ground_due.insert(chunk);
                continue;
            }
            count += 1;
            let rid = rendering.mesh_create();
            let mut arrays = ground_arrays(&mesh);
            let (finest, coarser) = ground_levels(&mesh);
            add_levelled_surface(rid, &mut arrays, finest, &coarser);
            match (ids, &self.palette) {
                (Some(ids), Some((palette, template))) => {
                    let material = chunk_material(template, palette, &mesh, &ids, cell);
                    rendering.mesh_surface_set_material(rid, 0, material.get_rid());
                    self.chunk_materials.insert(chunk, material);
                }
                _ => {
                    if let Some(material) = &self.ground_material {
                        rendering.mesh_surface_set_material(rid, 0, material.get_rid());
                    }
                }
            }
            let instance = rendering.instance_create2(rid, scenario);
            rendering.instance_set_transform(
                instance,
                Transform3D::new(Basis::IDENTITY, self.chunk_corner(chunk)),
            );
            Gi::Static.apply(instance);
            self.grounds.insert(chunk, (mesh, rid, instance));
            self.ground_builds += 1;
            self.ground_revisions.insert(chunk, self.ground_builds);
        }
        count
    }

    /// The scene [`Self::bake`] gives for the chunks from `from` to `to`, or why there is none.
    fn baked(&self, from: ChunkCoord, to: ChunkCoord) -> Result<Gd<PackedScene>, String> {
        let (Some(worker), Some(pack)) = (&self.worker, &self.pack) else {
            return Err("the node has not started".to_owned());
        };
        let mut root = Node3D::new_alloc();
        root.set_name("Baked");
        let result = self.bake_chunks(worker, pack, &mut root, from, to);
        let scene = result.and_then(|()| {
            own(&root.clone().upcast(), &root.clone().upcast());
            let mut scene = PackedScene::new_gd();
            match scene.pack(&root) {
                Error::OK => Ok(scene),
                error => Err(format!("packing the scene failed: {error:?}")),
            }
        });
        root.free();
        scene
    }

    /// Adds under `root` a node per chunk from `from` to `to` holding what [`Self::bake`] bakes.
    fn bake_chunks(
        &self,
        worker: &StageWorker,
        pack: &Pack,
        root: &mut Gd<Node3D>,
        from: ChunkCoord,
        to: ChunkCoord,
    ) -> Result<(), String> {
        let instance = |mesh: Gd<ArrayMesh>, name: &str, at: Vector3| {
            let mut instance = MeshInstance3D::new_alloc();
            instance.set_name(name);
            instance.set_mesh(&mesh);
            instance.set_position(at);
            instance
        };
        let body = |name: &str, shape: Gd<Shape3D>, at: Transform3D| {
            let mut body = StaticBody3D::new_alloc();
            body.set_name(name);
            let mut collision = CollisionShape3D::new_alloc();
            collision.set_shape(&shape);
            collision.set_transform(at);
            body.add_child(&collision);
            body
        };
        let bound = self.placements.kinds();
        let placing = Placing {
            worker,
            pack,
            rules: &self.rules,
            shape: self.chunk_shape(),
            cell: self.cell_size,
            bound: &bound,
        };
        for y in from.y..=to.y {
            for x in from.x..=to.x {
                let chunk = ChunkCoord::new(x, y, 0);
                let corner = self.chunk_corner(chunk);
                let mut holder = Node3D::new_alloc();
                holder.set_name(&format!("Chunk {x} {y}"));
                root.add_child(&holder);
                if !self.ground_stage.is_empty() {
                    let (ground, _, _) = self
                        .grounds
                        .get(&chunk)
                        .ok_or_else(|| format!("the ground of {chunk:?} is not built"))?;
                    let material = self
                        .chunk_materials
                        .get(&chunk)
                        .map(|material| material.clone().upcast::<Material>())
                        .or_else(|| self.ground_material.clone());
                    let (finest, coarser) = ground_levels(ground);
                    let mesh = levelled_mesh(
                        &mut ground_arrays(ground),
                        finest,
                        &coarser,
                        material.map(crate::shaders::self_contained).as_ref(),
                    );
                    holder.add_child(&instance(mesh, "Ground", corner));
                    if let Some((width, depth, heights, at)) = self.ground_height_map(chunk) {
                        let mut map = HeightMapShape3D::new_gd();
                        map.set_map_width(width);
                        map.set_map_depth(depth);
                        map.set_map_data(&heights);
                        holder.add_child(&body("GroundBody", map.upcast(), at));
                    }
                }
                for (layer, fluid) in [(&self.rock, false), (&self.fluid, true)] {
                    let Some(layer) = layer else {
                        continue;
                    };
                    let surface = layer
                        .built
                        .get(&chunk)
                        .ok_or_else(|| format!("the surface of {chunk:?} is not built"))?;
                    if surface.mesh.indices.is_empty() {
                        continue;
                    }
                    let mesh = levelled_mesh(
                        &mut self.surface_arrays(&surface.mesh, fluid),
                        &surface.mesh.indices,
                        &[],
                        self.surface_material(&surface.mesh, fluid)
                            .map(crate::shaders::self_contained)
                            .as_ref(),
                    );
                    let name = if fluid { "Fluid" } else { "Surface" };
                    holder.add_child(&instance(mesh, name, corner));
                    if !fluid {
                        let mut faces = ConcavePolygonShape3D::new_gd();
                        faces.set_faces(&surface_faces(&surface.mesh));
                        let at = Transform3D::new(Basis::IDENTITY, corner);
                        holder.add_child(&body("SurfaceBody", faces.upcast(), at));
                    }
                }
                // Everything so far is the generator's; a point placed as a node is the stage's.
                for mut part in holder.get_children().iter_shared() {
                    part.set_meta(BAKED_PART, &true.to_variant());
                }
                let mut listed = VarArray::new();
                if !bound.is_empty() {
                    for stage in pack.stage_names() {
                        let Some(items) = placement_items(&placing, stage, chunk) else {
                            continue;
                        };
                        let points = worker.points(stage, chunk).unwrap_or_default();
                        for (mut node, item) in self.placements.baked(&items)? {
                            let point = item.and_then(|item| {
                                points
                                    .iter()
                                    .find(|point| local_id(point.id.local) == item.id)
                            });
                            match point {
                                Some(point) => {
                                    let mut mark = VarDictionary::new();
                                    mark.set(&"stage".to_variant(), &stage.to_variant());
                                    mark.set(
                                        &"chunk".to_variant(),
                                        &to_vector(point.id.chunk).to_variant(),
                                    );
                                    mark.set(
                                        &"id".to_variant(),
                                        &local_id(point.id.local).to_variant(),
                                    );
                                    mark.set(
                                        &"at".to_variant(),
                                        &Vector2::new(point.position[0], point.position[1])
                                            .to_variant(),
                                    );
                                    mark.set(
                                        &"transform".to_variant(),
                                        &node.get_transform().to_variant(),
                                    );
                                    node.set_meta(BAKED_POINT, &mark.to_variant());
                                    listed.push(&mark.to_variant());
                                }
                                None => node.set_meta(BAKED_PART, &true.to_variant()),
                            }
                            holder.add_child(&node);
                        }
                    }
                }
                holder.set_meta(BAKED_CHUNK, &to_vector(chunk).to_variant());
                holder.set_meta(BAKED_POINTS, &listed.to_variant());
            }
        }
        Ok(())
    }

    /// Asks the stages' thread for the report of every chunk of `candidates_stage` that arrived,
    /// draws each report that arrived, and frees those of chunks dropped.
    fn update_candidates(&mut self, events: &[StageEvent]) {
        let stage = self.candidates_stage.to_string();
        if stage.is_empty() {
            return;
        }
        let (Some(worker), Some(pack)) = (&self.worker, &self.pack) else {
            return;
        };
        let Some(StageKind::Scatter { height, .. }) = pack.kind(&stage) else {
            return;
        };
        let Some(scenario) = self
            .base()
            .get_viewport()
            .and_then(|viewport| viewport.find_world_3d())
            .map(|world| world.get_scenario())
        else {
            return;
        };
        let mut rendering = RenderingServer::singleton();
        for event in events {
            match event {
                StageEvent::Generated {
                    stage: arrived,
                    chunk,
                } if *arrived == stage => {
                    worker.request_scatter_report(&stage, *chunk);
                }
                StageEvent::Dropped { stage: gone, chunk } if *gone == stage => {
                    if let Some((multimesh, instance, _)) = self.candidates.remove(chunk) {
                        rendering.free_rid(instance);
                        rendering.free_rid(multimesh);
                    }
                }
                StageEvent::Judged {
                    stage: judged,
                    chunk,
                } if *judged == stage => {
                    let (Some(report), Some(heights)) = (
                        worker.scatter_report(&stage, *chunk),
                        worker.field(height, *chunk),
                    ) else {
                        continue;
                    };
                    let [sx, sy] = [self.chunk_cells.x as f32, self.chunk_cells.y as f32];
                    let cell = self.cell_size;
                    let mut counts: BTreeMap<String, i64> = BTreeMap::new();
                    let mut buffer: Vec<f32> = Vec::with_capacity(report.len() * 16);
                    for judged in report {
                        let name = verdict_name(judged.verdict);
                        let colour = verdict_colour(judged.verdict);
                        *counts.entry(name).or_default() += 1;
                        let local = [
                            (judged.at[0] - chunk.x as f32 * sx).floor() as u32,
                            (judged.at[1] - chunk.y as f32 * sy).floor() as u32,
                        ];
                        let at = Vector3::new(
                            judged.at[0] * cell.x,
                            (heights.get(local[0], local[1]) + 0.5) * cell.y,
                            judged.at[1] * cell.z,
                        );
                        let size = 0.3 * cell.x;
                        buffer.extend_from_slice(&[
                            size, 0.0, 0.0, at.x, 0.0, size, 0.0, at.y, 0.0, 0.0, size, at.z,
                            colour.r, colour.g, colour.b, colour.a,
                        ]);
                    }
                    let multimesh = rendering.multimesh_create();
                    let mesh = self
                        .candidate_box
                        .get_or_insert_with(candidate_mesh)
                        .get_rid();
                    rendering.multimesh_set_mesh(multimesh, mesh);
                    rendering
                        .multimesh_allocate_data_ex(
                            multimesh,
                            report.len() as i32,
                            MultimeshTransformFormat::TRANSFORM_3D,
                        )
                        .color_format(true)
                        .done();
                    rendering.multimesh_set_buffer(
                        multimesh,
                        &PackedFloat32Array::from(buffer.as_slice()),
                    );
                    let instance = rendering.instance_create2(multimesh, scenario);
                    if let Some((old, old_instance, _)) = self
                        .candidates
                        .insert(*chunk, (multimesh, instance, counts))
                    {
                        rendering.free_rid(old_instance);
                        rendering.free_rid(old);
                    }
                }
                StageEvent::Generated { .. }
                | StageEvent::Dropped { .. }
                | StageEvent::Saved
                | StageEvent::Judged { .. } => {}
            }
        }
    }

    /// Updates the rock's surfaces, or the fluid's with `fluid`, from the chunks of its stage
    /// that `arrived` and are `gone`, drawing for `budget_ms` ([`Self::update_surfaces`]).
    ///
    /// Returns how many chunks got a surface.
    fn update_layer(
        &mut self,
        fluid: bool,
        arrived: &[ChunkCoord],
        gone: &[ChunkCoord],
        budget_ms: f64,
    ) -> usize {
        let taken = if fluid {
            self.fluid.take()
        } else {
            self.rock.take()
        };
        let Some(mut layer) = taken else {
            return 0;
        };
        let count = self.update_surfaces(&mut layer, fluid, arrived, gone, budget_ms);
        if fluid {
            self.fluid = Some(layer);
        } else {
            self.rock = Some(layer);
        }
        count
    }

    /// Meshes the surface of the chunks a newly arrived volume of `layer` may have completed on
    /// its surface thread, and draws those meshed, nearest the followed position first, for
    /// `budget_ms` and at least one, and frees the surface of every chunk that reads a volume that
    /// was dropped, which an edit's regeneration builds again.
    ///
    /// Returns how many chunks got a surface.
    fn update_surfaces(
        &mut self,
        layer: &mut VolumeLayer,
        fluid: bool,
        arrived: &[ChunkCoord],
        gone: &[ChunkCoord],
        budget_ms: f64,
    ) -> usize {
        let mut rendering = RenderingServer::singleton();
        for reader in gone.iter().copied().flat_map(ground_readers) {
            if let Some((mesh, instance)) = layer.built.remove(&reader).and_then(|s| s.drawn) {
                rendering.free_rid(instance);
                rendering.free_rid(mesh);
            }
            layer.surfaces.cancel(reader);
            layer.meshed.retain(|mesh| mesh.chunk != reader);
        }
        let Some(scenario) = self
            .base()
            .get_viewport()
            .and_then(|viewport| viewport.find_world_3d())
            .map(|world| world.get_scenario())
        else {
            return 0;
        };
        let Some(worker) = &self.worker else {
            return 0;
        };
        let voxel = self.cell_size.to_array();
        // Drops first: an edit drops a chunk and generates it again in one frame, and it is due.
        for chunk in gone {
            layer.due.remove(chunk);
        }
        layer
            .due
            .extend(arrived.iter().copied().flat_map(ground_readers));
        let focus = self.followed.unwrap_or(ChunkCoord::new(0, 0, 0));
        let distance = |chunk: ChunkCoord| (chunk.x - focus.x).abs().max((chunk.y - focus.y).abs());
        let mut due: Vec<ChunkCoord> = layer.due.iter().copied().collect();
        due.sort_by_key(|&chunk| (distance(chunk), chunk));
        for chunk in due {
            // Looked at now: meshing, built, or waiting for a volume around it, whose arrival
            // makes it due again.
            layer.due.remove(&chunk);
            if layer.built.contains_key(&chunk) || layer.surfaces.is_building(chunk) {
                continue;
            }
            let around: Option<Vec<_>> = (0..9)
                .map(|i| {
                    let at = ChunkCoord::new(chunk.x + i % 3 - 1, chunk.y + i / 3 - 1, chunk.z);
                    worker.shared(&layer.stage, at)
                })
                .collect();
            if let Some(around) = around.and_then(|around| around.try_into().ok()) {
                layer.surfaces.build(chunk, around, voxel);
            }
        }
        layer.meshed.extend(layer.surfaces.drain());
        layer
            .meshed
            .sort_by_key(|mesh| std::cmp::Reverse((distance(mesh.chunk), mesh.chunk)));
        let drawing = std::time::Instant::now();
        let mut count = 0;
        while let Some(mesh) = layer.meshed.pop() {
            let chunk = mesh.chunk;
            self.volume_revisions += 1;
            let drawn =
                (!mesh.indices.is_empty()).then(|| self.draw_surface(&mesh, scenario, fluid));
            let surface = Surface {
                mesh,
                drawn,
                revision: self.volume_revisions,
            };
            layer.built.insert(chunk, surface);
            count += 1;
            if elapsed_ms(drawing) >= budget_ms {
                break;
            }
        }
        self.volume_ms += elapsed_ms(drawing);
        count
    }

    /// Draws a chunk's surface of the rock, or of the fluid with `fluid`, in `scenario`, in its
    /// materials' colours for a stage with materials, and returns its `RenderingServer` mesh and
    /// instance.
    fn draw_surface(&self, mesh: &VolumeMesh, scenario: Rid, fluid: bool) -> (Rid, Rid) {
        let mut rendering = RenderingServer::singleton();
        let rid = rendering.mesh_create();
        let mut arrays = self.surface_arrays(mesh, fluid);
        add_levelled_surface(rid, &mut arrays, &mesh.indices, &[]);
        if let Some(material) = self.surface_material(mesh, fluid) {
            rendering.mesh_surface_set_material(rid, 0, material.get_rid());
        }
        let instance = rendering.instance_create2(rid, scenario);
        rendering.instance_set_transform(
            instance,
            Transform3D::new(Basis::IDENTITY, self.chunk_corner(mesh.chunk)),
        );
        // Fluid is see-through and moves with the rock it fills, so global illumination leaves
        // it out.
        if fluid { Gi::Off } else { Gi::Static }.apply(instance);
        (rid, instance)
    }

    /// A chunk's rock surface, or with `fluid` its fluid's, as a surface's arrays without its
    /// triangles: its vertices and normals, each vertex's material's colour from the palette for
    /// a stage with materials, and for fluid each vertex's glow in its first UV.
    fn surface_arrays(&self, mesh: &VolumeMesh, fluid: bool) -> VarArray {
        let palette = if fluid {
            &self.fluid_palette
        } else {
            &self.volume_palette
        };
        let mut arrays = VarArray::new();
        arrays.resize(ArrayType::MAX.ord() as usize, &Variant::nil());
        let vertices: PackedVector3Array = mesh
            .positions
            .iter()
            .map(|&[x, y, z]| Vector3::new(x, y, z))
            .collect();
        let normals: PackedVector3Array = mesh
            .normals
            .iter()
            .map(|&[x, y, z]| Vector3::new(x, y, z))
            .collect();
        arrays.set(ArrayType::VERTEX.ord() as usize, &vertices.to_variant());
        arrays.set(ArrayType::NORMAL.ord() as usize, &normals.to_variant());
        if !mesh.materials.is_empty() {
            let colours: PackedColorArray = mesh
                .materials
                .iter()
                .map(|&material| palette_colour(palette, usize::from(material)))
                .collect();
            arrays.set(ArrayType::COLOR.ord() as usize, &colours.to_variant());
        }
        // A fluid without materials glows nowhere, and its vertices carry no UV.
        if fluid && !mesh.materials.is_empty() {
            let glows: PackedVector2Array = mesh
                .materials
                .iter()
                .map(|&material| {
                    let glow = self.fluid_glow.get(usize::from(material)).unwrap_or(0.0);
                    Vector2::new(glow, 0.0)
                })
                .collect();
            arrays.set(ArrayType::TEX_UV.ord() as usize, &glows.to_variant());
        }
        arrays
    }

    /// The material a chunk's rock surface, or with `fluid` its fluid's, is drawn with: the one
    /// given, or else the vertex colours for fluid or rock with materials; none for Godot's
    /// default.
    fn surface_material(&self, mesh: &VolumeMesh, fluid: bool) -> Option<Gd<Material>> {
        if fluid {
            (self.fluid_material.clone()).or_else(|| self.fluid_colours.clone().map(Gd::upcast))
        } else {
            (self.volume_material.clone()).or_else(|| {
                (!mesh.materials.is_empty())
                    .then(|| self.vertex_colours.clone().map(Gd::upcast))
                    .flatten()
            })
        }
    }

    /// Draws the far ground of the coarse chunks that are due, nearest the followed position first
    /// and at most [`FAR_GROUNDS_PER_FRAME`] a frame: one mesh each, leaving out the chunks whose
    /// near ground is drawn and walled off where it meets them ([`far_ground`]). A coarse chunk is
    /// due when a field around it arrives, and when near ground comes or goes on or beside a
    /// lattice chunk it covers; one whose field is dropped is freed.
    fn update_far_ground(
        &mut self,
        arrived: &[ChunkCoord],
        gone: &[ChunkCoord],
        near_changed: &[ChunkCoord],
    ) {
        let stage = self.far_ground_stage.to_string();
        let (Some(worker), Some(pack)) = (&self.worker, &self.pack) else {
            return;
        };
        let Some(scale) = pack.scale(&stage) else {
            return;
        };
        let scale = scale as i32;
        let mut rendering = RenderingServer::singleton();
        for chunk in gone {
            self.far_due.remove(chunk);
            if let Some((mesh, instance)) = self.far_grounds.remove(chunk) {
                rendering.free_rid(instance);
                rendering.free_rid(mesh);
            }
        }
        for chunk in arrived {
            for (dx, dy) in (-1..=1).flat_map(|dy| (-1..=1).map(move |dx| (dx, dy))) {
                self.far_due
                    .insert(ChunkCoord::new(chunk.x + dx, chunk.y + dy, 0));
            }
        }
        for fine in near_changed {
            for (dx, dy) in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)] {
                self.far_due.insert(ChunkCoord::new(
                    (fine.x + dx).div_euclid(scale),
                    (fine.y + dy).div_euclid(scale),
                    0,
                ));
            }
        }
        let Some(scenario) = self
            .base()
            .get_viewport()
            .and_then(|viewport| viewport.find_world_3d())
            .map(|world| world.get_scenario())
        else {
            return;
        };
        let focus = self.followed.unwrap_or(ChunkCoord::new(0, 0, 0));
        let mut due: Vec<ChunkCoord> = self.far_due.iter().copied().collect();
        due.sort_by_key(|chunk| {
            let first = ChunkCoord::new(chunk.x * scale, chunk.y * scale, 0);
            (
                (first.x - focus.x).abs().max((first.y - focus.y).abs()),
                *chunk,
            )
        });
        let cell = self.cell_size.to_array();
        let mut built = Vec::new();
        for chunk in due {
            if built.len() == FAR_GROUNDS_PER_FRAME {
                break;
            }
            // Looked at now: built, or waiting for a field around it, whose arrival makes it due
            // again.
            self.far_due.remove(&chunk);
            let far = far_ground(
                chunk,
                scale as u32,
                |at| worker.field(&stage, at),
                cell,
                |fine| self.grounds.get(&fine).map(|(mesh, _, _)| mesh),
            );
            if let Some(far) = far {
                built.push(far);
            }
        }
        for far in built {
            // The near ground covers the whole coarse chunk: there is nothing of it to draw.
            if far.indices.is_empty() {
                if let Some((mesh, instance)) = self.far_grounds.remove(&far.chunk) {
                    rendering.free_rid(instance);
                    rendering.free_rid(mesh);
                }
                continue;
            }
            let rid = rendering.mesh_create();
            let mut arrays = VarArray::new();
            arrays.resize(ArrayType::MAX.ord() as usize, &Variant::nil());
            let vertices: PackedVector3Array = far
                .positions
                .iter()
                .map(|&[x, y, z]| Vector3::new(x, y, z))
                .collect();
            let normals: PackedVector3Array = far
                .normals
                .iter()
                .map(|&[x, y, z]| Vector3::new(x, y, z))
                .collect();
            arrays.set(ArrayType::VERTEX.ord() as usize, &vertices.to_variant());
            arrays.set(ArrayType::NORMAL.ord() as usize, &normals.to_variant());
            add_levelled_surface(rid, &mut arrays, &far.indices, &[]);
            if let Some(material) = &self.ground_material {
                rendering.mesh_surface_set_material(rid, 0, material.get_rid());
            }
            let instance = rendering.instance_create2(rid, scenario);
            let first = ChunkCoord::new(far.chunk.x * scale, far.chunk.y * scale, 0);
            rendering.instance_set_transform(
                instance,
                Transform3D::new(Basis::IDENTITY, self.chunk_corner(first)),
            );
            Gi::Static.apply(instance);
            if let Some((mesh, old)) = self.far_grounds.insert(far.chunk, (rid, instance)) {
                rendering.free_rid(old);
                rendering.free_rid(mesh);
            }
        }
    }

    /// Keeps one static body on every chunk within `collider_radius` of the followed chunk that
    /// has ground or a town with shapes, building it again when what it would hold changes, and
    /// frees the others.
    ///
    /// Returns how many chunks got a body.
    fn update_colliders(&mut self) -> usize {
        let (Some(pack), Some(focus)) = (self.pack.clone(), self.followed) else {
            return 0;
        };
        let radius = self.collider_radius;
        let within = |chunk: ChunkCoord| {
            radius >= 0 && (chunk.x - focus.x).abs().max((chunk.y - focus.y).abs()) <= radius
        };
        let solves = solve_stages(&pack);
        let contents = |this: &Self, chunk: ChunkCoord| this.body_contents(&solves, chunk);
        let mut physics = PhysicsServer3D::singleton();
        let stale: Vec<ChunkCoord> = self
            .bodies
            .iter()
            .filter(|(chunk, (_, _, held))| !within(**chunk) || *held != contents(self, **chunk))
            .map(|(chunk, _)| *chunk)
            .collect();
        for chunk in stale {
            if let Some((body, shape, _)) = self.bodies.remove(&chunk) {
                physics.free_rid(body);
                shape.into_iter().for_each(|shape| physics.free_rid(shape));
            }
        }
        let Some(space) = self
            .base()
            .get_viewport()
            .and_then(|viewport| viewport.find_world_3d())
            .map(|world| world.get_space())
        else {
            return 0;
        };
        let range = focus.x - radius.max(0)..=focus.x + radius.max(0);
        let mut wanted: Vec<(ChunkCoord, BodyContents)> = range
            .flat_map(|x| {
                (focus.y - radius.max(0)..=focus.y + radius.max(0))
                    .map(move |y| ChunkCoord::new(x, y, 0))
            })
            .filter(|chunk| radius >= 0 && !self.bodies.contains_key(chunk))
            .map(|chunk| (chunk, contents(self, chunk)))
            .filter(|(_, held)| {
                held.ground.is_some() || held.volume.is_some() || !held.towns.is_empty()
            })
            .collect();
        // Nearest first, and a few a frame, as the city node does.
        wanted.sort_by_key(|(chunk, _)| {
            (
                (chunk.x - focus.x).abs().max((chunk.y - focus.y).abs()),
                *chunk,
            )
        });
        let waiting = wanted.len();
        wanted.truncate(BODIES_PER_FRAME);
        let owner = u64::from_ne_bytes(self.base().instance_id().to_i64().to_ne_bytes());
        let building = std::time::Instant::now();
        let mut count = 0;
        for (chunk, held) in wanted {
            // A volume's surface is a concave shape of thousands of triangles, milliseconds to
            // build, so bodies stop at a time budget too.
            if count > 0 && elapsed_ms(building) >= BODIES_BUDGET_MS {
                break;
            }
            count += 1;
            let body = physics.body_create();
            physics.body_set_mode(body, BodyMode::STATIC);
            let mut shapes: Vec<Rid> = Vec::new();
            if held.ground.is_some() {
                shapes.extend(self.add_ground_shape(&mut physics, body, chunk));
            }
            if held.volume.is_some() {
                shapes.push(self.add_volume_shape(&mut physics, body, chunk));
            }
            for stage in &held.towns {
                let shapes = &self.collision_shapes;
                let Some((sets, lift)) =
                    self.town_sets(stage, chunk, |name| shapes.contains_key(name))
                else {
                    continue;
                };
                for set in sets {
                    let shape = shapes[&set.name].get_rid();
                    for row in set.transforms([1.0; 3]).chunks(12) {
                        let basis = Basis::from_rows(
                            Vector3::new(row[0], row[1], row[2]),
                            Vector3::new(row[4], row[5], row[6]),
                            Vector3::new(row[8], row[9], row[10]),
                        );
                        let at =
                            Transform3D::new(basis, Vector3::new(row[3], row[7] + lift, row[11]));
                        physics.body_add_shape_ex(body, shape).transform(at).done();
                    }
                }
            }
            physics.body_attach_object_instance_id(body, owner);
            physics.body_set_space(body, space);
            self.bodies.insert(chunk, (body, shapes, held));
        }
        self.bodies_pending = waiting - count;
        count
    }

    /// A chunk's ground as a height map: its vertices along x and z, each vertex's height in cells
    /// of the cell's width, and where the map stands in Godot's world, since a height map is
    /// centred on its origin and spaced a unit apart. None, with an error, for cells not as wide
    /// as they are deep, which a height map cannot hold.
    fn ground_height_map(
        &self,
        chunk: ChunkCoord,
    ) -> Option<(i32, i32, PackedFloat32Array, Transform3D)> {
        let (mesh, _, _) = &self.grounds[&chunk];
        let width = self.cell_size.x;
        if (self.cell_size.z - width).abs() > f32::EPSILON * width {
            godot_error!(
                "wave forge: a ground collider needs cells as wide as they are deep, not {}",
                self.cell_size
            );
            return None;
        }
        let heights: PackedFloat32Array = mesh.heights.iter().map(|h| h / width).collect();
        // The first vertex stands over the first column's centre.
        let centre = Vector3::new(
            (0.5 + (mesh.size[0] - 1) as f32 / 2.0) * width,
            0.0,
            (0.5 + (mesh.size[1] - 1) as f32 / 2.0) * width,
        );
        let at = Transform3D::new(
            Basis::from_scale(Vector3::ONE * width),
            self.chunk_corner(chunk) + centre,
        );
        Some((mesh.size[0] as i32, mesh.size[1] as i32, heights, at))
    }

    /// Adds a chunk's ground to `body` as a height map and returns the shape, which the caller
    /// frees with the body; none for cells a height map cannot hold.
    fn add_ground_shape(
        &self,
        physics: &mut Gd<PhysicsServer3D>,
        body: Rid,
        chunk: ChunkCoord,
    ) -> Option<Rid> {
        let (width, depth, heights, at) = self.ground_height_map(chunk)?;
        let (low, high) = heights
            .as_slice()
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), &h| {
                (low.min(h), high.max(h))
            });
        let mut data = VarDictionary::new();
        data.set(&"width".to_variant(), &width.to_variant());
        data.set(&"depth".to_variant(), &depth.to_variant());
        data.set(&"heights".to_variant(), &heights.to_variant());
        data.set(&"min_height".to_variant(), &low.to_variant());
        data.set(&"max_height".to_variant(), &high.to_variant());
        let shape = physics.heightmap_shape_create();
        physics.shape_set_data(shape, &data.to_variant());
        physics.body_add_shape_ex(body, shape).transform(at).done();
        Some(shape)
    }

    /// Adds a chunk's volume surface to `body` as a concave polygon and returns the shape, which the
    /// caller frees with the body. Only its front faces collide, which face from solid to empty.
    fn add_volume_shape(
        &self,
        physics: &mut Gd<PhysicsServer3D>,
        body: Rid,
        chunk: ChunkCoord,
    ) -> Rid {
        let mesh = &self
            .rock
            .as_ref()
            .expect("a volume shape is added while the volume is drawn")
            .built[&chunk]
            .mesh;
        let faces = surface_faces(mesh);
        let mut data = VarDictionary::new();
        data.set(&"faces".to_variant(), &faces.to_variant());
        data.set(&"backface_collision".to_variant(), &false.to_variant());
        let shape = physics.concave_polygon_shape_create();
        physics.shape_set_data(shape, &data.to_variant());
        let at = Transform3D::new(Basis::IDENTITY, self.chunk_corner(chunk));
        physics.body_add_shape_ex(body, shape).transform(at).done();
        shape
    }

    /// What a chunk's body holds: its ground and volume surface as built, and which of `solves`
    /// have a town there that the node has shapes for.
    fn body_contents(&self, solves: &[String], chunk: ChunkCoord) -> BodyContents {
        BodyContents {
            ground: self.ground_revisions.get(&chunk).copied(),
            volume: self
                .rock
                .as_ref()
                .and_then(|rock| rock.built.get(&chunk))
                .filter(|surface| !surface.mesh.indices.is_empty())
                .map(|surface| surface.revision),
            towns: if self.collision_shapes.is_empty() {
                Vec::new()
            } else {
                solves
                    .iter()
                    .filter(|stage| {
                        self.worker
                            .as_ref()
                            .is_some_and(|worker| worker.tiles(stage, chunk).is_some())
                    })
                    .cloned()
                    .collect()
            },
        }
    }

    /// Keeps a navigation region on every chunk within `navigation_radius` of the followed chunk,
    /// baked from the triangles its colliders hold and its neighbours', and baked again when
    /// what they hold changes; a bake is started only if `may_start`. Returns the chunks whose
    /// mesh went into the map this frame.
    fn update_navigation(&mut self, may_start: bool) -> Vec<ChunkCoord> {
        let (Some(pack), Some(focus)) = (self.pack.clone(), self.followed) else {
            return Vec::new();
        };
        let Some(map) = self
            .base()
            .get_viewport()
            .and_then(|viewport| viewport.find_world_3d())
            .map(|world| world.get_navigation_map())
        else {
            return Vec::new();
        };
        let radius = self.navigation_radius;
        let wanted: Vec<ChunkCoord> = (-radius..=radius)
            .flat_map(|dy| {
                (-radius..=radius).map(move |dx| ChunkCoord::new(focus.x + dx, focus.y + dy, 0))
            })
            .collect();
        let ready = self.navigation.settle(&wanted, map);
        if !may_start {
            return ready;
        }
        for (name, shape) in &self.collision_shapes {
            if self.shape_faces.contains_key(name) {
                continue;
            }
            // A shape's filled debug mesh is the one triangle form every Shape3D offers. Godot's
            // faces are clockwise seen from outside, and Recast's counter-clockwise, as the
            // ground's and the volume's are.
            let faces: Vec<[f32; 3]> = shape
                .get_debug_mesh()
                .map(|mesh| {
                    mesh.get_faces()
                        .as_slice()
                        .chunks(3)
                        .flat_map(|corners| [corners[0], corners[2], corners[1]])
                        .map(|corner| corner.to_array())
                        .collect()
                })
                .unwrap_or_default();
            if faces.is_empty() {
                godot_error!(
                    "wave forge: the collision shape of {name} has no faces to bake navigation \
                     from; is its debug_fill off?"
                );
            }
            self.shape_faces.insert(name.clone(), faces);
        }
        let solves = solve_stages(&pack);
        let shape = self.chunk_shape();
        let chunk_size = [
            shape.x as f32 * self.cell_size.x,
            shape.y as f32 * self.cell_size.z,
        ];
        // The ground and the volume's surface of a chunk are built from its neighbours' too, so a
        // chunk has them only where the world's bound holds every chunk around it.
        let in_world = |chunk: ChunkCoord| {
            pack.bound().is_none_or(|bound| {
                (-1..=1).all(|dy| {
                    (-1..=1).all(|dx| {
                        let min = [
                            (chunk.x + dx) as f32 * shape.x as f32,
                            (chunk.y + dy) as f32 * shape.y as f32,
                        ];
                        bound.meets(min, [min[0] + shape.x as f32, min[1] + shape.y as f32])
                    })
                })
            })
        };
        let cell_height = NavigationServer3D::singleton().map_get_cell_height(map);
        let mut navigation = std::mem::replace(&mut self.navigation, StageNavigation::new());
        let result = navigation.start_bake(
            map,
            self.navigation_template.as_ref(),
            wanted,
            focus,
            |chunk| {
                (-1..=1)
                    .flat_map(|dy| {
                        (-1..=1).map(move |dx| ChunkCoord::new(chunk.x + dx, chunk.y + dy, 0))
                    })
                    .map(|around| self.body_contents(&solves, around))
                    .collect()
            },
            |chunk, border| {
                wave_forge::surface_nav_source(
                    chunk,
                    chunk_size,
                    in_world,
                    |around| self.walkable_triangles(&solves, around),
                    border,
                    cell_height,
                )
            },
        );
        self.navigation = navigation;
        if let Err(error) = result {
            godot_error!("wave forge: navigation turned off: {error}");
            self.navigation_radius = -1;
        }
        ready
    }

    /// The boxes of a chunk's towns' solid cells in Godot's world, each town's raised to its site.
    fn town_occluder_boxes(&self, chunk: ChunkCoord) -> Vec<wave_forge::CellBox> {
        let (Some(pack), Some(worker)) = (&self.pack, &self.worker) else {
            return Vec::new();
        };
        let space = YUpSpace::new(self.chunk_shape(), self.cell_size.to_array());
        let mut boxes = Vec::new();
        for stage in solve_stages(pack) {
            let Some(town) = worker.tiles(&stage, chunk) else {
                continue;
            };
            let file = self
                .rules
                .get(town.rules.as_ref())
                .expect("the node loaded every rule set its Solve stages use");
            let tiles = Chunk {
                coord: chunk,
                tiles: town.tiles.to_vec().into_boxed_slice(),
                version: 1,
            };
            let lift = town.height * self.cell_size.y;
            boxes.extend(wave_forge::occluders(&tiles, file, &space).into_iter().map(
                |mut cell_box| {
                    cell_box.min[1] += lift;
                    cell_box.max[1] += lift;
                    cell_box
                },
            ));
        }
        boxes
    }

    /// Keeps occluders on every town chunk within `occluder_radius` of the followed chunk, building
    /// a few a frame, nearest first, and again when a chunk's town arrives anew; frees the others.
    fn update_occluders(&mut self, towns: &[ChunkCoord]) {
        let (Some(pack), Some(worker), Some(focus)) = (&self.pack, &self.worker, self.followed)
        else {
            return;
        };
        let radius = self.occluder_radius;
        let solves = solve_stages(pack);
        let held: std::collections::HashSet<ChunkCoord> = (-radius.max(0)..=radius.max(0))
            .flat_map(|dy| {
                (-radius.max(0)..=radius.max(0))
                    .map(move |dx| ChunkCoord::new(focus.x + dx, focus.y + dy, 0))
            })
            .filter(|&chunk| {
                solves
                    .iter()
                    .any(|stage| worker.tiles(stage, chunk).is_some())
            })
            .collect();
        let built: std::collections::HashSet<ChunkCoord> = self.occluders.chunks().collect();
        let plan = crate::radius::plan(
            radius,
            focus,
            &held,
            &built,
            &mut self.occluders_due,
            towns,
            BODIES_PER_FRAME,
        );
        let boxes: Vec<(ChunkCoord, Vec<wave_forge::CellBox>)> = plan
            .build
            .iter()
            .map(|&chunk| (chunk, self.town_occluder_boxes(chunk)))
            .collect();
        for chunk in plan.gone {
            self.occluders.drop_chunk(chunk);
        }
        let mut owner = self.base().clone();
        for (chunk, boxes) in boxes {
            self.occluders.build(&mut owner, chunk, &boxes);
        }
    }

    /// The triangles a chunk's colliders hold in Godot's world, counter-clockwise seen from where
    /// agents walk: its ground, its volume's surface, and its towns' shapes. None while its ground
    /// or surface is not built.
    fn walkable_triangles(&self, solves: &[String], chunk: ChunkCoord) -> Option<Vec<[f32; 3]>> {
        let corner = self.chunk_corner(chunk).to_array();
        let mut triangles = Vec::new();
        if !self.ground_stage.is_empty() {
            let (ground, _, _) = self.grounds.get(&chunk)?;
            triangles.extend(ground.surface_triangles(corner));
        }
        if let Some(rock) = &self.rock {
            triangles.extend(rock.built.get(&chunk)?.mesh.triangles(corner));
        }
        let faces = &self.shape_faces;
        for stage in solves {
            let Some((sets, lift)) = self.town_sets(stage, chunk, |name| faces.contains_key(name))
            else {
                continue;
            };
            for set in sets {
                triangles.extend(set.placed(&faces[&set.name], lift));
            }
        }
        Some(triangles)
    }

    fn free_bodies(&mut self) {
        let mut physics = PhysicsServer3D::singleton();
        for (_, (body, shape, _)) in self.bodies.drain() {
            physics.free_rid(body);
            shape.into_iter().for_each(|shape| physics.free_rid(shape));
        }
    }

    /// Once the bound scenes have loaded, gives every town module drawn from a lone mesh and given
    /// no shape by `set_collision_shape` its mesh as its collision shape, scaled to the cell, so a
    /// town drawn with no code is walked on and into as it is drawn.
    fn give_mesh_shapes(&mut self) {
        if self.mesh_shapes.is_some() {
            return;
        }
        let Some(meshes) = self.placements.lone_meshes() else {
            return;
        };
        let module = |name: &str| {
            self.rules
                .values()
                .any(|file| (0..file.num_tiles()).any(|tile| file.name(tile) == name))
        };
        let cell = self.cell_size;
        let mut given = BTreeSet::new();
        for (name, mesh) in meshes {
            if !module(&name) || self.collision_shapes.contains_key(&name) {
                continue;
            }
            let faces: PackedVector3Array = mesh
                .get_faces()
                .as_slice()
                .iter()
                .map(|&corner| corner * cell)
                .collect();
            let mut shape = ConcavePolygonShape3D::new_gd();
            shape.set_faces(&faces);
            self.collision_shapes.insert(name.clone(), shape.upcast());
            given.insert(name);
        }
        if !given.is_empty() {
            self.free_bodies();
            self.shape_faces.clear();
            self.navigation.clear();
        }
        self.mesh_shapes = Some(given);
    }

    /// Places the chunks of bound scenes that are due, within the frame's budget, and signals
    /// each node placed. Returns how many nodes were placed.
    fn update_placements(&mut self) -> usize {
        let (Some(worker), Some(pack)) = (&self.worker, &self.pack) else {
            return 0;
        };
        let Some(scenario) = self
            .base()
            .get_viewport()
            .and_then(|viewport| viewport.find_world_3d())
            .map(|world| world.get_scenario())
        else {
            return 0;
        };
        let bound: Vec<String> = self.placements.kinds();
        let placing = Placing {
            worker,
            pack,
            rules: &self.rules,
            shape: self.chunk_shape(),
            cell: self.cell_size,
            bound: &bound,
        };
        let items = |stage: &str, chunk: ChunkCoord| placement_items(&placing, stage, chunk);
        let focus = self.followed.unwrap_or(ChunkCoord::new(0, 0, 0));
        let budget = self.placement_budget_ms;
        let radius = self.promotion_radius;
        let near = |chunk: ChunkCoord| {
            radius < 0 || (chunk.x - focus.x).abs().max((chunk.y - focus.y).abs()) <= radius
        };
        let mut holder = self.base().clone();
        let result = self
            .placements
            .place(&mut holder, scenario, focus, budget, near, items);
        match result {
            Ok(spawned) => {
                let count = spawned.len();
                for (node, chunk, id) in spawned {
                    self.signals()
                        .instance_spawned()
                        .emit(&node, to_vector(chunk), id);
                }
                count
            }
            Err(error) => {
                godot_error!("wave forge: {error}");
                self.placements.clear();
                0
            }
        }
    }

    /// Grows grass on the chunks within `grass_radius` of the followed one whose ground is built,
    /// nearest first and a few a frame, and frees the grass of the others.
    fn update_grass(&mut self) {
        let Some(scenario) = self
            .base()
            .get_viewport()
            .and_then(|viewport| viewport.find_world_3d())
            .map(|world| world.get_scenario())
        else {
            return;
        };
        let (Some(worker), Some(grass)) = (&self.worker, &mut self.grass) else {
            return;
        };
        let focus = self.followed.unwrap_or(ChunkCoord::new(0, 0, 0));
        let radius = self.grass_radius;
        let distance = |chunk: ChunkCoord| (chunk.x - focus.x).abs().max((chunk.y - focus.y).abs());
        let grounds = &self.grounds;
        let keep = |chunk: ChunkCoord| distance(chunk) <= radius && grounds.contains_key(&chunk);
        let mut candidates: Vec<ChunkCoord> = grounds
            .keys()
            .copied()
            .filter(|&chunk| keep(chunk))
            .collect();
        candidates.sort_by_key(|&chunk| (distance(chunk), chunk));
        let (cell, chunk_cells) = (self.cell_size, self.chunk_cells);
        let corner = |chunk: ChunkCoord| {
            Vector3::new(
                chunk.x as f32 * chunk_cells.x.max(1) as f32 * cell.x,
                0.0,
                chunk.y as f32 * chunk_cells.y.max(1) as f32 * cell.z,
            )
        };
        let stage = self.grass_stage.to_string();
        grass.update(
            scenario,
            cell,
            corner,
            keep,
            &candidates,
            GRASS_PER_FRAME,
            |chunk| grounds.get(&chunk).map(|(mesh, _, _)| mesh),
            |chunk| worker.field(&stage, chunk),
        );
    }

    /// The inspector's "Start or regenerate".
    fn regenerate(&mut self) {
        self.start();
    }

    /// The inspector's "Reroll seed".
    fn reroll_seed(&mut self) {
        self.seed = i64::from(godot::global::randi() as u32);
        if self.worker.is_some() {
            self.start();
        }
    }

    /// The inspector's "Bake the view": the chunks within `view_radius` of the followed one, saved
    /// at `bake_path`, with an error if there is nothing followed or the bake fails.
    fn bake_view(&mut self) {
        let Some(followed) = self.followed else {
            godot_error!("wave forge: nothing is followed yet, so there is no view to bake");
            return;
        };
        let radius = self.view_radius.max(0);
        let from = Vector3i::new(followed.x - radius, followed.y - radius, 0);
        let to = Vector3i::new(followed.x + radius, followed.y + radius, 0);
        let Some(scene) = self.bake(from, to) else {
            return;
        };
        let saved = ResourceSaver::singleton()
            .save_ex(&scene)
            .path(&self.bake_path)
            .done();
        if saved == godot::global::Error::OK {
            godot_print!(
                "wave forge: baked chunks {from} to {to} into {}",
                self.bake_path
            );
        } else {
            godot_error!(
                "wave forge: the bake could not be saved at {}: {saved:?}",
                self.bake_path
            );
        }
    }

    /// Draws the sea under `chunk`, of `size` along Godot's x and z, making it the first time
    /// there is a material, a pack with water and a world to draw in.
    fn place_sea(&mut self, chunk: ChunkCoord, size: [f32; 2]) {
        if self.sea.is_none() {
            let level = self
                .pack
                .as_ref()
                .and_then(|pack| pack.water())
                .map(|water| water.level);
            let scenario = self
                .base()
                .get_viewport()
                .and_then(|viewport| viewport.find_world_3d())
                .map(|world| world.get_scenario());
            let (Some(material), Some(level), Some(scenario)) =
                (&self.sea_material, level, scenario)
            else {
                return;
            };
            // As wide as the view, and a chunk more on each side so its edge is never in it.
            let side = (2 * self.view_radius.max(0) + 3) as f32 * size[0].max(size[1]);
            self.sea = Some(crate::sea::Sea::new(
                material,
                side,
                level * self.cell_size.y,
                scenario,
            ));
        }
        if let Some(sea) = &self.sea {
            sea.place(Vector3::new(
                (chunk.x as f32 + 0.5) * size[0],
                0.0,
                (chunk.y as f32 + 0.5) * size[1],
            ));
        }
    }

    /// What is wrong with the node's stage settings for `pack`, each a sentence: a target or a
    /// setting naming no stage of the pack, or a stage of the wrong kind for the setting. `start`
    /// refuses to start with any, and the editor shows them as configuration warnings.
    fn stage_setting_problems(&self, pack: &Pack) -> Vec<String> {
        let mut problems = Vec::new();
        for target in self.targets.as_slice() {
            if pack.kind(&target.to_string()).is_none() {
                problems.push(format!(
                    "targets names {target}, which is no stage of the pack"
                ));
            }
        }
        let at_scale_one = |stage: &str| pack.scale(stage) == Some(1);
        let settings: [(&str, &GString, &str, Fits<'_>); 7] = [
            ("ground_stage", &self.ground_stage, "", &|_, _| true),
            ("grass_stage", &self.grass_stage, "", &|_, _| true),
            (
                "candidates_stage",
                &self.candidates_stage,
                "Scatter ",
                &|_, kind| matches!(kind, StageKind::Scatter { .. }),
            ),
            (
                "ground_material_stage",
                &self.ground_material_stage,
                "Rules, Area or Nearest ",
                &|_, kind| {
                    matches!(
                        kind,
                        StageKind::Rules { .. }
                            | StageKind::Area { .. }
                            | StageKind::Nearest { .. }
                    )
                },
            ),
            (
                "far_ground_stage",
                &self.far_ground_stage,
                "field ",
                &|_, kind| matches!(kind, StageKind::Field(_)),
            ),
            (
                "fluid_stage",
                &self.fluid_stage,
                "Volume, Carve or Aquifer ",
                &|stage, kind| {
                    at_scale_one(stage)
                        && matches!(
                            kind,
                            StageKind::Volume { .. }
                                | StageKind::Carve { .. }
                                | StageKind::Aquifer { .. }
                        )
                },
            ),
            (
                "volume_stage",
                &self.volume_stage,
                "Volume or Carve ",
                &|stage, kind| {
                    at_scale_one(stage)
                        && matches!(kind, StageKind::Volume { .. } | StageKind::Carve { .. })
                },
            ),
        ];
        for (setting, value, kinds, fits) in settings {
            if value.is_empty() {
                continue;
            }
            let stage = value.to_string();
            if !pack.kind(&stage).is_some_and(|kind| fits(&stage, kind)) {
                let scale = if kinds.starts_with("Volume") {
                    " at scale 1"
                } else {
                    ""
                };
                problems.push(format!(
                    "{setting} {stage:?} is no {kinds}stage of the pack{scale}"
                ));
            }
        }
        problems
    }

    /// The text of the pack the node generates: its `stack`'s, written as a pack file is, when it
    /// has one, otherwise `pack_file`'s.
    ///
    /// # Errors
    /// Why there is none: a stack that holds no valid pack, or a pack file that could not be read.
    fn pack_source(&self) -> Result<String, String> {
        match &self.stack {
            Some(stack) => {
                let text = stack
                    .clone()
                    .call("to_pack_text", &[])
                    .try_to::<GString>()
                    .map_err(|_| "the stack is no WaveForgeStack".to_owned())?
                    .to_string();
                if text.is_empty() {
                    return Err("the stack holds no valid pack".to_owned());
                }
                Ok(text)
            }
            None => {
                let text = FileAccess::get_file_as_string(&self.pack_file).to_string();
                if text.is_empty() {
                    return Err(format!("pack_file {} could not be read", self.pack_file));
                }
                Ok(text)
            }
        }
    }

    /// What the node's pack is called in a message: the stack, or `pack_file`'s path.
    fn pack_name(&self) -> String {
        match &self.stack {
            Some(_) => "the stack".to_owned(),
            None => self.pack_file.to_string(),
        }
    }

    /// The node's configuration warnings: its settings for its pack, and the project's and the
    /// scene's that would leave the world dark or without bodies ([`crate::warnings`]).
    fn warnings(&self) -> Vec<String> {
        let mut warnings = crate::warnings::lighting(&self.to_gd().upcast());
        if self.pack_file.is_empty() && self.stack.is_none() {
            warnings.push(
                "No pack_file: set one (*.world.ron), or choose a preset in the Wave Forge dock."
                    .to_owned(),
            );
        } else {
            if self.stack.is_some() && !self.pack_file.is_empty() {
                warnings.push(format!(
                    "Both a stack and pack_file {} are set: the stack is generated.",
                    self.pack_file
                ));
            }
            match self.pack_source() {
                Ok(text) => match Pack::parse(&text) {
                    Ok(pack) => warnings.extend(self.stage_setting_problems(&pack)),
                    Err(error) => warnings.push(format!("{}: {error}", self.pack_name())),
                },
                Err(error) => warnings.push(error),
            }
        }
        warnings.extend(crate::warnings::physics(self.collider_radius));
        warnings.extend(crate::warnings::occlusion(self.occluder_radius));
        if self.follow_camera {
            warnings.extend(crate::warnings::camera(&self.to_gd().upcast()));
        }
        warnings
    }

    /// The palette texture and the material each chunk's ground material is copied from.
    fn materials_template(&self) -> Result<(Gd<ImageTexture>, Gd<ShaderMaterial>), String> {
        let template = match &self.ground_material {
            None => {
                let mut material = ShaderMaterial::new_gd();
                material.set_shader(&crate::shaders::reference(
                    "ground.gdshader",
                    crate::shaders::GROUND,
                ));
                material
            }
            Some(material) => material.clone().try_cast::<ShaderMaterial>().map_err(|_| {
                "ground_material must be a ShaderMaterial when ground_material_stage is set"
                    .to_owned()
            })?,
        };
        let colours: Vec<u8> = (0..MAX_CATEGORIES)
            .flat_map(|index| {
                let colour = palette_colour(&self.ground_palette, index);
                [colour.r8(), colour.g8(), colour.b8(), 255]
            })
            .collect();
        let image = Image::create_from_data(
            MAX_CATEGORIES as i32,
            1,
            false,
            ImageFormat::RGBA8,
            &PackedByteArray::from(colours.as_slice()),
        )
        .ok_or("the ground palette could not be made")?;
        let palette = ImageTexture::create_from_image(&image)
            .ok_or("the ground palette could not be made")?;
        Ok((palette, template))
    }

    fn clear_ground_and_bodies(&mut self) {
        self.sea = None;
        if let Some(grass) = &mut self.grass {
            grass.clear();
        }
        let mut rendering = RenderingServer::singleton();
        for (_, (multimesh, instance, _)) in self.candidates.drain() {
            rendering.free_rid(instance);
            rendering.free_rid(multimesh);
        }
        self.far_due.clear();
        for (_, (mesh, instance)) in self.far_grounds.drain() {
            rendering.free_rid(instance);
            rendering.free_rid(mesh);
        }
        for (_, (_, mesh, instance)) in self.grounds.drain() {
            rendering.free_rid(instance);
            rendering.free_rid(mesh);
        }
        self.ground_revisions.clear();
        for layer in [&mut self.rock, &mut self.fluid].into_iter().flatten() {
            layer.clear();
        }
        // Freed after the meshes that draw with them.
        self.chunk_materials.clear();
        self.free_bodies();
        self.navigation.clear();
        for chunk in self.occluders.chunks().collect::<Vec<_>>() {
            self.occluders.drop_chunk(chunk);
        }
        self.placements.clear();
    }
}

/// The chunks of `layer` whose surface is built; none without the layer.
fn layer_chunks(layer: Option<&VolumeLayer>) -> Array<Vector3i> {
    layer.map_or_else(Array::new, |layer| {
        layer.built.keys().map(|&chunk| to_vector(chunk)).collect()
    })
}

/// A chunk's surface in `layer` for GDScript (`WaveForgeStages::volume_surface`); empty if it is
/// not built.
fn layer_surface(layer: Option<&VolumeLayer>, chunk: Vector3i) -> VarDictionary {
    let mut out = VarDictionary::new();
    let Some(Surface { mesh, .. }) = layer.and_then(|layer| layer.built.get(&from_vector(chunk)))
    else {
        return out;
    };
    let vectors = |values: &[[f32; 3]]| -> PackedVector3Array {
        values
            .iter()
            .map(|&[x, y, z]| Vector3::new(x, y, z))
            .collect()
    };
    let indices: PackedInt32Array = mesh
        .indices
        .chunks(3)
        .flat_map(|triangle| [triangle[0], triangle[2], triangle[1]])
        .map(|index| index as i32)
        .collect();
    out.set(
        &"positions".to_variant(),
        &vectors(&mesh.positions).to_variant(),
    );
    out.set(
        &"normals".to_variant(),
        &vectors(&mesh.normals).to_variant(),
    );
    out.set(&"indices".to_variant(), &indices.to_variant());
    out.set(
        &"materials".to_variant(),
        &PackedByteArray::from(mesh.materials.as_slice()).to_variant(),
    );
    out
}

/// The `RenderingServer` mesh a chunk's surface in `layer` is drawn with; an invalid RID if it is
/// not drawn.
fn layer_mesh(layer: Option<&VolumeLayer>, chunk: Vector3i) -> Rid {
    layer
        .and_then(|layer| layer.built.get(&from_vector(chunk)))
        .and_then(|surface| surface.drawn)
        .map_or(Rid::Invalid, |(mesh, _)| mesh)
}

/// A chunk's ground as a surface's arrays without its triangles: its vertices and normals.
fn ground_arrays(mesh: &GroundMesh) -> VarArray {
    let mut arrays = VarArray::new();
    arrays.resize(ArrayType::MAX.ord() as usize, &Variant::nil());
    let vertices: PackedVector3Array = mesh
        .positions
        .iter()
        .map(|&[x, y, z]| Vector3::new(x, y, z))
        .collect();
    let normals: PackedVector3Array = mesh
        .normals
        .iter()
        .map(|&[x, y, z]| Vector3::new(x, y, z))
        .collect();
    arrays.set(ArrayType::VERTEX.ord() as usize, &vertices.to_variant());
    arrays.set(ArrayType::NORMAL.ord() as usize, &normals.to_variant());
    arrays
}

/// A chunk's ground's finest triangles, and its coarser levels with how far each strays.
fn ground_levels(mesh: &GroundMesh) -> (&[u32], Vec<(&[u32], f32)>) {
    let [finest, coarser @ ..] = mesh.levels.as_slice() else {
        unreachable!("a ground has full detail at least");
    };
    let coarser = coarser
        .iter()
        .map(|level| (level.indices.as_slice(), level.error))
        .collect();
    (&finest.indices, coarser)
}

/// What a stroke reads in the node: the runtime it samples with, and the products its stages'
/// thread sent.
struct NodeCanvas<'a> {
    sampler: &'a Runtime,
    worker: &'a StageWorker,
}

impl Canvas for NodeCanvas<'_> {
    fn pack(&self) -> &Pack {
        self.sampler.pack()
    }

    fn chunk_size(&self) -> [u32; 2] {
        self.sampler.chunk_size()
    }

    fn sample(&self, stage: &str, at: [f32; 2]) -> Result<f32, StageError> {
        self.sampler.sample(stage, at)
    }

    fn points(&self, stage: &str, chunk: ChunkCoord) -> Option<&[Point]> {
        self.worker.points(stage, chunk)
    }
}

/// The brush a GDScript Dictionary describes ([`WaveForgeStages::paint`]); none if it names no
/// brush or lacks what its brush needs.
fn brush_of(brush: &VarDictionary) -> Option<Brush> {
    let text = |key: &str| Some(brush.get(key)?.try_to::<GString>().ok()?.to_string());
    let number = |key: &str| {
        let value = brush.get(key)?;
        value
            .try_to::<f64>()
            .ok()
            .or_else(|| value.try_to::<i64>().ok().map(|whole| whole as f64))
            .map(|value| value as f32)
    };
    Some(match text("brush")?.as_str() {
        "raise" => Brush::Raise {
            stage: text("stage")?,
            radius: number("radius")?,
            strength: number("strength")?,
        },
        "smooth" => Brush::Smooth {
            stage: text("stage")?,
            radius: number("radius")?,
            strength: number("strength")?,
        },
        "dig" => Brush::Dig {
            stage: text("stage")?,
            radius: number("radius")?,
        },
        "fill" => Brush::Fill {
            stage: text("stage")?,
            radius: number("radius")?,
        },
        "remove" => Brush::Remove {
            stages: brush
                .get("stages")?
                .try_to::<PackedStringArray>()
                .ok()?
                .as_slice()
                .iter()
                .map(ToString::to_string)
                .collect(),
            radius: number("radius")?,
        },
        _ => return None,
    })
}

/// The box a candidate is drawn with: unshaded, in each instance's colour.
fn candidate_mesh() -> Gd<BoxMesh> {
    let mut material = StandardMaterial3D::new_gd();
    material.set_shading_mode(ShadingMode::UNSHADED);
    material.set_flag(Flags::ALBEDO_FROM_VERTEX_COLOR, true);
    let mut mesh = BoxMesh::new_gd();
    mesh.set_material(&material);
    mesh
}

/// What a candidate came to, by name ([`WaveForgeStages::candidate_legend`]).
fn verdict_name(verdict: Result<(), Rejection>) -> String {
    match verdict {
        Ok(()) => "kept".to_owned(),
        Err(Rejection::Chance) => "chance".to_owned(),
        Err(Rejection::Height) => "height".to_owned(),
        Err(Rejection::Slope) => "slope".to_owned(),
        Err(Rejection::Condition(number)) => format!("condition {number}"),
        Err(Rejection::Water) => "water".to_owned(),
        Err(Rejection::Sites) => "sites".to_owned(),
        Err(Rejection::Blocked) => "blocked".to_owned(),
        Err(Rejection::Spacing) => "spacing".to_owned(),
    }
}

/// The colour a candidate is drawn in, by what it came to.
fn verdict_colour(verdict: Result<(), Rejection>) -> Color {
    verdict_colour_of(&verdict_name(verdict))
}

/// The colour of a verdict by its name: kept green, each modifier its own.
fn verdict_colour_of(name: &str) -> Color {
    match name {
        "kept" => Color::from_rgb(0.2, 0.9, 0.2),
        "chance" => Color::from_rgb(0.5, 0.5, 0.5),
        "height" => Color::from_rgb(0.2, 0.4, 1.0),
        "slope" => Color::from_rgb(1.0, 0.55, 0.1),
        "water" => Color::from_rgb(0.1, 0.9, 0.9),
        "sites" => Color::from_rgb(0.9, 0.1, 0.1),
        "blocked" => Color::from_rgb(0.55, 0.35, 0.15),
        "spacing" => Color::from_rgb(1.0, 0.9, 0.1),
        // Every condition of `when`.
        _ => Color::from_rgb(0.6, 0.2, 0.9),
    }
}

/// Parameter values from GDScript, name to number; none if any name is not a string or any value
/// not a number.
fn param_values(values: &VarDictionary) -> Option<BTreeMap<String, f32>> {
    values
        .iter_shared()
        .map(|(name, value)| {
            let name = name.try_to::<GString>().ok()?.to_string();
            let value = value
                .try_to::<f64>()
                .ok()
                .or_else(|| value.try_to::<i64>().ok().map(|whole| whole as f64))?;
            Some((name, value as f32))
        })
        .collect()
}

/// What builds the runtime a node generates with, on whatever thread calls it.
type Builder = Arc<dyn Fn() -> Result<Runtime, String> + Send + Sync>;

/// What each stage has generated and cost, by name: `products`, `ms` in all and `slowest_ms` for
/// one product.
fn stage_costs<'a>(timings: impl IntoIterator<Item = &'a (String, StageTiming)>) -> VarDictionary {
    let mut stages = VarDictionary::new();
    for (name, timing) in timings {
        let mut cost = VarDictionary::new();
        cost.set(
            &"products".to_variant(),
            &(timing.products as i64).to_variant(),
        );
        cost.set(&"ms".to_variant(), &timing.ms.to_variant());
        cost.set(&"slowest_ms".to_variant(), &timing.slowest_ms.to_variant());
        stages.set(
            &GString::from(name.as_str()).to_variant(),
            &cost.to_variant(),
        );
    }
    stages
}

/// Whether a stage, by name and kind, fits a setting ([`WaveForgeStages::stage_setting_problems`]).
type Fits<'a> = &'a dyn Fn(&str, &StageKind) -> bool;

/// A world run on a thread of its own ([`WaveForgeStages::run_world`]): what it reports, and the
/// flag that stops it. The thread is joined before the process exits.
struct WorldRunning {
    progress: std::sync::mpsc::Receiver<RunEvent>,
    cancel: Arc<std::sync::atomic::AtomicBool>,
}

// Nothing reads a dropped run's progress, and the process waits for its thread before exiting.
impl Drop for WorldRunning {
    fn drop(&mut self) {
        self.cancel
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

/// What a world run's thread has reported since the last look. A thread that stopped without
/// reporting its end, as a panic stops it, ends the run as failed, so the node does not wait on it.
fn received(progress: &std::sync::mpsc::Receiver<RunEvent>) -> Vec<RunEvent> {
    let mut events = Vec::new();
    loop {
        match progress.try_recv() {
            Ok(event) => events.push(event),
            Err(std::sync::mpsc::TryRecvError::Empty) => return events,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                if !matches!(events.last(), Some(RunEvent::Ended(_))) {
                    events.push(RunEvent::Ended(Err(
                        "its thread stopped without finishing; the panic above says why".to_owned(),
                    )));
                }
                return events;
            }
        }
    }
}

/// What a world run's thread reports.
enum RunEvent {
    /// One more chunk done.
    Progress(RunProgress),
    /// The run is over: where it got to, or why it failed.
    Ended(Result<RunProgress, String>),
}

/// The prefix of the properties the inspector shows a pack's parameters as.
const PARAMS: &str = "params/";

/// The metadata [`WaveForgeStages::bake`] leaves on a chunk's node: the chunk.
const BAKED_CHUNK: &str = "wave_forge_chunk";
/// On a chunk's node: every point it placed as a node, as each one's [`BAKED_POINT`].
const BAKED_POINTS: &str = "wave_forge_points";
/// On a node a stage placed as a point: its `stage`, the `chunk` and `id` of its positional id,
/// where it stood in cells (`at`) and its `transform`.
const BAKED_POINT: &str = "wave_forge_point";
/// On every other node the generator made.
const BAKED_PART: &str = "wave_forge_generated";

/// What names a baked point: its stage, and the chunk and id of its positional id.
fn point_key(mark: &VarDictionary) -> (String, Vector3i, i64) {
    (
        mark.at("stage").to::<GString>().to_string(),
        mark.at("chunk").to(),
        mark.at("id").to(),
    )
}

/// Makes `root` the owner of every node under `node`, so packing `root` keeps them. An instance of
/// a saved scene is kept as a reference to that scene, so its own nodes are left to it; an
/// instance of a scene made in memory is kept node by node.
fn own(node: &Gd<Node>, root: &Gd<Node>) {
    for mut child in node.get_children().iter_shared() {
        child.set_owner(root);
        if child.get_scene_file_path().is_empty() {
            own(&child, root);
        }
    }
}

/// A volume surface's triangles as a concave polygon's faces, three corners each in Godot's
/// winding: its front faces wind clockwise, the library's counter-clockwise.
fn surface_faces(mesh: &VolumeMesh) -> PackedVector3Array {
    mesh.indices
        .chunks(3)
        .flat_map(|triangle| [triangle[0], triangle[2], triangle[1]])
        .map(|index| {
            let [x, y, z] = mesh.positions[index as usize];
            Vector3::new(x, y, z)
        })
        .collect()
}

/// A town chunk's module instances of the modules `wanted` keeps, in chunks of `shape` with cells
/// of `cell`, from the rule set in `rules` it was solved with, and how far its site lifts them in
/// Godot's units; none outside every town or before the chunk arrives.
fn town_sets(
    worker: &StageWorker,
    rules: &BTreeMap<String, RuleFile>,
    shape: ChunkShape,
    cell: Vector3,
    stage: &str,
    chunk: ChunkCoord,
    wanted: impl Fn(&str) -> bool,
) -> Option<(Vec<InstanceSet>, f32)> {
    let town = worker.tiles(stage, chunk)?;
    // The rule set the town was solved with, which its row may have chosen over the stage's.
    let file = rules.get(town.rules.as_ref())?;
    let space = YUpSpace::new(shape, cell.to_array());
    let tiles = Chunk {
        coord: chunk,
        tiles: town.tiles.to_vec().into_boxed_slice(),
        version: 1,
    };
    let sets = wave_forge::instance_sets(&tiles, file, &space, wanted);
    Some((sets, town.height * cell.y))
}

/// What placements read a stage's chunk from: the stages and their pack, the rule sets towns are
/// solved with, the chunks' shape and cells, and the kinds bound to a scene.
struct Placing<'a> {
    worker: &'a StageWorker,
    pack: &'a Pack,
    rules: &'a BTreeMap<String, RuleFile>,
    shape: ChunkShape,
    cell: Vector3,
    bound: &'a [String],
}

/// What `stage` placed in `chunk` of the kinds `placing` binds, where each stands in Godot's
/// world: a point's kind, a piece's name, or a town's module, from the rule set its town was
/// solved with. None for a stage that places nothing or a chunk not generated.
fn placement_items(placing: &Placing, stage: &str, chunk: ChunkCoord) -> Option<Vec<Item>> {
    let &Placing {
        worker,
        pack,
        rules,
        shape,
        cell,
        bound,
    } = placing;
    let binds = |kind: &str| bound.iter().any(|known| known == kind);
    let place = |rows: [[f32; 3]; 3], [x, y, height]: [f32; 3]| {
        Transform3D::new(
            Basis::from_rows(
                Vector3::from_array(rows[0]),
                Vector3::from_array(rows[1]),
                Vector3::from_array(rows[2]),
            ),
            Vector3::new(x * cell.x, height * cell.y, y * cell.z),
        )
    };
    let items: Vec<Item> = match pack.kind(stage)? {
        StageKind::Scatter { .. }
        | StageKind::Embed { .. }
        | StageKind::Deposit { .. }
        | StageKind::Spawn { .. } => worker
            .points(stage, chunk)?
            .iter()
            .filter(|point| binds(&point.kind))
            .map(|point| Item {
                kind: point.kind.to_string(),
                transform: place(point.y_up_basis(), point.position),
                id: local_id(point.id.local),
                // The shader bends a plant by the wind over its stiffness in the mesh's
                // own units, which its scale then enlarges: a stiffness of its scale moves
                // every plant's tip the wind's strength, and a larger plant leans less.
                sway: [phase(point.id.local), point.scale],
                gi: Gi::Off,
            })
            .collect(),
        StageKind::Assemble { .. } | StageKind::Cave { .. } => worker
            .stamps(stage, chunk)?
            .iter()
            // A piece overlapping several chunks is placed by the one its id names.
            .filter(|stamp| stamp.id.chunk == chunk && binds(&stamp.piece))
            .map(|stamp| Item {
                kind: stamp.piece.to_string(),
                transform: place(stamp.y_up_basis(), stamp.position),
                id: local_id(stamp.id.local),
                sway: [phase(stamp.id.local), 1.0],
                gi: Gi::Static,
            })
            .collect(),
        StageKind::Solve { .. } => {
            let (sets, lift) = town_sets(worker, rules, shape, cell, stage, chunk, binds)?;
            sets.iter()
                .flat_map(|set| {
                    let transforms = set.transforms(cell.to_array());
                    let rows: Vec<[f32; 12]> = transforms
                        .chunks(12)
                        .map(|row| row.try_into().expect("twelve floats per instance"))
                        .collect();
                    rows.into_iter().zip(&set.ids).map(move |(row, id)| Item {
                        kind: set.name.clone(),
                        transform: Transform3D::new(
                            Basis::from_rows(
                                Vector3::new(row[0], row[1], row[2]),
                                Vector3::new(row[4], row[5], row[6]),
                                Vector3::new(row[8], row[9], row[10]),
                            ),
                            Vector3::new(row[3], row[7] + lift, row[11]),
                        ),
                        id: local_id(id.local),
                        sway: [0.0, 1.0],
                        gi: Gi::Static,
                    })
                })
                .collect()
        }
        _ => return None,
    };
    Some(items)
}

/// The colour of category `index` from `palette`, the categories past its end taking colours of
/// their own from their index.
fn palette_colour(palette: &PackedColorArray, index: usize) -> Color {
    palette
        .get(index)
        .unwrap_or_else(|| Color::from_hsv(index as f64 * 0.618_034 % 1.0, 0.45, 0.6))
}

/// A placement's phase in the wind as a fraction of a turn, from its id, so neighbours sway apart.
fn phase(local: u64) -> f32 {
    let mixed = (local ^ (local >> 29)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    (mixed >> 40) as f32 / (1u64 << 24) as f32
}

/// A chunk's copy of `template` holding the material `ids` of its ground's vertices, one texel each.
fn chunk_material(
    template: &Gd<ShaderMaterial>,
    palette: &Gd<ImageTexture>,
    mesh: &GroundMesh,
    ids: &[u8],
    cell: [f32; 3],
) -> Gd<ShaderMaterial> {
    let image = Image::create_from_data(
        mesh.size[0] as i32,
        mesh.size[1] as i32,
        false,
        ImageFormat::R8,
        &PackedByteArray::from(ids),
    )
    .expect("an image of one byte per vertex");
    let materials = ImageTexture::create_from_image(&image).expect("a texture of the image");
    let mut material = template.duplicate_resource();
    material.set_shader_parameter("wave_forge_materials", &materials.to_variant());
    material.set_shader_parameter("wave_forge_palette", &palette.to_variant());
    material.set_shader_parameter(
        "wave_forge_cell",
        &Vector2::new(cell[0], cell[2]).to_variant(),
    );
    material
}

#[cfg(test)]
mod tests {
    use super::{RunEvent, received};
    use wave_forge::stages::RunProgress;

    fn progress(done: usize) -> RunProgress {
        RunProgress {
            done,
            total: 4,
            skipped: 0,
            held: 0,
            stages: Vec::new(),
        }
    }

    #[test]
    fn a_world_run_whose_thread_stops_without_an_end_ends_failed() {
        let (sent, progress_events) = std::sync::mpsc::channel();
        sent.send(RunEvent::Progress(progress(1)))
            .expect("the receiver is alive");

        drop(sent);
        let events = received(&progress_events);

        assert!(matches!(events.first(), Some(RunEvent::Progress(_))));
        assert!(
            matches!(events.last(), Some(RunEvent::Ended(Err(_)))),
            "the run did not end"
        );
    }

    #[test]
    fn a_world_run_that_ended_ends_once() {
        let (sent, progress_events) = std::sync::mpsc::channel();
        sent.send(RunEvent::Ended(Ok(progress(4))))
            .expect("the receiver is alive");

        drop(sent);
        let events = received(&progress_events);

        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], RunEvent::Ended(Ok(_))));
    }
}
