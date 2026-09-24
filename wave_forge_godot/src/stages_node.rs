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

use crate::grass::{GRASS_SHADER, Grass};
use crate::placements::{Item, Placements};
use crate::timings::Timings;
use crate::{BODIES_PER_FRAME, RECENT_FRAMES, from_vector, local_id, to_vector};
use godot::classes::image::Format as ImageFormat;
use godot::classes::physics_server_3d::BodyMode;
use godot::classes::rendering_server::{ArrayType, PrimitiveType};
use godot::classes::{
    FastNoiseLite, FileAccess, INode, Image, ImageTexture, Material, Node, Node3D, PhysicsServer3D,
    ProjectSettings, RenderingServer, Shader, ShaderMaterial, Shape3D,
};
use godot::obj::EngineEnum;
use godot::prelude::*;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use wave_forge::loader::{RuleFile, parse_rule_file};
use wave_forge::noise::{
    CellularDistanceFunction, CellularReturnType, DomainWarpFractalType, DomainWarpType,
    FractalType, NoiseConfig, NoiseType,
};
use wave_forge::stages::regions::CurveId;
use wave_forge::stages::{
    Column, Edit, Edits, Facts, GivenRow, MAX_CATEGORIES, Pack, PointId, RowId, Runtime, Save,
    SiteId, StageEvent, StageKind, StageWorker, TableKind, Value,
};
use wave_forge::towns::WfcTowns;
use wave_forge::{
    Chunk, ChunkCoord, ChunkShape, FocusPoint, GroundMesh, InstanceSet, YUpSpace, ground,
    ground_materials, ground_readers,
};

/// Generates a world from a pack of stages around a position the game keeps handing it.
///
/// Set `pack_file`, the rule files its Solve stages name and the stages to generate, call
/// [`WaveForgeStages::start`], then [`WaveForgeStages::follow`] as the player moves, and read each
/// chunk the `stage_ready` signal names.
#[derive(GodotClass)]
#[class(base = Node)]
pub struct WaveForgeStages {
    base: Base<Node>,

    /// The pack of stages to generate: a `*.world.ron` file.
    #[export_group(name = "Pack")]
    #[export(file = "*.ron")]
    pack_file: GString,
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
    noises: VarDictionary,
    /// Whether to start as soon as the node enters the scene tree.
    #[export]
    start_on_ready: bool,

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
    /// A radius of its own for some targets, as stage name to chunks, the others keeping
    /// `view_radius`: ground far out, locations nearer and clutter nearest, say.
    #[export]
    target_radii: VarDictionary,

    /// The field stage the ground is built from, a height in cells per column; empty for no
    /// ground. It has to be generated, as a target or as what a target reads. A chunk's ground
    /// needs the fields of the chunks around it, so it reaches one chunk less than the view.
    #[export_group(name = "Ground")]
    #[export]
    ground_stage: GString,
    /// The material the ground is drawn with; none draws it with Godot's default.
    #[export]
    ground_material: Option<Gd<Material>>,
    /// A Rules or Area stage at the ground's scale whose categories are the ground's materials;
    /// empty for none. Each chunk's ground then gets its own copy of `ground_material`, which has to
    /// be a `ShaderMaterial` taking `wave_forge_materials`, `wave_forge_cell` and
    /// `wave_forge_palette`, or of the reference ground shader when `ground_material` is empty.
    #[export]
    ground_material_stage: GString,
    /// A colour per category of `ground_material_stage`, by index; categories past its end take
    /// colours of their own from their index.
    #[export]
    ground_palette: PackedColorArray,

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
    worker: Option<StageWorker>,
    followed: Option<ChunkCoord>,
    process_ms: Timings,
    /// Each town module's collision shape, by module name, for every Solve stage.
    collision_shapes: HashMap<String, Gd<Shape3D>>,
    /// Chunks whose ground may be buildable: a field around them arrived since they were last
    /// looked at.
    ground_due: std::collections::BTreeSet<ChunkCoord>,
    /// The chunks whose ground is built: its mesh, and its `RenderingServer` mesh and instance.
    grounds: HashMap<ChunkCoord, (GroundMesh, Rid, Rid)>,
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
    /// Signals not yet emitted, in the order their events arrived.
    pending: VecDeque<StageEvent>,
    /// Each chunk's static body and its ground's height map shape, which the body does not own,
    /// and what it holds, to tell when it has to be built again.
    bodies: HashMap<ChunkCoord, (Rid, Option<Rid>, BodyContents)>,
    /// Chunks within `collider_radius` still waiting for a body after the last frame.
    bodies_pending: usize,
    /// The scenes bound to kinds, and what each stage's chunk placed.
    placements: Placements,
}

/// The most `stage_ready` and `stage_dropped` signals one frame emits. A wide request can bring
/// thousands of products at once, and emitting each costs a microsecond or two before any handler
/// a game connected runs, so the rest wait for the next frames.
const SIGNALS_PER_FRAME: usize = 256;

/// The most chunks one frame gives ground. A wide request makes dozens buildable at once, and
/// building each takes tenths of a millisecond on Godot's thread, so the rest wait for the next
/// frames, nearest the followed position first.
const GROUNDS_PER_FRAME: usize = 8;

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
    /// Chunks whose ground was built, and the milliseconds that took.
    grounds: usize,
    grounds_ms: f64,
    /// Chunks whose body was built, and the milliseconds that took.
    bodies: usize,
    bodies_ms: f64,
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

/// What a chunk's body was built from: whether it has the ground, and which Solve stages' towns.
#[derive(Clone, Debug, PartialEq, Eq)]
struct BodyContents {
    ground: bool,
    towns: Vec<String>,
}

#[godot_api]
impl INode for WaveForgeStages {
    fn init(base: Base<Node>) -> Self {
        Self {
            base,
            pack_file: GString::new(),
            rules_files: VarDictionary::new(),
            noises: VarDictionary::new(),
            target_radii: VarDictionary::new(),
            targets: PackedStringArray::new(),
            start_on_ready: false,
            seed: 0,
            chunk_cells: Vector3i::new(8, 8, 8),
            cell_size: Vector3::ONE,
            view_radius: 2,
            pack: None,
            facts: None,
            edits: Edits::default(),
            sampler: None,
            rules: BTreeMap::new(),
            worker: None,
            followed: None,
            process_ms: Timings::new(RECENT_FRAMES),
            ground_stage: GString::new(),
            kernel_cache: GString::from("user://wave_forge/kernels"),
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
            grounds: HashMap::new(),
            ground_due: std::collections::BTreeSet::new(),
            bodies: HashMap::new(),
            bodies_pending: 0,
            scenes: VarDictionary::new(),
            placement_budget_ms: 2.0,
            promotion_radius: -1,
            placements: Placements::default(),
            slowest_frame: FrameCost::default(),
            pending: VecDeque::new(),
        }
    }

    /// Frees the ground's meshes and the bodies, which belong to the rendering and physics
    /// servers rather than to the node.
    fn exit_tree(&mut self) {
        self.clear_ground_and_bodies();
    }

    fn ready(&mut self) {
        if self.start_on_ready {
            self.start();
        }
    }

    /// Hands the frame whatever the stages' thread finished, as signals.
    fn process(&mut self, _delta: f64) {
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
        let ground_stage = self.ground_stage.to_string();
        let (mut arrived, mut gone) = (Vec::new(), Vec::new());
        if self.placements.any() {
            let placing = |stage: &str| {
                matches!(
                    self.pack.as_ref().and_then(|pack| pack.kind(stage)),
                    Some(StageKind::Scatter { .. } | StageKind::Assemble { .. })
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
                    | StageEvent::Saved => {}
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
                StageEvent::Dropped { stage, chunk } if *stage == ground_stage => gone.push(*chunk),
                StageEvent::Generated { .. } | StageEvent::Dropped { .. } | StageEvent::Saved => {}
            }
        }
        if let Some(save) = save {
            self.signals()
                .saved()
                .emit(&GString::from(save.to_ron().as_str()));
        }
        self.pending.extend(
            events
                .into_iter()
                .filter(|event| *event != StageEvent::Saved),
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
                StageEvent::Saved => unreachable!("a save is signalled as it arrives"),
            }
        }
        frame.signals_ms = elapsed_ms(signalling);
        let grounding = std::time::Instant::now();
        frame.grounds = self.update_ground(&arrived, &gone);
        frame.grounds_ms = elapsed_ms(grounding);
        let building = std::time::Instant::now();
        frame.bodies = self.update_colliders();
        frame.bodies_ms = elapsed_ms(building);
        self.update_grass();
        let placing = std::time::Instant::now();
        frame.placed = self.update_placements();
        frame.placements_ms = elapsed_ms(placing);
        frame.ms = elapsed_ms(processing);
        self.process_ms.push(frame.ms);
        if frame.ms > self.slowest_frame.ms {
            self.slowest_frame = frame;
        }
    }
}

#[godot_api]
impl WaveForgeStages {
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

    /// Loads `pack_file` and the rule sets its Solve stages name, and starts the stages' thread.
    /// Returns whether it could start; why not is reported as an error. A town solver builds its
    /// device on the stages' thread, so a failure there arrives as `generation_failed`.
    #[func]
    fn start(&mut self) -> bool {
        let text = FileAccess::get_file_as_string(&self.pack_file).to_string();
        if text.is_empty() {
            godot_error!("wave forge: pack_file {} could not be read", self.pack_file);
            return false;
        }
        let pack = match Pack::parse(&text) {
            Ok(pack) => Arc::new(pack),
            Err(error) => {
                godot_error!("wave forge: {}: {error}", self.pack_file);
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
        if !self.ground_material_stage.is_empty() {
            match self.materials_template(&pack) {
                Ok(template) => self.palette = Some(template),
                Err(error) => {
                    godot_error!("wave forge: {error}");
                    return false;
                }
            }
        } else {
            self.palette = None;
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
        let cache = (!self.kernel_cache.is_empty()).then(|| {
            std::path::PathBuf::from(
                ProjectSettings::singleton()
                    .globalize_path(&self.kernel_cache)
                    .to_string(),
            )
        });
        let mut noises = Vec::new();
        for (name, noise) in self.noises.iter_shared() {
            let name = name.to::<GString>().to_string();
            let Ok(noise) = noise.try_to::<Gd<FastNoiseLite>>() else {
                godot_error!("wave forge: noise {name:?} is not a FastNoiseLite");
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
                godot_error!("wave forge: {}: {error}", self.pack_file);
                return false;
            }
        };
        let mut sampler =
            match with_noises.clone()(Runtime::new(Arc::clone(&pack), seed, [shape.x, shape.y])) {
                Ok(sampler) => sampler,
                Err(error) => {
                    godot_error!("wave forge: {}: {error}", self.pack_file);
                    return false;
                }
            };
        sampler
            .set_facts(facts.clone())
            .expect("facts made for the sampler's pack and seed");
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
        // The towns' device is built on the stages' thread, which is where it is used.
        self.worker = Some(StageWorker::spawn(move || {
            let mut runtime = with_noises(Runtime::new(for_thread, seed, [shape.x, shape.y]))?;
            runtime
                .set_facts(thread_facts)
                .map_err(|error| error.to_string())?;
            if !solves {
                return Ok(runtime);
            }
            let mut towns = WfcTowns::new(shape);
            for (name, file) in rules {
                towns = match &cache {
                    Some(dir) => towns.with_rules(&name, file, |rules| {
                        wave_forge::towns::gpu_solver_cached(rules, dir)
                    }),
                    None => towns.with_rules(&name, file, wave_forge::towns::gpu_solver),
                }
                .map_err(|error| error.to_string())?;
            }
            runtime
                .with_towns(Box::new(towns))
                .map_err(|error| error.to_string())
        }));
        true
    }

    /// Generates around `position`, in Godot's world space. Call it as the player moves; it asks
    /// for new chunks only when the position enters another chunk.
    #[func]
    fn follow(&mut self, position: Vector3) {
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
                    .get(&target.to_variant())
                    .and_then(|radius| radius.try_to::<i64>().ok())
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
        GString::from(GROUND_SHADER)
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
        GString::from(VEGETATION_SHADER)
    }

    /// The reference grass shader's code, to copy into a shader of a game's own.
    #[func]
    fn grass_shader_code(&self) -> GString {
        GString::from(GRASS_SHADER)
    }

    /// The chunks whose ground is built.
    #[func]
    fn ground_chunks(&self) -> Array<Vector3i> {
        self.grounds.keys().map(|&chunk| to_vector(chunk)).collect()
    }

    /// The chunks that have a static body: their ground, and their towns' modules.
    #[func]
    fn collider_chunks(&self) -> Array<Vector3i> {
        self.bodies.keys().map(|&chunk| to_vector(chunk)).collect()
    }

    /// A Sites, TableSites or Locations stage's sites that overlap a chunk: what names each one,
    /// its `region` (Vector2i), the `row` (PackedInt64Array) of its table it stands for, or its
    /// `region` and `index` in a location table along with its `kind`; the chunks it covers from
    /// `min` up to but not including `max` (Vector2i, along the lattice's x and y); and its
    /// levelled `height` in cells. Empty if there are none or the chunk has not arrived.
    #[func]
    fn sites(&self, stage: GString, chunk: Vector3i) -> Array<VarDictionary> {
        let Some(sites) = self
            .worker
            .as_ref()
            .and_then(|worker| worker.sites(&stage.to_string(), from_vector(chunk)))
        else {
            return Array::new();
        };
        sites
            .iter()
            .map(|site| {
                let mut out = VarDictionary::new();
                name_site(&mut out, &site.id);
                if let Some(kind) = &site.kind {
                    out.set(&"kind".to_variant(), &GString::from(&**kind).to_variant());
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
            })
            .collect()
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

    /// What the node has cost Godot's thread: `process_ms_median`, `_p99` and `_max` over recent
    /// frames, once there are some; and what its slowest frame since the start spent the time on:
    /// `slowest_frame_ms` in all, `slowest_frame_events` signals emitted in
    /// `slowest_frame_signals_ms` (the handlers connected to them included),
    /// `slowest_frame_grounds` chunks given ground in `slowest_frame_grounds_ms`, and
    /// `slowest_frame_bodies` chunks given a body in `slowest_frame_bodies_ms`. And `stages`: what
    /// each stage has cost on the stages' thread, by name, as `products`, `ms` in all and
    /// `slowest_ms` for one product. And `pending_signals`, `pending_grounds` and
    /// `pending_colliders`: the signals, grounds and bodies waiting for a later frame.
    #[func]
    fn stats(&self) -> VarDictionary {
        let mut out = VarDictionary::new();
        let mut stages = VarDictionary::new();
        for (name, timing) in self.worker.iter().flat_map(|worker| worker.timings()) {
            let mut cost = VarDictionary::new();
            cost.set(
                &"products".to_variant(),
                &(timing.products as i64).to_variant(),
            );
            cost.set(&"ms".to_variant(), &timing.ms.to_variant());
            cost.set(&"slowest_ms".to_variant(), &timing.slowest_ms.to_variant());
            stages.set(&GString::from(name).to_variant(), &cost.to_variant());
        }
        out.set(&"stages".to_variant(), &stages.to_variant());
        out.set(
            &"pending_signals".to_variant(),
            &(self.pending.len() as i64).to_variant(),
        );
        out.set(
            &"pending_grounds".to_variant(),
            &(self.ground_due.len() as i64).to_variant(),
        );
        out.set(
            &"pending_colliders".to_variant(),
            &(self.bodies_pending as i64).to_variant(),
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
        let slowest = self.slowest_frame;
        for (key, value) in [
            ("slowest_frame_ms", slowest.ms),
            ("slowest_frame_signals_ms", slowest.signals_ms),
            ("slowest_frame_grounds_ms", slowest.grounds_ms),
            ("slowest_frame_bodies_ms", slowest.bodies_ms),
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
        let (pack, worker) = (self.pack.as_ref()?, self.worker.as_ref()?);
        let Some(StageKind::Solve { rules, .. }) = pack.kind(stage) else {
            return None;
        };
        let (file, town) = (self.rules.get(rules)?, worker.tiles(stage, chunk)?);
        let space = YUpSpace::new(self.chunk_shape(), self.cell_size.to_array());
        let tiles = Chunk {
            coord: chunk,
            tiles: town.tiles.to_vec().into_boxed_slice(),
            version: 1,
        };
        let sets = wave_forge::instance_sets(&tiles, file, &space, wanted);
        Some((sets, town.height * self.cell_size.y))
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
    /// completed, nearest the followed position first, and frees the ground of chunks whose own
    /// field was dropped.
    ///
    /// Returns how many chunks got ground.
    fn update_ground(&mut self, arrived: &[ChunkCoord], gone: &[ChunkCoord]) -> usize {
        let mut rendering = RenderingServer::singleton();
        for chunk in gone {
            if let Some((_, mesh, instance)) = self.grounds.remove(chunk) {
                rendering.free_rid(instance);
                rendering.free_rid(mesh);
            }
            self.chunk_materials.remove(chunk);
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
        self.ground_due
            .extend(arrived.iter().copied().flat_map(ground_readers));
        for chunk in gone {
            self.ground_due.remove(chunk);
        }
        let focus = self.followed.unwrap_or(ChunkCoord::new(0, 0, 0));
        let mut due: Vec<ChunkCoord> = self.ground_due.iter().copied().collect();
        due.sort_by_key(|chunk| {
            (
                (chunk.x - focus.x).abs().max((chunk.y - focus.y).abs()),
                *chunk,
            )
        });
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
        let count = built.len();
        for (chunk, mesh, ids) in built {
            let rid = rendering.mesh_create();
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
            let [finest, coarser @ ..] = mesh.levels.as_slice() else {
                unreachable!("a ground has full detail at least");
            };
            arrays.set(
                ArrayType::INDEX.ord() as usize,
                &godot_triangles(&finest.indices).to_variant(),
            );
            let errors: Vec<f32> = coarser.iter().map(|level| level.error).collect();
            let mut lods = VarDictionary::new();
            // A coarser level overwrites a finer one of the same key, so of levels that stray
            // alike the coarsest is drawn.
            for (key, level) in lod_keys(&errors).into_iter().zip(coarser) {
                lods.set(key, &godot_triangles(&level.indices).to_variant());
            }
            rendering
                .mesh_add_surface_from_arrays_ex(rid, PrimitiveType::TRIANGLES, &arrays)
                .lods(&lods)
                .done();
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
            self.grounds.insert(chunk, (mesh, rid, instance));
        }
        count
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
        let solves: Vec<String> = pack
            .stage_names()
            .filter(|name| matches!(pack.kind(name), Some(StageKind::Solve { .. })))
            .map(ToOwned::to_owned)
            .collect();
        let contents = |this: &Self, chunk: ChunkCoord| BodyContents {
            ground: this.grounds.contains_key(&chunk),
            towns: if this.collision_shapes.is_empty() {
                Vec::new()
            } else {
                solves
                    .iter()
                    .filter(|stage| {
                        this.worker
                            .as_ref()
                            .is_some_and(|worker| worker.tiles(stage, chunk).is_some())
                    })
                    .cloned()
                    .collect()
            },
        };
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
            .filter(|(_, held)| held.ground || !held.towns.is_empty())
            .collect();
        // Nearest first, and a few a frame, as the city node does.
        wanted.sort_by_key(|(chunk, _)| {
            (
                (chunk.x - focus.x).abs().max((chunk.y - focus.y).abs()),
                *chunk,
            )
        });
        self.bodies_pending = wanted.len().saturating_sub(BODIES_PER_FRAME);
        wanted.truncate(BODIES_PER_FRAME);
        let owner = u64::from_ne_bytes(self.base().instance_id().to_i64().to_ne_bytes());
        let count = wanted.len();
        for (chunk, held) in wanted {
            let body = physics.body_create();
            physics.body_set_mode(body, BodyMode::STATIC);
            let ground_shape = if held.ground {
                self.add_ground_shape(&mut physics, body, chunk)
            } else {
                None
            };
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
            self.bodies.insert(chunk, (body, ground_shape, held));
        }
        count
    }

    /// Adds a chunk's ground to `body` as a height map and returns the shape, which the caller
    /// frees with the body. A height map's samples are one unit apart, so the shape is scaled by
    /// the cell's width, and its heights divided by it, which needs cells as wide as they are deep.
    fn add_ground_shape(
        &self,
        physics: &mut Gd<PhysicsServer3D>,
        body: Rid,
        chunk: ChunkCoord,
    ) -> Option<Rid> {
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
        let (low, high) = heights
            .as_slice()
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), &h| {
                (low.min(h), high.max(h))
            });
        let mut data = VarDictionary::new();
        data.set(&"width".to_variant(), &(mesh.size[0] as i32).to_variant());
        data.set(&"depth".to_variant(), &(mesh.size[1] as i32).to_variant());
        data.set(&"heights".to_variant(), &heights.to_variant());
        data.set(&"min_height".to_variant(), &low.to_variant());
        data.set(&"max_height".to_variant(), &high.to_variant());
        let shape = physics.heightmap_shape_create();
        physics.shape_set_data(shape, &data.to_variant());
        // A height map is centred on its origin; the first vertex stands over the first column's
        // centre.
        let centre = Vector3::new(
            (0.5 + (mesh.size[0] - 1) as f32 / 2.0) * width,
            0.0,
            (0.5 + (mesh.size[1] - 1) as f32 / 2.0) * width,
        );
        let at = Transform3D::new(
            Basis::from_scale(Vector3::ONE * width),
            self.chunk_corner(chunk) + centre,
        );
        physics.body_add_shape_ex(body, shape).transform(at).done();
        Some(shape)
    }

    fn free_bodies(&mut self) {
        let mut physics = PhysicsServer3D::singleton();
        for (_, (body, shape, _)) in self.bodies.drain() {
            physics.free_rid(body);
            shape.into_iter().for_each(|shape| physics.free_rid(shape));
        }
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
        let cell = self.cell_size;
        let bound: Vec<String> = self.placements.kinds();
        let binds = |kind: &str| bound.iter().any(|known| known == kind);
        let items = |stage: &str, chunk: ChunkCoord| -> Option<Vec<Item>> {
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
                StageKind::Scatter { .. } => worker
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
                    })
                    .collect(),
                StageKind::Assemble { .. } => worker
                    .stamps(stage, chunk)?
                    .iter()
                    // A piece overlapping several chunks is placed by the one its id names.
                    .filter(|stamp| stamp.id.chunk == chunk && binds(&stamp.piece))
                    .map(|stamp| Item {
                        kind: stamp.piece.to_string(),
                        transform: place(stamp.y_up_basis(), stamp.position),
                        id: local_id(stamp.id.local),
                        sway: [phase(stamp.id.local), 1.0],
                    })
                    .collect(),
                _ => return None,
            };
            Some(items)
        };
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

    /// The palette texture and the material each chunk's ground material is copied from.
    fn materials_template(
        &self,
        pack: &Pack,
    ) -> Result<(Gd<ImageTexture>, Gd<ShaderMaterial>), String> {
        let stage = self.ground_material_stage.to_string();
        match pack.kind(&stage) {
            Some(StageKind::Rules { .. } | StageKind::Area { .. }) => {}
            _ => {
                return Err(format!(
                    "ground_material_stage {stage:?} is no Rules or Area stage of the pack"
                ));
            }
        }
        let template = match &self.ground_material {
            None => {
                let mut shader = Shader::new_gd();
                shader.set_code(GROUND_SHADER);
                let mut material = ShaderMaterial::new_gd();
                material.set_shader(&shader);
                material
            }
            Some(material) => material.clone().try_cast::<ShaderMaterial>().map_err(|_| {
                "ground_material must be a ShaderMaterial when ground_material_stage is set"
                    .to_owned()
            })?,
        };
        let colours: Vec<u8> = (0..MAX_CATEGORIES)
            .flat_map(|index| {
                let colour = self
                    .ground_palette
                    .get(index)
                    .unwrap_or_else(|| Color::from_hsv(index as f64 * 0.618_034 % 1.0, 0.45, 0.6));
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
        if let Some(grass) = &mut self.grass {
            grass.clear();
        }
        let mut rendering = RenderingServer::singleton();
        for (_, (_, mesh, instance)) in self.grounds.drain() {
            rendering.free_rid(instance);
            rendering.free_rid(mesh);
        }
        // Freed after the meshes that draw with them.
        self.chunk_materials.clear();
        self.free_bodies();
        self.placements.clear();
    }
}

/// The reference vegetation shader: plants bending in the global wind.
const VEGETATION_SHADER: &str = include_str!("shaders/vegetation.gdshader");

/// A placement's phase in the wind as a fraction of a turn, from its id, so neighbours sway apart.
fn phase(local: u64) -> f32 {
    let mixed = (local ^ (local >> 29)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    (mixed >> 40) as f32 / (1u64 << 24) as f32
}

/// A ground level's triangles for Godot: the library's are counter-clockwise seen from their front,
/// Godot's clockwise.
fn godot_triangles(indices: &[u32]) -> PackedInt32Array {
    indices
        .chunks(3)
        .flat_map(|triangle| [triangle[0], triangle[2], triangle[1]])
        .map(|index| index as i32)
        .collect()
}

/// The keys of the coarser ground levels in a surface's `lods`, from their errors in world units.
/// Godot draws a level while its key, projected to the screen, stays under the viewport's
/// `mesh_lod_threshold` in pixels, and stops at the first level that does not, in order of key; so
/// a key is at least every finer level's. It skips a key that is not positive, and a level that
/// strays nowhere is drawn at every distance, so such a key is the smallest positive float.
fn lod_keys(errors: &[f32]) -> Vec<f32> {
    errors
        .iter()
        .scan(f32::MIN_POSITIVE, |key, &error| {
            *key = key.max(error);
            Some(*key)
        })
        .collect()
}

/// The reference ground shader: a chunk's material ids per vertex, blended through a palette.
const GROUND_SHADER: &str = include_str!("shaders/ground.gdshader");

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
    use super::*;

    #[test]
    fn lod_keys_never_fall_below_a_finer_levels() {
        let keys = lod_keys(&[0.5, 0.25, 2.0]);

        assert_eq!(keys, [0.5, 0.5, 2.0]);
    }

    #[test]
    fn a_level_that_strays_nowhere_gets_a_positive_key() {
        let keys = lod_keys(&[0.0, 0.0]);

        assert_eq!(keys, [f32::MIN_POSITIVE, f32::MIN_POSITIVE]);
    }
}
