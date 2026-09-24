//! Wave Forge as a Godot GDExtension: a `WaveForgeWorld` node that generates chunks around
//! whatever the game tells it to follow.
//!
//! ```gdscript
//! extends Node3D
//!
//! func _ready() -> void:
//!     var world: WaveForgeWorld = $WaveForgeWorld
//!     world.chunk_updated.connect(_on_chunk_updated)
//!     world.load_rules(FileAccess.get_file_as_string("res://city.ron"))
//!     # What belongs on the ground and what belongs above it, by the tags and names the rule set
//!     # gives its modules. An empty layer allows everything.
//!     var layers: Array[PackedInt32Array] = [
//!         world.tiles_tagged("street_level"), PackedInt32Array(), world.tiles_named("air"),
//!     ]
//!     world.set_layer_tiles(layers)
//!     world.start()
//!
//! func _process(_delta: float) -> void:
//!     $WaveForgeWorld.follow($Player.global_position)
//!
//! func _on_chunk_updated(chunk: Vector3i) -> void:
//!     var tiles := $WaveForgeWorld.tiles_at(chunk)   # one tile index per cell
//!     for cell in tiles.size():
//!         var tile := tiles[cell]
//!         # One model per name, placed at the cell's centre and turned by the tile's rotation.
//! place(world.tile_name(tile), Transform3D(world.tile_basis(tile), world.cell_position(chunk,
//! cell)))
//! ```
//!
//! # Where the work happens
//!
//! Generation runs on a thread of its own, on a GPU device of its own, and the node drains its
//! events in `_process`. Nothing on Godot's side ever blocks on the device, and nothing on the
//! generating thread ever touches a Godot object, so the extension needs no thread-safety features
//! from godot-rust.
//!
//! A device of its own is not the only option the library offers: `ComputeBackend` in `wfc-gpu` is
//! there so that a backend over Godot's own `RenderingDevice` can replace it, which would put
//! generation on the device Godot already has. That backend does not exist yet; see
//! `docs/architecture/engine-integration.md`, "Godot: the solver stays on its own device", for what
//! it needs and what has to be measured to choose between the two.
//!
//! # Coordinates
//!
//! Godot is Y-up and the generator is Z-up. Godot's `xz` ground plane is therefore the lattice's
//! `xy`, and Godot's `y` is the lattice's `z`. `cell_size` says how large one cell is along each of
//! Godot's axes, which is what turns a position into a chunk.

use godot::classes::physics_server_3d::BodyMode;
use godot::classes::{
    FileAccess, INode, NavigationMesh, NavigationMeshSourceGeometryData3D, NavigationServer3D,
    Node, PhysicsServer3D, Shape3D,
};
use godot::prelude::*;
use std::collections::HashMap;
use timings::Timings;
use wave_forge::loader::RuleFile;
use wave_forge::{
    Builder, ChunkCoord, ChunkEvent, ChunkShape, FocusPoint, NavSourceError, Prior, RegionStatus,
    Ruleset, TileMask, Worker, WorldExtent, YUpSpace,
};

mod grass;
mod placements;
mod stages_node;
mod timings;

/// How many recent frames and navigation bakes `stats` summarises: a minute at 60 frames per
/// second, and a few hundred chunks.
const RECENT_FRAMES: usize = 3600;
const RECENT_BAKES: usize = 256;

struct WaveForgeExtension;

#[gdextension]
unsafe impl ExtensionLibrary for WaveForgeExtension {
    fn on_stage_init(stage: InitStage) {
        // As soon as the rendering server is there, before a scene loads, so a material compiled
        // anywhere finds the wind.
        if stage == InitStage::MainLoop {
            grass::ensure_wind();
        }
    }
}

/// Generates a world in chunks around a position the game keeps handing it.
///
/// Set the properties in the editor or from a script, load a rule set with
/// [`WaveForgeWorld::load_rules`], set the prior, call [`WaveForgeWorld::start`], then call
/// [`WaveForgeWorld::follow`] as the player moves and read the tiles of every chunk the
/// `chunk_updated` signal names.
#[derive(GodotClass)]
#[class(base = Node)]
pub struct WaveForgeWorld {
    base: Base<Node>,

    /// The rule set to generate from: a Wave Forge rule file, tiles with their adjacency or
    /// modules described by connectors. Read when the node starts on its own; a script can call
    /// [`WaveForgeWorld::load_rules`] instead.
    #[export_group(name = "Rules")]
    #[export(file = "*.ron")]
    rules_file: GString,
    /// Whether to load `rules_file` and start generating as soon as the node enters the scene
    /// tree. Leave it off to set the prior from a script first.
    #[export]
    start_on_ready: bool,

    /// Every choice in the world derives from this.
    #[export_group(name = "World")]
    #[export]
    seed: i64,
    /// The cells of one chunk, along the lattice's own axes (x, y across the ground, z up).
    #[export]
    chunk_cells: Vector3i,
    /// How large one cell is in Godot's world units, along Godot's axes.
    #[export]
    cell_size: Vector3,
    /// How many chunks the world is, along the lattice's axes. Zero on an axis means unbounded.
    #[export]
    world_chunks: Vector3i,

    /// How many chunks to keep generated around the followed position.
    #[export_group(name = "Streaming")]
    #[export]
    view_radius: i32,
    /// Chunks further than `view_radius` plus this many are dropped. Below one, a chunk that is
    /// about to be asked for again would be dropped, because the chunks a focus needs reach one
    /// beyond its radius. Zero keeps everything.
    #[export]
    evict_margin: i32,

    /// Chunks within this many of the followed chunk get colliders, from the shapes given with
    /// [`WaveForgeWorld::set_collision_shape`]. Keep it below `view_radius`: physics only matters
    /// near the player. Below zero, no chunk gets colliders.
    #[export_group(name = "Physics")]
    #[export]
    collider_radius: i32,

    /// Chunks within this many of the followed chunk get a navigation mesh, baked from the same
    /// shapes as their colliders. Below zero, none do.
    #[export_group(name = "Navigation")]
    #[export]
    navigation_radius: i32,
    /// The settings chunks are baked with: agent size, climb, slope, partitioning. Its cell size
    /// and height are replaced by the navigation map's, which they must match to merge.
    #[export]
    navigation_template: Option<Gd<NavigationMesh>>,

    /// Cells solved around a chunk and thrown away, so its borders can be completed. Changing it
    /// compiles other kernels.
    #[export_group(name = "Advanced")]
    #[export]
    halo: i32,
    /// Whether to compile the kernels a run needs when generation starts, rather than at the first
    /// batch that needs one.
    ///
    /// A kernel is specialised per region shape and per batch size rounded up to a power of two,
    /// and compiling one takes seconds on some drivers. It happens on the generating thread either
    /// way, so this decides whether the wait lands at the start or in the middle.
    #[export]
    warm_kernels: bool,

    /// Tiles allowed on each layer, from the world's lowest layer up, as set by
    /// [`WaveForgeWorld::set_layer_tiles`].
    layers: Vec<Vec<u32>>,
    /// Tiles banned on each of the world's six faces, as set by
    /// [`WaveForgeWorld::ban_tiles_on_face`].
    face_bans: [Vec<u32>; 6],

    /// The rule set [`WaveForgeWorld::load_rules`] read, which also says what each tile is.
    rules: Option<RuleFile>,
    /// The generating thread, once [`WaveForgeWorld::start`] has built it.
    worker: Option<Worker>,
    /// The chunk the last [`WaveForgeWorld::follow`] landed in, so an unmoved player asks nothing.
    followed: Option<ChunkCoord>,
    /// The collision shape of each module that has one, by module name.
    collision_shapes: HashMap<String, Gd<Shape3D>>,
    /// Each chunk's static body and the local id of the instance each of its shapes stands for, in
    /// shape order.
    bodies: HashMap<ChunkCoord, (Rid, Vec<u64>)>,
    /// Chunks whose tiles changed since their body was built, to build again.
    bodies_due: std::collections::BTreeSet<ChunkCoord>,
    /// Chunks within `collider_radius` still waiting for a body after the last frame.
    bodies_pending: usize,
    /// Each chunk's navigation region and the mesh being baked for it.
    navigation: HashMap<ChunkCoord, NavigationChunk>,
    /// The triangles of each module's collision shape, as the navigation bake reads them.
    shape_faces: HashMap<String, Vec<[f32; 3]>>,
    /// Milliseconds of Godot's thread that `process` took on each recent frame, and what the
    /// slowest frame since the start spent its time on.
    process_ms: Timings,
    slowest_frame: FrameCost,
    /// How many navigation bakes have finished, the milliseconds from asking for each recent one to
    /// its mesh being in place, and the milliseconds of Godot's thread each took to prepare and
    /// hand over, of that to gather its source triangles, and to put it in its region once baked.
    baked: i64,
    bake_ms: Timings,
    bake_start_ms: Timings,
    bake_source_ms: Timings,
    bake_finish_ms: Timings,
}

/// What one frame of `process` cost Godot's thread, and what it spent it on.
#[derive(Clone, Copy, Default)]
struct FrameCost {
    ms: f64,
    /// Events drained from the worker, and the milliseconds spent emitting them as signals,
    /// which includes the handlers a game connected to them.
    events: usize,
    signals_ms: f64,
    /// Chunks whose bodies were built, and the milliseconds that took.
    colliders: usize,
    colliders_ms: f64,
    navigation_ms: f64,
}

/// The most chunks one frame gives a collider body, in either node. A body costs Godot's thread
/// about 0.3 ms for a chunk of 512 boxes on the dev container, so three stay near a millisecond.
pub(crate) const BODIES_PER_FRAME: usize = 3;

fn elapsed_ms(since: std::time::Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1000.0
}

/// A chunk's navigation region and the mesh baked for it.
struct NavigationChunk {
    region: Rid,
    mesh: Gd<NavigationMesh>,
    /// When the running bake was asked for, if one is running.
    baking_since: Option<std::time::Instant>,
    /// Whether the chunk or a neighbour changed while it was baking, so it bakes again.
    stale: bool,
}

#[godot_api]
impl INode for WaveForgeWorld {
    fn init(base: Base<Node>) -> Self {
        Self {
            base,
            rules_file: GString::new(),
            start_on_ready: false,
            seed: 0,
            chunk_cells: Vector3i::new(8, 8, 8),
            cell_size: Vector3::ONE,
            view_radius: 2,
            halo: 1,
            evict_margin: 0,
            world_chunks: Vector3i::ZERO,
            warm_kernels: true,
            layers: Vec::new(),
            face_bans: Default::default(),
            rules: None,
            worker: None,
            followed: None,
            collider_radius: 1,
            collision_shapes: HashMap::new(),
            bodies: HashMap::new(),
            bodies_due: std::collections::BTreeSet::new(),
            bodies_pending: 0,
            navigation_radius: -1,
            navigation_template: None,
            navigation: HashMap::new(),
            shape_faces: HashMap::new(),
            process_ms: Timings::new(RECENT_FRAMES),
            slowest_frame: FrameCost::default(),
            baked: 0,
            bake_ms: Timings::new(RECENT_BAKES),
            bake_start_ms: Timings::new(RECENT_BAKES),
            bake_source_ms: Timings::new(RECENT_BAKES),
            bake_finish_ms: Timings::new(RECENT_BAKES),
        }
    }

    /// Frees the chunks' bodies and navigation regions, which belong to the physics and navigation
    /// servers rather than to the tree.
    fn exit_tree(&mut self) {
        let mut physics = PhysicsServer3D::singleton();
        for (_, (body, _)) in self.bodies.drain() {
            physics.free_rid(body);
        }
        let mut navigation = NavigationServer3D::singleton();
        for (_, chunk) in self.navigation.drain() {
            navigation.free_rid(chunk.region);
        }
    }

    /// Starts generating from `rules_file` if the node is set to start on its own.
    fn ready(&mut self) {
        if !self.start_on_ready {
            return;
        }
        if self.rules_file.is_empty() {
            godot_error!("wave forge: start_on_ready is set but rules_file is empty");
            return;
        }
        let text = FileAccess::get_file_as_string(&self.rules_file);
        if text.is_empty() {
            godot_error!("wave forge: {} could not be read", self.rules_file);
            return;
        }
        if self.load_rules(text) {
            self.start();
        }
    }

    /// Hands the frame whatever the generating thread finished, as signals.
    fn process(&mut self, _delta: f64) {
        let processing = std::time::Instant::now();
        let (events, failure) = match &mut self.worker {
            Some(worker) => (worker.drain(), worker.failure().map(ToOwned::to_owned)),
            None => return,
        };
        if let Some(reason) = failure {
            self.worker = None;
            godot_error!("wave forge: generation stopped: {reason}");
            self.signals()
                .generation_failed()
                .emit(&GString::from(&reason));
            return;
        }
        let mut frame = FrameCost {
            events: events.len(),
            ..FrameCost::default()
        };
        let mut updated = Vec::new();
        for event in events {
            match event {
                ChunkEvent::Updated(chunk) => {
                    updated.push(chunk);
                    self.signals().chunk_updated().emit(to_vector(chunk));
                }
                ChunkEvent::Evicted(chunk) => {
                    self.signals().chunk_evicted().emit(to_vector(chunk));
                }
                ChunkEvent::Failed { chunk, status } => {
                    self.signals()
                        .chunk_failed()
                        .emit(to_vector(chunk), status_name(status));
                }
            }
        }
        frame.signals_ms = elapsed_ms(processing);
        let colliding = std::time::Instant::now();
        frame.colliders = self.update_colliders(&updated);
        frame.colliders_ms = elapsed_ms(colliding);
        let navigating = std::time::Instant::now();
        self.update_navigation(&updated);
        frame.navigation_ms = elapsed_ms(navigating);
        frame.ms = elapsed_ms(processing);
        self.process_ms.push(frame.ms);
        if frame.ms > self.slowest_frame.ms {
            self.slowest_frame = frame;
        }
    }
}

#[godot_api]
impl WaveForgeWorld {
    /// A chunk's tiles are new or have changed, so anything built from them is stale. A repair
    /// reports every chunk it rewrote, not only the one it was repairing.
    #[signal]
    fn chunk_updated(chunk: Vector3i);

    /// A chunk left every focus and was dropped, so anything built from it can go too.
    #[signal]
    fn chunk_evicted(chunk: Vector3i);

    /// A chunk's navigation mesh is baked and in the navigation map; agents can path across it and
    /// into the neighbours that are ready too.
    #[signal]
    fn navigation_ready(chunk: Vector3i);

    /// A chunk could not be generated, and why. Asking again would fail the same way, so the
    /// generator leaves it alone; whether that leaves a hole is the game's decision.
    #[signal]
    fn chunk_failed(chunk: Vector3i, status: GString);

    /// Generation stopped altogether, and why. The node reports nothing further after this.
    #[signal]
    fn generation_failed(reason: GString);

    /// Reads a rule set in Wave Forge's RON format, either tiles with their adjacency or modules
    /// described by connectors, usually read with `FileAccess.get_file_as_string` so that it works
    /// inside an exported game.
    ///
    /// Once it is loaded, the tile functions say what each tile is, which is what a scene needs to
    /// set the prior by name or tag before [`WaveForgeWorld::start`]. Returns whether the rule set
    /// could be read; why not is reported as an error.
    #[func]
    fn load_rules(&mut self, rules: GString) -> bool {
        match wave_forge::loader::parse_rule_file(&rules.to_string()) {
            Ok(file) => {
                self.rules = Some(file);
                true
            }
            Err(error) => {
                godot_error!("wave forge: {error}");
                false
            }
        }
    }

    /// Starts generating from the rule set [`WaveForgeWorld::load_rules`] read, with the prior and
    /// properties set now.
    ///
    /// Returns whether generation could start. The GPU device and the first kernels are built on
    /// the generating thread, so a failure there arrives as `generation_failed` rather than here.
    #[func]
    fn start(&mut self) -> bool {
        let Some(file) = &self.rules else {
            godot_error!("wave forge: start() needs a rule set; call load_rules() first");
            return false;
        };
        let ruleset = match Ruleset::new(file.rules(), &file.tileset().weights) {
            Ok(ruleset) => ruleset,
            Err(error) => {
                godot_error!("wave forge: {error}");
                return false;
            }
        };
        let tiles = ruleset.num_tiles();
        if let Some(tile) = self.prior_tile_outside(tiles) {
            godot_error!(
                "wave forge: the prior names tile {tile}, but the rule set has {tiles} tiles"
            );
            return false;
        }
        let prior = self.prior(tiles);
        let extent = self.extent();
        let (seed, halo) = (self.seed as u64, self.halo.max(0) as u32);
        let warm = self.warm_kernels.then_some(self.view_radius.max(0) as u32);
        self.followed = None;
        // Everything here happens on the generating thread, including building the device and
        // compiling the kernels, so Godot's own thread never waits for either.
        self.worker = Some(Worker::spawn(move || {
            let mut world = Builder::new(ruleset, prior)
                .seed(seed)
                .extent(extent)
                .halo(halo)
                .build()?;
            if let Some(radius) = warm {
                let shapes = world.kernel_shapes(radius);
                world.solver_mut().warm(&shapes)?;
            }
            Ok(world)
        }));
        true
    }

    /// Restricts what each layer of the world may hold, from its lowest layer up.
    ///
    /// One array of tile indices per layer; the last one covers everything above it. This is how a
    /// world gets a ground layer and open sky: give the bottom layer the tiles that belong on the
    /// ground and the top layer the ones that belong in the air. An empty array means the layer
    /// allows everything.
    ///
    /// Call it before [`WaveForgeWorld::start`], which is when the world is built.
    #[func]
    fn set_layer_tiles(&mut self, layers: Array<PackedInt32Array>) {
        self.layers = layers
            .iter_shared()
            .map(|tiles| {
                tiles
                    .as_slice()
                    .iter()
                    .map(|&tile| tile.unsigned_abs())
                    .collect()
            })
            .collect();
    }

    /// Forbids `tiles` in the cells along one face of a bounded world, where nothing outside would
    /// continue them: a road may leave the world, a bridge to nowhere may not.
    ///
    /// `axis` is 0 to 5 for +x, -x, +y, -y, +z, -z, on the lattice's own axes. An axis the world is
    /// unbounded along has no face, so its ban never applies. Call it before
    /// [`WaveForgeWorld::start`].
    #[func]
    fn ban_tiles_on_face(&mut self, axis: i32, tiles: PackedInt32Array) {
        let Ok(axis) = usize::try_from(axis) else {
            godot_error!("wave forge: no axis {axis}");
            return;
        };
        if axis >= self.face_bans.len() {
            godot_error!("wave forge: no axis {axis}, the lattice has six");
            return;
        }
        self.face_bans[axis] = tiles
            .as_slice()
            .iter()
            .map(|&tile| tile.unsigned_abs())
            .collect();
    }

    /// Asks for the chunks around `position`, which is where the player is.
    ///
    /// Call it every frame: nothing happens until the position crosses into another chunk.
    #[func]
    fn follow(&mut self, position: Vector3) {
        let chunk = from_vector(self.chunk_at(position));
        if self.followed == Some(chunk) {
            return;
        }
        let radius = self.view_radius.max(0) as u32;
        let focus = [FocusPoint::new(chunk, radius)];
        let margin = self.evict_margin;
        if let Some(worker) = &self.worker {
            worker.request(&focus);
            if margin > 0 {
                worker.evict_outside(&focus, margin as u32);
            }
            self.followed = Some(chunk);
        }
    }

    /// One tile index per cell of a generated chunk, in `z`, then `y`, then `x` order, or an empty
    /// array when the chunk has not been generated.
    #[func]
    fn tiles_at(&self, chunk: Vector3i) -> PackedInt32Array {
        let Some(worker) = &self.worker else {
            return PackedInt32Array::new();
        };
        match worker.chunk(from_vector(chunk)) {
            Some(chunk) => chunk.tiles.iter().map(|&tile| i32::from(tile)).collect(),
            None => PackedInt32Array::new(),
        }
    }

    /// The chunk a position in Godot's world space falls in.
    #[func]
    fn chunk_at(&self, position: Vector3) -> Vector3i {
        to_vector(self.space().chunk_at(position.to_array()))
    }

    /// Where a chunk's lowest corner sits in Godot's world space.
    #[func]
    fn position_of(&self, chunk: Vector3i) -> Vector3 {
        Vector3::from_array(self.space().chunk_origin(from_vector(chunk)))
    }

    /// The centre of one cell of a chunk in Godot's world space. `cell` is an index into the
    /// chunk's [`WaveForgeWorld::tiles_at`].
    #[func]
    fn cell_position(&self, chunk: Vector3i, cell: i32) -> Vector3 {
        let cells = self.extent().shape().cells();
        match u32::try_from(cell) {
            Ok(cell) if cell < cells => {
                Vector3::from_array(self.space().cell_center(from_vector(chunk), cell))
            }
            _ => {
                godot_error!("wave forge: no cell {cell} in a chunk of {cells}");
                Vector3::ZERO
            }
        }
    }

    /// A chunk's placements of the modules named in `names` (every module if it is empty), ready
    /// to draw: one dictionary per module with its `name`, its `transforms` as a MultiMesh buffer
    /// (twelve floats per instance, for `RenderingServer.multimesh_set_buffer` on a multimesh of
    /// `TRANSFORM_3D` without colours or custom data), and each instance's stable `ids`. Each
    /// transform turns the module's unit model by its tile's rotation, scales it to the cell and
    /// puts it at the cell's centre. Empty for a chunk that has not been generated.
    ///
    /// An instance is known by its chunk and its id within the chunk, which stays the same in every
    /// run and session; its low 32 bits are the index of the instance's cell.
    #[func]
    fn instance_sets(&self, chunk: Vector3i, names: PackedStringArray) -> Array<VarDictionary> {
        let (Some(worker), Some(rules)) = (&self.worker, &self.rules) else {
            return Array::new();
        };
        let Some(generated) = worker.chunk(from_vector(chunk)) else {
            return Array::new();
        };
        let wanted: Vec<String> = names.as_slice().iter().map(ToString::to_string).collect();
        let drawn = |name: &str| wanted.is_empty() || wanted.iter().any(|w| w == name);
        wave_forge::instance_sets(generated, rules, &self.space(), drawn)
            .into_iter()
            .map(|set| {
                let ids: PackedInt64Array = set.ids.iter().map(|id| local_id(id.local)).collect();
                let mut out = VarDictionary::new();
                out.set(&"name".to_variant(), &GString::from(&set.name).to_variant());
                out.set(
                    &"transforms".to_variant(),
                    &PackedFloat32Array::from(set.transforms(self.cell_size.to_array()).as_slice())
                        .to_variant(),
                );
                out.set(&"ids".to_variant(), &ids.to_variant());
                out
            })
            .collect()
    }

    /// Gives every cell of `module` a collider of `shape`, turned by the tile's rotation and
    /// centred on the cell, in the chunks within `collider_radius`. The shape is sized for one cell
    /// in Godot's world units. Null takes the module's colliders away.
    #[func]
    fn set_collision_shape(&mut self, module: GString, shape: Option<Gd<Shape3D>>) {
        match shape {
            Some(shape) => self.collision_shapes.insert(module.to_string(), shape),
            None => self.collision_shapes.remove(&module.to_string()),
        };
        // Bodies and navigation meshes are rebuilt with the new shapes over the next frames.
        let mut physics = PhysicsServer3D::singleton();
        for (_, (body, _)) in self.bodies.drain() {
            physics.free_rid(body);
        }
        self.shape_faces.clear();
        let mut navigation = NavigationServer3D::singleton();
        for (_, chunk) in self.navigation.drain() {
            navigation.free_rid(chunk.region);
        }
    }

    /// The chunks whose latest navigation mesh is in the map. A chunk being baked again keeps its
    /// previous mesh in the map until the new one is ready, and is not listed meanwhile.
    #[func]
    fn navigation_chunks(&self) -> Array<Vector3i> {
        self.navigation
            .iter()
            .filter(|(_, chunk)| chunk.baking_since.is_none())
            .map(|(&coord, _)| to_vector(coord))
            .collect()
    }

    /// The instance a collision hit, from the `rid` and `shape` a ray or shape query reports: its
    /// `chunk` and its `id` within the chunk, as [`WaveForgeWorld::instance_sets`] gives them.
    /// Empty for a body or shape that is not one of this node's.
    #[func]
    fn collider_instance(&self, body: Rid, shape: i32) -> VarDictionary {
        let mut out = VarDictionary::new();
        let hit = self
            .bodies
            .iter()
            .find(|(_, (rid, _))| *rid == body)
            .and_then(|(&chunk, (_, locals))| {
                let local = usize::try_from(shape).ok().and_then(|i| locals.get(i))?;
                Some((chunk, *local))
            });
        if let Some((chunk, local)) = hit {
            out.set(&"chunk".to_variant(), &to_vector(chunk).to_variant());
            out.set(&"id".to_variant(), &local_id(local).to_variant());
        }
        out
    }

    /// The chunks that have colliders now.
    #[func]
    fn collider_chunks(&self) -> Array<Vector3i> {
        self.bodies.keys().map(|&chunk| to_vector(chunk)).collect()
    }

    /// How large one chunk is in Godot's world units.
    #[func]
    fn chunk_size(&self) -> Vector3 {
        Vector3::from_array(self.space().chunk_size())
    }

    /// How many tiles the loaded rule set has, or zero before [`WaveForgeWorld::load_rules`].
    #[func]
    fn tile_count(&self) -> i32 {
        self.rules.as_ref().map_or(0, |file| {
            i32::try_from(file.num_tiles()).expect("at most 256 tiles")
        })
    }

    /// What a tile is called: its own name in a tile set, its module's name in a module set, where
    /// every rotation of a module shares the name. Draw one model per name.
    #[func]
    fn tile_name(&self, tile: i32) -> GString {
        self.tile(tile)
            .map_or_else(GString::new, |(file, tile)| GString::from(file.name(tile)))
    }

    /// How far a tile is turned from its module, in quarter turns. Zero in a tile set.
    #[func]
    fn tile_rotation(&self, tile: i32) -> i32 {
        self.tile(tile)
            .map_or(0, |(file, tile)| i32::from(file.rotation(tile)))
    }

    /// The rotation to place a tile's model with: its turn from its module, about Godot's up axis.
    #[func]
    fn tile_basis(&self, tile: i32) -> Basis {
        self.tile(tile).map_or(Basis::IDENTITY, |(file, tile)| {
            Basis::from_axis_angle(Vector3::UP, YUpSpace::yaw(file.rotation(tile)))
        })
    }

    /// Every tile called `name`: one in a tile set, every rotation of the module in a module set.
    #[func]
    fn tiles_named(&self, name: GString) -> PackedInt32Array {
        self.rules
            .as_ref()
            .map_or_else(PackedInt32Array::new, |file| {
                to_packed(file.tiles_named(&name.to_string()))
            })
    }

    /// Every tile whose module carries `tag` in a module set. A tile set has no tags.
    #[func]
    fn tiles_tagged(&self, tag: GString) -> PackedInt32Array {
        self.rules
            .as_ref()
            .map_or_else(PackedInt32Array::new, |file| {
                to_packed(file.tiles_tagged(&tag.to_string()))
            })
    }

    /// The coordinates of every chunk generated and not yet dropped.
    #[func]
    fn generated_chunks(&self) -> Array<Vector3i> {
        match &self.worker {
            Some(worker) => worker
                .chunks()
                .map(|chunk| to_vector(chunk.coord))
                .collect(),
            None => Array::new(),
        }
    }

    /// What generation has cost so far: `batches`, `solved`, `repaired`, `rewritten_by_repair`,
    /// `failed`, `solver_ms`, `repair_batches` and `repair_ms`; how many navigation bakes have
    /// finished (`navigation_baked`) and the polygons of the meshes in place
    /// (`navigation_polygons`); and the chunks within `collider_radius` still waiting for a body
    /// (`pending_colliders`), since at most three are given one per frame.
    ///
    /// What the node's slowest frame since the start spent Godot's thread on: `slowest_frame_ms`
    /// in all, `slowest_frame_events` drained and `slowest_frame_signals_ms` emitting them (the
    /// handlers connected to them included), `slowest_frame_colliders` chunks given bodies in
    /// `slowest_frame_colliders_ms`, and `slowest_frame_navigation_ms`.
    ///
    /// And the `_median`, `_p99` and `_max` of recent timings in milliseconds, once there are
    /// some: `process_ms`, the node's own time on Godot's thread per frame; `navigation_bake_ms`,
    /// from asking for a chunk's bake to its mesh being in place; `navigation_start_ms` and
    /// `navigation_finish_ms`, Godot's thread preparing a bake and putting its mesh in the region;
    /// and `navigation_source_ms`, the part of preparing a bake that gathers its source triangles,
    /// the rest being their handover to Godot.
    #[func]
    fn stats(&self) -> Dictionary<GString, Variant> {
        let Some(worker) = &self.worker else {
            return Dictionary::new();
        };
        let stats = worker.stats();
        let mut out: Dictionary<GString, Variant> = Dictionary::new();
        out.set("batches", stats.batches);
        out.set("pending_colliders", self.bodies_pending as i64);
        out.set("solved", stats.solved);
        out.set("repaired", stats.repaired);
        out.set("rewritten_by_repair", stats.rewritten_by_repair);
        out.set("failed", stats.failed);
        out.set("solver_ms", stats.solver_ms);
        out.set("repair_batches", stats.repair_batches);
        out.set("repair_ms", stats.repair_ms);
        out.set("navigation_baked", self.baked);
        let polygons: i64 = self
            .navigation
            .values()
            .filter(|chunk| chunk.baking_since.is_none())
            .map(|chunk| i64::from(chunk.mesh.get_polygon_count()))
            .sum();
        out.set("navigation_polygons", polygons);
        let slowest = self.slowest_frame;
        out.set("slowest_frame_ms", slowest.ms);
        out.set("slowest_frame_events", slowest.events as i64);
        out.set("slowest_frame_signals_ms", slowest.signals_ms);
        out.set("slowest_frame_colliders", slowest.colliders as i64);
        out.set("slowest_frame_colliders_ms", slowest.colliders_ms);
        out.set("slowest_frame_navigation_ms", slowest.navigation_ms);
        for (name, timings) in [
            ("process", &self.process_ms),
            ("navigation_bake", &self.bake_ms),
            ("navigation_start", &self.bake_start_ms),
            ("navigation_source", &self.bake_source_ms),
            ("navigation_finish", &self.bake_finish_ms),
        ] {
            if let Some([median, p99, max]) = timings.summary() {
                out.set(&format!("{name}_ms_median"), median);
                out.set(&format!("{name}_ms_p99"), p99);
                out.set(&format!("{name}_ms_max"), max);
            }
        }
        out
    }

    /// Whether generation has been started and has not failed.
    #[func]
    fn is_generating(&self) -> bool {
        self.worker.is_some()
    }

    /// Gives the chunks within `collider_radius` of the followed chunk their bodies, rebuilding the
    /// ones whose tiles changed, and frees the bodies of chunks that are out of range or gone.
    ///
    /// A chunk is one static body with a shape per instance. Every shape is added before the body
    /// joins the space: Jolt rebuilds a body's compound shape on each shape added once it is in a
    /// space, which made a chunk of 200 boxes cost 3.1 ms instead of 0.12
    /// (docs/research/measurements.md).
    ///
    /// Returns how many chunks got a body.
    fn update_colliders(&mut self, updated: &[ChunkCoord]) -> usize {
        let (Some(worker), Some(rules), Some(focus)) = (&self.worker, &self.rules, self.followed)
        else {
            return 0;
        };
        let radius = self.collider_radius;
        let within = |chunk: ChunkCoord| {
            radius >= 0
                && (chunk.x - focus.x)
                    .abs()
                    .max((chunk.y - focus.y).abs())
                    .max((chunk.z - focus.z).abs())
                    <= radius
        };
        let mut physics = PhysicsServer3D::singleton();
        let gone: Vec<ChunkCoord> = self
            .bodies
            .keys()
            .copied()
            .filter(|&chunk| !within(chunk) || worker.chunk(chunk).is_none())
            .collect();
        for chunk in gone {
            if let Some((body, _)) = self.bodies.remove(&chunk) {
                physics.free_rid(body);
            }
        }
        self.bodies_due
            .extend(updated.iter().copied().filter(|&chunk| within(chunk)));
        self.bodies_due.retain(|&chunk| within(chunk));
        if self.collision_shapes.is_empty() {
            return 0;
        }
        let Some(space) = self
            .base()
            .get_viewport()
            .and_then(|viewport| viewport.find_world_3d())
            .map(|world| world.get_space())
        else {
            return 0;
        };
        let owner = u64::from_ne_bytes(self.base().instance_id().to_i64().to_ne_bytes());
        let layout = self.space();
        let mut wanted: Vec<ChunkCoord> = worker
            .chunks()
            .map(|chunk| chunk.coord)
            .filter(|&chunk| within(chunk))
            .filter(|chunk| !self.bodies.contains_key(chunk) || self.bodies_due.contains(chunk))
            .collect();
        // Nearest first, and a few a frame: each body costs Godot's thread a fraction of a
        // millisecond, and a turn of the player can make a whole ring of chunks due at once.
        wanted.sort_by_key(|chunk| {
            let distance = (chunk.x - focus.x)
                .abs()
                .max((chunk.y - focus.y).abs())
                .max((chunk.z - focus.z).abs());
            (distance, *chunk)
        });
        self.bodies_pending = wanted.len().saturating_sub(BODIES_PER_FRAME);
        wanted.truncate(BODIES_PER_FRAME);
        let built = wanted.len();
        for coord in wanted {
            self.bodies_due.remove(&coord);
            let chunk = worker
                .chunk(coord)
                .expect("chosen from the worker's chunks");
            let body = physics.body_create();
            physics.body_set_mode(body, BodyMode::STATIC);
            let mut ids = Vec::new();
            let sets = wave_forge::instance_sets(chunk, rules, &layout, |name| {
                self.collision_shapes.contains_key(name)
            });
            for set in sets {
                let shape = self.collision_shapes[&set.name].get_rid();
                for (row, &id) in set.transforms([1.0; 3]).chunks(12).zip(&set.ids) {
                    let basis = Basis::from_rows(
                        Vector3::new(row[0], row[1], row[2]),
                        Vector3::new(row[4], row[5], row[6]),
                        Vector3::new(row[8], row[9], row[10]),
                    );
                    let at = Transform3D::new(basis, Vector3::new(row[3], row[7], row[11]));
                    physics.body_add_shape_ex(body, shape).transform(at).done();
                    ids.push(id.local);
                }
            }
            physics.body_attach_object_instance_id(body, owner);
            physics.body_set_space(body, space);
            if let Some((old, _)) = self.bodies.insert(coord, (body, ids)) {
                physics.free_rid(old);
            }
        }
        built
    }

    /// Keeps a navigation mesh on every chunk within `navigation_radius` of the followed chunk:
    /// bakes the ones that have none, or whose tiles or neighbours changed, once all their
    /// neighbours are there; puts finished bakes into the map; frees the ones out of range.
    ///
    /// A chunk's source is `wave_forge::nav_source`: its own shapes and its neighbours' out to a
    /// border, as plain triangles, so the bake runs entirely on the navigation server's threads.
    /// Baked within its bounds and with that border, neighbouring chunks meet on edges built from
    /// the same geometry, whose vertices the map merges without edge connections
    /// (docs/architecture/engine-integration.md, "Systems other than rendering", from Godot's
    /// navigation chunk guidance). Finished bakes are found by asking the server each frame rather
    /// than through a callback, which the server calls from its own thread.
    fn update_navigation(&mut self, updated: &[ChunkCoord]) {
        let (Some(worker), Some(focus)) = (&self.worker, self.followed) else {
            return;
        };
        let radius = self.navigation_radius;
        let within = |chunk: ChunkCoord| {
            radius >= 0
                && (chunk.x - focus.x)
                    .abs()
                    .max((chunk.y - focus.y).abs())
                    .max((chunk.z - focus.z).abs())
                    <= radius
        };
        let mut server = NavigationServer3D::singleton();
        let gone: Vec<ChunkCoord> = self
            .navigation
            .keys()
            .copied()
            .filter(|&chunk| !within(chunk) || worker.chunk(chunk).is_none())
            .collect();
        for chunk in gone {
            if let Some(chunk) = self.navigation.remove(&chunk) {
                server.free_rid(chunk.region);
            }
        }
        let Some(map) = self
            .base()
            .get_viewport()
            .and_then(|viewport| viewport.find_world_3d())
            .map(|world| world.get_navigation_map())
        else {
            return;
        };
        let cell_size = server.map_get_cell_size(map);
        // Finished bakes go into the map.
        let mut ready = Vec::new();
        for (&coord, chunk) in &mut self.navigation {
            let Some(since) = chunk.baking_since else {
                continue;
            };
            if server.is_baking_navigation_mesh(&chunk.mesh) {
                continue;
            }
            let finishing = std::time::Instant::now();
            server.region_set_navigation_mesh(chunk.region, &chunk.mesh);
            chunk.baking_since = None;
            self.bake_finish_ms
                .push(finishing.elapsed().as_secs_f64() * 1000.0);
            self.bake_ms.push(since.elapsed().as_secs_f64() * 1000.0);
            self.baked += 1;
            ready.push(coord);
        }
        for coord in &ready {
            self.signals().navigation_ready().emit(to_vector(*coord));
        }
        if self.collision_shapes.is_empty() || radius < 0 {
            return;
        }
        for (name, shape) in &self.collision_shapes {
            if self.shape_faces.contains_key(name) {
                continue;
            }
            // A shape's filled debug mesh is the one triangle form every Shape3D offers; it has
            // triangles unless a CollisionShape3D showing it turned `debug_fill` off.
            let faces: Vec<[f32; 3]> = shape
                .get_debug_mesh()
                .map(|mesh| {
                    mesh.get_faces()
                        .as_slice()
                        .iter()
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
        let (Some(worker), Some(rules)) = (&self.worker, &self.rules) else {
            return;
        };
        server.map_set_use_edge_connections(map, false);
        let extent = self.extent();
        let layout = self.space();
        // Chunks that need a bake: in range, generated, and either without a mesh or changed,
        // with their neighbours, since it was baked.
        let touched = |chunk: ChunkCoord| {
            (-1..=1).any(|dx| {
                (-1..=1).any(|dy| {
                    updated.contains(&ChunkCoord::new(chunk.x + dx, chunk.y + dy, chunk.z))
                })
            })
        };
        let mut wanted: Vec<ChunkCoord> = worker
            .chunks()
            .map(|chunk| chunk.coord)
            .filter(|&chunk| within(chunk))
            .filter(|chunk| match self.navigation.get(chunk) {
                None => true,
                Some(existing) => touched(*chunk) || existing.stale,
            })
            .collect();
        if wanted.is_empty() {
            return;
        }
        // One bake is prepared per frame, nearest first: preparing one costs Godot's thread up to
        // 3 ms, and a focus crossing into a chunk makes several due at once.
        wanted.sort_by_key(|chunk| {
            (chunk.x - focus.x)
                .abs()
                .max((chunk.y - focus.y).abs())
                .max((chunk.z - focus.z).abs())
        });
        let cell_height = server.map_get_cell_height(map);
        let template = self
            .navigation_template
            .clone()
            .unwrap_or_else(NavigationMesh::new_gd);
        // Recast's own padding for tiles: the agent's radius in whole cells, and three more.
        let border = ((template.get_agent_radius() / cell_size).ceil() + 3.0) * cell_size;
        let mut started = false;
        for coord in wanted {
            if let Some(existing) = self.navigation.get_mut(&coord)
                && existing.baking_since.is_some()
            {
                existing.stale = true;
                continue;
            }
            if started {
                // A changed chunk that has a mesh keeps it until its turn; its stale mark is how
                // it gets that turn on a later frame.
                if let Some(existing) = self.navigation.get_mut(&coord) {
                    existing.stale = true;
                }
                continue;
            }
            let preparing = std::time::Instant::now();
            let faces = &self.shape_faces;
            let nav = match wave_forge::nav_source(
                coord,
                |chunk| worker.chunk(chunk),
                &extent,
                rules,
                &layout,
                |name| faces.get(name).map(Vec::as_slice),
                border,
            ) {
                Ok(nav) => nav,
                Err(NavSourceError::Missing(_)) => continue,
                Err(error) => {
                    godot_error!("wave forge: navigation turned off: {error}");
                    self.navigation_radius = -1;
                    return;
                }
            };
            started = true;
            self.bake_source_ms
                .push(preparing.elapsed().as_secs_f64() * 1000.0);
            // Handed over in one copy each; the setters keep the arrays they are given, where
            // `append_arrays` copies them again and rewrites every index.
            // The indices are in Recast's winding, counter-clockwise, where Godot's faces are
            // clockwise: the order that `add_faces` swaps for its callers. Unswapped, Recast takes
            // the underside of every face for its top and walks inside the ground.
            let triangles = i32::try_from(nav.triangles.len() / 9).expect("a source fits i32");
            let indices: Vec<i32> = (0..triangles)
                .flat_map(|triangle| [3 * triangle, 3 * triangle + 2, 3 * triangle + 1])
                .collect();
            let mut source = NavigationMeshSourceGeometryData3D::new_gd();
            source.set_vertices(&PackedFloat32Array::from(nav.triangles.as_slice()));
            source.set_indices(&PackedInt32Array::from(indices.as_slice()));
            let mut mesh = template.duplicate_resource();
            mesh.set_cell_size(cell_size);
            mesh.set_cell_height(cell_height);
            mesh.set_filter_baking_aabb(Aabb::new(
                Vector3::from_array(nav.bounds_origin),
                Vector3::from_array(nav.bounds_size),
            ));
            mesh.set_border_size(nav.border);
            server.bake_from_source_geometry_data_async(&mesh, &source);
            self.bake_start_ms
                .push(preparing.elapsed().as_secs_f64() * 1000.0);
            let region = match self.navigation.remove(&coord) {
                Some(existing) => existing.region,
                None => {
                    let region = server.region_create();
                    server.region_set_map(region, map);
                    server.region_set_enabled(region, true);
                    region
                }
            };
            self.navigation.insert(
                coord,
                NavigationChunk {
                    region,
                    mesh,
                    baking_since: Some(std::time::Instant::now()),
                    stale: false,
                },
            );
        }
    }

    /// The lattice in Godot's Y-up world space.
    fn space(&self) -> YUpSpace {
        YUpSpace::new(self.extent().shape(), self.cell_size.to_array())
    }

    /// The loaded rule set and a tile index a script passed, if both are valid; an error otherwise.
    fn tile(&self, tile: i32) -> Option<(&RuleFile, usize)> {
        let Some(file) = &self.rules else {
            godot_error!("wave forge: no rule set is loaded; call load_rules() first");
            return None;
        };
        match usize::try_from(tile) {
            Ok(index) if index < file.num_tiles() => Some((file, index)),
            _ => {
                godot_error!(
                    "wave forge: no tile {tile}, the rule set has {}",
                    file.num_tiles()
                );
                None
            }
        }
    }

    /// A tile index the prior names that the rule set does not have, if there is one.
    fn prior_tile_outside(&self, num_tiles: u32) -> Option<u32> {
        self.layers
            .iter()
            .chain(&self.face_bans)
            .flatten()
            .copied()
            .find(|&tile| tile >= num_tiles)
    }

    /// What a cell may hold before anything is decided: the layer masks and face bans a scene set,
    /// or everything if it set none.
    fn prior(&self, num_tiles: u32) -> Prior {
        let mask = |tiles: &[u32]| {
            let mut mask = TileMask::EMPTY;
            for &tile in tiles {
                mask.insert(tile);
            }
            mask
        };
        let mut prior = Prior::open(num_tiles);
        if !self.layers.is_empty() {
            let layer = |tiles: &Vec<u32>| {
                if tiles.is_empty() {
                    TileMask::all(num_tiles)
                } else {
                    mask(tiles)
                }
            };
            prior = prior.with_layers(self.layers.iter().map(layer).collect());
        }
        for (axis, tiles) in self.face_bans.iter().enumerate() {
            if !tiles.is_empty() {
                prior = prior.with_face_ban(axis, mask(tiles));
            }
        }
        prior
    }

    fn extent(&self) -> WorldExtent {
        let cells = self.chunk_cells;
        let shape = ChunkShape {
            x: cells.x.max(1) as u32,
            y: cells.y.max(1) as u32,
            z: cells.z.max(1) as u32,
        };
        let mut extent = WorldExtent::new(shape);
        let bounds = self.world_chunks;
        if bounds.x > 0 {
            extent = extent.with_x(0..bounds.x);
        }
        if bounds.y > 0 {
            extent = extent.with_y(0..bounds.y);
        }
        // A world that is one chunk tall is the usual case, and it is also what keeps a chunk's
        // vertical faces free of halo cells nothing will ever agree with.
        extent.with_z(0..bounds.z.max(1))
    }
}

/// An instance's id within its chunk as a Godot integer; ids stay below 2^63.
fn local_id(local: u64) -> i64 {
    i64::try_from(local).expect("an instance id stays below 2^63")
}

/// The lattice's coordinates as Godot sees them, which is the same order: the lattice's own axes.
const fn to_vector(chunk: ChunkCoord) -> Vector3i {
    Vector3i::new(chunk.x, chunk.y, chunk.z)
}

const fn from_vector(chunk: Vector3i) -> ChunkCoord {
    ChunkCoord::new(chunk.x, chunk.y, chunk.z)
}

/// Tile indices as a script receives them.
fn to_packed(tiles: Vec<usize>) -> PackedInt32Array {
    tiles
        .into_iter()
        .map(|tile| i32::try_from(tile).expect("at most 256 tiles"))
        .collect()
}

/// A status as a name a script can compare, rather than a number that would go stale.
const fn status_name(status: RegionStatus) -> &'static str {
    match status {
        RegionStatus::Solved => "solved",
        RegionStatus::Exhausted => "exhausted",
        RegionStatus::StepCap => "step_cap",
        RegionStatus::BorderContradiction => "border_contradiction",
        RegionStatus::Superseded => "superseded",
    }
}
