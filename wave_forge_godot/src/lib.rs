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
//!         place(world.tile_name(tile), Transform3D(world.tile_basis(tile), world.cell_position(chunk, cell)))
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
//! `docs/architecture.md` §8.2 and the Godot section of `docs/roadmap.md` for what it needs and
//! what has to be measured to choose between the two.
//!
//! # Coordinates
//!
//! Godot is Y-up and the generator is Z-up. Godot's `xz` ground plane is therefore the lattice's
//! `xy`, and Godot's `y` is the lattice's `z`. `cell_size` says how large one cell is along each of
//! Godot's axes, which is what turns a position into a chunk.

use godot::classes::{FileAccess, INode, Node};
use godot::prelude::*;
use wave_forge::loader::RuleFile;
use wave_forge::{
    Builder, ChunkCoord, ChunkEvent, ChunkShape, FocusPoint, Prior, RegionStatus, Ruleset,
    TileMask, Worker, WorldExtent, YUpSpace,
};

struct WaveForgeExtension;

#[gdextension]
unsafe impl ExtensionLibrary for WaveForgeExtension {}

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
        for event in events {
            match event {
                ChunkEvent::Updated(chunk) => {
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

    /// The centre of one cell of a chunk in Godot's world space. `cell` is an index into the chunk's
    /// [`WaveForgeWorld::tiles_at`].
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
                let ids: PackedInt64Array = set
                    .ids
                    .iter()
                    .map(|&id| i64::from_ne_bytes(id.to_ne_bytes()))
                    .collect();
                let mut out = VarDictionary::new();
                out.set(&"name".to_variant(), &GString::from(&set.name).to_variant());
                out.set(
                    &"transforms".to_variant(),
                    &PackedFloat32Array::from(set.transforms.as_slice()).to_variant(),
                );
                out.set(&"ids".to_variant(), &ids.to_variant());
                out
            })
            .collect()
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
    /// `failed`, `solver_ms`, `repair_batches` and `repair_ms`.
    #[func]
    fn stats(&self) -> Dictionary<GString, Variant> {
        let Some(worker) = &self.worker else {
            return Dictionary::new();
        };
        let stats = worker.stats();
        let mut out: Dictionary<GString, Variant> = Dictionary::new();
        out.set("batches", stats.batches);
        out.set("solved", stats.solved);
        out.set("repaired", stats.repaired);
        out.set("rewritten_by_repair", stats.rewritten_by_repair);
        out.set("failed", stats.failed);
        out.set("solver_ms", stats.solver_ms);
        out.set("repair_batches", stats.repair_batches);
        out.set("repair_ms", stats.repair_ms);
        out
    }

    /// Whether generation has been started and has not failed.
    #[func]
    fn is_generating(&self) -> bool {
        self.worker.is_some()
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
