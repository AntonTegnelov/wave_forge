//! Wave Forge as a Godot GDExtension: a `WaveForgeWorld` node that generates chunks around
//! whatever the game tells it to follow.
//!
//! ```gdscript
//! extends Node3D
//!
//! func _ready() -> void:
//!     var world: WaveForgeWorld = $WaveForgeWorld
//!     world.chunk_updated.connect(_on_chunk_updated)
//!     # What belongs on the ground and what belongs above it, if the rule set leaves that open.
//!     var layers: Array[PackedInt32Array] = [PackedInt32Array([ROAD, GRASS]), PackedInt32Array([AIR])]
//!     world.set_layer_tiles(layers)
//!     var rules := FileAccess.get_file_as_string("res://rules.ron")
//!     world.start(rules)
//!
//! func _process(_delta: float) -> void:
//!     $WaveForgeWorld.follow($Player.global_position)
//!
//! func _on_chunk_updated(chunk: Vector3i) -> void:
//!     var tiles := $WaveForgeWorld.tiles_at(chunk)   # one tile index per cell
//!     build_mesh_for(chunk, tiles)
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

use godot::classes::{INode, Node};
use godot::prelude::*;
use wave_forge::{
    Builder, ChunkCoord, ChunkEvent, ChunkShape, FocusPoint, Prior, RegionShape, RegionStatus,
    Ruleset, TileMask, Worker, WorldExtent,
};

struct WaveForgeExtension;

#[gdextension]
unsafe impl ExtensionLibrary for WaveForgeExtension {}

/// Generates a world in chunks around a position the game keeps handing it.
///
/// Set the properties in the editor or from a script, call [`WaveForgeWorld::start`] with a rule
/// set, then call [`WaveForgeWorld::follow`] as the player moves and read the tiles of every chunk
/// the `chunk_updated` signal names.
#[derive(GodotClass)]
#[class(base = Node)]
pub struct WaveForgeWorld {
    base: Base<Node>,

    /// Every choice in the world derives from this.
    #[export]
    seed: i64,
    /// The cells of one chunk, along the lattice's own axes (x, y across the ground, z up).
    #[export]
    chunk_cells: Vector3i,
    /// How large one cell is in Godot's world units, along Godot's axes.
    #[export]
    cell_size: Vector3,
    /// How many chunks to keep generated around the followed position.
    #[export]
    view_radius: i32,
    /// Cells solved around a chunk and thrown away, so its borders can be completed.
    #[export]
    halo: i32,
    /// Chunks further than `view_radius` plus this many are dropped. Below one, a chunk that is
    /// about to be asked for again would be dropped, because the chunks a focus needs reach one
    /// beyond its radius. Zero keeps everything.
    #[export]
    evict_margin: i32,
    /// How many chunks the world is, along the lattice's axes. Zero on an axis means unbounded.
    #[export]
    world_chunks: Vector3i,
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
            worker: None,
            followed: None,
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

    /// Starts generating from a rule set in Wave Forge's RON format, either tiles with their
    /// adjacency or modules described by connectors, usually read with
    /// `FileAccess.get_file_as_string` so that it works inside an exported game.
    ///
    /// Returns whether the rule set could be used. The GPU device and the first kernels are built
    /// on the generating thread, so a failure there arrives as `generation_failed` rather than
    /// here.
    #[func]
    fn start(&mut self, rules: GString) -> bool {
        let text = rules.to_string();
        let file = match wave_forge::loader::parse_rule_file(&text) {
            Ok(file) => file,
            Err(error) => {
                godot_error!("wave forge: {error}");
                return false;
            }
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
        let warm = self
            .warm_kernels
            .then(|| batch_capacities(self.view_radius.max(0) as u32));
        self.followed = None;
        // Everything here happens on the generating thread, including building the device and
        // compiling the kernels, so Godot's own thread never waits for either.
        self.worker = Some(Worker::spawn(move || {
            let mut world = Builder::new(ruleset, prior)
                .seed(seed)
                .extent(extent)
                .halo(halo)
                .build()?;
            if let Some(capacities) = warm {
                let config = world.config().clone();
                let region = |halo: u32| config.chunk.region(config.extent.halo(halo));
                let solver = world.solver_mut();
                // A repair solves one chunk, at any halo up to the widest the device fits.
                let repairs = (1..=config.repair.max_halo)
                    .map(region)
                    .filter(|shape| solver.fits(*shape))
                    .map(|shape| (1, shape));
                let shapes: Vec<(u32, RegionShape)> = capacities
                    .iter()
                    .map(|&batch| (batch, region(halo)))
                    .chain(repairs)
                    .collect();
                solver.warm(&shapes)?;
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
        let size = self.chunk_size();
        to_vector(ChunkCoord::new(
            (position.x / size.x).floor() as i32,
            (position.z / size.z).floor() as i32,
            (position.y / size.y).floor() as i32,
        ))
    }

    /// Where a chunk's lowest corner sits in Godot's world space.
    #[func]
    fn position_of(&self, chunk: Vector3i) -> Vector3 {
        let size = self.chunk_size();
        Vector3::new(
            chunk.x as f32 * size.x,
            chunk.z as f32 * size.y,
            chunk.y as f32 * size.z,
        )
    }

    /// How large one chunk is in Godot's world units.
    #[func]
    fn chunk_size(&self) -> Vector3 {
        Vector3::new(
            self.cell_size.x * self.chunk_cells.x.max(1) as f32,
            self.cell_size.y * self.chunk_cells.z.max(1) as f32,
            self.cell_size.z * self.chunk_cells.y.max(1) as f32,
        )
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
    /// `failed` and `solver_ms`.
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
        out
    }

    /// Whether generation has been started and has not failed.
    #[func]
    fn is_generating(&self) -> bool {
        self.worker.is_some()
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

/// A status as a name a script can compare, rather than a number that would go stale.
const fn status_name(status: RegionStatus) -> &'static str {
    match status {
        RegionStatus::Solved => "solved",
        RegionStatus::Exhausted => "exhausted",
        RegionStatus::StepCap => "step_cap",
        RegionStatus::BorderContradiction => "border_contradiction",
    }
}

/// The batch capacities a focus of `radius` chunks can dispatch, which is what a kernel is
/// specialised for.
///
/// A batch holds one parity of the chunks a focus asks for, and those include the ring of
/// neighbours the second parity reads, so a batch is at most half of a square two chunks wider than
/// the view. The solver rounds a batch up to a power of two, and a walking player dispatches every
/// size from one chunk to that, so every power of two up to it is needed.
fn batch_capacities(radius: u32) -> Vec<u32> {
    let across = 2 * radius + 3;
    let largest = (across * across).div_ceil(2).next_power_of_two();
    std::iter::successors(Some(1), |&capacity| {
        (capacity < largest).then_some(capacity * 2)
    })
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_batch_a_focus_can_dispatch_has_a_warmed_kernel() {
        for radius in 0..6u32 {
            // One parity of the square a focus asks for, and of the ring of neighbours the second
            // parity reads, is the most one batch can hold.
            let across = 2 * radius + 3;
            let largest: u32 = (across * across).div_ceil(2);

            let capacities = batch_capacities(radius);

            for batch in 1..=largest {
                // The solver rounds a batch up to a power of two and specialises for that.
                let capacity = batch.next_power_of_two();
                assert!(
                    capacities.contains(&capacity),
                    "radius {radius}: a batch of {batch} needs capacity {capacity}, warmed {capacities:?}"
                );
            }
        }
    }
}
