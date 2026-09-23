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

use crate::timings::Timings;
use crate::{RECENT_FRAMES, from_vector, local_id, to_vector};
use godot::classes::physics_server_3d::BodyMode;
use godot::classes::rendering_server::{ArrayType, PrimitiveType};
use godot::classes::{
    FileAccess, INode, Material, Node, PhysicsServer3D, RenderingServer, Shape3D,
};
use godot::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use wave_forge::loader::{RuleFile, parse_rule_file};
use wave_forge::stages::{Pack, Runtime, StageEvent, StageKind, StageWorker};
use wave_forge::towns::WfcTowns;
use wave_forge::{
    Chunk, ChunkCoord, ChunkShape, FocusPoint, GroundMesh, InstanceSet, YUpSpace, ground,
    ground_readers,
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

    /// The field stage the ground is built from, a height in cells per column; empty for no
    /// ground. It has to be generated, as a target or as what a target reads. A chunk's ground
    /// needs the fields of the chunks around it, so it reaches one chunk less than the view.
    #[export_group(name = "Ground")]
    #[export]
    ground_stage: GString,
    /// The material the ground is drawn with; none draws it with Godot's default.
    #[export]
    ground_material: Option<Gd<Material>>,

    /// How many chunks around the followed position get colliders: the ground, and every town's
    /// modules that have a shape (`set_collision_shape`). Below zero, none.
    #[export_group(name = "Physics")]
    #[export]
    collider_radius: i32,

    pack: Option<Arc<Pack>>,
    /// The rule sets Solve stages name, kept here too, to say what a town's tiles are.
    rules: BTreeMap<String, RuleFile>,
    worker: Option<StageWorker>,
    followed: Option<ChunkCoord>,
    process_ms: Timings,
    /// Each town module's collision shape, by module name, for every Solve stage.
    collision_shapes: HashMap<String, Gd<Shape3D>>,
    /// The chunks whose ground is built: its mesh, and its `RenderingServer` mesh and instance.
    grounds: HashMap<ChunkCoord, (GroundMesh, Rid, Rid)>,
    /// Each chunk's static body and its ground's height map shape, which the body does not own,
    /// and what it holds, to tell when it has to be built again.
    bodies: HashMap<ChunkCoord, (Rid, Option<Rid>, BodyContents)>,
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
            targets: PackedStringArray::new(),
            start_on_ready: false,
            seed: 0,
            chunk_cells: Vector3i::new(8, 8, 8),
            cell_size: Vector3::ONE,
            view_radius: 2,
            pack: None,
            rules: BTreeMap::new(),
            worker: None,
            followed: None,
            process_ms: Timings::new(RECENT_FRAMES),
            ground_stage: GString::new(),
            ground_material: None,
            collider_radius: 1,
            collision_shapes: HashMap::new(),
            grounds: HashMap::new(),
            bodies: HashMap::new(),
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
        for event in events {
            match event {
                StageEvent::Generated { stage, chunk } => {
                    if stage == ground_stage {
                        arrived.push(chunk);
                    }
                    self.signals()
                        .stage_ready()
                        .emit(&GString::from(&stage), to_vector(chunk));
                }
                StageEvent::Dropped { stage, chunk } => {
                    if stage == ground_stage {
                        gone.push(chunk);
                    }
                    self.signals()
                        .stage_dropped()
                        .emit(&GString::from(&stage), to_vector(chunk));
                }
            }
        }
        self.update_ground(&arrived, &gone);
        self.update_colliders();
        self.process_ms
            .push(processing.elapsed().as_secs_f64() * 1000.0);
    }
}

#[godot_api]
impl WaveForgeStages {
    /// A stage's product for a chunk is ready to read.
    #[signal]
    fn stage_ready(stage: GString, chunk: Vector3i);

    /// A stage's product for a chunk is no longer needed and was dropped.
    #[signal]
    fn stage_dropped(stage: GString, chunk: Vector3i);

    /// Generation stopped, and why.
    #[signal]
    fn generation_failed(reason: GString);

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
        let for_thread = Arc::clone(&pack);
        self.rules = rules.clone();
        self.pack = Some(pack);
        self.followed = None;
        self.clear_ground_and_bodies();
        // The towns' device is built on the stages' thread, which is where it is used.
        self.worker = Some(StageWorker::spawn(move || {
            let runtime = Runtime::new(for_thread, seed, [shape.x, shape.y]);
            if !solves {
                return Ok(runtime);
            }
            let mut towns = WfcTowns::new(shape);
            for (name, file) in rules {
                towns = towns
                    .with_rules(&name, file, wave_forge::towns::gpu_solver)
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
        let targets: Vec<String> = self
            .targets
            .as_slice()
            .iter()
            .map(ToString::to_string)
            .collect();
        let names: Vec<&str> = targets.iter().map(String::as_str).collect();
        worker.request(
            &[FocusPoint::new(chunk, self.view_radius.max(0) as u32)],
            &names,
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

    /// A Solve stage's town in a chunk: its `region` (Vector2i), the site's levelled `height`, in
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
        out.set(
            &"region".to_variant(),
            &Vector2i::new(town.region.0, town.region.1).to_variant(),
        );
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

    /// A Sites stage's sites that overlap a chunk: each one's `region` (Vector2i) that names it,
    /// the chunks it covers from `min` up to but not including `max` (Vector2i, along the lattice's
    /// x and y), and its levelled `height` in cells. Empty if there are none or the chunk has not
    /// arrived.
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
                out.set(
                    &"region".to_variant(),
                    &Vector2i::new(site.region.0, site.region.1).to_variant(),
                );
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
            let (sin, cos) = (point.turn * std::f32::consts::TAU).sin_cos();
            let [x, y, height] = point.position;
            transforms.extend([
                cos,
                0.0,
                sin,
                x * cell.x,
                0.0,
                1.0,
                0.0,
                height * cell.y,
                -sin,
                0.0,
                cos,
                y * cell.z,
            ]);
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
    /// frames, once there are some.
    #[func]
    fn stats(&self) -> VarDictionary {
        let mut out = VarDictionary::new();
        if let Some([median, p99, max]) = self.process_ms.summary() {
            out.set(&"process_ms_median".to_variant(), &median.to_variant());
            out.set(&"process_ms_p99".to_variant(), &p99.to_variant());
            out.set(&"process_ms_max".to_variant(), &max.to_variant());
        }
        out
    }
}

impl WaveForgeStages {
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

    /// Builds the ground of every chunk a newly arrived field may have completed, and frees the
    /// ground of chunks whose own field was dropped.
    fn update_ground(&mut self, arrived: &[ChunkCoord], gone: &[ChunkCoord]) {
        let mut rendering = RenderingServer::singleton();
        for chunk in gone {
            if let Some((_, mesh, instance)) = self.grounds.remove(chunk) {
                rendering.free_rid(instance);
                rendering.free_rid(mesh);
            }
        }
        let Some(worker) = &self.worker else {
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
        let stage = self.ground_stage.to_string();
        let cell = self.cell_size.to_array();
        let mut built = Vec::new();
        for &field in arrived {
            for chunk in ground_readers(field) {
                if self.grounds.contains_key(&chunk) || built.iter().any(|(c, _)| *c == chunk) {
                    continue;
                }
                if let Some(mesh) = ground(chunk, |at| worker.field(&stage, at), cell) {
                    built.push((chunk, mesh));
                }
            }
        }
        for (chunk, mesh) in built {
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
            // The library's triangles are counter-clockwise seen from above; Godot's front faces
            // are clockwise.
            let indices: PackedInt32Array = mesh
                .indices
                .chunks(3)
                .flat_map(|triangle| [triangle[0], triangle[2], triangle[1]])
                .map(|index| index as i32)
                .collect();
            arrays.set(ArrayType::VERTEX.ord() as usize, &vertices.to_variant());
            arrays.set(ArrayType::NORMAL.ord() as usize, &normals.to_variant());
            arrays.set(ArrayType::INDEX.ord() as usize, &indices.to_variant());
            rendering.mesh_add_surface_from_arrays(rid, PrimitiveType::TRIANGLES, &arrays);
            if let Some(material) = &self.ground_material {
                rendering.mesh_surface_set_material(rid, 0, material.get_rid());
            }
            let instance = rendering.instance_create2(rid, scenario);
            rendering.instance_set_transform(
                instance,
                Transform3D::new(Basis::IDENTITY, self.chunk_corner(chunk)),
            );
            self.grounds.insert(chunk, (mesh, rid, instance));
        }
    }

    /// Keeps one static body on every chunk within `collider_radius` of the followed chunk that
    /// has ground or a town with shapes, building it again when what it would hold changes, and
    /// frees the others.
    fn update_colliders(&mut self) {
        let (Some(pack), Some(focus)) = (self.pack.clone(), self.followed) else {
            return;
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
            return;
        };
        let range = focus.x - radius.max(0)..=focus.x + radius.max(0);
        let wanted: Vec<(ChunkCoord, BodyContents)> = range
            .flat_map(|x| {
                (focus.y - radius.max(0)..=focus.y + radius.max(0))
                    .map(move |y| ChunkCoord::new(x, y, 0))
            })
            .filter(|chunk| radius >= 0 && !self.bodies.contains_key(chunk))
            .map(|chunk| (chunk, contents(self, chunk)))
            .filter(|(_, held)| held.ground || !held.towns.is_empty())
            .collect();
        let owner = u64::from_ne_bytes(self.base().instance_id().to_i64().to_ne_bytes());
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

    fn clear_ground_and_bodies(&mut self) {
        let mut rendering = RenderingServer::singleton();
        for (_, (_, mesh, instance)) in self.grounds.drain() {
            rendering.free_rid(instance);
            rendering.free_rid(mesh);
        }
        self.free_bodies();
    }
}
