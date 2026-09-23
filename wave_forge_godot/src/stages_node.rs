//! A world generated from a pack of stages (docs/reference/godot.md), for Godot.
//!
//! The node is a thin surface over [`wave_forge::stages::StageWorker`]: it loads a pack and the
//! rule sets its Solve stages name, runs the stages on a thread of their own around the position
//! the game hands it, and gives each product to the game as Godot data: a field's values, a town
//! chunk's tiles and height, points as MultiMesh buffers per kind.

use crate::timings::Timings;
use crate::{RECENT_FRAMES, from_vector, local_id, to_vector};
use godot::classes::{FileAccess, INode, Node};
use godot::prelude::*;
use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::loader::{RuleFile, parse_rule_file};
use wave_forge::stages::{Pack, Runtime, StageEvent, StageKind, StageWorker};
use wave_forge::towns::WfcTowns;
use wave_forge::{Chunk, ChunkCoord, ChunkShape, FocusPoint, YUpSpace};

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

    pack: Option<Arc<Pack>>,
    /// The rule sets Solve stages name, kept here too, to say what a town's tiles are.
    rules: BTreeMap<String, RuleFile>,
    worker: Option<StageWorker>,
    followed: Option<ChunkCoord>,
    process_ms: Timings,
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
        }
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
        for event in events {
            match event {
                StageEvent::Generated { stage, chunk } => self
                    .signals()
                    .stage_ready()
                    .emit(&GString::from(&stage), to_vector(chunk)),
                StageEvent::Dropped { stage, chunk } => self
                    .signals()
                    .stage_dropped()
                    .emit(&GString::from(&stage), to_vector(chunk)),
            }
        }
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
        let stage = stage.to_string();
        let (Some(pack), Some(worker)) = (&self.pack, &self.worker) else {
            return Array::new();
        };
        let Some(StageKind::Solve { rules, .. }) = pack.kind(&stage) else {
            return Array::new();
        };
        let (Some(file), Some(town)) = (
            self.rules.get(rules),
            worker.tiles(&stage, from_vector(chunk)),
        ) else {
            return Array::new();
        };
        let cells = self.chunk_cells;
        let shape = ChunkShape {
            x: cells.x.max(1) as u32,
            y: cells.y.max(1) as u32,
            z: cells.z.max(1) as u32,
        };
        let space = YUpSpace::new(shape, self.cell_size.to_array());
        let tiles = Chunk {
            coord: from_vector(chunk),
            tiles: town.tiles.to_vec().into_boxed_slice(),
            version: 1,
        };
        let wanted: Vec<String> = names.as_slice().iter().map(ToString::to_string).collect();
        let drawn = |name: &str| wanted.is_empty() || wanted.iter().any(|w| w == name);
        let lift = town.height * self.cell_size.y;
        wave_forge::instance_sets(&tiles, file, &space, drawn)
            .into_iter()
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
